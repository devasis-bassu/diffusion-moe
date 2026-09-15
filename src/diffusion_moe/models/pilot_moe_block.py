"""A minimal, trainable diffusion-routed MoE block for splicing into a
PRETRAINED, otherwise-frozen HF causal LM's single decoder layer — the
"bounded pilot" from the Phase 1 investigation (see
reports/phase1_findings_report.md §3.3 and recommendation 4), testing
whether a TRAINED router + experts can close the gap between diffusion-only
(untrained) routing and the oracle specialization ceiling found at layers 2
and 4, without committing to full Phase 2 pretraining.

Unlike models/moe_layer.py's DiffusionMoELayer (which owns its own
RoPEMultiHeadAttention and replaces a WHOLE TransformerBlock in the
project's own from-scratch architecture), this block replaces ONLY a
pretrained decoder layer's `.mlp` submodule — attention and every other
layer stay frozen and pretrained; only this block's own parameters
(centroids + expert FFNs) are trained. Its forward(x) -> Tensor signature
matches what HF's MistralDecoderLayer.mlp expects, so splicing in is just
`layer.mlp = PilotMoEBlock(...)`.

Auxiliary routing info (router_logits, centroids, Psi_landmarks — needed by
training.losses.total_loss's load-balance and centroid-separation terms)
isn't returned directly, since the surrounding frozen decoder layer only
expects a plain tensor back — it's stashed on `self.last_aux` after every
forward call instead, for the training script to read out as
`{layer_idx: block.last_aux}` and pass straight to total_loss.

Clusters/fits the diffusion map on COSINE-normalized activations by
default, consistent with the investigation's finding that raw-Euclidean
distance is dominated by outlier-norm (attention-sink) tokens at several
layers, including some of these pilot's own candidate layers.
"""

from __future__ import annotations

import torch
from torch import nn

from diffusion_moe.geometry.multiscale import l2_normalize
from diffusion_moe.geometry.nystrom import NystromDiffusionMap
from diffusion_moe.models.expert_ffn import ExpertFFN
from diffusion_moe.models.moe_dispatch import dispatch_and_aggregate
from diffusion_moe.routing.centroids import ExpertCentroids
from diffusion_moe.routing.router import DiffusionRouter
from diffusion_moe.routing.separation import landmark_scale


class PilotMoEBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        n_experts: int,
        top_k: int,
        n_components: int = 32,
        n_landmarks: int = 128,
        diffusion_t: int = 3,
        alpha: float = 1.0,
        tau: float = 0.1,
        overlap_factor: float = 1.0,
        centroid_refresh_steps: int = 20,
        cosine: bool = True,
        random_state: int = 42,
        noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.n_components = n_components
        self.centroid_refresh_steps = centroid_refresh_steps
        self.cosine = cosine

        self.ndm = NystromDiffusionMap(
            n_landmarks=n_landmarks,
            n_components=n_components,
            t=diffusion_t,
            alpha=alpha,
            random_state=random_state,
        )
        self.centroids = ExpertCentroids(
            n_experts=n_experts, n_components=n_components, random_state=random_state
        )
        self.router = DiffusionRouter(n_experts=n_experts, top_k=top_k, tau=tau, noise_std=noise_std)
        self.experts = nn.ModuleList(
            [
                ExpertFFN(d_model, ffn_dim, n_experts, overlap_factor=overlap_factor)
                for _ in range(n_experts)
            ]
        )

        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))
        self.last_aux: dict[str, torch.Tensor] | None = None

    def _compute_diffusion_coords(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, seq, d_model). Returns Psi_t: (batch, seq,
        n_components). Refits the NystromDiffusionMap (landmarks +
        eigensystem) every centroid_refresh_steps calls during training;
        reuses the frozen landmarks via Nystrom extension in between —
        identical pattern to DiffusionMoELayer, just parameterized with a
        much smaller refresh interval since a pilot run has far fewer total
        steps than a full training run.
        """
        batch, seq_len, d_model = z.shape
        z_flat = z.detach().reshape(-1, d_model).float().cpu().numpy()
        if self.cosine:
            z_flat = l2_normalize(z_flat)

        should_refresh = int(self._step.item()) % self.centroid_refresh_steps == 0
        if should_refresh:
            psi_flat = self.ndm.fit_transform(z_flat)
        else:
            psi_flat = self.ndm.transform(z_flat)

        if self.training:
            self._step += 1

        psi = torch.from_numpy(psi_flat).to(dtype=z.dtype, device=z.device)
        return psi.reshape(batch, seq_len, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Psi_t = self._compute_diffusion_coords(x)

        if not self.centroids.is_initialised:
            self.centroids.initialise_from_batch(Psi_t)

        psi_landmarks = torch.from_numpy(self.ndm.psi_landmarks_).to(
            dtype=x.dtype, device=x.device
        )
        # Route on SCALE-NORMALIZED coordinates, not raw Psi_t. Real
        # diffusion coordinates (Psi_t = eigenvector * eigenvalue^t) can be
        # vanishingly small in absolute terms — verified on real Mistral-7B
        # activations at |Psi_t| ~ 1e-7 for diffusion_t=3 — which makes ANY
        # fixed, O(1)-scale tau (e.g. 0.1, inherited from base_config.yaml)
        # give a softmax that's numerically indistinguishable from uniform
        # regardless of which centroid is actually closest: a real training
        # run confirmed exactly this (tempered dispatch weights read exactly
        # 1/n_experts for every expert, on every token, for the entire run).
        # Dividing by scale (the typical landmark-to-landmark spacing) makes
        # tau operate in a consistent, dimensionless unit instead of this
        # layer's own — and, empirically, unpredictable — raw coordinate
        # magnitude.
        scale = landmark_scale(psi_landmarks)
        Psi_t_scaled = Psi_t / scale
        centroids_scaled = self.centroids() / scale
        psi_landmarks_scaled = psi_landmarks / scale

        gate_values, expert_indices, router_logits = self.router(Psi_t_scaled, centroids_scaled)
        output = dispatch_and_aggregate(x, gate_values, expert_indices, self.experts)

        # aux carries the SCALED versions throughout, so total_loss's
        # "untempered" load-balance softmax (which has no knowledge of tau or
        # this scale fix) and centroid_separation_loss both operate on the
        # same consistently-scaled coordinates the router itself used —
        # centroid_separation_loss's own scale-normalization is unaffected
        # (it's already scale-invariant by construction, so normalizing
        # twice is harmless).
        self.last_aux = {
            "router_logits": router_logits,
            "gate_values": gate_values,
            "expert_indices": expert_indices,
            "Psi_t": Psi_t_scaled,
            "centroids": centroids_scaled,
            "Psi_landmarks": psi_landmarks_scaled,
        }
        return output
