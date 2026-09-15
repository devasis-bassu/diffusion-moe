"""Diffusion-routed mixture-of-experts transformer layer."""

from __future__ import annotations

import torch
from torch import nn

from diffusion_moe.geometry.nystrom import NystromDiffusionMap
from diffusion_moe.models.attention import RoPEMultiHeadAttention
from diffusion_moe.models.expert_ffn import ExpertFFN
from diffusion_moe.models.moe_dispatch import dispatch_and_aggregate
from diffusion_moe.models.rmsnorm import RMSNorm
from diffusion_moe.routing.centroids import ExpertCentroids
from diffusion_moe.routing.router import DiffusionRouter
from diffusion_moe.routing.separation import landmark_scale


class DiffusionMoELayer(nn.Module):
    """Pre-norm block: RMSNorm -> RoPE attention -> residual, then a
    diffusion-routed mixture of ExpertFFNs -> residual, in place of a single
    dense FFN.

    Per-token routing works in diffusion coordinate space rather than the raw
    residual stream: post-attention activations z are mapped to diffusion
    coordinates Psi_t via a NystromDiffusionMap fit on this batch's tokens
    (landmarks/eigensystem are re-fit every `centroid_refresh_steps` calls and
    reused via Nystrom extension in between, since re-fitting is the expensive
    step), then routed to the nearest expert centroids in that space.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        n_experts: int,
        top_k: int,
        n_components: int,
        n_landmarks: int,
        diffusion_t: int = 3,
        alpha: float = 1.0,
        tau: float = 0.1,
        ffn_dim: int | None = None,
        overlap_factor: float = 1.0,
        rope_base: int = 10000,
        norm_eps: float = 1e-5,
        dropout: float = 0.0,
        centroid_refresh_steps: int = 500,
        noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        ffn_dim = ffn_dim if ffn_dim is not None else 4 * d_model

        self.n_experts = n_experts
        self.top_k = top_k
        self.n_components = n_components
        self.centroid_refresh_steps = centroid_refresh_steps

        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        self.attn = RoPEMultiHeadAttention(
            d_model, num_heads, max_seq_len, rope_base=rope_base, dropout=dropout
        )
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)

        self.ndm = NystromDiffusionMap(
            n_landmarks=n_landmarks, n_components=n_components, t=diffusion_t, alpha=alpha
        )
        self.centroids = ExpertCentroids(n_experts=n_experts, n_components=n_components)
        self.router = DiffusionRouter(n_experts=n_experts, top_k=top_k, tau=tau, noise_std=noise_std)
        self.experts = nn.ModuleList(
            [
                ExpertFFN(d_model, ffn_dim, n_experts, overlap_factor=overlap_factor)
                for _ in range(n_experts)
            ]
        )

        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

    def _compute_diffusion_coords(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, seq, d_model) post-attention activations. Returns
        Psi_t: (batch, seq, n_components). Refits the NystromDiffusionMap
        (landmarks + eigensystem) every centroid_refresh_steps calls; reuses
        the frozen landmarks via Nystrom extension in between. sklearn-backed,
        so this always runs on detached CPU float64 arrays, regardless of the
        model's device/dtype.
        """
        batch, seq_len, d_model = z.shape
        z_flat = z.detach().reshape(-1, d_model).cpu().numpy()

        should_refresh = int(self._step.item()) % self.centroid_refresh_steps == 0
        if should_refresh:
            psi_flat = self.ndm.fit_transform(z_flat)
        else:
            psi_flat = self.ndm.transform(z_flat)

        if self.training:
            self._step += 1

        psi = torch.from_numpy(psi_flat).to(dtype=z.dtype, device=z.device)
        return psi.reshape(batch, seq_len, -1)

    def _dispatch_and_aggregate(
        self, x: torch.Tensor, gate_values: torch.Tensor, expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """Routes tokens to experts and aggregates weighted outputs — shared
        with every other MoE layer variant (see moe_dispatch.py)."""
        return dispatch_and_aggregate(x, gate_values, expert_indices, self.experts)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z = self.attn(self.attn_norm(x), positions, mask)
        x = x + z

        Psi_t = self._compute_diffusion_coords(z)

        if not self.centroids.is_initialised:
            self.centroids.initialise_from_batch(Psi_t)

        psi_landmarks = torch.from_numpy(self.ndm.psi_landmarks_).to(
            dtype=x.dtype, device=x.device
        )
        # Route on SCALE-NORMALIZED coordinates, not raw Psi_t — real
        # diffusion coordinates (Psi_t = eigenvector * eigenvalue^t) can be
        # vanishingly small in absolute terms (verified on real Mistral-7B
        # activations at |Psi_t| ~ 1e-7 for diffusion_t=3), which makes ANY
        # fixed, O(1)-scale tau give a softmax numerically indistinguishable
        # from uniform regardless of which centroid is actually closest — a
        # real training run of this exact mechanism (via pilot_finetune.py's
        # PilotMoEBlock, which shares this router) confirmed exactly that
        # failure mode. Dividing by scale (the typical landmark-to-landmark
        # spacing) makes tau operate in a consistent, dimensionless unit
        # instead of this layer's own raw coordinate magnitude.
        scale = landmark_scale(psi_landmarks)
        Psi_t_scaled = Psi_t / scale
        centroids_scaled = self.centroids() / scale
        psi_landmarks_scaled = psi_landmarks / scale

        gate_values, expert_indices, router_logits = self.router(Psi_t_scaled, centroids_scaled)

        ffn_input = self.ffn_norm(x)
        expert_out = self._dispatch_and_aggregate(ffn_input, gate_values, expert_indices)
        output = x + expert_out

        aux = {
            "router_logits": router_logits,
            "gate_values": gate_values,
            "expert_indices": expert_indices,
            "Psi_t": Psi_t_scaled,
            "centroids": centroids_scaled,
            "Psi_landmarks": psi_landmarks_scaled,
        }
        return output, aux
