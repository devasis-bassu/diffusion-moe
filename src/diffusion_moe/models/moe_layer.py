"""Diffusion-routed mixture-of-experts transformer layer."""

from __future__ import annotations

import torch
from torch import nn

from diffusion_moe.geometry.multiscale import l2_normalize
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

    In addition to the n_experts routed experts, every token also passes
    through one mandatory shared_expert that sits entirely outside the
    router -- no gate value, no centroid, unconditionally applied every
    step regardless of what the diffusion router decides. Motivated by a
    real observation from the all-layers training run (see
    phase1_findings_report.md): with every layer routed, 23 of 24 layers
    showed real (if partial) load imbalance simultaneously, and task_loss
    plateaued well above where a comparably-sized dense model should land.
    Precedented in production MoE architectures (DeepSeekMoE's "shared
    expert isolation", similarly in Qwen2-MoE) for exactly this reason: a
    purely sparsely-routed layer has no path guaranteed consistent gradient
    signal every step, since everything is contingent on routing decisions
    -- common/generic computation ends up redundantly relearned by whichever
    expert wins the routing lottery that step, or not learned reliably at
    all if routing is noisy. The shared expert is architecturally identical
    to a routed ExpertFFN (same width, same overlap_factor) -- it's the
    *unconditional* application, not a different architecture, that makes
    it "shared". This does add real active compute per token (top_k+1
    experts fire instead of top_k), not a reallocation of existing capacity.
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
        cosine: bool = False,
        grad_accum_steps: int = 1,
    ) -> None:
        super().__init__()
        ffn_dim = ffn_dim if ffn_dim is not None else 4 * d_model

        self.n_experts = n_experts
        self.top_k = top_k
        self.n_components = n_components
        self.centroid_refresh_steps = centroid_refresh_steps
        # _step (below) increments once per forward() call, but Trainer
        # calls forward() once per MICRO-batch -- grad_accum_steps times per
        # logged/optimizer step, not once. Every other *_steps config value
        # (checkpoint_steps, eval_steps, log_steps) is measured in optimizer-
        # step units; without this, centroid_refresh_steps silently wasn't --
        # with the project's grad_accum_steps=2 default, refits were firing
        # every 250 logged steps, not the configured 500, for this entire
        # investigation (found by a sharp catch: a repeating loss-disruption
        # pattern visible at 250-step intervals that didn't match any
        # configured cadence). See _compute_diffusion_coords.
        self.grad_accum_steps = grad_accum_steps
        # Whether to fit/transform the diffusion map on L2-normalized (cosine
        # kernel) rather than raw activations. Off by default: the full
        # 32-layer multiscale sweep (phase1_findings_report.md §2.4) found
        # cosine is NOT a strict improvement -- raw Euclidean reports a
        # higher intrinsic dimension than cosine at 13 of 29 comparable
        # layers, meaning cosine discards real magnitude information at a
        # meaningful fraction of layers. The recommended usage (§7,
        # recommendation 2) is selective, not uniform: enable this only for
        # layers that actually need it -- confirmed to be 17, 20, and 31 for
        # Mistral-7B, the three layers that never connect under raw Euclidean
        # at ANY bandwidth tested, not just this kill-switch's default one.
        # PilotMoEBlock (the pilot fine-tune's version of this same block)
        # already has this option, defaulting to True there for unrelated
        # reasons specific to that one-layer splice; this brings the
        # production layer up to the same capability without inheriting that
        # default, matching the selective recommendation.
        self.cosine = cosine

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
        # Same width/overlap_factor as a routed expert -- see the class
        # docstring for why this exists and why "shared" means unconditional
        # application, not a different architecture.
        self.shared_expert = ExpertFFN(d_model, ffn_dim, n_experts, overlap_factor=overlap_factor)

        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

    def _compute_diffusion_coords(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, seq, d_model) post-attention activations. Returns
        Psi_t: (batch, seq, n_components). Refits the NystromDiffusionMap
        (landmarks + eigensystem) every centroid_refresh_steps *logged/
        optimizer* steps (centroid_refresh_steps * grad_accum_steps forward
        calls, since forward() runs once per micro-batch); reuses the frozen
        landmarks via Nystrom extension in between. sklearn-backed,
        so this always runs on detached CPU float64 arrays, regardless of the
        model's device/dtype.
        """
        batch, seq_len, d_model = z.shape
        # .float() before .numpy(): NumPy has no bfloat16 dtype, so this
        # crashes outright under real bf16 mixed-precision training (the
        # project's own default precision) without it -- found by actually
        # running a real step through Trainer, not caught by any prior
        # training-free diagnostic or by the pilot (whose separate
        # DtypeCastWrapper/PilotMoEBlock already cast to fp32 for other
        # reasons, incidentally avoiding this exact crash).
        z_flat = z.detach().reshape(-1, d_model).float().cpu().numpy()
        if self.cosine:
            z_flat = l2_normalize(z_flat)

        # self.ndm's fitted state (landmarks_, eigenvectors_, eps_, ...) lives
        # on a plain Python object, not an nn.Module -- it's never part of
        # state_dict(), so it does NOT survive a checkpoint save/load. Only
        # _step (a registered buffer) does. A fresh model loaded from a
        # checkpoint therefore has the *correct* _step but landmarks_ is None
        # -- without this check, should_refresh would be False whenever
        # _step isn't exactly on a centroid_refresh_steps boundary (the
        # common case), and .transform() would crash outright on the very
        # first forward pass: real training resume with an active MoE layer,
        # and any post-hoc analysis of a saved checkpoint (e.g.
        # scripts/expert_attribution.py), both hit this identically. Missed
        # by the original resume regression test because it used a dense
        # model (layers_to_replace=[]), never touching this path at all.
        should_refresh = (
            self.ndm.landmarks_ is None
            or int(self._step.item()) % (self.centroid_refresh_steps * self.grad_accum_steps) == 0
        )
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
        # Unconditional: every token, every step, no gate value -- entirely
        # outside the router's influence (see class docstring).
        shared_out = self.shared_expert(ffn_input)
        output = x + expert_out + shared_out

        aux = {
            "router_logits": router_logits,
            "gate_values": gate_values,
            "expert_indices": expert_indices,
            "Psi_t": Psi_t_scaled,
            "centroids": centroids_scaled,
            "Psi_landmarks": psi_landmarks_scaled,
        }
        return output, aux
