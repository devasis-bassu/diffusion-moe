"""Cosine-similarity baseline MoE layer — routes by similarity in the raw
d_model residual-stream space rather than diffusion coordinates, isolating
whether the diffusion-map geometry specifically helps versus just having
*some* learned geometric notion of token-to-expert similarity."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from diffusion_moe.models.attention import RoPEMultiHeadAttention
from diffusion_moe.models.expert_ffn import ExpertFFN
from diffusion_moe.models.moe_dispatch import dispatch_and_aggregate
from diffusion_moe.models.rmsnorm import RMSNorm
from diffusion_moe.routing.router import DiffusionRouter


class CosineMoELayer(nn.Module):
    """Pre-norm block: RMSNorm -> RoPE attention -> residual, then a
    cosine-similarity-routed mixture of ExpertFFNs -> residual.

    Routes on cosine_similarity(z_i, centroid_k) for K learned centroid
    vectors living directly in d_model space (unlike DiffusionMoELayer's
    centroids, which live in the n_components-dimensional diffusion
    coordinate space) — softmaxed with temperature tau and top-k gated via
    the same top_k_gate used by DiffusionRouter.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        n_experts: int,
        top_k: int,
        tau: float = 0.1,
        ffn_dim: int | None = None,
        overlap_factor: float = 1.0,
        rope_base: int = 10000,
        norm_eps: float = 1e-5,
        dropout: float = 0.0,
        use_shared_expert: bool = True,
    ) -> None:
        super().__init__()
        ffn_dim = ffn_dim if ffn_dim is not None else 4 * d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.tau = tau

        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        self.attn = RoPEMultiHeadAttention(
            d_model, num_heads, max_seq_len, rope_base=rope_base, dropout=dropout
        )
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)
        self.centroids = nn.Parameter(torch.randn(n_experts, d_model) * 0.02)
        self.experts = nn.ModuleList(
            [
                ExpertFFN(d_model, ffn_dim, n_experts, overlap_factor=overlap_factor)
                for _ in range(n_experts)
            ]
        )
        # Same width/overlap_factor as a routed expert, applied unconditionally
        # every token/step, no gate value -- see DiffusionMoELayer's docstring
        # for the full motivation. None when use_shared_expert=False.
        self.shared_expert = (
            ExpertFFN(d_model, ffn_dim, n_experts, overlap_factor=overlap_factor)
            if use_shared_expert
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z = self.attn(self.attn_norm(x), positions, mask)
        x = x + z

        z_normed = F.normalize(z, dim=-1)
        centroids_normed = F.normalize(self.centroids, dim=-1)
        router_logits = z_normed @ centroids_normed.t()  # (batch, seq, n_experts) cosine sims

        weights = F.softmax(router_logits / self.tau, dim=-1)
        gate_values, expert_indices = DiffusionRouter.top_k_gate(weights, self.top_k)

        ffn_input = self.ffn_norm(x)
        expert_out = dispatch_and_aggregate(ffn_input, gate_values, expert_indices, self.experts)
        output = x + expert_out
        if self.shared_expert is not None:
            output = output + self.shared_expert(ffn_input)

        aux = {
            "router_logits": router_logits,
            "gate_values": gate_values,
            "expert_indices": expert_indices,
        }
        return output, aux
