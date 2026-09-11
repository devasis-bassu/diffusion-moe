"""Randomly-routed baseline MoE layer — isolates whether routing (of any
kind) matters at all, by assigning experts uniformly at random per token."""

from __future__ import annotations

import torch
from torch import nn

from diffusion_moe.models.attention import RoPEMultiHeadAttention
from diffusion_moe.models.expert_ffn import ExpertFFN
from diffusion_moe.models.moe_dispatch import dispatch_and_aggregate
from diffusion_moe.models.rmsnorm import RMSNorm


class RandomMoELayer(nn.Module):
    """Pre-norm block: RMSNorm -> RoPE attention -> residual, then a
    uniform-random mixture of ExpertFFNs -> residual. No learned router
    parameters: each token's top-k experts are drawn uniformly at random
    (via top-k of i.i.d. uniform scores, equivalent to sampling k of n
    experts without replacement), with equal 1/top_k gate weight.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        n_experts: int,
        top_k: int,
        ffn_dim: int | None = None,
        overlap_factor: float = 1.0,
        rope_base: int = 10000,
        norm_eps: float = 1e-5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        ffn_dim = ffn_dim if ffn_dim is not None else 4 * d_model
        self.n_experts = n_experts
        self.top_k = top_k

        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        self.attn = RoPEMultiHeadAttention(
            d_model, num_heads, max_seq_len, rope_base=rope_base, dropout=dropout
        )
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)
        self.experts = nn.ModuleList(
            [
                ExpertFFN(d_model, ffn_dim, n_experts, overlap_factor=overlap_factor)
                for _ in range(n_experts)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z = self.attn(self.attn_norm(x), positions, mask)
        x = x + z

        batch, seq_len, _ = z.shape
        random_scores = torch.rand(batch, seq_len, self.n_experts, device=x.device)
        _, expert_indices = torch.topk(random_scores, self.top_k, dim=-1)
        gate_values = torch.full(
            (batch, seq_len, self.top_k), 1.0 / self.top_k, device=x.device, dtype=x.dtype
        )

        # Dense logits for the load-balance loss: uniform (zeros -> uniform
        # softmax) is the "balanced by construction" baseline for this router.
        router_logits = torch.zeros(batch, seq_len, self.n_experts, device=x.device, dtype=x.dtype)

        ffn_input = self.ffn_norm(x)
        expert_out = dispatch_and_aggregate(ffn_input, gate_values, expert_indices, self.experts)
        output = x + expert_out

        aux = {
            "router_logits": router_logits,
            "gate_values": gate_values,
            "expert_indices": expert_indices,
        }
        return output, aux
