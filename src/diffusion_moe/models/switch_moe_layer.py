"""Switch Transformer baseline MoE layer — a standard learned linear router,
isolating what (if anything) the diffusion-geometric routing contributes
over the field's existing learned-routing approach."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from diffusion_moe.models.attention import RoPEMultiHeadAttention
from diffusion_moe.models.expert_ffn import ExpertFFN
from diffusion_moe.models.moe_dispatch import dispatch_and_aggregate
from diffusion_moe.models.rmsnorm import RMSNorm
from diffusion_moe.routing.load_balance import switch_load_balance_loss
from diffusion_moe.routing.router import DiffusionRouter


class SwitchMoELayer(nn.Module):
    """Pre-norm block: RMSNorm -> RoPE attention -> residual, then a
    Switch-Transformer-routed mixture of ExpertFFNs -> residual.

    The router is a single learned linear projection to n_experts logits,
    softmax, top-k — no geometric or distance-based structure at all. The
    standard Switch auxiliary load-balance loss (Fedus et al., 2021) is
    computed here and exposed via aux["switch_aux_loss"], in addition to the
    same dense router_logits/gate_values/expert_indices every other MoE
    layer variant returns (so the generic CV^2 load-balance/total_loss
    pipeline still applies uniformly across variants for comparison).
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
        self.router = nn.Linear(d_model, n_experts, bias=False)
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

        router_logits = self.router(z)  # (batch, seq, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        gate_values, expert_indices = DiffusionRouter.top_k_gate(router_probs, self.top_k)

        switch_aux_loss = switch_load_balance_loss(router_probs, expert_indices, self.n_experts)

        ffn_input = self.ffn_norm(x)
        expert_out = dispatch_and_aggregate(ffn_input, gate_values, expert_indices, self.experts)
        output = x + expert_out

        aux = {
            "router_logits": router_logits,
            "gate_values": gate_values,
            "expert_indices": expert_indices,
            "switch_aux_loss": switch_aux_loss,
        }
        return output, aux
