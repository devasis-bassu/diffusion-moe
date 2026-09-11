"""Pre-norm block: RMSNorm -> RoPE attention -> residual -> RMSNorm -> SwiGLU FFN -> residual."""

from __future__ import annotations

import torch
from torch import nn

from diffusion_moe.models.attention import RoPEMultiHeadAttention
from diffusion_moe.models.ffn import FeedForward
from diffusion_moe.models.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        ffn_dim: int | None = None,
        rope_base: int = 10000,
        norm_eps: float = 1e-5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        self.attn = RoPEMultiHeadAttention(
            d_model, num_heads, max_seq_len, rope_base=rope_base, dropout=dropout
        )
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)
        self.ffn = FeedForward(d_model, hidden_dim=ffn_dim)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), positions, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x
