"""Multi-head self-attention with rotary positional embeddings."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from diffusion_moe.models.rope import RotaryEmbedding


class RoPEMultiHeadAttention(nn.Module):
    """Projects to Q/K/V, applies RoPE to Q and K, then scaled dot-product attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int | None = None,
        rope_base: int = 10000,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else d_model // num_heads
        if self.head_dim * num_heads == 0:
            raise ValueError("num_heads and head_dim must both be positive")
        inner_dim = self.num_heads * self.head_dim

        self.q_proj = nn.Linear(d_model, inner_dim, bias=False)
        self.k_proj = nn.Linear(d_model, inner_dim, bias=False)
        self.v_proj = nn.Linear(d_model, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, d_model, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len, base=rope_base)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: (batch, seq, d_model). positions: (batch, seq). mask: additive/bool
        attention mask broadcastable to (batch, num_heads, seq, seq); if None, a
        causal mask is used (standard for pretraining)."""
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.rotary.apply_rope(q, k, positions)

        dropout_p = self.dropout if self.training else 0.0
        if mask is not None:
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=False
            )
        else:
            attn_out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, is_causal=True
            )

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out_proj(attn_out)
