"""Rotary positional embeddings (RoPE) with interleaved dimension pairing."""

from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Precomputes RoPE cos/sin tables and rotates Q/K with interleaved pairing.

    Frequencies follow theta_j = base^(-2j / head_dim) for j in [0, head_dim/2)
    (Su et al., 2021 — RoFormer). Dimensions are paired as (0,1), (2,3), ... and
    each pair is rotated by theta_j at the token's position.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: int = 10000) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        self.head_dim = head_dim
        self.base = base

        j = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = base ** (-j / head_dim)  # theta_j, shape (head_dim / 2,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._cached_seq_len = 0
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, head_dim / 2)
        cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1)  # (seq_len, head_dim)
        sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        self._cached_seq_len = seq_len

    @staticmethod
    def _rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
        """Maps (x0, x1, x2, x3, ...) -> (-x1, x0, -x3, x2, ...)."""
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def _cos_sin(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        max_pos = int(positions.max().item()) + 1 if positions.numel() > 0 else 0
        if max_pos > self._cached_seq_len:
            self._build_cache(max_pos)
        cos = self.cos_cached[positions]  # (..., head_dim)
        sin = self.sin_cached[positions]
        return cos, sin

    def apply_rope(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotates q and k by their position-dependent angles.

        q, k: (batch, num_heads, seq_len, head_dim)
        positions: (batch, seq_len) integer position ids
        """
        cos, sin = self._cos_sin(positions)
        cos = cos.unsqueeze(1).to(dtype=q.dtype)  # (batch, 1, seq_len, head_dim)
        sin = sin.unsqueeze(1).to(dtype=q.dtype)

        q_rot = q * cos + self._rotate_interleaved(q) * sin
        k_rot = k * cos + self._rotate_interleaved(k) * sin
        return q_rot, k_rot
