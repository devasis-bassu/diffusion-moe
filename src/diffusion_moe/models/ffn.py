"""SwiGLU feed-forward network."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FeedForward(nn.Module):
    """SwiGLU variant: down_proj(silu(gate_proj(x)) * up_proj(x))."""

    def __init__(self, d_model: int, hidden_dim: int | None = None, bias: bool = False) -> None:
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else 4 * d_model
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
