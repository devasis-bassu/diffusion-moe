"""Per-expert SwiGLU feed-forward network, sized for a mixture of experts."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ExpertFFN(nn.Module):
    """SwiGLU FFN for one expert: down_proj(silu(gate_proj(x)) * up_proj(x)).

    Same forward API as FeedForward, but sized relative to the dense FFN width
    it replaces: hidden_dim = ffn_dim // n_experts * overlap_factor. At
    overlap_factor=1, K experts together cost about the same FLOPs per token as
    one dense FFN of width ffn_dim (since only top_k of K experts fire per
    token); overlap_factor > 1 gives each expert extra capacity at the cost of
    more active FLOPs per token.
    """

    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        n_experts: int,
        overlap_factor: float = 1.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        hidden_dim = max(1, int(ffn_dim // n_experts * overlap_factor))
        self.hidden_dim = hidden_dim
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
