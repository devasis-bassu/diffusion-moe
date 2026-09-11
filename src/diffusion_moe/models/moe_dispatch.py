"""Token-to-expert dispatch/aggregation shared by every MoE layer variant
(DiffusionMoELayer, CosineMoELayer, SwitchMoELayer, RandomMoELayer)."""

from __future__ import annotations

import torch
from torch import nn


def dispatch_and_aggregate(
    x: torch.Tensor,
    gate_values: torch.Tensor,
    expert_indices: torch.Tensor,
    experts: nn.ModuleList,
) -> torch.Tensor:
    """Routes each token to its top-k selected experts and aggregates their
    weighted outputs.

    Tokens are gathered into contiguous per-expert buckets (so each expert
    runs once over a batch of its assigned tokens, rather than once per
    token) via index_select, then scattered back to their original positions.

    x: (batch, seq, d_model). gate_values, expert_indices: (batch, seq, top_k).
    """
    batch, seq_len, d_model = x.shape
    top_k = expert_indices.shape[-1]
    n_experts = len(experts)
    n_tokens = batch * seq_len

    flat_x = x.reshape(n_tokens, d_model)
    flat_gate = gate_values.reshape(n_tokens * top_k)
    flat_expert = expert_indices.reshape(n_tokens * top_k)

    # Gather: each token's row repeated once per selected slot, in order.
    token_ids = torch.arange(n_tokens, device=x.device).repeat_interleave(top_k)
    tokens_rep = flat_x.index_select(0, token_ids)

    # Gather into contiguous per-expert buckets.
    sort_order = torch.argsort(flat_expert)
    sorted_tokens = tokens_rep.index_select(0, sort_order)
    counts = torch.bincount(flat_expert, minlength=n_experts)
    offsets = torch.cumsum(counts, dim=0)

    sorted_output = torch.empty_like(sorted_tokens)
    start = 0
    for expert_id in range(n_experts):
        end = int(offsets[expert_id].item())
        if end > start:
            sorted_output[start:end] = experts[expert_id](sorted_tokens[start:end])
        start = end

    # Scatter back to the original (repeat_interleave) slot order.
    output_rep = torch.empty_like(sorted_tokens)
    output_rep.scatter_(0, sort_order.unsqueeze(-1).expand(-1, d_model), sorted_output)

    # Weight by gate value and sum each token's top_k contributions.
    weighted = output_rep * flat_gate.unsqueeze(-1)
    weighted = weighted.reshape(n_tokens, top_k, d_model)
    return weighted.sum(dim=1).reshape(batch, seq_len, d_model)
