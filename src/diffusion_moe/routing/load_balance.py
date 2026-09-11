"""Load-balancing auxiliary loss for expert routing."""

from __future__ import annotations

import torch


def coefficient_of_variation_loss(gate_values: torch.Tensor, n_experts: int) -> torch.Tensor:
    """CV^2 load-balance penalty: the squared coefficient of variation of the
    total gating mass routed to each expert.

    gate_values: (..., n_experts) dense per-expert gate weights — e.g. the
    router's softmax distribution over all experts before top-k sparsification.
    The last dimension must be exactly n_experts so each column can be
    attributed to a specific expert; pass the dense distribution, not the
    sparse top-k output.

    A perfectly balanced router (equal average load on every expert) gives
    CV^2 = 0; a collapsed router (all mass on one expert) approaches its
    maximum, n_experts - 1.
    """
    if gate_values.shape[-1] != n_experts:
        raise ValueError(
            f"gate_values last dim ({gate_values.shape[-1]}) must equal n_experts "
            f"({n_experts}) — pass the dense per-expert distribution, not the "
            "sparse top-k output."
        )

    load = gate_values.reshape(-1, n_experts).mean(dim=0)  # (n_experts,)
    mean = load.mean()
    std = load.std(unbiased=False)
    cv = std / (mean + 1e-8)
    return cv**2


def switch_load_balance_loss(
    router_probs: torch.Tensor, expert_indices: torch.Tensor, n_experts: int
) -> torch.Tensor:
    """The original Switch Transformer auxiliary load-balance loss (Fedus et
    al., 2021): n_experts * sum_i f_i * P_i, where f_i is the fraction of
    (token, slot) assignments actually dispatched to expert i (the hard
    top-k decision) and P_i is the mean router probability — the dense
    softmax, before top-k — assigned to expert i. Both f and P are uniform
    (1/n_experts each) when routing is perfectly balanced, giving the loss
    its minimum value of 1; it grows as dispatch and/or probability mass
    concentrate onto fewer experts. Reduces to the paper's original top-1
    formula when top_k=1.

    router_probs: (..., n_experts) dense softmax distribution (pre-top-k).
    expert_indices: (..., top_k) the selected expert ids (post-top-k).
    """
    flat_probs = router_probs.reshape(-1, n_experts)
    flat_indices = expert_indices.reshape(-1)

    P = flat_probs.mean(dim=0)  # (n_experts,) mean router probability per expert
    dispatch_counts = torch.bincount(flat_indices, minlength=n_experts).float()
    f = dispatch_counts / dispatch_counts.sum()  # fraction of dispatches per expert

    return n_experts * torch.sum(f * P)
