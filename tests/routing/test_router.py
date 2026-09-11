"""Tests for DiffusionRouter, including the tau -> 0 hard-routing limit."""

import torch

from diffusion_moe.routing.router import DiffusionRouter

BATCH, SEQ_LEN, N_COMPONENTS, N_EXPERTS = 2, 16, 32, 8


def _inputs():
    Psi_t = torch.randn(BATCH, SEQ_LEN, N_COMPONENTS)
    centroids = torch.randn(N_EXPERTS, N_COMPONENTS)
    return Psi_t, centroids


def test_output_shapes():
    router = DiffusionRouter(n_experts=N_EXPERTS, top_k=2, tau=0.5)
    Psi_t, centroids = _inputs()
    gate_values, expert_indices, router_logits = router(Psi_t, centroids)

    assert gate_values.shape == (BATCH, SEQ_LEN, 2)
    assert expert_indices.shape == (BATCH, SEQ_LEN, 2)
    assert expert_indices.dtype == torch.long
    assert router_logits.shape == (BATCH, SEQ_LEN, N_EXPERTS)


def test_gate_values_renormalised_to_sum_to_one():
    router = DiffusionRouter(n_experts=N_EXPERTS, top_k=3, tau=0.5)
    Psi_t, centroids = _inputs()
    gate_values, _, _ = router(Psi_t, centroids)
    sums = gate_values.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_router_logits_match_manual_negative_squared_distance():
    router = DiffusionRouter(n_experts=N_EXPERTS, top_k=1, tau=0.5)
    Psi_t, centroids = _inputs()
    _, _, router_logits = router(Psi_t, centroids)

    manual = -torch.cdist(Psi_t, centroids.unsqueeze(0).expand(BATCH, -1, -1)) ** 2
    assert torch.allclose(router_logits, manual, atol=1e-4)


def test_top_k_gate_picks_largest_weights():
    weights = torch.tensor([[0.1, 0.6, 0.05, 0.25]])
    gate_values, expert_indices = DiffusionRouter.top_k_gate(weights, k=2)
    assert expert_indices.tolist() == [[1, 3]]
    assert torch.allclose(gate_values.sum(dim=-1), torch.ones(1))
    assert gate_values[0, 0] > gate_values[0, 1]


def test_tau_to_zero_assigns_all_weight_to_nearest_centroid():
    """As tau -> 0, softmax(logits / tau) hardens to a one-hot on the argmax
    logit, i.e. the nearest centroid — this is the routing sanity check the
    whole diffusion-routing design depends on."""
    torch.manual_seed(0)
    Psi_t = torch.randn(BATCH, SEQ_LEN, N_COMPONENTS)
    centroids = torch.randn(N_EXPERTS, N_COMPONENTS)

    router = DiffusionRouter(n_experts=N_EXPERTS, top_k=1, tau=1e-6)
    gate_values, expert_indices, _ = router(Psi_t, centroids)

    # ground truth: nearest centroid by brute-force Euclidean distance
    dists = torch.cdist(Psi_t, centroids.unsqueeze(0).expand(BATCH, -1, -1))
    expected_nearest = dists.argmin(dim=-1)

    assert torch.equal(expert_indices.squeeze(-1), expected_nearest)
    assert torch.allclose(gate_values.squeeze(-1), torch.ones(BATCH, SEQ_LEN), atol=1e-4)


def test_rejects_top_k_greater_than_n_experts():
    import pytest

    with pytest.raises(ValueError):
        DiffusionRouter(n_experts=4, top_k=5)
