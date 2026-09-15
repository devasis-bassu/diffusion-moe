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


def test_noise_std_zero_reproduces_exact_prior_behavior():
    """Default noise_std=0.0 must be a pure no-op -- existing callers
    (DiffusionMoELayer, PilotMoEBlock) and every test above must see
    identical behaviour to before noisy gating was added."""
    router = DiffusionRouter(n_experts=N_EXPERTS, top_k=2, tau=0.5, noise_std=0.0)
    Psi_t, centroids = _inputs()
    router.train()

    _, _, router_logits = router(Psi_t, centroids)
    manual = -torch.cdist(Psi_t, centroids.unsqueeze(0).expand(BATCH, -1, -1)) ** 2
    assert torch.allclose(router_logits, manual, atol=1e-4)


def test_noise_not_applied_in_eval_mode():
    """Noisy gating is a training-time-only regularisation technique (Shazeer
    et al., 2017) -- eval-mode routing must stay deterministic regardless of
    noise_std, since noise at inference would make dispatch non-reproducible
    for no benefit."""
    router = DiffusionRouter(n_experts=N_EXPERTS, top_k=2, tau=0.5, noise_std=5.0)
    Psi_t, centroids = _inputs()
    router.eval()

    torch.manual_seed(0)
    _, _, logits_a = router(Psi_t, centroids)
    torch.manual_seed(1)
    _, _, logits_b = router(Psi_t, centroids)

    assert torch.equal(logits_a, logits_b)


def test_noise_applied_in_training_mode_varies_across_calls():
    router = DiffusionRouter(n_experts=N_EXPERTS, top_k=2, tau=0.5, noise_std=5.0)
    Psi_t, centroids = _inputs()
    router.train()

    torch.manual_seed(0)
    _, _, logits_a = router(Psi_t, centroids)
    torch.manual_seed(1)
    _, _, logits_b = router(Psi_t, centroids)

    assert not torch.equal(logits_a, logits_b)


def test_noise_reduces_load_imbalance_from_a_biased_centroid_layout():
    """The whole point of noisy gating: when one expert centroid sits closer
    to the tokens than the rest (the "early leader" pattern a real pilot run
    showed -- k-means++ initialising a centroid close to the data mode,
    which then dominates the softmax for the entire run), noise should
    measurably reduce how concentrated dispatch is, on average across many
    i.i.d. noise draws -- not eliminate the imbalance (that's what the
    separate load-balance loss is for), just soften it.

    Controlled geometry, not random centroids: every non-winner expert is
    placed at an identical, exact squared distance (via orthogonal unit-axis
    offsets, so cross terms vanish) so the "winner's edge" is a known,
    tunable quantity relative to tau -- avoids the trap of random centroid
    placement either saturating the softmax completely (no room left for
    noise to matter) or barely biasing it at all (nothing to reduce),
    depending on the draw.
    """
    n_experts = 8
    n_tokens = 500  # many i.i.d. per-token noise draws at an identical point
    tau = 0.5
    gap = 0.7071  # gap^2 = 0.5 -> a real but non-saturating edge at this tau

    Psi_t = torch.zeros(1, n_tokens, N_COMPONENTS)
    centroids = torch.zeros(n_experts, N_COMPONENTS)
    for i in range(1, n_experts):
        centroids[i, i] = gap  # orthogonal offset: sq_dist = gap^2 for every non-winner

    def max_expert_share(router: DiffusionRouter, seed: int) -> float:
        torch.manual_seed(seed)
        weights = torch.nn.functional.softmax(
            router(Psi_t, centroids)[2] / router.tau, dim=-1
        )
        return float(weights.mean(dim=(0, 1)).max())

    clean_router = DiffusionRouter(n_experts=n_experts, top_k=2, tau=tau, noise_std=0.0)
    noisy_router = DiffusionRouter(n_experts=n_experts, top_k=2, tau=tau, noise_std=1.0)
    clean_router.train()
    noisy_router.train()

    clean_shares = [max_expert_share(clean_router, seed) for seed in range(5)]
    noisy_shares = [max_expert_share(noisy_router, seed) for seed in range(5)]

    assert sum(noisy_shares) / len(noisy_shares) < sum(clean_shares) / len(clean_shares)
