"""Tests for ExpertCentroids k-means++ initialisation."""

import torch

from diffusion_moe.routing.centroids import ExpertCentroids
from diffusion_moe.routing.separation import centroid_separation_loss, landmark_scale

N_EXPERTS, N_COMPONENTS = 8, 32


def test_initial_shape_before_init():
    centroids = ExpertCentroids(n_experts=N_EXPERTS, n_components=N_COMPONENTS)
    assert centroids.centroids.shape == (N_EXPERTS, N_COMPONENTS)
    assert not centroids.is_initialised


def test_initialise_from_batch_sets_flag_and_shape():
    centroids = ExpertCentroids(n_experts=N_EXPERTS, n_components=N_COMPONENTS)
    Psi_t = torch.randn(4, 16, N_COMPONENTS)  # (batch, seq, n_components)
    centroids.initialise_from_batch(Psi_t)

    assert centroids.is_initialised
    assert centroids.centroids.shape == (N_EXPERTS, N_COMPONENTS)


def test_seeded_centroids_match_actual_batch_points():
    """kmeans++ seeding picks real data points as centers, so every centroid
    should exactly match some input token.

    Distance is computed via direct elementwise subtraction, not torch.cdist:
    cdist's squared-distance-expansion algorithm (||a||^2 - 2a.b + ||b||^2)
    suffers float32 catastrophic cancellation for near-zero distances, turning
    an exact match into a deceptive ~1e-3 apparent distance after the sqrt.
    """
    centroids = ExpertCentroids(n_experts=N_EXPERTS, n_components=N_COMPONENTS)
    Psi_t = torch.randn(4, 16, N_COMPONENTS)
    centroids.initialise_from_batch(Psi_t)

    flat = Psi_t.reshape(-1, N_COMPONENTS)
    for row in centroids.centroids:
        max_abs_diff = (flat - row.unsqueeze(0)).abs().amax(dim=-1)
        assert max_abs_diff.min().item() == 0.0


def test_second_call_is_a_noop():
    centroids = ExpertCentroids(n_experts=N_EXPERTS, n_components=N_COMPONENTS)
    Psi_t_a = torch.randn(4, 16, N_COMPONENTS)
    centroids.initialise_from_batch(Psi_t_a)
    values_after_first = centroids.centroids.detach().clone()

    Psi_t_b = torch.randn(4, 16, N_COMPONENTS) * 100  # very different batch
    centroids.initialise_from_batch(Psi_t_b)

    assert torch.equal(centroids.centroids.detach(), values_after_first)


def test_handles_fewer_points_than_experts():
    centroids = ExpertCentroids(n_experts=N_EXPERTS, n_components=N_COMPONENTS)
    Psi_t = torch.randn(1, 3, N_COMPONENTS)  # only 3 tokens, but 8 experts
    centroids.initialise_from_batch(Psi_t)
    assert centroids.centroids.shape == (N_EXPERTS, N_COMPONENTS)
    assert torch.all(torch.isfinite(centroids.centroids))


def test_gradients_flow_through_centroids():
    centroids = ExpertCentroids(n_experts=N_EXPERTS, n_components=N_COMPONENTS)
    loss = centroids.centroids.sum()
    loss.backward()
    assert centroids.centroids.grad is not None
    assert torch.all(centroids.centroids.grad == 1.0)


def test_forward_returns_centroids_tensor():
    centroids = ExpertCentroids(n_experts=N_EXPERTS, n_components=N_COMPONENTS)
    assert centroids() is centroids.centroids


def test_clip_norm_caps_only_centroids_exceeding_max_norm():
    centroids = ExpertCentroids(n_experts=4, n_components=3)
    with torch.no_grad():
        centroids.centroids.copy_(
            torch.tensor(
                [
                    [3.0, 0.0, 0.0],  # norm 3 -> exceeds cap, should shrink
                    [10.0, 0.0, 0.0],  # norm 10 -> exceeds cap, should shrink
                    [0.5, 0.0, 0.0],  # norm 0.5 -> within cap, unchanged
                    [0.0, 0.0, 0.0],  # norm 0 -> unchanged, no divide-by-zero
                ]
            )
        )

    centroids.clip_norm_(max_norm=2.0)
    norms = centroids.centroids.norm(dim=-1)

    assert torch.isclose(norms[0], torch.tensor(2.0), atol=1e-5)
    assert torch.isclose(norms[1], torch.tensor(2.0), atol=1e-5)
    assert torch.isclose(norms[2], torch.tensor(0.5), atol=1e-5)  # untouched
    assert torch.isclose(norms[3], torch.tensor(0.0), atol=1e-5)  # untouched


def test_clip_norm_preserves_direction():
    centroids = ExpertCentroids(n_experts=1, n_components=3)
    with torch.no_grad():
        centroids.centroids.copy_(torch.tensor([[3.0, 4.0, 0.0]]))  # norm 5

    centroids.clip_norm_(max_norm=1.0)

    direction = centroids.centroids / centroids.centroids.norm(dim=-1, keepdim=True)
    expected_direction = torch.tensor([[0.6, 0.8, 0.0]])
    assert torch.allclose(direction, expected_direction, atol=1e-5)


def test_clip_norm_is_a_no_op_gradient_context():
    """clip_norm_ mutates .data directly and shouldn't require or produce
    autograd tracking -- callable straight after optimizer.step() with no
    extra no_grad() wrapping needed at the call site."""
    centroids = ExpertCentroids(n_experts=2, n_components=3)
    with torch.no_grad():
        centroids.centroids.copy_(torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]]))

    centroids.clip_norm_(max_norm=1.0)  # no torch.no_grad() at call site

    assert centroids.centroids.requires_grad
    assert torch.allclose(centroids.centroids.norm(dim=-1), torch.tensor([1.0, 1.0]), atol=1e-5)


def test_clip_norm_prevents_the_runaway_growth_a_real_pilot_run_hit():
    """Direct reproduction of the failure mode found in the first real
    training run: repeatedly maximizing centroid_separation_loss's gradient
    with no counter-force grows centroid norms without bound (by design —
    see test_separation.py's test_more_spread_centroids_give_more_negative_loss).
    Periodic clip_norm_ (as scripts/pilot_finetune.py now does after every
    optimizer step) should keep norms bounded near the data's own scale
    instead.
    """
    landmarks = torch.randn(50, N_COMPONENTS, generator=torch.Generator().manual_seed(0))
    scale = landmark_scale(landmarks)

    centroids_unclipped = ExpertCentroids(n_experts=4, n_components=N_COMPONENTS)
    centroids_clipped = ExpertCentroids(n_experts=4, n_components=N_COMPONENTS)
    with torch.no_grad():
        centroids_clipped.centroids.copy_(centroids_unclipped.centroids)

    for _ in range(200):
        for centroids, do_clip in [(centroids_unclipped, False), (centroids_clipped, True)]:
            loss = centroid_separation_loss(centroids.centroids, landmarks)
            loss.backward()
            with torch.no_grad():
                centroids.centroids -= 5.0 * centroids.centroids.grad  # plain gradient step
                centroids.centroids.grad = None
            if do_clip:
                centroids.clip_norm_(max_norm=3.0 * scale)

    unclipped_max_norm = centroids_unclipped.centroids.norm(dim=-1).max()
    clipped_max_norm = centroids_clipped.centroids.norm(dim=-1).max()

    assert unclipped_max_norm > 5 * scale  # reproduces the unbounded blowup
    assert clipped_max_norm <= 3.0 * scale + 1e-4  # stays within the cap
