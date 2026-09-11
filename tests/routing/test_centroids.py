"""Tests for ExpertCentroids k-means++ initialisation."""

import torch

from diffusion_moe.routing.centroids import ExpertCentroids

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
