"""Tests for centroid_separation_loss."""

import torch

from diffusion_moe.routing.separation import centroid_separation_loss

N_EXPERTS, N_COMPONENTS, N_LANDMARKS = 4, 8, 50


def _landmarks(seed=0):
    return torch.randn(N_LANDMARKS, N_COMPONENTS, generator=torch.Generator().manual_seed(seed))


def test_collapsed_centroids_give_zero_loss():
    centroids = torch.ones(N_EXPERTS, N_COMPONENTS)  # all identical
    loss = centroid_separation_loss(centroids, _landmarks())
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_more_spread_centroids_give_more_negative_loss():
    landmarks = _landmarks()
    tight = torch.randn(N_EXPERTS, N_COMPONENTS) * 0.01
    spread = torch.randn(N_EXPERTS, N_COMPONENTS) * 5.0

    loss_tight = centroid_separation_loss(tight, landmarks)
    loss_spread = centroid_separation_loss(spread, landmarks)
    assert loss_spread < loss_tight  # more separation -> more negative loss


def test_loss_is_negative_for_nontrivial_centroids():
    centroids = torch.randn(N_EXPERTS, N_COMPONENTS)
    loss = centroid_separation_loss(centroids, _landmarks())
    assert loss.item() < 0.0


def test_scale_invariance_when_centroids_and_landmarks_scale_together():
    """The landmark-spread normalisation should make the loss invariant to a
    uniform rescaling of the whole diffusion coordinate space."""
    landmarks = _landmarks()
    centroids = torch.randn(N_EXPERTS, N_COMPONENTS)

    loss_base = centroid_separation_loss(centroids, landmarks)
    loss_scaled = centroid_separation_loss(centroids * 10.0, landmarks * 10.0)
    assert torch.isclose(loss_base, loss_scaled, atol=1e-4)


def test_gradients_flow_to_centroids():
    centroids = torch.randn(N_EXPERTS, N_COMPONENTS, requires_grad=True)
    loss = centroid_separation_loss(centroids, _landmarks())
    loss.backward()
    assert centroids.grad is not None
    assert torch.any(centroids.grad != 0)
