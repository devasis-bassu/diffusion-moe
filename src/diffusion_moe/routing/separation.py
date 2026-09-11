"""Centroid separation auxiliary loss for expert routing."""

from __future__ import annotations

import torch


def centroid_separation_loss(centroids: torch.Tensor, Psi_landmarks: torch.Tensor) -> torch.Tensor:
    """Negative mean pairwise diffusion distance between expert centroids —
    minimising this loss pushes centroids apart so experts specialise on
    distinct regions of diffusion space rather than collapsing together.

    centroids: (n_experts, n_components). Psi_landmarks: (n_landmarks,
    n_components) — the Nystrom landmark diffusion coordinates. The diffusion
    metric itself is just Euclidean distance in diffusion coordinates (the
    defining property of diffusion maps); Psi_landmarks is used to read off the
    typical landmark-to-landmark spread at this layer and normalise by it, so
    the loss stays on a comparable scale across layers and training progress
    instead of growing unboundedly with raw centroid distance.
    """
    n_experts = centroids.shape[0]
    centroid_dists = torch.cdist(centroids, centroids, p=2)
    centroid_mask = ~torch.eye(n_experts, dtype=torch.bool, device=centroids.device)
    mean_centroid_dist = centroid_dists[centroid_mask].mean()

    n_landmarks = Psi_landmarks.shape[0]
    landmark_dists = torch.cdist(Psi_landmarks, Psi_landmarks, p=2)
    landmark_mask = ~torch.eye(n_landmarks, dtype=torch.bool, device=Psi_landmarks.device)
    scale = landmark_dists[landmark_mask].mean() + 1e-8

    return -mean_centroid_dist / scale
