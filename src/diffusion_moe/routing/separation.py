"""Centroid separation auxiliary loss for expert routing."""

from __future__ import annotations

import torch


def landmark_scale(Psi_landmarks: torch.Tensor) -> torch.Tensor:
    """Mean pairwise landmark-to-landmark distance — the natural scale of
    this layer's diffusion coordinates. Used both to normalise
    centroid_separation_loss (below) and, separately, to cap how far
    ExpertCentroids.clip_norm_ lets centroids drift from that scale: the loss
    itself has no upper bound on raw centroid separation (verified by
    test_separation.py's test_more_spread_centroids_give_more_negative_loss,
    intentionally, so changing that here isn't the fix), so nothing about the
    loss stops training from pushing centroids arbitrarily far outside the
    range real data actually occupies — a real 300-step pilot run reproduced
    exactly this: centroid_separation_loss grew from -3 to -19,185 while the
    load-balance loss simultaneously read exactly 0.0 the entire run,
    consistent with the router's distances becoming centroid-norm-dominated
    (every token roughly equidistant from every runaway-scale centroid) and
    collapsing toward uniform, uninformative dispatch. clip_norm_ is the
    external safeguard for that failure mode; this loss's formula is
    unchanged.

    The epsilon guarding against a literal zero mean distance (landmarks
    exactly coincident) is deliberately tiny (1e-30), not the more
    conventional 1e-8: this function is also used (via PilotMoEBlock /
    DiffusionMoELayer's router-input rescaling) on raw diffusion coordinates
    whose TRUE scale can itself be smaller than 1e-8 — real Mistral-7B
    activations measured at ~1e-7, and small eigenvalues raised to even a
    modest diffusion_t push this smaller still. A 1e-8 epsilon would then
    silently dominate the result instead of guarding it, capping how much
    rescaling actually happens and defeating the fix it's used for.
    """
    n_landmarks = Psi_landmarks.shape[0]
    landmark_dists = torch.cdist(Psi_landmarks, Psi_landmarks, p=2)
    landmark_mask = ~torch.eye(n_landmarks, dtype=torch.bool, device=Psi_landmarks.device)
    return landmark_dists[landmark_mask].mean() + 1e-30


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

    scale = landmark_scale(Psi_landmarks)

    return -mean_centroid_dist / scale
