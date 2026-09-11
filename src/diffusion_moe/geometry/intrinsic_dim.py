"""Intrinsic dimensionality estimation from diffusion map eigenvalues."""

from __future__ import annotations

import numpy as np


def estimate_intrinsic_dim(
    eigenvalues: np.ndarray, t: int, threshold: float = 0.95
) -> int:
    """Estimates the intrinsic dimension r* via the energy-retention formula.

    Each diffusion coordinate i carries "energy" lambda_i^(2t) (the variance
    contributed by that coordinate at diffusion time t). r* is the smallest
    number of leading components whose cumulative energy reaches `threshold`
    of the total energy.

    `eigenvalues` should already exclude the trivial top eigenvalue (~1) of the
    Markov matrix, e.g. NystromDiffusionMap.eigenvalues_.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    energy = np.clip(eigenvalues, a_min=0.0, a_max=None) ** (2 * t)

    total = energy.sum()
    if total <= 0:
        return len(eigenvalues)

    cumulative_ratio = np.cumsum(energy) / total
    r_star = int(np.searchsorted(cumulative_ratio, threshold) + 1)
    return min(r_star, len(eigenvalues))


def spectral_gap(eigenvalues: np.ndarray) -> float:
    """Returns lambda_1 - lambda_2, the gap between the two leading eigenvalues.

    A large gap after the r*-th eigenvalue indicates the remaining components
    are noise rather than additional manifold structure.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    return float(eigenvalues[0] - eigenvalues[1])
