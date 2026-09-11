"""Coifman-Lafon anisotropic density normalisation for diffusion maps."""

from __future__ import annotations

import numpy as np


def coifman_lafon_normalise(K: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Applies the alpha-density normalisation, then row-normalises to a Markov matrix.

    K: (n, n) kernel matrix (e.g. from gaussian_kernel).
    alpha: density-normalisation strength (Coifman & Lafon, 2006). alpha=0 gives
        the classical graph-Laplacian normalisation; alpha=1 approximates the
        Laplace-Beltrami operator on the manifold independent of the sampling
        density; alpha=0.5 gives the Fokker-Planck diffusion.

    Returns P: (n, n) row-stochastic Markov transition matrix. P is reversible
    w.r.t. its stationary distribution, so it is similar to a symmetric matrix
    and has real, non-negative eigenvalues.
    """
    d = K.sum(axis=1)
    d_alpha = np.power(d, alpha)
    K_alpha = K / np.outer(d_alpha, d_alpha)

    row_sums = K_alpha.sum(axis=1, keepdims=True)
    P = K_alpha / row_sums
    return P
