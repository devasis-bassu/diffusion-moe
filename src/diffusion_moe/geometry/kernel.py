"""Gaussian kernel and bandwidth selection for diffusion maps."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform


def gaussian_kernel(Z: np.ndarray, eps: float) -> np.ndarray:
    """Pairwise Gaussian kernel K_ij = exp(-||z_i - z_j||^2 / eps).

    Z: (n, d) array of points. eps: kernel bandwidth (the denominator, i.e. 2*sigma^2).
    Returns K: (n, n) symmetric PSD kernel matrix.
    """
    sq_dists = squareform(pdist(Z, metric="sqeuclidean"))
    return np.exp(-sq_dists / eps)


def gaussian_kernel_cross(X: np.ndarray, Y: np.ndarray, eps: float) -> np.ndarray:
    """Cross Gaussian kernel between two point sets: K_ij = exp(-||x_i - y_j||^2 / eps).

    Used by the Nystrom extension to compare new points X against fixed landmarks Y.
    Returns K: (n_X, n_Y).
    """
    sq_dists = cdist(X, Y, metric="sqeuclidean")
    return np.exp(-sq_dists / eps)


def bandwidth_median_heuristic(Z: np.ndarray) -> float:
    """Sets eps to the median pairwise squared Euclidean distance (excluding
    self-distances) — the standard median heuristic for Gaussian kernel bandwidth.
    """
    sq_dists = pdist(Z, metric="sqeuclidean")
    eps = float(np.median(sq_dists))
    if eps <= 0:
        eps = float(np.finfo(np.float64).eps)
    return eps
