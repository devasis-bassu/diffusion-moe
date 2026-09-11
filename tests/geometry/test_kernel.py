"""Tests for gaussian_kernel, gaussian_kernel_cross, bandwidth_median_heuristic."""

import numpy as np

from diffusion_moe.geometry.kernel import (
    bandwidth_median_heuristic,
    gaussian_kernel,
    gaussian_kernel_cross,
)


def test_gaussian_kernel_diagonal_is_one():
    Z = np.random.RandomState(0).randn(10, 3)
    K = gaussian_kernel(Z, eps=1.0)
    assert np.allclose(np.diag(K), 1.0)


def test_gaussian_kernel_symmetric_and_bounded():
    Z = np.random.RandomState(0).randn(10, 3)
    K = gaussian_kernel(Z, eps=1.0)
    assert np.allclose(K, K.T)
    assert np.all(K > 0)
    assert np.all(K <= 1.0 + 1e-12)


def test_smaller_eps_decays_faster():
    Z = np.array([[0.0, 0.0], [1.0, 0.0]])
    K_small_eps = gaussian_kernel(Z, eps=0.1)
    K_large_eps = gaussian_kernel(Z, eps=10.0)
    assert K_small_eps[0, 1] < K_large_eps[0, 1]


def test_bandwidth_median_heuristic_matches_manual_median():
    Z = np.array([[0.0], [1.0], [3.0]])
    # pairwise sq dists: (0,1)->1, (0,3)->9, (1,3)->4 ; median = 4
    eps = bandwidth_median_heuristic(Z)
    assert np.isclose(eps, 4.0)


def test_bandwidth_median_heuristic_handles_identical_points():
    Z = np.zeros((5, 2))
    eps = bandwidth_median_heuristic(Z)
    assert eps > 0  # guarded against a zero bandwidth


def test_gaussian_kernel_cross_shape():
    X = np.random.RandomState(0).randn(4, 3)
    Y = np.random.RandomState(1).randn(6, 3)
    K = gaussian_kernel_cross(X, Y, eps=1.0)
    assert K.shape == (4, 6)


def test_gaussian_kernel_cross_matches_gaussian_kernel_when_same_set():
    Z = np.random.RandomState(0).randn(5, 3)
    K_full = gaussian_kernel(Z, eps=2.0)
    K_cross = gaussian_kernel_cross(Z, Z, eps=2.0)
    assert np.allclose(K_full, K_cross)
