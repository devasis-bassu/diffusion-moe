"""Tests for coifman_lafon_normalise."""

import numpy as np

from diffusion_moe.geometry.kernel import gaussian_kernel
from diffusion_moe.geometry.markov import coifman_lafon_normalise


def _random_kernel(n=8, seed=0):
    Z = np.random.RandomState(seed).randn(n, 3)
    return gaussian_kernel(Z, eps=2.0)


def test_rows_sum_to_one():
    K = _random_kernel()
    for alpha in (0.0, 0.5, 1.0):
        P = coifman_lafon_normalise(K, alpha=alpha)
        assert np.allclose(P.sum(axis=1), 1.0)


def test_output_shape_matches_input():
    K = _random_kernel(n=6)
    P = coifman_lafon_normalise(K, alpha=1.0)
    assert P.shape == K.shape


def test_alpha_changes_normalisation_on_nonuniform_density():
    """With non-uniformly-spaced points, alpha=1 (density-independent) should
    differ from alpha=0 (no density correction)."""
    Z = np.array([[0.0], [0.1], [0.2], [5.0], [5.1], [5.2]])
    K = gaussian_kernel(Z, eps=1.0)
    P0 = coifman_lafon_normalise(K, alpha=0.0)
    P1 = coifman_lafon_normalise(K, alpha=1.0)
    assert not np.allclose(P0, P1)


def test_nonnegative_entries():
    K = _random_kernel()
    P = coifman_lafon_normalise(K, alpha=1.0)
    assert np.all(P >= 0)
