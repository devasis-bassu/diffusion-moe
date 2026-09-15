"""Tests for diffusion_eigenvectors."""

from unittest.mock import patch

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence

from diffusion_moe.geometry.eigensolver import diffusion_eigenvectors
from diffusion_moe.geometry.kernel import gaussian_kernel
from diffusion_moe.geometry.markov import coifman_lafon_normalise


def _markov_matrix(n=50, seed=0):
    Z = np.random.RandomState(seed).randn(n, 3)
    K = gaussian_kernel(Z, eps=2.0)
    return coifman_lafon_normalise(K, alpha=1.0)


def test_shapes():
    P = _markov_matrix(n=50)
    eigenvalues, eigenvectors = diffusion_eigenvectors(P, n_components=10)
    assert eigenvalues.shape == (10,)
    assert eigenvectors.shape == (50, 10)


def test_top_eigenvalue_is_one():
    """A row-stochastic matrix always has 1 as its largest eigenvalue."""
    P = _markov_matrix(n=50)
    eigenvalues, _ = diffusion_eigenvectors(P, n_components=10)
    assert np.isclose(eigenvalues[0], 1.0, atol=1e-6)


def test_eigenvalues_sorted_descending():
    P = _markov_matrix(n=50)
    eigenvalues, _ = diffusion_eigenvectors(P, n_components=10)
    assert np.all(np.diff(eigenvalues) <= 1e-10)


def test_eigenvalues_bounded_by_one():
    P = _markov_matrix(n=50)
    eigenvalues, _ = diffusion_eigenvectors(P, n_components=10)
    assert np.all(eigenvalues <= 1.0 + 1e-6)


def test_caps_components_when_more_requested_than_available():
    P = _markov_matrix(n=5)
    eigenvalues, eigenvectors = diffusion_eigenvectors(P, n_components=10)
    # ARPACK needs k < n - 1, so at most n - 2 = 3 eigenpairs come back
    assert eigenvalues.shape[0] == 3
    assert eigenvectors.shape == (5, 3)


def test_dense_fallback_for_tiny_matrix():
    P = _markov_matrix(n=2)
    eigenvalues, eigenvectors = diffusion_eigenvectors(P, n_components=10)
    assert eigenvalues.shape[0] == 2
    assert eigenvectors.shape == (2, 2)


def test_dense_fallback_when_arpack_fails_to_converge():
    """Reproduces a real crash from a full 32-layer multiscale bandwidth
    sweep: at some scales in the dyadic eps ladder, P's eigenspectrum becomes
    near-degenerate/tightly clustered enough that ARPACK's iterative Lanczos
    method (scipy.sparse.linalg.eigs) raises ArpackNoConvergence instead of
    returning a result -- which previously crashed the whole sweep uncaught,
    losing 7 of 32 layers' worth of real GPU compute. ARPACK failing here
    isn't reproducible deterministically, so this mocks the failure directly
    to confirm the dense fallback actually engages and still returns a
    correctly-shaped, valid result.
    """
    P = _markov_matrix(n=50)
    with patch(
        "diffusion_moe.geometry.eigensolver.eigs",
        side_effect=ArpackNoConvergence("no convergence", eigenvalues=[], eigenvectors=[]),
    ):
        eigenvalues, eigenvectors = diffusion_eigenvectors(P, n_components=10)

    assert eigenvalues.shape == (10,)
    assert eigenvectors.shape == (50, 10)
    assert np.isclose(eigenvalues[0], 1.0, atol=1e-6)
