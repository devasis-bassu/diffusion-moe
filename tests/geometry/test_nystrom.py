"""Tests for NystromDiffusionMap."""

import numpy as np

from diffusion_moe.geometry.nystrom import NystromDiffusionMap


def _make_data(n=200, seed=0):
    return np.random.RandomState(seed).randn(n, 5)


def test_fit_transform_shape():
    Z = _make_data(n=200)
    ndm = NystromDiffusionMap(n_landmarks=32, n_components=10, t=3, alpha=1.0)
    Psi = ndm.fit_transform(Z)
    assert Psi.shape == (200, 10)
    assert np.all(np.isfinite(Psi))


def test_eigenvalues_and_eigenvectors_exclude_trivial_component():
    Z = _make_data(n=200)
    ndm = NystromDiffusionMap(n_landmarks=32, n_components=10, t=3, alpha=1.0)
    ndm.fit(Z)
    assert ndm.eigenvalues_.shape == (10,)
    assert ndm.eigenvectors_.shape == (32, 10)
    # the trivial eigenvalue (~1) has been dropped
    assert ndm.eigenvalues_[0] < 1.0 - 1e-6


def test_landmarks_clipped_to_n_when_fewer_points_than_landmarks():
    Z = _make_data(n=10)
    ndm = NystromDiffusionMap(n_landmarks=128, n_components=3, t=1, alpha=1.0)
    ndm.fit(Z)
    assert ndm.landmarks_.shape[0] == 10


def test_transform_before_fit_raises():
    ndm = NystromDiffusionMap(n_landmarks=32, n_components=10)
    import pytest

    with pytest.raises(RuntimeError):
        ndm.transform(_make_data(n=5))


def test_transform_reproduces_fit_transform_on_same_data():
    """Nystrom extension applied to the training points should closely match
    fit_transform's own output (it's the same formula, called twice)."""
    Z = _make_data(n=150)
    ndm = NystromDiffusionMap(n_landmarks=40, n_components=5, t=2, alpha=1.0)
    Psi_fit = ndm.fit_transform(Z)
    Psi_again = ndm.transform(Z)
    assert np.allclose(Psi_fit, Psi_again)


def test_eigenvalues_decrease_diffusion_time_shrinks_coordinates():
    Z = _make_data(n=150)
    ndm_t1 = NystromDiffusionMap(n_landmarks=40, n_components=5, t=1, alpha=1.0, random_state=0)
    ndm_t5 = NystromDiffusionMap(n_landmarks=40, n_components=5, t=5, alpha=1.0, random_state=0)
    Psi_t1 = ndm_t1.fit_transform(Z)
    Psi_t5 = ndm_t5.fit_transform(Z)
    # eigenvalues < 1, so raising diffusion time shrinks coordinate magnitude
    assert np.abs(Psi_t5).mean() < np.abs(Psi_t1).mean()
