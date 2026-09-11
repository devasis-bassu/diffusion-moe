"""Tests for estimate_intrinsic_dim and spectral_gap."""

import numpy as np

from diffusion_moe.geometry.intrinsic_dim import estimate_intrinsic_dim, spectral_gap


def test_spectral_gap_basic():
    eigenvalues = np.array([0.9, 0.85, 0.3, 0.1])
    assert np.isclose(spectral_gap(eigenvalues), 0.05)


def test_estimate_intrinsic_dim_two_dominant_components():
    # Two components carry almost all the energy at t=1 -> r* should be 2.
    eigenvalues = np.array([0.99, 0.98, 0.05, 0.02, 0.01])
    r_star = estimate_intrinsic_dim(eigenvalues, t=1, threshold=0.95)
    assert r_star == 2


def test_estimate_intrinsic_dim_all_components_needed_for_high_threshold():
    eigenvalues = np.array([0.5, 0.5, 0.5, 0.5])
    r_star = estimate_intrinsic_dim(eigenvalues, t=1, threshold=0.999)
    assert r_star == 4


def test_estimate_intrinsic_dim_higher_t_concentrates_energy():
    eigenvalues = np.array([0.99, 0.9, 0.5, 0.3])
    r_star_low_t = estimate_intrinsic_dim(eigenvalues, t=1, threshold=0.9)
    r_star_high_t = estimate_intrinsic_dim(eigenvalues, t=10, threshold=0.9)
    assert r_star_high_t <= r_star_low_t


def test_estimate_intrinsic_dim_handles_all_zero_eigenvalues():
    eigenvalues = np.zeros(5)
    r_star = estimate_intrinsic_dim(eigenvalues, t=1, threshold=0.95)
    assert r_star == 5
