"""Tests for the dyadic multiscale diffusion map analysis.

Validates the core claim behind it: a single kernel bandwidth can be wrong
in either direction (too small -> fragmentation, too large -> collapse), so
these tests check that sweeping a dyadic ladder (1) recovers the known
Swiss-roll intrinsic dimension in a stable window rather than at one lucky
scale, (2) flags genuine disconnection from norm outliers at small scales
and shows it resolving as the scale grows, and (3) shows that switching the
metric to cosine/angular distance (via l2_normalize) removes that
disconnection entirely when the outliers differ from the bulk only in raw
norm, not direction — the same claim about attention-sink-style tokens made
in the extract_geometry.py investigation this module grew out of.
"""

import numpy as np
from sklearn.datasets import make_swiss_roll

from diffusion_moe.geometry.multiscale import (
    dyadic_eps_ladder,
    find_stable_window,
    l2_normalize,
    multiscale_diffusion_analysis,
)

SEED = 0


def test_l2_normalize_gives_unit_norm_rows():
    rng = np.random.RandomState(SEED)
    Z = rng.randn(50, 8) * rng.uniform(0.1, 1000, size=(50, 1))
    normalized = l2_normalize(Z)
    norms = np.linalg.norm(normalized, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_l2_normalize_preserves_direction():
    rng = np.random.RandomState(SEED)
    Z = rng.randn(50, 8)
    normalized = l2_normalize(Z)
    cos_sim = np.sum(Z * normalized, axis=1) / np.linalg.norm(Z, axis=1)
    np.testing.assert_allclose(cos_sim, 1.0, atol=1e-10)


def test_dyadic_eps_ladder_centered_and_monotonic():
    ladder = dyadic_eps_ladder(eps_center=1.0, n_scales=7, base=2.0)
    assert len(ladder) == 7
    assert ladder[3] == 1.0  # centered: k=0 term is exactly eps_center
    assert ladder == sorted(ladder)  # monotonically increasing
    np.testing.assert_allclose(ladder, [2.0**k for k in range(-3, 4)])


def test_dyadic_eps_ladder_rejects_invalid_input():
    import pytest

    with pytest.raises(ValueError):
        dyadic_eps_ladder(eps_center=1.0, n_scales=0)
    with pytest.raises(ValueError):
        dyadic_eps_ladder(eps_center=0.0, n_scales=5)


def test_swiss_roll_has_stable_window_at_intrinsic_dim_two():
    X, _ = make_swiss_roll(n_samples=1000, noise=0.05, random_state=SEED)

    result = multiscale_diffusion_analysis(
        X, n_landmarks=200, n_components=10, t=5, n_scales=7, cosine=False,
        random_state=SEED,
    )
    scales = result["scales"]

    # The scale nearest the median heuristic (eps_ratio == 1) should match
    # the point-estimate result already established in test_swiss_roll.py.
    center = next(s for s in scales if s["eps_ratio_to_median"] == 1.0)
    assert center["r_star"] == 2
    assert not center["likely_disconnected"]

    # And it shouldn't be a lucky one-off: a real plateau of >= 2 adjacent
    # scales should agree on r*=2, exactly the kind of stability multiscale
    # analysis is meant to distinguish from a single arbitrary eps choice.
    window = find_stable_window(scales, min_run=2)
    assert len(window) >= 2
    assert all(scales[i]["r_star"] == 2 for i in window)


def test_disconnection_from_norm_outliers_resolves_at_larger_scale():
    rng = np.random.RandomState(SEED)
    main_cluster = rng.randn(195, 8)
    outliers = rng.randn(5, 8) * 0.1 + np.array([12, 0, 0, 0, 0, 0, 0, 0])
    Z = np.vstack([main_cluster, outliers])

    result = multiscale_diffusion_analysis(
        Z, n_landmarks=32, n_components=5, t=3, n_scales=7, cosine=False,
        random_state=SEED,
    )
    scales = result["scales"]

    # Smallest scale tested: bandwidth is far too small to bridge the
    # outliers, so the graph should show the disconnection signature.
    assert scales[0]["likely_disconnected"]
    # Largest scale tested: bandwidth has grown well past the outlier gap,
    # so the components should have merged back into one connected graph.
    assert not scales[-1]["likely_disconnected"]


def test_cosine_metric_resolves_disconnection_from_norm_only_outliers():
    """Same-direction, different-norm outliers (the attention-sink /
    massive-activation pattern) disconnect the raw-Euclidean kernel graph at
    every scale small enough to resolve the bulk cluster's own structure,
    but should NOT disconnect the cosine/angular kernel graph at all, since
    normalizing to unit norm discards the one axis (raw magnitude) the
    outliers differ on.
    """
    rng = np.random.RandomState(SEED)
    direction = np.zeros(8)
    direction[0] = 1.0
    main_cluster = direction + 0.05 * rng.randn(195, 8)
    outliers = 1000 * direction + 0.05 * rng.randn(5, 8)
    Z = np.vstack([main_cluster, outliers])

    euclidean = multiscale_diffusion_analysis(
        Z, n_landmarks=32, n_components=5, t=3, n_scales=5, cosine=False,
        random_state=SEED,
    )
    cosine = multiscale_diffusion_analysis(
        Z, n_landmarks=32, n_components=5, t=3, n_scales=5, cosine=True,
        random_state=SEED,
    )

    assert any(s["likely_disconnected"] for s in euclidean["scales"])
    assert not any(s["likely_disconnected"] for s in cosine["scales"])


def test_find_stable_window_picks_longest_non_disconnected_run():
    scales = [
        {"r_star": 1, "likely_disconnected": True},
        {"r_star": 5, "likely_disconnected": False},
        {"r_star": 3, "likely_disconnected": False},
        {"r_star": 3, "likely_disconnected": False},
        {"r_star": 3, "likely_disconnected": False},
        {"r_star": 8, "likely_disconnected": False},
    ]
    window = find_stable_window(scales, min_run=2)
    assert window == [2, 3, 4]


def test_find_stable_window_empty_when_all_disconnected():
    scales = [{"r_star": 1, "likely_disconnected": True} for _ in range(5)]
    assert find_stable_window(scales, min_run=2) == []
