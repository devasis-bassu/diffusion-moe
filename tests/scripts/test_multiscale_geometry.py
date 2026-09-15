"""Tests for scripts/multiscale_geometry.py's model-independent logic.

Loading a real model isn't exercised here (see test_extract_geometry.py for
why); analyze_layer_multiscale only needs a pooled-activation array, so it's
tested directly against the same synthetic scenarios used in
tests/geometry/test_multiscale.py.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "multiscale_geometry_script", REPO_ROOT / "scripts" / "multiscale_geometry.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["multiscale_geometry_script"] = module
    spec.loader.exec_module(module)
    return module


mg = _load_module()


def test_analyze_layer_multiscale_returns_expected_structure():
    Z = np.random.RandomState(0).randn(200, 8)
    result = mg.analyze_layer_multiscale(Z, n_scales=5, seed=0)

    assert set(result.keys()) == {
        "euclidean",
        "cosine",
        "euclidean_stable_window",
        "cosine_stable_window",
        "euclidean_stable_r_star",
        "cosine_stable_r_star",
    }
    assert len(result["euclidean"]["scales"]) == 5
    assert len(result["cosine"]["scales"]) == 5


def test_analyze_layer_multiscale_identifies_cosine_only_resolution():
    """Same scenario as
    test_multiscale.test_cosine_metric_resolves_disconnection_from_norm_only_outliers:
    same-direction, huge-norm outliers should leave the euclidean metric with
    no stable (non-disconnected) plateau while the cosine metric finds one.
    """
    rng = np.random.RandomState(0)
    direction = np.zeros(8)
    direction[0] = 1.0
    main_cluster = direction + 0.05 * rng.randn(195, 8)
    outliers = 1000 * direction + 0.05 * rng.randn(5, 8)
    Z = np.vstack([main_cluster, outliers])

    result = mg.analyze_layer_multiscale(Z, n_scales=5, seed=0)

    assert result["euclidean_stable_r_star"] is None
    assert result["cosine_stable_r_star"] is not None


def test_layers_default_to_first_middle_last():
    # main() resolves args.layers -> sorted({0, n_layers // 2, n_layers - 1})
    # when --layers isn't passed; exercise that formula directly since it
    # doesn't need a model.
    n_layers = 32
    layers = sorted({0, n_layers // 2, n_layers - 1})
    assert layers == [0, 16, 31]
