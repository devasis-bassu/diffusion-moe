"""Tests for scripts/extract_geometry.py.

Loading a real Llama-2/Mistral-7B and streaming real Wikipedia isn't possible
in this offline environment (gated, multi-GB downloads), so these tests use a
tiny locally-constructed MistralForCausalLM (same architecture/module layout,
random weights, no download) and a fake in-memory data stream in place of the
real Wikipedia stream.
"""

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import MistralConfig, MistralForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "extract_geometry_script", REPO_ROOT / "scripts" / "extract_geometry.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["extract_geometry_script"] = module
    spec.loader.exec_module(module)
    return module


eg = _load_module()

VOCAB_SIZE, D_MODEL, N_LAYERS, SEQ_LEN = 50, 16, 3, 12


def _tiny_model(n_layers=N_LAYERS, hidden_size=D_MODEL):
    config = MistralConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    model = MistralForCausalLM(config)
    model.eval()
    return model


class _FakeTokenizedDataset(IterableDataset):
    """Yields fixed-length, fully-valid (no padding) token sequences —
    stands in for a real streamed-and-tokenized Wikipedia batch."""

    def __init__(self, n_examples=20, seq_len=SEQ_LEN, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.examples = [
            torch.randint(0, VOCAB_SIZE, (seq_len,), generator=g).tolist()
            for _ in range(n_examples)
        ]

    def __iter__(self):
        for ids in self.examples:
            yield {"input_ids": ids}


def _fake_loader(batch_size=4, n_examples=20, seq_len=SEQ_LEN, pad_token_id=0):
    from functools import partial

    from diffusion_moe.data.dataset import collate_fn

    dataset = _FakeTokenizedDataset(n_examples=n_examples, seq_len=seq_len)
    collate = partial(collate_fn, pad_token_id=pad_token_id)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate)


def test_attention_output_capture_matches_layer_count_and_shape():
    model = _tiny_model()
    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))

    with eg.AttentionOutputCapture(model) as capture:
        model(input_ids)

    assert len(capture.outputs) == N_LAYERS
    for out in capture.outputs:
        assert out.shape == (2, SEQ_LEN, D_MODEL)


def test_attention_output_capture_rejects_non_llama_family_model():
    class NotALlamaModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

    import pytest

    with pytest.raises(ValueError):
        eg.AttentionOutputCapture(NotALlamaModel())


def test_subsample_valid_tokens_excludes_padding():
    z = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    gen = torch.Generator().manual_seed(0)

    sampled = eg.subsample_valid_tokens(z, mask, n_tokens=100, generator=gen)
    # only 2 + 3 = 5 valid positions total, even though 100 were requested
    assert sampled.shape == (5, 3)


def test_subsample_valid_tokens_caps_at_n_tokens():
    z = torch.randn(2, 10, 4)
    mask = torch.ones(2, 10, dtype=torch.long)
    gen = torch.Generator().manual_seed(0)
    sampled = eg.subsample_valid_tokens(z, mask, n_tokens=5, generator=gen)
    assert sampled.shape == (5, 4)


def test_compute_tokens_per_batch_keeps_total_pool_bounded_as_sequences_grow():
    """The bug that OOM-killed a real run: a FIXED per-batch quota makes total
    pooled memory scale linearly with n_sequences. The derived per-batch
    quota must shrink as n_sequences grows, keeping the total roughly at
    max_pool_size regardless."""
    small = eg.compute_tokens_per_batch(max_pool_size=8192, n_sequences=20, batch_size=8)
    large = eg.compute_tokens_per_batch(max_pool_size=8192, n_sequences=1000, batch_size=8)

    assert large < small  # more sequences -> smaller per-batch quota
    # total pooled tokens stays near the budget in both cases, not scaling
    # linearly with n_sequences the way a fixed quota would
    small_total = small * math.ceil(20 / 8)
    large_total = large * math.ceil(1000 / 8)
    assert small_total <= 8192
    assert large_total <= 8192 + 125  # off by at most one quota-per-batch of rounding


def test_compute_tokens_per_batch_realistic_mistral_7b_scale():
    """The exact scenario that caused the real OOM: n_sequences=1000,
    batch_size=8, 32 layers, d_model=4096, pre+post, float32. Total memory
    should now land in the single-digit GB range, not ~125GB."""
    tokens_per_batch = eg.compute_tokens_per_batch(
        max_pool_size=eg.MAX_POOL_SIZE, n_sequences=1000, batch_size=8
    )
    n_batches = math.ceil(1000 / 8)
    total_pooled_per_layer = tokens_per_batch * n_batches

    d_model, n_layers = 4096, 32
    total_bytes = total_pooled_per_layer * d_model * 4 * 2 * n_layers  # float32, pre+post
    total_gb = total_bytes / (1024**3)

    assert total_gb < 10  # was ~125GB before the fix


def test_compute_tokens_per_batch_never_returns_zero():
    # even an enormous n_sequences shouldn't drive the per-batch quota to 0
    # (that would silently pool nothing at all)
    assert eg.compute_tokens_per_batch(max_pool_size=100, n_sequences=1_000_000, batch_size=8) >= 1


def test_collect_layer_activations_returns_float32_with_bf16_model():
    """The real script now loads models in bf16 (see main()'s dtype choice) —
    confirm collect_layer_activations still produces plain float32 numpy
    arrays end to end, not bf16 (which has no native numpy dtype)."""
    model = _tiny_model().to(torch.bfloat16)
    loader = _fake_loader(batch_size=4, n_examples=16)

    pre_arrays, post_arrays = eg.collect_layer_activations(
        model, loader, device="cpu", tokens_per_batch=6, seed=0
    )

    for arr in pre_arrays + post_arrays:
        assert arr.dtype == np.float32
        assert np.all(np.isfinite(arr))


def test_collect_layer_activations_shapes():
    model = _tiny_model()
    loader = _fake_loader(batch_size=4, n_examples=16)

    pre_arrays, post_arrays = eg.collect_layer_activations(
        model, loader, device="cpu", tokens_per_batch=6, seed=0
    )

    assert len(pre_arrays) == N_LAYERS
    assert len(post_arrays) == N_LAYERS
    # 4 batches of size 4, 6 tokens sampled per batch -> 24 pooled tokens
    for arr in pre_arrays + post_arrays:
        assert arr.shape == (24, D_MODEL)
        # float32, not float64: NystromDiffusionMap upcasts internally itself,
        # one layer at a time — doing it here too, for all layers held in
        # memory at once, is exactly what caused the real 7B-model OOM.
        assert arr.dtype == np.float32


def test_collect_layer_activations_excludes_given_token_ids():
    """Recommendation 6 (phase1_findings_report.md §7): excluding known
    outlier tokens (e.g. the newline token responsible for most of layers
    1/2/4's disconnection) from geometry pooling entirely. Deterministic
    setup: each sequence is 11 tokens of id 1 plus one token of id 2 (the
    "excluded" one) -- with tokens_per_batch set high enough to capture
    every valid position, pooled count must drop by exactly one token per
    sequence once id 2 is excluded.
    """
    model = _tiny_model()
    n_examples, batch_size = 8, 4

    class _WithSentinelToken(IterableDataset):
        def __iter__(self):
            for _ in range(n_examples):
                yield {"input_ids": [1] * (SEQ_LEN - 1) + [2]}

    from functools import partial

    from diffusion_moe.data.dataset import collate_fn

    loader = DataLoader(
        _WithSentinelToken(),
        batch_size=batch_size,
        collate_fn=partial(collate_fn, pad_token_id=0),
    )

    pre_unfiltered, _ = eg.collect_layer_activations(
        model, loader, device="cpu", tokens_per_batch=1000, seed=0
    )
    pre_filtered, _ = eg.collect_layer_activations(
        model, loader, device="cpu", tokens_per_batch=1000, seed=0, excluded_token_ids={2}
    )

    n_batches = n_examples // batch_size
    assert pre_unfiltered[0].shape[0] == n_examples * SEQ_LEN
    # one sentinel token per sequence removed, every batch
    assert pre_filtered[0].shape[0] == n_examples * SEQ_LEN - n_examples
    assert pre_filtered[0].shape[0] == pre_unfiltered[0].shape[0] - n_examples
    assert n_batches > 0  # sanity: the loader actually yields multiple batches


def test_nystrom_approximation_error_zero_at_reference_and_decreasing():
    Z = np.random.RandomState(0).randn(300, 8)
    errors = eg.nystrom_approximation_error(Z, m_values=[16, 32, 64, 128], n_components=5)

    assert set(errors.keys()) == {16, 32, 64, 128}
    assert errors[128] == 0.0  # the reference (max m) has zero error vs itself
    assert errors[16] >= errors[32] >= errors[64]  # more landmarks -> better approx


def test_analyze_layer_returns_expected_keys_and_sane_values():
    pre_Z = np.random.RandomState(0).randn(200, 8)
    post_Z = np.random.RandomState(1).randn(200, 8)

    result = eg.analyze_layer(
        pre_Z, post_Z, n_landmarks=32, n_components=5, m_values=[8, 16, 32]
    )

    assert set(result.keys()) == {
        "r_star_post_attention",
        "r_star_pre_attention",
        "spectral_gap",
        "nystrom_error",
        "top_eigenvalues",
        "likely_disconnected",
        "post_eps",
        "pre_eps",
        "post_pool_diagnostics",
        "pre_pool_diagnostics",
    }
    assert 1 <= result["r_star_post_attention"] <= 5
    assert 1 <= result["r_star_pre_attention"] <= 5
    assert set(result["nystrom_error"].keys()) == {"8", "16", "32"}
    assert len(result["top_eigenvalues"]) == 5
    assert isinstance(result["likely_disconnected"], bool)
    assert result["post_eps"] > 0
    assert result["post_pool_diagnostics"]["n_tokens"] == 200
    assert result["post_pool_diagnostics"]["duplicate_fraction"] == 0.0


def test_analyze_layer_flags_disconnected_clusters():
    # A large main cluster plus a handful of extreme-norm outlier points
    # (mimicking real attention-sink / massive-activation tokens) so that
    # >95% of pairwise distances are small, within-cluster ones: the
    # median-heuristic bandwidth locks onto that small scale, making the
    # ~1e4-away outliers get ~zero kernel affinity to everything else. That
    # near-isolated component gives the landmark Markov chain a second
    # eigenvalue at/near 1, which should trip the disconnection flag rather
    # than be read as a tiny, genuine intrinsic dimension.
    rng = np.random.RandomState(0)
    main_cluster = rng.randn(195, 8)
    outliers = rng.randn(5, 8) * 0.1 + np.array([1e4, 0, 0, 0, 0, 0, 0, 0])
    post_Z = np.vstack([main_cluster, outliers])
    pre_Z = np.vstack([main_cluster, outliers])

    result = eg.analyze_layer(
        pre_Z, post_Z, n_landmarks=32, n_components=5, m_values=[8, 16, 32]
    )

    assert result["likely_disconnected"] is True
    assert result["top_eigenvalues"][0] > 0.29
    # The diagnostics should point at the outlier-norm hypothesis (large
    # token_norm_max_to_median), not the duplicate-landmark hypothesis
    # (near-zero eps / high duplicate_fraction) — these 200 points are all
    # distinct floats.
    assert result["post_pool_diagnostics"]["token_norm_max_to_median"] > 100
    assert result["post_pool_diagnostics"]["duplicate_fraction"] == 0.0
    assert result["post_eps"] > np.finfo(np.float64).eps * 10


def test_run_geometry_extraction_end_to_end_with_tiny_model():
    model = _tiny_model()
    loader = _fake_loader(batch_size=4, n_examples=16)

    results = eg.run_geometry_extraction(
        model, loader, device="cpu", d_model=D_MODEL, tokens_per_batch=6, seed=0
    )

    assert results["d_model"] == D_MODEL
    assert results["n_layers"] == N_LAYERS
    assert len(results["layers"]) == N_LAYERS
    for i, layer in enumerate(results["layers"]):
        assert layer["layer"] == i
        assert layer["n_tokens_pooled"] == 24


def test_save_results_writes_valid_json(tmp_path):
    results = {
        "d_model": 64,
        "n_layers": 2,
        "layers": [
            {"layer": 0, "r_star_post_attention": 3, "spectral_gap": 0.1},
            {"layer": 1, "r_star_post_attention": 4, "spectral_gap": 0.2},
        ],
    }
    path = eg.save_results(results, tmp_path, "org/model-name")
    assert path.name == "org_model-name_geometry.json"
    assert path.exists()

    with open(path) as f:
        loaded = json.load(f)
    assert loaded == results


def test_plot_results_writes_png(tmp_path):
    results = {
        "d_model": 64,
        "n_layers": 3,
        "layers": [
            {
                "layer": i,
                "r_star_post_attention": 3 + i,
                "spectral_gap": 0.1 * (i + 1),
                "nystrom_error": {"32": 0.5, "64": 0.2, "128": 0.05},
            }
            for i in range(3)
        ],
    }
    path = eg.plot_results(results, tmp_path, "org/model-name")
    assert path.name == "org_model-name_geometry.png"
    assert path.exists()
    assert path.stat().st_size > 0
