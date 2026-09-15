"""Tests for scripts/ffn_specialization.py's model-independent logic and its
combined activation-collection loop (see test_extract_geometry.py for why a
tiny locally-constructed MistralForCausalLM + fake data stream stand in for
the real model/dataset here).
"""

import importlib.util
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import MistralConfig, MistralForCausalLM

from diffusion_moe.data.dataset import collate_fn

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "ffn_specialization_script", REPO_ROOT / "scripts" / "ffn_specialization.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["ffn_specialization_script"] = module
    spec.loader.exec_module(module)
    return module


fs = _load_module()

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
    def __init__(self, n_examples=40, seq_len=SEQ_LEN, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.examples = [
            torch.randint(0, VOCAB_SIZE, (seq_len,), generator=g).tolist()
            for _ in range(n_examples)
        ]

    def __iter__(self):
        for ids in self.examples:
            yield {"input_ids": ids}


def _fake_loader(batch_size=4, n_examples=40, seq_len=SEQ_LEN, pad_token_id=0):
    dataset = _FakeTokenizedDataset(n_examples=n_examples, seq_len=seq_len)
    collate = partial(collate_fn, pad_token_id=pad_token_id)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate)


def test_collect_z_and_mlp_activations_shapes_and_alignment():
    model = _tiny_model()
    loader = _fake_loader(batch_size=4, n_examples=40)

    z_arrays, mlp_arrays = fs.collect_z_and_mlp_activations(
        model, loader, device="cpu", tokens_per_batch=6, seed=0
    )

    assert len(z_arrays) == N_LAYERS
    assert len(mlp_arrays) == N_LAYERS
    for z, mlp in zip(z_arrays, mlp_arrays):
        assert z.shape[1] == D_MODEL
        assert mlp.shape[1] == D_MODEL * 2  # intermediate_size
        # same pooled-token count in both, since they're sampled together
        assert z.shape[0] == mlp.shape[0]
        assert np.isfinite(z).all()
        assert np.isfinite(mlp).all()


def test_analyze_layer_specialization_returns_expected_structure():
    rng = np.random.RandomState(0)
    post_z = rng.randn(150, 8)
    mlp_activations = rng.rand(150, 20)

    result = fs.analyze_layer_specialization(
        post_z, mlp_activations, n_experts_values=[2, 4], seed=0
    )

    assert result["d_ff"] == 20
    assert set(result["by_n_experts"].keys()) == {"2", "4"}
    for k, stats in result["by_n_experts"].items():
        assert set(stats.keys()) == {"diffusion", "oracle", "agreement"}
        for role in ("diffusion", "oracle"):
            assert stats[role]["width_k"] == 20 // int(k)
            assert stats[role]["n_clusters"] == int(k)
            assert "mean_specialization_gain" in stats[role]
            assert "mean_pairwise_jaccard" in stats[role]
        assert "adjusted_rand_index" in stats["agreement"]
        assert "normalized_mutual_info" in stats["agreement"]


def test_analyze_layer_specialization_clustering_is_reasonably_balanced():
    """Regression test for a clustering degeneracy found on the real model:
    plain k-means on RAW post_z let a handful of extreme-norm outlier tokens
    get isolated as their own landmarks, collapsing every other token into
    one dominant cluster at every K tested — not a meaningful semantic
    partition. analyze_layer_specialization clusters on cosine-normalized
    coordinates specifically to avoid this; verify it actually does, even
    when post_z contains the same kind of extreme-norm outliers that broke
    the raw-Euclidean version on real data.
    """
    rng = np.random.RandomState(0)
    main_cluster = rng.randn(190, 8)
    outliers = rng.randn(10, 8) * 0.1 + np.array([1e4, 0, 0, 0, 0, 0, 0, 0])
    post_z = np.vstack([main_cluster, outliers])
    mlp_activations = rng.rand(200, 20)

    result = fs.analyze_layer_specialization(
        post_z, mlp_activations, n_experts_values=[4], seed=0
    )

    per_cluster = result["by_n_experts"]["4"]["diffusion"]["per_cluster"]
    cluster_sizes = [c["n_tokens"] for c in per_cluster.values()]
    largest_cluster_fraction = max(cluster_sizes) / sum(cluster_sizes)
    # the raw-Euclidean failure mode saw >99% of tokens collapse into one
    # cluster; a balanced 4-way split should keep every cluster well under
    # that
    assert largest_cluster_fraction < 0.9


def test_analyze_layer_specialization_oracle_ceiling_at_least_matches_diffusion():
    """The oracle clusters directly on the activations being measured, so it
    should never do WORSE than diffusion clustering at finding specialization
    in that same activation space — it's the ceiling by construction.
    """
    rng = np.random.RandomState(0)
    d_ff, n_clusters, width_k, n_per_cluster = 20, 4, 5, 40
    true_labels = np.repeat(np.arange(n_clusters), n_per_cluster)
    mlp_activations = rng.rand(n_clusters * n_per_cluster, d_ff) * 0.05
    for c in range(n_clusters):
        block = slice(c * width_k, (c + 1) * width_k)
        idx = true_labels == c
        mlp_activations[idx, block] = 4.0 + rng.rand(n_per_cluster, width_k) * 0.5
    # post_z carries no information about the true cluster structure
    post_z = rng.randn(n_clusters * n_per_cluster, 8)

    result = fs.analyze_layer_specialization(
        post_z, mlp_activations, n_experts_values=[4], seed=0
    )
    stats = result["by_n_experts"]["4"]

    oracle_gain = stats["oracle"]["mean_specialization_gain"]
    diffusion_gain = stats["diffusion"]["mean_specialization_gain"]
    assert oracle_gain >= diffusion_gain
    assert oracle_gain > 0.5
