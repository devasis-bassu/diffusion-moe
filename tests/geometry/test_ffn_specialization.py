"""Tests for the FFN-width-reduction diagnostic (geometry/ffn_specialization.py).

Validates the two extremes the analysis is meant to distinguish: neurons
that are genuinely cluster-specific (own top-k neurons cover the cluster's
activation mass far better than a shared global top-k does, and clusters'
top-k sets barely overlap) vs. neurons used identically regardless of
cluster (own coverage matches global coverage, top-k sets fully overlap) —
the latter would falsify the width-reduction hypothesis for that data
independent of whatever the diffusion map's r* says about routing.
"""

import numpy as np
import torch
from transformers import MistralConfig, MistralForCausalLM

from diffusion_moe.geometry.ffn_specialization import (
    MLPInputCapture,
    cluster_agreement,
    cluster_tokens,
    neuron_specialization_analysis,
    neuron_usage,
    oracle_cluster_tokens_by_activation,
)

SEED = 0


def test_neuron_usage_is_mean_abs_activation():
    activations = np.array([[1.0, -2.0, 0.0], [3.0, 2.0, 0.0], [-4.0, 0.0, 0.0]])
    usage = neuron_usage(activations)
    np.testing.assert_allclose(usage, [8.0 / 3, 4.0 / 3, 0.0])


def test_cluster_tokens_returns_requested_number_of_clusters():
    rng = np.random.RandomState(SEED)
    Psi = rng.randn(100, 5)
    labels = cluster_tokens(Psi, n_clusters=4, random_state=SEED)
    assert labels.shape == (100,)
    assert set(np.unique(labels).tolist()) == {0, 1, 2, 3}


def test_cluster_tokens_deterministic_given_same_seed():
    rng = np.random.RandomState(SEED)
    Psi = rng.randn(100, 5)
    labels_a = cluster_tokens(Psi, n_clusters=4, random_state=SEED)
    labels_b = cluster_tokens(Psi, n_clusters=4, random_state=SEED)
    np.testing.assert_array_equal(labels_a, labels_b)


def test_specialization_analysis_detects_disjoint_specialized_clusters():
    """Each cluster's tokens fire almost exclusively on their own dedicated
    block of neurons -> own top-k should capture ~all the cluster's mass,
    a shared global top-k should capture much less, and different clusters'
    top-k sets shouldn't overlap at all.
    """
    rng = np.random.RandomState(SEED)
    d_ff, n_clusters, width_k, n_per_cluster = 20, 4, 5, 60

    labels = np.repeat(np.arange(n_clusters), n_per_cluster)
    activations = rng.rand(n_clusters * n_per_cluster, d_ff) * 0.05
    for c in range(n_clusters):
        block = slice(c * width_k, (c + 1) * width_k)
        idx = labels == c
        activations[idx, block] = 4.0 + rng.rand(n_per_cluster, width_k) * 0.5

    result = neuron_specialization_analysis(activations, labels, width_k)

    assert result["mean_specialization_gain"] > 0.5
    assert result["mean_pairwise_jaccard"] == 0.0
    for stats in result["per_cluster"].values():
        assert stats["own_coverage"] > 0.9
        assert stats["specialization_gain"] > 0.5


def test_specialization_analysis_finds_no_gain_when_neurons_used_uniformly():
    """Every cluster draws from the identical neuron-usage distribution ->
    own-cluster top-k should match the globally-shared top-k almost exactly:
    zero specialization gain, full overlap. This is the case that would
    falsify the width-reduction hypothesis regardless of r*.
    """
    rng = np.random.RandomState(SEED)
    d_ff, n_clusters, width_k, n_per_cluster = 20, 4, 5, 200

    means = np.linspace(0.1, 5.0, d_ff)
    labels = np.repeat(np.arange(n_clusters), n_per_cluster)
    activations = rng.rand(n_clusters * n_per_cluster, d_ff) * means[None, :] * 2

    result = neuron_specialization_analysis(activations, labels, width_k)

    assert abs(result["mean_specialization_gain"]) < 0.05
    assert result["mean_pairwise_jaccard"] > 0.9


def test_specialization_analysis_width_k_clipped_to_d_ff():
    rng = np.random.RandomState(SEED)
    activations = rng.rand(40, 6)
    labels = np.repeat([0, 1], 20)
    result = neuron_specialization_analysis(activations, labels, width_k=1000)
    assert result["width_k"] == 6


def _tiny_model(n_layers=3, hidden_size=16):
    config = MistralConfig(
        vocab_size=50,
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


def test_mlp_input_capture_shapes_match_intermediate_size():
    model = _tiny_model(n_layers=3, hidden_size=16)
    batch, seq_len = 2, 8
    input_ids = torch.randint(0, 50, (batch, seq_len))

    with MLPInputCapture(model) as capture:
        with torch.no_grad():
            model(input_ids)

    assert len(capture.outputs) == 3  # one per decoder layer
    for layer_output in capture.outputs:
        assert layer_output.shape == (batch, seq_len, 32)  # intermediate_size = 16*2


def test_mlp_input_capture_removes_hooks_on_exit():
    model = _tiny_model(n_layers=2, hidden_size=16)
    with MLPInputCapture(model) as capture:
        pass
    assert capture._handles == []

    input_ids = torch.randint(0, 50, (1, 4))
    with torch.no_grad():
        model(input_ids)
    assert capture.outputs == []  # hooks were removed, so nothing new appended


def test_mlp_input_capture_rejects_non_llama_family_model():
    class NotALlamaModel(torch.nn.Module):
        pass

    import pytest

    with pytest.raises(ValueError):
        MLPInputCapture(NotALlamaModel())


def _disjoint_specialized_activations(
    seed=SEED, d_ff=20, n_clusters=4, width_k=5, n_per_cluster=60
):
    rng = np.random.RandomState(seed)
    true_labels = np.repeat(np.arange(n_clusters), n_per_cluster)
    activations = rng.rand(n_clusters * n_per_cluster, d_ff) * 0.05
    for c in range(n_clusters):
        block = slice(c * width_k, (c + 1) * width_k)
        idx = true_labels == c
        activations[idx, block] = 4.0 + rng.rand(n_per_cluster, width_k) * 0.5
    return activations, true_labels


def test_oracle_cluster_tokens_recovers_disjoint_specialized_clusters():
    """Oracle clustering (on the activations themselves) should recover the
    same disjoint-block structure used in
    test_specialization_analysis_detects_disjoint_specialized_clusters, and
    measuring specialization against those oracle labels shows near-total
    specialization — the ceiling this framework is meant to establish.
    """
    activations, true_labels = _disjoint_specialized_activations()

    oracle_labels = oracle_cluster_tokens_by_activation(
        activations, n_clusters=4, n_components=8, random_state=SEED
    )
    result = neuron_specialization_analysis(activations, oracle_labels, width_k=5)

    assert result["mean_specialization_gain"] > 0.5
    assert result["mean_pairwise_jaccard"] < 0.1
    # oracle clustering should recover the true block structure (allowing
    # for arbitrary label numbering — ARI/NMI are permutation-invariant)
    agreement = cluster_agreement(true_labels, oracle_labels)
    assert agreement["adjusted_rand_index"] > 0.9


def test_cluster_agreement_identical_labels():
    labels = np.array([0, 0, 1, 1, 2, 2])
    agreement = cluster_agreement(labels, labels)
    assert agreement["adjusted_rand_index"] == 1.0
    assert agreement["normalized_mutual_info"] == 1.0


def test_cluster_agreement_permutation_invariant():
    labels_a = np.array([0, 0, 1, 1, 2, 2])
    labels_b = np.array([5, 5, 9, 9, 1, 1])  # same partition, different label ids
    agreement = cluster_agreement(labels_a, labels_b)
    assert agreement["adjusted_rand_index"] == 1.0
    assert agreement["normalized_mutual_info"] == 1.0


def test_cluster_agreement_independent_labels_near_chance():
    rng = np.random.RandomState(SEED)
    labels_a = rng.randint(0, 4, size=2000)
    labels_b = rng.randint(0, 4, size=2000)  # independent of labels_a
    agreement = cluster_agreement(labels_a, labels_b)
    assert abs(agreement["adjusted_rand_index"]) < 0.05
    assert agreement["normalized_mutual_info"] < 0.05


def test_oracle_specialization_stays_high_when_diffusion_signal_is_uninformative():
    """The core value of this decomposition: oracle specialization stays
    high regardless of whether the ROUTING signal (e.g. diffusion
    coordinates) is any good, while agreement between oracle and a
    diffusion-like clustering built from pure noise stays near zero — the
    two independently-answerable pieces the module docstring describes.
    """
    activations, _ = _disjoint_specialized_activations()
    rng = np.random.RandomState(SEED)

    oracle_labels = oracle_cluster_tokens_by_activation(
        activations, n_clusters=4, n_components=8, random_state=SEED
    )
    oracle_result = neuron_specialization_analysis(activations, oracle_labels, width_k=5)

    # a "diffusion clustering" built from coordinates carrying NO information
    # about the true activation structure
    uninformative_coords = rng.randn(activations.shape[0], 8)
    diffusion_like_labels = cluster_tokens(
        uninformative_coords, n_clusters=4, random_state=SEED
    )
    diffusion_result = neuron_specialization_analysis(activations, diffusion_like_labels, width_k=5)

    assert oracle_result["mean_specialization_gain"] > 0.5
    assert diffusion_result["mean_specialization_gain"] < oracle_result["mean_specialization_gain"]

    agreement = cluster_agreement(oracle_labels, diffusion_like_labels)
    assert agreement["adjusted_rand_index"] < 0.3
