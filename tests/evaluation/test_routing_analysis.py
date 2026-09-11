"""Tests for RoutingAnalyser."""

import math

import pytest
import torch

from diffusion_moe.evaluation.routing_analysis import RoutingAnalyser
from diffusion_moe.models.moe_model import DiffusionMoETransformer

VOCAB_SIZE, SEQ_LEN, D_MODEL, N_LAYERS = 30, 8, 16, 2
N_EXPERTS, TOP_K = 4, 2


def _model(layers_to_replace=None):
    return DiffusionMoETransformer(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=4, max_seq_len=SEQ_LEN,
        n_experts=N_EXPERTS, top_k=TOP_K, n_components=3, n_landmarks=8, diffusion_t=2,
        centroid_refresh_steps=1000, layers_to_replace=layers_to_replace,
    )


def test_raises_without_any_moe_layer():
    model = _model(layers_to_replace=[])
    with pytest.raises(ValueError):
        RoutingAnalyser(model)


def test_hooks_all_moe_layers():
    model = _model()  # default: every layer is MoE
    analyser = RoutingAnalyser(model)
    assert set(analyser.moe_layers.keys()) == set(range(N_LAYERS))


def test_expert_token_counts_totals_match_slot_assignments():
    model = _model()
    analyser = RoutingAnalyser(model)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))

    with analyser:
        model(input_ids)

    counts = analyser.expert_token_counts()
    total = sum(counts.values())
    # 2 layers * (batch=2 * seq=8 * top_k=2) assignments each
    assert total == N_LAYERS * (2 * SEQ_LEN * TOP_K)
    assert set(counts.keys()) == set(range(N_EXPERTS))


def test_expert_token_counts_can_be_restricted_to_one_layer():
    model = _model()
    analyser = RoutingAnalyser(model)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))

    with analyser:
        model(input_ids)

    counts_layer0 = analyser.expert_token_counts(layer_idx=0)
    assert sum(counts_layer0.values()) == 2 * SEQ_LEN * TOP_K


def test_routing_entropy_per_expert_sums_to_aggregate_entropy():
    model = _model()
    analyser = RoutingAnalyser(model)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))

    with analyser:
        model(input_ids)

    per_expert = analyser.routing_entropy_per_expert()
    counts = analyser.expert_token_counts()
    total = sum(counts.values())
    manual_aggregate = -sum(
        (c / total) * math.log2(c / total) for c in counts.values() if c > 0
    )
    assert math.isclose(sum(per_expert.values()), manual_aggregate, rel_tol=1e-6)


def test_routing_entropy_zero_for_starved_expert():
    """An expert with zero assignments contributes exactly 0 to the entropy."""
    model = _model()
    analyser = RoutingAnalyser(model)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))

    with analyser:
        model(input_ids)

    counts = analyser.expert_token_counts()
    per_expert = analyser.routing_entropy_per_expert()
    for expert_id, count in counts.items():
        if count == 0:
            assert per_expert[expert_id] == 0.0


def test_centroid_distances_over_training_one_value_per_forward_call():
    model = _model()
    analyser = RoutingAnalyser(model)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))

    with analyser:
        model(input_ids)
        model(input_ids)
        model(input_ids)

    distances = analyser.centroid_distances_over_training(layer_idx=0)
    assert len(distances) == 3
    assert all(d >= 0 for d in distances)


def test_hooks_removed_after_context_exit():
    model = _model()
    analyser = RoutingAnalyser(model)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))

    with analyser:
        model(input_ids)
    n_records_after_exit = sum(len(r) for r in analyser.records.values())

    model(input_ids)  # outside the context — should NOT be recorded
    n_records_now = sum(len(r) for r in analyser.records.values())

    assert n_records_now == n_records_after_exit


def test_reentering_context_resets_records():
    model = _model()
    analyser = RoutingAnalyser(model)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))

    with analyser:
        model(input_ids)
    with analyser:
        pass  # no forward calls this time

    assert sum(len(r) for r in analyser.records.values()) == 0
