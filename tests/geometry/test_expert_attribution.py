"""Tests for expert_attribution.py: per-layer/per-expert/per-token routing
analytics built on top of DiffusionMoETransformer's own router_outputs."""

import torch

from diffusion_moe.geometry.expert_attribution import (
    per_expert_top_tokens,
    per_layer_expert_load,
    plot_expert_load_heatmap,
    sequence_expert_trace,
)

N_EXPERTS = 4


def _router_outputs(expert_indices: torch.Tensor, gate_values: torch.Tensor | None = None):
    """expert_indices: (batch, seq, top_k) long. Builds a minimal aux dict
    with just what these functions read (router_logits only for its shape,
    used to infer n_experts)."""
    batch, seq, top_k = expert_indices.shape
    if gate_values is None:
        gate_values = torch.ones(batch, seq, top_k) / top_k
    router_logits = torch.zeros(batch, seq, N_EXPERTS)
    return {
        0: {
            "expert_indices": expert_indices,
            "gate_values": gate_values,
            "router_logits": router_logits,
        }
    }


def test_per_layer_expert_load_counts_top1_selections():
    # batch=1, seq=4, top_k=1: experts [0, 0, 2, 3]
    expert_indices = torch.tensor([[[0], [0], [2], [3]]])
    attention_mask = torch.ones(1, 4)
    router_outputs = _router_outputs(expert_indices)

    load = per_layer_expert_load(router_outputs, attention_mask)

    assert load[0] == [2, 0, 1, 1]  # expert 0 x2, expert 1 x0, expert 2 x1, expert 3 x1


def test_per_layer_expert_load_excludes_padded_positions():
    expert_indices = torch.tensor([[[0], [0], [1], [1]]])
    attention_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])  # last two are padding
    router_outputs = _router_outputs(expert_indices)

    load = per_layer_expert_load(router_outputs, attention_mask)

    assert load[0] == [2, 0, 0, 0]  # only the two valid (expert-0) positions counted


def test_per_layer_expert_load_top_k_slot_selects_second_choice():
    # top_k=2: slot 0 is always expert 0, slot 1 varies
    expert_indices = torch.tensor([[[0, 1], [0, 2], [0, 3]]])
    attention_mask = torch.ones(1, 3)
    router_outputs = _router_outputs(expert_indices)

    load_slot0 = per_layer_expert_load(router_outputs, attention_mask, top_k_slot=0)
    load_slot1 = per_layer_expert_load(router_outputs, attention_mask, top_k_slot=1)

    assert load_slot0[0] == [3, 0, 0, 0]
    assert load_slot1[0] == [0, 1, 1, 1]


def test_per_expert_top_tokens_groups_by_expert_and_counts():
    # token ids [10, 10, 10, 20, 20, 30] -> experts [0, 0, 0, 1, 1, 0]
    input_ids = torch.tensor([[10, 10, 10, 20, 20, 30]])
    expert_indices = torch.tensor([[[0], [0], [0], [1], [1], [0]]])
    attention_mask = torch.ones(1, 6)
    router_outputs = _router_outputs(expert_indices)

    result = per_expert_top_tokens(router_outputs, input_ids, attention_mask, min_count=1)

    expert_0_tokens = {d["token_id"]: d["count"] for d in result[0][0]}
    assert expert_0_tokens == {10: 3, 30: 1}
    expert_1_tokens = {d["token_id"]: d["count"] for d in result[0][1]}
    assert expert_1_tokens == {20: 2}
    assert result[0][2] == []
    assert result[0][3] == []


def test_per_expert_top_tokens_respects_min_count():
    input_ids = torch.tensor([[10, 10, 20]])
    expert_indices = torch.tensor([[[0], [0], [0]]])
    attention_mask = torch.ones(1, 3)
    router_outputs = _router_outputs(expert_indices)

    result = per_expert_top_tokens(router_outputs, input_ids, attention_mask, min_count=2)

    token_ids = {d["token_id"] for d in result[0][0]}
    assert token_ids == {10}  # 20 occurred once, below min_count=2


def test_per_expert_top_tokens_respects_attention_mask():
    input_ids = torch.tensor([[10, 20]])
    expert_indices = torch.tensor([[[0], [0]]])
    attention_mask = torch.tensor([[1.0, 0.0]])  # second position is padding
    router_outputs = _router_outputs(expert_indices)

    result = per_expert_top_tokens(router_outputs, input_ids, attention_mask, min_count=1)

    token_ids = {d["token_id"] for d in result[0][0]}
    assert token_ids == {10}


def test_per_expert_top_tokens_sorted_by_count_descending():
    input_ids = torch.tensor([[1, 2, 2, 3, 3, 3]])
    expert_indices = torch.tensor([[[0], [0], [0], [0], [0], [0]]])
    attention_mask = torch.ones(1, 6)
    router_outputs = _router_outputs(expert_indices)

    result = per_expert_top_tokens(router_outputs, input_ids, attention_mask, min_count=1)

    counts_in_order = [d["count"] for d in result[0][0]]
    assert counts_in_order == sorted(counts_in_order, reverse=True)


def test_sequence_expert_trace_returns_per_position_detail():
    input_ids = torch.tensor([[100, 200, 300]])
    expert_indices = torch.tensor([[[1], [2], [1]]])
    gate_values = torch.tensor([[[0.9], [0.7], [0.6]]])
    router_outputs = _router_outputs(expert_indices, gate_values)

    trace = sequence_expert_trace(router_outputs, input_ids, layer_idx=0)

    expected = [
        {"position": 0, "token_id": 100, "expert_id": 1, "gate_weight": 0.9},
        {"position": 1, "token_id": 200, "expert_id": 2, "gate_weight": 0.7},
        {"position": 2, "token_id": 300, "expert_id": 1, "gate_weight": 0.6},
    ]
    assert len(trace) == len(expected)
    for actual, exp in zip(trace, expected):
        assert actual["position"] == exp["position"]
        assert actual["token_id"] == exp["token_id"]
        assert actual["expert_id"] == exp["expert_id"]
        assert abs(actual["gate_weight"] - exp["gate_weight"]) < 1e-5


def test_sequence_expert_trace_selects_the_right_batch_element():
    input_ids = torch.tensor([[1, 1], [2, 2]])
    expert_indices = torch.tensor([[[0], [0]], [[3], [3]]])
    router_outputs = _router_outputs(expert_indices)

    trace = sequence_expert_trace(router_outputs, input_ids, layer_idx=0, seq_idx=1)

    assert all(d["token_id"] == 2 and d["expert_id"] == 3 for d in trace)


def test_plot_expert_load_heatmap_writes_png(tmp_path):
    per_layer_load = {0: [10, 0, 0, 0], 1: [3, 3, 2, 2]}
    output_path = tmp_path / "heatmap.png"

    plot_expert_load_heatmap(per_layer_load, str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0
