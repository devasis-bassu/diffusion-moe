"""Tests for scripts/expert_attribution.py's own pooling logic (merging
per-layer expert load and per-expert token counts across multiple collected
batches) -- the underlying per-batch analysis functions are already covered
by tests/geometry/test_expert_attribution.py."""

import importlib.util
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "expert_attribution_script", REPO_ROOT / "scripts" / "expert_attribution.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["expert_attribution_script"] = module
    spec.loader.exec_module(module)
    return module


ea = _load_module()

N_EXPERTS = 3


def _router_outputs(expert_indices: torch.Tensor):
    router_logits = torch.zeros(*expert_indices.shape[:-1], N_EXPERTS)
    gate_values = torch.ones_like(expert_indices, dtype=torch.float) / expert_indices.shape[-1]
    return {0: {"expert_indices": expert_indices, "gate_values": gate_values,
                "router_logits": router_logits}}


def test_pool_per_layer_expert_load_sums_across_batches():
    batch_1 = _router_outputs(torch.tensor([[[0], [0], [1]]]))
    batch_2 = _router_outputs(torch.tensor([[[0], [2], [2]]]))
    masks = [torch.ones(1, 3), torch.ones(1, 3)]

    pooled = ea.pool_per_layer_expert_load(masks, [batch_1, batch_2])

    assert pooled[0] == [3, 1, 2]  # expert 0: 2+1=3, expert 1: 1+0=1, expert 2: 0+2=2


def test_pool_per_layer_expert_load_respects_masks_per_batch():
    batch_1 = _router_outputs(torch.tensor([[[0], [0]]]))
    masks = [torch.tensor([[1.0, 0.0]])]  # second position padded

    pooled = ea.pool_per_layer_expert_load(masks, [batch_1])

    assert pooled[0] == [1, 0, 0]


def test_pool_per_expert_top_tokens_merges_counts_across_batches():
    input_ids_1 = torch.tensor([[10, 10]])
    input_ids_2 = torch.tensor([[10, 20]])
    batch_1 = _router_outputs(torch.tensor([[[0], [0]]]))
    batch_2 = _router_outputs(torch.tensor([[[0], [1]]]))
    masks = [torch.ones(1, 2), torch.ones(1, 2)]

    class _FakeTokenizer:
        def decode(self, ids):
            return f"tok{ids[0]}"

    pooled = ea.pool_per_expert_top_tokens(
        [input_ids_1, input_ids_2], masks, [batch_1, batch_2], _FakeTokenizer(),
        top_n=10, min_count=1,
    )

    expert_0_counts = {d["token_id"]: d["count"] for d in pooled[0][0]}
    assert expert_0_counts == {10: 3}  # token 10 routed to expert 0 in all 3 occurrences
    expert_1_counts = {d["token_id"]: d["count"] for d in pooled[0][1]}
    assert expert_1_counts == {20: 1}


def test_pool_per_expert_top_tokens_decodes_text():
    input_ids = torch.tensor([[42]])
    batch = _router_outputs(torch.tensor([[[0]]]))
    masks = [torch.ones(1, 1)]

    class _FakeTokenizer:
        def decode(self, ids):
            return f"<{ids[0]}>"

    pooled = ea.pool_per_expert_top_tokens(
        [input_ids], masks, [batch], _FakeTokenizer(), top_n=10, min_count=1
    )

    assert pooled[0][0][0]["text"] == "<42>"


def test_pool_per_expert_top_tokens_applies_min_count_after_merging():
    """min_count should apply to the MERGED count across all batches, not
    per-batch -- a token appearing once in each of two batches (merged
    count 2) should survive min_count=2, even though neither batch alone
    would have met it."""
    input_ids_1 = torch.tensor([[10]])
    input_ids_2 = torch.tensor([[10]])
    batch_1 = _router_outputs(torch.tensor([[[0]]]))
    batch_2 = _router_outputs(torch.tensor([[[0]]]))
    masks = [torch.ones(1, 1), torch.ones(1, 1)]

    class _FakeTokenizer:
        def decode(self, ids):
            return str(ids[0])

    pooled = ea.pool_per_expert_top_tokens(
        [input_ids_1, input_ids_2], masks, [batch_1, batch_2], _FakeTokenizer(),
        top_n=10, min_count=2,
    )

    assert len(pooled[0][0]) == 1
    assert pooled[0][0][0]["count"] == 2
