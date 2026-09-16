"""Per-token/per-layer expert-attribution analytics: which tokens actually
get routed to which expert, in which layer -- for inspecting whether
diffusion-MoE routing tracks anything interpretable, and for diagnosing
collapse concretely (which specific tokens dominate a collapsed expert,
whether collapse is uniform across layers or concentrated) rather than
reading only the aggregate load_loss/layer_{idx} scalars.

No new hooks needed: DiffusionMoETransformer.forward already returns
router_outputs = {layer_idx: aux} with aux["expert_indices"]/["gate_values"]
per MoE layer -- everything here is analysis on top of that, not capture.

Complements, not duplicates, evaluation/routing_analysis.py::RoutingAnalyser:
that one hooks every DiffusionMoELayer to ACCUMULATE aggregate stats
(expert_token_counts, routing_entropy_per_expert, centroid drift) across
MANY forward calls over a training run or eval pass, but never sees the
original input_ids (the hook fires inside the layer, with no reference back
to token identity) -- it can tell you an expert is starved, not what kind of
token it's starved of. This module runs on ONE forward call's real
router_outputs + input_ids together, trading multi-call accumulation for
the token-identity and per-sequence detail RoutingAnalyser structurally
can't provide.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import torch


def per_layer_expert_load(
    router_outputs: dict[int, dict[str, torch.Tensor]],
    attention_mask: torch.Tensor,
    top_k_slot: int = 0,
) -> dict[int, list[int]]:
    """For every layer in router_outputs, counts how many valid (non-padded)
    tokens had each expert in their `top_k_slot` selection (0 = the primary,
    highest-gate-weight expert per token; 1 = the second choice, etc.).

    Returns {layer_idx: [count_expert_0, count_expert_1, ...]} -- the
    cheapest possible whole-network view of collapse: scan this across every
    layer at once to see whether collapse is concentrated in a few layers or
    spread across all of them (see phase1_findings_report.md's "what happens
    if a layer keeps forcing a single expert" discussion). Use
    plot_expert_load_heatmap to visualize this directly.
    """
    flat_mask = attention_mask.reshape(-1).bool().cpu()
    result: dict[int, list[int]] = {}
    for layer_idx, aux in router_outputs.items():
        n_experts = aux["router_logits"].shape[-1]
        primary_expert = aux["expert_indices"][..., top_k_slot].reshape(-1).cpu()
        valid_experts = primary_expert[flat_mask]
        counts = torch.bincount(valid_experts, minlength=n_experts)
        result[layer_idx] = counts.tolist()
    return result


def per_expert_top_tokens(
    router_outputs: dict[int, dict[str, torch.Tensor]],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    top_k_slot: int = 0,
    top_n: int = 15,
    min_count: int = 3,
) -> dict[int, dict[int, list[dict[str, Any]]]]:
    """For every layer and every expert, the most frequent token ids routed
    to that expert (via `top_k_slot`'s selection) -- same "what does this
    bucket actually contain" question sink_diagnostics.py's
    per_token_id_norm_summary answers for norm outliers, applied to routing
    instead. Token ids occurring fewer than `min_count` times for a given
    expert are dropped (matches per_token_id_norm_summary's convention: a
    single occurrence isn't a reliable per-token-id statistic).

    Returns {layer_idx: {expert_id: [{"token_id", "count"}, ...]}}, each
    expert's list sorted by count descending. Decoding token ids to text is
    left to the caller (this module has no tokenizer dependency), matching
    sink_diagnostics.py's own convention.
    """
    flat_mask = attention_mask.reshape(-1).bool().cpu()
    flat_tokens = input_ids.reshape(-1).cpu()

    result: dict[int, dict[int, list[dict[str, Any]]]] = {}
    for layer_idx, aux in router_outputs.items():
        n_experts = aux["router_logits"].shape[-1]
        primary_expert = aux["expert_indices"][..., top_k_slot].reshape(-1).cpu()

        valid_tokens = flat_tokens[flat_mask]
        valid_experts = primary_expert[flat_mask]

        per_expert: dict[int, list[dict[str, Any]]] = {}
        for expert_id in range(n_experts):
            tokens_for_expert = valid_tokens[valid_experts == expert_id]
            counts = Counter(tokens_for_expert.tolist())
            per_expert[expert_id] = [
                {"token_id": int(token_id), "count": int(count)}
                for token_id, count in counts.most_common(top_n)
                if count >= min_count
            ]
        result[layer_idx] = per_expert
    return result


def sequence_expert_trace(
    router_outputs: dict[int, dict[str, torch.Tensor]],
    input_ids: torch.Tensor,
    layer_idx: int,
    seq_idx: int = 0,
    top_k_slot: int = 0,
) -> list[dict[str, Any]]:
    """For ONE sequence (seq_idx into the batch) and ONE layer, the raw
    per-position material for a "color each token by its expert"
    visualization: which expert each token was routed to (top_k_slot's
    selection) and the gate weight it got there, in sequence order.

    Returns a list (one entry per position) of
    {"position", "token_id", "expert_id", "gate_weight"}. Decoding token ids
    to text and the actual rendering are left to the caller.
    """
    aux = router_outputs[layer_idx]
    expert_ids = aux["expert_indices"][seq_idx, :, top_k_slot].cpu()
    gate_weights = aux["gate_values"][seq_idx, :, top_k_slot].cpu()
    tokens = input_ids[seq_idx].cpu()

    return [
        {
            "position": pos,
            "token_id": int(tokens[pos]),
            "expert_id": int(expert_ids[pos]),
            "gate_weight": float(gate_weights[pos]),
        }
        for pos in range(tokens.shape[0])
    ]


def plot_expert_load_heatmap(
    per_layer_load: dict[int, list[int]], output_path: str, title: str = "Expert load by layer"
) -> None:
    """Layers x experts heatmap of each expert's SHARE of tokens (not raw
    count, so layers are comparable regardless of how many tokens they
    pooled) -- the single-glance view of which layers collapse and onto
    which expert(s), across an entire network at once. A collapsed layer
    shows one bright cell and the rest dark; a balanced layer shows uniform
    shading across its row.
    """
    import matplotlib.pyplot as plt

    layers = sorted(per_layer_load.keys())
    n_experts = len(next(iter(per_layer_load.values())))
    shares = np.zeros((len(layers), n_experts))
    for row, layer_idx in enumerate(layers):
        counts = np.array(per_layer_load[layer_idx], dtype=np.float64)
        total = counts.sum()
        shares[row] = counts / total if total > 0 else 0.0

    fig, ax = plt.subplots(figsize=(max(6, n_experts * 0.6), max(4, len(layers) * 0.35)))
    im = ax.imshow(shares, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n_experts))
    ax.set_xticklabels([f"E{e}" for e in range(n_experts)])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{layer_idx}" for layer_idx in layers])
    ax.set_xlabel("expert")
    ax.set_ylabel("layer")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="share of tokens (top-1 expert)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
