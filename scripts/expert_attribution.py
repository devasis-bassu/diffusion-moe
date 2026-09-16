"""Per-token/per-layer expert-attribution report for a trained (or
in-progress) DiffusionMoETransformer checkpoint: which tokens actually get
routed to which expert, in which layer, plus a whole-network heatmap and a
few example sequences traced token-by-token -- the tool for turning "layer N
collapsed" (a number) into "here's what it collapsed onto, and why that
might make sense" (something you can look at).

Complements scripts/evaluate.py's routing diagnostics (aggregate counts/
entropy via RoutingAnalyser, no token identity) with the token-identity and
per-sequence detail that requires seeing input_ids alongside router_outputs.

Usage:
    python scripts/expert_attribution.py --checkpoint checkpoints/step_2000.pt
    python scripts/expert_attribution.py --checkpoint checkpoints/step_2000.pt \\
        --layers 0 5 10 15 20 --n_sequences 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir

from diffusion_moe.data.dataloader import build_dataloaders
from diffusion_moe.data.tokenizer import TokenizerWrapper
from diffusion_moe.geometry.expert_attribution import (
    per_expert_top_tokens,
    per_layer_expert_load,
    plot_expert_load_heatmap,
    sequence_expert_trace,
)
from diffusion_moe.models.model_factory import build_model_from_config
from diffusion_moe.utils.device import get_device
from diffusion_moe.utils.env import load_env

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config-name", type=str, default="base_config")
    parser.add_argument("--device", type=str, default=get_device())
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="Restrict analysis to these layer indices (default: every MoE layer present).",
    )
    parser.add_argument("--n_batches", type=int, default=20, help="Batches to pool from val data.")
    parser.add_argument("--n_example_sequences", type=int, default=5)
    parser.add_argument("--top_n_tokens", type=int, default=15)
    parser.add_argument("--min_token_count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/expert_attribution")
    return parser.parse_args()


def load_config(config_name: str) -> Any:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS_DIR)):
        return compose(config_name=config_name)


def load_model_from_checkpoint(checkpoint_path: str, cfg: Any, device: str) -> torch.nn.Module:
    model = build_model_from_config(cfg)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def collect_router_outputs_over_batches(
    model: torch.nn.Module, loader: Any, device: str, n_batches: int, layers: list[int] | None
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[dict[int, dict[str, torch.Tensor]]]]:
    """Runs up to n_batches real batches through the model, returning the
    input_ids/attention_mask/router_outputs from each -- kept as a list of
    per-batch results (not concatenated) so per_expert_top_tokens etc. can
    be pooled across all of them while sequence_expert_trace can still
    index into one specific original batch/sequence for the example traces.
    """
    all_input_ids, all_masks, all_router_outputs = [], [], []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        _, _, router_outputs = model(input_ids)
        if layers is not None:
            router_outputs = {k: v for k, v in router_outputs.items() if k in layers}
        all_input_ids.append(input_ids)
        all_masks.append(attention_mask)
        all_router_outputs.append(router_outputs)
    return all_input_ids, all_masks, all_router_outputs


def pool_per_layer_expert_load(
    all_masks: list[torch.Tensor], all_router_outputs: list[dict[int, dict[str, torch.Tensor]]]
) -> dict[int, list[int]]:
    """Sums per_layer_expert_load across every collected batch."""
    pooled: dict[int, list[int]] = {}
    for mask, router_outputs in zip(all_masks, all_router_outputs):
        batch_load = per_layer_expert_load(router_outputs, mask)
        for layer_idx, counts in batch_load.items():
            if layer_idx not in pooled:
                pooled[layer_idx] = list(counts)
            else:
                pooled[layer_idx] = [a + b for a, b in zip(pooled[layer_idx], counts)]
    return pooled


def pool_per_expert_top_tokens(
    all_input_ids: list[torch.Tensor],
    all_masks: list[torch.Tensor],
    all_router_outputs: list[dict[int, dict[str, torch.Tensor]]],
    tokenizer: TokenizerWrapper,
    top_n: int,
    min_count: int,
) -> dict[int, dict[int, list[dict[str, Any]]]]:
    """Merges per-batch token counts per (layer, expert), then re-sorts and
    truncates -- pooling across batches instead of just using the last one,
    so rare-but-real tokens for a lightly-loaded expert aren't lost to
    whichever single batch happened to be looked at.
    """
    from collections import Counter

    merged: dict[int, dict[int, Counter]] = {}
    for input_ids, mask, router_outputs in zip(all_input_ids, all_masks, all_router_outputs):
        batch_result = per_expert_top_tokens(
            router_outputs, input_ids, mask, top_n=10**9, min_count=1
        )
        for layer_idx, per_expert in batch_result.items():
            merged.setdefault(layer_idx, {})
            for expert_id, entries in per_expert.items():
                counter = merged[layer_idx].setdefault(expert_id, Counter())
                for entry in entries:
                    counter[entry["token_id"]] += entry["count"]

    final: dict[int, dict[int, list[dict[str, Any]]]] = {}
    for layer_idx, per_expert in merged.items():
        final[layer_idx] = {}
        for expert_id, counter in per_expert.items():
            final[layer_idx][expert_id] = [
                {"token_id": token_id, "text": tokenizer.decode([token_id]), "count": count}
                for token_id, count in counter.most_common(top_n)
                if count >= min_count
            ]
    return final


def main() -> None:
    load_env()
    args = parse_args()
    torch.manual_seed(args.seed)
    cfg = load_config(args.config_name)

    model = load_model_from_checkpoint(args.checkpoint, cfg, args.device)
    tokenizer = TokenizerWrapper(cfg.data.tokenizer)
    _, val_loader = build_dataloaders(cfg)

    all_input_ids, all_masks, all_router_outputs = collect_router_outputs_over_batches(
        model, val_loader, args.device, args.n_batches, args.layers
    )
    if not all_router_outputs or not all_router_outputs[0]:
        raise ValueError(
            "No MoE layers found in router_outputs -- check --layers matches "
            "layers_to_replace, and that the checkpoint actually has any DiffusionMoELayer."
        )

    per_layer_load = pool_per_layer_expert_load(all_masks, all_router_outputs)
    top_tokens = pool_per_expert_top_tokens(
        all_input_ids, all_masks, all_router_outputs, tokenizer, args.top_n_tokens,
        args.min_token_count,
    )

    layers_present = sorted(all_router_outputs[0].keys())
    n_examples = min(args.n_example_sequences, all_input_ids[0].shape[0])
    example_traces = {
        layer_idx: [
            {
                "text": [tokenizer.decode([d["token_id"]]) for d in trace],
                "expert_id": [d["expert_id"] for d in trace],
                "gate_weight": [d["gate_weight"] for d in trace],
            }
            for seq_idx in range(n_examples)
            for trace in [
                sequence_expert_trace(all_router_outputs[0], all_input_ids[0], layer_idx, seq_idx)
            ]
        ]
        for layer_idx in layers_present
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = Path(args.checkpoint).stem

    results = {
        "checkpoint": args.checkpoint,
        "per_layer_expert_load": per_layer_load,
        "per_expert_top_tokens": {
            str(layer_idx): {str(e): v for e, v in per_expert.items()}
            for layer_idx, per_expert in top_tokens.items()
        },
        "example_sequence_traces": {str(k): v for k, v in example_traces.items()},
    }
    json_path = output_dir / f"{run_name}_expert_attribution.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    png_path = output_dir / f"{run_name}_expert_load_heatmap.png"
    plot_expert_load_heatmap(per_layer_load, str(png_path), title=f"Expert load — {run_name}")

    print(f"Saved {json_path}")
    print(f"Saved {png_path}")
    print("\nPer-layer top expert share:")
    for layer_idx in layers_present:
        counts = per_layer_load[layer_idx]
        total = sum(counts) or 1
        top_expert = max(range(len(counts)), key=lambda e: counts[e])
        share = counts[top_expert] / total
        flag = "  <-- collapsed" if share > 0.9 else ""
        print(f"  layer {layer_idx:3d}: top expert {top_expert} at {share:.1%} of traffic{flag}")


if __name__ == "__main__":
    main()
