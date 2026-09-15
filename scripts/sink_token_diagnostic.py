"""Identifies which token positions AND identities drive the norm-outlier
disconnection found at layers 1, 2, 4, and 31 (see extract_geometry.py,
multiscale_geometry.py).

The position-based half was originally built to distinguish two stories:
  - compounding structural artifact: the layer-31 spike is the same
    sink/BOS-anchored mechanism established at layers 1/2/4, its magnitude
    just compounding through the residual stream with depth. Predicted
    outliers anchored to a fixed START position at every flagged layer.
  - selection sharpening: the layer-31 spike is a distinct, prediction-
    related process. Predicted outliers shifting toward the END of each
    sequence at layer 31, unlike the early layers.

The first real run found neither: outliers weren't anchored to position 0
at all, they were anchored to a specific TOKEN IDENTITY (the newline
character) wherever it occurred — still consistent with "compounding
artifact" (same token, same mechanism, growing magnitude with depth), just
via a delimiter-token carrier rather than the textbook BOS-token one. The
per-token-identity half below extends that: is elevated norm specific to
newline, or shared by punctuation/delimiter tokens more broadly?

Runs a modest number of real sequences (no large pooling needed — this is a
qualitative look at a few hundred tokens, not a statistical estimate) and
reports, per layer, both the position distribution AND the per-token-id
norm ranking of the highest-norm tokens, plus decoded text.

    python scripts/sink_token_diagnostic.py \\
        --model mistralai/Mistral-7B-v0.1 \\
        --dataset wikitext \\
        --n_sequences 50 \\
        --layers 1 2 4 15 31
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from diffusion_moe.data.dataset import StreamingTextDataset, collate_fn
from diffusion_moe.data.tokenizer import TokenizerWrapper
from diffusion_moe.geometry.activation_capture import AttentionOutputCapture
from diffusion_moe.geometry.sink_diagnostics import (
    compare_delimiter_vs_content_norms,
    per_token_id_norm_summary,
    summarize_outlier_positions,
    token_norms_from_capture,
)
from diffusion_moe.utils.device import get_device
from diffusion_moe.utils.env import load_env

TOP_FRAC = 0.01
N_EXAMPLES_TO_PRINT = 8
MIN_TOKEN_ID_COUNT = 3
N_TOP_TOKEN_IDS_TO_PRINT = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--dataset", type=str, default="wikipedia")
    parser.add_argument("--n_sequences", type=int, default=50)
    parser.add_argument("--device", type=str, default=get_device())
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 4, 15, 31])
    parser.add_argument("--top_frac", type=float, default=TOP_FRAC)
    parser.add_argument("--min_token_id_count", type=int, default=MIN_TOKEN_ID_COUNT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/geometry")
    return parser.parse_args()


@torch.no_grad()
def collect_norms_and_tokens(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    layers: list[int],
    max_seq_len: int,
    pad_token_id: int,
) -> tuple[dict[int, list[Any]], list[Any], list[Any]]:
    """Runs every batch once, returns per-layer lists of (batch, seq) norm
    arrays (one array per batch, to be concatenated by the caller) plus the
    raw input_ids for every batch (for decoding outlier tokens back to text
    afterward — norms alone don't tell you WHAT the outlier token was).

    collate_fn pads each batch only to ITS OWN longest sequence, so
    different batches can have different widths — every batch here is
    right-padded up to the fixed `max_seq_len` (the tokenizer's own
    truncation cap, so always >= any batch's width) before being returned,
    so the caller can safely concatenate across batches along dim 0.
    """
    per_layer_norms: dict[int, list[Any]] = {layer: [] for layer in layers}
    all_input_ids: list[Any] = []
    all_masks: list[Any] = []

    with AttentionOutputCapture(model) as capture:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            capture.outputs = []
            model(input_ids, attention_mask=attention_mask)

            pad_width = max_seq_len - input_ids.shape[1]
            for layer in layers:
                norms = token_norms_from_capture(capture.outputs[layer], attention_mask)
                if pad_width > 0:
                    norms = np.pad(norms, ((0, 0), (0, pad_width)), constant_values=np.nan)
                per_layer_norms[layer].append(norms)

            ids_padded = input_ids.cpu()
            mask_padded = attention_mask.cpu()
            if pad_width > 0:
                ids_padded = torch.nn.functional.pad(
                    ids_padded, (0, pad_width), value=pad_token_id
                )
                mask_padded = torch.nn.functional.pad(mask_padded, (0, pad_width), value=0)
            all_input_ids.append(ids_padded)
            all_masks.append(mask_padded)

    return per_layer_norms, all_input_ids, all_masks


def main() -> None:
    load_env()
    args = parse_args()
    torch.manual_seed(args.seed)

    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    model.to(args.device)
    model.eval()

    tokenizer = TokenizerWrapper(args.model)
    dataset = StreamingTextDataset(
        args.dataset, tokenizer, max_seq_len=args.max_seq_len, take=args.n_sequences,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
    )

    per_layer_norms, all_input_ids, all_masks = collect_norms_and_tokens(
        model, loader, args.device, args.layers, args.max_seq_len, tokenizer.pad_token_id
    )

    input_ids_cat = torch.cat(all_input_ids, dim=0)
    mask_cat = torch.cat(all_masks, dim=0).numpy()

    input_ids_np = input_ids_cat.numpy()

    all_results = {}
    for layer in args.layers:
        norms_cat = np.concatenate(per_layer_norms[layer], axis=0)
        result = summarize_outlier_positions(norms_cat, mask_cat, top_frac=args.top_frac)

        token_stats = per_token_id_norm_summary(
            norms_cat, mask_cat, input_ids_np, min_count=args.min_token_id_count
        )
        token_texts = {
            s["token_id"]: tokenizer.decode([s["token_id"]], skip_special_tokens=False)
            for s in token_stats
        }
        delimiter_summary = compare_delimiter_vs_content_norms(token_stats, token_texts)

        result["per_token_id_norm_summary"] = token_stats[:N_TOP_TOKEN_IDS_TO_PRINT]
        result["delimiter_vs_content"] = delimiter_summary
        all_results[str(layer)] = result

        print(f"\nLayer {layer}: {result['n_flagged']}/{result['n_valid_tokens']} tokens flagged "
              f"(top {args.top_frac:.1%}), median_norm={result['median_norm']:.2f}")
        print(f"  frac at position 0 (sequence START): {result['frac_at_position_0']:.2f}")
        print(f"  frac at last valid position (sequence END): "
              f"{result['frac_at_last_position']:.2f}")
        print(f"  median dist_from_start={result['median_dist_from_start']:.1f}, "
              f"median dist_from_end={result['median_dist_from_end']:.1f}")

        print(f"  sample outlier tokens (top {N_EXAMPLES_TO_PRINT}):")
        for outlier in result["outliers"][:N_EXAMPLES_TO_PRINT]:
            token_id = int(input_ids_cat[outlier["batch_idx"], outlier["position"]])
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            print(
                f"    pos={outlier['position']:3d} (from_start={outlier['dist_from_start']:3d}, "
                f"from_end={outlier['dist_from_end']:3d}) norm={outlier['norm']:8.1f} "
                f"token_id={token_id:6d} text={token_text!r}"
            )

        d = delimiter_summary
        print(f"  delimiter-like tokens ({d['n_delimiter_token_ids']} distinct ids): "
              f"mean_norm={d['delimiter_mean_norm']}")
        print(f"  content tokens ({d['n_content_token_ids']} distinct ids): "
              f"mean_norm={d['content_mean_norm']}")
        print(f"  top {min(10, len(token_stats))} token ids by mean norm "
              f"(min_count={args.min_token_id_count}):")
        for s in token_stats[:10]:
            text = token_texts[s["token_id"]]
            print(
                f"    token_id={s['token_id']:6d} text={text!r:12} "
                f"count={s['count']:5d} mean_norm={s['mean_norm']:8.2f}"
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = args.model.replace("/", "_")
    json_path = output_dir / f"{safe_name}_sink_token_diagnostic.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {json_path}")


if __name__ == "__main__":
    main()
