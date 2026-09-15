"""Dyadic multiscale extension of the Phase-1 geometry kill switch.

extract_geometry.py commits to a single kernel bandwidth (the median
heuristic) per layer. The investigation that led here found that choice can
be actively misleading: on the real Mistral-7B run, several layers showed a
leading non-trivial eigenvalue at/near 1 (`likely_disconnected` in that
script's output) — the signature of a kernel graph fragmenting around a
handful of outlier-norm (attention-sink) tokens — which trivially collapses
r* to ~2 without reflecting any real low-dimensional structure.

This script re-analyzes the SAME captured activations at a dyadic ladder of
bandwidths around the median heuristic, in both raw-Euclidean and
cosine/angular metrics (see geometry/multiscale.py), and reports, per layer:
whether a stable r* plateau exists at all, and whether switching to the
cosine metric (which discards each token's raw activation norm — the exact
axis attention-sink tokens are outliers on) removes the disconnection.

Kept to a handful of representative layers by default (--layers), since a
full 32-layer x N-scale sweep costs roughly N times a single extract_geometry.py
run; widen it once the representative layers show where to look.

    python scripts/multiscale_geometry.py \\
        --model mistralai/Mistral-7B-v0.1 \\
        --n_sequences 200 \\
        --layers 0 8 16 24 31
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from diffusion_moe.data.dataset import StreamingTextDataset, collate_fn
from diffusion_moe.data.tokenizer import TokenizerWrapper
from diffusion_moe.geometry.activation_capture import (
    collect_layer_activations,
    compute_tokens_per_batch,
)
from diffusion_moe.geometry.multiscale import find_stable_window, multiscale_diffusion_analysis
from diffusion_moe.utils.device import get_device
from diffusion_moe.utils.env import load_env

N_LANDMARKS = 128
N_COMPONENTS = 32
DIFFUSION_T = 3
ALPHA = 1.0
N_SCALES = 9
BASE = 2.0
MAX_POOL_SIZE = 8192
DEFAULT_LAYERS = None  # None -> first/middle/last, resolved once n_layers is known


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--dataset", type=str, default="wikipedia")
    parser.add_argument("--n_sequences", type=int, default=200)
    parser.add_argument("--device", type=str, default=get_device())
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_pool_size", type=int, default=MAX_POOL_SIZE)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULT_LAYERS,
        help="Layer indices to analyze (default: first/middle/last).",
    )
    parser.add_argument("--n_scales", type=int, default=N_SCALES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/geometry")
    return parser.parse_args()


def analyze_layer_multiscale(post_Z, n_scales: int, seed: int) -> dict[str, Any]:
    """Runs the dyadic sweep in both metrics and summarizes whether a stable,
    non-disconnected r* plateau exists in each — the two candidate
    disconnection causes (degenerate bandwidth vs. genuine outlier-norm
    tokens) are distinguished by whether going cosine fixes it.
    """
    euclidean = multiscale_diffusion_analysis(
        post_Z, n_landmarks=N_LANDMARKS, n_components=N_COMPONENTS, t=DIFFUSION_T,
        alpha=ALPHA, n_scales=n_scales, base=BASE, cosine=False, random_state=seed,
    )
    cosine = multiscale_diffusion_analysis(
        post_Z, n_landmarks=N_LANDMARKS, n_components=N_COMPONENTS, t=DIFFUSION_T,
        alpha=ALPHA, n_scales=n_scales, base=BASE, cosine=True, random_state=seed,
    )
    euclidean_window = find_stable_window(euclidean["scales"])
    cosine_window = find_stable_window(cosine["scales"])
    return {
        "euclidean": euclidean,
        "cosine": cosine,
        "euclidean_stable_window": euclidean_window,
        "cosine_stable_window": cosine_window,
        "euclidean_stable_r_star": (
            euclidean["scales"][euclidean_window[0]]["r_star"] if euclidean_window else None
        ),
        "cosine_stable_r_star": (
            cosine["scales"][cosine_window[0]]["r_star"] if cosine_window else None
        ),
    }


def plot_layer(result: dict[str, Any], layer_idx: int, output_dir: Path, model_name: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in [(axes[0], "euclidean", "raw Euclidean"), (axes[1], "cosine", "cosine")]:
        scales = result[key]["scales"]
        ratios = [s["eps_ratio_to_median"] for s in scales]
        r_stars = [s["r_star"] for s in scales]
        colors = ["tab:red" if s["likely_disconnected"] else "tab:blue" for s in scales]
        ax.scatter(ratios, r_stars, c=colors)
        ax.plot(ratios, r_stars, color="gray", alpha=0.4, zorder=0)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("eps / median_heuristic_eps")
        ax.set_ylabel("r*")
        ax.set_title(f"{title} (red = likely disconnected)")
    fig.suptitle(f"{model_name} — layer {layer_idx}")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    path = output_dir / f"{safe_name}_multiscale_layer{layer_idx}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


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

    tokens_per_batch = compute_tokens_per_batch(
        args.max_pool_size, args.n_sequences, args.batch_size
    )
    _, post_arrays = collect_layer_activations(
        model, loader, args.device, tokens_per_batch, args.seed
    )

    n_layers = len(post_arrays)
    layers = args.layers or sorted({0, n_layers // 2, n_layers - 1})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = args.model.replace("/", "_")

    all_results = {}
    for layer_idx in layers:
        print(f"Layer {layer_idx}: sweeping {args.n_scales} scales x 2 metrics...")
        result = analyze_layer_multiscale(post_arrays[layer_idx], args.n_scales, args.seed)
        all_results[str(layer_idx)] = result
        plot_path = plot_layer(result, layer_idx, output_dir, args.model)

        euc_r = result["euclidean_stable_r_star"]
        cos_r = result["cosine_stable_r_star"]
        euc_len = len(result["euclidean_stable_window"])
        cos_len = len(result["cosine_stable_window"])
        print(f"  euclidean stable r* = {euc_r}  (window len {euc_len})")
        print(f"  cosine    stable r* = {cos_r}  (window len {cos_len})")
        if euc_r is None and cos_r is not None:
            print("  -> cosine metric resolves a disconnection the raw metric doesn't: "
                  "consistent with outlier-norm (attention-sink) tokens, not a genuine "
                  "degenerate bandwidth.")
        elif euc_r is None and cos_r is None:
            print("  -> no stable r* plateau in EITHER metric at any scale tested: "
                  "widen --n_scales before drawing conclusions about this layer.")
        print(f"  Saved {plot_path}")

    json_path = output_dir / f"{safe_name}_multiscale.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
