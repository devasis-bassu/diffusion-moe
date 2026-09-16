"""Phase 1 kill-switch: does the manifold hypothesis hold for a pretrained
model's hidden states?

Runs a pretrained causal LM over streamed Wikipedia text and, at every layer,
fits a diffusion map on its post-attention activations to estimate the
intrinsic dimension r*. Run this BEFORE committing to any training run: if r*
is not << d_model, the diffusion-routing design has no low-dimensional
structure to exploit, and the project should stop here to revisit the kernel
design rather than spend compute training on top of it.

    python scripts/extract_geometry.py \\
        --model mistralai/Mistral-7B-v0.1 \\
        --n_sequences 1000 \\
        --device cuda
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
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from diffusion_moe.data.dataset import StreamingTextDataset, collate_fn
from diffusion_moe.data.tokenizer import TokenizerWrapper
from diffusion_moe.geometry.activation_capture import (
    AttentionOutputCapture,
    collect_layer_activations,
    compute_tokens_per_batch,
    subsample_valid_tokens,
)
from diffusion_moe.geometry.intrinsic_dim import estimate_intrinsic_dim, spectral_gap
from diffusion_moe.geometry.nystrom import NystromDiffusionMap
from diffusion_moe.utils.device import get_device
from diffusion_moe.utils.env import load_env

__all__ = [
    "AttentionOutputCapture",
    "collect_layer_activations",
    "compute_tokens_per_batch",
    "subsample_valid_tokens",
]

NYSTROM_M_VALUES = [32, 64, 128, 256]
N_LANDMARKS = 128
N_COMPONENTS = 32
DIFFUSION_T = 3
ALPHA = 1.0
INTRINSIC_DIM_THRESHOLD = 0.95
# Total pooled tokens kept PER LAYER, regardless of n_sequences. 32x the
# largest landmark budget (256) is comfortable oversampling for KMeans/Nystrom
# without the memory blowing up as n_sequences grows — see
# compute_tokens_per_batch's docstring for why this has to be a total budget,
# not a fixed per-batch amount.
MAX_POOL_SIZE = 8192


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--n_sequences", type=int, default=1000)
    parser.add_argument("--device", type=str, default=get_device())
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument(
        "--max_pool_size",
        type=int,
        default=MAX_POOL_SIZE,
        help="Total pooled tokens kept per layer across the whole run (not per "
        "batch) — bounds memory regardless of --n_sequences.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/geometry")
    parser.add_argument("--dataset", type=str, default="wikipedia")
    parser.add_argument(
        "--local_data_files",
        type=str,
        nargs="+",
        default=None,
        help="Local parquet file path(s) to read --dataset's train split from directly, "
        "bypassing Hub streaming -- see StreamingTextDataset's local_data_files docstring "
        "for why this exists (a recurring flaky-CDN stall, hit on two different datasets "
        "across this investigation).",
    )
    parser.add_argument(
        "--exclude_token_ids",
        type=int,
        nargs="*",
        default=None,
        help="Token ids to drop from geometry pooling entirely (recommendation 6 in "
        "phase1_findings_report.md) — e.g. the newline token (id 13 for Mistral-7B's "
        "tokenizer) responsible for most of the kernel-disconnection pathology at "
        "layers 1/2/4. Not specified by default, so behavior is unchanged unless "
        "explicitly requested.",
    )
    return parser.parse_args()


def nystrom_approximation_error(
    Z: np.ndarray,
    m_values: list[int],
    n_components: int = N_COMPONENTS,
    t: int = DIFFUSION_T,
    alpha: float = ALPHA,
    random_state: int = 42,
) -> dict[int, float]:
    """Measures Nystrom approximation quality at each landmark budget m.

    Diffusion coordinates aren't uniquely defined (sign flips per eigenvector,
    arbitrary rotation within near-degenerate eigenspaces), so raw coordinates
    from different landmark counts aren't directly comparable. Pairwise
    diffusion distances ARE invariant to that ambiguity, so error is the
    relative Frobenius distance between pairwise-distance matrices, each
    computed from its own landmark budget's Nystrom map, referenced against
    the largest m tested (a true full decomposition on every pooled token
    would defeat the point of testing Nystrom's approximation cost at all).
    """
    reference_m = max(m_values)
    reference_ndm = NystromDiffusionMap(
        n_landmarks=reference_m, n_components=n_components, t=t, alpha=alpha,
        random_state=random_state,
    )
    reference_psi = reference_ndm.fit_transform(Z)
    reference_dist = _pairwise_dist(reference_psi)

    errors: dict[int, float] = {}
    for m in m_values:
        if m == reference_m:
            errors[m] = 0.0
            continue
        ndm = NystromDiffusionMap(
            n_landmarks=m, n_components=n_components, t=t, alpha=alpha,
            random_state=random_state,
        )
        psi = ndm.fit_transform(Z)
        dist = _pairwise_dist(psi)
        rel_error = np.linalg.norm(dist - reference_dist) / (np.linalg.norm(reference_dist) + 1e-12)
        errors[m] = float(rel_error)
    return errors


def _pairwise_dist(X: np.ndarray) -> np.ndarray:
    from scipy.spatial.distance import pdist, squareform

    return squareform(pdist(X, metric="euclidean"))


def pool_diagnostics(Z: np.ndarray) -> dict[str, float]:
    """Cheap, no-model-inference-required stats on a pooled-token array that
    distinguish the two disconnection hypotheses without rerunning anything:

    - token_norm_max_to_median: a handful of outlier-norm tokens (e.g.
      attention-sink / massive-activation positions, well documented for
      Llama/Mistral-family models) show up as a huge ratio here even though
      only a few rows are affected.
    - duplicate_fraction: bf16 has ~3 decimal digits of precision, so if many
      pooled activations round to identical bf16 values, this fraction will
      be far from 0 — that alone can degenerate the median-heuristic
      bandwidth to ~0 (see bandwidth_median_heuristic's zero-median fallback).
    """
    norms = np.linalg.norm(Z, axis=1)
    median_norm = float(np.median(norms))
    max_norm = float(norms.max())
    n_unique = int(np.unique(Z, axis=0).shape[0])
    n_total = int(Z.shape[0])
    return {
        "token_norm_min": float(norms.min()),
        "token_norm_median": median_norm,
        "token_norm_max": max_norm,
        "token_norm_max_to_median": max_norm / median_norm if median_norm > 0 else float("inf"),
        "n_tokens": n_total,
        "n_unique_tokens": n_unique,
        "duplicate_fraction": float(1.0 - n_unique / n_total) if n_total > 0 else 0.0,
    }


def analyze_layer(
    pre_Z: np.ndarray,
    post_Z: np.ndarray,
    n_landmarks: int = N_LANDMARKS,
    n_components: int = N_COMPONENTS,
    t: int = DIFFUSION_T,
    alpha: float = ALPHA,
    m_values: list[int] = NYSTROM_M_VALUES,
    random_state: int = 42,
) -> dict[str, Any]:
    """Runs the full per-layer geometry analysis: intrinsic dim and spectral
    gap from a fixed-n_landmarks diffusion map on post-attention activations,
    the pre-vs-post-attention r* comparison, and the Nystrom-error sweep.
    """
    post_ndm = NystromDiffusionMap(
        n_landmarks=n_landmarks, n_components=n_components, t=t, alpha=alpha,
        random_state=random_state,
    )
    post_ndm.fit(post_Z)
    r_star_post = estimate_intrinsic_dim(
        post_ndm.eigenvalues_, t=t, threshold=INTRINSIC_DIM_THRESHOLD
    )
    delta = spectral_gap(post_ndm.eigenvalues_)

    pre_ndm = NystromDiffusionMap(
        n_landmarks=n_landmarks, n_components=n_components, t=t, alpha=alpha,
        random_state=random_state,
    )
    pre_ndm.fit(pre_Z)
    r_star_pre = estimate_intrinsic_dim(
        pre_ndm.eigenvalues_, t=t, threshold=INTRINSIC_DIM_THRESHOLD
    )

    nystrom_error = nystrom_approximation_error(
        post_Z, m_values, n_components=n_components, t=t, alpha=alpha, random_state=random_state
    )

    # NystromDiffusionMap.eigenvalues_ already excludes the trivial top
    # eigenvalue (~1, the constant eigenvector). If the largest REMAINING
    # eigenvalue is still ~1, the landmark Markov chain has more than one
    # eigenvalue at/near 1 — i.e. the kernel graph is disconnected (or
    # numerically so) into near-isolated components, most likely because a
    # few landmarks sit on extreme-norm outlier tokens (e.g. attention-sink /
    # BOS positions, well documented for Llama/Mistral-family models) that
    # the median-heuristic bandwidth can't bridge. When that happens, a tiny
    # r* reflects graph fragmentation, not a genuine low-dimensional
    # manifold, and should NOT be read as confirming the manifold hypothesis.
    likely_disconnected = bool(post_ndm.eigenvalues_[0] > 0.999)

    return {
        "r_star_post_attention": r_star_post,
        "r_star_pre_attention": r_star_pre,
        "spectral_gap": delta,
        "nystrom_error": {str(m): err for m, err in nystrom_error.items()},
        "top_eigenvalues": post_ndm.eigenvalues_[:5].tolist(),
        "likely_disconnected": likely_disconnected,
        # eps_ near float64 machine epsilon (~2.2e-16) means the zero-median
        # fallback in bandwidth_median_heuristic fired — i.e. many landmark
        # points were exact/near duplicates (see pool_diagnostics below for
        # the token-level evidence of that).
        "post_eps": post_ndm.eps_,
        "pre_eps": pre_ndm.eps_,
        "post_pool_diagnostics": pool_diagnostics(post_Z),
        "pre_pool_diagnostics": pool_diagnostics(pre_Z),
    }


def run_geometry_extraction(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    d_model: int,
    tokens_per_batch: int = 512,
    seed: int = 42,
    excluded_token_ids: set[int] | None = None,
) -> dict[str, Any]:
    """Full pipeline over an already-constructed model/loader. Kept separate
    from main() so tests can drive it with a tiny local model and a mocked
    data stream, without any network access."""
    pre_arrays, post_arrays = collect_layer_activations(
        model, loader, device, tokens_per_batch, seed, excluded_token_ids=excluded_token_ids
    )

    layers = []
    for layer_idx, (pre_Z, post_Z) in enumerate(zip(pre_arrays, post_arrays)):
        layer_result = analyze_layer(pre_Z, post_Z, random_state=seed)
        layer_result["layer"] = layer_idx
        layer_result["n_tokens_pooled"] = post_Z.shape[0]
        layers.append(layer_result)

    return {"d_model": d_model, "n_layers": len(layers), "layers": layers}


def save_results(results: dict[str, Any], output_dir: Path, model_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    path = output_dir / f"{safe_name}_geometry.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def plot_results(results: dict[str, Any], output_dir: Path, model_name: str) -> Path:
    layers = results["layers"]
    layer_ids = [layer["layer"] for layer in layers]
    r_stars = [layer["r_star_post_attention"] for layer in layers]
    gaps = [layer["spectral_gap"] for layer in layers]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(layer_ids, r_stars, marker="o")
    axes[0].axhline(results["d_model"], color="gray", linestyle="--", label="d_model")
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("r*")
    axes[0].set_title("Intrinsic dimension vs layer")
    axes[0].legend()

    axes[1].plot(layer_ids, gaps, marker="o", color="tab:orange")
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("spectral gap (lambda_1 - lambda_2)")
    axes[1].set_title("Spectral gap vs layer")

    selected_indices = sorted(set([0, len(layers) // 2, len(layers) - 1]))
    for idx in selected_indices:
        layer = layers[idx]
        m_values = sorted(int(m) for m in layer["nystrom_error"])
        errors = [layer["nystrom_error"][str(m)] for m in m_values]
        axes[2].plot(m_values, errors, marker="o", label=f"layer {layer['layer']}")
    axes[2].set_xlabel("n_landmarks (m)")
    axes[2].set_ylabel("relative Nystrom error")
    axes[2].set_title("Nystrom approximation error vs m")
    axes[2].legend()

    fig.suptitle(model_name)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    path = output_dir / f"{safe_name}_geometry.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    load_env()
    args = parse_args()
    torch.manual_seed(args.seed)

    # bf16 halves memory and is meaningfully faster than fp32 on both CUDA and
    # MPS (Apple Silicon GPUs support it fine); fp32 is only needed on plain CPU.
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    model.to(args.device)
    model.eval()

    tokenizer = TokenizerWrapper(args.model)
    dataset = StreamingTextDataset(
        args.dataset,
        tokenizer,
        max_seq_len=args.max_seq_len,
        take=args.n_sequences,
        seed=args.seed,
        local_data_files=args.local_data_files,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
    )

    d_model = model.config.hidden_size
    tokens_per_batch = compute_tokens_per_batch(
        args.max_pool_size, args.n_sequences, args.batch_size
    )
    excluded_token_ids = set(args.exclude_token_ids) if args.exclude_token_ids else None
    results = run_geometry_extraction(
        model,
        loader,
        args.device,
        d_model,
        tokens_per_batch=tokens_per_batch,
        seed=args.seed,
        excluded_token_ids=excluded_token_ids,
    )
    results["excluded_token_ids"] = sorted(excluded_token_ids) if excluded_token_ids else []
    results["model_name"] = args.model
    results["n_sequences"] = args.n_sequences
    results["dataset"] = args.dataset

    output_dir = Path(args.output_dir)
    json_path = save_results(results, output_dir, args.model)
    png_path = plot_results(results, output_dir, args.model)

    mean_r_star = sum(layer["r_star_post_attention"] for layer in results["layers"]) / len(
        results["layers"]
    )
    n_disconnected = sum(1 for layer in results["layers"] if layer["likely_disconnected"])

    print(f"Saved {json_path}")
    print(f"Saved {png_path}")
    print(f"Mean r* = {mean_r_star:.1f} vs d_model = {d_model}")

    if n_disconnected > 0:
        print(
            f"WARNING: {n_disconnected}/{len(results['layers'])} layers show a "
            "leading non-trivial eigenvalue ~1 — the landmark kernel graph is "
            "likely fragmenting into near-disconnected components (e.g. from "
            "outlier/attention-sink tokens), which trivially deflates r* without "
            "reflecting a genuine low-dimensional manifold. Do NOT treat this run "
            "as confirming the manifold hypothesis until that's ruled out — see "
            "'likely_disconnected' per layer in the saved JSON."
        )
        # Cheap breakdown of the two candidate causes, so the JSON doesn't
        # have to be hand-inspected to tell them apart:
        #  - near-machine-epsilon eps_ -> zero-median fallback fired ->
        #    duplicate/near-duplicate landmark points (bf16-rounding collapse)
        #  - large token_norm_max_to_median with few duplicates -> genuine
        #    outlier-norm tokens (attention-sink / massive-activation)
        min_eps = min(layer["post_eps"] for layer in results["layers"])
        max_dup_frac = max(
            layer["post_pool_diagnostics"]["duplicate_fraction"] for layer in results["layers"]
        )
        max_norm_ratio = max(
            layer["post_pool_diagnostics"]["token_norm_max_to_median"]
            for layer in results["layers"]
        )
        print(
            f"  diagnostics: min post_eps={min_eps:.3g} "
            f"(machine eps={np.finfo(np.float64).eps:.3g}), "
            f"max duplicate_fraction={max_dup_frac:.3f}, "
            f"max token_norm_max_to_median={max_norm_ratio:.1f}"
        )
    # "<<" (much less than) is the guide's actual criterion, not merely "less
    # than" — an order-of-magnitude margin is used here so a weakly-compressed
    # r* (e.g. d_model/2) can't slip through as a false "holds".
    elif mean_r_star < d_model / 10:
        print("r* << d_model: manifold hypothesis holds — proceed to training.")
    else:
        print("r* is NOT much smaller than d_model: revisit the kernel design before training.")


if __name__ == "__main__":
    main()
