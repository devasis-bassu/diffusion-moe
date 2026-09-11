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
import math
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
from diffusion_moe.geometry.intrinsic_dim import estimate_intrinsic_dim, spectral_gap
from diffusion_moe.geometry.nystrom import NystromDiffusionMap
from diffusion_moe.utils.device import get_device
from diffusion_moe.utils.env import load_env

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
    return parser.parse_args()


class AttentionOutputCapture:
    """Registers forward hooks on every decoder layer's self-attention
    submodule to capture its raw output — the post-attention, pre-FFN-residual
    point DiffusionMoELayer actually routes from. HF's own
    `output_hidden_states` only exposes states at layer *boundaries* (after
    the full attention+FFN block), one per layer, so it can't give us this
    intermediate point; a hook on `.self_attn` is the only way to reach it.

    Assumes a Llama/Mistral-family model (model.model.layers, each with a
    .self_attn submodule) — true of both target models and most HF causal LMs
    that share that code path.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise ValueError(
                "Expected a Llama/Mistral-family model exposing `model.model.layers` "
                "(a ModuleList of decoder layers, each with a `.self_attn` submodule)."
            )
        self.layers = model.model.layers
        self.outputs: list[torch.Tensor] = []
        self._handles: list[Any] = []

    def _hook(self, module: torch.nn.Module, inputs: Any, output: Any) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        self.outputs.append(tensor.detach())

    def __enter__(self) -> "AttentionOutputCapture":
        self.outputs = []
        self._handles = [
            layer.self_attn.register_forward_hook(self._hook) for layer in self.layers
        ]
        return self

    def __exit__(self, *exc_info: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def subsample_valid_tokens(
    z: torch.Tensor,
    attention_mask: torch.Tensor,
    n_tokens: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Samples up to n_tokens rows without replacement from the non-padded
    positions of z. z: (batch, seq, d_model). attention_mask: (batch, seq),
    1 for real tokens / 0 for padding — padded positions carry no meaningful
    representation and would pollute the geometry estimate.
    """
    flat_z = z.reshape(-1, z.shape[-1])
    flat_mask = attention_mask.reshape(-1).bool()
    valid = flat_z[flat_mask]
    n = min(n_tokens, valid.shape[0])
    if n == 0:
        return valid
    idx = torch.randperm(valid.shape[0], generator=generator)[:n]
    return valid[idx]


def compute_tokens_per_batch(max_pool_size: int, n_sequences: int, batch_size: int) -> int:
    """Derives how many tokens to pool from EACH batch, given a target TOTAL
    pooled-token budget PER LAYER for the whole run.

    A fixed per-batch quota (the original design) makes total pooled memory
    scale linearly with n_sequences: at n_sequences=1000 on the real 32-layer,
    4096-dim Mistral-7B, pooling 512 tokens/batch across ~125 batches, in
    float64, for both pre- and post-attention, across all 32 layers held in
    memory at once, needs ~125GB — which is exactly what OOM-killed a real
    run on a 48GB machine. Deriving the per-batch quota from a fixed total
    budget instead keeps memory roughly constant (~max_pool_size tokens/layer)
    no matter how large n_sequences gets.
    """
    expected_n_batches = max(1, math.ceil(n_sequences / batch_size))
    return max(1, max_pool_size // expected_n_batches)


@torch.no_grad()
def collect_layer_activations(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    tokens_per_batch: int,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Runs the model over every batch in `loader`, subsampling
    `tokens_per_batch` valid tokens per batch at every layer, and pools them
    across batches. Returns (pre_attention, post_attention), each a list
    (one entry per layer) of (n_pooled_tokens, d_model) float32 arrays.

    `tokens_per_batch` should be derived from a total per-layer budget via
    compute_tokens_per_batch, not passed as a large fixed constant — see that
    function's docstring for why a fixed per-batch quota doesn't scale.
    """
    generator = torch.Generator().manual_seed(seed)
    pre_pool: list[list[torch.Tensor]] = None
    post_pool: list[list[torch.Tensor]] = None

    with AttentionOutputCapture(model) as capture:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            capture.outputs = []  # hooks only append; clear before each batch's forward pass
            outputs = model(
                input_ids, attention_mask=attention_mask, output_hidden_states=True
            )
            n_layers = len(capture.outputs)

            if pre_pool is None:
                pre_pool = [[] for _ in range(n_layers)]
                post_pool = [[] for _ in range(n_layers)]

            # one shared sample of valid-token positions per batch, reused
            # across every layer's pre/post tensors for a like-for-like
            # pre-vs-post-attention comparison at the same tokens
            flat_mask = attention_mask.reshape(-1).bool()
            n_valid = int(flat_mask.sum().item())
            n = min(tokens_per_batch, n_valid)
            sample_idx = torch.randperm(n_valid, generator=generator)[:n]
            d_model_dim = outputs.hidden_states[0].shape[-1]

            for layer_idx in range(n_layers):
                pre = outputs.hidden_states[layer_idx].reshape(-1, d_model_dim)
                post = capture.outputs[layer_idx].reshape(-1, d_model_dim)
                pre_pool[layer_idx].append(pre[flat_mask][sample_idx].cpu())
                post_pool[layer_idx].append(post[flat_mask][sample_idx].cpu())

    # Chunks are kept in the model's own compute dtype (e.g. bf16) while
    # accumulating, then cast to float32 only once here — bf16 has no native
    # numpy representation (.numpy() would raise), and NystromDiffusionMap
    # already upcasts to float64 internally itself, one layer at a time, so
    # doing it here too (on all layers held in memory at once) would only
    # double memory for no benefit.
    pre_arrays = [torch.cat(layer_chunks, dim=0).float().numpy() for layer_chunks in pre_pool]
    post_arrays = [torch.cat(layer_chunks, dim=0).float().numpy() for layer_chunks in post_pool]
    return pre_arrays, post_arrays


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

    return {
        "r_star_post_attention": r_star_post,
        "r_star_pre_attention": r_star_pre,
        "spectral_gap": delta,
        "nystrom_error": {str(m): err for m, err in nystrom_error.items()},
        "top_eigenvalues": post_ndm.eigenvalues_[:5].tolist(),
    }


def run_geometry_extraction(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    d_model: int,
    tokens_per_batch: int = 512,
    seed: int = 42,
) -> dict[str, Any]:
    """Full pipeline over an already-constructed model/loader. Kept separate
    from main() so tests can drive it with a tiny local model and a mocked
    data stream, without any network access."""
    pre_arrays, post_arrays = collect_layer_activations(
        model, loader, device, tokens_per_batch, seed
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
        "wikipedia", tokenizer, max_seq_len=args.max_seq_len, take=args.n_sequences, seed=args.seed
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
    results = run_geometry_extraction(
        model, loader, args.device, d_model, tokens_per_batch=tokens_per_batch, seed=args.seed
    )
    results["model_name"] = args.model
    results["n_sequences"] = args.n_sequences

    output_dir = Path(args.output_dir)
    json_path = save_results(results, output_dir, args.model)
    png_path = plot_results(results, output_dir, args.model)

    mean_r_star = sum(layer["r_star_post_attention"] for layer in results["layers"]) / len(
        results["layers"]
    )
    print(f"Saved {json_path}")
    print(f"Saved {png_path}")
    print(f"Mean r* = {mean_r_star:.1f} vs d_model = {d_model}")
    if mean_r_star < d_model * 0.5:
        print("r* << d_model: manifold hypothesis holds — proceed to training.")
    else:
        print("r* is NOT much smaller than d_model: revisit the kernel design before training.")


if __name__ == "__main__":
    main()
