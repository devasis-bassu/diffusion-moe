"""Tests the FFN-width-reduction hypothesis directly, as distinct from the
routing-manifold hypothesis extract_geometry.py checks.

A low intrinsic dimension r* (extract_geometry.py's kill switch) is evidence
that token routing CAN be organized around a coarse, low-frequency manifold.
It is NOT evidence that each routed region's FFN computation is
correspondingly simpler — the diffusion coordinates are an explicit
low-pass filter, and the FFN's whole job (per this project's own framing) is
to resolve exactly the high-frequency detail that filter discards. The
width-reduction claim baked into expert_ffn.py (d_ff^(k) ~= d_ff / K) and
the G2 success criterion rest on a separate, unverified assumption: that
tokens sharing a coarse diffusion cluster also share which FFN "pattern
detector" neurons (Geva et al.) they need.

This script tests that assumption on the PRETRAINED DENSE model directly:
for each candidate expert count K, it clusters tokens by diffusion
coordinates (standing in for DiffusionRouter's assignment) and measures, per
cluster, how much of that cluster's real FFN activation mass a width-k
(=d_ff/K) slice of ITS OWN top neurons captures versus a cluster-agnostic
globally-shared slice of the same size. A large gap (see
geometry/ffn_specialization.py's neuron_specialization_analysis) supports
narrow per-cluster experts; a gap near zero falsifies it independent of r*.

BUT a null result against diffusion clustering alone is ambiguous: the dense
model was never trained with any incentive to organize around diffusion
boundaries, so "diffusion clustering finds no specialization" could mean
either "no specialization is achievable here" or "specialization is
achievable, this particular untrained routing signal just isn't finding
it" — a real MoE's load-balancing loss would actively reshape neuron usage
during training in a way nothing here simulates. To separate these, this
script ALSO clusters tokens directly by their own FFN activation pattern
(the oracle — the best-case partition for this exact metric, establishing
an achievable-specialization ceiling independent of any routing signal) and
measures how much of that ceiling the diffusion clustering already recovers
(Adjusted Rand Index / Normalized Mutual Info). Weak oracle specialization
means the FFN isn't separable by ANY grouping — real evidence against width
reduction at that layer. Strong oracle specialization with low diffusion-
recovery means specialization is achievable, just not by this fixed
signal — a router/training problem, not a fundamental one.

    python scripts/ffn_specialization.py \\
        --model mistralai/Mistral-7B-v0.1 \\
        --n_sequences 200 \\
        --layers 1 2 4 31 15 \\
        --n_experts 4 8 16
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
    compute_tokens_per_batch,
)
from diffusion_moe.geometry.ffn_specialization import (
    MLPInputCapture,
    cluster_agreement,
    cluster_tokens,
    neuron_specialization_analysis,
    oracle_cluster_tokens_by_activation,
)
from diffusion_moe.geometry.multiscale import l2_normalize
from diffusion_moe.geometry.nystrom import NystromDiffusionMap
from diffusion_moe.utils.device import get_device
from diffusion_moe.utils.env import load_env

N_LANDMARKS = 128
N_COMPONENTS = 32
DIFFUSION_T = 3
ALPHA = 1.0
MAX_POOL_SIZE = 8192
DEFAULT_N_EXPERTS = [4, 8, 16]  # matches configs/base_config.yaml's own sweep grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--dataset", type=str, default="wikipedia")
    parser.add_argument("--n_sequences", type=int, default=200)
    parser.add_argument("--device", type=str, default=get_device())
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_pool_size", type=int, default=MAX_POOL_SIZE)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--n_experts", type=int, nargs="+", default=DEFAULT_N_EXPERTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/geometry")
    return parser.parse_args()


@torch.no_grad()
def collect_z_and_mlp_activations(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    tokens_per_batch: int,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Like activation_capture.collect_layer_activations, but captures BOTH
    the post-attention z (for diffusion coordinates / cluster assignment)
    AND the MLP's down_proj input (the FFN neuron activations to analyze)
    from the SAME forward pass, subsampled with the SAME token indices per
    batch — needed so a cluster label derived from z_i lines up with the
    right token's FFN activation, not a different one.

    Returns (post_attention_z, mlp_activations), each a list (one entry per
    layer) of (n_pooled_tokens, dim) float32 arrays.
    """
    generator = torch.Generator().manual_seed(seed)
    z_pool: list[list[torch.Tensor]] = None
    mlp_pool: list[list[torch.Tensor]] = None

    with AttentionOutputCapture(model) as attn_capture, MLPInputCapture(model) as mlp_capture:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            attn_capture.outputs = []
            mlp_capture.outputs = []
            model(input_ids, attention_mask=attention_mask)
            n_layers = len(attn_capture.outputs)

            if z_pool is None:
                z_pool = [[] for _ in range(n_layers)]
                mlp_pool = [[] for _ in range(n_layers)]

            flat_mask = attention_mask.reshape(-1).bool()
            n_valid = int(flat_mask.sum().item())
            n = min(tokens_per_batch, n_valid)
            sample_idx = torch.randperm(n_valid, generator=generator)[:n]

            for layer_idx in range(n_layers):
                z = attn_capture.outputs[layer_idx]
                mlp = mlp_capture.outputs[layer_idx]
                d_z = z.shape[-1]
                d_mlp = mlp.shape[-1]
                z_pool[layer_idx].append(
                    z.reshape(-1, d_z)[flat_mask][sample_idx].cpu()
                )
                mlp_pool[layer_idx].append(
                    mlp.reshape(-1, d_mlp)[flat_mask][sample_idx].cpu()
                )

    z_arrays = [torch.cat(chunks, dim=0).float().numpy() for chunks in z_pool]
    mlp_arrays = [torch.cat(chunks, dim=0).float().numpy() for chunks in mlp_pool]
    return z_arrays, mlp_arrays


def analyze_layer_specialization(
    post_z: np.ndarray, mlp_activations: np.ndarray, n_experts_values: list[int], seed: int
) -> dict[str, Any]:
    """Clusters tokens by diffusion coordinates fit on COSINE-normalized
    activations, not raw post_z. An earlier raw-Euclidean version of this
    function fed plain k-means a diffusion map that's known to be
    near-disconnected at outlier-norm-affected layers (1, 2, 4, 31 — see
    extract_geometry.py's findings): the handful of extreme-norm tokens get
    isolated as their own landmarks/eigencomponents, and k-means duly peels
    them into tiny singleton "clusters" while everything else collapses into
    one dominant cluster — not a meaningful semantic partition at any K, and
    it happened even at the non-flagged control layer (15) tested during
    this investigation, since k-means finds whatever the diffusion
    coordinates hand it. The multiscale sweep already showed cosine
    normalization removes this disconnection at the default (auto
    median-heuristic) bandwidth for exactly these layers, without needing to
    know which token identity is responsible — so it's used here as the
    default clustering input. mlp_activations (what specialization is
    actually measured against) stays raw and unnormalized; only the
    clustering input changes.
    """
    post_z_for_clustering = l2_normalize(post_z)
    ndm = NystromDiffusionMap(
        n_landmarks=N_LANDMARKS, n_components=N_COMPONENTS, t=DIFFUSION_T, alpha=ALPHA,
        random_state=seed,
    )
    Psi = ndm.fit_transform(post_z_for_clustering)
    d_ff = mlp_activations.shape[1]

    by_k = {}
    for n_experts in n_experts_values:
        width_k = d_ff // n_experts

        diffusion_labels = cluster_tokens(Psi, n_clusters=n_experts, random_state=seed)
        diffusion_result = neuron_specialization_analysis(
            mlp_activations, diffusion_labels, width_k
        )

        # Oracle: the best-case partition for THIS metric, clustering tokens
        # directly by their own FFN activation pattern rather than by
        # diffusion coordinates. Establishes a ceiling on achievable
        # specialization, independent of whether diffusion geometry (or any
        # other routing signal) can find it — see module docstring.
        oracle_labels = oracle_cluster_tokens_by_activation(
            mlp_activations, n_clusters=n_experts, n_components=N_COMPONENTS, random_state=seed
        )
        oracle_result = neuron_specialization_analysis(mlp_activations, oracle_labels, width_k)

        agreement = cluster_agreement(diffusion_labels, oracle_labels)

        by_k[str(n_experts)] = {
            "diffusion": diffusion_result,
            "oracle": oracle_result,
            "agreement": agreement,
        }

    return {"d_ff": d_ff, "by_n_experts": by_k}


def plot_layer(result: dict[str, Any], layer_idx: int, output_dir: Path, model_name: str) -> Path:
    n_experts_values = sorted(int(k) for k in result["by_n_experts"])
    diffusion_gains = [
        result["by_n_experts"][str(k)]["diffusion"]["mean_specialization_gain"]
        for k in n_experts_values
    ]
    oracle_gains = [
        result["by_n_experts"][str(k)]["oracle"]["mean_specialization_gain"]
        for k in n_experts_values
    ]
    ari = [
        result["by_n_experts"][str(k)]["agreement"]["adjusted_rand_index"]
        for k in n_experts_values
    ]

    x = np.arange(len(n_experts_values))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(
        x - width / 2, oracle_gains, width, color="tab:green", alpha=0.8, label="oracle (ceiling)"
    )
    ax1.bar(
        x + width / 2, diffusion_gains, width, color="tab:blue", alpha=0.8, label="diffusion"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(k) for k in n_experts_values])
    ax1.set_xlabel("n_experts (K)")
    ax1.set_ylabel("mean specialization gain")
    ax1.axhline(0, color="gray", linewidth=0.8)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(x, ari, color="tab:red", marker="o", label="diffusion-vs-oracle ARI")
    ax2.set_ylabel("adjusted rand index (diffusion vs. oracle)", color="tab:red")
    ax2.set_ylim(-0.05, 1.05)

    fig.suptitle(
        f"{model_name} — layer {layer_idx}\n"
        "(green = achievable ceiling, blue = what diffusion routing finds, "
        "red = how much of the ceiling it recovers)"
    )
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    path = output_dir / f"{safe_name}_ffn_specialization_layer{layer_idx}.png"
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
    z_arrays, mlp_arrays = collect_z_and_mlp_activations(
        model, loader, args.device, tokens_per_batch, args.seed
    )

    n_layers = len(z_arrays)
    layers = args.layers or sorted({0, n_layers // 2, n_layers - 1})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = args.model.replace("/", "_")

    all_results = {}
    for layer_idx in layers:
        print(f"Layer {layer_idx}: clustering at K in {args.n_experts}...")
        result = analyze_layer_specialization(
            z_arrays[layer_idx], mlp_arrays[layer_idx], args.n_experts, args.seed
        )
        all_results[str(layer_idx)] = result
        plot_path = plot_layer(result, layer_idx, output_dir, args.model)

        for k in sorted(int(x) for x in args.n_experts):
            stats = result["by_n_experts"][str(k)]
            diff, oracle, agree = stats["diffusion"], stats["oracle"], stats["agreement"]
            diff_gain, diff_jacc = diff["mean_specialization_gain"], diff["mean_pairwise_jaccard"]
            oracle_gain = oracle["mean_specialization_gain"]
            ari = agree["adjusted_rand_index"]

            if oracle_gain < 0.05:
                verdict = (
                    "NO achievable specialization at this layer, by ANY partition -- "
                    "not a router problem, the FFN itself isn't separable here"
                )
            elif diff_gain > 0.15 and diff_jacc < 0.5:
                verdict = "diffusion routing already finds real specialization"
            elif ari > 0.3:
                verdict = "achievable, and diffusion routing partially recovers it"
            else:
                verdict = (
                    "achievable (oracle ceiling exists) but diffusion routing ISN'T finding it "
                    "-- router/training problem, not a fundamental impossibility"
                )

            print(
                f"  K={k:3d}: oracle_gain={oracle_gain:+.3f} diffusion_gain={diff_gain:+.3f} "
                f"ARI={ari:+.3f} -> {verdict}"
            )
        print(f"  Saved {plot_path}")

    json_path = output_dir / f"{safe_name}_ffn_specialization.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
