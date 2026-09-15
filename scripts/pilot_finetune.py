"""Bounded pilot fine-tune: does a TRAINED diffusion router + experts close
the gap between diffusion-only routing and the oracle specialization
ceiling found at layers 2 and 4 (reports/phase1_findings_report.md §3.3)?

Every diagnostic in this investigation up to now was training-free, which
left one question structurally unanswerable: the dense model was never
trained with any incentive to organize around diffusion-cluster boundaries,
so "diffusion clustering finds weak specialization in the frozen model"
can't distinguish "not achievable" from "achievable, just not learned yet."
A real MoE's load-balancing loss actively reshapes neuron usage during
training in a way nothing training-free can simulate.

This script tests that directly, at minimum cost: freeze the entire
pretrained Mistral-7B-v0.1, splice a single PilotMoEBlock in place of ONE
decoder layer's dense MLP (default layer 2 — the strongest oracle-ceiling
candidate found), and train ONLY that block's parameters (centroids +
narrow expert FFNs) for a small number of steps with gradient checkpointing
on the frozen backbone. Everything else — attention, embeddings, every
other layer — stays exactly as pretrained.

Success signal: does task_loss fall substantially below its step-0 value
(and ideally approach the dense baseline's loss on the same batch) while
load_loss stays low (routing doesn't collapse onto one expert)? That's
direct evidence the architecture can learn to specialize here, distinct
from whether it already happens to for free.

    python scripts/pilot_finetune.py \\
        --model mistralai/Mistral-7B-v0.1 \\
        --dataset wikitext \\
        --layer 2 \\
        --n_experts 8 --top_k 2 \\
        --steps 300
"""

from __future__ import annotations

import argparse
import json
import os
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from diffusion_moe.data.dataset import StreamingTextDataset, collate_fn
from diffusion_moe.data.tokenizer import TokenizerWrapper
from diffusion_moe.models.pilot_moe_block import PilotMoEBlock
from diffusion_moe.routing.separation import landmark_scale
from diffusion_moe.training.losses import total_loss
from diffusion_moe.training.optimizer import build_optimizer, build_scheduler
from diffusion_moe.utils.device import get_device
from diffusion_moe.utils.env import load_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--dataset", type=str, default="wikipedia")
    parser.add_argument(
        "--local_data_files",
        type=str,
        nargs="+",
        default=None,
        help="Local parquet file path(s) to read --dataset's train split from directly, "
        "bypassing Hub streaming entirely. Use when Hub streaming is stalling on a flaky "
        "CDN path (hit twice in this project's history) -- pre-fetch the files with "
        "huggingface_hub.hf_hub_download (robust, resumable) or rsync them from a machine "
        "with a working connection, then pass their paths here.",
    )
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--n_experts", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--n_components", type=int, default=32)
    parser.add_argument("--n_landmarks", type=int, default=128)
    parser.add_argument("--diffusion_t", type=int, default=3)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.0,
        help="Std of Gaussian noise added to router logits during training only "
        "(noisy top-k gating, Shazeer et al. 2017) -- targets the 'early leader' "
        "load-imbalance pattern found in real pilot runs, where whichever expert "
        "centroids land closest to the data at k-means++ init dominate the "
        "softmax for the whole run regardless of seed. 0.0 = off (default, exact "
        "prior behavior).",
    )
    parser.add_argument("--centroid_refresh_steps", type=int, default=20)
    parser.add_argument(
        "--centroid_max_radius_factor",
        type=float,
        default=3.0,
        help="Caps each centroid's norm at this multiple of the current landmark scale "
        "after every optimizer step — without this, centroid_separation_loss's gradient "
        "has no upper bound on how far it pushes centroids apart, and a real run showed "
        "this diverges (sep_loss magnitude grew ~4 orders of magnitude over 300 steps "
        "while load_loss degenerated to exactly 0, consistent with the router collapsing "
        "toward uniform dispatch once centroids escaped the data's own coordinate range).",
    )
    parser.add_argument("--mu", type=float, default=0.01, help="load-balance loss weight")
    parser.add_argument("--nu", type=float, default=0.05, help="separation loss weight")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=get_device())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="results/pilot")
    parser.add_argument("--wandb_project", type=str, default="diffusion-moe")
    return parser.parse_args()


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


class DtypeCastWrapper(nn.Module):
    """Keeps the trainable PilotMoEBlock's parameters in fp32 for training
    stability while the frozen backbone runs in bf16 — casts x to fp32 at
    the block's input and back to the backbone's dtype at its output, so it
    drops straight into a bf16 decoder layer's `.mlp` slot unmodified.
    """

    def __init__(self, block: PilotMoEBlock, compute_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.block = block.to(dtype=compute_dtype)
        self.compute_dtype = compute_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x.to(self.compute_dtype))
        return out.to(x.dtype)

    @property
    def last_aux(self) -> dict[str, torch.Tensor] | None:
        return self.block.last_aux


@torch.no_grad()
def dense_baseline_loss(model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor) -> float:
    """One forward pass through the model AS-PRETRAINED (before the MoE
    splice), on the very first batch — a reference point for "how good was
    the original dense computation at this layer," so the pilot's own loss
    trajectory has something concrete to compare against besides its own
    step 0.
    """
    from diffusion_moe.data.dataset import IGNORE_INDEX

    outputs = model(input_ids)
    vocab_size = outputs.logits.shape[-1]
    loss = nn.functional.cross_entropy(
        outputs.logits.reshape(-1, vocab_size), labels.reshape(-1), ignore_index=IGNORE_INDEX
    )
    return float(loss.item())


def main() -> None:
    load_env()
    args = parse_args()
    torch.manual_seed(args.seed)

    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    model.to(args.device)

    tokenizer = TokenizerWrapper(args.model)
    dataset = StreamingTextDataset(
        args.dataset,
        tokenizer,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        local_data_files=args.local_data_files,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
    )

    freeze_all(model)

    # Dense baseline: one forward pass through the ORIGINAL, unmodified model
    # (still has its real pretrained dense mlp at this point) on a single
    # held-out batch, before the MLP gets spliced out — a concrete reference
    # point for "how good was the original dense computation here," so the
    # pilot's own loss trajectory has something to compare against besides
    # its own step 0.
    loader_iter = iter(loader)
    baseline_batch = next(loader_iter)
    dense_baseline_task_loss = dense_baseline_loss(
        model,
        baseline_batch["input_ids"].to(args.device),
        baseline_batch["labels"].to(args.device),
    )
    print(f"Dense baseline task_loss (original mlp, layer {args.layer}'s input batch held out): "
          f"{dense_baseline_task_loss:.4f}")

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # required for checkpointing through a frozen embedding

    target_layer = model.model.layers[args.layer]
    pilot_block = PilotMoEBlock(
        d_model=model.config.hidden_size,
        ffn_dim=model.config.intermediate_size,
        n_experts=args.n_experts,
        top_k=args.top_k,
        n_components=args.n_components,
        n_landmarks=args.n_landmarks,
        diffusion_t=args.diffusion_t,
        tau=args.tau,
        centroid_refresh_steps=args.centroid_refresh_steps,
        cosine=True,
        random_state=args.seed,
        noise_std=args.noise_std,
    )
    wrapped_block = DtypeCastWrapper(pilot_block, compute_dtype=torch.float32).to(args.device)
    target_layer.mlp = wrapped_block
    model.train()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,} total "
          f"({100 * n_trainable / n_total:.3f}%) — layer {args.layer}'s MoE block only")

    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, warmup_steps=args.warmup_steps, total_steps=args.steps)

    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        import wandb

        wandb.init(project=args.wandb_project, config=vars(args))

    history: list[dict[str, Any]] = []
    step = 0

    for batch in loader_iter:
        if step >= args.steps:
            break
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        labels = batch["labels"].to(args.device)

        outputs = model(input_ids, attention_mask=attention_mask)
        aux = {args.layer: wrapped_block.last_aux}
        result = total_loss(outputs.logits, labels, aux, mu=args.mu, nu=args.nu)

        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], args.grad_clip
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # centroid_separation_loss's gradient has no upper bound on how far
        # it pushes centroids apart (by design — see test_separation.py) —
        # cap norms back to a sane multiple of THIS layer's own diffusion-
        # coordinate scale after every step, or a real run diverges (see
        # --centroid_max_radius_factor's help text).
        psi_landmarks = torch.from_numpy(pilot_block.ndm.psi_landmarks_).to(
            dtype=pilot_block.centroids.centroids.dtype,
            device=pilot_block.centroids.centroids.device,
        )
        current_scale = landmark_scale(psi_landmarks)
        pilot_block.centroids.clip_norm_(max_norm=args.centroid_max_radius_factor * current_scale)

        # Per-expert dense load, not just the aggregate CV^2 scalar -- lets
        # us tell apart "collapse always lands on the same expert" (would
        # point at an initialization/architecture bias) from "collapse
        # target drifts around" (more consistent with a noisy/unstable
        # optimization landscape rather than a structural favorite).
        with torch.no_grad():
            per_expert_load = (
                torch.nn.functional.softmax(wrapped_block.last_aux["router_logits"], dim=-1)
                .reshape(-1, args.n_experts)
                .mean(dim=0)
                .tolist()
            )

        record = {
            "step": step,
            "task_loss": float(result["task_loss"]),
            "load_loss": float(result["load_loss"]),
            "sep_loss": float(result["sep_loss"]),
            "centroid_norm_mean": float(
                pilot_block.centroids.centroids.norm(dim=-1).mean()
            ),
            "per_expert_load": per_expert_load,
            "argmax_expert": int(torch.tensor(per_expert_load).argmax()),
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(record)
        if use_wandb:
            import wandb

            wandb.log(record, step=step)
        if step % args.log_every == 0:
            print(
                f"step {step:4d}: task_loss={record['task_loss']:.4f} "
                f"load_loss={record['load_loss']:.4f} sep_loss={record['sep_loss']:.4f}"
            )
        step += 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = args.model.replace("/", "_")
    result_path = output_dir / f"{safe_name}_layer{args.layer}_pilot.json"

    first_task_loss = history[0]["task_loss"] if history else None
    last_task_loss = history[-1]["task_loss"] if history else None
    summary = {
        "args": vars(args),
        "history": history,
        "dense_baseline_task_loss": dense_baseline_task_loss,
        "first_task_loss": first_task_loss,
        "last_task_loss": last_task_loss,
        "task_loss_delta": (
            last_task_loss - first_task_loss
            if first_task_loss is not None and last_task_loss is not None
            else None
        ),
        "mean_load_loss_last_20pct": (
            sum(h["load_loss"] for h in history[-max(1, len(history) // 5):])
            / max(1, len(history[-max(1, len(history) // 5):]))
            if history
            else None
        ),
    }
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\ndense baseline task_loss={dense_baseline_task_loss}")
    print(f"first task_loss={first_task_loss}, last task_loss={last_task_loss}")
    print(f"Saved {result_path}")
    if use_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
