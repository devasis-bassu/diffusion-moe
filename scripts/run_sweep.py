"""Phase 2 hyperparameter sweep: router type x n_experts x top_k x nu_sep,
grid search via wandb.sweep(), each run training 5B tokens on The Pile at
300M scale, evaluated on wikitext perplexity at the end.

    python scripts/run_sweep.py --dry_run   # 100 steps, verify the pipeline end-to-end
    python scripts/run_sweep.py             # launch the full 72-run sweep
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
import wandb
from hydra import compose, initialize_config_dir

from diffusion_moe.data.dataloader import build_dataloaders
from diffusion_moe.evaluation.perplexity import compute_perplexity
from diffusion_moe.models.model_factory import build_model_from_config
from diffusion_moe.training.optimizer import build_optimizer, build_scheduler
from diffusion_moe.training.trainer import Trainer
from diffusion_moe.utils.env import load_env

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"

SWEEP_CONFIG: dict[str, Any] = {
    "method": "grid",
    "metric": {"name": "val_wikitext_perplexity", "goal": "minimize"},
    "parameters": {
        "router": {"values": ["diffusion", "cosine", "switch", "random"]},
        "n_experts": {"values": [4, 8, 16]},
        "top_k": {"values": [1, 2]},
        "nu_sep": {"values": [0.0, 0.01, 0.1]},
    },
}

RUN_TOKENS = 5_000_000_000
DRY_RUN_STEPS = 100
LOG_STEPS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run only 100 steps (one fixed hyperparameter combo, no wandb sweep) "
        "to verify the pipeline end-to-end before launching the full sweep.",
    )
    parser.add_argument("--project", type=str, default="diffusion-moe")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/sweep")
    parser.add_argument("--config-name", type=str, default="300m")
    return parser.parse_args()


def load_base_config(config_name: str) -> Any:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS_DIR)):
        return compose(config_name=config_name)


def apply_sweep_overrides(cfg: Any, sweep_params: dict[str, Any]) -> Any:
    """Applies one grid point's hyperparameters onto a copy of the base
    config, plus the sweep-wide fixed settings (The Pile, 5B tokens, log
    every 50 steps)."""
    cfg = cfg.copy()
    cfg.model.router = sweep_params["router"]
    cfg.routing.n_experts = sweep_params["n_experts"]
    cfg.routing.top_k = sweep_params["top_k"]
    cfg.routing.nu_sep = sweep_params["nu_sep"]
    cfg.data.dataset = "the_pile"
    cfg.training.total_tokens = RUN_TOKENS
    cfg.training.log_steps = LOG_STEPS
    return cfg


def compute_total_steps(cfg: Any) -> int:
    tokens_per_step = cfg.data.batch_size * cfg.data.max_seq_len * cfg.training.grad_accum_steps
    return max(1, cfg.training.total_tokens // tokens_per_step)


def active_flop_fraction(top_k: int, n_experts: int, overlap_factor: float = 1.0) -> float:
    """Fraction of a dense FFN's FLOPs actually spent per token by the MoE
    layer. Each of the top_k active experts is sized ffn_dim/n_experts*
    overlap_factor versus a dense FFN's full ffn_dim, so active FLOPs per
    token scale as top_k * overlap_factor / n_experts relative to dense."""
    return top_k * overlap_factor / n_experts


def _centroid_tensor(layer: torch.nn.Module) -> torch.Tensor | None:
    """Returns a layer's learned centroid tensor, whichever form it takes:
    DiffusionMoELayer wraps it in an ExpertCentroids module (.centroids.centroids),
    CosineMoELayer holds it directly as an nn.Parameter (.centroids). None for
    router types without centroids at all (switch, random)."""
    centroids = getattr(layer, "centroids", None)
    if isinstance(centroids, torch.Tensor):
        return centroids
    inner = getattr(centroids, "centroids", None)
    return inner if isinstance(inner, torch.Tensor) else None


def compute_mean_centroid_distance(model: torch.nn.Module) -> float | None:
    """Mean pairwise centroid distance, averaged over every MoE layer that
    has learned centroids. None if no layer does (switch/random routers, or
    a fully dense model) — the sweep should skip logging this metric then."""
    distances = []
    for layer in model.blocks:
        centroids = _centroid_tensor(layer)
        if centroids is None or centroids.shape[0] < 2:
            continue
        n = centroids.shape[0]
        pairwise = torch.cdist(centroids, centroids, p=2)
        mask = ~torch.eye(n, dtype=torch.bool, device=centroids.device)
        distances.append(pairwise[mask].mean().item())
    return sum(distances) / len(distances) if distances else None


def train_one_run(cfg: Any, max_steps: int, checkpoint_dir: str) -> dict[str, Any]:
    """Trains one hyperparameter combination end to end: builds model/data/
    optimizer/scheduler, runs the training loop with the sweep's per-50-step
    logging (task/load/sep loss, perplexity, mean centroid distance, routing
    CV, active FLOP fraction — all via Trainer.train_step/._log, so grad
    accumulation/mixed precision/clipping stay centralised in Trainer), then
    evaluates on wikitext and checkpoints.

    Returns the last logged metrics dict, plus val_wikitext_perplexity — for
    a --dry_run smoke test, or a caller that wants the final numbers directly
    rather than re-reading them from wandb.
    """
    model = build_model_from_config(cfg)
    train_loader, val_loader = build_dataloaders(cfg)
    optimizer = build_optimizer(model, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scheduler = build_scheduler(
        optimizer, warmup_steps=cfg.training.warmup_steps, total_steps=max_steps
    )
    trainer = Trainer(
        model, optimizer, scheduler, train_loader, val_loader, cfg, checkpoint_dir=checkpoint_dir
    )

    trainer.model.train()
    train_iter = iter(trainer.train_loader)
    last_metrics: dict[str, Any] = {}

    while trainer.step < max_steps:
        micro_batches = []
        for _ in range(trainer.grad_accum_steps):
            try:
                micro_batches.append(next(train_iter))
            except StopIteration:
                train_iter = iter(trainer.train_loader)
                micro_batches.append(next(train_iter))

        metrics = trainer.train_step(micro_batches)

        if trainer.step % trainer.log_steps == 0:
            metrics = dict(metrics)
            metrics["lr"] = trainer.scheduler.get_last_lr()[0]
            metrics["perplexity"] = math.exp(min(metrics["task_loss"], 20.0))
            metrics["routing_cv"] = math.sqrt(max(metrics["load_loss"], 0.0))
            mean_dist = compute_mean_centroid_distance(trainer.model)
            if mean_dist is not None:
                metrics["mean_centroid_distance"] = mean_dist
            metrics["active_flop_fraction"] = active_flop_fraction(
                cfg.routing.top_k, cfg.routing.n_experts
            )
            trainer._log(metrics, trainer.step)
            last_metrics = metrics

    wikitext_cfg = cfg.copy()
    wikitext_cfg.data.dataset = "wikitext"
    _, wikitext_loader = build_dataloaders(wikitext_cfg)
    wikitext_result = compute_perplexity(trainer.model, wikitext_loader)
    last_metrics["val_wikitext_perplexity"] = wikitext_result["perplexity"]
    trainer._log({"val_wikitext_perplexity": wikitext_result["perplexity"]}, trainer.step)

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(Path(checkpoint_dir) / "final.pt")

    return last_metrics


def make_sweep_run_fn(base_cfg: Any, max_steps: int, checkpoint_root: str):
    """Returns the function wandb.agent calls for each grid point: reads that
    trial's hyperparameters from wandb.config (populated by the agent before
    calling this), builds its config, and trains it."""

    def run_fn() -> None:
        wandb.init()
        sweep_params = {k: wandb.config[k] for k in SWEEP_CONFIG["parameters"]}
        cfg = apply_sweep_overrides(base_cfg, sweep_params)
        run_name = "_".join(f"{k}-{v}" for k, v in sorted(sweep_params.items()))
        checkpoint_dir = str(Path(checkpoint_root) / run_name)
        train_one_run(cfg, max_steps, checkpoint_dir)

    return run_fn


def main() -> None:
    load_env()
    args = parse_args()
    base_cfg = load_base_config(args.config_name)
    first_combo = {param: spec["values"][0] for param, spec in SWEEP_CONFIG["parameters"].items()}

    if args.dry_run:
        cfg = apply_sweep_overrides(base_cfg, first_combo)
        metrics = train_one_run(
            cfg, max_steps=DRY_RUN_STEPS, checkpoint_dir=str(Path(args.checkpoint_dir) / "dry_run")
        )
        print("Dry run complete:", metrics)
        return

    max_steps = compute_total_steps(apply_sweep_overrides(base_cfg, first_combo))
    sweep_id = wandb.sweep(SWEEP_CONFIG, project=args.project)
    wandb.agent(sweep_id, function=make_sweep_run_fn(base_cfg, max_steps, args.checkpoint_dir))


if __name__ == "__main__":
    main()
