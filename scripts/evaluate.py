"""Evaluation entrypoint: loads a checkpoint, runs perplexity and routing
diagnostics (and optionally lm-evaluation-harness tasks), and saves
results/eval/{run_name}_results.json.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --run_name my_run
    python scripts/evaluate.py --checkpoint a.pt --compare b.pt
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
from diffusion_moe.evaluation.benchmarks import run_lm_eval
from diffusion_moe.evaluation.perplexity import compute_perplexity
from diffusion_moe.evaluation.routing_analysis import RoutingAnalyser
from diffusion_moe.models.model_factory import build_model_from_config
from diffusion_moe.utils.device import get_device
from diffusion_moe.utils.env import load_env

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--compare", type=str, default=None, help="A second checkpoint to compare against."
    )
    parser.add_argument("--config-name", type=str, default="base_config")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/eval")
    parser.add_argument("--device", type=str, default=get_device())
    parser.add_argument(
        "--lm_eval_tasks",
        type=str,
        nargs="*",
        default=[],
        help=(
            "Optional lm-evaluation-harness tasks to also run. Requires an "
            "HF-compatible model — DiffusionMoETransformer isn't one, so this "
            "will fail unless you've wrapped it accordingly."
        ),
    )
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


def _run_routing_analysis(
    model: torch.nn.Module, loader: Any, device: str
) -> dict[str, Any] | None:
    """Returns routing diagnostics if the model has any DiffusionMoELayer,
    else None (e.g. a fully-dense ablation checkpoint)."""
    try:
        analyser = RoutingAnalyser(model)
    except ValueError:
        return None

    with analyser:
        for batch in loader:
            model(batch["input_ids"].to(device))

    return {
        "expert_token_counts": analyser.expert_token_counts(),
        "routing_entropy_per_expert": analyser.routing_entropy_per_expert(),
    }


def evaluate_checkpoint(
    checkpoint_path: str, cfg: Any, device: str, lm_eval_tasks: list[str]
) -> dict[str, Any]:
    """Runs every compatible evaluation on one checkpoint: perplexity and
    bits-per-byte always; expert routing diagnostics if the model has any
    DiffusionMoELayer; lm-evaluation-harness tasks if requested."""
    model = load_model_from_checkpoint(checkpoint_path, cfg, device)
    _, val_loader = build_dataloaders(cfg)

    results: dict[str, Any] = {"checkpoint": checkpoint_path}
    results.update(compute_perplexity(model, val_loader))

    routing = _run_routing_analysis(model, val_loader, device)
    if routing is not None:
        results.update(routing)

    if lm_eval_tasks:
        try:
            tokenizer = TokenizerWrapper(cfg.data.tokenizer)
            results["lm_eval"] = run_lm_eval(model, tokenizer, lm_eval_tasks)
        except Exception as exc:  # noqa: BLE001 - report and continue, don't crash the run
            print(f"lm_eval tasks failed ({exc.__class__.__name__}: {exc}); skipping.")
            results["lm_eval_error"] = str(exc)

    return results


def save_results(results: dict[str, Any], output_dir: Path, run_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{run_name}_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def _run_name_from_checkpoint(checkpoint_path: str) -> str:
    return Path(checkpoint_path).stem


def print_comparison_table(results_a: dict[str, Any], results_b: dict[str, Any]) -> None:
    scalar_keys = [
        key
        for key in results_a
        if key != "checkpoint" and isinstance(results_a[key], (int, float, str))
    ]
    name_a = Path(results_a["checkpoint"]).name
    name_b = Path(results_b["checkpoint"]).name

    header = f"{'metric':<28}{name_a:>20}{name_b:>20}"
    print(header)
    print("-" * len(header))
    for key in scalar_keys:
        val_a, val_b = results_a.get(key, "-"), results_b.get(key, "-")
        val_a_str = f"{val_a:.4f}" if isinstance(val_a, float) else str(val_a)
        val_b_str = f"{val_b:.4f}" if isinstance(val_b, float) else str(val_b)
        print(f"{key:<28}{val_a_str:>20}{val_b_str:>20}")


def main() -> None:
    load_env()
    args = parse_args()
    cfg = load_config(args.config_name)

    run_name = args.run_name or _run_name_from_checkpoint(args.checkpoint)
    results = evaluate_checkpoint(args.checkpoint, cfg, args.device, args.lm_eval_tasks)
    path = save_results(results, Path(args.output_dir), run_name)
    print(f"Saved {path}")
    print(json.dumps({k: v for k, v in results.items() if not isinstance(v, dict)}, indent=2))

    if args.compare:
        compare_run_name = _run_name_from_checkpoint(args.compare)
        compare_results = evaluate_checkpoint(args.compare, cfg, args.device, args.lm_eval_tasks)
        compare_path = save_results(compare_results, Path(args.output_dir), compare_run_name)
        print(f"Saved {compare_path}")
        print()
        print_comparison_table(results, compare_results)


if __name__ == "__main__":
    main()
