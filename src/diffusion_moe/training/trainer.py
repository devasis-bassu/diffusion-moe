"""Training loop: gradient accumulation, mixed precision, checkpointing, logging."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.nn.parallel import DistributedDataParallel

from diffusion_moe.data.dataset import IGNORE_INDEX
from diffusion_moe.training.losses import total_loss
from diffusion_moe.utils.device import (
    all_reduce_mean,
    get_device,
    get_local_rank,
    is_distributed,
    is_main_process,
)

try:
    import wandb
except ImportError:  # wandb is optional locally; required only for real training runs
    wandb = None


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    """Reads `key` from cfg, whether cfg is a dict, OmegaConf node, or namespace."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


class Trainer:
    """Drives training for a model whose forward(input_ids) returns
    (logits, activations, router_outputs) — DiffusionMoETransformer's
    signature (router_outputs may be empty for a fully dense model).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_loader: Iterable,
        val_loader: Iterable,
        config: Any,
        device: str | None = None,
        checkpoint_dir: str | Path = "checkpoints",
    ) -> None:
        self.device = device or get_device()
        self.device_type = self.device.split(":")[0]
        # Under a torchrun-launched multi-GPU run, each rank owns one CUDA
        # device (LOCAL_RANK) and wraps the model in DDP, which all-reduces
        # gradients across ranks automatically during .backward() — no change
        # needed to the training loop itself. MPS has no DDP backend, so
        # local (single-Mac) runs are always plain single-device training.
        self.is_ddp = is_distributed() and self.device_type == "cuda"
        if self.is_ddp:
            self.device = f"cuda:{get_local_rank()}"

        model = model.to(self.device)
        self.model = (
            DistributedDataParallel(model, device_ids=[get_local_rank()])
            if self.is_ddp
            else model
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        training_cfg = _get(config, "training", {})
        routing_cfg = _get(config, "routing", {})
        wandb_cfg = _get(config, "wandb", {})

        self.grad_accum_steps = _get(training_cfg, "grad_accum_steps", 1)
        self.grad_clip = _get(training_cfg, "grad_clip", 1.0)
        self.checkpoint_steps = _get(training_cfg, "checkpoint_steps", 5000)
        self.eval_steps = _get(training_cfg, "eval_steps", 1000)
        self.log_steps = _get(training_cfg, "log_steps", 50)
        self.precision = _get(training_cfg, "precision", "bf16")
        self.mu = _get(routing_cfg, "mu_load", 0.01)
        self.nu = _get(routing_cfg, "nu_sep", 0.05)

        self.amp_enabled = self.precision in ("bf16", "fp16")
        self.amp_dtype = _AMP_DTYPES.get(self.precision, torch.float32)
        self.use_scaler = self.precision == "fp16"
        self.scaler = torch.amp.GradScaler(self.device_type, enabled=self.use_scaler)

        self.checkpoint_dir = Path(checkpoint_dir)
        # Only rank 0 creates the checkpoint dir / writes files / logs, to
        # avoid every rank racing on the same paths and spamming stdout.
        if is_main_process():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.step = 0
        self.best_val_ppl = float("inf")

        self.use_wandb = (
            is_main_process() and bool(os.environ.get("WANDB_API_KEY")) and wandb is not None
        )
        if self.use_wandb:
            wandb.init(project=_get(wandb_cfg, "project", "diffusion-moe"))

    @property
    def raw_model(self) -> torch.nn.Module:
        """The underlying model, unwrapped from DistributedDataParallel if
        wrapped — checkpointing and anything that needs to reach model
        internals (e.g. centroids) should go through this, not self.model,
        since DDP prefixes state_dict keys with "module." and doesn't expose
        submodule attributes directly."""
        return self.model.module if self.is_ddp else self.model

    def _log(self, metrics: dict[str, float], step: int) -> None:
        if not is_main_process():
            return
        if self.use_wandb:
            wandb.log(metrics, step=step)
        else:
            formatted = " ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics.items()
            )
            print(f"step={step} {formatted}")

    def _forward_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        with torch.amp.autocast(
            device_type=self.device_type, dtype=self.amp_dtype, enabled=self.amp_enabled
        ):
            logits, _, router_outputs = self.model(input_ids)
            return total_loss(logits, labels, router_outputs, self.mu, self.nu)

    def train_step(self, micro_batches: list[dict[str, torch.Tensor]]) -> dict[str, float]:
        """One optimizer step, gradient-accumulated over `micro_batches`."""
        self.optimizer.zero_grad(set_to_none=True)
        n = len(micro_batches)
        totals = {"loss": 0.0, "task_loss": 0.0, "load_loss": 0.0, "sep_loss": 0.0}

        for micro_batch in micro_batches:
            losses = self._forward_loss(micro_batch)
            scaled_loss = losses["loss"] / n

            if self.use_scaler:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            totals["loss"] += scaled_loss.item()
            for key in ("task_loss", "load_loss", "sep_loss"):
                totals[key] += losses[key].item() / n

        if self.use_scaler:
            self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        if self.use_scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.scheduler.step()
        self.step += 1

        # DDP already all-reduces *gradients* during backward(); these are
        # locally-computed *logging* metrics, each rank having only seen its
        # own shard's micro-batches, so they're separately averaged here for
        # a globally-representative number (a no-op outside a distributed run).
        return {key: all_reduce_mean(value, self.device) for key, value in totals.items()}

    @torch.no_grad()
    def evaluate(self) -> float:
        """Returns validation perplexity (token-count-weighted across batches)."""
        was_training = self.model.training
        self.model.eval()

        total_nll, total_tokens = 0.0, 0
        for batch in self.val_loader:
            labels = batch["labels"].to(self.device)
            losses = self._forward_loss(batch)
            n_valid = int((labels != IGNORE_INDEX).sum().item())
            total_nll += losses["task_loss"].item() * n_valid
            total_tokens += n_valid

        if was_training:
            self.model.train()

        avg_nll = total_nll / max(1, total_tokens)
        return math.exp(min(avg_nll, 20.0))  # cap to avoid inf on a garbage model

    def save_checkpoint(self, path: str | Path) -> None:
        """Only rank 0 writes, to avoid every rank racing on the same file —
        a no-op elsewhere. Saves raw_model's state_dict (unwrapped from DDP,
        which would otherwise prefix every key with "module.")."""
        if not is_main_process():
            return
        state = {
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "best_val_ppl": self.best_val_ppl,
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.raw_model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.step = state["step"]
        self.best_val_ppl = state["best_val_ppl"]

    def train(self, max_steps: int) -> None:
        self.model.train()
        train_iter = iter(self.train_loader)

        while self.step < max_steps:
            micro_batches = []
            for _ in range(self.grad_accum_steps):
                try:
                    micro_batches.append(next(train_iter))
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    micro_batches.append(next(train_iter))

            metrics = self.train_step(micro_batches)

            if self.step % self.log_steps == 0:
                log_metrics = dict(metrics)
                log_metrics["lr"] = self.scheduler.get_last_lr()[0]
                self._log(log_metrics, self.step)

            if self.step % self.eval_steps == 0:
                val_ppl = self.evaluate()
                self._log({"val_ppl": val_ppl}, self.step)
                if val_ppl < self.best_val_ppl:
                    self.best_val_ppl = val_ppl
                    self.save_checkpoint(self.checkpoint_dir / "best.pt")

            if self.step % self.checkpoint_steps == 0:
                self.save_checkpoint(self.checkpoint_dir / f"step_{self.step}.pt")
