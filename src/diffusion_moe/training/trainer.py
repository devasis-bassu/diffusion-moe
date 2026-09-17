"""Training loop: gradient accumulation, mixed precision, checkpointing, logging."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Iterable

import torch
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel

from diffusion_moe.data.dataset import IGNORE_INDEX
from diffusion_moe.routing.separation import landmark_scale
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
        # Numbered checkpoints (step_N.pt) accumulate forever otherwise --
        # a real disk-full crash on an actual run (8 checkpoints x ~5.7GB
        # each on a 60GB disk) killed training right at its second-to-last
        # step. best.pt is exempt: it's not necessarily the most recent
        # numbered one, and is small in count regardless (only overwritten
        # when val_ppl actually improves).
        self.keep_last_n_checkpoints = _get(training_cfg, "keep_last_n_checkpoints", 2)
        self.eval_steps = _get(training_cfg, "eval_steps", 1000)
        self.log_steps = _get(training_cfg, "log_steps", 50)
        self.precision = _get(training_cfg, "precision", "bf16")
        self.mu = _get(routing_cfg, "mu_load", 0.01)
        self.nu = _get(routing_cfg, "nu_sep", 0.05)
        # centroid_separation_loss's gradient has no upper bound on how far it
        # pushes expert centroids apart (by design -- see test_separation.py);
        # a real pilot run showed this diverge ~4 orders of magnitude over 300
        # steps once nothing was capping it, collapsing the router toward
        # uniform dispatch (see reports/phase1_findings_report.md §3.4, bug 1).
        # The pilot script itself applies ExpertCentroids.clip_norm_() after
        # every optimizer step as the fix -- this mirrors that here, in the
        # actual production training loop, since the pilot's own bespoke loop
        # was the only caller and this Trainer never inherited the safeguard.
        self.centroid_max_radius_factor = _get(routing_cfg, "centroid_max_radius_factor", 3.0)

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
            # Previously omitted entirely: wandb.init() was never given
            # `config=`, so no run's actual hyperparameters (routing.top_k,
            # model.router, use_shared_expert, ...) were ever recorded in
            # wandb -- confirming what a given run actually used required
            # SSHing into the instance and reading its (often still-buffered,
            # see Trainer's own stdout-buffering notes elsewhere) log file
            # by hand. OmegaConf.to_container flattens a real Hydra
            # DictConfig into a plain, JSON-serializable dict; falls back to
            # the config object as-is for the plain-dict configs the test
            # suite uses, which OmegaConf.is_config correctly rejects.
            wandb_config = (
                OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config
            )
            wandb.init(project=_get(wandb_cfg, "project", "diffusion-moe"), config=wandb_config)

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

    def _clip_centroid_norms(self) -> None:
        """Caps each DiffusionMoELayer's expert centroids at a multiple of
        that layer's own current landmark_scale, after every optimizer step
        -- see the note on centroid_max_radius_factor in __init__. Only
        DiffusionMoELayer blocks have both `centroids` and `ndm` (the cosine/
        switch/random router variants and plain dense TransformerBlocks
        don't), and ndm.psi_landmarks_ is None until the first forward pass
        has fit it, which has always already happened by the time this is
        called (train_step always does a forward pass before optimizer.step()).
        """
        for block in self.raw_model.blocks:
            centroids = getattr(block, "centroids", None)
            ndm = getattr(block, "ndm", None)
            if centroids is None or ndm is None or ndm.psi_landmarks_ is None:
                continue
            psi_landmarks = torch.from_numpy(ndm.psi_landmarks_).to(
                dtype=centroids.centroids.dtype, device=centroids.centroids.device
            )
            scale = landmark_scale(psi_landmarks)
            centroids.clip_norm_(max_norm=self.centroid_max_radius_factor * scale)

    def train_step(self, micro_batches: list[dict[str, torch.Tensor]]) -> dict[str, float]:
        """One optimizer step, gradient-accumulated over `micro_batches`."""
        self.optimizer.zero_grad(set_to_none=True)
        n = len(micro_batches)
        totals: dict[str, float] = {"loss": 0.0}

        for micro_batch in micro_batches:
            losses = self._forward_loss(micro_batch)

            if not torch.isfinite(losses["loss"]):
                # clip_grad_norm_ can't repair this: the norm of a NaN/Inf
                # gradient is itself NaN/Inf, so the very next optimizer.step()
                # would silently set every parameter to NaN -- permanently,
                # since nothing downstream can recover from that. Abort loudly
                # here, before backward(), instead of training on (and
                # checkpointing) poisoned weights for however many steps until
                # something else happens to crash outright.
                detail = ", ".join(
                    f"{k}={v.item():.6g}" for k, v in losses.items() if k != "loss"
                )
                raise FloatingPointError(
                    f"Non-finite loss at step {self.step} "
                    f"(loss={losses['loss'].item()}): {detail}"
                )

            scaled_loss = losses["loss"] / n

            if self.use_scaler:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            totals["loss"] += scaled_loss.item()
            # Everything else (task_loss/load_loss/sep_loss, plus any
            # per-layer load_loss/layer_N, sep_loss/layer_N keys total_loss
            # adds when more than one MoE layer is active) is already
            # detached -- accumulate whatever keys are actually present
            # rather than a fixed tuple, so this doesn't need to know in
            # advance how many MoE layers the model has.
            for key, value in losses.items():
                if key == "loss":
                    continue
                totals[key] = totals.get(key, 0.0) + value.item() / n

        if self.use_scaler:
            self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        if self.use_scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self._clip_centroid_norms()
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

    def _prune_old_checkpoints(self) -> None:
        """Deletes numbered checkpoints (step_N.pt) beyond the most recent
        keep_last_n_checkpoints, oldest first. best.pt is untouched -- it's
        a separate file, not part of this numbered sequence. A no-op if
        keep_last_n_checkpoints <= 0 (unlimited retention, opt-in)."""
        if not is_main_process() or self.keep_last_n_checkpoints <= 0:
            return
        numbered = sorted(
            self.checkpoint_dir.glob("step_*.pt"),
            key=lambda p: int(p.stem.removeprefix("step_")),
        )
        for path in numbered[: -self.keep_last_n_checkpoints]:
            path.unlink(missing_ok=True)

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

        # Resuming from a checkpoint restores self.step, but a freshly built
        # train_loader always starts its (deterministically shuffled, fixed
        # seed) stream from the beginning -- without this, "resume" would
        # silently re-serve the same batches the original run already
        # trained on, rather than continuing into unseen data. Burn through
        # exactly the micro-batches already consumed before resuming real
        # training. Assumes grad_accum_steps (and the rest of the data
        # config) matches what produced the checkpoint -- same requirement
        # base_config.yaml already documents for layers_to_replace on
        # resume, not a new constraint this introduces.
        for _ in range(self.step * self.grad_accum_steps):
            try:
                next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                next(train_iter)

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
                self._prune_old_checkpoints()
