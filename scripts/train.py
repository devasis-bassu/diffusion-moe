"""Training entrypoint: builds model/dataloaders/optimizer/scheduler and runs
Trainer.train().

Local runs use a single device (CUDA > MPS > CPU, auto-detected by Trainer).
Remote runs on vast.ai use multiple CUDA GPUs via torch.distributed +
DistributedDataParallel, launched with torchrun:

    python scripts/train.py                                    # local: single device
    python scripts/train.py --config-name=1.3b                  # 1.3B model, still local
    python scripts/train.py resume_from_checkpoint=checkpoints/best.pt
    torchrun --nproc_per_node=NUM_GPUS scripts/train.py          # vast.ai: multi-GPU DDP
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from diffusion_moe.data.dataloader import build_dataloaders
from diffusion_moe.models.model_factory import build_model_from_config
from diffusion_moe.training.optimizer import build_optimizer, build_scheduler
from diffusion_moe.training.trainer import Trainer
from diffusion_moe.utils.device import cleanup_distributed, is_main_process, setup_distributed
from diffusion_moe.utils.env import load_env


def compute_total_steps(cfg: DictConfig) -> int:
    tokens_per_step = cfg.data.batch_size * cfg.data.max_seq_len * cfg.training.grad_accum_steps
    return max(1, cfg.training.total_tokens // tokens_per_step)


def run(cfg: DictConfig) -> Trainer:
    """Builds model/dataloaders/optimizer/scheduler/Trainer, optionally resumes
    from a checkpoint, then trains. Returns the Trainer (its final `step` and
    `best_val_ppl` are useful to callers/tests without re-parsing stdout).

    Under torchrun (WORLD_SIZE>1), Trainer detects the distributed run and
    wraps the model in DistributedDataParallel itself; setup_distributed()
    here just makes sure the process group exists first.
    """
    setup_distributed()
    try:
        model = build_model_from_config(cfg)
        train_loader, val_loader = build_dataloaders(cfg)
        optimizer = build_optimizer(
            model, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
        )

        total_steps = compute_total_steps(cfg)
        scheduler = build_scheduler(
            optimizer, warmup_steps=cfg.training.warmup_steps, total_steps=total_steps
        )

        trainer = Trainer(model, optimizer, scheduler, train_loader, val_loader, cfg)

        resume_path = cfg.get("resume_from_checkpoint")
        if resume_path:
            trainer.load_checkpoint(resume_path)

        trainer.train(max_steps=total_steps)
        return trainer
    finally:
        cleanup_distributed()


@hydra.main(version_base=None, config_path="../configs", config_name="base_config")
def main(cfg: DictConfig) -> None:
    load_env()
    if is_main_process():
        print(OmegaConf.to_yaml(cfg))
    run(cfg)


if __name__ == "__main__":
    main()
