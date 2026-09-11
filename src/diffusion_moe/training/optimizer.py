"""Optimizer and learning-rate scheduler construction."""

from __future__ import annotations

import math

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> AdamW:
    """AdamW with weight decay excluded from 1D parameters — biases and norm
    (RMSNorm/LayerNorm) weights are all 1D, while every real weight matrix
    (Linear, Embedding) is 2D+, so this single shape-based rule captures
    exactly the intended exclusion without needing to inspect module types.
    """
    decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return AdamW(param_groups, lr=lr)


def build_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> LambdaLR:
    """Linear warmup over `warmup_steps`, then cosine decay to 0 by `total_steps`."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
