"""Tests for build_optimizer and build_scheduler."""

import torch
from torch import nn

from diffusion_moe.training.optimizer import build_optimizer, build_scheduler


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8, bias=True)
        self.norm = nn.LayerNorm(8)

    def forward(self, x):
        return self.norm(self.linear(x))


def test_weight_matrices_get_weight_decay_biases_and_norms_dont():
    model = _TinyModel()
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.1)

    assert len(optimizer.param_groups) == 2
    decay_group, no_decay_group = optimizer.param_groups
    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0

    decay_param_ids = {id(p) for p in decay_group["params"]}
    no_decay_param_ids = {id(p) for p in no_decay_group["params"]}

    assert id(model.linear.weight) in decay_param_ids
    assert id(model.linear.bias) in no_decay_param_ids
    assert id(model.norm.weight) in no_decay_param_ids
    assert id(model.norm.bias) in no_decay_param_ids


def test_optimizer_covers_all_trainable_params_exactly_once():
    model = _TinyModel()
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.1)

    all_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    expected_ids = [id(p) for p in model.parameters() if p.requires_grad]

    assert sorted(all_ids) == sorted(expected_ids)
    assert len(all_ids) == len(set(all_ids))  # no duplicates


def test_optimizer_is_adamw_with_correct_lr():
    model = _TinyModel()
    optimizer = build_optimizer(model, lr=5e-4, weight_decay=0.1)
    assert isinstance(optimizer, torch.optim.AdamW)
    for group in optimizer.param_groups:
        assert group["lr"] == 5e-4


def test_scheduler_linear_warmup_then_cosine_decay():
    model = _TinyModel()
    optimizer = build_optimizer(model, lr=1.0, weight_decay=0.0)
    scheduler = build_scheduler(optimizer, warmup_steps=10, total_steps=100)

    lrs = []
    for step in range(101):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    # warmup: strictly increasing up to step 10
    assert lrs[0] < lrs[5] < lrs[9]
    # peak near the end of warmup
    assert max(lrs[:15]) == lrs[10] or abs(lrs[10] - 1.0) < 1e-6
    # cosine decay: near zero by the end
    assert lrs[-1] < 0.01
    # decreasing overall after warmup
    assert lrs[50] > lrs[90]


def test_scheduler_handles_zero_warmup_steps():
    model = _TinyModel()
    optimizer = build_optimizer(model, lr=1.0, weight_decay=0.0)
    scheduler = build_scheduler(optimizer, warmup_steps=0, total_steps=10)
    # should not raise, and lr at step 0 should be at or near the peak (cos(0)=1)
    assert optimizer.param_groups[0]["lr"] > 0.99
    for _ in range(10):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] < 0.01
