"""Tests for coefficient_of_variation_loss."""

import pytest
import torch

from diffusion_moe.routing.load_balance import coefficient_of_variation_loss

N_EXPERTS = 4


def test_perfectly_balanced_load_gives_zero_cv():
    gate_values = torch.full((100, N_EXPERTS), 1.0 / N_EXPERTS)
    loss = coefficient_of_variation_loss(gate_values, N_EXPERTS)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_collapsed_load_gives_larger_cv_than_balanced():
    balanced = torch.full((100, N_EXPERTS), 1.0 / N_EXPERTS)
    collapsed = torch.zeros(100, N_EXPERTS)
    collapsed[:, 0] = 1.0  # all mass on expert 0

    loss_balanced = coefficient_of_variation_loss(balanced, N_EXPERTS)
    loss_collapsed = coefficient_of_variation_loss(collapsed, N_EXPERTS)
    assert loss_collapsed > loss_balanced


def test_raises_on_shape_mismatch():
    gate_values = torch.rand(10, 3)  # last dim != n_experts
    with pytest.raises(ValueError):
        coefficient_of_variation_loss(gate_values, N_EXPERTS)


def test_handles_leading_batch_and_seq_dims():
    gate_values = torch.full((2, 16, N_EXPERTS), 1.0 / N_EXPERTS)
    loss = coefficient_of_variation_loss(gate_values, N_EXPERTS)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_partial_imbalance_between_extremes():
    balanced = torch.full((100, N_EXPERTS), 1.0 / N_EXPERTS)
    collapsed = torch.zeros(100, N_EXPERTS)
    collapsed[:, 0] = 1.0
    slightly_off = torch.full((100, N_EXPERTS), 1.0 / N_EXPERTS)
    slightly_off[:, 0] += 0.1
    slightly_off[:, 1] -= 0.1

    loss_balanced = coefficient_of_variation_loss(balanced, N_EXPERTS)
    loss_slightly_off = coefficient_of_variation_loss(slightly_off, N_EXPERTS)
    loss_collapsed = coefficient_of_variation_loss(collapsed, N_EXPERTS)
    assert loss_balanced < loss_slightly_off < loss_collapsed
