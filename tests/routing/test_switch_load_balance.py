"""Tests for switch_load_balance_loss (the standard Switch Transformer
auxiliary load-balance loss)."""

import torch

from diffusion_moe.routing.load_balance import switch_load_balance_loss

N_EXPERTS = 4


def test_uniform_routing_gives_minimum_loss_of_one():
    n = 100
    router_probs = torch.full((n, N_EXPERTS), 1.0 / N_EXPERTS)
    # dispatch each expert to an equal share of tokens
    expert_indices = torch.arange(n).remainder(N_EXPERTS).unsqueeze(-1)

    loss = switch_load_balance_loss(router_probs, expert_indices, N_EXPERTS)
    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-5)


def test_fully_collapsed_routing_gives_maximum_loss_of_n_experts():
    n = 100
    router_probs = torch.zeros(n, N_EXPERTS)
    router_probs[:, 0] = 1.0  # every token's softmax is peaked on expert 0
    expert_indices = torch.zeros(n, 1, dtype=torch.long)  # every token dispatched to expert 0

    loss = switch_load_balance_loss(router_probs, expert_indices, N_EXPERTS)
    assert torch.isclose(loss, torch.tensor(float(N_EXPERTS)), atol=1e-5)


def test_partial_imbalance_between_extremes():
    n = 100
    uniform_probs = torch.full((n, N_EXPERTS), 1.0 / N_EXPERTS)
    uniform_indices = torch.arange(n).remainder(N_EXPERTS).unsqueeze(-1)
    uniform_loss = switch_load_balance_loss(uniform_probs, uniform_indices, N_EXPERTS)

    skewed_probs = uniform_probs.clone()
    skewed_probs[:, 0] += 0.3
    skewed_probs[:, 1] -= 0.3
    skewed_indices = torch.zeros(n, 1, dtype=torch.long)
    skewed_loss = switch_load_balance_loss(skewed_probs, skewed_indices, N_EXPERTS)

    assert uniform_loss < skewed_loss


def test_handles_top_k_greater_than_one():
    n = 50
    router_probs = torch.full((n, N_EXPERTS), 1.0 / N_EXPERTS)
    expert_indices = torch.randint(0, N_EXPERTS, (n, 2))  # top_k=2
    loss = switch_load_balance_loss(router_probs, expert_indices, N_EXPERTS)
    assert loss.item() > 0


def test_gradients_flow_through_router_probs():
    router_probs = torch.full((20, N_EXPERTS), 1.0 / N_EXPERTS, requires_grad=True)
    expert_indices = torch.randint(0, N_EXPERTS, (20, 1))
    loss = switch_load_balance_loss(router_probs, expert_indices, N_EXPERTS)
    loss.backward()
    assert router_probs.grad is not None
