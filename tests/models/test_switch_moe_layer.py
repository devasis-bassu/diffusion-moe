"""Tests for SwitchMoELayer: standard learned linear router + Switch aux loss."""

import torch
import torch.nn.functional as F

from diffusion_moe.models.switch_moe_layer import SwitchMoELayer
from diffusion_moe.routing.load_balance import switch_load_balance_loss

BATCH, SEQ_LEN, D_MODEL, N_EXPERTS, TOP_K = 2, 16, 64, 4, 2


def _layer():
    return SwitchMoELayer(
        d_model=D_MODEL, num_heads=8, max_seq_len=32, n_experts=N_EXPERTS, top_k=TOP_K
    )


def _inputs():
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    positions = torch.arange(SEQ_LEN).unsqueeze(0).expand(BATCH, -1)
    return x, positions


def test_output_and_aux_shapes():
    layer = _layer()
    x, positions = _inputs()
    out, aux = layer(x, positions)

    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)
    assert aux["router_logits"].shape == (BATCH, SEQ_LEN, N_EXPERTS)
    assert aux["gate_values"].shape == (BATCH, SEQ_LEN, TOP_K)
    assert aux["expert_indices"].shape == (BATCH, SEQ_LEN, TOP_K)
    assert "switch_aux_loss" in aux
    assert aux["switch_aux_loss"].dim() == 0  # scalar


def test_router_is_a_single_linear_layer():
    layer = _layer()
    assert isinstance(layer.router, torch.nn.Linear)
    assert layer.router.weight.shape == (N_EXPERTS, D_MODEL)


def test_router_logits_match_manual_linear_projection():
    layer = _layer()
    x, positions = _inputs()
    z = layer.attn(layer.attn_norm(x), positions)
    _, aux = layer(x, positions)
    manual = layer.router(z)
    assert torch.allclose(aux["router_logits"], manual, atol=1e-5)


def test_switch_aux_loss_matches_standalone_function():
    layer = _layer()
    x, positions = _inputs()
    _, aux = layer(x, positions)

    router_probs = F.softmax(aux["router_logits"], dim=-1)
    manual = switch_load_balance_loss(router_probs, aux["expert_indices"], N_EXPERTS)
    assert torch.isclose(aux["switch_aux_loss"], manual)


def test_gradients_flow_to_router_attention_and_experts():
    layer = _layer()
    x, positions = _inputs()
    out, aux = layer(x, positions)
    (out.sum() + aux["switch_aux_loss"]).backward()
    assert layer.router.weight.grad is not None
    assert layer.attn.q_proj.weight.grad is not None
    assert any(e.gate_proj.weight.grad is not None for e in layer.experts)
