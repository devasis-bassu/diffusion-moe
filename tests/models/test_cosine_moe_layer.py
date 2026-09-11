"""Tests for CosineMoELayer: cosine-similarity routing in d_model space."""

import torch
import torch.nn.functional as F

from diffusion_moe.models.cosine_moe_layer import CosineMoELayer

BATCH, SEQ_LEN, D_MODEL, N_EXPERTS, TOP_K = 2, 16, 64, 4, 2


def _layer(**overrides):
    kwargs = dict(
        d_model=D_MODEL, num_heads=8, max_seq_len=32, n_experts=N_EXPERTS, top_k=TOP_K, tau=0.5
    )
    kwargs.update(overrides)
    return CosineMoELayer(**kwargs)


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


def test_centroids_live_in_d_model_space():
    layer = _layer()
    assert layer.centroids.shape == (N_EXPERTS, D_MODEL)


def test_router_logits_are_bounded_cosine_similarities():
    layer = _layer()
    x, positions = _inputs()
    _, aux = layer(x, positions)
    assert torch.all(aux["router_logits"] >= -1.0 - 1e-5)
    assert torch.all(aux["router_logits"] <= 1.0 + 1e-5)


def test_router_logits_match_manual_cosine_similarity():
    layer = _layer()
    x, positions = _inputs()
    z = layer.attn(layer.attn_norm(x), positions)
    _, aux = layer(x, positions)

    z_normed = F.normalize(z, dim=-1)
    centroids_normed = F.normalize(layer.centroids, dim=-1)
    manual = z_normed @ centroids_normed.t()
    assert torch.allclose(aux["router_logits"], manual, atol=1e-5)


def test_tau_to_zero_hardens_to_nearest_centroid():
    torch.manual_seed(0)
    layer = _layer(top_k=1, tau=1e-6)
    x, positions = _inputs()
    _, aux = layer(x, positions)
    assert torch.allclose(aux["gate_values"].squeeze(-1), torch.ones(BATCH, SEQ_LEN), atol=1e-4)


def test_gradients_flow_to_centroids_attention_and_experts():
    layer = _layer()
    x, positions = _inputs()
    out, _ = layer(x, positions)
    out.sum().backward()
    assert layer.centroids.grad is not None
    assert layer.attn.q_proj.weight.grad is not None
    assert any(e.gate_proj.weight.grad is not None for e in layer.experts)
