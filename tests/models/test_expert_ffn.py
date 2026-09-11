"""Tests for ExpertFFN: sizing formula and output shape."""

import torch

from diffusion_moe.models.expert_ffn import ExpertFFN

BATCH, SEQ_LEN, D_MODEL = 2, 16, 64


def test_hidden_dim_formula():
    ffn = ExpertFFN(d_model=D_MODEL, ffn_dim=256, n_experts=4, overlap_factor=1.0)
    assert ffn.hidden_dim == 256 // 4  # 64


def test_hidden_dim_scales_with_overlap_factor():
    ffn = ExpertFFN(d_model=D_MODEL, ffn_dim=256, n_experts=4, overlap_factor=2.0)
    assert ffn.hidden_dim == (256 // 4) * 2


def test_output_shape_matches_input():
    ffn = ExpertFFN(d_model=D_MODEL, ffn_dim=256, n_experts=4)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out = ffn(x)
    assert out.shape == x.shape


def test_hidden_dim_at_least_one():
    ffn = ExpertFFN(d_model=D_MODEL, ffn_dim=4, n_experts=16, overlap_factor=0.1)
    assert ffn.hidden_dim >= 1
