"""Tests for the SwiGLU FeedForward module."""

import torch

from diffusion_moe.models.ffn import FeedForward

BATCH, SEQ_LEN, D_MODEL = 2, 16, 64


def test_output_shape_matches_input():
    ffn = FeedForward(d_model=D_MODEL)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out = ffn(x)
    assert out.shape == x.shape


def test_default_hidden_dim_is_4x_d_model():
    ffn = FeedForward(d_model=D_MODEL)
    assert ffn.gate_proj.out_features == 4 * D_MODEL
    assert ffn.up_proj.out_features == 4 * D_MODEL


def test_custom_hidden_dim():
    ffn = FeedForward(d_model=D_MODEL, hidden_dim=128)
    assert ffn.gate_proj.out_features == 128
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    assert ffn(x).shape == x.shape
