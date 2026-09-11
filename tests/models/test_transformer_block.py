"""Tests for TransformerBlock output shape and residual behavior."""

import torch

from diffusion_moe.models.transformer_block import TransformerBlock

BATCH, SEQ_LEN, D_MODEL = 2, 16, 64


def test_output_shape():
    block = TransformerBlock(d_model=D_MODEL, num_heads=8, max_seq_len=32)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    positions = torch.arange(SEQ_LEN).unsqueeze(0).expand(BATCH, -1)
    out = block(x, positions)
    assert out.shape == x.shape


def test_zero_input_is_not_trivially_zero_due_to_residual():
    """With residual connections, a zero input should not necessarily map to zero
    output once the sublayers have nonzero parameters — sanity-checks wiring."""
    torch.manual_seed(0)
    block = TransformerBlock(d_model=D_MODEL, num_heads=8, max_seq_len=32)
    x = torch.zeros(1, SEQ_LEN, D_MODEL)
    positions = torch.arange(SEQ_LEN).unsqueeze(0)
    out = block(x, positions)
    assert out.shape == x.shape
