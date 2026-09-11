"""Tests for RoPEMultiHeadAttention output shapes and masking."""

import torch

from diffusion_moe.models.attention import RoPEMultiHeadAttention

BATCH, SEQ_LEN, D_MODEL = 2, 16, 64


def _inputs():
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    positions = torch.arange(SEQ_LEN).unsqueeze(0).expand(BATCH, -1)
    return x, positions


def test_output_shape_default_causal():
    attn = RoPEMultiHeadAttention(d_model=D_MODEL, num_heads=8, max_seq_len=32)
    x, positions = _inputs()
    out = attn(x, positions)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)


def test_output_shape_with_explicit_mask():
    attn = RoPEMultiHeadAttention(d_model=D_MODEL, num_heads=8, max_seq_len=32)
    x, positions = _inputs()
    causal = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool))
    mask = causal.unsqueeze(0).unsqueeze(0).expand(BATCH, 1, SEQ_LEN, SEQ_LEN)
    out = attn(x, positions, mask=mask)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)


def test_custom_head_dim():
    attn = RoPEMultiHeadAttention(d_model=D_MODEL, num_heads=4, head_dim=32, max_seq_len=32)
    x, positions = _inputs()
    out = attn(x, positions)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)


def test_causal_mask_blocks_future_tokens():
    """Changing a future token must not change an earlier position's output
    under the default causal masking."""
    torch.manual_seed(0)
    attn = RoPEMultiHeadAttention(d_model=D_MODEL, num_heads=8, max_seq_len=32)
    attn.eval()
    x, positions = _inputs()

    out_a = attn(x, positions)

    x_perturbed = x.clone()
    x_perturbed[:, -1, :] += 10.0  # perturb only the last token
    out_b = attn(x_perturbed, positions)

    assert torch.allclose(out_a[:, :-1, :], out_b[:, :-1, :], atol=1e-5)
    assert not torch.allclose(out_a[:, -1, :], out_b[:, -1, :])
