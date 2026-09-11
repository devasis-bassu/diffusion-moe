"""Tests for RotaryEmbedding: cache shapes, theta formula, norm-preservation."""

import torch

from diffusion_moe.models.rope import RotaryEmbedding


def test_theta_formula():
    head_dim = 8
    rotary = RotaryEmbedding(head_dim=head_dim, max_seq_len=4, base=10000)
    j = torch.arange(0, head_dim, 2, dtype=torch.float32)
    expected = 10000 ** (-j / head_dim)
    assert torch.allclose(rotary.inv_freq, expected)


def test_cache_shapes():
    rotary = RotaryEmbedding(head_dim=16, max_seq_len=32)
    assert rotary.cos_cached.shape == (32, 16)
    assert rotary.sin_cached.shape == (32, 16)


def test_cache_grows_for_out_of_range_positions():
    rotary = RotaryEmbedding(head_dim=8, max_seq_len=4)
    positions = torch.arange(10).unsqueeze(0)
    q = torch.randn(1, 2, 10, 8)
    k = torch.randn(1, 2, 10, 8)
    q_rot, k_rot = rotary.apply_rope(q, k, positions)
    assert q_rot.shape == q.shape
    assert rotary.cos_cached.shape[0] >= 10


def test_apply_rope_preserves_shape_and_norm():
    batch, num_heads, seq_len, head_dim = 2, 4, 16, 8
    rotary = RotaryEmbedding(head_dim=head_dim, max_seq_len=32)
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
    q = torch.randn(batch, num_heads, seq_len, head_dim)
    k = torch.randn(batch, num_heads, seq_len, head_dim)

    q_rot, k_rot = rotary.apply_rope(q, k, positions)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    # Rotation is an orthogonal (norm-preserving) transform per (2i, 2i+1) pair.
    assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-4)
    assert torch.allclose(k_rot.norm(dim=-1), k.norm(dim=-1), atol=1e-4)


def test_position_zero_is_identity():
    head_dim = 8
    rotary = RotaryEmbedding(head_dim=head_dim, max_seq_len=4)
    positions = torch.zeros(1, 1, dtype=torch.long)
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)
    q_rot, k_rot = rotary.apply_rope(q, k, positions)
    # theta * position=0 -> cos=1, sin=0 -> rotation is the identity
    assert torch.allclose(q_rot, q, atol=1e-6)
    assert torch.allclose(k_rot, k, atol=1e-6)


def test_rejects_odd_head_dim():
    import pytest

    with pytest.raises(ValueError):
        RotaryEmbedding(head_dim=7, max_seq_len=4)
