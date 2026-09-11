"""Tests for BaseTransformer: logits shape, activation dict, weight tying."""

import torch

from diffusion_moe.models.base_model import BaseTransformer

BATCH, SEQ_LEN, D_MODEL, N_LAYERS, VOCAB_SIZE = 2, 16, 64, 3, 1000


def _model(**overrides):
    kwargs = dict(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=8,
        max_seq_len=32,
    )
    kwargs.update(overrides)
    return BaseTransformer(**kwargs)


def test_logits_shape():
    model = _model()
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    logits, _ = model(input_ids)
    assert logits.shape == (BATCH, SEQ_LEN, VOCAB_SIZE)


def test_activations_keyed_by_layer_index():
    model = _model()
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    _, activations = model(input_ids)

    assert set(activations.keys()) == set(range(N_LAYERS))
    for layer_idx, act in activations.items():
        assert act.shape == (BATCH, SEQ_LEN, D_MODEL)


def test_default_positions_are_arange():
    model = _model()
    input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
    logits_default, _ = model(input_ids)

    positions = torch.arange(SEQ_LEN).unsqueeze(0)
    logits_explicit, _ = model(input_ids, positions=positions)

    assert torch.allclose(logits_default, logits_explicit)


def test_weight_tying_enabled_by_default():
    model = _model()
    assert model.lm_head.weight is model.token_embedding.weight


def test_weight_tying_can_be_disabled():
    model = _model(tie_embeddings=False)
    assert model.lm_head.weight is not model.token_embedding.weight
