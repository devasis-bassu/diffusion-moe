"""Tests for scripts/sink_token_diagnostic.py's activation-collection wiring
(see test_extract_geometry.py for why a tiny locally-constructed
MistralForCausalLM + fake data stream stand in for the real model here).

Critically, the fake dataset here yields VARIABLE-length sequences, the way
a real streamed corpus does — collate_fn pads each batch only to its own
batch's longest sequence, so different batches end up with different
widths. An earlier fixed-length fake dataset masked exactly this: every
batch happened to collate to the same width, so the tests passed while the
real run crashed on `torch.cat` across batches of different widths.
"""

import importlib.util
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import MistralConfig, MistralForCausalLM

from diffusion_moe.data.dataset import collate_fn

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "sink_token_diagnostic_script", REPO_ROOT / "scripts" / "sink_token_diagnostic.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["sink_token_diagnostic_script"] = module
    spec.loader.exec_module(module)
    return module


std = _load_module()

VOCAB_SIZE, D_MODEL, N_LAYERS, MAX_SEQ_LEN, PAD_TOKEN_ID = 50, 16, 3, 12, 0


def _tiny_model(n_layers=N_LAYERS, hidden_size=D_MODEL):
    config = MistralConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    model = MistralForCausalLM(config)
    model.eval()
    return model


class _FakeTokenizedDataset(IterableDataset):
    """Yields VARIABLE-length sequences (min_len..max_seq_len), the way real
    streamed-and-tokenized documents do — every batch's collated width can
    differ from every other batch's.
    """

    def __init__(self, n_examples=20, min_len=4, max_len=MAX_SEQ_LEN, seed=0):
        g = torch.Generator().manual_seed(seed)
        lengths = torch.randint(min_len, max_len + 1, (n_examples,), generator=g)
        self.examples = [
            torch.randint(0, VOCAB_SIZE, (int(length),), generator=g).tolist()
            for length in lengths
        ]

    def __iter__(self):
        for ids in self.examples:
            yield {"input_ids": ids}


def _fake_loader(
    batch_size=4, n_examples=20, min_len=4, max_len=MAX_SEQ_LEN, pad_token_id=PAD_TOKEN_ID
):
    dataset = _FakeTokenizedDataset(n_examples=n_examples, min_len=min_len, max_len=max_len)
    collate = partial(collate_fn, pad_token_id=pad_token_id)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate)


def test_collect_norms_and_tokens_pads_every_batch_to_max_seq_len():
    model = _tiny_model()
    loader = _fake_loader(batch_size=4, n_examples=20)

    per_layer_norms, all_input_ids, all_masks = std.collect_norms_and_tokens(
        model, loader, device="cpu", layers=[0, 2],
        max_seq_len=MAX_SEQ_LEN, pad_token_id=PAD_TOKEN_ID,
    )

    assert set(per_layer_norms.keys()) == {0, 2}
    n_batches = len(all_input_ids)
    assert len(per_layer_norms[0]) == n_batches
    for batch_norms, batch_ids, batch_mask in zip(
        per_layer_norms[0], all_input_ids, all_masks
    ):
        assert batch_norms.shape == batch_ids.shape == batch_mask.shape == (4, MAX_SEQ_LEN)


def test_collect_norms_and_tokens_concatenates_across_variable_width_batches():
    """Regression test for the exact bug hit on the real model: batches of
    different collated widths must concatenate cleanly along dim 0, the way
    main() does, without a torch.cat/np.concatenate shape error.
    """
    model = _tiny_model()
    loader = _fake_loader(batch_size=3, n_examples=17, min_len=2, max_len=MAX_SEQ_LEN)

    per_layer_norms, all_input_ids, all_masks = std.collect_norms_and_tokens(
        model, loader, device="cpu", layers=[1],
        max_seq_len=MAX_SEQ_LEN, pad_token_id=PAD_TOKEN_ID,
    )

    input_ids_cat = torch.cat(all_input_ids, dim=0)
    mask_cat = torch.cat(all_masks, dim=0).numpy()
    norms_cat = np.concatenate(per_layer_norms[1], axis=0)

    assert input_ids_cat.shape[1] == MAX_SEQ_LEN
    assert mask_cat.shape == norms_cat.shape == (input_ids_cat.shape[0], MAX_SEQ_LEN)


def test_collect_norms_and_tokens_valid_positions_are_finite():
    model = _tiny_model()
    loader = _fake_loader(batch_size=4, n_examples=20)

    per_layer_norms, _, all_masks = std.collect_norms_and_tokens(
        model, loader, device="cpu", layers=[1],
        max_seq_len=MAX_SEQ_LEN, pad_token_id=PAD_TOKEN_ID,
    )

    for batch_norms, batch_mask in zip(per_layer_norms[1], all_masks):
        valid = batch_mask.numpy().astype(bool)
        assert np.isfinite(batch_norms[valid]).all()
        assert np.isnan(batch_norms[~valid]).all() or valid.all()
