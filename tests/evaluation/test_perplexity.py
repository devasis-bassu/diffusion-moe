"""Tests for compute_perplexity: perplexity and bits-per-byte."""

import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from diffusion_moe.data.dataset import IGNORE_INDEX
from diffusion_moe.evaluation.perplexity import compute_perplexity
from diffusion_moe.models.moe_model import DiffusionMoETransformer

VOCAB_SIZE, SEQ_LEN, D_MODEL = 30, 8, 16


class _TinyDataset(Dataset):
    def __init__(self, n=12, seed=0, with_bytes=False):
        g = torch.Generator().manual_seed(seed)
        self.ids = torch.randint(0, VOCAB_SIZE, (n, SEQ_LEN), generator=g)
        self.with_bytes = with_bytes

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):
        ids = self.ids[idx]
        labels = torch.cat([ids[1:], torch.tensor([IGNORE_INDEX])])
        item = {
            "input_ids": ids,
            "attention_mask": torch.ones(SEQ_LEN, dtype=torch.long),
            "labels": labels,
        }
        if self.with_bytes:
            item["num_bytes"] = torch.tensor(20)  # arbitrary fixed byte count
        return item


def _model():
    m = DiffusionMoETransformer(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=2, n_heads=4, max_seq_len=SEQ_LEN,
        n_experts=4, top_k=2, n_components=3, n_landmarks=8, diffusion_t=2,
        centroid_refresh_steps=1000, layers_to_replace=[],
    )
    m.eval()
    return m


def test_returns_finite_perplexity_and_nan_bpb_without_num_bytes():
    model = _model()
    loader = DataLoader(_TinyDataset(with_bytes=False), batch_size=4)

    result = compute_perplexity(model, loader)

    assert set(result.keys()) == {"perplexity", "bits_per_byte"}
    assert result["perplexity"] > 0
    assert result["perplexity"] == result["perplexity"]  # not NaN
    assert math.isnan(result["bits_per_byte"])


def test_bits_per_byte_finite_with_num_bytes():
    model = _model()
    loader = DataLoader(_TinyDataset(with_bytes=True), batch_size=4)

    result = compute_perplexity(model, loader)

    assert not math.isnan(result["bits_per_byte"])
    assert result["bits_per_byte"] > 0


def test_matches_manual_perplexity_computation():
    model = _model()
    loader = DataLoader(_TinyDataset(n=8, with_bytes=False), batch_size=8)

    result = compute_perplexity(model, loader)

    with torch.no_grad():
        batch = next(iter(loader))
        logits, _, _ = model(batch["input_ids"])
        manual_nll = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), batch["labels"].reshape(-1), ignore_index=IGNORE_INDEX
        )
    assert math.isclose(result["perplexity"], math.exp(manual_nll.item()), rel_tol=1e-4)


def test_restores_training_mode():
    model = _model()
    loader = DataLoader(_TinyDataset(), batch_size=4)

    model.train()
    compute_perplexity(model, loader)
    assert model.training

    model.eval()
    compute_perplexity(model, loader)
    assert not model.training


def test_works_with_two_tuple_model_return():
    """compute_perplexity should also accept a model whose forward returns
    just (logits, activations), like BaseTransformer."""

    class TwoTupleWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids):
            logits, activations, _ = self.inner(input_ids)
            return logits, activations

    model = TwoTupleWrapper(_model())
    loader = DataLoader(_TinyDataset(), batch_size=4)
    result = compute_perplexity(model, loader)
    assert result["perplexity"] > 0
