"""Tests for build_dataloaders — mocks TokenizerWrapper and StreamingTextDataset
so no network access or real tokenizer download is needed."""

import torch
from torch.utils.data import IterableDataset

from diffusion_moe.data import dataloader as dataloader_module
from diffusion_moe.data.dataloader import build_dataloaders

FIXED_EXAMPLES = [
    {"input_ids": [1, 2, 3]},
    {"input_ids": [4, 5, 6, 7]},
    {"input_ids": [8, 9]},
    {"input_ids": [10, 11, 12]},
]


class FakeTokenizerWrapper:
    def __init__(self, pretrained_name: str) -> None:
        self.pretrained_name = pretrained_name
        self.pad_token_id = 0


class FakeStreamingTextDataset(IterableDataset):
    def __init__(
        self,
        dataset_name,
        tokenizer,
        max_seq_len,
        split="train",
        skip=0,
        take=None,
        seed=42,
        num_shards=1,
        shard_index=0,
    ) -> None:
        self.num_shards = num_shards
        self.shard_index = shard_index
        examples = FIXED_EXAMPLES[skip:]
        if take is not None:
            examples = examples[:take]
        self.examples = examples

    def __iter__(self):
        return iter(self.examples)


def _patch(monkeypatch):
    monkeypatch.setattr(dataloader_module, "TokenizerWrapper", FakeTokenizerWrapper)
    monkeypatch.setattr(dataloader_module, "StreamingTextDataset", FakeStreamingTextDataset)


def _make_config(**overrides):
    cfg = {
        "data": {
            "dataset": "the_pile",
            "tokenizer": "fake/model",
            "max_seq_len": 16,
            "batch_size": 2,
            "num_workers": 0,
            "val_tokens": 16,  # -> val_n_examples = 1
        }
    }
    cfg["data"].update(overrides)
    return cfg


def test_build_dataloaders_returns_two_loaders(monkeypatch):
    _patch(monkeypatch)
    train_loader, val_loader = build_dataloaders(_make_config())
    assert train_loader is not None
    assert val_loader is not None


def test_train_batch_shapes_and_dtypes(monkeypatch):
    _patch(monkeypatch)
    train_loader, _ = build_dataloaders(_make_config())
    batch = next(iter(train_loader))

    assert set(batch.keys()) == {"input_ids", "attention_mask", "labels", "num_bytes"}
    assert batch["input_ids"].shape[0] == 2  # batch_size
    for key in ("input_ids", "attention_mask", "labels"):
        assert batch[key].dtype == torch.long


def test_val_loader_uses_held_out_examples(monkeypatch):
    _patch(monkeypatch)
    # val_tokens=16, max_seq_len=16 -> val_n_examples = 1 -> val gets FIXED_EXAMPLES[:1]
    # train gets FIXED_EXAMPLES[1:] (skip=1)
    train_loader, val_loader = build_dataloaders(_make_config())

    val_batch = next(iter(val_loader))
    assert val_batch["input_ids"].shape[0] == 1

    train_examples = list(train_loader.dataset)
    assert train_examples == FIXED_EXAMPLES[1:]


def test_num_workers_gt_zero_sets_prefetch_factor(monkeypatch):
    _patch(monkeypatch)
    train_loader, _ = build_dataloaders(_make_config(num_workers=2, prefetch_factor=3))
    assert train_loader.num_workers == 2
    assert train_loader.prefetch_factor == 3


def test_train_dataset_sharded_by_ddp_rank_and_world_size(monkeypatch):
    _patch(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")

    train_loader, val_loader = build_dataloaders(_make_config())

    assert train_loader.dataset.num_shards == 4
    assert train_loader.dataset.shard_index == 2
    # the val set is deliberately left unsharded across ranks
    assert val_loader.dataset.num_shards == 1
    assert val_loader.dataset.shard_index == 0


def test_train_dataset_unsharded_outside_distributed_run(monkeypatch):
    _patch(monkeypatch)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)

    train_loader, _ = build_dataloaders(_make_config())
    assert train_loader.dataset.num_shards == 1
    assert train_loader.dataset.shard_index == 0
