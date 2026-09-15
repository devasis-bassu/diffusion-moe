"""Tests for StreamingTextDataset and collate_fn — mocks datasets.load_dataset."""

import pytest
import torch

from diffusion_moe.data import dataset as dataset_module
from diffusion_moe.data.dataset import IGNORE_INDEX, StreamingTextDataset, collate_fn


class FakeHFStream:
    """Stand-in for a HuggingFace streaming IterableDataset."""

    def __init__(self, examples: list[dict], shard_calls: list | None = None) -> None:
        self.examples = list(examples)
        self._shard_calls = shard_calls if shard_calls is not None else []

    def shuffle(self, seed=None, buffer_size=None):
        return self

    def skip(self, n: int):
        return FakeHFStream(self.examples[n:], self._shard_calls)

    def take(self, n: int):
        return FakeHFStream(self.examples[:n], self._shard_calls)

    def shard(self, num_shards: int, index: int):
        self._shard_calls.append((num_shards, index))
        # simple contiguous split, good enough to test wiring
        per_shard = max(1, len(self.examples) // num_shards)
        start = index * per_shard
        end = start + per_shard if index < num_shards - 1 else len(self.examples)
        return FakeHFStream(self.examples[start:end], self._shard_calls)

    def __iter__(self):
        return iter(self.examples)


class FakeTokenizer:
    """Deterministic stand-in for TokenizerWrapper.encode/decode."""

    def encode(self, text: str, max_length: int) -> list[int]:
        ids = [3 + (ord(c) % 50) for c in text]
        return ids[:max_length]

    def decode(self, ids: list[int]) -> str:
        return "x" * len(ids)


SAMPLE_EXAMPLES = [
    {"text": "hello world"},
    {"text": ""},  # empty -> skipped
    {"text": "a"},  # single token -> skipped (< 2 tokens)
    {"text": "b" * 100},  # longer than max_seq_len -> truncated
    {"text": "ok"},
]


def test_unknown_dataset_name_raises():
    with pytest.raises(ValueError):
        StreamingTextDataset("not_a_real_dataset", FakeTokenizer(), max_seq_len=16)


def test_invalid_shard_index_raises():
    with pytest.raises(ValueError):
        StreamingTextDataset(
            "the_pile", FakeTokenizer(), max_seq_len=16, num_shards=4, shard_index=4
        )


def test_no_shard_call_when_num_shards_is_one(monkeypatch):
    shard_calls = []
    monkeypatch.setattr(
        dataset_module,
        "load_dataset",
        lambda *a, **k: FakeHFStream(SAMPLE_EXAMPLES, shard_calls),
    )
    ds = StreamingTextDataset("the_pile", FakeTokenizer(), max_seq_len=10)
    list(ds)
    assert shard_calls == []


def test_shard_called_with_ddp_rank_and_world_size(monkeypatch):
    shard_calls = []
    monkeypatch.setattr(
        dataset_module,
        "load_dataset",
        lambda *a, **k: FakeHFStream(SAMPLE_EXAMPLES, shard_calls),
    )
    ds = StreamingTextDataset(
        "the_pile", FakeTokenizer(), max_seq_len=10, num_shards=2, shard_index=1
    )
    list(ds)
    assert shard_calls == [(2, 1)]


def test_streaming_dataset_filters_and_truncates(monkeypatch):
    monkeypatch.setattr(
        dataset_module, "load_dataset", lambda *a, **k: FakeHFStream(SAMPLE_EXAMPLES)
    )
    ds = StreamingTextDataset("the_pile", FakeTokenizer(), max_seq_len=10)
    items = list(ds)

    # empty text and single-char "a" should both be filtered out
    assert len(items) == 3
    for item in items:
        assert 2 <= len(item["input_ids"]) <= 10
        assert all(isinstance(tok, int) for tok in item["input_ids"])


def test_streaming_dataset_supports_wikipedia(monkeypatch):
    monkeypatch.setattr(
        dataset_module, "load_dataset", lambda *a, **k: FakeHFStream(SAMPLE_EXAMPLES)
    )
    ds = StreamingTextDataset("wikipedia", FakeTokenizer(), max_seq_len=10)
    items = list(ds)
    assert len(items) == 3


def test_local_data_files_bypasses_hub_repo_and_uses_local_parquet_builder(monkeypatch):
    """The whole point of local_data_files: load_dataset must be called with
    the local `parquet` builder and the given file paths, NOT the registry's
    Hub repo id — this is what actually avoids the network path that hung
    indefinitely on a real remote run (see the constructor's docstring)."""
    captured_calls = []

    def fake_load_dataset(*args, **kwargs):
        captured_calls.append((args, kwargs))
        return FakeHFStream(SAMPLE_EXAMPLES)

    monkeypatch.setattr(dataset_module, "load_dataset", fake_load_dataset)
    ds = StreamingTextDataset(
        "wikitext",
        FakeTokenizer(),
        max_seq_len=10,
        local_data_files=["/tmp/fake-train-0.parquet", "/tmp/fake-train-1.parquet"],
    )
    list(ds)

    assert len(captured_calls) == 1
    args, kwargs = captured_calls[0]
    assert args == ("parquet",) or kwargs.get("path") == "parquet"
    assert kwargs["data_files"] == {"train": ["/tmp/fake-train-0.parquet", "/tmp/fake-train-1.parquet"]}
    assert kwargs["streaming"] is True
    # Must NOT reference the Hub repo id anywhere in the call.
    assert "Salesforce/wikitext" not in str((args, kwargs))


def test_local_data_files_reads_a_real_local_parquet_file(tmp_path):
    """End-to-end, not just call-argument wiring: write a tiny real parquet
    file and confirm StreamingTextDataset actually reads real rows from it
    with no Hub interaction at all (no monkeypatch of load_dataset here)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_path = tmp_path / "train.parquet"
    table = pa.table({"text": ["hello world", "", "a second real row of text"]})
    pq.write_table(table, parquet_path)

    ds = StreamingTextDataset(
        "wikitext",
        FakeTokenizer(),
        max_seq_len=10,
        local_data_files=[str(parquet_path)],
    )
    items = list(ds)

    # The empty-text row is filtered out, same as the registry-streamed path.
    assert len(items) == 2
    for item in items:
        assert all(isinstance(tok, int) for tok in item["input_ids"])


def test_collate_fn_pads_and_builds_shifted_labels():
    batch = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]
    out = collate_fn(batch, pad_token_id=0)

    assert set(out.keys()) == {"input_ids", "attention_mask", "labels", "num_bytes"}
    assert out["input_ids"].shape == (2, 3)
    assert out["attention_mask"].shape == (2, 3)
    assert out["labels"].shape == (2, 3)
    for tensor in out.values():
        assert tensor.dtype == torch.long

    assert out["input_ids"].tolist() == [[1, 2, 3], [4, 5, 0]]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    # labels are input_ids shifted left by one; final + padded positions ignored
    assert out["labels"].tolist() == [[2, 3, IGNORE_INDEX], [5, IGNORE_INDEX, IGNORE_INDEX]]
