"""Streaming text dataset for causal LM pretraining."""

from __future__ import annotations

from typing import Any, Iterator

import torch
from datasets import IterableDataset as HFIterableDataset
from datasets import load_dataset
from torch.utils.data import IterableDataset

from diffusion_moe.data.tokenizer import TokenizerWrapper

# Maps a short dataset key (as used in configs/base_config.yaml) to the
# HuggingFace Hub dataset path/config needed to stream it.
DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "the_pile": {
        "path": "monology/pile-uncopyrighted",
        "name": None,
        "text_field": "text",
    },
    "wikipedia": {
        "path": "wikimedia/wikipedia",
        "name": "20231101.en",
        "text_field": "text",
    },
    "wikitext": {
        # The bare "wikitext" repo id is deprecated on the Hub (redirects,
        # which datasets' HfFileSystem URI parsing doesn't follow) — the
        # dataset now lives under the Salesforce namespace.
        "path": "Salesforce/wikitext",
        "name": "wikitext-103-raw-v1",
        "text_field": "text",
    },
}

IGNORE_INDEX = -100


class StreamingTextDataset(IterableDataset):
    """Streams raw text from a supported HF dataset and tokenizes it on the fly.

    Yields dicts of {"input_ids": list[int]} truncated to max_seq_len. No padding
    is applied here — that happens in collate_fn so batches pad only to their own
    longest sequence.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer: TokenizerWrapper,
        max_seq_len: int,
        split: str = "train",
        skip: int = 0,
        take: int | None = None,
        seed: int = 42,
        num_shards: int = 1,
        shard_index: int = 0,
        local_data_files: list[str] | None = None,
    ) -> None:
        """local_data_files: optional list of local parquet file paths to read
        directly instead of streaming from the Hub. A real, recurring
        reliability problem, not a hypothetical one: this project's own repo
        streaming has stalled indefinitely twice on two different datasets
        (a 130+ minute wikipedia CDN stall; a wikitext stall traced to HF's
        newer "Xet" CDN backend issuing many small, separately-connected
        byte-range requests that can hang on a slow/lossy connection).
        Crucially, `hf_hub_download`-ing the files ahead of time does NOT
        help streaming mode on its own — confirmed directly: streaming reads
        go through fsspec's HfFileSystem, which re-fetches over the network
        regardless of what's already sitting in the local hub cache. Passing
        the already-downloaded files' paths here bypasses that: `load_dataset`
        is called with the local `parquet` builder instead of the registry's
        Hub repo id, so reading never touches the network at all. `dataset_name`
        is still required and still selects `text_field` from the registry
        (schema, not source) — the local files must match that dataset's schema.
        """
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Supported: {list(DATASET_REGISTRY)}"
            )
        if not 0 <= shard_index < num_shards:
            raise ValueError(f"shard_index ({shard_index}) must be in [0, num_shards={num_shards})")
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split
        self.skip = skip
        self.take = take
        self.seed = seed
        self.num_shards = num_shards
        self.shard_index = shard_index
        self.local_data_files = local_data_files
        self._registry_entry = DATASET_REGISTRY[dataset_name]

    def _build_hf_stream(self) -> HFIterableDataset:
        entry = self._registry_entry
        if self.local_data_files is not None:
            stream = load_dataset(
                "parquet",
                data_files={self.split: self.local_data_files},
                split=self.split,
                streaming=True,
            )
        else:
            stream = load_dataset(
                entry["path"],
                name=entry["name"],
                split=self.split,
                streaming=True,
            )
        # Each DDP rank gets a disjoint slice of the underlying source shards
        # BEFORE shuffling, so ranks train on non-overlapping data — the
        # standard pattern for streaming IterableDatasets under DDP (there's
        # no DistributedSampler equivalent for a stream with no fixed length).
        if self.num_shards > 1:
            stream = stream.shard(num_shards=self.num_shards, index=self.shard_index)
        stream = stream.shuffle(seed=self.seed, buffer_size=10_000)
        if self.skip:
            stream = stream.skip(self.skip)
        if self.take is not None:
            stream = stream.take(self.take)
        return stream

    def __iter__(self) -> Iterator[dict[str, Any]]:
        text_field = self._registry_entry["text_field"]
        for example in self._build_hf_stream():
            text = example[text_field]
            if not text:
                continue
            input_ids = self.tokenizer.encode(text, max_length=self.max_seq_len)
            if len(input_ids) < 2:
                continue
            # Decode the retained (possibly truncated) tokens back to text so
            # bits-per-byte (compute_perplexity) has the exact byte length of
            # what the model was actually evaluated on, not the original doc.
            num_bytes = len(self.tokenizer.decode(input_ids).encode("utf-8"))
            yield {"input_ids": input_ids, "num_bytes": num_bytes}


def collate_fn(
    batch: list[dict[str, Any]], pad_token_id: int
) -> dict[str, torch.Tensor]:
    """Pads a batch of variable-length token sequences and builds shifted labels.

    Returns input_ids, attention_mask (both (batch, seq_len) LongTensors) and
    labels, where labels[i] is input_ids shifted left by one position (i.e.
    labels[i][t] = input_ids[i][t + 1]) with the final position and any padding
    set to IGNORE_INDEX so the loss ignores them. Also returns num_bytes, a
    (batch,) LongTensor of each example's UTF-8 byte length (0 if the dataset
    didn't supply one) — used by compute_perplexity for bits-per-byte.
    """
    max_len = max(len(ex["input_ids"]) for ex in batch)

    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    num_bytes = torch.zeros(len(batch), dtype=torch.long)

    for i, example in enumerate(batch):
        ids = torch.as_tensor(example["input_ids"], dtype=torch.long)
        seq_len = ids.shape[0]
        input_ids[i, :seq_len] = ids
        attention_mask[i, :seq_len] = 1
        if seq_len > 1:
            labels[i, : seq_len - 1] = ids[1:]
        num_bytes[i] = example.get("num_bytes", 0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "num_bytes": num_bytes,
    }
