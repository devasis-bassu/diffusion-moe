"""Builds train/val DataLoaders from a config."""

from __future__ import annotations

from functools import partial
from typing import Any

from torch.utils.data import DataLoader

from diffusion_moe.data.dataset import StreamingTextDataset, collate_fn
from diffusion_moe.data.tokenizer import TokenizerWrapper
from diffusion_moe.utils.device import get_rank, get_world_size


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    """Reads `key` from cfg, whether cfg is a dict, OmegaConf node, or namespace."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_dataloaders(config: Any) -> tuple[DataLoader, DataLoader]:
    """Builds (train_loader, val_loader) from a config exposing a `data` section.

    Expected fields under config.data: dataset, tokenizer, max_seq_len, batch_size,
    num_workers, val_tokens, and optionally prefetch_factor, local_data_files.
    """
    data_cfg = _get(config, "data", config)

    dataset_name = _get(data_cfg, "dataset")
    tokenizer_name = _get(data_cfg, "tokenizer")
    max_seq_len = _get(data_cfg, "max_seq_len")
    batch_size = _get(data_cfg, "batch_size")
    num_workers = _get(data_cfg, "num_workers", 0)
    prefetch_factor = _get(data_cfg, "prefetch_factor", 2)
    val_tokens = _get(data_cfg, "val_tokens", 0)
    seed = _get(data_cfg, "seed", 42)
    # See StreamingTextDataset's own docstring: bypasses Hub streaming (a
    # recurring flaky-CDN stall hit repeatedly across this investigation) by
    # reading already-downloaded local parquet files directly. None by
    # default, so behavior is unchanged unless explicitly set.
    local_data_files = _get(data_cfg, "local_data_files", None)

    tokenizer = TokenizerWrapper(tokenizer_name)
    collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)

    val_n_examples = max(1, val_tokens // max_seq_len) if val_tokens else 0

    # Under DDP, each rank streams a disjoint shard of the training data (see
    # StreamingTextDataset.shard) — a no-op (num_shards=1) outside a
    # distributed run. The val set is deliberately left unsharded: it's
    # already tiny, and every rank evaluating the same held-out examples is
    # simpler and harmless (a bit of redundant compute, not a correctness issue).
    train_dataset = StreamingTextDataset(
        dataset_name,
        tokenizer,
        max_seq_len,
        split="train",
        skip=val_n_examples,
        seed=seed,
        num_shards=get_world_size(),
        shard_index=get_rank(),
        local_data_files=local_data_files,
    )
    val_dataset = StreamingTextDataset(
        dataset_name,
        tokenizer,
        max_seq_len,
        split="train",
        take=val_n_examples or 1,
        seed=seed,
        local_data_files=local_data_files,
    )

    loader_kwargs: dict[str, Any] = {"num_workers": num_workers}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=0,
    )

    return train_loader, val_loader
