"""Perplexity and bits-per-byte evaluation."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F

from diffusion_moe.data.dataset import IGNORE_INDEX


def _forward_logits(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Normalises model(input_ids) to just its logits, whether the model
    returns (logits, activations) or (logits, activations, router_outputs)."""
    outputs = model(input_ids)
    return outputs[0] if isinstance(outputs, tuple) else outputs


@torch.no_grad()
def compute_perplexity(model: torch.nn.Module, dataloader: Iterable) -> dict[str, float]:
    """Runs the model in eval mode over `dataloader` and returns perplexity
    and bits-per-byte.

    Perplexity = exp(mean NLL per valid token) — the standard token-level
    metric, comparable across runs using the same tokenizer.

    Bits-per-byte normalises by the underlying text's UTF-8 byte length
    instead of token count, making it comparable ACROSS tokenizers with
    different token-to-byte ratios. It requires each batch to carry
    "num_bytes" (as diffusion_moe.data.dataset.collate_fn provides); batches
    without it yield NaN for bits_per_byte (perplexity is unaffected).

    Returns {"perplexity": float, "bits_per_byte": float}.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = _forward_logits(model, input_ids)
        vocab_size = logits.shape[-1]
        batch_nll_sum = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        n_valid = int((labels != IGNORE_INDEX).sum().item())

        total_nll += batch_nll_sum.item()
        total_tokens += n_valid
        if "num_bytes" in batch:
            total_bytes += int(batch["num_bytes"].sum().item())

    if was_training:
        model.train()

    mean_nll = total_nll / max(1, total_tokens)
    perplexity = math.exp(min(mean_nll, 20.0))  # cap to avoid inf on a garbage model
    bits_per_byte = (total_nll / total_bytes) / math.log(2) if total_bytes > 0 else float("nan")

    return {"perplexity": perplexity, "bits_per_byte": bits_per_byte}
