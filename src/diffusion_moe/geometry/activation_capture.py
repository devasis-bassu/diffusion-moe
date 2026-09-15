"""Shared machinery for pulling pooled per-layer activations out of a running
HF causal LM, for scripts that fit diffusion maps on those activations
(e.g. extract_geometry.py, multiscale_geometry.py) without re-implementing
hook registration and memory-bounded pooling in each one.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


class AttentionOutputCapture:
    """Registers forward hooks on every decoder layer's self-attention
    submodule to capture its raw output — the post-attention, pre-FFN-residual
    point DiffusionMoELayer actually routes from. HF's own
    `output_hidden_states` only exposes states at layer *boundaries* (after
    the full attention+FFN block), one per layer, so it can't give us this
    intermediate point; a hook on `.self_attn` is the only way to reach it.

    Assumes a Llama/Mistral-family model (model.model.layers, each with a
    .self_attn submodule) — true of both target models and most HF causal LMs
    that share that code path.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise ValueError(
                "Expected a Llama/Mistral-family model exposing `model.model.layers` "
                "(a ModuleList of decoder layers, each with a `.self_attn` submodule)."
            )
        self.layers = model.model.layers
        self.outputs: list[torch.Tensor] = []
        self._handles: list[Any] = []

    def _hook(self, module: torch.nn.Module, inputs: Any, output: Any) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        self.outputs.append(tensor.detach())

    def __enter__(self) -> "AttentionOutputCapture":
        self.outputs = []
        self._handles = [
            layer.self_attn.register_forward_hook(self._hook) for layer in self.layers
        ]
        return self

    def __exit__(self, *exc_info: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def subsample_valid_tokens(
    z: torch.Tensor,
    attention_mask: torch.Tensor,
    n_tokens: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Samples up to n_tokens rows without replacement from the non-padded
    positions of z. z: (batch, seq, d_model). attention_mask: (batch, seq),
    1 for real tokens / 0 for padding — padded positions carry no meaningful
    representation and would pollute the geometry estimate.
    """
    flat_z = z.reshape(-1, z.shape[-1])
    flat_mask = attention_mask.reshape(-1).bool()
    valid = flat_z[flat_mask]
    n = min(n_tokens, valid.shape[0])
    if n == 0:
        return valid
    idx = torch.randperm(valid.shape[0], generator=generator)[:n]
    return valid[idx]


def compute_tokens_per_batch(max_pool_size: int, n_sequences: int, batch_size: int) -> int:
    """Derives how many tokens to pool from EACH batch, given a target TOTAL
    pooled-token budget PER LAYER for the whole run.

    A fixed per-batch quota (the original design) makes total pooled memory
    scale linearly with n_sequences: at n_sequences=1000 on the real 32-layer,
    4096-dim Mistral-7B, pooling 512 tokens/batch across ~125 batches, in
    float64, for both pre- and post-attention, across all 32 layers held in
    memory at once, needs ~125GB — which is exactly what OOM-killed a real
    run on a 48GB machine. Deriving the per-batch quota from a fixed total
    budget instead keeps memory roughly constant (~max_pool_size tokens/layer)
    no matter how large n_sequences gets.
    """
    expected_n_batches = max(1, math.ceil(n_sequences / batch_size))
    return max(1, max_pool_size // expected_n_batches)


@torch.no_grad()
def collect_layer_activations(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    tokens_per_batch: int,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Runs the model over every batch in `loader`, subsampling
    `tokens_per_batch` valid tokens per batch at every layer, and pools them
    across batches. Returns (pre_attention, post_attention), each a list
    (one entry per layer) of (n_pooled_tokens, d_model) float32 arrays.

    `tokens_per_batch` should be derived from a total per-layer budget via
    compute_tokens_per_batch, not passed as a large fixed constant — see that
    function's docstring for why a fixed per-batch quota doesn't scale.
    """
    generator = torch.Generator().manual_seed(seed)
    pre_pool: list[list[torch.Tensor]] = None
    post_pool: list[list[torch.Tensor]] = None

    with AttentionOutputCapture(model) as capture:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            capture.outputs = []  # hooks only append; clear before each batch's forward pass
            outputs = model(
                input_ids, attention_mask=attention_mask, output_hidden_states=True
            )
            n_layers = len(capture.outputs)

            if pre_pool is None:
                pre_pool = [[] for _ in range(n_layers)]
                post_pool = [[] for _ in range(n_layers)]

            # one shared sample of valid-token positions per batch, reused
            # across every layer's pre/post tensors for a like-for-like
            # pre-vs-post-attention comparison at the same tokens
            flat_mask = attention_mask.reshape(-1).bool()
            n_valid = int(flat_mask.sum().item())
            n = min(tokens_per_batch, n_valid)
            sample_idx = torch.randperm(n_valid, generator=generator)[:n]
            d_model_dim = outputs.hidden_states[0].shape[-1]

            for layer_idx in range(n_layers):
                pre = outputs.hidden_states[layer_idx].reshape(-1, d_model_dim)
                post = capture.outputs[layer_idx].reshape(-1, d_model_dim)
                pre_pool[layer_idx].append(pre[flat_mask][sample_idx].cpu())
                post_pool[layer_idx].append(post[flat_mask][sample_idx].cpu())

    # Chunks are kept in the model's own compute dtype (e.g. bf16) while
    # accumulating, then cast to float32 only once here — bf16 has no native
    # numpy representation (.numpy() would raise), and NystromDiffusionMap
    # already upcasts to float64 internally itself, one layer at a time, so
    # doing it here too (on all layers held in memory at once) would only
    # double memory for no benefit.
    pre_arrays = [torch.cat(layer_chunks, dim=0).float().numpy() for layer_chunks in pre_pool]
    post_arrays = [torch.cat(layer_chunks, dim=0).float().numpy() for layer_chunks in post_pool]
    return pre_arrays, post_arrays
