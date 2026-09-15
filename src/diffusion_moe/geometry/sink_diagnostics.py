"""Identifies WHICH token positions and identities drive the norm outliers
behind the disconnection artifacts found at layers 1, 2, 4, and 31 — the
investigation so far established that outlier-norm tokens fragment the
raw-Euclidean kernel graph, but not which tokens those outliers actually are.

The position-based diagnostic (summarize_outlier_positions) was originally
built to distinguish two stories:
  - "compounding structural artifact": the layer-31 spike is the SAME
    sink/BOS-anchored mechanism established at layers 1/2/4, its magnitude
    just compounding through the residual stream with depth. Predicted
    outliers clustering at a FIXED position near the START of the sequence.
  - "selection sharpening": the layer-31 spike is a distinct, prediction-
    related process. Predicted outliers shifting toward the END of the
    sequence at layer 31, unlike the early layers.

Neither predicted what was actually found on the real model: outliers are
NOT anchored to position 0 at all (frac_at_position_0 = 0.00 at every
flagged layer) — they're anchored to a specific TOKEN IDENTITY (the newline
character) wherever it occurs. That still supports "compounding artifact"
over "selection sharpening" (same token, same mechanism, growing magnitude
with depth), just via a different concrete carrier than assumed. The
per-token-identity functions below (per_token_id_norm_summary,
compare_delimiter_vs_content_norms) extend the diagnostic to check whether
this is specific to newline or a broader class of low-content/punctuation
tokens.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def token_norms_from_capture(
    layer_output: torch.Tensor, attention_mask: torch.Tensor
) -> np.ndarray:
    """L2 norm of each token's post-attention vector. layer_output: (batch,
    seq, d_model). attention_mask: (batch, seq). Returns (batch, seq) float
    array with padded positions set to NaN, so they never qualify as
    outliers and never pollute rank/percentile statistics.
    """
    norms = layer_output.float().norm(dim=-1).cpu().numpy()
    mask = attention_mask.cpu().numpy().astype(bool)
    return np.where(mask, norms, np.nan)


def summarize_outlier_positions(
    norms: np.ndarray, attention_mask: np.ndarray, top_frac: float = 0.01
) -> dict[str, Any]:
    """Finds the top `top_frac` fraction of valid tokens by norm (pooled
    across the whole batch) and reports, for each, its position from the
    START of its own sequence and from the END (i.e. how many valid tokens
    follow it) — the two candidate stories above make opposite predictions
    about which of these stays small.

    norms: (batch, seq) from token_norms_from_capture. attention_mask:
    (batch, seq), 1 for valid tokens.
    """
    mask = attention_mask.astype(bool)
    seq_lengths = mask.sum(axis=1)  # valid tokens per sequence
    batch_idx, positions = np.where(mask)
    values = norms[batch_idx, positions]

    n_valid = len(values)
    n_top = max(1, int(np.ceil(top_frac * n_valid)))
    order = np.argsort(values)[::-1][:n_top]

    top_batch = batch_idx[order]
    top_pos = positions[order]
    top_values = values[order]
    dist_from_start = top_pos
    dist_from_end = seq_lengths[top_batch] - 1 - top_pos

    return {
        "n_valid_tokens": int(n_valid),
        "n_flagged": int(n_top),
        "median_norm": float(np.nanmedian(values)),
        "outliers": [
            {
                "batch_idx": int(b),
                "position": int(p),
                "norm": float(v),
                "dist_from_start": int(ds),
                "dist_from_end": int(de),
            }
            for b, p, v, ds, de in zip(
                top_batch, top_pos, top_values, dist_from_start, dist_from_end
            )
        ],
        "median_dist_from_start": float(np.median(dist_from_start)),
        "median_dist_from_end": float(np.median(dist_from_end)),
        "frac_at_position_0": float(np.mean(dist_from_start == 0)),
        "frac_at_last_position": float(np.mean(dist_from_end == 0)),
    }


def per_token_id_norm_summary(
    norms: np.ndarray, attention_mask: np.ndarray, input_ids: np.ndarray, min_count: int = 5
) -> list[dict[str, Any]]:
    """Aggregates norm statistics per UNIQUE token id across every valid
    occurrence, not just the extreme top-frac cut summarize_outlier_positions
    uses. That cut is biased toward whichever token simply occurs most often
    in the tail (which is exactly why newline dominated it) — this instead
    answers "which token identities are elevated on average," so rarer but
    still-elevated token classes (e.g. specific punctuation) aren't hidden
    behind newline's sheer occurrence count.

    Token ids appearing fewer than `min_count` times are dropped (a single
    lucky/unlucky occurrence isn't a reliable per-token-id statistic).
    Sorted by mean_norm descending.

    norms, attention_mask: (batch, seq) as elsewhere in this module.
    input_ids: (batch, seq) int array of token ids (same shape).
    """
    mask = attention_mask.astype(bool)
    ids = input_ids[mask]
    values = norms[mask]

    unique_ids, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    stats: list[dict[str, Any]] = []
    for i, token_id in enumerate(unique_ids):
        if counts[i] < min_count:
            continue
        token_values = values[inverse == i]
        stats.append(
            {
                "token_id": int(token_id),
                "count": int(counts[i]),
                "mean_norm": float(np.mean(token_values)),
                "median_norm": float(np.median(token_values)),
                "max_norm": float(np.max(token_values)),
            }
        )
    stats.sort(key=lambda s: s["mean_norm"], reverse=True)
    return stats


def is_delimiter_like(text: str) -> bool:
    """True if the decoded token contains no alphanumeric characters — a
    cheap, tokenizer-agnostic proxy for "low-content, punctuation/whitespace
    -only" tokens (newlines, quotes, dashes, periods, ...), the class the
    real-model diagnostic found the sink-carrying token (newline) belongs
    to.
    """
    return not any(c.isalnum() for c in text)


def compare_delimiter_vs_content_norms(
    per_token_stats: list[dict[str, Any]], token_texts: dict[int, str]
) -> dict[str, Any]:
    """Splits per_token_id_norm_summary's output into delimiter-like vs.
    content token ids (via is_delimiter_like on each token's decoded text)
    and compares their mean_norm distributions — answers whether elevated
    norm is a property of the WHOLE delimiter/punctuation class, or
    specific to newline alone.

    per_token_stats: output of per_token_id_norm_summary. token_texts: maps
    token_id -> its decoded text (computed by the caller, since decoding
    needs a real tokenizer this module doesn't depend on).
    """
    delim_norms: list[float] = []
    content_norms: list[float] = []
    delim_details: list[dict[str, Any]] = []
    for stat in per_token_stats:
        text = token_texts.get(stat["token_id"], "")
        if is_delimiter_like(text):
            delim_norms.append(stat["mean_norm"])
            delim_details.append({**stat, "text": text})
        else:
            content_norms.append(stat["mean_norm"])

    delim_details.sort(key=lambda d: d["mean_norm"], reverse=True)

    return {
        "n_delimiter_token_ids": len(delim_norms),
        "n_content_token_ids": len(content_norms),
        "delimiter_mean_norm": float(np.mean(delim_norms)) if delim_norms else None,
        "content_mean_norm": float(np.mean(content_norms)) if content_norms else None,
        "delimiter_median_norm": float(np.median(delim_norms)) if delim_norms else None,
        "content_median_norm": float(np.median(content_norms)) if content_norms else None,
        "top_delimiter_tokens": delim_details[:10],
    }
