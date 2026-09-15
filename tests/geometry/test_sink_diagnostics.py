"""Tests for the outlier-position diagnostic. Validates that it correctly
distinguishes the two competing stories from the investigation: outliers
anchored at a fixed start position (compounding sink artifact) vs. outliers
anchored at the end of each sequence (selection-sharpening story).
"""

import numpy as np
import torch

from diffusion_moe.geometry.sink_diagnostics import (
    compare_delimiter_vs_content_norms,
    is_delimiter_like,
    per_token_id_norm_summary,
    summarize_outlier_positions,
    token_norms_from_capture,
)


def test_token_norms_masks_padding_as_nan():
    layer_output = torch.zeros(2, 4, 3)
    layer_output[0, 0] = torch.tensor([3.0, 4.0, 0.0])  # norm 5
    attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])

    norms = token_norms_from_capture(layer_output, attention_mask)

    assert norms.shape == (2, 4)
    assert norms[0, 0] == 5.0
    assert np.isnan(norms[0, 2])
    assert np.isnan(norms[0, 3])
    assert not np.isnan(norms[1]).any()


def test_summarize_outlier_positions_detects_start_anchored_sink():
    """Every sequence has one huge-norm token at position 0 (fixed start
    position, like a BOS-anchored sink) — outliers should cluster at
    dist_from_start=0 regardless of how long each sequence is.
    """
    rng = np.random.RandomState(0)
    batch, seq_len = 20, 10
    norms = rng.rand(batch, seq_len) * 2 + 1  # normal tokens: norm ~[1,3]
    norms[:, 0] = 1000.0  # sink at position 0 in every sequence
    attention_mask = np.ones((batch, seq_len), dtype=int)

    result = summarize_outlier_positions(norms, attention_mask, top_frac=0.05)

    assert result["frac_at_position_0"] == 1.0
    assert result["median_dist_from_start"] == 0.0
    # end distance varies with sequence length here (all length 10 -> fixed
    # 9), so this alone doesn't distinguish the stories; start-anchoring does
    for outlier in result["outliers"]:
        assert outlier["dist_from_start"] == 0


def test_summarize_outlier_positions_detects_end_anchored_selection():
    """Every sequence has one huge-norm token at its LAST valid position
    (varying start distance across sequences of different lengths, but
    always distance 0 from the end) — outliers should cluster at
    dist_from_end=0, not at a fixed start position.
    """
    rng = np.random.RandomState(0)
    batch, seq_len = 20, 10
    seq_lengths = rng.randint(4, seq_len + 1, size=batch)
    norms = rng.rand(batch, seq_len) * 2 + 1
    attention_mask = np.zeros((batch, seq_len), dtype=int)
    for b, length in enumerate(seq_lengths):
        attention_mask[b, :length] = 1
        norms[b, length - 1] = 1000.0  # outlier at the last valid position

    result = summarize_outlier_positions(norms, attention_mask, top_frac=0.05)

    assert result["frac_at_last_position"] == 1.0
    assert result["median_dist_from_end"] == 0.0
    # start distance varies with sequence length here, so it should NOT be
    # concentrated at 0 the way the sink-anchored case is
    assert result["frac_at_position_0"] < 1.0


def test_summarize_outlier_positions_reports_expected_keys_and_counts():
    rng = np.random.RandomState(0)
    norms = rng.rand(10, 6)
    attention_mask = np.ones((10, 6), dtype=int)

    result = summarize_outlier_positions(norms, attention_mask, top_frac=0.1)

    assert result["n_valid_tokens"] == 60
    assert result["n_flagged"] == 6  # ceil(0.1 * 60)
    assert len(result["outliers"]) == 6
    for key in ("batch_idx", "position", "norm", "dist_from_start", "dist_from_end"):
        assert key in result["outliers"][0]


def test_per_token_id_norm_summary_ranks_by_mean_norm_and_drops_rare_ids():
    # token id 7 is rare (appears twice) but huge-norm; token id 3 is
    # common (appears 10x) with a smaller but still elevated mean; token id
    # 1 is common and low-norm. min_count=5 should drop id 7 entirely, even
    # though its raw values are the most extreme in the data.
    input_ids = np.array([[1] * 10, [3] * 10, [7, 7] + [1] * 8])
    norms = np.array(
        [[1.0] * 10, [5.0] * 10, [500.0, 500.0] + [1.0] * 8]
    )
    attention_mask = np.ones_like(input_ids)

    stats = per_token_id_norm_summary(norms, attention_mask, input_ids, min_count=5)

    token_ids_reported = {s["token_id"] for s in stats}
    assert 7 not in token_ids_reported  # dropped: only 2 occurrences < min_count
    assert token_ids_reported == {1, 3}
    # sorted descending by mean_norm -> token 3 (mean 5.0) before token 1 (mean ~1.0)
    assert stats[0]["token_id"] == 3
    assert stats[0]["count"] == 10
    assert stats[0]["mean_norm"] == 5.0


def test_is_delimiter_like():
    assert is_delimiter_like("\n") is True
    assert is_delimiter_like(".") is True
    assert is_delimiter_like(" - ") is True
    assert is_delimiter_like("director") is False
    assert is_delimiter_like("2") is False  # digits count as alphanumeric
    assert is_delimiter_like("") is True  # no alnum chars present


def test_compare_delimiter_vs_content_norms_separates_groups_correctly():
    per_token_stats = [
        {"token_id": 13, "count": 50, "mean_norm": 100.0, "median_norm": 95.0, "max_norm": 200.0},
        {"token_id": 46, "count": 30, "mean_norm": 40.0, "median_norm": 38.0, "max_norm": 60.0},
        {"token_id": 99, "count": 40, "mean_norm": 2.0, "median_norm": 2.0, "max_norm": 3.0},
    ]
    token_texts = {13: "\n", 46: ".", 99: "director"}

    result = compare_delimiter_vs_content_norms(per_token_stats, token_texts)

    assert result["n_delimiter_token_ids"] == 2
    assert result["n_content_token_ids"] == 1
    assert result["delimiter_mean_norm"] == 70.0  # mean of [100, 40]
    assert result["content_mean_norm"] == 2.0
    assert result["top_delimiter_tokens"][0]["token_id"] == 13  # sorted desc
    assert result["top_delimiter_tokens"][0]["text"] == "\n"


def test_compare_delimiter_vs_content_norms_handles_empty_group():
    per_token_stats = [
        {"token_id": 99, "count": 40, "mean_norm": 2.0, "median_norm": 2.0, "max_norm": 3.0},
    ]
    result = compare_delimiter_vs_content_norms(per_token_stats, {99: "director"})

    assert result["n_delimiter_token_ids"] == 0
    assert result["delimiter_mean_norm"] is None
    assert result["top_delimiter_tokens"] == []
