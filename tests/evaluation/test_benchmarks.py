"""Tests for run_lm_eval — mocks lm_eval.simple_evaluate/HFLM so no real
benchmark data is downloaded or evaluated."""

import pytest

from diffusion_moe.evaluation import benchmarks as benchmarks_module
from diffusion_moe.evaluation.benchmarks import _extract_primary_metric, run_lm_eval


def test_rejects_unsupported_tasks():
    with pytest.raises(ValueError):
        run_lm_eval(model=object(), tokenizer=object(), tasks=["not_a_real_task"])


def test_extract_primary_metric_prefers_word_perplexity_for_wikitext():
    metrics = {"word_perplexity,none": 12.3, "byte_perplexity,none": 1.5}
    assert _extract_primary_metric("wikitext", metrics) == 12.3


def test_extract_primary_metric_prefers_acc_norm_for_hellaswag():
    metrics = {"acc,none": 0.5, "acc_norm,none": 0.6}
    assert _extract_primary_metric("hellaswag", metrics) == 0.6


def test_extract_primary_metric_falls_back_to_any_numeric_value():
    metrics = {"some_unexpected_metric,none": 0.42}
    assert _extract_primary_metric("mmlu", metrics) == 0.42


def test_extract_primary_metric_raises_when_nothing_numeric():
    with pytest.raises(ValueError):
        _extract_primary_metric("mmlu", {"alias": "mmlu"})


def test_raises_import_error_when_lm_eval_unavailable(monkeypatch):
    monkeypatch.setattr(benchmarks_module, "lm_eval", None)
    with pytest.raises(ImportError):
        run_lm_eval(model=object(), tokenizer=object(), tasks=["wikitext"])


def test_run_lm_eval_wires_hflm_and_extracts_metrics(monkeypatch):
    calls = {}

    class FakeHFLM:
        def __init__(self, pretrained, tokenizer):
            calls["hflm_init"] = {"pretrained": pretrained, "tokenizer": tokenizer}

    class FakeLmEval:
        @staticmethod
        def simple_evaluate(model, tasks):
            calls["simple_evaluate"] = {"model": model, "tasks": tasks}
            return {
                "results": {
                    "wikitext": {"word_perplexity,none": 15.0},
                    "hellaswag": {"acc,none": 0.4, "acc_norm,none": 0.55},
                }
            }

    monkeypatch.setattr(benchmarks_module, "lm_eval", FakeLmEval)
    monkeypatch.setattr(benchmarks_module, "HFLM", FakeHFLM)

    fake_model = object()
    fake_tokenizer = object()
    result = run_lm_eval(fake_model, fake_tokenizer, ["wikitext", "hellaswag"])

    assert result == {"wikitext": 15.0, "hellaswag": 0.55}
    assert calls["hflm_init"]["pretrained"] is fake_model
    assert calls["hflm_init"]["tokenizer"] is fake_tokenizer
    assert calls["simple_evaluate"]["tasks"] == ["wikitext", "hellaswag"]
