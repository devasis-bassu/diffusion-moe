"""lm-evaluation-harness wrapper for standard LM benchmarks.

Requires `pip install lm-eval accelerate` (listed in requirements.txt).
"""

from __future__ import annotations

from typing import Any

try:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
except ImportError:  # lm-eval is heavy; only required when actually benchmarking
    lm_eval = None
    HFLM = None

SUPPORTED_TASKS = {"wikitext", "lambada_openai", "hellaswag", "arc_challenge", "mmlu"}

# Preferred metric name (matched against the harness's "metric,filter" key,
# split on the comma) for each task's headline number, in priority order.
# wikitext/lambada_openai are perplexity-style; the rest are accuracy-style.
_METRIC_PREFERENCE: dict[str, list[str]] = {
    "wikitext": ["word_perplexity", "perplexity"],
    "lambada_openai": ["perplexity", "acc"],
    "hellaswag": ["acc_norm", "acc"],
    "arc_challenge": ["acc_norm", "acc"],
    "mmlu": ["acc"],
}


def _extract_primary_metric(task: str, metrics: dict[str, Any]) -> float:
    """Picks the headline metric for a task out of the harness's raw metrics
    dict, which keys entries as "metric_name,filter_name" (e.g. "acc,none")."""
    for preferred in _METRIC_PREFERENCE.get(task, ["acc"]):
        for key, value in metrics.items():
            metric_name = key.split(",")[0]
            if metric_name == preferred and isinstance(value, (int, float)):
                return float(value)
    for value in metrics.values():
        if isinstance(value, (int, float)):
            return float(value)
    raise ValueError(f"No numeric metric found for task '{task}' in {metrics!r}")


def run_lm_eval(model: Any, tokenizer: Any, tasks: list[str]) -> dict[str, float]:
    """Runs `tasks` via lm-evaluation-harness and returns {task: headline_metric}
    — accuracy for hellaswag/arc_challenge/mmlu, perplexity for
    wikitext/lambada_openai.

    model, tokenizer: a HuggingFace-compatible model and tokenizer (e.g. a
    DiffusionMoETransformer isn't directly HF-compatible; use the pretrained
    HF model loaded in scripts/extract_geometry.py-style workflows, or wrap
    a custom model to satisfy HFLM's expected interface).
    """
    if lm_eval is None:
        raise ImportError(
            "lm_eval is not installed. Run `pip install lm-eval accelerate` to use run_lm_eval."
        )

    unsupported = set(tasks) - SUPPORTED_TASKS
    if unsupported:
        raise ValueError(
            f"Unsupported tasks: {sorted(unsupported)}. Supported: {sorted(SUPPORTED_TASKS)}"
        )

    lm = HFLM(pretrained=model, tokenizer=tokenizer)
    harness_output = lm_eval.simple_evaluate(model=lm, tasks=list(tasks))
    task_results = harness_output["results"]

    return {task: _extract_primary_metric(task, task_results[task]) for task in tasks}
