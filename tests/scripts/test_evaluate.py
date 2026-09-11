"""Tests for scripts/evaluate.py's wiring: config loading, checkpoint
loading, running evaluations, saving results, and the --compare table.
"""

import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from diffusion_moe.data.dataset import IGNORE_INDEX
from diffusion_moe.models.model_factory import build_model_from_config
from diffusion_moe.models.moe_model import DiffusionMoETransformer
from diffusion_moe.training.optimizer import build_optimizer, build_scheduler
from diffusion_moe.training.trainer import Trainer

REPO_ROOT = Path(__file__).resolve().parents[2]

VOCAB_SIZE, SEQ_LEN, D_MODEL = 30, 8, 16


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_script", REPO_ROOT / "scripts" / "evaluate.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_script"] = module
    spec.loader.exec_module(module)
    return module


evaluate_script = _load_module()


class _TinyDataset(Dataset):
    def __init__(self, n=12, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.ids = torch.randint(0, VOCAB_SIZE, (n, SEQ_LEN), generator=g)

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):
        ids = self.ids[idx]
        labels = torch.cat([ids[1:], torch.tensor([IGNORE_INDEX])])
        return {
            "input_ids": ids,
            "attention_mask": torch.ones(SEQ_LEN, dtype=torch.long),
            "labels": labels,
        }


def _fake_build_dataloaders(_cfg):
    train_loader = DataLoader(_TinyDataset(n=12, seed=0), batch_size=4)
    val_loader = DataLoader(_TinyDataset(n=8, seed=1), batch_size=4)
    return train_loader, val_loader


def _tiny_cfg(**overrides):
    from hydra import compose, initialize

    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(
            config_name="base_config",
            overrides=[
                "model.vocab_size=30",
                "model.d_model=16",
                "model.n_layers=2",
                "model.n_heads=4",
                "model.max_seq_len=8",
                "model.ffn_dim=32",
                "routing.n_experts=4",
                "routing.top_k=2",
                "routing.n_components=3",
                "routing.n_landmarks=8",
                "routing.diffusion_t=2",
                "routing.centroid_refresh_steps=1000",
                *[f"{k}={v}" for k, v in overrides.items()],
            ],
        )
    return cfg


def _save_checkpoint(tmp_path, layers_to_replace, name="ckpt.pt"):
    # Built the same way scripts/train.py builds a checkpointed model — via
    # build_model_from_config — so the saved architecture always matches what
    # a later evaluate.py reconstruction from *this same returned cfg* will
    # produce. (layers_to_replace must agree between save and load, exactly
    # as it would have to for a real checkpoint.)
    cfg = _tiny_cfg(**{"layers_to_replace": list(layers_to_replace)})
    model = build_model_from_config(cfg)
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.0)
    scheduler = build_scheduler(optimizer, warmup_steps=0, total_steps=10)
    trainer = Trainer(
        model, optimizer, scheduler,
        DataLoader(_TinyDataset(), batch_size=4), DataLoader(_TinyDataset(), batch_size=4),
        config={"training": {"precision": "fp32"}, "routing": {}, "wandb": {}},
        device="cpu", checkpoint_dir=str(tmp_path / "ckpt_dir"),
    )
    path = tmp_path / name
    trainer.save_checkpoint(path)
    return path, cfg


def test_load_config_loads_base_config():
    cfg = evaluate_script.load_config("base_config")
    assert cfg.model.d_model == 1024


def test_load_model_from_checkpoint_restores_weights(tmp_path):
    ckpt_path, cfg = _save_checkpoint(tmp_path, layers_to_replace=[])

    model = evaluate_script.load_model_from_checkpoint(str(ckpt_path), cfg, device="cpu")
    assert isinstance(model, DiffusionMoETransformer)
    assert not model.training  # loaded in eval mode


def test_evaluate_checkpoint_dense_model_has_no_routing_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_script, "build_dataloaders", _fake_build_dataloaders)
    ckpt_path, cfg = _save_checkpoint(tmp_path, layers_to_replace=[])

    results = evaluate_script.evaluate_checkpoint(str(ckpt_path), cfg, "cpu", lm_eval_tasks=[])

    assert "perplexity" in results
    assert "bits_per_byte" in results
    assert "expert_token_counts" not in results


def test_evaluate_checkpoint_moe_model_has_routing_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_script, "build_dataloaders", _fake_build_dataloaders)
    ckpt_path, cfg = _save_checkpoint(tmp_path, layers_to_replace=[0])

    results = evaluate_script.evaluate_checkpoint(str(ckpt_path), cfg, "cpu", lm_eval_tasks=[])

    assert "expert_token_counts" in results
    assert "routing_entropy_per_expert" in results


def test_evaluate_checkpoint_lm_eval_failure_is_caught_not_raised(tmp_path, monkeypatch):
    """DiffusionMoETransformer isn't HF-compatible, so requesting lm_eval
    tasks should fail gracefully (recorded in results) rather than crash."""
    monkeypatch.setattr(evaluate_script, "build_dataloaders", _fake_build_dataloaders)
    ckpt_path, cfg = _save_checkpoint(tmp_path, layers_to_replace=[])

    results = evaluate_script.evaluate_checkpoint(
        str(ckpt_path), cfg, "cpu", lm_eval_tasks=["wikitext"]
    )
    assert "lm_eval_error" in results


def test_evaluate_checkpoint_lm_eval_success_path(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_script, "build_dataloaders", _fake_build_dataloaders)
    monkeypatch.setattr(
        evaluate_script, "run_lm_eval", lambda model, tokenizer, tasks: {"wikitext": 12.3}
    )
    monkeypatch.setattr(evaluate_script, "TokenizerWrapper", lambda name: object())
    ckpt_path, cfg = _save_checkpoint(tmp_path, layers_to_replace=[])

    results = evaluate_script.evaluate_checkpoint(
        str(ckpt_path), cfg, "cpu", lm_eval_tasks=["wikitext"]
    )
    assert results["lm_eval"] == {"wikitext": 12.3}
    assert "lm_eval_error" not in results


def test_save_results_writes_json(tmp_path):
    results = {"checkpoint": "a.pt", "perplexity": 12.5, "bits_per_byte": 1.2}
    path = evaluate_script.save_results(results, tmp_path, "my_run")
    assert path.name == "my_run_results.json"
    with open(path) as f:
        assert json.load(f) == results


def test_print_comparison_table_includes_both_checkpoint_names(capsys):
    results_a = {"checkpoint": "a.pt", "perplexity": 10.0, "bits_per_byte": 1.0}
    results_b = {"checkpoint": "b.pt", "perplexity": 8.0, "bits_per_byte": 0.9}
    evaluate_script.print_comparison_table(results_a, results_b)
    out = capsys.readouterr().out
    assert "a.pt" in out
    assert "b.pt" in out
    assert "perplexity" in out


def test_main_with_compare_saves_two_result_files_and_prints_table(tmp_path, monkeypatch, capsys):
    # main() calls the real load_env() first; spy on it (rather than a bare
    # no-op) so this test both proves the wiring is there AND never loads
    # this repo's actual .env (pytest's cwd is the repo root) or mutates the
    # real process environment with real secrets.
    load_env_calls = []
    monkeypatch.setattr(evaluate_script, "load_env", lambda: load_env_calls.append(1))
    monkeypatch.setattr(evaluate_script, "build_dataloaders", _fake_build_dataloaders)
    # both checkpoints must share one architecture cfg for --compare, exactly
    # as real usage requires: it's a single load_config() call for both.
    ckpt_a, cfg = _save_checkpoint(tmp_path, layers_to_replace=[], name="a.pt")
    ckpt_b, _ = _save_checkpoint(tmp_path, layers_to_replace=[], name="b.pt")

    monkeypatch.setattr(evaluate_script, "load_config", lambda config_name: cfg)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--checkpoint", str(ckpt_a),
            "--compare", str(ckpt_b),
            "--output_dir", str(tmp_path / "results"),
        ],
    )

    evaluate_script.main()

    assert (tmp_path / "results" / "a_results.json").exists()
    assert (tmp_path / "results" / "b_results.json").exists()
    out = capsys.readouterr().out
    assert "a.pt" in out
    assert "b.pt" in out
    assert load_env_calls == [1]
