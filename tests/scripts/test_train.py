"""Tests for scripts/train.py's wiring: config composition, model/optimizer/
scheduler construction, and the resume_from_checkpoint flag.

scripts/ isn't part of the installed package (it's a set of standalone CLI
entrypoints, not importable via `diffusion_moe.*`), so the module is loaded
directly from its file path.
"""

import importlib.util
import sys
from pathlib import Path

import torch
from hydra import compose, initialize
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


def _load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_script", REPO_ROOT / "scripts" / "train.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_script"] = module
    spec.loader.exec_module(module)
    return module


train_script = _load_train_module()


class _TinyDataset(Dataset):
    def __init__(self, n=16, seed=0, seq_len=8, vocab_size=32):
        g = torch.Generator().manual_seed(seed)
        self.ids = torch.randint(0, vocab_size, (n, seq_len), generator=g)
        self.seq_len = seq_len

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):
        ids = self.ids[idx]
        labels = torch.cat([ids[1:], torch.tensor([-100])])
        return {
            "input_ids": ids,
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "labels": labels,
        }


def _fake_build_dataloaders(_cfg):
    train_loader = DataLoader(_TinyDataset(n=16, seed=0), batch_size=4)
    val_loader = DataLoader(_TinyDataset(n=8, seed=1), batch_size=4)
    return train_loader, val_loader


def _tiny_cfg(**overrides):
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(
            config_name="base_config",
            overrides=[
                "model.vocab_size=32",
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
                "data.max_seq_len=8",
                "data.batch_size=4",
                "training.total_tokens=200",  # tiny: forces total_steps small
                "training.warmup_steps=0",
                "training.grad_accum_steps=1",
                "training.precision=fp32",
                "training.checkpoint_steps=100",
                "training.eval_steps=100",
                "training.log_steps=1",
                *[f"{k}={v}" for k, v in overrides.items()],
            ],
        )
    return cfg


def test_configs_compose_for_both_model_sizes():
    with initialize(version_base=None, config_path="../../configs"):
        cfg_300m = compose(config_name="300m")
        cfg_13b = compose(config_name="1.3b")
    assert cfg_300m.model.d_model == 1024
    assert cfg_13b.model.d_model == 2048


def test_build_model_from_config_matches_cfg_dims():
    cfg = _tiny_cfg()
    model = train_script.build_model_from_config(cfg)
    assert model.d_model == 16
    assert len(model.blocks) == 2


def test_run_seed_makes_model_init_reproducible(tmp_path, monkeypatch):
    """Real gap this fixes: cfg.seed was never applied to torch's global RNG
    anywhere in this pipeline (only data.seed, controlling shuffle order,
    was wired through) -- model weight init, the dominant source of run-to-
    run variation, drew from whatever unseeded state the process happened
    to start in. Same cfg.seed, same init; different cfg.seed, different
    init."""
    monkeypatch.setattr(train_script, "build_dataloaders", _fake_build_dataloaders)
    monkeypatch.chdir(tmp_path)

    cfg_a = _tiny_cfg(seed=123)
    trainer_a = train_script.run(cfg_a)
    first_param_a = next(trainer_a.model.parameters()).clone()

    cfg_b = _tiny_cfg(seed=123)
    trainer_b = train_script.run(cfg_b)
    first_param_b = next(trainer_b.model.parameters()).clone()

    cfg_c = _tiny_cfg(seed=456)
    trainer_c = train_script.run(cfg_c)
    first_param_c = next(trainer_c.model.parameters()).clone()

    assert torch.equal(first_param_a, first_param_b)  # same seed -> same init
    assert not torch.equal(first_param_a, first_param_c)  # different seed -> different init


def test_compute_total_steps():
    cfg = _tiny_cfg()
    # tokens_per_step = batch_size(4) * max_seq_len(8) * grad_accum_steps(1) = 32
    # total_tokens=200 -> total_steps = 200 // 32 = 6
    assert train_script.compute_total_steps(cfg) == 6


def test_compute_total_steps_floors_at_one():
    cfg = _tiny_cfg(**{"training.total_tokens": 1})
    assert train_script.compute_total_steps(cfg) == 1


def test_run_trains_and_checkpoints(tmp_path, monkeypatch):
    cfg = _tiny_cfg()

    monkeypatch.setattr(train_script, "build_dataloaders", _fake_build_dataloaders)
    monkeypatch.chdir(tmp_path)  # Trainer's default checkpoint_dir="checkpoints" is relative

    trainer = train_script.run(cfg)

    assert trainer.step == train_script.compute_total_steps(cfg)
    assert (tmp_path / "checkpoints").exists()


def test_run_resumes_from_checkpoint(tmp_path, monkeypatch):
    cfg = _tiny_cfg()

    monkeypatch.setattr(train_script, "build_dataloaders", _fake_build_dataloaders)
    monkeypatch.chdir(tmp_path)

    first_trainer = train_script.run(cfg)
    # eval/checkpoint steps are set to 100 (never triggered in a 6-step run),
    # so save one explicitly to resume from.
    saved_path = tmp_path / "manual_resume.pt"
    first_trainer.save_checkpoint(saved_path)

    resumed_cfg = _tiny_cfg(**{"resume_from_checkpoint": str(saved_path)})
    monkeypatch.setattr(train_script, "build_dataloaders", _fake_build_dataloaders)
    # a fresh total_tokens=0 run would immediately satisfy max_steps if step
    # already >= total_steps; keep the same total_tokens so it "continues"
    resumed_trainer = train_script.run(resumed_cfg)

    # resumed run starts already at first_trainer's final step, and since
    # total_steps is the same, train() should be a no-op loop (step unchanged)
    assert resumed_trainer.step == first_trainer.step


def test_run_calls_setup_and_cleanup_distributed(tmp_path, monkeypatch):
    """setup_distributed()/cleanup_distributed() are no-ops outside a
    torchrun-launched job, but run() must always call them — the cleanup in
    particular needs to fire even on error, hence the try/finally."""
    cfg = _tiny_cfg()
    monkeypatch.setattr(train_script, "build_dataloaders", _fake_build_dataloaders)
    monkeypatch.chdir(tmp_path)

    calls = []
    monkeypatch.setattr(train_script, "setup_distributed", lambda: calls.append("setup"))
    monkeypatch.setattr(train_script, "cleanup_distributed", lambda: calls.append("cleanup"))

    train_script.run(cfg)
    assert calls == ["setup", "cleanup"]


def test_run_calls_cleanup_distributed_even_on_error(tmp_path, monkeypatch):
    cfg = _tiny_cfg()

    def _broken_build_dataloaders(_cfg):
        raise RuntimeError("boom")

    monkeypatch.setattr(train_script, "build_dataloaders", _broken_build_dataloaders)
    monkeypatch.chdir(tmp_path)

    calls = []
    monkeypatch.setattr(train_script, "setup_distributed", lambda: calls.append("setup"))
    monkeypatch.setattr(train_script, "cleanup_distributed", lambda: calls.append("cleanup"))

    import pytest

    with pytest.raises(RuntimeError):
        train_script.run(cfg)
    assert calls == ["setup", "cleanup"]
