"""Tests for scripts/run_sweep.py: sweep config, per-run overrides, custom
metric computation, the dry-run pipeline, and wandb sweep/agent wiring —
all mocked so nothing touches a real wandb account or downloads real data.
"""

import importlib.util
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from diffusion_moe.data.dataset import IGNORE_INDEX
from diffusion_moe.models.moe_model import DiffusionMoETransformer

REPO_ROOT = Path(__file__).resolve().parents[2]

VOCAB_SIZE, SEQ_LEN, D_MODEL = 30, 8, 16


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_sweep_script", REPO_ROOT / "scripts" / "run_sweep.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_sweep_script"] = module
    spec.loader.exec_module(module)
    return module


run_sweep = _load_module()


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
            "num_bytes": torch.tensor(20),
        }


def _fake_build_dataloaders(_cfg):
    train_loader = DataLoader(_TinyDataset(n=12, seed=0), batch_size=4)
    val_loader = DataLoader(_TinyDataset(n=8, seed=1), batch_size=4)
    return train_loader, val_loader


def _tiny_cfg(router="diffusion"):
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
                f"model.router={router}",
                "routing.n_experts=4",
                "routing.top_k=2",
                "routing.n_components=3",
                "routing.n_landmarks=8",
                "routing.diffusion_t=2",
                "routing.centroid_refresh_steps=1000",
                "data.max_seq_len=8",
                "data.batch_size=4",
                "training.total_tokens=200",
                "training.warmup_steps=0",
                "training.grad_accum_steps=1",
                "training.precision=fp32",
                "training.checkpoint_steps=100000",
                "training.eval_steps=100000",
                "training.log_steps=1",
            ],
        )
    return cfg


def _model(router="diffusion", layers_to_replace=None):
    kwargs = dict(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=2, n_heads=4, max_seq_len=SEQ_LEN,
        n_experts=4, top_k=2, n_components=3, n_landmarks=8, diffusion_t=2,
        centroid_refresh_steps=1000, router=router,
    )
    if layers_to_replace is not None:
        kwargs["layers_to_replace"] = layers_to_replace
    return DiffusionMoETransformer(**kwargs)


def test_sweep_config_structure():
    cfg = run_sweep.SWEEP_CONFIG
    assert cfg["method"] == "grid"
    params = cfg["parameters"]
    assert params["router"]["values"] == ["diffusion", "cosine", "switch", "random"]
    assert params["n_experts"]["values"] == [4, 8, 16]
    assert params["top_k"]["values"] == [1, 2]
    assert params["nu_sep"]["values"] == [0.0, 0.01, 0.1]


def test_apply_sweep_overrides_sets_all_fields_without_mutating_base():
    base_cfg = _tiny_cfg()
    original_router = base_cfg.model.router

    sweep_params = {"router": "switch", "n_experts": 8, "top_k": 1, "nu_sep": 0.01}
    cfg = run_sweep.apply_sweep_overrides(base_cfg, sweep_params)

    assert cfg.model.router == "switch"
    assert cfg.routing.n_experts == 8
    assert cfg.routing.top_k == 1
    assert cfg.routing.nu_sep == 0.01
    assert cfg.data.dataset == "the_pile"
    assert cfg.training.total_tokens == run_sweep.RUN_TOKENS
    assert cfg.training.log_steps == run_sweep.LOG_STEPS
    # base config untouched
    assert base_cfg.model.router == original_router


def test_active_flop_fraction():
    assert run_sweep.active_flop_fraction(top_k=2, n_experts=8) == 0.25
    assert run_sweep.active_flop_fraction(top_k=1, n_experts=4) == 0.25
    assert run_sweep.active_flop_fraction(top_k=2, n_experts=8, overlap_factor=2.0) == 0.5


def test_compute_mean_centroid_distance_diffusion_router():
    model = _model(router="diffusion")
    distance = run_sweep.compute_mean_centroid_distance(model)
    assert distance is not None
    assert distance >= 0.0


def test_compute_mean_centroid_distance_cosine_router():
    model = _model(router="cosine")
    distance = run_sweep.compute_mean_centroid_distance(model)
    assert distance is not None
    assert distance >= 0.0


def test_compute_mean_centroid_distance_none_for_switch_and_random():
    assert run_sweep.compute_mean_centroid_distance(_model(router="switch")) is None
    assert run_sweep.compute_mean_centroid_distance(_model(router="random")) is None


def test_compute_mean_centroid_distance_averages_only_moe_layers():
    """A mix of one diffusion-MoE layer and one dense TransformerBlock should
    average over just the MoE layer, not error on the dense one."""
    model = _model(router="diffusion", layers_to_replace=[0])
    distance = run_sweep.compute_mean_centroid_distance(model)
    assert distance is not None


def test_train_one_run_logs_expected_metrics_and_saves_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(run_sweep, "build_dataloaders", _fake_build_dataloaders)
    cfg = _tiny_cfg(router="diffusion")
    checkpoint_dir = str(tmp_path / "run1")

    metrics = run_sweep.train_one_run(cfg, max_steps=3, checkpoint_dir=checkpoint_dir)

    for key in ("task_loss", "load_loss", "sep_loss", "perplexity", "routing_cv",
                "active_flop_fraction", "val_wikitext_perplexity", "mean_centroid_distance"):
        assert key in metrics, f"missing {key}"

    assert (Path(checkpoint_dir) / "final.pt").exists()


def test_train_one_run_switch_router_has_no_centroid_distance(tmp_path, monkeypatch):
    monkeypatch.setattr(run_sweep, "build_dataloaders", _fake_build_dataloaders)
    cfg = _tiny_cfg(router="switch")
    checkpoint_dir = str(tmp_path / "run2")

    metrics = run_sweep.train_one_run(cfg, max_steps=2, checkpoint_dir=checkpoint_dir)
    assert "mean_centroid_distance" not in metrics


def test_dry_run_does_not_touch_wandb_sweep_or_agent(tmp_path, monkeypatch):
    # main() calls the real load_env() first; spy on it (rather than a bare
    # no-op) so this test both proves the wiring is there AND never loads
    # this repo's actual .env (pytest's cwd is the repo root) or mutates the
    # real process environment with real secrets.
    load_env_calls = []
    monkeypatch.setattr(run_sweep, "load_env", lambda: load_env_calls.append(1))
    monkeypatch.setattr(run_sweep, "build_dataloaders", _fake_build_dataloaders)
    monkeypatch.setattr(run_sweep, "load_base_config", lambda config_name: _tiny_cfg())

    def _fail(*a, **k):
        raise AssertionError("wandb.sweep/agent should not be called in --dry_run")

    monkeypatch.setattr(run_sweep.wandb, "sweep", _fail)
    monkeypatch.setattr(run_sweep.wandb, "agent", _fail)
    monkeypatch.setattr(
        sys, "argv", ["run_sweep.py", "--dry_run", "--checkpoint_dir", str(tmp_path / "ckpt")]
    )

    run_sweep.main()  # should complete without raising
    assert (tmp_path / "ckpt" / "dry_run" / "final.pt").exists()
    assert load_env_calls == [1]


def test_main_without_dry_run_wires_wandb_sweep_and_agent(tmp_path, monkeypatch):
    load_env_calls = []
    monkeypatch.setattr(run_sweep, "load_env", lambda: load_env_calls.append(1))
    monkeypatch.setattr(run_sweep, "load_base_config", lambda config_name: _tiny_cfg())

    calls = {}

    def fake_sweep(sweep_config, project):
        calls["sweep_config"] = sweep_config
        calls["project"] = project
        return "fake-sweep-id"

    def fake_agent(sweep_id, function):
        calls["sweep_id"] = sweep_id
        calls["function"] = function

    monkeypatch.setattr(run_sweep.wandb, "sweep", fake_sweep)
    monkeypatch.setattr(run_sweep.wandb, "agent", fake_agent)
    monkeypatch.setattr(
        sys, "argv", ["run_sweep.py", "--checkpoint_dir", str(tmp_path / "ckpt")]
    )

    run_sweep.main()

    assert calls["sweep_config"] == run_sweep.SWEEP_CONFIG
    assert calls["sweep_id"] == "fake-sweep-id"
    assert callable(calls["function"])
    assert load_env_calls == [1]


def test_make_sweep_run_fn_reads_wandb_config_and_calls_train_one_run(tmp_path, monkeypatch):
    base_cfg = _tiny_cfg()
    captured = {}

    def fake_train_one_run(cfg, max_steps, checkpoint_dir):
        captured["cfg"] = cfg
        captured["max_steps"] = max_steps
        captured["checkpoint_dir"] = checkpoint_dir
        return {}

    class FakeWandb:
        config = {"router": "cosine", "n_experts": 8, "top_k": 1, "nu_sep": 0.1}

        @staticmethod
        def init(**kwargs):
            pass

    monkeypatch.setattr(run_sweep, "train_one_run", fake_train_one_run)
    monkeypatch.setattr(run_sweep, "wandb", FakeWandb)

    run_fn = run_sweep.make_sweep_run_fn(base_cfg, max_steps=10, checkpoint_root=str(tmp_path))
    run_fn()

    assert captured["cfg"].model.router == "cosine"
    assert captured["cfg"].routing.n_experts == 8
    assert captured["max_steps"] == 10
    assert "router-cosine" in captured["checkpoint_dir"]
