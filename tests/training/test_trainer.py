"""Tests for Trainer: accumulation, checkpointing, evaluation, and logging."""

import torch
from torch.utils.data import DataLoader, Dataset

from diffusion_moe.data.dataset import IGNORE_INDEX
from diffusion_moe.models.moe_model import DiffusionMoETransformer
from diffusion_moe.routing.separation import landmark_scale
from diffusion_moe.training import trainer as trainer_module
from diffusion_moe.training.optimizer import build_optimizer, build_scheduler
from diffusion_moe.training.trainer import Trainer

VOCAB_SIZE = 50
SEQ_LEN = 8
D_MODEL = 16


class _TinyLMDataset(Dataset):
    def __init__(self, n_examples: int = 16, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(0, VOCAB_SIZE, (n_examples, SEQ_LEN), generator=g)

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.input_ids[idx]
        labels = torch.cat([ids[1:], torch.tensor([IGNORE_INDEX])])
        attention_mask = torch.ones(SEQ_LEN, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": attention_mask, "labels": labels}


def _make_model(layers_to_replace=(), **overrides):
    kwargs = dict(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=2,
        n_heads=4,
        max_seq_len=SEQ_LEN,
        n_experts=4,
        top_k=2,
        n_components=3,
        n_landmarks=8,
        diffusion_t=2,
        centroid_refresh_steps=1000,  # avoid repeated KMeans refits during short tests
        layers_to_replace=list(layers_to_replace),
    )
    kwargs.update(overrides)
    return DiffusionMoETransformer(**kwargs)


def _make_trainer(
    tmp_path, model=None, layers_to_replace=(), training_overrides=None, device_override=None
):
    model = model or _make_model(layers_to_replace=layers_to_replace)
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.0)
    # warmup_steps=0: LR is at its cosine peak from step 0 (warmup_steps=1 would
    # make the very first optimizer.step() a no-op, since lr_lambda(0) == 0).
    scheduler = build_scheduler(optimizer, warmup_steps=0, total_steps=100)

    train_loader = DataLoader(_TinyLMDataset(n_examples=16, seed=0), batch_size=4)
    val_loader = DataLoader(_TinyLMDataset(n_examples=8, seed=1), batch_size=4)

    training_cfg = {
        "grad_accum_steps": 1,
        "grad_clip": 1.0,
        "checkpoint_steps": 5,
        "eval_steps": 5,
        "log_steps": 1,
        "precision": "fp32",
    }
    if training_overrides:
        training_cfg.update(training_overrides)

    config = {
        "training": training_cfg,
        "routing": {"mu_load": 0.01, "nu_sep": 0.05},
        "wandb": {"project": "test-project"},
    }
    return Trainer(
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        config,
        device=device_override or "cpu",
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )


def test_dense_model_train_step_updates_step_and_params(tmp_path):
    t = _make_trainer(tmp_path, layers_to_replace=[])
    before = {n: p.clone() for n, p in t.model.named_parameters()}

    metrics = t.train_step([next(iter(t.train_loader))])

    assert t.step == 1
    assert "loss" in metrics and "task_loss" in metrics
    changed = any(not torch.equal(before[n], p) for n, p in t.model.named_parameters())
    assert changed


def test_gradient_accumulation_produces_one_optimizer_step_per_call(tmp_path):
    t = _make_trainer(tmp_path, layers_to_replace=[], training_overrides={"grad_accum_steps": 3})
    batches = [next(iter(t.train_loader)) for _ in range(3)]
    t.train_step(batches)
    assert t.step == 1


def test_evaluate_returns_finite_positive_perplexity(tmp_path):
    t = _make_trainer(tmp_path, layers_to_replace=[])
    ppl = t.evaluate()
    assert ppl > 0
    assert ppl == ppl  # not NaN
    assert ppl != float("inf")


def test_evaluate_leaves_model_in_original_mode(tmp_path):
    t = _make_trainer(tmp_path, layers_to_replace=[])
    t.model.train()
    t.evaluate()
    assert t.model.training

    t.model.eval()
    t.evaluate()
    assert not t.model.training


def test_train_creates_checkpoints_and_best_checkpoint(tmp_path):
    t = _make_trainer(tmp_path, layers_to_replace=[])
    t.train(max_steps=6)

    assert t.step == 6
    assert (t.checkpoint_dir / "best.pt").exists()
    assert (t.checkpoint_dir / "step_5.pt").exists()


def test_resume_from_checkpoint_restores_step_and_weights(tmp_path):
    t = _make_trainer(tmp_path, layers_to_replace=[])
    t.train(max_steps=3)
    saved_step = t.step
    t.save_checkpoint(t.checkpoint_dir / "manual.pt")

    t2 = _make_trainer(tmp_path, layers_to_replace=[])
    t2.load_checkpoint(t.checkpoint_dir / "manual.pt")

    assert t2.step == saved_step
    assert t2.best_val_ppl == t.best_val_ppl
    for (_, p1), (_, p2) in zip(t.model.named_parameters(), t2.model.named_parameters()):
        assert torch.equal(p1, p2)


def test_logs_to_stdout_when_wandb_not_configured(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    t = _make_trainer(tmp_path, layers_to_replace=[])
    assert not t.use_wandb

    t._log({"loss": 1.234}, 7)
    captured = capsys.readouterr()
    assert "step=7" in captured.out
    assert "loss=1.2340" in captured.out


def test_uses_wandb_when_api_key_set_and_wandb_available(tmp_path, monkeypatch):
    calls = {"init": [], "log": []}

    class FakeWandb:
        @staticmethod
        def init(**kwargs):
            calls["init"].append(kwargs)

        @staticmethod
        def log(metrics, step=None):
            calls["log"].append((metrics, step))

    monkeypatch.setattr(trainer_module, "wandb", FakeWandb)
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")

    t = _make_trainer(tmp_path, layers_to_replace=[])
    assert t.use_wandb
    assert len(calls["init"]) == 1

    t._log({"loss": 1.0}, 0)
    assert calls["log"] == [({"loss": 1.0}, 0)]


def test_does_not_use_wandb_without_api_key_even_if_wandb_available(tmp_path, monkeypatch):
    class FakeWandb:
        @staticmethod
        def init(**kwargs):
            raise AssertionError("wandb.init should not be called without WANDB_API_KEY")

    monkeypatch.setattr(trainer_module, "wandb", FakeWandb)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    t = _make_trainer(tmp_path, layers_to_replace=[])
    assert not t.use_wandb


def test_moe_layer_produces_nonzero_aux_losses(tmp_path):
    """Integration check: with a real MoE layer, total_loss's aux components
    should be nontrivial rather than the empty-router_outputs zero default."""
    t = _make_trainer(tmp_path, layers_to_replace=[0])
    metrics = t.train_step([next(iter(t.train_loader))])
    assert metrics["load_loss"] != 0.0 or metrics["sep_loss"] != 0.0


def test_train_step_exposes_per_layer_load_loss_for_multiple_moe_layers(tmp_path):
    """With two DiffusionMoELayers active, train_step's returned metrics must
    break load_loss down per layer, not just the aggregate mean -- otherwise
    a full collapse at one layer masked by balance at another is invisible
    to anything watching training (the actual motivation: run all layers as
    diffusion-MoE and see, per layer, what happens)."""
    t = _make_trainer(tmp_path, layers_to_replace=[0, 1])
    metrics = t.train_step([next(iter(t.train_loader))])

    assert "load_loss/layer_0" in metrics
    assert "load_loss/layer_1" in metrics
    assert "load_loss/layer_2" not in metrics  # layer 2 wasn't replaced
    manual_mean = (metrics["load_loss/layer_0"] + metrics["load_loss/layer_1"]) / 2
    assert abs(metrics["load_loss"] - manual_mean) < 1e-5


def test_train_step_clips_centroid_norms_after_optimizer_step(tmp_path):
    """The actual gap this closes: ExpertCentroids.clip_norm_() existed but
    was only ever called by scripts/pilot_finetune.py's own bespoke training
    loop -- the production Trainer never called it at all, so real training
    through Trainer/DiffusionMoETransformer (unlike the pilot) had no
    protection against centroid_separation_loss's unbounded-growth gradient
    (see reports/phase1_findings_report.md §3.4, bug 1; recommendation 4
    flagged this fix as unverified in the production path). Reproducing the
    full multi-hundred-step organic blowup through Adam-optimized training on
    tiny synthetic data isn't practical (gradient clipping already bounds
    single-step movement, and the existing test_clip_norm_prevents_the_
    runaway_growth_a_real_pilot_run_hit in test_centroids.py already proves
    clip_norm_ itself works) -- this instead directly tests the wiring gap:
    a deliberately-inflated centroid must come back down within bounds after
    one real train_step.
    """
    t = _make_trainer(tmp_path, layers_to_replace=[0])
    # One real step first, so ndm.psi_landmarks_ is populated (fit during the
    # forward pass) before we inflate anything.
    t.train_step([next(iter(t.train_loader))])

    block = t.model.blocks[0]
    scale = landmark_scale(
        torch.from_numpy(block.ndm.psi_landmarks_).to(block.centroids.centroids.dtype)
    )
    with torch.no_grad():
        block.centroids.centroids[0] = torch.full_like(block.centroids.centroids[0], 1000.0)
    assert block.centroids.centroids.norm(dim=-1).max() > 50 * scale  # sanity: really is huge

    t.train_step([next(iter(t.train_loader))])

    assert block.centroids.centroids.norm(dim=-1).max() <= t.centroid_max_radius_factor * scale * 1.5


def test_dense_model_train_step_does_not_error_without_centroids(tmp_path):
    """_clip_centroid_norms must be a safe no-op for dense TransformerBlocks
    (no `centroids`/`ndm` attributes at all) and for the non-diffusion router
    variants -- it shouldn't assume every block is a DiffusionMoELayer."""
    t = _make_trainer(tmp_path, layers_to_replace=[])
    t.train_step([next(iter(t.train_loader))])  # must not raise


def test_centroid_max_radius_factor_read_from_routing_config(tmp_path):
    t = _make_trainer(
        tmp_path,
        layers_to_replace=[0],
        training_overrides=None,
    )
    t.config["routing"]["centroid_max_radius_factor"] = 7.5
    t2 = Trainer(
        t.model,
        t.optimizer,
        t.scheduler,
        t.train_loader,
        t.val_loader,
        t.config,
        device="cpu",
        checkpoint_dir=str(tmp_path / "checkpoints2"),
    )
    assert t2.centroid_max_radius_factor == 7.5


def test_grad_clipping_bounds_gradient_norm(tmp_path):
    t = _make_trainer(tmp_path, layers_to_replace=[], training_overrides={"grad_clip": 0.01})
    t.train_step([next(iter(t.train_loader))])
    total_norm = torch.norm(
        torch.stack(
            [p.grad.detach().norm() for p in t.model.parameters() if p.grad is not None]
        )
    )
    # clipping isn't exact post-hoc (params already stepped), but the applied
    # gradient should have been rescaled down to (approximately) grad_clip
    assert total_norm < 1.0


# --- DDP-awareness -----------------------------------------------------------
#
# There's no real multi-GPU CUDA hardware in this environment, so none of
# these exercise an actual torch.distributed process group or real
# DistributedDataParallel. Instead they verify Trainer's OWN decision logic
# (when to wrap, how raw_model unwraps, who does I/O) by substituting a fake
# DDP wrapper — the same approach already used for wandb in this file.
# Real multi-GPU behavior (gradient all-reduce during backward, actual NCCL
# communication) is PyTorch's own responsibility, not Trainer's.


class _FakeDDP(torch.nn.Module):
    """Stands in for torch.nn.parallel.DistributedDataParallel: a thin
    pass-through wrapper exposing the same .module attribute and forward()
    delegation real DDP provides, without needing a real process group."""

    def __init__(self, module, device_ids=None):
        super().__init__()
        self.module = module
        self.device_ids = device_ids

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _make_ddp_trainer(tmp_path, monkeypatch, local_rank=0):
    """A Trainer built as if launched under torchrun with device="cuda", but
    with DistributedDataParallel faked and Module.to() short-circuited so it
    never touches real CUDA — see the module-level comment above."""
    monkeypatch.setattr(trainer_module, "is_distributed", lambda: True)
    monkeypatch.setattr(trainer_module, "get_local_rank", lambda: local_rank)
    monkeypatch.setattr(trainer_module, "DistributedDataParallel", _FakeDDP)
    monkeypatch.setattr(torch.nn.Module, "to", lambda self, *a, **k: self)

    return _make_trainer(tmp_path, layers_to_replace=[], device_override="cuda")


def test_ddp_wrapping_is_applied_when_distributed_and_cuda(tmp_path, monkeypatch):
    t = _make_ddp_trainer(tmp_path, monkeypatch, local_rank=2)
    assert t.is_ddp is True
    assert isinstance(t.model, _FakeDDP)
    assert t.device == "cuda:2"


def test_ddp_not_applied_when_device_is_not_cuda(tmp_path, monkeypatch):
    """MPS has no DDP backend — even if somehow launched with WORLD_SIZE>1,
    a non-CUDA device must never be wrapped."""
    monkeypatch.setattr(trainer_module, "is_distributed", lambda: True)
    monkeypatch.setattr(trainer_module, "DistributedDataParallel", _FakeDDP)
    t = _make_trainer(tmp_path, layers_to_replace=[], device_override="cpu")
    assert t.is_ddp is False
    assert not isinstance(t.model, _FakeDDP)


def test_raw_model_unwraps_ddp(tmp_path, monkeypatch):
    t = _make_ddp_trainer(tmp_path, monkeypatch)
    assert t.raw_model is t.model.module
    assert not isinstance(t.raw_model, _FakeDDP)


def test_raw_model_is_model_itself_when_not_ddp(tmp_path):
    t = _make_trainer(tmp_path, layers_to_replace=[])
    assert t.raw_model is t.model


def test_checkpoint_saved_through_raw_model_has_no_module_prefix(tmp_path, monkeypatch):
    t = _make_ddp_trainer(tmp_path, monkeypatch)
    path = t.checkpoint_dir / "ddp.pt"
    t.save_checkpoint(path)

    state = torch.load(path, weights_only=False)
    assert not any(key.startswith("module.") for key in state["model"])


def test_load_checkpoint_restores_into_raw_model_when_ddp(tmp_path, monkeypatch):
    t = _make_ddp_trainer(tmp_path, monkeypatch)
    path = t.checkpoint_dir / "ddp.pt"
    t.save_checkpoint(path)  # tensors are genuinely CPU-resident (Module.to was faked as a no-op)

    t2 = _make_ddp_trainer(tmp_path, monkeypatch)
    # load_checkpoint maps onto t2.device, which is the fake "cuda:N" string —
    # real torch.load would then genuinely try (and fail) to touch CUDA. Force
    # map_location back to "cpu", matching where the tensors actually live;
    # this environment has no real CUDA to test the true remapping against.
    original_load = torch.load
    monkeypatch.setattr(
        trainer_module.torch,
        "load",
        lambda p, map_location=None, weights_only=False: original_load(
            p, map_location="cpu", weights_only=weights_only
        ),
    )
    t2.load_checkpoint(path)
    for (_, p1), (_, p2) in zip(
        t.raw_model.named_parameters(), t2.raw_model.named_parameters()
    ):
        assert torch.equal(p1, p2)


def test_save_checkpoint_is_noop_on_non_main_rank(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer_module, "is_main_process", lambda: False)
    t = _make_trainer(tmp_path, layers_to_replace=[])
    path = t.checkpoint_dir / "should_not_exist.pt"
    t.save_checkpoint(path)
    assert not path.exists()


def test_log_is_noop_on_non_main_rank(tmp_path, monkeypatch, capsys):
    t = _make_trainer(tmp_path, layers_to_replace=[])
    monkeypatch.setattr(trainer_module, "is_main_process", lambda: False)
    t._log({"loss": 1.0}, 0)
    assert capsys.readouterr().out == ""


def test_train_step_metrics_pass_through_cross_rank_averaging(tmp_path, monkeypatch):
    """Verifies train_step's returned metrics go through all_reduce_mean
    (already unit-tested for correctness in tests/utils/test_device.py) —
    here we just confirm the wiring actually calls it, via a spy that
    records every (key, value) it's asked to average."""
    t = _make_trainer(tmp_path, layers_to_replace=[])
    calls = []

    def spy(value, device):
        calls.append(value)
        return value

    monkeypatch.setattr(trainer_module, "all_reduce_mean", spy)
    metrics = t.train_step([next(iter(t.train_loader))])

    assert len(calls) == len(metrics) == 4  # loss, task_loss, load_loss, sep_loss
    assert set(calls) <= {metrics[k] for k in metrics}
