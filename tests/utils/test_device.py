"""Tests for device/distributed-environment detection."""

import torch

from diffusion_moe.utils import device as device_module


def test_get_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert device_module.get_device() == "cuda"


def test_get_device_prefers_mps_over_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert device_module.get_device() == "mps"


def test_get_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert device_module.get_device() == "cpu"


def test_is_distributed_false_by_default(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert not device_module.is_distributed()


def test_is_distributed_true_when_world_size_gt_one(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert device_module.is_distributed()


def test_is_distributed_false_when_world_size_is_one(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert not device_module.is_distributed()


def test_rank_helpers_read_env_vars(monkeypatch):
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "8")
    assert device_module.get_rank() == 2
    assert device_module.get_local_rank() == 1
    assert device_module.get_world_size() == 8


def test_rank_helpers_default_to_single_process(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert device_module.get_rank() == 0
    assert device_module.get_local_rank() == 0
    assert device_module.get_world_size() == 1


def test_is_main_process_true_for_rank_zero(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    assert device_module.is_main_process()


def test_is_main_process_false_for_other_ranks(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    assert not device_module.is_main_process()


def test_setup_distributed_is_noop_when_not_distributed(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    def _fail(*a, **k):
        raise AssertionError("init_process_group should not be called")

    monkeypatch.setattr(device_module.dist, "init_process_group", _fail)
    device_module.setup_distributed()  # should not raise


def test_setup_distributed_is_noop_when_already_initialized(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(device_module.dist, "is_initialized", lambda: True)

    def _fail(*a, **k):
        raise AssertionError("init_process_group should not be called again")

    monkeypatch.setattr(device_module.dist, "init_process_group", _fail)
    device_module.setup_distributed()  # should not raise


def test_setup_distributed_initializes_process_group_when_distributed(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(device_module.dist, "is_initialized", lambda: False)

    calls = {}

    def fake_init_process_group(backend):
        calls["backend"] = backend

    monkeypatch.setattr(device_module.dist, "init_process_group", fake_init_process_group)
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda idx: calls.setdefault("cuda_device", idx)
    )

    device_module.setup_distributed(backend="nccl")
    assert calls["backend"] == "nccl"
    assert calls["cuda_device"] == 1


def test_cleanup_distributed_is_noop_when_not_initialized(monkeypatch):
    monkeypatch.setattr(device_module.dist, "is_initialized", lambda: False)

    def _fail(*a, **k):
        raise AssertionError("destroy_process_group should not be called")

    monkeypatch.setattr(device_module.dist, "destroy_process_group", _fail)
    device_module.cleanup_distributed()  # should not raise


def test_all_reduce_mean_is_noop_outside_distributed(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert device_module.all_reduce_mean(3.5, device="cpu") == 3.5


def test_all_reduce_mean_averages_across_ranks(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setattr(device_module.dist, "is_initialized", lambda: True)

    def fake_all_reduce(tensor, op):
        tensor.mul_(4)  # simulate 4 ranks all holding the same value, summed

    monkeypatch.setattr(device_module.dist, "all_reduce", fake_all_reduce)

    result = device_module.all_reduce_mean(2.0, device="cpu")
    assert result == 2.0  # sum(2.0*4)/4 == 2.0
