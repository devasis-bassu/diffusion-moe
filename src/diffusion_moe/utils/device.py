"""Device and distributed-environment detection.

Local runs use the single device a machine actually exposes (CUDA if
present, else Apple's MPS, else CPU — there is no multi-MPS: Apple Silicon
exposes exactly one GPU per machine in PyTorch today). Remote runs on
vast.ai use multiple CUDA GPUs via torch.distributed + DistributedDataParallel,
launched with torchrun (which sets RANK/WORLD_SIZE/LOCAL_RANK env vars —
that's how we detect we're running distributed at all).
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def get_device() -> str:
    """Picks the single best local device: cuda > mps > cpu.

    Under a torchrun-launched distributed job this returns "cuda" (the
    per-rank GPU is selected separately via LOCAL_RANK — see get_local_rank).
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_distributed() -> bool:
    """True when launched via torchrun (or any launcher setting the standard
    RANK/WORLD_SIZE/LOCAL_RANK env vars) with more than one process."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    """True on the single process that should do checkpointing, wandb
    logging, and console output — always true outside a distributed run."""
    return get_rank() == 0


def setup_distributed(backend: str = "nccl") -> None:
    """Initialises the default process group if launched distributed and not
    already initialised. No-ops in a single-process run. MPS has no DDP
    backend, so this is CUDA-only in practice (vast.ai); calling it on a
    non-distributed local MPS/CPU run is always a safe no-op.
    """
    if not is_distributed() or dist.is_initialized():
        return
    dist.init_process_group(backend=backend)
    torch.cuda.set_device(get_local_rank())


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_mean(value: float, device: str) -> float:
    """Averages a scalar metric across all ranks — used for logging losses
    that are otherwise only computed on each rank's local micro-batch.
    No-ops (returns value unchanged) outside a distributed run."""
    if not (is_distributed() and dist.is_initialized()):
        return value
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / get_world_size()).item()
