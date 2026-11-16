"""Distributed launch and data-partitioning helpers for QuanTopo training."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TypeVar

import torch
import torch.distributed as distributed
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DistributedSampler

ModuleT = TypeVar("ModuleT", bound=nn.Module)


def global_to_local_batch(global_batch: int, world_size: int) -> int:
    """Convert a configured global batch into an exact per-rank batch."""

    if global_batch < 1 or world_size < 1:
        raise ValueError("global batch and world size must be positive")
    if global_batch % world_size:
        raise ValueError(
            f"global batch {global_batch} must be divisible by world size {world_size}"
        )
    return global_batch // world_size


@dataclass
class DistributedContext:
    """Process-local view of a torchrun launch."""

    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    owns_process_group: bool

    @classmethod
    def from_environment(cls, requested_device: str) -> DistributedContext:
        names = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
        values = {name: os.environ.get(name) for name in names}
        present = [value is not None for value in values.values()]
        if any(present) and not all(present):
            missing = ", ".join(name for name, value in values.items() if value is None)
            raise ValueError(f"incomplete distributed launch environment; missing {missing}")

        if not any(present):
            return cls(0, 1, 0, torch.device(requested_device), False)

        try:
            rank = int(values["RANK"] or "")
            world_size = int(values["WORLD_SIZE"] or "")
            local_rank = int(values["LOCAL_RANK"] or "")
        except ValueError as error:
            raise ValueError("distributed launch ranks must be integers") from error
        if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
            raise ValueError("invalid distributed rank, local rank, or world size")

        requested = torch.device(requested_device)
        if requested.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available")
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = requested
            backend = "gloo"

        owns_process_group = False
        if world_size > 1:
            if not distributed.is_available():
                raise RuntimeError("torch.distributed is unavailable")
            if not distributed.is_initialized():
                distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)
                owns_process_group = True
            elif distributed.get_rank() != rank or distributed.get_world_size() != world_size:
                raise RuntimeError("initialized process group does not match launch environment")
        return cls(rank, world_size, local_rank, device, owns_process_group)

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    def barrier(self) -> None:
        if self.is_distributed and distributed.is_initialized():
            distributed.barrier()

    def close(self) -> None:
        if self.owns_process_group and distributed.is_initialized():
            distributed.destroy_process_group()
            self.owns_process_group = False


def make_sampler(
    dataset: Dataset[object],
    context: DistributedContext,
    seed: int,
) -> DistributedSampler[object] | None:
    if not context.is_distributed:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=True,
        seed=seed,
        drop_last=False,
    )


def wrap_ddp(module: ModuleT, context: DistributedContext) -> nn.Module:
    if not context.is_distributed:
        return module
    if context.device.type == "cuda":
        return DistributedDataParallel(
            module,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
        )
    return DistributedDataParallel(module)
