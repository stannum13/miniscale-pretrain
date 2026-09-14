from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    owns_process_group: bool = False

    @classmethod
    def from_environment(cls) -> "DistributedContext":
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        owns = False
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
            owns = True
        return cls(rank, local_rank, world_size, device, owns)

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.distributed:
            dist.barrier()

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def mean(self, value: torch.Tensor) -> torch.Tensor:
        if self.distributed:
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            value /= self.world_size
        return value

    def close(self) -> None:
        if self.owns_process_group and dist.is_initialized():
            dist.destroy_process_group()


def reset_peak_memory(context: DistributedContext) -> None:
    if context.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(context.device)


def peak_memory_bytes(context: DistributedContext) -> int:
    if context.device.type == "cuda":
        local = torch.tensor(torch.cuda.max_memory_allocated(context.device), device=context.device)
        if context.distributed:
            dist.all_reduce(local, op=dist.ReduceOp.MAX)
        return int(local.item())
    return 0
