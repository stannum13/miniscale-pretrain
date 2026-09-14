from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from distributed.runtime import DistributedContext


@dataclass
class CommunicationTimer:
    elapsed_ms: float = 0.0

    def reset(self) -> None:
        self.elapsed_ms = 0.0


def _timed_allreduce(state: CommunicationTimer, bucket):
    started = time.perf_counter()
    future = dist.all_reduce(bucket.buffer(), async_op=True).get_future()

    def complete(result):
        state.elapsed_ms += (time.perf_counter() - started) * 1000
        return result.value()[0].div_(dist.get_world_size())

    return future.then(complete)


def wrap_model(
    model: nn.Module,
    strategy: str,
    context: DistributedContext,
    *,
    bf16: bool,
) -> tuple[nn.Module, CommunicationTimer]:
    timer = CommunicationTimer()
    model = model.to(context.device)
    if strategy == "single":
        if context.world_size != 1:
            raise ValueError("single strategy requires WORLD_SIZE=1")
        return model, timer
    if not dist.is_initialized():
        raise RuntimeError(f"{strategy} requires an initialized process group")
    if strategy == "ddp":
        device_ids = [context.local_rank] if context.device.type == "cuda" else None
        wrapped = DistributedDataParallel(
            model, device_ids=device_ids, gradient_as_bucket_view=True, static_graph=False
        )
        wrapped.register_comm_hook(timer, _timed_allreduce)
        return wrapped, timer
    if strategy == "fsdp":
        if context.device.type != "cuda":
            raise RuntimeError("FSDP full sharding requires CUDA in this project")
        from torch.distributed.fsdp import (
            BackwardPrefetch,
            FullyShardedDataParallel,
            MixedPrecision,
            ShardingStrategy,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from model.transformer import TransformerBlock

        policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})
        mixed = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.bfloat16) if bf16 else None
        wrapped = FullyShardedDataParallel(
            model,
            auto_wrap_policy=policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            forward_prefetch=True,
            limit_all_gathers=True,
            use_orig_params=True,
            device_id=context.device,
        )
        return wrapped, timer
    raise ValueError(f"unknown strategy: {strategy}")


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model
