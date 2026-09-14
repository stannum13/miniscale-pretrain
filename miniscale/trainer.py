from __future__ import annotations

import math
import os
import random
import signal
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from checkpoint import load_checkpoint, save_checkpoint
from data.dataset import TokenShardDataset, ensure_synthetic_dataset
from distributed.runtime import DistributedContext, peak_memory_bytes, reset_peak_memory
from distributed.strategies import wrap_model
from miniscale.config import TrainConfig
from miniscale.metrics import BenchmarkRecord, StepTimer, mfu, write_jsonl
from model.transformer import Transformer, estimate_model_flops


@dataclass
class TrainResult:
    start_step: int
    sample_cursor_start: int
    end_step: int
    losses: list[float]
    stopped_early: bool
    last_checkpoint: Path


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _schedule(config: TrainConfig):
    def multiplier(step: int) -> float:
        if config.warmup_steps and step < config.warmup_steps:
            return (step + 1) / config.warmup_steps
        span = max(1, config.max_steps - config.warmup_steps)
        progress = min(1.0, max(0.0, (step - config.warmup_steps) / span))
        minimum = config.min_learning_rate / config.learning_rate
        return minimum + 0.5 * (1.0 - minimum) * (1.0 + math.cos(math.pi * progress))
    return multiplier


def _maximum(value: float, context: DistributedContext) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=context.device)
    if context.distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _mean(value: float, context: DistributedContext) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=context.device)
    return float(context.mean(tensor).item())


def _latest_or_save(
    checkpoint_root: Path,
    model,
    optimizer,
    scheduler,
    context: DistributedContext,
    completed: int,
    cursor: int,
    config: TrainConfig,
) -> Path:
    path = checkpoint_root / f"step-{completed:08d}"
    if (path / "COMPLETE").exists():
        return path
    return save_checkpoint(checkpoint_root, model, optimizer, scheduler, context,
                           step=completed, sample_cursor=cursor, config=config)


def run_training(
    config: TrainConfig,
    *,
    run_id: str | None = None,
    stop_after: int | None = None,
    crash_after: int | None = None,
) -> TrainResult:
    context = DistributedContext.from_environment()
    config.validate(context.world_size)
    _seed_everything(config.seed)
    if config.data.synthetic:
        ensure_synthetic_dataset(config.data.directory, config.model.vocab_size)
    dataset = TokenShardDataset(config.data.directory, config.sequence_length, config.data.seed)
    raw_model = Transformer(config.model)
    model, communication = wrap_model(raw_model, config.strategy, context, bf16=config.bf16)
    if config.compile:
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay, fused=context.device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _schedule(config))
    accumulation = config.gradient_accumulation_steps(context.world_size)
    start_step = 0
    sample_cursor = 0
    if config.resume:
        resumed = load_checkpoint(config.resume, model, optimizer, scheduler, context, config=config)
        start_step, sample_cursor = resumed.step, resumed.sample_cursor
    sample_cursor_start = sample_cursor
    identifier = run_id or config.run_name
    run_root = Path(config.output_dir) / identifier
    checkpoint_root = run_root / "checkpoints"
    steps_path = run_root / "steps.jsonl"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    reset_peak_memory(context)
    losses: list[float] = []
    measurements: list[dict[str, float]] = []
    last_checkpoint: Path | None = None
    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    old_term = signal.signal(signal.SIGTERM, request_stop)
    try:
        for step in range(start_step, config.max_steps):
            context.barrier()
            context.synchronize()
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            communication.reset()
            timer = StepTimer(context.device)
            local_loss = 0.0
            for micro_step in range(accumulation):
                tokens, _ = dataset.rank_batch(
                    sample_cursor, micro_step, rank=context.rank, world_size=context.world_size,
                    micro_batch_size=config.micro_batch_size,
                )
                tokens = tokens.to(context.device, non_blocking=True)
                synchronize_gradients = micro_step == accumulation - 1
                sync_context = nullcontext() if synchronize_gradients or not hasattr(model, "no_sync") else model.no_sync()
                autocast = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if config.bf16 and context.device.type == "cuda" else nullcontext()
                )
                with sync_context:
                    with timer.phase("forward"):
                        with autocast:
                            output = model(tokens, labels=tokens)
                            if output.loss is None:
                                raise RuntimeError("model did not return training loss")
                            loss = output.loss / accumulation
                    local_loss += float(loss.detach())
                    with timer.phase("backward"):
                        loss.backward()
            with timer.phase("optimizer"):
                if config.strategy == "fsdp" and hasattr(model, "clip_grad_norm_"):
                    gradient_norm = model.clip_grad_norm_(config.grad_clip)
                else:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                scheduler.step()
            phases = timer.finish()
            context.synchronize()
            step_seconds = _maximum(time.perf_counter() - step_started, context)
            loss_value = _mean(local_loss, context)
            grad_value = _mean(float(gradient_norm), context)
            completed = step + 1
            sample_cursor += config.global_batch_size
            losses.append(loss_value)
            tokens_per_step = config.global_batch_size * config.sequence_length
            step_metrics = {
                "step": completed,
                "sample_cursor": sample_cursor,
                "loss": loss_value,
                "gradient_norm": grad_value,
                "step_time_ms": step_seconds * 1000,
                "tokens_per_second": tokens_per_step / step_seconds,
                "tokens_per_second_per_gpu": tokens_per_step / step_seconds / context.world_size,
                "forward_ms": _maximum(phases.get("forward", 0.0), context),
                "backward_ms": _maximum(phases.get("backward", 0.0), context),
                "optimizer_ms": _maximum(phases.get("optimizer", 0.0), context),
                "communication_ms": _maximum(communication.elapsed_ms, context) if config.strategy == "ddp" else None,
                "peak_memory_bytes": peak_memory_bytes(context),
            }
            if context.is_main:
                write_jsonl(steps_path, step_metrics)
            if config.warmup_steps <= step < config.warmup_steps + config.measured_steps:
                measurements.append(step_metrics)
            if completed % config.checkpoint_interval == 0:
                last_checkpoint = _latest_or_save(checkpoint_root, model, optimizer, scheduler,
                                                  context, completed, sample_cursor, config)
            if crash_after is not None and completed >= crash_after:
                context.barrier()
                os._exit(86)
            if stop_requested or (stop_after is not None and completed >= stop_after):
                last_checkpoint = _latest_or_save(checkpoint_root, model, optimizer, scheduler,
                                                  context, completed, sample_cursor, config)
                return TrainResult(start_step, sample_cursor_start, completed, losses, True, last_checkpoint)
        last_checkpoint = _latest_or_save(checkpoint_root, model, optimizer, scheduler,
                                          context, config.max_steps, sample_cursor, config)
        if measurements and context.is_main:
            count = len(measurements)
            average = lambda key: sum(float(row[key]) for row in measurements) / count
            seconds = average("step_time_ms") / 1000
            tokens_per_step = config.global_batch_size * config.sequence_length
            flop_count = estimate_model_flops(model, tokens_per_step, config.sequence_length)
            record = BenchmarkRecord(
                run_id=identifier, model=config.run_name, gpu_count=context.world_size,
                strategy=config.strategy, micro_batch_size=config.micro_batch_size,
                global_batch_size=config.global_batch_size,
                gradient_accumulation_steps=accumulation, sequence_length=config.sequence_length,
                measured_steps=count, step_time_ms=average("step_time_ms"),
                tokens_per_second=tokens_per_step / seconds,
                tokens_per_second_per_gpu=tokens_per_step / seconds / context.world_size,
                scaling_efficiency=None, peak_memory_bytes=max(int(row["peak_memory_bytes"]) for row in measurements),
                mfu=mfu(flop_count, seconds, context.world_size, config.hardware_peak_tflops),
                forward_ms=average("forward_ms"), backward_ms=average("backward_ms"),
                optimizer_ms=average("optimizer_ms"),
                communication_ms=(average("communication_ms") if config.strategy == "ddp" else None),
                loss=average("loss"), gradient_norm=average("gradient_norm"),
            )
            write_jsonl(run_root / "benchmark.jsonl", record)
        return TrainResult(start_step, sample_cursor_start, config.max_steps, losses, False, last_checkpoint)
    finally:
        signal.signal(signal.SIGTERM, old_term)
        context.close()
