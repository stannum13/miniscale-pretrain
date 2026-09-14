from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import torch


def mfu(model_flops: float, step_seconds: float, gpu_count: int, peak_tflops_per_gpu: float) -> float | None:
    if peak_tflops_per_gpu <= 0:
        return None
    return model_flops / step_seconds / (gpu_count * peak_tflops_per_gpu * 1e12)


def scaling_efficiency(single_gpu_tokens_per_second: float, distributed_tokens_per_second: float, gpu_count: int) -> float:
    if gpu_count <= 0 or single_gpu_tokens_per_second <= 0:
        raise ValueError("GPU count and baseline throughput must be positive")
    return distributed_tokens_per_second / (gpu_count * single_gpu_tokens_per_second)


@dataclass
class BenchmarkRecord:
    run_id: str
    model: str
    gpu_count: int
    strategy: str
    micro_batch_size: int
    global_batch_size: int
    gradient_accumulation_steps: int
    sequence_length: int
    measured_steps: int
    step_time_ms: float
    tokens_per_second: float
    tokens_per_second_per_gpu: float
    scaling_efficiency: float | None
    peak_memory_bytes: int
    mfu: float | None
    forward_ms: float
    backward_ms: float
    optimizer_ms: float
    communication_ms: float | None
    loss: float
    gradient_norm: float
    device_type: str
    backend: str
    hardware: str
    torch_version: str
    cuda_version: str | None
    experiment_key: str


def write_jsonl(path: str | Path, record: BenchmarkRecord | dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(record) if isinstance(record, BenchmarkRecord) else record
    line = json.dumps(payload, sort_keys=True) + "\n"
    descriptor = os.open(destination, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, line.encode())
    finally:
        os.close(descriptor)


class StepTimer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.values: dict[str, float] = {}
        self._events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if self.device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            self._events.setdefault(name, []).append((start, end))
        else:
            start_time = time.perf_counter()
            yield
            self.values[name] = self.values.get(name, 0.0) + (time.perf_counter() - start_time) * 1000

    def finish(self) -> dict[str, float]:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self.values.update({
                name: sum(start.elapsed_time(end) for start, end in events)
                for name, events in self._events.items()
            })
        return dict(self.values)
