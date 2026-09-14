from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 32_768
    hidden_size: int = 768
    num_layers: int = 16
    num_heads: int = 12
    num_kv_heads: int = 12
    intermediate_size: int = 2_048
    max_sequence_length: int = 2_048
    rope_theta: float = 10_000.0
    dropout: float = 0.0
    activation_checkpointing: bool = False

    def estimated_parameters(self) -> int:
        # Tied token embedding + attention Q/K/V/O + gated MLP + norms.
        head_dim = self.hidden_size // self.num_heads
        q = self.hidden_size * self.hidden_size
        kv = 2 * self.hidden_size * self.num_kv_heads * head_dim
        out = self.hidden_size * self.hidden_size
        mlp = 3 * self.hidden_size * self.intermediate_size
        norms = 2 * self.hidden_size
        return self.vocab_size * self.hidden_size + self.num_layers * (q + kv + out + mlp + norms) + self.hidden_size

    def validate(self) -> None:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.max_sequence_length < 2:
            raise ValueError("max_sequence_length must be at least 2")


@dataclass
class DataConfig:
    directory: str = "data/processed/fineweb-edu"
    seed: int = 17
    synthetic: bool = False


@dataclass
class TrainConfig:
    run_name: str = "miniscale"
    seed: int = 1337
    strategy: str = "single"
    bf16: bool = True
    compile: bool = False
    sequence_length: int = 1_024
    micro_batch_size: int = 2
    global_batch_size: int = 32
    max_steps: int = 100
    warmup_steps: int = 10
    measured_steps: int = 50
    learning_rate: float = 3.0e-4
    min_learning_rate: float = 3.0e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    checkpoint_interval: int = 50
    log_interval: int = 1
    output_dir: str = "results/runs"
    resume: str | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def gradient_accumulation_steps(self, world_size: int) -> int:
        denominator = self.micro_batch_size * world_size
        if self.global_batch_size % denominator:
            raise ValueError(
                f"global_batch_size={self.global_batch_size} must be divisible by "
                f"micro_batch_size*world_size={denominator}"
            )
        return self.global_batch_size // denominator

    def validate(self, world_size: int = 1) -> None:
        self.model.validate()
        self.gradient_accumulation_steps(world_size)
        if self.strategy not in {"single", "ddp", "fsdp"}:
            raise ValueError("strategy must be single, ddp, or fsdp")
        if self.sequence_length > self.model.max_sequence_length:
            raise ValueError("sequence_length exceeds model.max_sequence_length")
        if min(self.max_steps, self.micro_batch_size, self.global_batch_size) <= 0:
            raise ValueError("steps and batch sizes must be positive")
        if self.warmup_steps + self.measured_steps > self.max_steps:
            raise ValueError("warmup_steps + measured_steps cannot exceed max_steps")


def _construct(data: dict[str, Any]) -> TrainConfig:
    values = dict(data)
    values["model"] = ModelConfig(**values.get("model", {}))
    values["data"] = DataConfig(**values.get("data", {}))
    return TrainConfig(**values)


def load_config(path: str | Path) -> TrainConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    cfg = _construct(raw)
    cfg.validate()
    return cfg
