from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import torch

from miniscale.config import TrainConfig


def manifest_sha256(config: TrainConfig) -> str:
    return hashlib.sha256((Path(config.data.directory) / "manifest.json").read_bytes()).hexdigest()


def hardware_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return platform.machine() or platform.processor() or "unknown-cpu"


def experiment_key(config: TrainConfig, hardware: str) -> str:
    controlled = {
        "model": asdict(config.model),
        "sequence_length": config.sequence_length,
        "micro_batch_size": config.micro_batch_size,
        "global_batch_size": config.global_batch_size,
        "bf16": config.bf16,
        "compile": config.compile,
        "dataset_manifest_sha256": manifest_sha256(config),
        "hardware": hardware,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    encoded = json.dumps(controlled, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
