from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
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
        "seed": config.seed,
        "data_seed": config.data.seed,
        "max_steps": config.max_steps,
        "warmup_steps": config.warmup_steps,
        "measured_steps": config.measured_steps,
        "learning_rate": config.learning_rate,
        "min_learning_rate": config.min_learning_rate,
        "weight_decay": config.weight_decay,
        "beta1": config.beta1,
        "beta2": config.beta2,
        "grad_clip": config.grad_clip,
        "dataset_manifest_sha256": manifest_sha256(config),
        "hardware": hardware,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    encoded = json.dumps(controlled, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_state() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _normalized_config(config: TrainConfig) -> dict:
    payload = asdict(config)
    payload["resume"] = None
    return payload


def prepare_run_directory(
    run_root: str | Path,
    config: TrainConfig,
    *,
    device_type: str,
    backend: str,
    hardware: str,
    world_size: int,
    is_resume: bool,
) -> Path:
    root = Path(run_root)
    path = root / "run.json"
    commit, dirty = _source_state()
    dataset_manifest = json.loads((Path(config.data.directory) / "manifest.json").read_text(encoding="utf-8"))
    identity = {
        "config": _normalized_config(config),
        "dataset_manifest_sha256": manifest_sha256(config),
        "dataset_manifest": dataset_manifest,
        "device_type": device_type,
        "backend": backend,
        "hardware": hardware,
        "world_size": world_size,
        "git_commit": commit,
        "git_dirty": dirty,
    }
    if path.exists():
        if not is_resume:
            raise FileExistsError(f"run already exists; use a new --run-id or --resume: {root}")
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable = {key: existing.get(key) for key in identity}
        if comparable != identity:
            raise ValueError("existing run manifest does not match resume configuration or provenance")
        return path
    if is_resume:
        raise ValueError(f"run manifest is missing for resume: {path}")
    root.mkdir(parents=True, exist_ok=False)
    payload = {"format_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), **identity}
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path
