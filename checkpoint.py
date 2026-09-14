from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from distributed.runtime import DistributedContext
from distributed.strategies import unwrap_model
from miniscale.config import TrainConfig


@dataclass(frozen=True)
class ResumeState:
    step: int
    sample_cursor: int


def _fingerprint(config: TrainConfig) -> str:
    payload = asdict(config)
    payload["resume"] = None
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _dataset_manifest_hash(config: TrainConfig) -> str:
    manifest = Path(config.data.directory) / "manifest.json"
    if not manifest.is_file():
        raise ValueError(f"dataset manifest is missing: {manifest}")
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _is_fsdp(model: nn.Module) -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
    except ImportError:
        return False
    return isinstance(model, FullyShardedDataParallel)


def _state_for_save(model: nn.Module, optimizer: torch.optim.Optimizer) -> tuple[dict, dict]:
    if not _is_fsdp(model):
        return unwrap_model(model).state_dict(), optimizer.state_dict()
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardedOptimStateDictConfig,
        ShardedStateDictConfig,
        StateDictType,
    )
    with FSDP.state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
        ShardedStateDictConfig(offload_to_cpu=True),
        ShardedOptimStateDictConfig(offload_to_cpu=True),
    ):
        return model.state_dict(), FSDP.optim_state_dict(model, optimizer)


def _load_states(model: nn.Module, optimizer: torch.optim.Optimizer, model_state: dict, optimizer_state: dict) -> None:
    if not _is_fsdp(model):
        unwrap_model(model).load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        return
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardedOptimStateDictConfig,
        ShardedStateDictConfig,
        StateDictType,
    )
    with FSDP.state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
        ShardedStateDictConfig(offload_to_cpu=True),
        ShardedOptimStateDictConfig(offload_to_cpu=True),
    ):
        model.load_state_dict(model_state)
        optimizer.load_state_dict(FSDP.optim_state_dict_to_load(model, optimizer, optimizer_state))


def save_checkpoint(
    root: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    context: DistributedContext,
    *,
    step: int,
    sample_cursor: int,
    config: TrainConfig,
) -> Path:
    destination = Path(root) / f"step-{step:08d}"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "COMPLETE").exists():
        raise FileExistsError(f"checkpoint already complete: {destination}")
    model_state, optimizer_state = _state_for_save(model, optimizer)
    payload = {
        "model": model_state,
        "optimizer": optimizer_state,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng": _rng_state(),
        "step": step,
        "sample_cursor": sample_cursor,
    }
    rank_file = destination / f"rank-{context.rank:05d}.pt"
    temporary = rank_file.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, rank_file)
    context.barrier()
    if context.is_main:
        metadata = {
            "format_version": 1,
            "world_size": context.world_size,
            "strategy": config.strategy,
            "config_sha256": _fingerprint(config),
            "dataset_manifest_sha256": _dataset_manifest_hash(config),
            "step": step,
            "sample_cursor": sample_cursor,
        }
        metadata_temp = destination / "metadata.json.tmp"
        metadata_temp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(metadata_temp, destination / "metadata.json")
        (destination / "COMPLETE").touch()
    context.barrier()
    return destination


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    context: DistributedContext,
    *,
    config: TrainConfig,
) -> ResumeState:
    source = Path(path)
    if not (source / "COMPLETE").exists():
        raise ValueError(f"checkpoint is incomplete: {source}")
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata["world_size"] != context.world_size:
        raise ValueError("checkpoint world size differs; rank-local RNG/shards are topology-specific")
    if metadata["config_sha256"] != _fingerprint(config):
        raise ValueError("checkpoint configuration fingerprint mismatch")
    if metadata.get("dataset_manifest_sha256") != _dataset_manifest_hash(config):
        raise ValueError("checkpoint dataset manifest fingerprint mismatch")
    # RNG state tensors must remain CPU ByteTensors for torch.set_rng_state.
    # Optimizer/model loaders migrate tensors to their parameter devices.
    payload = torch.load(source / f"rank-{context.rank:05d}.pt", map_location="cpu", weights_only=False)
    _load_states(model, optimizer, payload["model"], payload["optimizer"])
    if scheduler is not None and payload["scheduler"] is not None:
        scheduler.load_state_dict(payload["scheduler"])
    _restore_rng(payload["rng"])
    context.barrier()
    return ResumeState(step=int(payload["step"]), sample_cursor=int(payload["sample_cursor"]))
