from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import subprocess
import uuid
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path

import torch

from miniscale.config import TrainConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def manifest_sha256(config: TrainConfig) -> str:
    return hashlib.sha256((Path(config.data.directory) / "manifest.json").read_bytes()).hexdigest()


def hardware_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return platform.machine() or platform.processor() or "unknown-cpu"


def digest_files(root: str | Path, relative_paths: list[str]) -> str:
    base = Path(root)
    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        path = base / relative
        digest.update(relative.encode() + b"\0")
        if not path.exists() and not path.is_symlink():
            digest.update(b"<deleted>\0")
            continue
        digest.update(str(stat.S_IMODE(path.lstat().st_mode)).encode() + b"\0")
        content = os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _is_excluded(path: Path, excluded_roots: list[Path]) -> bool:
    resolved = path.resolve()
    return any(resolved == root or root in resolved.parents for root in excluded_roots)


def _fallback_source_paths(root: Path, excluded_roots: list[Path]) -> list[str]:
    allowed_suffixes = {".py", ".yaml", ".yml", ".toml", ".sh", ".md"}
    paths = []
    for path in root.rglob("*"):
        if _is_excluded(path, excluded_roots) or any(part in {".git", "__pycache__"} for part in path.parts):
            continue
        if path.is_file() and (path.suffix in allowed_suffixes or path.name == "Makefile"):
            paths.append(str(path.relative_to(root)))
    return paths


def _working_tree_digest(
    *,
    repo_root: str | Path = REPOSITORY_ROOT,
    excluded_roots: list[str | Path] | None = None,
) -> str:
    root = Path(repo_root).resolve()
    exclusions = [Path(path).resolve() for path in (excluded_roots or [])]
    if root in exclusions:
        raise ValueError("cannot exclude the repository root from source identity")
    try:
        raw = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        )
        paths = [
            item.decode() for item in raw.split(b"\0") if item
            and not _is_excluded(root / item.decode(), exclusions)
        ]
    except (OSError, subprocess.CalledProcessError):
        paths = _fallback_source_paths(root, exclusions)
    if not paths:
        raise RuntimeError(f"no source files available for provenance under {root}")
    return digest_files(root, paths)


def experiment_key(
    config: TrainConfig,
    hardware: str,
    *,
    source_state_digest: str | None = None,
) -> str:
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
        "source_digest": source_state_digest or _working_tree_digest(),
    }
    encoded = json.dumps(controlled, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_state(*, excluded_roots: list[str | Path] | None = None) -> tuple[str, bool, str]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(REPOSITORY_ROOT), "status", "--porcelain"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip())
        return commit, dirty, _working_tree_digest(excluded_roots=excluded_roots)
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", True, _working_tree_digest(excluded_roots=excluded_roots)


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
    excluded = [root, Path(config.data.directory)]
    commit, dirty, source_digest = _source_state(excluded_roots=excluded)
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
        "source_digest": source_digest,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "experiment_key": experiment_key(config, hardware, source_state_digest=source_digest),
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
    payload = {
        "format_version": 1,
        "run_uuid": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        **identity,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path
