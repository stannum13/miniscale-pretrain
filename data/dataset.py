from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def write_token_shards(
    shards: Iterable[Sequence[int]],
    directory: str | Path,
    *,
    dataset: dict[str, str],
    tokenizer: dict[str, str],
) -> Path:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, int | str]] = []
    total = 0
    minimum: int | None = None
    maximum: int | None = None
    for index, tokens in enumerate(shards):
        array = np.asarray(tokens, dtype="<u4")
        if not len(array):
            continue
        path = root / f"shard-{index:05d}.bin"
        temporary = path.with_suffix(".bin.tmp")
        array.tofile(temporary)
        os.replace(temporary, path)
        records.append({"file": path.name, "tokens": int(len(array)), "sha256": _sha256(path)})
        total += len(array)
        shard_min, shard_max = int(array.min()), int(array.max())
        minimum = shard_min if minimum is None else min(minimum, shard_min)
        maximum = shard_max if maximum is None else max(maximum, shard_max)
    if not records:
        raise ValueError("at least one non-empty token shard is required")
    manifest = {
        "format_version": 1,
        "dtype": "uint32-le",
        "dataset": dataset,
        "tokenizer": tokenizer,
        "total_tokens": total,
        "min_token_id": minimum,
        "max_token_id": maximum,
        "shards": records,
    }
    destination = root / "manifest.json"
    _atomic_json(manifest, destination)
    return destination


def ensure_synthetic_dataset(directory: str | Path, vocab_size: int, tokens: int = 16_384) -> Path:
    root = Path(directory)
    manifest = root / "manifest.json"
    if manifest.exists():
        return manifest
    values = ((np.arange(tokens, dtype=np.uint64) * 73 + 19) % vocab_size).astype(np.uint32)
    return write_token_shards(
        [values], root,
        dataset={"name": "deterministic-synthetic", "revision": "generator-v1"},
        tokenizer={"name": "integer-fixture", "revision": "generator-v1"},
    )


class TokenShardDataset:
    """Random-access packed token sequences with a deterministic global order."""

    def __init__(
        self, directory: str | Path, sequence_length: int, seed: int, *, verify_hashes: bool = True
    ) -> None:
        self.root = Path(directory)
        self.sequence_length = sequence_length
        self.seed = seed
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("format_version") != 1:
            raise ValueError("unsupported token manifest format")
        self.shards: list[np.memmap] = []
        self._cumulative_windows: list[int] = []
        windows = 0
        actual_minimum: int | None = None
        actual_maximum: int | None = None
        for record in self.manifest["shards"]:
            path = self.root / record["file"]
            if verify_hashes and _sha256(path) != record["sha256"]:
                raise ValueError(f"hash mismatch for {path}")
            array = np.memmap(path, mode="r", dtype="<u4")
            if verify_hashes:
                shard_minimum, shard_maximum = int(np.min(array)), int(np.max(array))
                actual_minimum = shard_minimum if actual_minimum is None else min(actual_minimum, shard_minimum)
                actual_maximum = shard_maximum if actual_maximum is None else max(actual_maximum, shard_maximum)
            available = len(array) - sequence_length + 1
            if available <= 0:
                continue
            self.shards.append(array)
            windows += available
            self._cumulative_windows.append(windows)
        if windows <= 0:
            raise ValueError("no shard is long enough for the requested sequence length")
        if verify_hashes and "min_token_id" in self.manifest:
            recorded = (int(self.manifest["min_token_id"]), int(self.manifest["max_token_id"]))
            actual = (actual_minimum, actual_maximum)
            if recorded != actual:
                raise ValueError(f"token bounds mismatch: manifest={recorded}, shards={actual}")
        self.max_token_id = actual_maximum if verify_hashes else self.manifest.get("max_token_id")
        self.num_windows = windows
        # An affine permutation gives O(1) stable shuffle without a huge index array.
        multiplier = 2 * (seed % max(1, windows)) + 1
        while math.gcd(multiplier, windows) != 1:
            multiplier += 2
        self._multiplier = multiplier
        self._offset = (seed * 0x9E3779B1) % windows

    def validate_vocab_size(self, vocab_size: int) -> None:
        maximum = self.max_token_id
        if maximum is None:  # Backward-compatible validation for format-v1 manifests.
            maximum = max(int(np.max(shard)) for shard in self.shards)
        if int(maximum) >= vocab_size:
            raise ValueError(
                f"dataset contains token id {maximum}, outside model vocabulary size {vocab_size}"
            )

    def _window(self, global_index: int) -> tuple[int, int]:
        permuted = (self._multiplier * (global_index % self.num_windows) + self._offset) % self.num_windows
        shard_index = bisect.bisect_right(self._cumulative_windows, permuted)
        previous = 0 if shard_index == 0 else self._cumulative_windows[shard_index - 1]
        return shard_index, permuted - previous

    def sample(self, global_index: int) -> Tensor:
        shard_index, start = self._window(global_index)
        values = np.asarray(self.shards[shard_index][start : start + self.sequence_length], dtype=np.int64)
        return torch.from_numpy(values.copy())

    def rank_batch(
        self,
        sample_cursor: int,
        micro_step: int,
        *,
        rank: int,
        world_size: int,
        micro_batch_size: int,
    ) -> tuple[Tensor, list[int]]:
        base = sample_cursor + micro_step * world_size * micro_batch_size + rank * micro_batch_size
        indices = list(range(base, base + micro_batch_size))
        return torch.stack([self.sample(index) for index in indices]), indices
