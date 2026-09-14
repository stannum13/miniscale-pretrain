from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch

from scripts.process_runner import run_process_group


ROOT = Path(__file__).resolve().parents[1]


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_step_logs(baseline: Path, candidate: Path, *, tolerance: float) -> float:
    expected, actual = _records(baseline), _records(candidate)
    expected_positions = [(row["step"], row["sample_cursor"]) for row in expected]
    actual_positions = [(row["step"], row["sample_cursor"]) for row in actual]
    assert actual_positions == expected_positions, (
        f"step/cursor mismatch: expected {expected_positions}, got {actual_positions}"
    )
    errors = [abs(float(a["loss"]) - float(b["loss"])) for a, b in zip(expected, actual)]
    maximum = max(errors, default=0.0)
    assert maximum <= tolerance, f"loss mismatch {maximum} exceeds tolerance {tolerance}"
    return maximum


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _launch(output: Path, world_size: int, use_cuda: bool, timeout_seconds: float) -> None:
    strategy = "single" if world_size == 1 else "ddp"
    command = [
        "torchrun", "--nnodes=1", "--node-rank=0", f"--nproc-per-node={world_size}",
        "--master-addr=127.0.0.1", f"--master-port={_free_port()}", "train.py",
        "--config", "configs/smoke.yaml", "--strategy", strategy,
        "--output-dir", str(output), "--run-id", f"world-{world_size}",
    ]
    environment = os.environ.copy()
    if not use_cuda:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = run_process_group(
        command, cwd=ROOT, env=environment, timeout_seconds=timeout_seconds
    )
    if completed.returncode:
        raise RuntimeError(f"world-size {world_size} failed:\n{completed.stdout}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify controlled 1/2/4-rank loss agreement")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds per torchrun launch")
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available() if args.device == "auto" else args.device == "cuda"
    if use_cuda and torch.cuda.device_count() < 4:
        raise SystemExit("CUDA correctness check requires at least four visible GPUs")
    errors: dict[str, float] = {}
    run_manifests: dict[str, dict] = {}
    artifact_hashes: dict[str, dict[str, str]] = {}
    with tempfile.TemporaryDirectory(prefix="miniscale-agreement-") as temporary:
        output = Path(temporary)
        for world_size in (1, 2, 4):
            _launch(output, world_size, use_cuda, args.timeout)
        baseline = output / "world-1/steps.jsonl"
        for world_size in (2, 4):
            errors[str(world_size)] = compare_step_logs(
                baseline, output / f"world-{world_size}/steps.jsonl", tolerance=args.tolerance
            )
        for world_size in (1, 2, 4):
            run_root = output / f"world-{world_size}"
            run_manifests[str(world_size)] = json.loads((run_root / "run.json").read_text())
            files = [run_root / "steps.jsonl", run_root / "checkpoints/step-00000004/metadata.json"]
            files.extend(
                run_root / f"checkpoints/step-00000004/rank-{rank:05d}.pt"
                for rank in range(world_size)
            )
            artifact_hashes[str(world_size)] = {
                str(path.relative_to(run_root)): _sha256(path) for path in files
            }
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "device_type": "cuda" if use_cuda else "cpu",
        "world_sizes": [1, 2, 4],
        "global_batch_size": 8,
        "maximum_loss_absolute_error": errors,
        "tolerance": args.tolerance,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "run_manifests": run_manifests,
        "artifact_sha256": artifact_hashes,
    }
    destination = ROOT / "results/correctness" / f'{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
