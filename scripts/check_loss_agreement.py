from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


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


def _launch(output: Path, world_size: int, use_cuda: bool) -> None:
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
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify controlled 1/2/4-rank loss agreement")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available() if args.device == "auto" else args.device == "cuda"
    if use_cuda and torch.cuda.device_count() < 4:
        raise SystemExit("CUDA correctness check requires at least four visible GPUs")
    errors: dict[str, float] = {}
    with tempfile.TemporaryDirectory(prefix="miniscale-agreement-") as temporary:
        output = Path(temporary)
        for world_size in (1, 2, 4):
            _launch(output, world_size, use_cuda)
        baseline = output / "world-1/steps.jsonl"
        for world_size in (2, 4):
            errors[str(world_size)] = compare_step_logs(
                baseline, output / f"world-{world_size}/steps.jsonl", tolerance=args.tolerance
            )
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
    }
    destination = ROOT / "results/correctness" / f'{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
