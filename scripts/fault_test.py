from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def build_train_command(arguments: tuple[str, ...], *, world_size: int, port: int) -> list[str]:
    return [
        "torchrun", "--nnodes=1", "--node-rank=0", f"--nproc-per-node={world_size}",
        "--master-addr=127.0.0.1", f"--master-port={port}", "train.py", *arguments,
    ]


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def run(
    *arguments: str,
    world_size: int,
    use_cuda: bool,
    expect_failure: bool = False,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    if not use_cuda:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        build_train_command(arguments, world_size=world_size, port=free_port()),
        cwd=ROOT, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    valid = completed.returncode != 0 if expect_failure else completed.returncode == 0
    if not valid:
        expectation = "nonzero" if expect_failure else "zero"
        raise RuntimeError(f"command returned {completed.returncode}, expected {expectation}:\n{completed.stdout}")
    if expect_failure and "86" not in completed.stdout:
        raise RuntimeError(f"failure did not contain intentional exit code 86:\n{completed.stdout}")
    return completed


def equal(left, right, path: str = "state") -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0, msg=lambda msg: f"{path}: {msg}")
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right, err_msg=path)
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            equal(a, b, f"{path}[{index}]")
    else:
        assert left == right, f"{path}: {left!r} != {right!r}"


def read_steps(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description="Hard-crash and exact-resume checkpoint test")
    parser.add_argument("--world-size", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    visible_cuda = torch.cuda.device_count()
    use_cuda = visible_cuda >= args.world_size if args.device == "auto" else args.device == "cuda"
    if use_cuda and visible_cuda < args.world_size:
        raise SystemExit(f"requested {args.world_size} CUDA ranks but only {visible_cuda} GPUs are visible")
    strategy = "single" if args.world_size == 1 else "ddp"
    with tempfile.TemporaryDirectory(prefix="miniscale-fault-") as temporary:
        output = Path(temporary)
        common = (
            "--config", "configs/smoke.yaml", "--strategy", strategy,
            "--output-dir", str(output),
        )
        run(*common, "--run-id", "baseline", world_size=args.world_size, use_cuda=use_cuda)
        run(*common, "--run-id", "recovered", "--crash-after", "3",
            world_size=args.world_size, use_cuda=use_cuda, expect_failure=True)
        resume = output / "recovered/checkpoints/step-00000002"
        assert (resume / "COMPLETE").exists(), "last pre-failure checkpoint is incomplete"
        run(*common, "--run-id", "recovered", "--resume", str(resume),
            world_size=args.world_size, use_cuda=use_cuda)
        for rank in range(args.world_size):
            name = f"rank-{rank:05d}.pt"
            baseline_path = output / "baseline/checkpoints/step-00000004" / name
            recovered_path = output / "recovered/checkpoints/step-00000004" / name
            baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
            recovered = torch.load(recovered_path, map_location="cpu", weights_only=False)
            equal(baseline, recovered, path=f"rank-{rank}")
        baseline_steps = read_steps(output / "baseline/steps.jsonl")
        recovered_steps = read_steps(output / "recovered/steps.jsonl")
        assert [row["step"] for row in recovered_steps] == [1, 2, 3, 4]
        recovered_tail = recovered_steps[-2:]
        assert [row["step"] for row in recovered_tail] == [3, 4]
        assert [row["sample_cursor"] for row in recovered_tail] == [24, 32]
        for expected, actual in zip(baseline_steps[2:], recovered_tail):
            assert abs(expected["loss"] - actual["loss"]) <= 1e-7
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "pass",
            "failure_exit_code": 86,
            "device_type": "cuda" if use_cuda else "cpu",
            "world_size": args.world_size,
            "strategy": strategy,
            "resumed_from_step": 2,
            "final_step": recovered["step"],
            "final_sample_cursor": recovered["sample_cursor"],
            "loss_continuity_max_abs_error": max(
                abs(a["loss"] - b["loss"]) for a, b in zip(baseline_steps[2:], recovered_tail)
            ),
            "checkpoint_contents_equal": True,
        }
    destination = ROOT / "results/fault-tests" / (
        f'{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}-world{args.world_size}.json'
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
