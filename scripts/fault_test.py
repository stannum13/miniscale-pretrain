from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def run(*arguments: str, expected: int = 0) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [sys.executable, "train.py", *arguments], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if completed.returncode != expected:
        raise RuntimeError(f"command returned {completed.returncode}, expected {expected}:\n{completed.stdout}")
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
    with tempfile.TemporaryDirectory(prefix="miniscale-fault-") as temporary:
        output = Path(temporary)
        common = ("--config", "configs/smoke.yaml", "--output-dir", str(output))
        run(*common, "--run-id", "baseline")
        run(*common, "--run-id", "recovered", "--crash-after", "3", expected=86)
        resume = output / "recovered/checkpoints/step-00000002"
        assert (resume / "COMPLETE").exists(), "last pre-failure checkpoint is incomplete"
        run(*common, "--run-id", "recovered", "--resume", str(resume))
        baseline_checkpoint = output / "baseline/checkpoints/step-00000004/rank-00000.pt"
        recovered_checkpoint = output / "recovered/checkpoints/step-00000004/rank-00000.pt"
        baseline = torch.load(baseline_checkpoint, map_location="cpu", weights_only=False)
        recovered = torch.load(recovered_checkpoint, map_location="cpu", weights_only=False)
        equal(baseline, recovered)
        baseline_steps = read_steps(output / "baseline/steps.jsonl")
        recovered_steps = read_steps(output / "recovered/steps.jsonl")
        recovered_tail = recovered_steps[-2:]
        assert [row["step"] for row in recovered_tail] == [3, 4]
        assert [row["sample_cursor"] for row in recovered_tail] == [24, 32]
        for expected, actual in zip(baseline_steps[2:], recovered_tail):
            assert abs(expected["loss"] - actual["loss"]) <= 1e-7
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "pass",
            "failure_exit_code": 86,
            "resumed_from_step": 2,
            "final_step": recovered["step"],
            "final_sample_cursor": recovered["sample_cursor"],
            "loss_continuity_max_abs_error": max(
                abs(a["loss"] - b["loss"]) for a, b in zip(baseline_steps[2:], recovered_tail)
            ),
            "checkpoint_contents_equal": True,
        }
    destination = ROOT / "results/fault-tests" / f'{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
