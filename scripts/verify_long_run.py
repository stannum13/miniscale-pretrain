from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _checkpoint_metadata(run_root: Path, step: int, global_batch_size: int) -> tuple[dict, str]:
    checkpoint = run_root / "checkpoints" / f"step-{step:08d}"
    assert (checkpoint / "COMPLETE").is_file(), f"checkpoint {step} is incomplete"
    path = checkpoint / "metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["step"] == step, f"checkpoint step mismatch at {step}"
    assert payload["sample_cursor"] == step * global_batch_size, (
        f"checkpoint sample cursor mismatch at {step}"
    )
    return payload, hashlib.sha256(path.read_bytes()).hexdigest()


def verify_long_run(
    run_root: Path,
    precrash_log: Path,
    *,
    resume_step: int,
    crash_step: int,
    expected_steps: int,
    global_batch_size: int,
    crash_code: int,
) -> dict:
    assert crash_code != 0, "intentional crash did not fail"
    recovered = _rows(run_root / "steps.jsonl")
    before_crash = _rows(precrash_log)
    assert int(before_crash[-1]["step"]) == crash_step, "pre-crash log ended at the wrong step"
    assert int(before_crash[-1]["sample_cursor"]) == crash_step * global_batch_size, (
        "pre-crash sample cursor ended at the wrong position"
    )
    marker = json.loads((run_root / "intentional-crash.json").read_text(encoding="utf-8"))
    assert marker == {
        "exit_code": 86,
        "step": crash_step,
        "sample_cursor": crash_step * global_batch_size,
    }, "intentional crash marker is missing or inconsistent"
    assert [row["step"] for row in recovered] == list(range(1, expected_steps + 1)), (
        "recovered step sequence is not continuous"
    )
    expected_cursors = [step * global_batch_size for step in range(1, expected_steps + 1)]
    assert [row["sample_cursor"] for row in recovered] == expected_cursors, (
        "recovered sample cursor sequence reset or skipped data"
    )
    assert all(
        math.isfinite(float(row[metric]))
        for row in recovered for metric in ("loss", "gradient_norm")
    ), "loss or gradient norm is non-finite"

    recovered_by_step = {int(row["step"]): row for row in recovered}
    replayed = [row for row in before_crash if resume_step < int(row["step"]) <= expected_steps]
    assert replayed, "pre-crash log contains no replayed steps"
    loss_errors = [
        abs(float(row["loss"]) - float(recovered_by_step[int(row["step"])]["loss"]))
        for row in replayed
    ]
    grad_errors = [
        abs(float(row["gradient_norm"]) - float(recovered_by_step[int(row["step"])]["gradient_norm"]))
        for row in replayed
    ]
    assert max(loss_errors) <= 1e-6, "replayed loss trajectory diverged after resume"
    assert max(grad_errors) <= 1e-6, "replayed gradient trajectory diverged after resume"

    resume_metadata, resume_hash = _checkpoint_metadata(run_root, resume_step, global_batch_size)
    final_metadata, final_hash = _checkpoint_metadata(run_root, expected_steps, global_batch_size)
    assert resume_metadata["world_size"] == final_metadata["world_size"] == 4
    assert resume_metadata["strategy"] == final_metadata["strategy"] == "fsdp"
    return {
        "status": "pass",
        "crash_exit_code": crash_code,
        "intentional_exit_code": 86,
        "crash_step": crash_step,
        "resume_step": resume_step,
        "final_step": expected_steps,
        "final_sample_cursor": expected_cursors[-1],
        "replayed_steps_compared": len(replayed),
        "max_replayed_loss_error": max(loss_errors),
        "max_replayed_gradient_norm_error": max(grad_errors),
        "resume_metadata_sha256": resume_hash,
        "final_metadata_sha256": final_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Attest long-run hard-crash recovery")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--precrash-log", type=Path, required=True)
    parser.add_argument("--resume-step", type=int, required=True)
    parser.add_argument("--crash-step", type=int, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--crash-code", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_long_run(
        args.run_root, args.precrash_log, resume_step=args.resume_step,
        crash_step=args.crash_step, expected_steps=args.expected_steps,
        global_batch_size=args.global_batch_size,
        crash_code=args.crash_code,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
