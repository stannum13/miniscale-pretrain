import json
from pathlib import Path

import pytest

from scripts.verify_long_run import verify_long_run


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_long_run_attestation_proves_cursor_and_replay_continuity(tmp_path: Path) -> None:
    run = tmp_path / "long"
    rows = [
        {"step": step, "sample_cursor": step * 32, "loss": 5.0 - step / 1000,
         "gradient_norm": 1.0}
        for step in range(1, 7)
    ]
    write_jsonl(run / "steps.jsonl", rows)
    write_jsonl(tmp_path / "precrash.jsonl", rows[:5])
    (run / "intentional-crash.json").write_text(json.dumps({
        "exit_code": 86, "step": 5, "sample_cursor": 160
    }))
    for step in (4, 6):
        checkpoint = run / "checkpoints" / f"step-{step:08d}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "COMPLETE").touch()
        (checkpoint / "metadata.json").write_text(json.dumps({
            "step": step, "sample_cursor": step * 32, "world_size": 4, "strategy": "fsdp"
        }))

    result = verify_long_run(
        run, tmp_path / "precrash.jsonl", resume_step=4, crash_step=5, expected_steps=6,
        global_batch_size=32, crash_code=1,
    )
    assert result["status"] == "pass"
    assert result["replayed_steps_compared"] == 1
    assert result["max_replayed_loss_error"] == 0
    assert result["final_sample_cursor"] == 192


def test_long_run_attestation_rejects_data_reset(tmp_path: Path) -> None:
    run = tmp_path / "long"
    rows = [{"step": 1, "sample_cursor": 0, "loss": 1.0, "gradient_norm": 1.0}]
    write_jsonl(run / "steps.jsonl", rows)
    write_jsonl(tmp_path / "precrash.jsonl", rows)
    (run / "intentional-crash.json").write_text(json.dumps({
        "exit_code": 86, "step": 1, "sample_cursor": 32
    }))
    with pytest.raises(AssertionError, match="sample cursor"):
        verify_long_run(
            run, tmp_path / "precrash.jsonl", resume_step=0, crash_step=1, expected_steps=1,
            global_batch_size=32, crash_code=1,
        )
