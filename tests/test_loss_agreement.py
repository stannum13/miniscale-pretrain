import json

import pytest

from scripts.check_loss_agreement import compare_step_logs


def write_log(path, losses, cursors) -> None:
    path.write_text("".join(
        json.dumps({"step": index + 1, "loss": loss, "sample_cursor": cursor}) + "\n"
        for index, (loss, cursor) in enumerate(zip(losses, cursors))
    ))


def test_loss_agreement_reports_maximum_error(tmp_path) -> None:
    baseline = tmp_path / "one.jsonl"
    candidate = tmp_path / "four.jsonl"
    write_log(baseline, [4.0, 3.0], [8, 16])
    write_log(candidate, [4.0 + 1e-8, 3.0], [8, 16])
    assert compare_step_logs(baseline, candidate, tolerance=1e-6) == pytest.approx(1e-8)


def test_loss_agreement_rejects_data_cursor_reset(tmp_path) -> None:
    baseline = tmp_path / "one.jsonl"
    candidate = tmp_path / "two.jsonl"
    write_log(baseline, [4.0, 3.0], [8, 16])
    write_log(candidate, [4.0, 3.0], [8, 8])
    with pytest.raises(AssertionError, match="step/cursor"):
        compare_step_logs(baseline, candidate, tolerance=1e-6)
