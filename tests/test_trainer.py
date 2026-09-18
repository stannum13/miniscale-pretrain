import json
from pathlib import Path

import pytest
import torch

from miniscale.config import load_config
from distributed.runtime import DistributedContext
from miniscale.trainer import _collective_local, _profile_settings, run_training


def configured(tmp_path: Path, name: str):
    cfg = load_config("configs/smoke.yaml")
    cfg.run_name = name
    cfg.output_dir = str(tmp_path / name)
    cfg.data.directory = str(tmp_path / "tokens")
    cfg.max_steps = 4
    cfg.warmup_steps = 0
    cfg.measured_steps = 4
    cfg.checkpoint_interval = 2
    return cfg


def checkpoint_model(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path / "rank-00000.pt", map_location="cpu", weights_only=False)["model"]


def test_interrupted_resume_matches_uninterrupted_loss_and_weights(tmp_path: Path) -> None:
    baseline = run_training(configured(tmp_path, "baseline"), run_id="baseline")
    interrupted_cfg = configured(tmp_path, "resumed")
    interrupted = run_training(interrupted_cfg, run_id="resumed", stop_after=2)
    assert interrupted.stopped_early
    interrupted_cfg.resume = str(interrupted.last_checkpoint)
    resumed = run_training(interrupted_cfg, run_id="resumed")
    assert resumed.start_step == 2 and resumed.sample_cursor_start == 16
    assert resumed.losses == pytest.approx(baseline.losses[2:], rel=0, abs=1e-7)
    log_path = Path(interrupted_cfg.output_dir) / "resumed/steps.jsonl"
    assert [json.loads(line)["step"] for line in log_path.read_text().splitlines()] == [1, 2, 3, 4]
    baseline_state = checkpoint_model(baseline.last_checkpoint)
    resumed_state = checkpoint_model(resumed.last_checkpoint)
    for key in baseline_state:
        torch.testing.assert_close(baseline_state[key], resumed_state[key], rtol=0, atol=1e-7)


def test_reusing_run_id_without_resume_is_rejected(tmp_path: Path) -> None:
    cfg = configured(tmp_path, "collision")
    run_training(cfg, run_id="same")
    with pytest.raises(FileExistsError, match="already exists"):
        run_training(cfg, run_id="same")


def test_local_preflight_errors_are_labeled() -> None:
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="dataset preflight failed on rank 0: broken"):
        _collective_local(context, "dataset", lambda: (_ for _ in ()).throw(ValueError("broken")))


def test_profile_settings_are_explicit_and_bounded(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MINISCALE_PROFILE_DIR", raising=False)
    assert _profile_settings(tmp_path, rank=0) is None

    monkeypatch.setenv("MINISCALE_PROFILE_DIR", "profiles/cuda")
    monkeypatch.setenv("MINISCALE_PROFILE_WAIT", "20")
    monkeypatch.setenv("MINISCALE_PROFILE_ACTIVE", "3")
    settings = _profile_settings(tmp_path, rank=2)
    assert settings == (tmp_path / "profiles/cuda", 20, 3, "rank-00002")

    monkeypatch.setenv("MINISCALE_PROFILE_ACTIVE", "0")
    with pytest.raises(ValueError, match="PROFILE_ACTIVE"):
        _profile_settings(tmp_path, rank=0)
