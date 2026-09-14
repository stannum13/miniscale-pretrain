import json

import pytest

from data.dataset import ensure_synthetic_dataset
from miniscale.config import load_config
from miniscale.provenance import experiment_key, prepare_run_directory


def test_compiled_and_eager_runs_have_different_experiment_keys(tmp_path) -> None:
    eager = load_config("configs/smoke.yaml")
    eager.data.directory = str(tmp_path / "tokens")
    ensure_synthetic_dataset(eager.data.directory, eager.model.vocab_size)
    compiled = load_config("configs/smoke.yaml")
    compiled.data.directory = eager.data.directory
    compiled.compile = True
    assert experiment_key(eager, "fixture-gpu") != experiment_key(compiled, "fixture-gpu")


def test_training_schedule_and_seed_are_part_of_experiment_identity(tmp_path) -> None:
    baseline = load_config("configs/smoke.yaml")
    baseline.data.directory = str(tmp_path / "tokens")
    ensure_synthetic_dataset(baseline.data.directory, baseline.model.vocab_size)
    changed = load_config("configs/smoke.yaml")
    changed.data.directory = baseline.data.directory
    original = experiment_key(baseline, "fixture-gpu")
    changed.seed += 1
    assert experiment_key(changed, "fixture-gpu") != original
    changed.seed = baseline.seed
    changed.learning_rate *= 2
    assert experiment_key(changed, "fixture-gpu") != original


def test_run_manifest_is_immutable_and_records_dataset_revisions(tmp_path) -> None:
    cfg = load_config("configs/smoke.yaml")
    cfg.data.directory = str(tmp_path / "tokens")
    ensure_synthetic_dataset(cfg.data.directory, cfg.model.vocab_size)
    run_root = tmp_path / "run"
    path = prepare_run_directory(
        run_root, cfg, device_type="cpu", backend="none", hardware="fixture",
        world_size=1, is_resume=False,
    )
    payload = json.loads(path.read_text())
    assert payload["dataset_manifest"]["dataset"]["revision"] == "generator-v1"
    assert payload["dataset_manifest"]["tokenizer"]["revision"] == "generator-v1"
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_run_directory(
            run_root, cfg, device_type="cpu", backend="none", hardware="fixture",
            world_size=1, is_resume=False,
        )


def test_resume_requires_matching_run_manifest(tmp_path) -> None:
    cfg = load_config("configs/smoke.yaml")
    cfg.data.directory = str(tmp_path / "tokens")
    ensure_synthetic_dataset(cfg.data.directory, cfg.model.vocab_size)
    run_root = tmp_path / "run"
    prepare_run_directory(run_root, cfg, device_type="cpu", backend="none",
                          hardware="fixture", world_size=1, is_resume=False)
    cfg.resume = "checkpoint"
    prepare_run_directory(run_root, cfg, device_type="cpu", backend="none",
                          hardware="fixture", world_size=1, is_resume=True)
    cfg.seed += 1
    with pytest.raises(ValueError, match="manifest does not match"):
        prepare_run_directory(run_root, cfg, device_type="cpu", backend="none",
                              hardware="fixture", world_size=1, is_resume=True)
