import json
import subprocess
from pathlib import Path

import pytest

from data.dataset import ensure_synthetic_dataset
from miniscale.config import load_config
from miniscale.provenance import (
    _working_tree_dirty,
    _working_tree_digest,
    digest_files,
    experiment_key,
    prepare_run_directory,
)


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
    assert payload["source_digest"]
    assert payload["python_version"]
    assert payload["torch_version"]
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


def test_source_digest_and_experiment_key_change_with_source_bytes(tmp_path) -> None:
    source = tmp_path / "train.py"
    source.write_text("first")
    first = digest_files(tmp_path, ["train.py"])
    source.write_text("second")
    second = digest_files(tmp_path, ["train.py"])
    assert first != second
    cfg = load_config("configs/smoke.yaml")
    cfg.data.directory = str(tmp_path / "tokens")
    ensure_synthetic_dataset(cfg.data.directory, cfg.model.vocab_size)
    assert experiment_key(cfg, "fixture", source_state_digest=first) != experiment_key(
        cfg, "fixture", source_state_digest=second
    )


def test_source_discovery_is_independent_of_caller_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = load_config(Path(__file__).parents[1] / "configs/smoke.yaml")
    cfg.data.directory = str(tmp_path / "tokens")
    ensure_synthetic_dataset(cfg.data.directory, cfg.model.vocab_size)
    path = prepare_run_directory(tmp_path / "output/run", cfg, device_type="cpu",
                                 backend="none", hardware="fixture", world_size=1,
                                 is_resume=False)
    payload = json.loads(path.read_text())
    assert payload["source_digest"] not in {"", "unknown"}
    assert payload["git_commit"] != "unknown"


def test_source_digest_excludes_generated_output_tree(tmp_path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    source = tmp_path / "train.py"
    source.write_text("source")
    subprocess.run(["git", "add", "train.py"], cwd=tmp_path, check=True)
    before = _working_tree_digest(repo_root=tmp_path, excluded_roots=[tmp_path / "runs"])
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/metrics.jsonl").write_text("generated")
    after = _working_tree_digest(repo_root=tmp_path, excluded_roots=[tmp_path / "runs"])
    assert after == before
    source.write_text("changed")
    assert _working_tree_digest(repo_root=tmp_path, excluded_roots=[tmp_path / "runs"]) != before


def test_source_digest_never_excludes_tracked_files(tmp_path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    tracked = tmp_path / "data/dataset.py"
    tracked.parent.mkdir()
    tracked.write_text("first")
    subprocess.run(["git", "add", "data/dataset.py"], cwd=tmp_path, check=True)
    before = _working_tree_digest(repo_root=tmp_path, excluded_roots=[tmp_path / "data"])
    tracked.write_text("second")
    after = _working_tree_digest(repo_root=tmp_path, excluded_roots=[tmp_path / "data"])
    assert after != before


def test_dirty_state_ignores_generated_untracked_outputs_but_not_tracked_changes(tmp_path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    source = tmp_path / "train.py"
    source.write_text("first")
    subprocess.run(["git", "add", "train.py"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "initial"],
        cwd=tmp_path, check=True,
    )
    output = tmp_path / "custom-output"
    output.mkdir()
    (output / "sibling-run.json").write_text("generated")
    assert not _working_tree_dirty(repo_root=tmp_path, excluded_roots=[output])
    source.write_text("second")
    assert _working_tree_dirty(repo_root=tmp_path, excluded_roots=[output])
