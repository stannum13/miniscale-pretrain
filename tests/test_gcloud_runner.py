from decimal import Decimal
from pathlib import Path

import pytest
import scripts.gcloud_runner as gcloud_runner

from scripts.gcloud_runner import (
    CloudSettings,
    artifact_bucket,
    build_bucket_create_command,
    build_create_command,
    estimate_max_cost,
    quota_requirements_met,
    validate_launch,
    validate_published_source,
)


def settings() -> CloudSettings:
    return CloudSettings(
        project="fixture-project",
        zone="us-central1-a",
        instance="miniscale-fixture",
        bucket="gs://fixture-bucket/miniscale-fixture",
        source_commit="a" * 40,
        dataset_revision="b" * 40,
        tokenizer_revision="c" * 40,
    )


def test_cost_guard_rejects_worst_case_over_authorized_ceiling() -> None:
    assert estimate_max_cost(Decimal("4.50"), Decimal("6"), Decimal("1")) == Decimal("28.00")
    with pytest.raises(ValueError, match="authorized ceiling"):
        validate_launch(settings(), max_cost=Decimal("31"), quotas_ready=True)


def test_create_command_is_spot_auto_deleting_and_narrowly_scoped(tmp_path: Path) -> None:
    startup = tmp_path / "startup.sh"
    startup.write_text("#!/usr/bin/env bash\n")
    command = build_create_command(settings(), startup)
    joined = " ".join(command)
    assert command[:4] == ["gcloud", "compute", "instances", "create"]
    assert "--machine-type=g2-standard-48" in command
    assert "--provisioning-model=SPOT" in command
    assert "--max-run-duration=6h" in command
    assert "--instance-termination-action=DELETE" in command
    assert "--image-family=common-cu129-ubuntu-2404-nvidia-580" in command
    assert "--scopes=storage-rw,logging-write,compute-rw" in command
    assert "--service-account=miniscale-runner@fixture-project.iam.gserviceaccount.com" in command
    assert "cloud-platform" not in joined
    assert f"--metadata-from-file=startup-script={startup}" in command
    assert "source-commit=" + "a" * 40 in joined
    assert "dataset-revision=" + "b" * 40 in joined
    assert "tokenizer-revision=" + "c" * 40 in joined
    assert "artifact-uri=gs://fixture-bucket/miniscale-fixture/fixture-run" in joined


def test_artifact_bucket_is_regional_and_lifecycle_managed() -> None:
    assert artifact_bucket(settings().bucket) == "gs://fixture-bucket"
    command = build_bucket_create_command(settings())
    assert command == [
        "gcloud", "storage", "buckets", "create", "gs://fixture-bucket",
        "--project=fixture-project", "--location=us-central1",
        "--uniform-bucket-level-access",
    ]


def test_quota_gate_requires_all_four_limits() -> None:
    global_quotas = {"GPUS_ALL_REGIONS": (4, 0), "CPUS_ALL_REGIONS": (64, 8)}
    regional_quotas = {"PREEMPTIBLE_NVIDIA_L4_GPUS": (4, 0), "PREEMPTIBLE_CPUS": (48, 0)}
    assert quota_requirements_met(global_quotas, regional_quotas)
    regional_quotas["PREEMPTIBLE_NVIDIA_L4_GPUS"] = (1, 0)
    assert not quota_requirements_met(global_quotas, regional_quotas)


def test_launch_refuses_incomplete_quotas_or_placeholder_revisions() -> None:
    with pytest.raises(RuntimeError, match="quota"):
        validate_launch(settings(), max_cost=Decimal("28"), quotas_ready=False)
    invalid = settings()
    invalid = CloudSettings(**{**invalid.__dict__, "dataset_revision": "main"})
    with pytest.raises(ValueError, match="immutable 40-character"):
        validate_launch(invalid, max_cost=Decimal("28"), quotas_ready=True)


def test_source_must_be_clean_head_and_published(monkeypatch) -> None:
    commit = "a" * 40
    outputs = iter(("", commit + "\n", f"{commit}\trefs/heads/main\n"))
    monkeypatch.setattr(gcloud_runner, "_command_output", lambda _command: next(outputs))
    validate_published_source(commit)

    outputs = iter((" M train.py\n", commit + "\n", f"{commit}\trefs/heads/main\n"))
    monkeypatch.setattr(gcloud_runner, "_command_output", lambda _command: next(outputs))
    with pytest.raises(RuntimeError, match="worktree"):
        validate_published_source(commit)


def test_startup_job_has_terminal_upload_and_all_required_gpu_gates() -> None:
    script = Path("scripts/gcloud_startup.sh").read_text()
    assert "trap finish EXIT" in script
    assert "make correctness" in script
    assert "scripts/fault_test.py --world-size 2 --device cuda" in script
    assert "scripts/fault_test.py --world-size 4 --device cuda" in script
    assert "MINISCALE_PROFILE_DIR=profiles/150m-2gpu-ddp" in script
    assert "--output-dir /var/tmp/miniscale-profile-run" in script
    assert "--output-dir results/profiles" not in script
    assert "make benchmark" in script
    assert "DISCARD_CHECKPOINTS=1" in script
    assert "--crash-after 123" in script
    assert "step-00000100" in script
    assert "steps.precrash.jsonl" in script
    assert "scripts/verify_long_run.py" in script
    assert "gcloud storage rsync" in script
    assert "shutdown -h now" in script


def test_makefile_exposes_guarded_cloud_operations() -> None:
    makefile = Path("Makefile").read_text()
    for target in ("cloud-status:", "cloud-launch:", "cloud-logs:", "cloud-download:", "cloud-delete:"):
        assert target in makefile
    assert "CLOUD_RUNNER = python scripts/gcloud_runner.py" in makefile
    assert "$(CLOUD_RUNNER) create" in makefile
