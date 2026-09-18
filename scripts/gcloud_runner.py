from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Mapping, Sequence


AUTHORIZED_CEILING = Decimal("30")
CONSERVATIVE_VM_HOURLY_RATE = Decimal("4.50")
MAX_HOURS = Decimal("6")
DISK_AND_NETWORK_BUFFER = Decimal("1")
REPOSITORY_URL = "https://github.com/stannum13/miniscale-pretrain.git"


@dataclass(frozen=True)
class CloudSettings:
    project: str
    zone: str
    instance: str
    bucket: str
    source_commit: str
    dataset_revision: str
    tokenizer_revision: str
    cloud_run_id: str = "fixture-run"
    machine_type: str = "g2-standard-48"
    image_family: str = "common-cu129-ubuntu-2404-nvidia-580"
    image_project: str = "deeplearning-platform-release"
    max_run_duration: str = "6h"
    peak_tflops: str = "121"

    @property
    def service_account(self) -> str:
        return f"miniscale-runner@{self.project}.iam.gserviceaccount.com"


def estimate_max_cost(hourly_rate: Decimal, max_hours: Decimal, disk_buffer: Decimal) -> Decimal:
    return (hourly_rate * max_hours + disk_buffer).quantize(Decimal("0.01"))


def _immutable_revision(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{40}", value))


def validate_launch(settings: CloudSettings, *, max_cost: Decimal, quotas_ready: bool) -> None:
    if max_cost > AUTHORIZED_CEILING:
        raise ValueError(
            f"worst-case cost ${max_cost} exceeds authorized ceiling ${AUTHORIZED_CEILING}"
        )
    if not quotas_ready:
        raise RuntimeError("required GCloud quota is not ready")
    for label, revision in (
        ("source", settings.source_commit),
        ("dataset", settings.dataset_revision),
        ("tokenizer", settings.tokenizer_revision),
    ):
        if not _immutable_revision(revision):
            raise ValueError(f"{label} revision must be an immutable 40-character commit")
    if not settings.bucket.startswith("gs://"):
        raise ValueError("artifact bucket must be a gs:// URI")


def _command_output(command: Sequence[str]) -> str:
    return subprocess.run(list(command), check=True, text=True, capture_output=True).stdout


def validate_published_source(source_commit: str) -> None:
    if _command_output(["git", "status", "--porcelain"]):
        raise RuntimeError("worktree must be clean before a cloud launch")
    head = _command_output(["git", "rev-parse", "HEAD"]).strip()
    if head != source_commit:
        raise RuntimeError(f"source commit {source_commit} is not the local HEAD {head}")
    remote_refs = _command_output(["git", "ls-remote", "--heads", "origin"])
    if not any(line.startswith(f"{source_commit}\t") for line in remote_refs.splitlines()):
        raise RuntimeError(f"source commit {source_commit} is not published on origin")


def build_create_command(settings: CloudSettings, startup_script: Path) -> list[str]:
    metadata = ",".join(
        (
            f"repository-url={REPOSITORY_URL}",
            f"source-commit={settings.source_commit}",
            f"dataset-revision={settings.dataset_revision}",
            f"tokenizer-revision={settings.tokenizer_revision}",
            f"artifact-uri={settings.bucket.rstrip('/')}/{settings.cloud_run_id}",
            f"peak-tflops={settings.peak_tflops}",
        )
    )
    return [
        "gcloud", "compute", "instances", "create", settings.instance,
        f"--project={settings.project}",
        f"--zone={settings.zone}",
        f"--machine-type={settings.machine_type}",
        "--provisioning-model=SPOT",
        "--maintenance-policy=TERMINATE",
        "--no-restart-on-failure",
        f"--max-run-duration={settings.max_run_duration}",
        "--instance-termination-action=DELETE",
        f"--image-family={settings.image_family}",
        f"--image-project={settings.image_project}",
        "--boot-disk-size=200GB",
        "--boot-disk-type=pd-balanced",
        f"--service-account={settings.service_account}",
        "--scopes=storage-rw,logging-write,compute-rw",
        f"--metadata={metadata}",
        f"--metadata-from-file=startup-script={startup_script}",
    ]


def artifact_bucket(uri: str) -> str:
    match = re.fullmatch(r"(gs://[^/]+)(?:/.*)?", uri)
    if not match:
        raise ValueError("artifact bucket must be a gs:// URI")
    return match.group(1)


def build_bucket_create_command(settings: CloudSettings) -> list[str]:
    region = settings.zone.rsplit("-", 1)[0]
    return [
        "gcloud", "storage", "buckets", "create", artifact_bucket(settings.bucket),
        f"--project={settings.project}", f"--location={region}",
        "--uniform-bucket-level-access",
    ]


def ensure_artifact_bucket(settings: CloudSettings) -> int:
    bucket = artifact_bucket(settings.bucket)
    described = subprocess.run(
        ["gcloud", "storage", "buckets", "describe", bucket, f"--project={settings.project}"],
        check=False, text=True, capture_output=True,
    )
    if described.returncode and subprocess.run(build_bucket_create_command(settings), check=False).returncode:
        return 1
    lifecycle = Path(__file__).with_name("gcloud_lifecycle.json")
    if subprocess.run(
        ["gcloud", "storage", "buckets", "update", bucket,
         f"--project={settings.project}", f"--lifecycle-file={lifecycle}"],
        check=False,
    ).returncode:
        return 1
    account = settings.service_account
    described_account = subprocess.run(
        ["gcloud", "iam", "service-accounts", "describe", account,
         f"--project={settings.project}"],
        check=False, text=True, capture_output=True,
    )
    if described_account.returncode and subprocess.run(
        ["gcloud", "iam", "service-accounts", "create", "miniscale-runner",
         f"--project={settings.project}", "--display-name=MiniScale GPU runner"],
        check=False,
    ).returncode:
        return 1
    if subprocess.run(
        ["gcloud", "storage", "buckets", "add-iam-policy-binding", bucket,
         f"--project={settings.project}", f"--member=serviceAccount:{account}",
         "--role=roles/storage.objectAdmin", "--quiet"],
        check=False,
    ).returncode:
        return 1
    role = "miniscaleSelfDelete"
    described_role = subprocess.run(
        ["gcloud", "iam", "roles", "describe", role, f"--project={settings.project}"],
        check=False, text=True, capture_output=True,
    )
    if described_role.returncode and subprocess.run(
        ["gcloud", "iam", "roles", "create", role, f"--project={settings.project}",
         "--title=MiniScale VM self delete",
         "--permissions=compute.instances.delete,compute.instances.get", "--stage=GA"],
        check=False,
    ).returncode:
        return 1
    return subprocess.run(
        ["gcloud", "projects", "add-iam-policy-binding", settings.project,
         f"--member=serviceAccount:{account}",
         f"--role=projects/{settings.project}/roles/{role}", "--quiet"],
        check=False,
    ).returncode


def quota_requirements_met(
    global_quotas: Mapping[str, tuple[float, float]],
    regional_quotas: Mapping[str, tuple[float, float]],
) -> bool:
    required_global = {"GPUS_ALL_REGIONS": 4, "CPUS_ALL_REGIONS": 48}
    required_regional = {"PREEMPTIBLE_NVIDIA_L4_GPUS": 4, "PREEMPTIBLE_CPUS": 48}
    return all(
        key in global_quotas and global_quotas[key][0] - global_quotas[key][1] >= needed
        for key, needed in required_global.items()
    ) and all(
        key in regional_quotas and regional_quotas[key][0] - regional_quotas[key][1] >= needed
        for key, needed in required_regional.items()
    )


def _quota_map(payload: Mapping[str, object]) -> dict[str, tuple[float, float]]:
    quotas = payload.get("quotas", [])
    assert isinstance(quotas, list)
    result: dict[str, tuple[float, float]] = {}
    for item in quotas:
        assert isinstance(item, dict)
        result[str(item["metric"])] = (float(item["limit"]), float(item.get("usage", 0)))
    return result


def _json_command(command: Sequence[str]) -> dict:
    completed = subprocess.run(list(command), check=True, text=True, capture_output=True)
    return json.loads(completed.stdout)


def current_quotas(project: str, region: str) -> tuple[dict, dict]:
    global_payload = _json_command(
        ["gcloud", "compute", "project-info", "describe", f"--project={project}", "--format=json"]
    )
    regional_payload = _json_command(
        ["gcloud", "compute", "regions", "describe", region, f"--project={project}", "--format=json"]
    )
    return _quota_map(global_payload), _quota_map(regional_payload)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Cost- and quota-gated GCloud experiment runner")
    result.add_argument(
        "operation", choices=("status", "bucket", "create", "logs", "download", "delete")
    )
    result.add_argument("--project", default="project-1178f0de-10fb-4e7e-8e4")
    result.add_argument("--zone", default="us-central1-a")
    result.add_argument("--instance", default="miniscale-pretrain")
    result.add_argument("--bucket", default="gs://miniscale-pretrain-888484963419/miniscale-pretrain")
    result.add_argument("--source-commit")
    result.add_argument("--dataset-revision")
    result.add_argument("--tokenizer-revision")
    result.add_argument("--cloud-run-id")
    result.add_argument("--dry-run", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    region = args.zone.rsplit("-", 1)[0]
    global_quotas, regional_quotas = current_quotas(args.project, region)
    ready = quota_requirements_met(global_quotas, regional_quotas)
    status = {
        "ready": ready,
        "authorized_ceiling_usd": str(AUTHORIZED_CEILING),
        "worst_case_cost_usd": str(
            estimate_max_cost(CONSERVATIVE_VM_HOURLY_RATE, MAX_HOURS, DISK_AND_NETWORK_BUFFER)
        ),
        "global": {key: global_quotas.get(key) for key in ("GPUS_ALL_REGIONS", "CPUS_ALL_REGIONS")},
        "regional": {
            key: regional_quotas.get(key)
            for key in ("PREEMPTIBLE_NVIDIA_L4_GPUS", "PREEMPTIBLE_CPUS")
        },
    }
    if args.operation == "status":
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0 if ready else 3
    settings = CloudSettings(
        project=args.project,
        zone=args.zone,
        instance=args.instance,
        bucket=args.bucket,
        source_commit=args.source_commit or "",
        dataset_revision=args.dataset_revision or "",
        tokenizer_revision=args.tokenizer_revision or "",
        cloud_run_id=args.cloud_run_id or (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
            + (args.source_commit or "unknown")[:8]
        ),
    )
    if args.operation == "bucket":
        return ensure_artifact_bucket(settings)
    if args.operation == "create":
        max_cost = estimate_max_cost(
            CONSERVATIVE_VM_HOURLY_RATE, MAX_HOURS, DISK_AND_NETWORK_BUFFER
        )
        validate_launch(settings, max_cost=max_cost, quotas_ready=ready)
        validate_published_source(settings.source_commit)
        command = build_create_command(settings, Path(__file__).with_name("gcloud_startup.sh"))
        if args.dry_run:
            print(json.dumps(command, indent=2))
            return 0
        return subprocess.run(command, check=False).returncode
    if args.operation == "logs":
        command = [
            "gcloud", "compute", "instances", "get-serial-port-output", args.instance,
            f"--project={args.project}", f"--zone={args.zone}", "--port=1",
        ]
    elif args.operation == "download":
        command = ["gcloud", "storage", "cp", "--recursive", f"{args.bucket}/", "results/cloud/"]
    else:
        command = [
            "gcloud", "compute", "instances", "delete", args.instance,
            f"--project={args.project}", f"--zone={args.zone}", "--quiet",
        ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
