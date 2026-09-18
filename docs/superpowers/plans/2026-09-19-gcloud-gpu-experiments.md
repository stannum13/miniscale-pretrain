# GCloud GPU Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the pending 1/2/4-GPU correctness, scaling, profiling, and fault-recovery experiments on a cost-bounded Google Cloud Spot VM and publish only measured evidence.

**Architecture:** A pure Python launcher builds and validates `gcloud` commands without shell interpolation, refuses configurations whose worst-case VM runtime exceeds the authorized USD 30 ceiling, and creates one auto-deleting 4×L4 VM. A startup script clones the pinned repository commit, prepares pinned data, executes staged gates, uploads artifacts to a dedicated Cloud Storage prefix, and deletes the VM even when a stage fails.

**Tech Stack:** Python 3.10+, Google Cloud CLI/Compute Engine/Cloud Storage, Bash, PyTorch/CUDA, pytest.

## Global Constraints

- Total authorized Google Cloud spend is USD 30.
- Use one `g2-standard-48` Spot VM with four NVIDIA L4 GPUs for hardware-consistent 1/2/4 comparisons.
- Apply an independent six-hour Compute Engine maximum runtime with automatic deletion.
- Never treat CPU diagnostics, failed runs, or estimates as GPU measurements.
- Never launch until global GPU, regional Spot L4, regional Spot CPU, and global CPU quotas are granted.
- Keep global batch size constant within each model's scaling comparison.
- Commit and push runner changes before launching so provenance points to an immutable source commit.

---

### Task 1: Cost- and quota-gated GCloud launcher

**Files:**
- Create: `scripts/gcloud_runner.py`
- Create: `tests/test_gcloud_runner.py`

**Interfaces:**
- Produces `estimate_max_cost(hourly_rate: Decimal, max_hours: Decimal, disk_buffer: Decimal) -> Decimal`.
- Produces `build_create_command(settings: CloudSettings, startup_script: Path) -> list[str]`.
- Produces CLI operations `status`, `create`, `logs`, `download`, and `delete`.

- [ ] Write tests proving a run over USD 30 is rejected; the create command contains Spot provisioning, `6h` maximum runtime, delete-on-termination, pinned image family, and no broad OAuth scope; and `create` refuses incomplete quota status.
- [ ] Run `pytest tests/test_gcloud_runner.py -q`; expect failure because the module does not exist.
- [ ] Implement immutable settings, decimal cost validation, structured quota inspection, subprocess argument lists, and explicit instance lifecycle operations.
- [ ] Re-run the focused tests; expect pass.
- [ ] Commit launcher and tests.

### Task 2: Reproducible staged startup job

**Files:**
- Create: `scripts/gcloud_startup.sh`
- Modify: `scripts/gcloud_runner.py`
- Modify: `tests/test_gcloud_runner.py`

**Interfaces:**
- The launcher passes repository URL, source commit, dataset/tokenizer revisions, bucket URI, and peak BF16 TFLOPS as instance metadata.
- The startup job writes a terminal status marker plus command logs and experiment artifacts to `gs://.../<run-id>/`.

- [ ] Add tests proving required metadata is present and rejecting placeholder revisions or a dirty/unpushed source commit.
- [ ] Run focused tests; expect failure for missing metadata validation.
- [ ] Implement clone/checkout, environment installation, pinned corpus preparation, CUDA correctness, 150M pilot, complete benchmark matrix, profiling capture, selected longer 400M run, intentional termination/resume, report generation, artifact upload, and shutdown cleanup.
- [ ] Re-run focused and full tests; expect pass.
- [ ] Commit the startup workflow.

### Task 3: Operational documentation and launch

**Files:**
- Modify: `README.md`
- Modify: `SCALE_STATE.md`
- Modify: `Makefile`

**Interfaces:**
- `make cloud-status`, `make cloud-launch`, `make cloud-logs`, `make cloud-download`, and `make cloud-delete` wrap the launcher.

- [ ] Add documentation for the USD 30 guard, quotas, automatic deletion, Spot interruption semantics, artifact location, and exact launch/status/download commands.
- [ ] Run `pytest -q`, `make cloud-status`, shell syntax checks, and launcher dry-run; expect pass without creating a VM.
- [ ] Commit and push the immutable source revision.
- [ ] Poll quota status; when all requirements are granted, launch exactly one VM and monitor through artifact upload and deletion.
- [ ] Download artifacts, independently verify manifests and measurements, update `REPORT.md` and `SCALE_STATE.md`, run the release gate, commit, and push.

## Self-Review

- Spec coverage: cost boundary, quota gate, automatic deletion, provenance, staged measurements, artifact recovery, report update, and publication are all assigned.
- Placeholder scan: runtime inputs that must be real are explicitly rejected rather than left as implementation placeholders.
- Type consistency: launcher settings and command builders are introduced in Task 1 and extended, not renamed, in Task 2.
