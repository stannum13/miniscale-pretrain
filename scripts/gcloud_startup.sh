#!/usr/bin/env bash
set -Eeuo pipefail

workspace=/opt/miniscale-pretrain
job_log=/var/log/miniscale-pretrain.log
status_file=/var/tmp/miniscale-status.json
source_commit=unknown
artifact_uri=
project_id=
instance_name=
instance_zone=

retry() {
  local attempt
  for attempt in 1 2 3; do
    if "$@"; then
      return 0
    fi
    sleep "$((attempt * 2))"
  done
  return 1
}

write_status() {
  local code="$1"
  python3 - "$status_file" "$code" "$source_commit" <<'PY'
import json, sys
from datetime import datetime, timezone
path, code, commit = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump({
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "exit_code": int(code),
        "source_commit": commit,
        "status": "pass" if int(code) == 0 else "fail",
    }, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

upload_artifacts() {
  [[ -n "$artifact_uri" ]] || return 1
  if [[ -d "$workspace/results" ]]; then
    retry gcloud storage rsync --recursive "$workspace/results" "$artifact_uri/results" || return 1
  fi
  if [[ -d "$workspace/profiles" ]]; then
    retry gcloud storage rsync --recursive "$workspace/profiles" "$artifact_uri/profiles" || return 1
  fi
  if [[ -f "$workspace/REPORT.md" ]]; then
    retry gcloud storage cp "$workspace/REPORT.md" "$workspace/SCALE_STATE.md" \
      "$artifact_uri/" || return 1
  fi
  retry gcloud storage cp "$job_log" "$artifact_uri/job.log" || return 1
}

finish() {
  local code=$?
  local final_code="$code"
  trap - EXIT
  write_status "$final_code"
  if [[ -n "$artifact_uri" ]]; then
    if ! upload_artifacts; then
      final_code=74
      write_status "$final_code"
    fi
    if ! retry gcloud storage cp "$job_log" "$status_file" "$artifact_uri/"; then
      final_code=74
      write_status "$final_code"
      retry gcloud storage cp "$status_file" "$artifact_uri/" || true
    fi
  fi
  sync
  if [[ -n "$project_id" && -n "$instance_name" && -n "$instance_zone" ]]; then
    retry gcloud compute instances delete "$instance_name" --project="$project_id" \
      --zone="$instance_zone" --quiet --async || shutdown -h now || true
  else
    shutdown -h now || true
  fi
  exit "$final_code"
}
trap finish EXIT
exec > >(tee -a "$job_log") 2>&1

metadata() {
  retry curl --fail --silent --show-error \
    -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

instance_metadata() {
  retry curl --fail --silent --show-error \
    -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/$1"
}

project_id="$(instance_metadata project/project-id)"
instance_name="$(instance_metadata instance/name)"
instance_zone="$(instance_metadata instance/zone)"
instance_zone="${instance_zone##*/}"
repository_url="$(metadata repository-url)"
source_commit="$(metadata source-commit)"
dataset_revision="$(metadata dataset-revision)"
tokenizer_revision="$(metadata tokenizer-revision)"
artifact_uri="$(metadata artifact-uri)"
peak_tflops="$(metadata peak-tflops)"

echo "Starting miniscale-pretrain at source commit $source_commit"
nvidia-smi
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y git python3-venv

# Fail before expensive setup if the dedicated VM identity cannot persist evidence.
probe=/var/tmp/miniscale-write-probe
printf 'artifact write preflight\n' > "$probe"
retry gcloud storage cp "$probe" "$artifact_uri/write-probe.txt"
retry gcloud storage rm "$artifact_uri/write-probe.txt"

rm -rf "$workspace"
git clone "$repository_url" "$workspace"
cd "$workspace"
git checkout --detach "$source_commit"
test "$(git rev-parse HEAD)" = "$source_commit"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,data]'
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 4, torch.cuda.device_count()
print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpus": torch.cuda.device_count()})
PY

export HF_HOME=/opt/hf-cache
export DATASET_REVISION="$dataset_revision"
export TOKENIZER_REVISION="$tokenizer_revision"
scripts/prepare_data.sh --max-tokens 20000000 --shard-tokens 5000000

make correctness
upload_artifacts
python scripts/fault_test.py --world-size 2 --device cuda --timeout 300
python scripts/fault_test.py --world-size 4 --device cuda --timeout 300
upload_artifacts

# Profile a separate run outside results so profiler overhead cannot enter the report.
MINISCALE_PROFILE_DIR=profiles/150m-2gpu-ddp \
MINISCALE_PROFILE_WAIT=20 MINISCALE_PROFILE_ACTIVE=3 \
torchrun --standalone --nproc-per-node=2 bench.py \
  --config configs/150m.yaml --strategy ddp --run-id profile-150m-2gpu-ddp \
  --output-dir /var/tmp/miniscale-profile-run --peak-tflops "$peak_tflops"
upload_artifacts

# Each launcher prunes its checkpoint after preserving the small benchmark record.
DISCARD_CHECKPOINTS=1 PEAK_TFLOPS="$peak_tflops" make benchmark
upload_artifacts

set +e
torchrun --standalone --nproc-per-node=4 train.py \
  --config configs/400m.yaml --strategy fsdp --output-dir results/runs \
  --run-id long-400m --max-steps 300 --crash-after 123 \
  --peak-tflops "$peak_tflops"
crash_code=$?
set -e
if [[ "$crash_code" -eq 0 ]]; then
  echo "Intentional crash command unexpectedly succeeded" >&2
  exit 1
fi
cp results/runs/long-400m/steps.jsonl results/runs/long-400m/steps.precrash.jsonl
resume_path=results/runs/long-400m/checkpoints/step-00000100
test -f "$resume_path/COMPLETE"
torchrun --standalone --nproc-per-node=4 train.py \
  --config configs/400m.yaml --strategy fsdp --output-dir results/runs \
  --run-id long-400m --max-steps 300 --resume "$resume_path" \
  --peak-tflops "$peak_tflops"

python scripts/verify_long_run.py \
  --run-root results/runs/long-400m \
  --precrash-log results/runs/long-400m/steps.precrash.jsonl \
  --resume-step 100 --crash-step 123 --expected-steps 300 --global-batch-size 32 \
  --crash-code "$crash_code" --output results/long-run-attestation.json
make report
pytest -q

# The fault-test artifacts attest payload equivalence. Keep hashes and logs,
# not multi-GB checkpoint payloads.
find results/runs/long-400m/checkpoints -mindepth 1 -maxdepth 1 \
  -type d -exec rm -rf '{}' +
upload_artifacts
echo "All GCloud experiment stages completed"
