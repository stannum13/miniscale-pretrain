#!/usr/bin/env bash
set -euo pipefail

model="${MODEL:-150m}"
gpu_count="${GPU_COUNT:-2}"
strategy="${STRATEGY:-ddp}"
: "${PEAK_TFLOPS:?Set PEAK_TFLOPS to this GPU's dense BF16 peak}"
if [[ "$gpu_count" != "2" && "$gpu_count" != "4" ]]; then
  echo "GPU_COUNT must be 2 or 4" >&2
  exit 2
fi
run_id="${RUN_ID:-${model}-${gpu_count}gpu-${strategy}-$(date -u +%Y%m%dT%H%M%SZ)}"

torchrun --standalone --nproc-per-node="$gpu_count" bench.py \
  --config "configs/${model}.yaml" --strategy "$strategy" --run-id "$run_id" \
  --peak-tflops "$PEAK_TFLOPS"
