#!/usr/bin/env bash
set -euo pipefail

model="${MODEL:-150m}"
: "${PEAK_TFLOPS:?Set PEAK_TFLOPS to this GPU's dense BF16 peak}"
run_id="${RUN_ID:-${model}-1gpu-single-$(date -u +%Y%m%dT%H%M%SZ)}"

torchrun --standalone --nproc-per-node=1 bench.py \
  --config "configs/${model}.yaml" --strategy single --run-id "$run_id" \
  --peak-tflops "$PEAK_TFLOPS"
