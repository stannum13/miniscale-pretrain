#!/usr/bin/env bash
set -euo pipefail

: "${PEAK_TFLOPS:?Set PEAK_TFLOPS to this GPU's dense BF16 peak}"
models="${MODELS:-150m 400m 1b}"
gpu_counts="${GPU_COUNTS:-1 2 4}"

for model in $models; do
  for gpu_count in $gpu_counts; do
    if [[ "$gpu_count" == "1" ]]; then
      MODEL="$model" PEAK_TFLOPS="$PEAK_TFLOPS" scripts/single_gpu.sh
    else
      for strategy in ddp fsdp; do
        MODEL="$model" GPU_COUNT="$gpu_count" STRATEGY="$strategy" \
          PEAK_TFLOPS="$PEAK_TFLOPS" scripts/distributed.sh
      done
    fi
  done
done
