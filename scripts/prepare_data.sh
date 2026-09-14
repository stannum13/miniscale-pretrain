#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_REVISION:?Set DATASET_REVISION to an immutable FineWeb-Edu commit hash}"
: "${TOKENIZER_REVISION:?Set TOKENIZER_REVISION to an immutable tokenizer commit hash}"

python -m data.prepare \
  --dataset-revision "$DATASET_REVISION" \
  --tokenizer-revision "$TOKENIZER_REVISION" \
  "$@"
