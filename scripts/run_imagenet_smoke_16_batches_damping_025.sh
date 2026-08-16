#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAIN_EPOCHS=1
export MAX_BATCHES_PER_EPOCH=16

SECONDS=0
if "$script_dir/run_imagenet_train_damping_025.sh" "$@"; then
    echo "imagenet_smoke_total_seconds=$SECONDS"
else
    status=$?
    echo "imagenet_smoke_failed_after_seconds=$SECONDS" >&2
    exit "$status"
fi
