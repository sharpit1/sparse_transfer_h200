#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$script_dir/run_imagenet_train.sh" 0.25 "$@"
"$script_dir/run_imagenet_train.sh" 0.5 "$@"
