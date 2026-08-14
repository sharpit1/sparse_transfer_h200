#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "usage: $0 <0.25|0.5>" >&2
    exit 2
fi

damping="$1"
case "$damping" in
    0.25|0.5) ;;
    *)
        echo "damping must be 0.25 or 0.5" >&2
        exit 2
        ;;
esac

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python}"
gpu_id="${GPU_ID:-0}"
data_root="$root/data/nips2017_smoke"
out_root="${OUT_ROOT:-/app/output/nips2017_smoke}"
generator_mode="${GENERATOR_MODE:-isolated}"

case "$generator_mode" in
    isolated|legacy) ;;
    *)
        echo "GENERATOR_MODE must be isolated or legacy" >&2
        exit 2
        ;;
esac

if [[ ! -f "$data_root/images.csv" ]]; then
    echo "missing NIPS2017 smoke metadata: $data_root/images.csv" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$gpu_id"
export TORCH_HOME="${TORCH_HOME:-$root/.cache/torch}"
export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

mkdir -p "$TORCH_HOME" "$out_root"

exec "$python_bin" -u "$root/third_party/GPG/DDSC_GPG_train.py" \
    --train_dir "$data_root/images" \
    --train_csv "$data_root/images.csv" \
    --model_type res50 \
    --generator_mode "$generator_mode" \
    --eps 10 \
    --target -1 \
    --batch_size 16 \
    --sample_per_class 0 \
    --n_iters 1 \
    --epochs 5 \
    --lr 2.25e-5 \
    --lam_1 0.0001 \
    --lam_2 0.0001 \
    --lam_3 0.0001 \
    --pb full \
    --load_CP New \
    --out-dir "$out_root" \
    --device cuda:0 \
    --num_workers 0 \
    --seed 42 \
    --decoder_width 128 \
    --decoder_num_blocks 3 \
    --decoder_upsample_backend transpose \
    --save_every 1 \
    --ddsc_target_density 0.10 \
    --ddsc_warmup_epochs 2 \
    --ddsc_ema_decay 0.0 \
    --ddsc_mass 1.0 \
    --ddsc_damping "$damping" \
    --ddsc_restoring_gain 0.0001 \
    --ddsc_dt 1.0 \
    --ddsc_lambda1_min 0.0
