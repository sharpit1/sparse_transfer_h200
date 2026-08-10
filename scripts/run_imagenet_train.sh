#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
    echo "usage: $0 <0.25|0.5> [imagenet-train-dir]" >&2
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
out_root="${OUT_ROOT:-$root/runs/imagenet_train}"

is_imagenet_train_dir() {
    local candidate="$1"
    local class_count
    [[ -d "$candidate" ]] || return 1
    class_count="$(find "$candidate" -mindepth 1 -maxdepth 1 -type d | wc -l)"
    [[ "$class_count" -eq 1000 ]]
}

resolve_imagenet_train_dir() {
    local explicit_path="${1:-}"
    local input_root
    local candidate
    local -a candidates=()

    if [[ -n "$explicit_path" ]]; then
        if is_imagenet_train_dir "$explicit_path"; then
            printf '%s\n' "$explicit_path"
            return 0
        fi
        echo "explicit ImageNet path is not a 1000-class ImageFolder: $explicit_path" >&2
        return 1
    fi

    if [[ -n "${IMAGENET_TRAIN_DIR:-}" ]]; then
        candidates+=("$IMAGENET_TRAIN_DIR")
    fi

    candidates+=(
        "$root/data/imagenet/train"
        "$root/data/ILSVRC2012_img_train"
        "/app/scratch/datasets/imagenet/train"
        "/app/scratch/dataset/imagenet/train"
        "/app/scratch/imagenet/train"
        "/app/scratch/ILSVRC2012_img_train"
        "/datasets/imagenet/train"
        "/datasets/ILSVRC2012_img_train"
        "/data/imagenet/train"
        "/data/ILSVRC2012_img_train"
    )

    input_root="$(dirname "$root")"
    shopt -s nullglob
    candidates+=(
        "$input_root"/*/train
        "$input_root"/*/ILSVRC2012_img_train
        "$input_root"/*/*/train
    )
    shopt -u nullglob

    for candidate in "${candidates[@]}"; do
        if is_imagenet_train_dir "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    echo "ImageNet train directory was not found." >&2
    echo "Expected an ImageFolder root containing exactly 1000 class directories." >&2
    echo "Pass the path as the second argument or set IMAGENET_TRAIN_DIR in the FaaS environment." >&2
    return 1
}

train_dir="$(resolve_imagenet_train_dir "${2:-}")"
echo "imagenet_train_dir=$train_dir"
echo "batch_size=16 epochs=15 warmup_epochs=2 damping=$damping"

export CUDA_VISIBLE_DEVICES="$gpu_id"
export TORCH_HOME="${TORCH_HOME:-$root/.cache/torch}"
export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

mkdir -p "$TORCH_HOME" "$out_root"

exec "$python_bin" -u "$root/third_party/GPG/DDSC_GPG_train.py" \
    --train_dir "$train_dir" \
    --model_type res50 \
    --eps 10 \
    --target -1 \
    --batch_size 16 \
    --sample_per_class 0 \
    --n_iters 1 \
    --epochs 15 \
    --lr 2.25e-5 \
    --lam_1 0.0001 \
    --lam_2 0.0001 \
    --lam_3 0.0001 \
    --pb full \
    --load_CP New \
    --out-dir "$out_root" \
    --device cuda:0 \
    --num_workers 8 \
    --worker_timeout_seconds 120 \
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
