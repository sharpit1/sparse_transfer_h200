#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -gt 1 ]]; then
    echo "usage: $0 [imagenet-train-dir]" >&2
    exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python}"
gpu_id="${GPU_ID:-0}"
out_root="${OUT_ROOT:-/app/output/sharpit1}"
generator_mode="${GENERATOR_MODE:-legacy}"

case "$generator_mode" in
    legacy|isolated) ;;
    *)
        echo "GENERATOR_MODE must be legacy or isolated" >&2
        exit 2
        ;;
esac

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
        "/app/data/ImageNet-2012/train"
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
    echo "Pass the path as the first argument or set IMAGENET_TRAIN_DIR." >&2
    return 1
}

train_dir="$(resolve_imagenet_train_dir "${1:-}")"

export CUDA_VISIBLE_DEVICES="$gpu_id"
export TORCH_HOME="${TORCH_HOME:-$root/.cache/torch}"
export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

mkdir -p "$TORCH_HOME" "$out_root"

generator_mode_suffix=""
if [[ "$generator_mode" == "isolated" ]]; then
    generator_mode_suffix="_gen_isolated"
fi

stage1_output_dir="$out_root/GPG_res50_tar_-1_eps_10_Load_New_lam1_0.0001_lam3_0.0001_pb_full${generator_mode_suffix}"
stage1_checkpoint="$stage1_output_dir/GN_res50_11.pth"

common_args=(
    --train_dir "$train_dir"
    --model_type res50
    --generator_mode "$generator_mode"
    --eps 10
    --target -1
    --batch_size 16
    --sample_per_class 0
    --n_iters 1
    --lr 2.25e-5
    --lam_2 0.0001
    --lam_3 0.0001
    --pb full
    --out-dir "$out_root"
)

echo "imagenet_train_dir=$train_dir"
echo "stage1=epochs 1-12, warmup epochs 1-3, lam_1=0.0001 after warmup"
echo "stage2=Continue from $stage1_checkpoint, 3 epochs, lam_1=0.00021"
echo "batch_size=16 n_iters=1 generator_mode=$generator_mode gpu=$gpu_id"

"$python_bin" -u "$root/third_party/GPG/GPG_train.py" \
    "${common_args[@]}" \
    --epochs 12 \
    --lam_1 0.0001 \
    --load_CP New

if [[ ! -s "$stage1_checkpoint" ]]; then
    echo "stage-1 checkpoint was not created: $stage1_checkpoint" >&2
    exit 1
fi

exec "$python_bin" -u "$root/third_party/GPG/GPG_train.py" \
    "${common_args[@]}" \
    --epochs 3 \
    --lam_1 0.00021 \
    --load_CP Continue \
    --CP_path "$stage1_checkpoint"
