#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd -- "${script_dir}/.." && pwd)"
trainer="${root}/third_party/GPG/DDSC_GPG_train.py"
evaluator="${root}/third_party/GPG/tools/evaluate_ddsc_gpg_all_model_t.py"
eval_bootstrapper="${root}/third_party/GPG/tools/bootstrap_transfer_eval.py"
openmmlab_vit_checkpoint="${OPENMMLAB_VIT_CHECKPOINT:-${root}/artifacts/pretrained/vit-base-p16_pt-32xb128-mae_in1k_20220623-4c544545.pth}"

python_bin="${PYTHON:-python}"
gpu_id="${GPU_ID:-0}"
device="${DEVICE:-cuda:0}"
out_root="${OUT_ROOT:-/app/output/gpg_ddsc_lam1_0001}"
epochs="${EPOCHS:-15}"
batch_size="${BATCH_SIZE:-16}"
num_workers="${NUM_WORKERS:-8}"
worker_timeout_seconds="${WORKER_TIMEOUT_SECONDS:-120}"
lambda1_init="${LAMBDA1_INIT:-0.0001}"
ddsc_restoring_gain="${DDSC_RESTORING_GAIN:-}"
ddsc_target_density="${DDSC_TARGET_DENSITY:-0.10}"
ddsc_warmup_epochs="${DDSC_WARMUP_EPOCHS:-2}"
ddsc_damping="${DDSC_DAMPING:-0.25}"
layer1_dropout_mode="${LAYER1_DROPOUT_MODE:-off}"
layer1_dropout_p="${LAYER1_DROPOUT_P:-0.7}"
layer1_dropout_channel_ratio="${LAYER1_DROPOUT_CHANNEL_RATIO:-0.3}"
layer1_dropout_hf_ratio="${LAYER1_DROPOUT_HF_RATIO:-0.35}"
layer1_dropout_eot_samples="${LAYER1_DROPOUT_EOT_SAMPLES:-1}"
layer1_dropout_eot_reduction="${LAYER1_DROPOUT_EOT_REDUCTION:-logits}"
load_cp="${LOAD_CP:-New}"
cp_path="${CP_PATH:-}"
run_transfer_eval="${RUN_TRANSFER_EVAL:-1}"
eval_batch_size="${EVAL_BATCH_SIZE:-128}"
eval_num_workers="${EVAL_NUM_WORKERS:-4}"
eval_samples="${EVAL_SAMPLES:-0}"
eval_state_every_batches="${EVAL_STATE_EVERY_BATCHES:-25}"
eval_device="${EVAL_DEVICE:-${device}}"
default_eval_models="dense161 vgg16 incv3 res50 WideRes50 EffNetB6 deit tnt Swin_Tiny twins vit deit_base tnt_base swin_small swin_base twins_base vit_small vit_base"
eval_models="${EVAL_MODELS:-${default_eval_models}}"
eval_resume="${EVAL_RESUME:-1}"
auto_install_eval_deps="${AUTO_INSTALL_EVAL_DEPS:-1}"
auto_download_eval_assets="${AUTO_DOWNLOAD_EVAL_ASSETS:-1}"
eval_deps_dir="${EVAL_DEPS_DIR:-}"
eval_deps_wheelhouse="${EVAL_DEPS_WHEELHOUSE:-}"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat >&2 <<EOF
usage: $0 [imagenet-train-dir] [imagenet-val-dir]

New DDSC-GPG train + transfer-ASR evaluation (default):
  bash $0 [imagenet-train-dir] [imagenet-val-dir]

Continue an architecture-mode DDSC-GPG run:
  LOAD_CP=Continue CP_PATH=/path/to/DDSC_gpg_res50_epoch_0011.train.pth \\
  EPOCHS=15 bash $0 [imagenet-train-dir] [imagenet-val-dir]

Training only:
  RUN_TRANSFER_EVAL=0 bash $0 [imagenet-train-dir]

Frequency-channel dropout for the GPG attack objective:
  LAYER1_DROPOUT_MODE=frequency_channel \
  LAYER1_DROPOUT_EOT_REDUCTION=loss bash $0 [train-dir] [val-dir]

EPOCHS is the total target epoch count, not the number of additional epochs.
All trajectory and runtime settings must match the checkpoint exactly.
Evaluation defaults to 18 unique victim implementations and requires
cached/downloadable pretrained weights. Set EVAL_MODELS to override the list.
Missing evaluator packages and the OpenMMLab ViT checkpoint are bootstrapped
by default. Set AUTO_INSTALL_EVAL_DEPS=0 or AUTO_DOWNLOAD_EVAL_ASSETS=0 to
require pre-provisioned dependencies or assets instead.
EOF
    exit 2
}

[[ "$#" -le 2 ]] || usage
case "${load_cp}" in
    New|Continue) ;;
    *) fail "LOAD_CP must be New or Continue" ;;
esac
case "${run_transfer_eval}" in
    0|1) ;;
    *) fail "RUN_TRANSFER_EVAL must be 0 or 1" ;;
esac
case "${eval_resume}" in
    0|1) ;;
    *) fail "EVAL_RESUME must be 0 or 1" ;;
esac
case "${auto_install_eval_deps}" in
    0|1) ;;
    *) fail "AUTO_INSTALL_EVAL_DEPS must be 0 or 1" ;;
esac
case "${auto_download_eval_assets}" in
    0|1) ;;
    *) fail "AUTO_DOWNLOAD_EVAL_ASSETS must be 0 or 1" ;;
esac
case "${layer1_dropout_mode}" in
    off|frequency_channel) ;;
    *) fail "LAYER1_DROPOUT_MODE must be off or frequency_channel" ;;
esac
case "${layer1_dropout_eot_reduction}" in
    logits|loss) ;;
    *) fail "LAYER1_DROPOUT_EOT_REDUCTION must be logits or loss" ;;
esac
if [[ "${layer1_dropout_mode}" == "frequency_channel" \
    && "${layer1_dropout_eot_reduction}" != "loss" ]]; then
    fail "GPG frequency-channel dropout requires LAYER1_DROPOUT_EOT_REDUCTION=loss"
fi

for integer_setting in \
    "EPOCHS=${epochs}" \
    "BATCH_SIZE=${batch_size}" \
    "NUM_WORKERS=${num_workers}" \
    "EVAL_BATCH_SIZE=${eval_batch_size}" \
    "EVAL_NUM_WORKERS=${eval_num_workers}" \
    "EVAL_SAMPLES=${eval_samples}" \
    "EVAL_STATE_EVERY_BATCHES=${eval_state_every_batches}" \
    "LAYER1_DROPOUT_EOT_SAMPLES=${layer1_dropout_eot_samples}"
do
    name="${integer_setting%%=*}"
    value="${integer_setting#*=}"
    [[ "${value}" =~ ^[0-9]+$ ]] || fail "${name} must be an integer"
done
(( epochs > 0 )) || fail "EPOCHS must be positive"
(( batch_size > 0 )) || fail "BATCH_SIZE must be positive"
(( eval_batch_size > 0 )) || fail "EVAL_BATCH_SIZE must be positive"
(( eval_state_every_batches > 0 )) || \
    fail "EVAL_STATE_EVERY_BATCHES must be positive"
(( layer1_dropout_eot_samples > 0 )) || \
    fail "LAYER1_DROPOUT_EOT_SAMPLES must be positive"

is_imagenet_imagefolder() {
    local candidate="$1"
    local class_count
    [[ -d "${candidate}" ]] || return 1
    class_count="$(find "${candidate}" -mindepth 1 -maxdepth 1 -type d | wc -l)"
    [[ "${class_count}" -eq 1000 ]]
}

resolve_imagenet_train_dir() {
    local explicit_path="${1:-}"
    local input_root
    local candidate
    local -a candidates=()

    if [[ -n "${explicit_path}" ]]; then
        is_imagenet_imagefolder "${explicit_path}" || {
            printf 'Explicit ImageNet path is not a 1000-class ImageFolder: %s\n' \
                "${explicit_path}" >&2
            return 1
        }
        printf '%s\n' "${explicit_path}"
        return 0
    fi

    if [[ -n "${IMAGENET_TRAIN_DIR:-}" ]]; then
        candidates+=("${IMAGENET_TRAIN_DIR}")
    fi
    candidates+=(
        "/app/data/ImageNet-2012/train"
        "${root}/data/imagenet/train"
        "${root}/data/ILSVRC2012_img_train"
        "/app/scratch/datasets/imagenet/train"
        "/app/scratch/dataset/imagenet/train"
        "/app/scratch/imagenet/train"
        "/app/scratch/ILSVRC2012_img_train"
        "/datasets/imagenet/train"
        "/datasets/ILSVRC2012_img_train"
        "/data/imagenet/train"
        "/data/ILSVRC2012_img_train"
    )

    input_root="$(dirname -- "${root}")"
    shopt -s nullglob
    candidates+=(
        "${input_root}"/*/train
        "${input_root}"/*/ILSVRC2012_img_train
        "${input_root}"/*/*/train
    )
    shopt -u nullglob

    for candidate in "${candidates[@]}"; do
        if is_imagenet_imagefolder "${candidate}"; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

train_dir="$(resolve_imagenet_train_dir "${1:-}")" || \
    fail "ImageNet train directory was not found"

resolve_imagenet_val_dir() {
    local explicit_path="${1:-}"
    local candidate
    local -a candidates=()

    if [[ -n "${explicit_path}" ]]; then
        is_imagenet_imagefolder "${explicit_path}" || {
            printf 'Explicit ImageNet val path is not a 1000-class ImageFolder: %s\n' \
                "${explicit_path}" >&2
            return 1
        }
        printf '%s\n' "${explicit_path}"
        return 0
    fi

    if [[ -n "${IMAGENET_VAL_ROOT:-}" ]]; then
        candidates+=("${IMAGENET_VAL_ROOT}")
    fi
    candidates+=(
        "/app/data/ImageNet-2012/val"
        "${root}/data/imagenet/val"
        "${root}/data/ILSVRC2012_img_val"
        "/app/scratch/datasets/imagenet/val"
        "/app/scratch/dataset/imagenet/val"
        "/app/scratch/imagenet/val"
        "/datasets/imagenet/val"
        "/data/imagenet/val"
    )
    for candidate in "${candidates[@]}"; do
        if is_imagenet_imagefolder "${candidate}"; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

val_dir=""
eval_model_args=()
eval_model_list=()
eval_pythonpath=""
if [[ "${run_transfer_eval}" == "1" ]]; then
    val_dir="$(resolve_imagenet_val_dir "${2:-}")" || \
        fail "ImageNet validation directory was not found"
    [[ -f "${evaluator}" ]] || fail "Evaluator is missing: ${evaluator}"
    [[ -f "${eval_bootstrapper}" ]] || \
        fail "Evaluation bootstrapper is missing: ${eval_bootstrapper}"
    "${python_bin}" - "${train_dir}" "${val_dir}" "${eval_samples}" <<'PY'
import sys
from pathlib import Path

from torchvision.datasets import ImageFolder

train_root, val_root = map(Path, sys.argv[1:3])
requested_samples = int(sys.argv[3])
train_classes = sorted(path.name for path in train_root.iterdir() if path.is_dir())
val_dataset = ImageFolder(str(val_root))
if train_classes != list(val_dataset.classes):
    raise SystemExit("ImageNet train/val class-directory mappings differ")
if requested_samples == 0 and len(val_dataset) != 50_000:
    raise SystemExit(
        f"full transfer evaluation requires 50000 validation images; "
        f"found={len(val_dataset)}"
    )
if requested_samples > len(val_dataset):
    raise SystemExit(
        f"EVAL_SAMPLES exceeds the validation set: requested={requested_samples}, "
        f"available={len(val_dataset)}"
    )
print(f"imagenet_val_samples={len(val_dataset)} classes={len(val_dataset.classes)}")
PY
    if [[ -n "${eval_models}" ]]; then
        read -r -a eval_model_list <<< "${eval_models}"
        (( ${#eval_model_list[@]} > 0 )) || fail "EVAL_MODELS is empty"
        eval_model_args+=(--models "${eval_model_list[@]}")
    fi
    needs_openmmlab_vit=0
    if [[ -z "${eval_models}" ]]; then
        needs_openmmlab_vit=1
    else
        for model_name in "${eval_model_list[@]}"; do
            [[ "${model_name}" == "vit" ]] && needs_openmmlab_vit=1
        done
    fi

    eval_python_tag="$(
        "${python_bin}" -c \
            'import platform, sys; print(f"py{sys.version_info.major}.{sys.version_info.minor}-{platform.machine()}")'
    )" || fail "Cannot determine the evaluator Python ABI"
    if [[ -z "${eval_deps_dir}" ]]; then
        eval_deps_dir="${root}/.cache/eval-deps/${eval_python_tag}"
    fi
    eval_pythonpath="${eval_deps_dir}${PYTHONPATH:+:${PYTHONPATH}}"
    eval_bootstrap_args=(
        "${python_bin}" "${eval_bootstrapper}"
        --deps-dir "${eval_deps_dir}"
        --auto-install-deps "${auto_install_eval_deps}"
        --auto-download-assets "${auto_download_eval_assets}"
    )
    if [[ -n "${eval_deps_wheelhouse}" ]]; then
        eval_bootstrap_args+=(--wheelhouse "${eval_deps_wheelhouse}")
    fi
    if [[ "${needs_openmmlab_vit}" == "1" ]]; then
        eval_bootstrap_args+=(
            --require-vit
            --vit-checkpoint "${openmmlab_vit_checkpoint}"
        )
    fi
    pip_cache_dir="${PIP_CACHE_DIR:-${root}/.cache/pip}"
    mkdir -p -- "${pip_cache_dir}" "${eval_deps_dir}"
    printf 'eval_dependency_target=%s auto_install=%s\n' \
        "${eval_deps_dir}" "${auto_install_eval_deps}"
    printf 'auto_download_eval_assets=%s\n' "${auto_download_eval_assets}"
    PIP_CACHE_DIR="${pip_cache_dir}" PYTHONPATH="${eval_pythonpath}" \
        "${eval_bootstrap_args[@]}" || \
        fail "Transfer evaluation dependency/asset bootstrap failed"
fi

checkpoint_args=()
if [[ "${load_cp}" == "Continue" ]]; then
    [[ -f "${cp_path}" ]] || fail "CP_PATH does not exist: ${cp_path}"
    [[ "${cp_path}" == *.train.pth ]] || \
        fail "CP_PATH must be a DDSC training sidecar ending in .train.pth"

    "${python_bin}" - "${cp_path}" "${lambda1_init}" \
        "${ddsc_restoring_gain}" <<'PY'
import math
import sys

import torch

checkpoint_path = sys.argv[1]
requested_lambda1 = float(sys.argv[2])
requested_gain_text = sys.argv[3]
try:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
except TypeError:
    payload = torch.load(checkpoint_path, map_location="cpu")
if not isinstance(payload, tuple) or len(payload) != 2:
    raise SystemExit("CP_PATH is not a DDSC training checkpoint tuple")
checkpoint_format, payload = payload
if checkpoint_format not in {"ddsc_gpg_training_v8", "ddsc_gpg_training_v9"}:
    raise SystemExit(
        "CP_PATH is not a compatible architecture-mode DDSC training checkpoint"
    )
if not isinstance(payload, dict) or payload.get("kind") != "training":
    raise SystemExit("CP_PATH does not contain a DDSC training payload")
train_args = payload.get("train_args")
if not isinstance(train_args, dict) or train_args.get("architecture_mode") != "gpg":
    raise SystemExit("CP_PATH was not created with architecture_mode=gpg")
stored_lambda1 = train_args.get("lam_1")
if not isinstance(stored_lambda1, (int, float)) or not math.isclose(
    float(stored_lambda1), requested_lambda1, rel_tol=0.0, abs_tol=0.0
):
    raise SystemExit(
        f"LAMBDA1_INIT must equal the checkpoint value: stored={stored_lambda1!r}, "
        f"requested={requested_lambda1!r}"
    )
stored_gain = train_args.get("ddsc_restoring_gain")
if not requested_gain_text:
    if stored_gain is not None:
        raise SystemExit(
            "Checkpoint used an explicit ddsc_restoring_gain; set "
            f"DDSC_RESTORING_GAIN={stored_gain}"
        )
else:
    requested_gain = float(requested_gain_text)
    if not isinstance(stored_gain, (int, float)) or not math.isclose(
        float(stored_gain), requested_gain, rel_tol=0.0, abs_tol=0.0
    ):
        raise SystemExit(
            "DDSC_RESTORING_GAIN must equal the checkpoint value: "
            f"stored={stored_gain!r}, requested={requested_gain!r}"
        )
PY
    checkpoint_args+=(--CP_path "${cp_path}")
elif [[ -n "${cp_path}" ]]; then
    fail "CP_PATH is set but LOAD_CP=New"
fi

restoring_gain_args=()
if [[ -n "${ddsc_restoring_gain}" ]]; then
    restoring_gain_args+=(--ddsc_restoring_gain "${ddsc_restoring_gain}")
fi

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export TORCH_HOME="${TORCH_HOME:-${root}/.cache/torch}"
export HF_HOME="${HF_HOME:-${root}/.cache/huggingface}"
export OPENMMLAB_VIT_CHECKPOINT="${openmmlab_vit_checkpoint}"
export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

mkdir -p -- "${TORCH_HOME}" "${HF_HOME}" "${out_root}"

printf 'architecture_mode=gpg\n'
printf 'imagenet_train_dir=%s\n' "${train_dir}"
printf 'load_cp=%s checkpoint=%s\n' "${load_cp}" "${cp_path:-none}"
printf 'epochs_total=%s batch_size=%s lambda1_init=%s\n' \
    "${epochs}" "${batch_size}" "${lambda1_init}"
printf 'ddsc_restoring_gain=%s\n' "${ddsc_restoring_gain:-default(lam_1)}"
printf 'layer1_dropout_mode=%s p=%s channel_ratio=%s hf_ratio=%s eot_samples=%s eot_reduction=%s\n' \
    "${layer1_dropout_mode}" "${layer1_dropout_p}" \
    "${layer1_dropout_channel_ratio}" "${layer1_dropout_hf_ratio}" \
    "${layer1_dropout_eot_samples}" "${layer1_dropout_eot_reduction}"
printf 'out_root=%s gpu_id=%s device=%s\n' \
    "${out_root}" "${gpu_id}" "${device}"
if [[ "${run_transfer_eval}" == "1" ]]; then
    printf 'imagenet_val_root=%s eval_device=%s eval_samples=%s\n' \
        "${val_dir}" "${eval_device}" "${eval_samples}"
fi

trainer_command=(
    "${python_bin}" -u "${trainer}"
    --train_dir "${train_dir}"
    --model_type res50
    --architecture_mode gpg
    --eps 10
    --target -1
    --batch_size "${batch_size}"
    --sample_per_class 0
    --n_iters 1
    --epochs "${epochs}"
    --lr 2.25e-5
    --lam_1 "${lambda1_init}"
    --lam_2 0.0001
    --lam_3 0.0001
    --pb full
    --load_CP "${load_cp}"
    "${checkpoint_args[@]}"
    --out-dir "${out_root}"
    --device "${device}"
    --num_workers "${num_workers}"
    --worker_timeout_seconds "${worker_timeout_seconds}"
    --seed 42
    --decoder_width 128
    --decoder_num_blocks 3
    --decoder_upsample_backend transpose
    --decoder_mode shared
    --layer1_dropout_mode "${layer1_dropout_mode}"
    --layer1_dropout_p "${layer1_dropout_p}"
    --layer1_dropout_channel_ratio "${layer1_dropout_channel_ratio}"
    --layer1_dropout_hf_ratio "${layer1_dropout_hf_ratio}"
    --layer1_dropout_eot_samples "${layer1_dropout_eot_samples}"
    --layer1_dropout_eot_reduction "${layer1_dropout_eot_reduction}"
    --save_every 1
    --ddsc_target_density "${ddsc_target_density}"
    --ddsc_warmup_epochs "${ddsc_warmup_epochs}"
    --ddsc_ema_decay 0.0
    --ddsc_mass 1.0
    --ddsc_damping "${ddsc_damping}"
    "${restoring_gain_args[@]}"
    --ddsc_dt 1.0
    --ddsc_lambda1_min 0.0
)

run_stamp="$(date +%Y%m%d_%H%M%S)"
train_stdout_log="${out_root}/gpg_ddsc_damping025_train_${run_stamp}.stdout.log"
printf 'trainer_command='
printf '%q ' "${trainer_command[@]}"
printf '\ntrain_stdout_log=%s\n' "${train_stdout_log}"
"${trainer_command[@]}" 2>&1 | tee "${train_stdout_log}"

training_output_path="$(
    sed -n 's/^output_path=//p' "${train_stdout_log}" | tail -n 1
)"
[[ -n "${training_output_path}" ]] || \
    fail "Trainer did not report output_path"
[[ -d "${training_output_path}" ]] || \
    fail "Trainer output directory does not exist: ${training_output_path}"
training_output_path="$(cd -- "${training_output_path}" && pwd -P)"
printf 'training_completed output_path=%s\n' "${training_output_path}"

if [[ "${run_transfer_eval}" == "0" ]]; then
    exit 0
fi

printf -v final_epoch_tag '%04d' "$((epochs - 1))"
inference_checkpoint="${training_output_path}/DDSC_gpg_res50_epoch_${final_epoch_tag}.inference.pth"
[[ -f "${inference_checkpoint}" ]] || \
    fail "Final inference checkpoint is missing: ${inference_checkpoint}"

eval_out_dir="${EVAL_OUT_DIR:-${training_output_path}/transfer_asr_18models}"
mkdir -p -- "${eval_out_dir}"
eval_resume_args=()
if [[ "${eval_resume}" == "0" ]]; then
    eval_resume_args+=(--no-resume)
fi
evaluator_command=(
    "${python_bin}" -u "${evaluator}"
    --checkpoint "${inference_checkpoint}"
    --imagenet-val-root "${val_dir}"
    --out-dir "${eval_out_dir}"
    "${eval_model_args[@]}"
    --batch-size "${eval_batch_size}"
    --num-workers "${eval_num_workers}"
    --samples "${eval_samples}"
    --seed 42
    --device "${eval_device}"
    --state-every-batches "${eval_state_every_batches}"
    "${eval_resume_args[@]}"
)

eval_stdout_log="${eval_out_dir}/evaluate_all_model_t.stdout.log"
printf 'evaluator_command='
printf '%q ' "${evaluator_command[@]}"
printf '\neval_stdout_log=%s\n' "${eval_stdout_log}"
PYTHONPATH="${eval_pythonpath}" \
    "${evaluator_command[@]}" 2>&1 | tee "${eval_stdout_log}"

results_json="${eval_out_dir}/results.json"
[[ -s "${results_json}" ]] || fail "Evaluation results are missing: ${results_json}"
"${python_bin}" - "${results_json}" <<'PY'
import json
import math
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
if payload.get("status") != "complete":
    raise SystemExit("transfer evaluation did not complete")
perturbation = payload.get("perturbation")
if not isinstance(perturbation, dict):
    raise SystemExit("transfer evaluation contains no perturbation metrics")
active_ratio = perturbation.get("mean_mask_active_ratio")
if (
    isinstance(active_ratio, bool)
    or not isinstance(active_ratio, (int, float))
    or not math.isfinite(float(active_ratio))
    or not 0.0 <= float(active_ratio) <= 1.0
):
    raise SystemExit("transfer evaluation contains an invalid mean mask active ratio")
rows = payload.get("models")
if not isinstance(rows, list) or not rows:
    raise SystemExit("transfer evaluation contains no model rows")
transfer = [row for row in rows if row.get("model_t") != "res50"]
if not transfer:
    raise SystemExit("transfer evaluation contains no non-source victim")
print(f"mean_active_pixel_percent={100.0 * float(active_ratio):.6f}")
print("transfer_asr_clean_correct_percent")
transfer_values = []
for row in transfer:
    value = row.get("untarget_asr_clean_correct_percent")
    if not isinstance(value, (int, float)):
        raise SystemExit(
            f"transfer ASR is undefined for {row.get('model_t')}: "
            "no clean-correct evaluation samples"
        )
    transfer_values.append(float(value))
    print(f"{row['model_t']}={float(value):.6f}")
mean_asr = sum(transfer_values) / len(transfer_values)
print(f"transfer_mean={mean_asr:.6f}")
PY
printf 'transfer_results_json=%s\n' "${results_json}"
printf 'transfer_results_csv=%s\n' "${eval_out_dir}/model_t_metrics.csv"
