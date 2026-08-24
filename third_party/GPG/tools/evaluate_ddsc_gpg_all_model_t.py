"""Evaluate one DDSC or source-generator checkpoint against every victim.

The attack is generated once per ImageNet batch and shared by all victim
models.  This preserves Eval_GPG's prediction-flip metric while avoiding eleven
repeated DDSC/source passes.  It also records ground-truth conditioned ASR and
correct per-image perturbation norms for reporting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchvision
from PIL import __version__ as PILLOW_VERSION
from timm import layers as timm_layers
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets.folder import pil_loader
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


GPG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = GPG_ROOT.parents[1]
if str(GPG_ROOT) not in sys.path:
    sys.path.insert(0, str(GPG_ROOT))

from DDSC_GPG_train import (  # noqa: E402
    attack_model_contract,
    build_generator_from_inference_checkpoint,
    validate_module_state_dict_finite,
)
from ddsc_architecture_modes import (  # noqa: E402
    EGSStructuredMask,
    architecture_mode_from_generator_type,
    build_original_generator,
    canonical_architecture_mode,
    egs_conditioner_contract,
    forward_generator_inference,
    generator_architecture_metadata,
    generator_type_for_architecture_mode,
    validate_egs_conditioner_contract,
)
from generators_ddsc_gpg import module_state_sha256  # noqa: E402
from generators_modify import GeneratorResnet as GPGGeneratorResnet  # noqa: E402


MODEL_ORDER = (
    "dense161",
    "vgg16",
    "incv3",
    "res50",
    "WideRes50",
    "EffNetB6",
    "deit",
    "tnt",
    "Swin_Tiny",
    "twins",
    "vit",
    "deit_small",
    "deit_base",
    "tnt_small",
    "tnt_base",
    "swin_small",
    "swin_base",
    "twins_small",
    "twins_base",
    "vit_small",
    "vit_base",
)

MODEL_IMPLEMENTATIONS = {
    "dense161": "torchvision.densenet161/IMAGENET1K_V1",
    "vgg16": "torchvision.vgg16/IMAGENET1K_V1",
    "incv3": "torchvision.inception_v3/IMAGENET1K_V1",
    "res50": "torchvision.resnet50/IMAGENET1K_V1",
    "WideRes50": "torchvision.wide_resnet50_2/IMAGENET1K_V1",
    "EffNetB6": "torchvision.efficientnet_b6/IMAGENET1K_V1",
    "deit": "timm/deit_small_patch16_224.fb_in1k",
    "tnt": "timm/tnt_s_patch16_224.in1k",
    "Swin_Tiny": "timm/swin_tiny_patch4_window7_224.ms_in1k",
    "twins": "timm/twins_pcpvt_small.in1k",
    "vit": "OpenMMLab/vit-base-p16_32xb128-mae_in1k",
    "deit_small": "timm/deit_small_patch16_224.fb_in1k",
    "deit_base": "timm/deit_base_patch16_224.fb_in1k",
    "tnt_small": "timm/tnt_s_patch16_224.in1k",
    "tnt_base": "timm/tnt_b_patch16_224.in1k",
    "swin_small": "timm/swin_small_patch4_window7_224.ms_in1k",
    "swin_base": "timm/swin_base_patch4_window7_224.ms_in1k",
    "twins_small": "timm/twins_pcpvt_small.in1k",
    "twins_base": "timm/twins_pcpvt_base.in1k",
    "vit_small": "timm/vit_small_patch16_224.augreg_in21k_ft_in1k",
    "vit_base": "timm/vit_base_patch16_224.augreg_in21k_ft_in1k",
}

_DEFAULT_OPENMMLAB_MAE_VIT_CHECKPOINT = (
    REPO_ROOT
    / "artifacts"
    / "pretrained"
    / "vit-base-p16_pt-32xb128-mae_in1k_20220623-4c544545.pth"
)
OPENMMLAB_MAE_VIT_CHECKPOINT = Path(
    os.environ.get(
        "OPENMMLAB_VIT_CHECKPOINT",
        str(_DEFAULT_OPENMMLAB_MAE_VIT_CHECKPOINT),
    )
).expanduser()
OPENMMLAB_MAE_VIT_SHA256 = (
    "4c544545d50657b87c62ca2de1a5da8d5f12abfc80e386bfb9a37dfbdf5b3e08"
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PROGRESS_STATE_SCHEMA = 3
EVALUATION_CONTRACT_SCHEMA = 4
METRICS_SCHEMA = 1
RAW_ARCHITECTURE_MODES = ("gpg", "tsaa", "egs_tsaa", "egs_tssa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a DDSC inference checkpoint or raw GPG/TSAA/EGS-TSSA "
            "state_dict against all Eval_GPG model_t values."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--architecture-mode",
        "--raw-architecture-mode",
        dest="architecture_mode",
        choices=RAW_ARCHITECTURE_MODES,
        help=(
            "generator architecture for a raw state_dict checkpoint; GPG can "
            "usually be inferred, while TSAA and EGS-TSSA require this option"
        ),
    )
    parser.add_argument(
        "--model-type",
        "--raw-model-type",
        dest="model_type",
        choices=("res50", "incv3"),
        help="source model type for a raw checkpoint",
    )
    parser.add_argument(
        "--eps-pixels",
        "--raw-eps-pixels",
        dest="eps_pixels",
        type=float,
        help="pixel-space perturbation budget for a raw checkpoint",
    )
    parser.add_argument(
        "--target",
        "--raw-target",
        dest="target",
        type=int,
        help="training target for a raw checkpoint; defaults to -1 (untargeted)",
    )
    parser.add_argument(
        "--completed-epoch",
        "--raw-completed-epoch",
        dest="completed_epoch",
        type=int,
        help="optional epoch provenance for a raw checkpoint; defaults to 0",
    )
    parser.add_argument(
        "--egs-tsaa-tk",
        "--egs-tssa-tk",
        dest="egs_tsaa_tk",
        type=float,
        help="EGS-TSSA top-k spatial fraction for a raw checkpoint; defaults to 0.6",
    )
    parser.add_argument(
        "--gpg-generator-mode",
        choices=("auto", "legacy", "isolated"),
        default="auto",
        help="GPG encoder mode for a raw checkpoint",
    )
    parser.add_argument("--imagenet-val-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_ORDER, default=list(MODEL_ORDER)
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--samples", type=int, default=0, help="0 evaluates all samples"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--state-every-batches", type=int, default=25)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _raw_state_dict_from_object(
    checkpoint_object: Any,
) -> tuple[dict[str, torch.Tensor], Mapping[str, Any], str]:
    """Extract a generator state_dict from common source-training formats."""

    if not isinstance(checkpoint_object, Mapping):
        raise ValueError(
            "raw checkpoint must be a state_dict or a mapping containing one"
        )
    metadata = checkpoint_object
    wrapper_key = "root"
    candidate: Any = checkpoint_object
    if not checkpoint_object or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in checkpoint_object.items()
    ):
        candidate = None
        for key in ("generator_state_dict", "state_dict", "netG", "model"):
            value = checkpoint_object.get(key)
            if (
                isinstance(value, Mapping)
                and value
                and all(
                    isinstance(name, str) and isinstance(tensor, torch.Tensor)
                    for name, tensor in value.items()
                )
            ):
                candidate = value
                wrapper_key = key
                break
    if not isinstance(candidate, Mapping) or not candidate:
        raise ValueError(
            "raw checkpoint does not contain a tensor-only generator state_dict"
        )
    state_dict = dict(candidate)
    if all(name.startswith("module.") for name in state_dict):
        state_dict = {
            name[len("module.") :]: tensor for name, tensor in state_dict.items()
        }
        wrapper_key = f"{wrapper_key}:module_prefix_stripped"
    validate_module_state_dict_finite(
        state_dict,
        field_name="raw_generator_state_dict",
    )
    return state_dict, metadata, wrapper_key


def _metadata_value(
    args: argparse.Namespace,
    metadata: Mapping[str, Any],
    argument_name: str,
    metadata_name: str,
    default: Any = None,
) -> Any:
    argument = getattr(args, argument_name, None)
    if argument is not None:
        return argument
    return metadata.get(metadata_name, default)


def _raw_architecture_mode(
    args: argparse.Namespace,
    metadata: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
) -> str:
    requested = getattr(args, "architecture_mode", None)
    if requested is not None:
        return canonical_architecture_mode(requested)
    generator_type = metadata.get("generator_type")
    if isinstance(generator_type, str):
        try:
            return architecture_mode_from_generator_type(generator_type)
        except ValueError:
            pass
    architecture = metadata.get("architecture")
    if isinstance(architecture, Mapping) and isinstance(
        architecture.get("architecture_mode"), str
    ):
        return canonical_architecture_mode(architecture["architecture_mode"])
    if any(name.startswith(("Grad_block", "isolated_encoder.")) for name in state_dict):
        return "gpg"
    raise ValueError(
        "raw TSAA and EGS-TSSA checkpoints have the same state schema; pass "
        "--architecture-mode tsaa or --architecture-mode egs_tsaa"
    )


def _gpg_encoder_mode(
    args: argparse.Namespace,
    state_dict: Mapping[str, torch.Tensor],
) -> str:
    inferred = (
        "isolated"
        if any(name.startswith("isolated_encoder.") for name in state_dict)
        else "legacy"
    )
    requested = getattr(args, "gpg_generator_mode", "auto")
    if requested == "auto":
        return inferred
    if requested != inferred:
        raise ValueError(
            "--gpg-generator-mode disagrees with the raw checkpoint state schema"
        )
    return requested


def _isolated_gpg_architecture_metadata(
    generator: torch.nn.Module,
    *,
    eps: float,
    inception: bool,
) -> dict[str, Any]:
    return {
        "generator_type": generator_type_for_architecture_mode("gpg"),
        "architecture_mode": "gpg",
        "source_method": "GPG",
        "class_name": "GeneratorResnet",
        "encoder_mode": "isolated",
        "inception_crop": inception,
        "eps": eps,
        "state_entry_count": len(generator.state_dict()),
        "parameter_count": sum(
            parameter.numel() for parameter in generator.parameters()
        ),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in generator.parameters()
            if parameter.requires_grad
        ),
    }


def build_generator_from_compatible_checkpoint(
    path: str | os.PathLike[str],
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, Mapping[str, Any]]:
    """Load either a self-describing DDSC checkpoint or a source raw state_dict."""

    with torch.inference_mode(False), torch.no_grad():
        checkpoint_object = torch.load(path, map_location="cpu", weights_only=True)
    if type(checkpoint_object) is tuple:
        return build_generator_from_inference_checkpoint(path)

    state_dict, metadata, wrapper_key = _raw_state_dict_from_object(checkpoint_object)
    architecture_mode = _raw_architecture_mode(args, metadata, state_dict)
    if architecture_mode == "simple":
        raise ValueError("raw DDSC simple-generator checkpoints are unsupported")
    model_type = _metadata_value(args, metadata, "model_type", "model_type")
    if model_type not in {"res50", "incv3"}:
        raise ValueError(
            "raw checkpoints do not encode the source input size; pass "
            "--model-type res50 or --model-type incv3"
        )
    eps_pixels = _metadata_value(args, metadata, "eps_pixels", "eps_pixels")
    if (
        not isinstance(eps_pixels, (int, float))
        or isinstance(eps_pixels, bool)
        or not math.isfinite(float(eps_pixels))
        or float(eps_pixels) <= 0.0
    ):
        raise ValueError(
            "raw checkpoints do not encode the perturbation budget; pass a "
            "finite positive --eps-pixels value"
        )
    eps_pixels = float(eps_pixels)
    eps = eps_pixels / 255.0
    inception = model_type == "incv3"

    gpg_encoder_mode = "legacy"
    if architecture_mode == "gpg":
        gpg_encoder_mode = _gpg_encoder_mode(args, state_dict)
    if architecture_mode == "gpg" and gpg_encoder_mode == "isolated":
        if model_type != "res50":
            raise ValueError("isolated GPG raw checkpoints require --model-type res50")
        generator = GPGGeneratorResnet(
            inception=False,
            eps=eps,
            evaluate=True,
            encoder_mode="isolated",
            encoder_backbone=torchvision.models.resnet50(weights=None),
        ).float()
    else:
        generator = build_original_generator(
            architecture_mode,
            inception=inception,
            eps=eps,
            inference=True,
        ).float()

    egs_tk: float | None = None
    if architecture_mode == "egs_tsaa":
        architecture = metadata.get("architecture")
        metadata_tk = (
            architecture.get("egs_tk") if isinstance(architecture, Mapping) else None
        )
        egs_tk = _metadata_value(
            args,
            metadata,
            "egs_tsaa_tk",
            "egs_tsaa_tk",
            metadata_tk if metadata_tk is not None else 0.6,
        )
        if (
            not isinstance(egs_tk, (int, float))
            or isinstance(egs_tk, bool)
            or not math.isfinite(float(egs_tk))
            or not 0.0 < float(egs_tk) <= 1.0
        ):
            raise ValueError("--egs-tsaa-tk must be finite and in (0, 1]")
        egs_tk = float(egs_tk)
        setattr(generator, "ddsc_egs_tk", egs_tk)

    try:
        generator.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"raw checkpoint state schema is incompatible with {architecture_mode}: {exc}"
        ) from exc
    validate_module_state_dict_finite(
        generator.state_dict(),
        field_name="loaded_raw_generator_state_dict",
    )
    generator.eval()
    if architecture_mode == "gpg" and gpg_encoder_mode == "isolated":
        architecture_metadata = _isolated_gpg_architecture_metadata(
            generator,
            eps=eps,
            inception=inception,
        )
    else:
        architecture_metadata = generator_architecture_metadata(
            generator,
            architecture_mode,
        )
    architecture_metadata = dict(architecture_metadata)
    architecture_metadata["raw_checkpoint_wrapper"] = wrapper_key

    target = _metadata_value(args, metadata, "target", "target", -1)
    completed_epoch = _metadata_value(
        args,
        metadata,
        "completed_epoch",
        "completed_epoch",
        0,
    )
    if type(target) is not int or target < -1 or target >= 1000:
        raise ValueError("--target must be -1 or an ImageNet class index")
    if type(completed_epoch) is not int or completed_epoch < 0:
        raise ValueError("--completed-epoch must be a nonnegative integer")
    stored_conditioner = metadata.get("conditioner_contract")
    if architecture_mode != "egs_tsaa" or not isinstance(stored_conditioner, Mapping):
        stored_conditioner = None
    payload = {
        "kind": "raw_state_dict",
        "generator_type": generator_type_for_architecture_mode(architecture_mode),
        "model_type": model_type,
        "target": target,
        "eps_pixels": eps_pixels,
        "image_size": 299 if model_type == "incv3" else 224,
        "completed_epoch": completed_epoch,
        "architecture": architecture_metadata,
        "conditioner_contract": stored_conditioner,
        "generator_state_dict": state_dict,
    }
    return generator, payload


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_openmmlab_mae_vit() -> torch.nn.Module:
    """Build the exact ImageNet classifier intended by Eval_GPG's vit entry."""
    checkpoint_path = OPENMMLAB_MAE_VIT_CHECKPOINT
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "missing official OpenMMLab ViT classifier checkpoint: "
            f"{checkpoint_path}"
        )
    actual_sha256 = sha256_file(checkpoint_path)
    if actual_sha256 != OPENMMLAB_MAE_VIT_SHA256:
        raise RuntimeError(
            "OpenMMLab ViT checkpoint hash mismatch: "
            f"expected {OPENMMLAB_MAE_VIT_SHA256}, got {actual_sha256}"
        )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise RuntimeError("unexpected OpenMMLab ViT checkpoint structure")

    converted: dict[str, torch.Tensor] = {}
    for key, tensor in payload["state_dict"].items():
        if key.startswith("data_preprocessor."):
            continue
        if key == "backbone.cls_token":
            converted["cls_token"] = tensor
        elif key == "backbone.pos_embed":
            converted["pos_embed"] = tensor
        elif key.startswith("backbone.patch_embed.projection."):
            converted[
                key.replace("backbone.patch_embed.projection.", "patch_embed.proj.")
            ] = tensor
        elif key.startswith("backbone.layers."):
            mapped = key.replace("backbone.layers.", "blocks.", 1)
            mapped = mapped.replace(".ln1.", ".norm1.")
            mapped = mapped.replace(".ln2.", ".norm2.")
            mapped = mapped.replace(".ffn.layers.0.0.", ".mlp.fc1.")
            mapped = mapped.replace(".ffn.layers.1.", ".mlp.fc2.")
            converted[mapped] = tensor
        elif key.startswith("backbone.ln1."):
            converted[key.replace("backbone.ln1.", "norm.")] = tensor
        elif key.startswith("head.layers.head."):
            converted[key.replace("head.layers.head.", "head.")] = tensor
        else:
            raise RuntimeError(f"unmapped OpenMMLab ViT state key: {key}")

    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    incompatible = model.load_state_dict(converted, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"ViT strict load failed: {incompatible}")
    return model


def build_model(name: str) -> torch.nn.Module:
    if name == "dense161":
        return torchvision.models.densenet161(
            weights=torchvision.models.DenseNet161_Weights.IMAGENET1K_V1
        )
    if name == "vgg16":
        return torchvision.models.vgg16(
            weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
        )
    if name == "incv3":
        return torchvision.models.inception_v3(
            weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
        )
    if name == "res50":
        return torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        )
    if name == "WideRes50":
        return torchvision.models.wide_resnet50_2(
            weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        )
    if name == "EffNetB6":
        return torchvision.models.efficientnet_b6(
            weights=torchvision.models.EfficientNet_B6_Weights.IMAGENET1K_V1
        )
    if name == "vit":
        return build_openmmlab_mae_vit()
    timm_names = {
        "deit": "deit_small_patch16_224.fb_in1k",
        "tnt": "tnt_s_patch16_224.in1k",
        "Swin_Tiny": "swin_tiny_patch4_window7_224.ms_in1k",
        "twins": "twins_pcpvt_small.in1k",
        "deit_small": "deit_small_patch16_224.fb_in1k",
        "deit_base": "deit_base_patch16_224.fb_in1k",
        "tnt_small": "tnt_s_patch16_224.in1k",
        "tnt_base": "tnt_b_patch16_224.in1k",
        "swin_small": "swin_small_patch4_window7_224.ms_in1k",
        "swin_base": "swin_base_patch4_window7_224.ms_in1k",
        "twins_small": "twins_pcpvt_small.in1k",
        "twins_base": "twins_pcpvt_base.in1k",
        "vit_small": "vit_small_patch16_224.augreg_in21k_ft_in1k",
        "vit_base": "vit_base_patch16_224.augreg_in21k_ft_in1k",
    }
    try:
        timm_name = timm_names[name]
    except KeyError as exc:
        raise ValueError(f"unsupported model_t: {name}") from exc
    return timm.create_model(timm_name, pretrained=True)


def victim_model_state_contract(model: torch.nn.Module) -> dict[str, Any]:
    """Hash a CPU victim state before the module is moved to the evaluator GPU.

    ``module_state_sha256`` sorts state keys and combines each key, dtype,
    shape, and contiguous tensor bytes.  Requiring CPU state here prevents the
    provenance pass from ever creating a second copy of a victim on GPU.
    """

    state_dict = model.state_dict()
    total_tensor_bytes = 0
    for name, tensor in state_dict.items():
        if not isinstance(name, str) or not name:
            raise ValueError("victim state_dict contains an invalid key")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"victim state_dict value is not a tensor: {name}")
        if tensor.device.type != "cpu":
            raise ValueError("victim state must be hashed before device transfer")
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise ValueError(f"victim state tensor layout is unsupported: {name}")
        total_tensor_bytes += tensor.numel() * tensor.element_size()
    return {
        "schema": 1,
        "kind": "sorted_key_dtype_shape_and_tensor_bytes_sha256",
        "tensor_count": len(state_dict),
        "total_tensor_bytes": total_tensor_bytes,
        "byte_order": sys.byteorder,
        "sha256": module_state_sha256(model),
    }


def victim_model_execution_contract(model: torch.nn.Module) -> dict[str, Any]:
    """Record non-state fused-attention choices that affect victim forwards."""

    entries: list[dict[str, Any]] = []
    for module_name, module in model.named_modules():
        if not hasattr(module, "fused_attn"):
            continue
        fused_attn = getattr(module, "fused_attn")
        if type(fused_attn) is not bool:
            raise ValueError(
                "victim fused_attn execution policy must be a plain boolean: "
                f"{module_name or '<root>'}"
            )
        entries.append({"name": module_name, "fused_attn": fused_attn})
    entries.sort(key=lambda entry: entry["name"])
    encoded = json.dumps(
        entries,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "schema": 1,
        "kind": "named_module_fused_attn_execution_v1",
        "fields": ["name", "fused_attn"],
        "entry_count": len(entries),
        "entries": entries,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def image_loader_contract() -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": "explicit_torchvision_pil_loader_v1",
        "callable": f"{pil_loader.__module__}.{pil_loader.__qualname__}",
        "forced_backend": "PIL",
        "torchvision_image_backend": str(torchvision.get_image_backend()),
    }


def input_transform_contract(image_size: int = 224) -> dict[str, Any]:
    if image_size not in {224, 299}:
        raise ValueError("generator image_size must be 224 or 299")
    resize_size = 256 if image_size == 224 else 300
    return {
        "schema": 1,
        "ordered_steps": [
            {
                "name": "Resize",
                "size": resize_size,
                "interpolation": "bilinear",
                "antialias": True,
            },
            {"name": "CenterCrop", "size": image_size},
            {"name": "ToTensor"},
        ],
        "normalization": {
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
        "victim_prediction_resize": {
            "incv3_size": [299, 299],
            "other_model_size": [224, 224],
            "mode": "bilinear",
            "align_corners": False,
        },
    }


def build_evaluation_transform(image_size: int = 224) -> transforms.Compose:
    if image_size not in {224, 299}:
        raise ValueError("generator image_size must be 224 or 299")
    resize_size = 256 if image_size == 224 else 300
    return transforms.Compose(
        [
            transforms.Resize(
                resize_size,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


def build_evaluation_dataset(
    data_root: Path,
    image_size: int = 224,
) -> datasets.ImageFolder:
    return datasets.ImageFolder(
        str(data_root),
        transform=build_evaluation_transform(image_size),
        loader=pil_loader,
    )


def normalize(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def predict(model: torch.nn.Module, name: str, images: torch.Tensor) -> torch.Tensor:
    prediction_size = 299 if name == "incv3" else 224
    if tuple(images.shape[-2:]) != (prediction_size, prediction_size):
        images = F.interpolate(
            images,
            size=(prediction_size, prediction_size),
            mode="bilinear",
            align_corners=False,
        )
    logits = model(normalize(images))
    if not isinstance(logits, torch.Tensor):
        logits = logits.logits
    return logits.argmax(dim=1)


def empty_model_metrics() -> dict[str, int]:
    return {
        "n": 0,
        "clean_correct": 0,
        "adv_correct": 0,
        "adv_correct_on_clean": 0,
        "attack_success_on_clean": 0,
        "prediction_flip_count": 0,
    }


def empty_perturbation_metrics() -> dict[str, float]:
    return {
        "n": 0,
        "mask_active_ratio_sum": 0.0,
        "applied_l0_sum": 0.0,
        "applied_l1_sum": 0.0,
        "applied_l2_sum": 0.0,
        "applied_linf_sum": 0.0,
        "actual_l0_sum": 0.0,
        "actual_l1_sum": 0.0,
        "actual_l2_sum": 0.0,
        "actual_linf_sum": 0.0,
        "legacy_l0_sum": 0.0,
        "legacy_l1_sum": 0.0,
        "legacy_l2_sum": 0.0,
        "legacy_linf_last_batch": 0.0,
    }


def update_model_metrics(
    metrics: dict[str, int],
    clean_prediction: torch.Tensor,
    adv_prediction: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    clean_correct = clean_prediction.eq(labels)
    adv_correct = adv_prediction.eq(labels)
    metrics["n"] += int(labels.numel())
    metrics["clean_correct"] += int(clean_correct.sum().item())
    metrics["adv_correct"] += int(adv_correct.sum().item())
    metrics["adv_correct_on_clean"] += int((clean_correct & adv_correct).sum().item())
    metrics["attack_success_on_clean"] += int(
        (clean_correct & ~adv_correct).sum().item()
    )
    metrics["prediction_flip_count"] += int(
        clean_prediction.ne(adv_prediction).sum().item()
    )


def per_sample_norms(tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    flat = tensor.flatten(1)
    return {
        "l0": flat.ne(0).sum(dim=1),
        "l1": flat.abs().sum(dim=1),
        "l2": flat.norm(p=2, dim=1),
        "linf": flat.abs().amax(dim=1),
    }


def update_perturbation_metrics(
    metrics: dict[str, float],
    images: torch.Tensor,
    adv: torch.Tensor,
    adv_inf: torch.Tensor,
    adv_0: torch.Tensor,
) -> None:
    batch = int(images.shape[0])
    applied = adv_0 * adv_inf
    actual = adv - images
    applied_norms = per_sample_norms(applied)
    actual_norms = per_sample_norms(actual)
    metrics["n"] += batch
    metrics["mask_active_ratio_sum"] += float(
        adv_0.flatten(1).ne(0).float().mean(dim=1).sum().item()
    )
    for key in ("l0", "l1", "l2", "linf"):
        metrics[f"applied_{key}_sum"] += float(applied_norms[key].sum().item())
        metrics[f"actual_{key}_sum"] += float(actual_norms[key].sum().item())
    # Preserve the exact legacy Eval_GPG aggregation for comparability.
    metrics["legacy_l0_sum"] += float(torch.norm(adv_0, p=0).item())
    metrics["legacy_l1_sum"] += float(torch.norm(applied, p=1).item())
    metrics["legacy_l2_sum"] += float(torch.norm(applied, p=2).item())
    metrics["legacy_linf_last_batch"] = float(
        torch.norm(applied, p=float("inf")).item()
    )


def finalize_model(name: str, metrics: dict[str, int]) -> dict[str, Any]:
    n = metrics["n"]
    clean = metrics["clean_correct"]
    return {
        "model_t": name,
        "implementation": MODEL_IMPLEMENTATIONS[name],
        **metrics,
        "clean_accuracy_percent": 100.0 * clean / n,
        "adv_accuracy_percent": 100.0 * metrics["adv_correct"] / n,
        "prediction_flip_rate_percent": 100.0 * metrics["prediction_flip_count"] / n,
        "untarget_asr_clean_correct_percent": (
            100.0 * metrics["attack_success_on_clean"] / clean if clean else None
        ),
    }


def finalize_perturbation(
    metrics: dict[str, float], image_size: int
) -> dict[str, float]:
    n = int(metrics["n"])
    result = dict(metrics)
    result.update(
        {
            "mean_mask_active_ratio": metrics["mask_active_ratio_sum"] / n,
            "mean_applied_l0": metrics["applied_l0_sum"] / n,
            "mean_applied_l1": metrics["applied_l1_sum"] / n,
            "mean_applied_l2": metrics["applied_l2_sum"] / n,
            "mean_applied_linf": metrics["applied_linf_sum"] / n,
            "mean_actual_l0": metrics["actual_l0_sum"] / n,
            "mean_actual_l1": metrics["actual_l1_sum"] / n,
            "mean_actual_l2": metrics["actual_l2_sum"] / n,
            "mean_actual_linf": metrics["actual_linf_sum"] / n,
            "legacy_sparsity_l0": metrics["legacy_l0_sum"]
            / n
            / (image_size * image_size),
            "legacy_l1": metrics["legacy_l1_sum"] / n,
            "legacy_l2_batch_dependent": metrics["legacy_l2_sum"] / n,
            "legacy_linf_last_batch": metrics["legacy_linf_last_batch"],
        }
    )
    return result


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _canonical_json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_json_value(value: Any) -> Any:
    """Return the exact JSON representation used by persisted progress files."""

    return json.loads(_canonical_json_text(value))


def evaluator_source_contract() -> dict[str, Any]:
    """Fingerprint every local source file that defines evaluation semantics."""

    source_paths = {
        "evaluator": Path(__file__).resolve(),
        "checkpoint_loader": GPG_ROOT / "DDSC_GPG_train.py",
        "architecture_adapter": GPG_ROOT / "ddsc_architecture_modes.py",
        "simple_generator": GPG_ROOT / "generators_ddsc_gpg.py",
        "frequency_dropout": GPG_ROOT / "ddsc_layer1_dropout_eot.py",
    }
    contract: dict[str, Any] = {}
    repository_root = REPO_ROOT.resolve()
    for name, source_path in source_paths.items():
        resolved = source_path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"evaluation source is missing: {resolved}")
        try:
            display_path = resolved.relative_to(repository_root).as_posix()
        except ValueError:
            display_path = str(resolved)
        contract[name] = {
            "path": display_path,
            "sha256": sha256_file(resolved),
        }
    return contract


def _optional_bool_attribute(owner: Any, name: str) -> dict[str, Any]:
    if not hasattr(owner, name):
        return {"available": False, "value": None}
    value = getattr(owner, name)
    if type(value) is not bool:
        raise RuntimeError(f"runtime policy {name} is not a plain boolean")
    return {"available": True, "value": value}


def _optional_bool_call(owner: Any, name: str) -> dict[str, Any]:
    function = getattr(owner, name, None)
    if not callable(function):
        return {"available": False, "value": None}
    value = function()
    if type(value) is not bool:
        raise RuntimeError(f"runtime policy {name} did not return a plain boolean")
    return {"available": True, "value": value}


def _optional_string_call(owner: Any, name: str) -> dict[str, Any]:
    function = getattr(owner, name, None)
    if not callable(function):
        return {"available": False, "value": None}
    value = function()
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"runtime policy {name} did not return a nonempty string")
    return {"available": True, "value": value}


def _autocast_device_contract(device_type: str) -> dict[str, Any]:
    enabled_function = getattr(torch, "is_autocast_enabled", None)
    enabled: bool | None = None
    if callable(enabled_function):
        try:
            enabled = enabled_function(device_type)
        except TypeError:
            legacy_enabled_name = (
                "is_autocast_cpu_enabled"
                if device_type == "cpu"
                else "is_autocast_enabled"
            )
            legacy_enabled = getattr(torch, legacy_enabled_name, None)
            if callable(legacy_enabled):
                enabled = legacy_enabled()
    if enabled is not None and type(enabled) is not bool:
        raise RuntimeError("autocast enabled policy is not a plain boolean")

    dtype_function = getattr(torch, "get_autocast_dtype", None)
    dtype: torch.dtype | None = None
    if callable(dtype_function):
        try:
            dtype = dtype_function(device_type)
        except TypeError:
            dtype = None
    if dtype is None:
        legacy_dtype_name = (
            "get_autocast_cpu_dtype"
            if device_type == "cpu"
            else "get_autocast_gpu_dtype"
        )
        legacy_dtype = getattr(torch, legacy_dtype_name, None)
        if callable(legacy_dtype):
            dtype = legacy_dtype()
    if dtype is not None and not isinstance(dtype, torch.dtype):
        raise RuntimeError("autocast dtype policy did not return torch.dtype")
    return {
        "enabled": {
            "available": enabled is not None,
            "value": enabled,
        },
        "dtype": {
            "available": dtype is not None,
            "value": str(dtype) if dtype is not None else None,
        },
    }


def framework_device_contract(device: torch.device) -> dict[str, Any]:
    """Capture framework and numerical-runtime choices that can affect outputs."""

    device_index: int | None = None
    device_details: dict[str, Any] = {"type": device.type, "index": None}
    if device.type == "cuda":
        device_index = (
            torch.cuda.current_device() if device.index is None else int(device.index)
        )
        properties = torch.cuda.get_device_properties(device_index)
        device_details.update(
            {
                "index": device_index,
                "name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "total_memory_bytes": properties.total_memory,
                "multi_processor_count": properties.multi_processor_count,
            }
        )
    else:
        if device.index is not None:
            device_details["index"] = int(device.index)
        device_details.update(
            {
                "machine": platform.machine(),
                "processor": platform.processor(),
            }
        )

    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pillow_version": PILLOW_VERSION,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "timm_version": timm.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "default_dtype": str(torch.get_default_dtype()),
        "evaluation_dtype": "torch.float32",
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_algorithms_warn_only": _optional_bool_call(
            torch, "is_deterministic_algorithms_warn_only_enabled"
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_enabled": _optional_bool_attribute(torch.backends.cudnn, "enabled"),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cuda_matmul_allow_fp16_reduced_precision_reduction": (
            _optional_bool_attribute(
                torch.backends.cuda.matmul,
                "allow_fp16_reduced_precision_reduction",
            )
        ),
        "cuda_matmul_allow_bf16_reduced_precision_reduction": (
            _optional_bool_attribute(
                torch.backends.cuda.matmul,
                "allow_bf16_reduced_precision_reduction",
            )
        ),
        "cuda_matmul_allow_fp16_accumulation": _optional_bool_attribute(
            torch.backends.cuda.matmul,
            "allow_fp16_accumulation",
        ),
        "mkldnn_enabled": _optional_bool_attribute(torch.backends.mkldnn, "enabled"),
        "mkldnn_deterministic": _optional_bool_attribute(
            torch.backends.mkldnn, "deterministic"
        ),
        "float32_matmul_precision": _optional_string_call(
            torch, "get_float32_matmul_precision"
        ),
        "timm_use_fused_attn": _optional_bool_call(timm_layers, "use_fused_attn"),
        "sdpa": {
            "flash": _optional_bool_call(torch.backends.cuda, "flash_sdp_enabled"),
            "memory_efficient": _optional_bool_call(
                torch.backends.cuda, "mem_efficient_sdp_enabled"
            ),
            "math": _optional_bool_call(torch.backends.cuda, "math_sdp_enabled"),
            "cudnn": _optional_bool_call(torch.backends.cuda, "cudnn_sdp_enabled"),
            "fp16_bf16_reduction_math": _optional_bool_call(
                torch.backends.cuda,
                "fp16_bf16_reduction_math_sdp_allowed",
            ),
        },
        "autocast": {
            "cpu": _autocast_device_contract("cpu"),
            "cuda": _autocast_device_contract("cuda"),
            "cache_enabled": _optional_bool_call(torch, "is_autocast_cache_enabled"),
        },
        "device": device_details,
    }


def ordered_dataset_manifest(
    dataset: datasets.ImageFolder,
    data_root: Path,
    dataset_n: int,
) -> dict[str, Any]:
    """Hash the exact ordered ImageFolder prefix consumed by this evaluation.

    The digest covers normalized relative paths, class indices, file sizes, and
    streaming SHA-256 values of the image bytes.
    """

    if type(dataset_n) is not int or dataset_n <= 0 or dataset_n > len(dataset.samples):
        raise ValueError("dataset_n is outside the ImageFolder sample range")
    root = data_root.resolve()
    digest = hashlib.sha256()
    for ordinal, (sample_path, class_index) in enumerate(dataset.samples[:dataset_n]):
        resolved = Path(sample_path).resolve()
        try:
            relative_path = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"ImageFolder sample escapes the declared dataset root: {resolved}"
            ) from exc
        if type(class_index) is not int or class_index < 0:
            raise ValueError(f"invalid ImageFolder class index at ordinal {ordinal}")
        stat_before = resolved.stat()
        file_size = stat_before.st_size
        content_sha256 = sha256_file(resolved)
        stat_after = resolved.stat()
        if (
            stat_after.st_size != stat_before.st_size
            or stat_after.st_mtime_ns != stat_before.st_mtime_ns
        ):
            raise RuntimeError(
                f"ImageFolder sample changed while it was fingerprinted: {resolved}"
            )
        record = json.dumps(
            [ordinal, relative_path, class_index, file_size, content_sha256],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(record).to_bytes(8, byteorder="big", signed=False))
        digest.update(record)

    class_contract = {
        "classes": list(dataset.classes),
        "class_to_idx": dict(dataset.class_to_idx),
    }
    class_bytes = json.dumps(
        class_contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema": 2,
        "kind": "ordered_imagefolder_prefix_with_content_sha256",
        "sample_count": dataset_n,
        "record_fields": [
            "ordinal",
            "relative_path",
            "class_index",
            "file_size",
            "content_sha256",
        ],
        "record_encoding": "length-prefixed canonical-json/utf-8",
        "ordered_records_sha256": digest.hexdigest(),
        "class_count": len(dataset.classes),
        "class_mapping_sha256": hashlib.sha256(class_bytes).hexdigest(),
    }


def metrics_contract(models: list[str]) -> dict[str, Any]:
    final_model_metric_keys = [
        "model_t",
        "implementation",
        *empty_model_metrics(),
        "clean_accuracy_percent",
        "adv_accuracy_percent",
        "prediction_flip_rate_percent",
        "untarget_asr_clean_correct_percent",
    ]
    final_perturbation_metric_keys = [
        *empty_perturbation_metrics(),
        "mean_mask_active_ratio",
        "mean_applied_l0",
        "mean_applied_l1",
        "mean_applied_l2",
        "mean_applied_linf",
        "mean_actual_l0",
        "mean_actual_l1",
        "mean_actual_l2",
        "mean_actual_linf",
        "legacy_sparsity_l0",
        "legacy_l1",
        "legacy_l2_batch_dependent",
        "legacy_linf_last_batch",
    ]
    return {
        "schema": METRICS_SCHEMA,
        "model_order": list(models),
        "model_metric_keys": list(empty_model_metrics()),
        "perturbation_metric_keys": list(empty_perturbation_metrics()),
        "final_model_metric_keys": final_model_metric_keys,
        "final_perturbation_metric_keys": final_perturbation_metric_keys,
        "model_metric_values": "nonnegative plain integers; n equals next_index",
        "perturbation_metric_values": (
            "n is a nonnegative plain integer; remaining values are finite "
            "nonnegative JSON numbers"
        ),
        "prediction_metric_semantics": "Eval_GPG untargeted argmax/v1",
        "perturbation_metric_semantics": "per-sample plus legacy Eval_GPG/v1",
    }


def state_contract(
    args: argparse.Namespace,
    *,
    checkpoint_sha256: str,
    checkpoint_payload: Any,
    dataset_manifest: dict[str, Any],
    device: torch.device,
    victim_model_contracts: dict[str, dict[str, Any]],
    victim_model_execution_contracts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    architecture_mode = architecture_mode_from_generator_type(
        checkpoint_payload["generator_type"]
    )
    if list(victim_model_contracts) != list(args.models):
        raise ValueError("victim model state-contract order differs from --models")
    if list(victim_model_execution_contracts) != list(args.models):
        raise ValueError("victim model execution-contract order differs from --models")
    checkpoint_metadata = {
        key: checkpoint_payload[key]
        for key in (
            "kind",
            "generator_type",
            "model_type",
            "target",
            "eps_pixels",
            "image_size",
            "completed_epoch",
            "architecture",
            "conditioner_contract",
        )
    }
    contract = {
        "schema": EVALUATION_CONTRACT_SCHEMA,
        "evaluator_sources": evaluator_source_contract(),
        "framework_and_device": framework_device_contract(device),
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "sha256": checkpoint_sha256,
            "metadata": checkpoint_metadata,
        },
        "architecture_mode": architecture_mode,
        "egs_conditioner_contract": checkpoint_payload["conditioner_contract"],
        "dataset": {
            "root": str(Path(args.imagenet_val_root).resolve()),
            "requested_samples": args.samples,
            "ordered_manifest": dataset_manifest,
            "image_loader": image_loader_contract(),
        },
        "models": [
            {
                "name": name,
                "implementation": MODEL_IMPLEMENTATIONS[name],
                "state_dict": victim_model_contracts[name],
                "execution": victim_model_execution_contracts[name],
            }
            for name in args.models
        ],
        "openmmlab_mae_vit": {
            "checkpoint_path": str(OPENMMLAB_MAE_VIT_CHECKPOINT.resolve()),
            "expected_sha256": OPENMMLAB_MAE_VIT_SHA256,
        },
        "input_transform": input_transform_contract(
            int(checkpoint_payload["image_size"])
        ),
        "data_loader": {
            "subset": "range(next_index, dataset.sample_count)",
            "batch_size": args.batch_size,
            "shuffle": False,
            "num_workers": args.num_workers,
            "pin_memory": device.type == "cuda",
            "drop_last": False,
            "persistent_workers": args.num_workers > 0,
        },
        "seed": args.seed,
        "metrics": metrics_contract(list(args.models)),
    }
    return _canonical_json_value(contract)


def _require_plain_int(value: Any, field_name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"saved progress {field_name} must be a plain integer")
    return value


def _availability_record_is_valid(
    record: Any,
    *,
    value_type: type,
) -> bool:
    if not isinstance(record, dict) or set(record) != {"available", "value"}:
        return False
    available = record["available"]
    if type(available) is not bool:
        return False
    value = record["value"]
    if not available:
        return value is None
    if value_type is bool:
        return type(value) is bool
    return isinstance(value, value_type) and (value_type is not str or bool(value))


def _victim_execution_contract_is_valid(contract: Any) -> bool:
    required_keys = {
        "schema",
        "kind",
        "fields",
        "entry_count",
        "entries",
        "sha256",
    }
    if not isinstance(contract, dict) or set(contract) != required_keys:
        return False
    if type(contract["schema"]) is not int or contract["schema"] != 1:
        return False
    if contract["kind"] != "named_module_fused_attn_execution_v1":
        return False
    if contract["fields"] != ["name", "fused_attn"]:
        return False
    entries = contract["entries"]
    if not isinstance(entries, list):
        return False
    if type(contract["entry_count"]) is not int or contract["entry_count"] != len(
        entries
    ):
        return False
    names: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"name", "fused_attn"}:
            return False
        if not isinstance(entry["name"], str) or type(entry["fused_attn"]) is not bool:
            return False
        names.append(entry["name"])
    if names != sorted(names) or len(names) != len(set(names)):
        return False
    encoded = json.dumps(
        entries,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    expected_sha256 = hashlib.sha256(encoded).hexdigest()
    return contract["sha256"] == expected_sha256


def _validate_framework_policy_contract(contract: Any) -> None:
    if not isinstance(contract, dict):
        raise ValueError("evaluation framework contract must be a JSON object")
    plain_bool_keys = {
        "deterministic_algorithms",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cudnn_allow_tf32",
        "cuda_matmul_allow_tf32",
    }
    if any(type(contract.get(key)) is not bool for key in plain_bool_keys):
        raise ValueError("evaluation framework boolean policy is malformed")
    for key in (
        "deterministic_algorithms_warn_only",
        "cudnn_enabled",
        "mkldnn_enabled",
        "mkldnn_deterministic",
        "timm_use_fused_attn",
        "cuda_matmul_allow_fp16_reduced_precision_reduction",
        "cuda_matmul_allow_bf16_reduced_precision_reduction",
        "cuda_matmul_allow_fp16_accumulation",
    ):
        if not _availability_record_is_valid(contract.get(key), value_type=bool):
            raise ValueError(f"evaluation framework policy is malformed: {key}")
    if not _availability_record_is_valid(
        contract.get("float32_matmul_precision"), value_type=str
    ):
        raise ValueError("evaluation float32 matmul policy is malformed")
    sdpa = contract.get("sdpa")
    if not isinstance(sdpa, dict) or set(sdpa) != {
        "flash",
        "memory_efficient",
        "math",
        "cudnn",
        "fp16_bf16_reduction_math",
    }:
        raise ValueError("evaluation SDPA policy keys are malformed")
    if any(
        not _availability_record_is_valid(sdpa[key], value_type=bool) for key in sdpa
    ):
        raise ValueError("evaluation SDPA policy value is malformed")
    autocast = contract.get("autocast")
    if not isinstance(autocast, dict) or set(autocast) != {
        "cpu",
        "cuda",
        "cache_enabled",
    }:
        raise ValueError("evaluation autocast policy keys are malformed")
    if not _availability_record_is_valid(autocast["cache_enabled"], value_type=bool):
        raise ValueError("evaluation autocast cache policy is malformed")
    for device_type in ("cpu", "cuda"):
        device_policy = autocast[device_type]
        if not isinstance(device_policy, dict) or set(device_policy) != {
            "enabled",
            "dtype",
        }:
            raise ValueError("evaluation autocast device policy keys are malformed")
        if not _availability_record_is_valid(
            device_policy["enabled"], value_type=bool
        ) or not _availability_record_is_valid(device_policy["dtype"], value_type=str):
            raise ValueError("evaluation autocast device policy is malformed")


def validate_progress_state(
    saved: Any,
    *,
    expected_contract: dict[str, Any],
    dataset_n: int,
    model_names: list[str],
) -> tuple[int, dict[str, dict[str, int]], dict[str, float]]:
    """Validate a persisted accumulator before any value is trusted."""

    if type(dataset_n) is not int or dataset_n <= 0:
        raise ValueError("dataset_n must be a positive plain integer")
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("model_names must be nonempty and unique")
    if not isinstance(expected_contract, dict):
        raise ValueError("evaluation contract must be a JSON object")
    if (
        type(expected_contract.get("schema")) is not int
        or expected_contract.get("schema") != EVALUATION_CONTRACT_SCHEMA
    ):
        raise ValueError("evaluation contract schema is invalid")
    _validate_framework_policy_contract(expected_contract.get("framework_and_device"))
    contract_dataset = expected_contract.get("dataset")
    contract_manifest = (
        contract_dataset.get("ordered_manifest")
        if isinstance(contract_dataset, dict)
        else None
    )
    contract_sample_count = (
        contract_manifest.get("sample_count")
        if isinstance(contract_manifest, dict)
        else None
    )
    if type(contract_sample_count) is not int or contract_sample_count != dataset_n:
        raise ValueError("evaluation contract dataset sample count is inconsistent")
    if (
        type(contract_manifest.get("schema")) is not int
        or contract_manifest.get("schema") != 2
        or contract_manifest.get("kind")
        != "ordered_imagefolder_prefix_with_content_sha256"
        or contract_manifest.get("record_fields")
        != [
            "ordinal",
            "relative_path",
            "class_index",
            "file_size",
            "content_sha256",
        ]
    ):
        raise ValueError("evaluation contract dataset manifest schema is inconsistent")
    loader_contract = contract_dataset.get("image_loader")
    if (
        not isinstance(loader_contract, dict)
        or set(loader_contract)
        != {
            "schema",
            "kind",
            "callable",
            "forced_backend",
            "torchvision_image_backend",
        }
        or type(loader_contract["schema"]) is not int
        or loader_contract["schema"] != 1
        or loader_contract["kind"] != "explicit_torchvision_pil_loader_v1"
        or loader_contract["callable"]
        != f"{pil_loader.__module__}.{pil_loader.__qualname__}"
        or loader_contract["forced_backend"] != "PIL"
        or not isinstance(loader_contract["torchvision_image_backend"], str)
    ):
        raise ValueError("evaluation contract image loader policy is inconsistent")
    try:
        transform_matches = _canonical_json_text(
            expected_contract.get("input_transform")
        ) == _canonical_json_text(input_transform_contract())
    except (TypeError, ValueError):
        transform_matches = False
    if not transform_matches:
        raise ValueError("evaluation contract input transform is inconsistent")
    contract_models = expected_contract.get("models")
    if (
        not isinstance(contract_models, list)
        or len(contract_models) != len(model_names)
        or any(
            not isinstance(entry, dict)
            or set(entry) != {"name", "implementation", "state_dict", "execution"}
            or entry.get("name") != model_name
            or entry.get("implementation") != MODEL_IMPLEMENTATIONS[model_name]
            or not isinstance(entry.get("state_dict"), dict)
            or set(entry["state_dict"])
            != {
                "schema",
                "kind",
                "tensor_count",
                "total_tensor_bytes",
                "byte_order",
                "sha256",
            }
            or type(entry["state_dict"].get("schema")) is not int
            or entry["state_dict"].get("schema") != 1
            or entry["state_dict"].get("kind")
            != "sorted_key_dtype_shape_and_tensor_bytes_sha256"
            or type(entry["state_dict"].get("tensor_count")) is not int
            or entry["state_dict"].get("tensor_count", 0) <= 0
            or type(entry["state_dict"].get("total_tensor_bytes")) is not int
            or entry["state_dict"].get("total_tensor_bytes", 0) <= 0
            or entry["state_dict"].get("byte_order") not in {"little", "big"}
            or not isinstance(entry["state_dict"].get("sha256"), str)
            or len(entry["state_dict"].get("sha256", "")) != 64
            or any(
                character not in "0123456789abcdef"
                for character in entry["state_dict"].get("sha256", "")
            )
            or not _victim_execution_contract_is_valid(entry.get("execution"))
            for entry, model_name in zip(contract_models, model_names)
        )
    ):
        raise ValueError("evaluation contract model order is inconsistent")
    contract_metrics = expected_contract.get("metrics", {})
    if (
        not isinstance(contract_metrics, dict)
        or type(contract_metrics.get("schema")) is not int
        or contract_metrics.get("schema") != METRICS_SCHEMA
        or contract_metrics.get("model_order") != model_names
        or contract_metrics.get("model_metric_keys") != list(empty_model_metrics())
        or contract_metrics.get("perturbation_metric_keys")
        != list(empty_perturbation_metrics())
    ):
        raise ValueError("evaluation contract metrics schema is inconsistent")
    if not isinstance(saved, dict):
        raise ValueError("saved progress must be a JSON object")
    required_keys = {
        "schema",
        "contract",
        "next_index",
        "model_metrics",
        "perturbation_metrics",
    }
    allowed_keys = required_keys | {"complete"}
    saved_keys = set(saved)
    if not required_keys.issubset(saved_keys) or not saved_keys.issubset(allowed_keys):
        raise ValueError(
            "saved progress keys mismatch: "
            f"missing={sorted(required_keys - saved_keys)}, "
            f"unexpected={sorted(saved_keys - allowed_keys)}"
        )
    schema = _require_plain_int(saved["schema"], "schema")
    if schema != PROGRESS_STATE_SCHEMA:
        raise ValueError(f"unsupported saved progress schema: {schema}")
    try:
        contract_matches = _canonical_json_text(
            saved["contract"]
        ) == _canonical_json_text(expected_contract)
    except (TypeError, ValueError):
        contract_matches = False
    if not contract_matches:
        raise ValueError("saved progress contract differs from this evaluation")

    next_index = _require_plain_int(saved["next_index"], "next_index")
    if next_index < 0 or next_index > dataset_n:
        raise ValueError("saved progress next_index is outside the dataset range")
    batch_size = expected_contract.get("data_loader", {}).get("batch_size")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("evaluation contract has an invalid batch size")
    if next_index != dataset_n and next_index % batch_size != 0:
        raise ValueError("saved progress next_index is not a completed batch boundary")
    if "complete" in saved:
        if saved["complete"] is not True:
            raise ValueError("saved progress complete must be true when present")
        if next_index != dataset_n:
            raise ValueError("complete saved progress does not cover the dataset")

    model_metrics = saved["model_metrics"]
    if not isinstance(model_metrics, dict) or list(model_metrics) != model_names:
        raise ValueError("saved progress model metric names/order mismatch")
    expected_model_keys = list(empty_model_metrics())
    for model_name in model_names:
        values = model_metrics[model_name]
        if not isinstance(values, dict) or list(values) != expected_model_keys:
            raise ValueError(
                f"saved progress metric keys mismatch for model {model_name}"
            )
        for metric_name, value in values.items():
            count = _require_plain_int(
                value, f"model_metrics.{model_name}.{metric_name}"
            )
            if count < 0:
                raise ValueError(
                    f"saved progress model metric is negative: {model_name}.{metric_name}"
                )
        n = values["n"]
        if n != next_index:
            raise ValueError(f"saved progress n mismatch for model {model_name}")
        for metric_name in expected_model_keys[1:]:
            if values[metric_name] > n:
                raise ValueError(
                    f"saved progress count exceeds n: {model_name}.{metric_name}"
                )
        if values["adv_correct_on_clean"] > values["clean_correct"]:
            raise ValueError("saved adv_correct_on_clean exceeds clean_correct")
        if values["adv_correct_on_clean"] > values["adv_correct"]:
            raise ValueError("saved adv_correct_on_clean exceeds adv_correct")
        if values["attack_success_on_clean"] != (
            values["clean_correct"] - values["adv_correct_on_clean"]
        ):
            raise ValueError("saved clean-conditioned attack counts are inconsistent")
        minimum_flips = (
            values["attack_success_on_clean"]
            + values["adv_correct"]
            - values["adv_correct_on_clean"]
        )
        if values["prediction_flip_count"] < minimum_flips:
            raise ValueError("saved prediction flip count is below its implied minimum")
        if values["prediction_flip_count"] > n - values["adv_correct_on_clean"]:
            raise ValueError("saved prediction flip count exceeds its implied maximum")

    perturbation_metrics = saved["perturbation_metrics"]
    expected_perturbation_keys = list(empty_perturbation_metrics())
    if (
        not isinstance(perturbation_metrics, dict)
        or list(perturbation_metrics) != expected_perturbation_keys
    ):
        raise ValueError("saved progress perturbation metric keys mismatch")
    perturbation_n = _require_plain_int(
        perturbation_metrics["n"], "perturbation_metrics.n"
    )
    if perturbation_n != next_index:
        raise ValueError("saved perturbation metric n differs from next_index")
    for metric_name in expected_perturbation_keys[1:]:
        value = perturbation_metrics[metric_name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"saved perturbation metric must be numeric: {metric_name}"
            )
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(
                f"saved perturbation metric must be finite/nonnegative: {metric_name}"
            )
    ratio_roundoff_tolerance = 1e-6 * max(1, perturbation_n)
    if (
        perturbation_metrics["mask_active_ratio_sum"]
        > perturbation_n + ratio_roundoff_tolerance
    ):
        raise ValueError("saved mask_active_ratio_sum exceeds n")
    if perturbation_n == 0 and any(
        perturbation_metrics[key] != 0 for key in expected_perturbation_keys[1:]
    ):
        raise ValueError("zero-sample progress contains perturbation accumulators")

    return next_index, model_metrics, perturbation_metrics


def progress_state_payload(
    contract: dict[str, Any],
    next_index: int,
    model_metrics: dict[str, dict[str, int]],
    perturbation_metrics: dict[str, float],
    *,
    complete: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": PROGRESS_STATE_SCHEMA,
        "contract": contract,
        "next_index": next_index,
        "model_metrics": model_metrics,
        "perturbation_metrics": perturbation_metrics,
    }
    if complete:
        payload["complete"] = True
    return payload


def invalidate_progress_state_for_fresh_run(path: Path) -> None:
    """Atomically replace any stale resumable state before expensive setup."""

    atomic_json(
        path,
        {
            "schema": PROGRESS_STATE_SCHEMA,
            "status": "invalidated",
            "reason": "fresh --no-resume setup has not completed",
        },
    )


def initialize_fresh_progress_state(
    path: Path,
    *,
    contract: dict[str, Any],
    dataset_n: int,
    model_names: list[str],
) -> tuple[dict[str, dict[str, int]], dict[str, float]]:
    model_metrics = {name: empty_model_metrics() for name in model_names}
    perturbation_metrics = empty_perturbation_metrics()
    payload = progress_state_payload(
        contract,
        0,
        model_metrics,
        perturbation_metrics,
    )
    validate_progress_state(
        payload,
        expected_contract=contract,
        dataset_n=dataset_n,
        model_names=model_names,
    )
    atomic_json(path, payload)
    return model_metrics, perturbation_metrics


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if (
        args.batch_size <= 0
        or args.num_workers < 0
        or args.samples < 0
        or args.state_every_batches <= 0
    ):
        raise ValueError(
            "batch size/state interval must be positive; workers/samples must be "
            "non-negative"
        )
    if len(set(args.models)) != len(args.models):
        raise ValueError("--models must not contain duplicate entries")

    checkpoint = Path(args.checkpoint).resolve()
    data_root = Path(args.imagenet_val_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "progress_state.json"
    result_json = out_dir / "results.json"
    result_csv = out_dir / "model_t_metrics.csv"
    if not args.resume:
        invalidate_progress_state_for_fresh_run(state_path)

    seed_everything(args.seed)
    torch.set_num_threads(3)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    generator, checkpoint_payload = build_generator_from_compatible_checkpoint(
        checkpoint,
        args,
    )
    checkpoint_payload = dict(checkpoint_payload)
    architecture_mode = architecture_mode_from_generator_type(
        checkpoint_payload["generator_type"]
    )
    if checkpoint_payload["target"] != -1:
        raise ValueError("this runner currently reports untargeted metrics only")
    source_model_type = str(checkpoint_payload["model_type"])
    image_size = int(checkpoint_payload["image_size"])
    eps = float(checkpoint_payload["eps_pixels"]) / 255.0
    generator = generator.to(device).eval()
    generator.requires_grad_(False)

    dataset = build_evaluation_dataset(data_root, image_size)
    dataset_n = min(len(dataset), args.samples) if args.samples else len(dataset)
    if dataset_n <= 0:
        raise ValueError("ImageNet validation dataset is empty")

    checkpoint_sha256 = sha256_file(checkpoint)
    dataset_manifest = ordered_dataset_manifest(dataset, data_root, dataset_n)
    print(f"dataset_n={dataset_n} classes={len(dataset.classes)}", flush=True)
    print(f"models={list(args.models)}", flush=True)
    models: dict[str, torch.nn.Module] = {}
    victim_model_contracts: dict[str, dict[str, Any]] = {}
    victim_model_execution_contracts: dict[str, dict[str, Any]] = {}
    egs_attack_model_contract: dict[str, Any] | None = None
    for name in args.models:
        print(f"loading_model {name} {MODEL_IMPLEMENTATIONS[name]}", flush=True)
        model = build_model(name).to(device="cpu", dtype=torch.float32).eval()
        model.requires_grad_(False)
        victim_model_contracts[name] = victim_model_state_contract(model)
        victim_model_execution_contracts[name] = victim_model_execution_contract(model)
        victim_sha256 = victim_model_contracts[name]["sha256"]
        print(
            f"victim_state_sha256 {name} {victim_sha256}",
            flush=True,
        )
        if architecture_mode == "egs_tsaa" and name == source_model_type:
            egs_attack_model_contract = attack_model_contract(
                model,
                model_type=source_model_type,
            )
        models[name] = model.to(device=device)
    egs_source_model: torch.nn.Module | None = None
    egs_conditioner: EGSStructuredMask | None = None
    egs_topk_fraction: float | None = None
    if architecture_mode == "egs_tsaa":
        egs_source_model = models.get(source_model_type)
        if egs_source_model is None:
            egs_source_model = build_model(source_model_type).to(
                device="cpu", dtype=torch.float32
            )
            egs_source_model.eval()
            egs_source_model.requires_grad_(False)
            egs_attack_model_contract = attack_model_contract(
                egs_source_model,
                model_type=source_model_type,
            )
            egs_source_model.to(device=device)
        if egs_attack_model_contract is None:
            raise RuntimeError("EGS source-model contract was not initialized")
        egs_topk_fraction = float(checkpoint_payload["architecture"]["egs_tk"])
        actual_conditioner_contract = egs_conditioner_contract(
            model_type=source_model_type,
            image_size=image_size,
            topk_fraction=egs_topk_fraction,
            attack_model_contract=egs_attack_model_contract,
        )
        stored_conditioner_contract = checkpoint_payload.get("conditioner_contract")
        if stored_conditioner_contract is not None:
            validate_egs_conditioner_contract(
                stored_conditioner_contract,
                actual_contract=actual_conditioner_contract,
            )
        checkpoint_payload["conditioner_contract"] = actual_conditioner_contract

    contract = state_contract(
        args,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_payload=checkpoint_payload,
        dataset_manifest=dataset_manifest,
        device=device,
        victim_model_contracts=victim_model_contracts,
        victim_model_execution_contracts=victim_model_execution_contracts,
    )
    next_index = 0
    model_metrics = {name: empty_model_metrics() for name in args.models}
    perturbation_metrics = empty_perturbation_metrics()
    if args.resume and state_path.exists():
        with state_path.open("r", encoding="utf-8") as handle:
            saved = json.load(handle)
        next_index, model_metrics, perturbation_metrics = validate_progress_state(
            saved,
            expected_contract=contract,
            dataset_n=dataset_n,
            model_names=list(args.models),
        )
        print(f"resume next_index={next_index}/{dataset_n}", flush=True)
    elif not args.resume:
        model_metrics, perturbation_metrics = initialize_fresh_progress_state(
            state_path,
            contract=contract,
            dataset_n=dataset_n,
            model_names=list(args.models),
        )

    if device.type == "cuda":
        print(
            "cuda_after_load allocated_mib={:.1f} reserved_mib={:.1f}".format(
                torch.cuda.memory_allocated() / 2**20,
                torch.cuda.memory_reserved() / 2**20,
            ),
            flush=True,
        )

    indices = range(next_index, dataset_n)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    started = time.time()
    processed = next_index
    if egs_source_model is not None:
        if egs_topk_fraction is None:
            raise RuntimeError("EGS-TSSA conditioner contract was not initialized")
        egs_conditioner = EGSStructuredMask(
            egs_source_model,
            model_type=source_model_type,
            image_size=image_size,
            topk_fraction=egs_topk_fraction,
        )
    try:
        progress = tqdm(loader, desc="generator all model_t", dynamic_ncols=True)
        for batch_index, (images, labels) in enumerate(progress, start=1):
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            structured_mask: torch.Tensor | None = None
            if egs_conditioner is not None:
                _, structured_mask = egs_conditioner.clean_logits_and_mask(
                    normalize(images)
                )
            with torch.no_grad():
                adv, adv_inf, adv_0, _ = forward_generator_inference(
                    generator,
                    architecture_mode,
                    images,
                    eps,
                    structured_mask=structured_mask,
                )
                update_perturbation_metrics(
                    perturbation_metrics, images, adv, adv_inf, adv_0
                )
                for name, model in models.items():
                    clean_prediction = predict(model, name, images)
                    adv_prediction = predict(model, name, adv)
                    update_model_metrics(
                        model_metrics[name], clean_prediction, adv_prediction, labels
                    )
                processed += int(labels.numel())
                progress.set_postfix(samples=f"{processed}/{dataset_n}")
                if batch_index % args.state_every_batches == 0:
                    progress_payload = progress_state_payload(
                        contract,
                        processed,
                        model_metrics,
                        perturbation_metrics,
                    )
                    validate_progress_state(
                        progress_payload,
                        expected_contract=contract,
                        dataset_n=dataset_n,
                        model_names=list(args.models),
                    )
                    atomic_json(
                        state_path,
                        progress_payload,
                    )
    finally:
        if egs_conditioner is not None:
            egs_conditioner.close()

    final_progress_payload = progress_state_payload(
        contract,
        processed,
        model_metrics,
        perturbation_metrics,
        complete=True,
    )
    validate_progress_state(
        final_progress_payload,
        expected_contract=contract,
        dataset_n=dataset_n,
        model_names=list(args.models),
    )
    elapsed = time.time() - started
    rows = [finalize_model(name, model_metrics[name]) for name in args.models]
    perturbation = finalize_perturbation(
        perturbation_metrics,
        image_size=image_size,
    )
    payload = {
        "status": "complete",
        "contract": contract,
        "checkpoint_metadata": {
            key: checkpoint_payload[key]
            for key in (
                "generator_type",
                "model_type",
                "target",
                "eps_pixels",
                "image_size",
                "completed_epoch",
                "architecture",
            )
        },
        "runtime": {
            "elapsed_sec_this_invocation": elapsed,
            "completed_at_unix": time.time(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "timm": timm.__version__,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "models": rows,
        "perturbation": perturbation,
        "metric_notes": {
            "prediction_flip_rate_percent": "Legacy Eval_GPG untargeted FR.",
            "untarget_asr_clean_correct_percent": "GT-based ASR over each model's clean-correct subset.",
            "applied_norms": "Per-image norms of adv_0 * adv_inf before image clipping.",
            "actual_norms": "Per-image norms of adv - clean after clipping.",
            "legacy_l2_batch_dependent": "Exact Eval_GPG aggregation; depends on batch size.",
            "legacy_linf_last_batch": "Exact Eval_GPG behavior; only the final batch.",
        },
    }
    atomic_json(result_json, payload)
    write_csv(result_csv, rows)
    atomic_json(state_path, final_progress_payload)
    print(f"completed samples={processed} elapsed_sec={elapsed:.1f}", flush=True)
    print(f"results_json={result_json}", flush=True)
    print(f"results_csv={result_csv}", flush=True)


if __name__ == "__main__":
    main()
