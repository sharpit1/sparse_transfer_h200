"""Dual-mode GPG trainer with DDSC lambda-1 control.

This file is based on ``GPG_train.py`` and supports two generator modes:

1. adapt the sparse-loss multiplier lambda-1 once per controlled epoch from
   the observed binary spatial support; and
2. select either the original learned dual-encoder GPG generator (``legacy``)
   or an ImageNet ResNet-50 V1 layer1 prefix with the parameter-reduced
   shared-lite decoder (``isolated``); and
3. optionally regularize only the attack-objective ResNet-50 with isolated
   layer1 frequency-channel dropout and clean-inclusive EOT.  The clean-label
   model and, when selected, the generator's frozen encoder remain
   dropout-free.

The original GPG sparse loss remains a sum, so the DDSC restoring-gain default
is resolved in the same numerical scale as ``--lam_1``.  Values such as 2 or 4
from mean-loss DDSC experiments must not be copied directly into this trainer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import platform
import random
import threading
import time
import uuid
from collections import OrderedDict, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image, __version__ as PILLOW_VERSION
from torch.utils.data import Subset
from tqdm import tqdm

try:
    from .generators_ddsc_gpg import (
        DDSCGPGGenerator,
        GENERATOR_TYPE as ISOLATED_GENERATOR_TYPE,
        IMAGENET_MEAN,
        IMAGENET_STD,
        module_state_sha256,
        parameter_count,
    )
    from .generators_legacy_gpg import (
        LEGACY_GENERATOR_TYPE,
        LegacyGPGGenerator,
    )
    from .ddsc_layer1_dropout_eot import (
        IsolatedResNet50Layer1ChannelDropoutEOT,
        attack_model_loss_and_logits,
    )
except ImportError:
    from generators_ddsc_gpg import (  # type: ignore[no-redef]
        DDSCGPGGenerator,
        GENERATOR_TYPE as ISOLATED_GENERATOR_TYPE,
        IMAGENET_MEAN,
        IMAGENET_STD,
        module_state_sha256,
        parameter_count,
    )
    from generators_legacy_gpg import (  # type: ignore[no-redef]
        LEGACY_GENERATOR_TYPE,
        LegacyGPGGenerator,
    )
    from ddsc_layer1_dropout_eot import (  # type: ignore[no-redef]
        IsolatedResNet50Layer1ChannelDropoutEOT,
        attack_model_loss_and_logits,
    )


LOGGER = logging.getLogger("ddsc_gpg")
_RUN_TRAINING_LOCK = threading.Lock()
CHECKPOINT_FORMAT = "ddsc_gpg_training_v8"
DROPOUT_TRAINING_CHECKPOINT_FORMAT = "ddsc_gpg_training_v7"
LEGACY_TRAINING_CHECKPOINT_FORMAT = "ddsc_gpg_training_v6"
INFERENCE_CHECKPOINT_FORMAT = "ddsc_gpg_inference_v3"
PREVIOUS_INFERENCE_CHECKPOINT_FORMAT = "ddsc_gpg_inference_v2"
LEGACY_GPG_PARAMETER_COUNT = 8_592_516
GENERATOR_MODES = ("isolated", "legacy")
GENERATOR_TYPES_BY_MODE = {
    "isolated": ISOLATED_GENERATOR_TYPE,
    "legacy": LEGACY_GENERATOR_TYPE,
}
ISOLATED_DECODER_DEFAULTS = {
    "decoder_width": 128,
    "decoder_num_blocks": 3,
    "decoder_upsample_backend": "transpose",
}
DEFAULT_WORKER_TIMEOUT_SECONDS = 120.0
LAYER1_DROPOUT_DEFAULTS = {
    "layer1_dropout_mode": "off",
    "layer1_dropout_p": 0.0,
    "layer1_dropout_channel_ratio": 0.3,
    "layer1_dropout_hf_ratio": 0.35,
    "layer1_dropout_eot_samples": 1,
    "layer1_dropout_eot_reduction": "logits",
}

# These fields change the optimization trajectory or the data presented to the
# model.  Resume is deliberately fail-closed instead of silently mixing two
# experiments.  Operational fields such as epochs, output directory,
# save_every, num_workers, and worker timeout may change.
RESUME_EXACT_ARGS = (
    "train_dir",
    "train_csv",
    "model_type",
    "generator_mode",
    "eps",
    "target",
    "batch_size",
    "max_batches_per_epoch",
    "sample_per_class",
    "n_iters",
    "lr",
    "lam_1",
    "lam_2",
    "lam_3",
    "pb",
    "seed",
    "device",
    "decoder_width",
    "decoder_num_blocks",
    "decoder_upsample_backend",
    *LAYER1_DROPOUT_DEFAULTS,
    "ddsc_target_density",
    "ddsc_warmup_epochs",
    "ddsc_ema_decay",
    "ddsc_mass",
    "ddsc_damping",
    "ddsc_restoring_gain",
    "ddsc_dt",
    "ddsc_lambda1_min",
    "ddsc_lambda1_max",
)


def generator_type_for_mode(generator_mode: str) -> str:
    try:
        return GENERATOR_TYPES_BY_MODE[generator_mode]
    except KeyError as exc:
        raise ValueError(
            f"unsupported generator_mode: {generator_mode!r}"
        ) from exc


def generator_mode_for_type(generator_type: str) -> str:
    matches = [
        mode
        for mode, candidate_type in GENERATOR_TYPES_BY_MODE.items()
        if generator_type == candidate_type
    ]
    if len(matches) != 1:
        raise ValueError(f"unsupported generator_type: {generator_type!r}")
    return matches[0]


def _legacy_feature_guidance_contract(
    architecture: Mapping[str, Any],
) -> str:
    """Classify legacy metadata without trusting it for state reconstruction."""

    encoder = architecture.get("encoder")
    if not isinstance(encoder, Mapping):
        return "invalid"
    gradient_branch = encoder.get("gradient_branch")
    feature_guidance = architecture.get("legacy_feature_guidance")
    if isinstance(gradient_branch, Mapping):
        if (
            gradient_branch.get("present") is True
            and gradient_branch.get("frozen") is False
            and gradient_branch.get("used_by_ddsc") is True
            and feature_guidance == "enabled_trainable_by_ddsc"
        ):
            return "trainable"
        if (
            gradient_branch.get("present") is True
            and gradient_branch.get("frozen") is True
            and gradient_branch.get("used_by_ddsc") is False
            and feature_guidance == "preserved_frozen_unused_by_ddsc"
        ):
            return "frozen_unused"
    if (
        gradient_branch is True
        and feature_guidance == "available_but_unused_by_ddsc"
    ):
        return "unversioned_unused"
    return "invalid"


def _legacy_inference_architecture_matches_current(
    architecture: Mapping[str, Any],
    current_architecture: Mapping[str, Any],
) -> bool:
    """Allow only the historical feature-guidance metadata difference."""

    if _legacy_feature_guidance_contract(architecture) not in {
        "frozen_unused",
        "unversioned_unused",
    }:
        return False
    encoder = architecture.get("encoder")
    current_encoder = current_architecture.get("encoder")
    if not isinstance(encoder, Mapping) or not isinstance(
        current_encoder,
        Mapping,
    ):
        return False
    normalized_architecture = dict(architecture)
    normalized_encoder = dict(encoder)
    normalized_encoder["gradient_branch"] = current_encoder.get(
        "gradient_branch"
    )
    normalized_architecture["encoder"] = normalized_encoder
    normalized_architecture["legacy_feature_guidance"] = (
        current_architecture.get("legacy_feature_guidance")
    )
    return _values_equal_exact(normalized_architecture, current_architecture)


@dataclass(frozen=True)
class DDSCControllerConfig:
    """Second-order feedback-controller configuration for lambda-1."""

    ema_decay: float
    mass: float
    damping: float
    restoring_gain: float
    integration_dt: float
    lambda1_init: float
    lambda1_min: float
    lambda1_max: float | None


@dataclass(frozen=True)
class DDSCControllerState:
    lambda1: float
    velocity: float = 0.0
    support_ema: float | None = None
    error: float | None = None
    update_index: int = 0
    clipped_low_count: int = 0
    clipped_high_count: int = 0


def validate_controller_config(
    config: DDSCControllerConfig,
) -> DDSCControllerConfig:
    values = {
        "ema_decay": config.ema_decay,
        "mass": config.mass,
        "damping": config.damping,
        "restoring_gain": config.restoring_gain,
        "integration_dt": config.integration_dt,
        "lambda1_init": config.lambda1_init,
        "lambda1_min": config.lambda1_min,
    }
    if config.lambda1_max is not None:
        values["lambda1_max"] = config.lambda1_max
    for name, value in values.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"controller {name} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"controller {name} must be finite")
    if not 0.0 <= config.ema_decay < 1.0:
        raise ValueError("controller ema_decay must be in [0, 1)")
    if config.mass <= 0.0:
        raise ValueError("controller mass must be positive")
    if config.damping < 0.0 or config.restoring_gain < 0.0:
        raise ValueError("controller damping/gain must be non-negative")
    if config.integration_dt <= 0.0:
        raise ValueError("controller integration_dt must be positive")
    if config.lambda1_min < 0.0:
        raise ValueError("controller lambda1_min must be non-negative")
    if config.lambda1_max is not None and config.lambda1_max < config.lambda1_min:
        raise ValueError("controller lambda1_max must be >= lambda1_min")
    if not config.lambda1_min <= config.lambda1_init:
        raise ValueError("controller lambda1_init is below lambda1_min")
    if config.lambda1_max is not None and config.lambda1_init > config.lambda1_max:
        raise ValueError("controller lambda1_init is above lambda1_max")
    return config


def initial_controller_state(
    config: DDSCControllerConfig,
) -> DDSCControllerState:
    validate_controller_config(config)
    return DDSCControllerState(lambda1=float(config.lambda1_init))


def validate_controller_state(
    state: DDSCControllerState,
    config: DDSCControllerConfig,
) -> DDSCControllerState:
    """Validate deserialized controller state before it can affect a loss."""

    validate_controller_config(config)
    named_scalar_values: list[tuple[str, float | int]] = [
        ("lambda1", state.lambda1),
        ("velocity", state.velocity),
    ]
    if state.support_ema is not None:
        named_scalar_values.append(("support_ema", state.support_ema))
    if state.error is not None:
        named_scalar_values.append(("error", state.error))
    if any(
        not isinstance(value, (int, float)) or isinstance(value, bool)
        for _, value in named_scalar_values
    ):
        raise ValueError("controller state scalars must be numeric")
    if not all(
        math.isfinite(float(value)) for _, value in named_scalar_values
    ):
        raise ValueError("controller state contains non-finite values")
    if any(type(value) is not int for value in (
        state.update_index,
        state.clipped_low_count,
        state.clipped_high_count,
    )):
        raise ValueError("controller counters must be plain integers")
    if not config.lambda1_min <= state.lambda1:
        raise ValueError("controller state lambda1 lies below its lower bound")
    if config.lambda1_max is not None and state.lambda1 > config.lambda1_max:
        raise ValueError("controller state lambda1 lies above its upper bound")
    if state.support_ema is not None and state.support_ema < 0.0:
        raise ValueError("controller support_ema must be non-negative")
    counters = (
        state.update_index,
        state.clipped_low_count,
        state.clipped_high_count,
    )
    if min(counters) < 0:
        raise ValueError("controller counters must be non-negative")
    if state.clipped_low_count + state.clipped_high_count > state.update_index:
        raise ValueError("controller clipping counters exceed update_index")
    if state.update_index == 0:
        if state.support_ema is not None or state.error is not None:
            raise ValueError("unupdated controller cannot contain EMA/error state")
    elif state.support_ema is None or state.error is None:
        raise ValueError("updated controller must contain EMA and error state")
    return state


def ddsc_controller_transition(
    state: DDSCControllerState,
    config: DDSCControllerConfig,
    *,
    observed_k: float,
    target_k: int,
) -> tuple[DDSCControllerState, dict[str, float | bool | int]]:
    """Apply one sample-weighted support feedback update.

    The update uses semi-implicit Euler order: velocity first, followed by
    lambda-1 using the new velocity.  Clipping resets outward velocity to avoid
    wind-up at either bound.
    """

    validate_controller_state(state, config)
    if not math.isfinite(observed_k) or observed_k < 0.0:
        raise ValueError("observed_k must be finite and non-negative")
    if target_k <= 0:
        raise ValueError("target_k must be positive")
    if state.lambda1 < config.lambda1_min or (
        config.lambda1_max is not None and state.lambda1 > config.lambda1_max
    ):
        raise ValueError("controller state lambda1 lies outside its bounds")

    support_ema = (
        float(observed_k)
        if state.support_ema is None
        else config.ema_decay * state.support_ema
        + (1.0 - config.ema_decay) * float(observed_k)
    )
    error = (support_ema - float(target_k)) / float(target_k)
    acceleration = (
        config.restoring_gain * error - config.damping * state.velocity
    ) / config.mass
    velocity_unclipped = state.velocity + config.integration_dt * acceleration
    lambda1_unclipped = (
        state.lambda1 + config.integration_dt * velocity_unclipped
    )
    intermediates = {
        "support_ema": support_ema,
        "error": error,
        "acceleration": acceleration,
        "velocity_unclipped": velocity_unclipped,
        "lambda1_unclipped": lambda1_unclipped,
    }
    non_finite = [
        name for name, value in intermediates.items() if not math.isfinite(value)
    ]
    if non_finite:
        raise ValueError(
            "controller transition produced non-finite intermediates: "
            f"{non_finite}"
        )
    clipped_low = lambda1_unclipped < config.lambda1_min
    clipped_high = (
        config.lambda1_max is not None
        and lambda1_unclipped > config.lambda1_max
    )
    lambda1_after = max(config.lambda1_min, lambda1_unclipped)
    if config.lambda1_max is not None:
        lambda1_after = min(config.lambda1_max, lambda1_after)

    velocity_after = velocity_unclipped
    at_low_bound = lambda1_after <= config.lambda1_min
    at_high_bound = (
        config.lambda1_max is not None
        and lambda1_after >= config.lambda1_max
    )
    outward_low = at_low_bound and velocity_unclipped < 0.0
    outward_high = at_high_bound and velocity_unclipped > 0.0
    if outward_low or outward_high:
        velocity_after = 0.0

    next_state = DDSCControllerState(
        lambda1=float(lambda1_after),
        velocity=float(velocity_after),
        support_ema=float(support_ema),
        error=float(error),
        update_index=state.update_index + 1,
        clipped_low_count=state.clipped_low_count + int(clipped_low),
        clipped_high_count=state.clipped_high_count + int(clipped_high),
    )
    diagnostics: dict[str, float | bool | int] = {
        "observed_k": float(observed_k),
        "target_k": int(target_k),
        "support_ema": float(support_ema),
        "normalized_error": float(error),
        "acceleration": float(acceleration),
        "velocity_before": float(state.velocity),
        "velocity_after": float(velocity_after),
        "lambda1_before": float(state.lambda1),
        "lambda1_unclipped": float(lambda1_unclipped),
        "lambda1_after": float(lambda1_after),
        "clipped_low": bool(clipped_low),
        "clipped_high": bool(clipped_high),
        "anti_windup_applied": bool(outward_low or outward_high),
    }
    return next_state, diagnostics


def load_controller_state(
    payload: Mapping[str, Any],
    config: DDSCControllerConfig | None = None,
) -> DDSCControllerState:
    expected = {field.name for field in DDSCControllerState.__dataclass_fields__.values()}
    if set(payload) != expected:
        raise ValueError(
            "controller state keys mismatch: "
            f"missing={sorted(expected - set(payload))}, "
            f"unexpected={sorted(set(payload) - expected)}"
        )
    for key in ("lambda1", "velocity"):
        if not isinstance(payload[key], (int, float)) or isinstance(
            payload[key], bool
        ):
            raise ValueError(f"controller_state.{key} must be numeric")
    for key in ("support_ema", "error"):
        if payload[key] is not None and (
            not isinstance(payload[key], (int, float))
            or isinstance(payload[key], bool)
        ):
            raise ValueError(f"controller_state.{key} must be numeric or None")
    for key in ("update_index", "clipped_low_count", "clipped_high_count"):
        _require_plain_int(payload[key], f"controller_state.{key}")
    state = DDSCControllerState(
        lambda1=float(payload["lambda1"]),
        velocity=float(payload["velocity"]),
        support_ema=(
            None
            if payload["support_ema"] is None
            else float(payload["support_ema"])
        ),
        error=None if payload["error"] is None else float(payload["error"]),
        update_index=int(payload["update_index"]),
        clipped_low_count=int(payload["clipped_low_count"]),
        clipped_high_count=int(payload["clipped_high_count"]),
    )
    if config is None:
        scalar_values = [state.lambda1, state.velocity]
        if state.support_ema is not None:
            scalar_values.append(state.support_ema)
        if state.error is not None:
            scalar_values.append(state.error)
        if not all(math.isfinite(value) for value in scalar_values):
            raise ValueError("controller state contains non-finite values")
        if min(
            state.update_index,
            state.clipped_low_count,
            state.clipped_high_count,
        ) < 0:
            raise ValueError("controller counters must be non-negative")
        return state
    return validate_controller_state(state, config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train DDSC-GPG with dynamic lambda-1 using a selectable legacy "
            "or isolated generator"
        )
    )
    parser.add_argument(
        "--train_dir",
        default="/home/dataset/imagenet-1k/train",
        help="path to an ImageFolder root or the NIPS2017 image directory",
    )
    parser.add_argument(
        "--train_csv",
        default="",
        help=(
            "optional NIPS2017 images.csv; when set, TrueLabel is read from "
            "the CSV instead of inferring labels from ImageFolder directories"
        ),
    )
    parser.add_argument(
        "--model_type",
        choices=("res50", "incv3"),
        default="res50",
        help="attack surrogate model",
    )
    parser.add_argument(
        "--generator_mode",
        "--generator-mode",
        dest="generator_mode",
        choices=GENERATOR_MODES,
        default="isolated",
        help=(
            "isolated uses the frozen ResNet-50 layer1 shared-lite generator; "
            "legacy uses the original learned dual-encoder GPG generator"
        ),
    )
    parser.add_argument(
        "--layer1_dropout_mode",
        "--layer1-dropout-mode",
        dest="layer1_dropout_mode",
        choices=("off", "frequency_channel"),
        default=LAYER1_DROPOUT_DEFAULTS["layer1_dropout_mode"],
        help=(
            "isolated attack-objective dropout over ResNet-50 layer1; clean "
            "labels and the generator encoder remain dropout-free"
        ),
    )
    parser.add_argument(
        "--layer1_dropout_p",
        "--layer1-dropout-p",
        dest="layer1_dropout_p",
        type=float,
        default=LAYER1_DROPOUT_DEFAULTS["layer1_dropout_p"],
        help="drop probability for eligible layer1 channels",
    )
    parser.add_argument(
        "--layer1_dropout_channel_ratio",
        "--layer1-dropout-channel-ratio",
        dest="layer1_dropout_channel_ratio",
        type=float,
        default=LAYER1_DROPOUT_DEFAULTS["layer1_dropout_channel_ratio"],
        help="fraction of per-sample high-frequency layer1 channels eligible",
    )
    parser.add_argument(
        "--layer1_dropout_hf_ratio",
        "--layer1-dropout-hf-ratio",
        dest="layer1_dropout_hf_ratio",
        type=float,
        default=LAYER1_DROPOUT_DEFAULTS["layer1_dropout_hf_ratio"],
        help="center low-frequency ratio removed before channel ranking",
    )
    parser.add_argument(
        "--layer1_dropout_eot_samples",
        "--layer1-dropout-eot-samples",
        dest="layer1_dropout_eot_samples",
        type=int,
        default=LAYER1_DROPOUT_DEFAULTS["layer1_dropout_eot_samples"],
        help=(
            "number of stochastic members; one clean member is added and the "
            "ResNet suffix batch grows by samples+1"
        ),
    )
    parser.add_argument(
        "--layer1_dropout_eot_reduction",
        "--layer1-dropout-eot-reduction",
        dest="layer1_dropout_eot_reduction",
        choices=("logits", "loss"),
        default=LAYER1_DROPOUT_DEFAULTS["layer1_dropout_eot_reduction"],
        help="average member logits or member losses for attack objectives",
    )
    parser.add_argument("--eps", type=int, default=10, help="perturbation budget")
    parser.add_argument("--target", type=int, default=-1, help="-1 if untargeted")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--sample_per_class",
        "--samples_per_class",
        dest="sample_per_class",
        type=int,
        default=0,
        help="maximum samples per class; 0 uses all samples",
    )
    parser.add_argument("--n_iters", type=int, default=1)
    parser.add_argument(
        "--max_batches_per_epoch",
        type=int,
        default=0,
        help="maximum training batches per epoch; 0 processes the full loader",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2.25e-5)
    parser.add_argument(
        "--lam_1",
        type=float,
        default=0.0001,
        help="initial controlled sparse-loss multiplier",
    )
    parser.add_argument("--lam_2", type=float, default=0.0001)
    parser.add_argument("--lam_3", type=float, default=0.0001)
    parser.add_argument("--pb", choices=("full", "half"), default="full")
    parser.add_argument("--load_CP", choices=("New", "Continue"), default="New")
    parser.add_argument(
        "--CP_path",
        default="",
        help="DDSC training sidecar (.train.pth), required for Continue",
    )
    parser.add_argument("--out-dir", default="Train_DDSC_GPG")
    parser.add_argument(
        "--device",
        default="cuda",
        help="training device; exact continuation supports CPU and CUDA only",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="DataLoader workers; values above zero use a persistent pool",
    )
    parser.add_argument(
        "--worker_timeout_seconds",
        "--worker-timeout-seconds",
        dest="worker_timeout_seconds",
        type=float,
        default=DEFAULT_WORKER_TIMEOUT_SECONDS,
        help=(
            "maximum seconds to wait for a worker-produced batch; only "
            "applies when num_workers > 0"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--decoder_width",
        type=int,
        default=ISOLATED_DECODER_DEFAULTS["decoder_width"],
    )
    parser.add_argument(
        "--decoder_num_blocks",
        type=int,
        default=ISOLATED_DECODER_DEFAULTS["decoder_num_blocks"],
    )
    parser.add_argument(
        "--decoder_upsample_backend",
        choices=("transpose", "nearest_conv"),
        default=ISOLATED_DECODER_DEFAULTS["decoder_upsample_backend"],
        help="shared-lite backend; transpose matches the frozen-SPGD default",
    )
    parser.add_argument("--save_every", type=int, default=1)

    parser.add_argument(
        "--ddsc_target_density",
        "--ddsc-target-density",
        dest="ddsc_target_density",
        type=float,
        default=0.10,
        help="target fraction of active spatial mask pixels",
    )
    parser.add_argument(
        "--ddsc_warmup_epochs",
        "--ddsc-warmup-epochs",
        dest="ddsc_warmup_epochs",
        type=int,
        default=1,
        help="GPG-compatible epochs with lambda-1 fixed at zero",
    )
    parser.add_argument("--ddsc_ema_decay", type=float, default=0.0)
    parser.add_argument("--ddsc_mass", type=float, default=1.0)
    parser.add_argument("--ddsc_damping", type=float, default=0.25)
    parser.add_argument(
        "--ddsc_restoring_gain",
        type=float,
        default=None,
        help="defaults to --lam_1 to preserve GPG sum-loss scale",
    )
    parser.add_argument("--ddsc_dt", type=float, default=1.0)
    parser.add_argument("--ddsc_lambda1_min", type=float, default=0.0)
    parser.add_argument("--ddsc_lambda1_max", type=float, default=None)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if torch.get_default_dtype() != torch.float32:
        raise ValueError(
            "DDSC-GPG training requires the torch default dtype to be float32"
        )
    try:
        factory_device = torch.empty(0).device.type
    except Exception as exc:
        raise ValueError(
            "DDSC-GPG training requires tensor factories to default to CPU"
        ) from exc
    if (
        torch.get_default_device().type != "cpu"
        or factory_device != "cpu"
    ):
        raise ValueError(
            "DDSC-GPG training requires the torch default/factory device to be CPU; "
            "use --device to select the training device"
        )
    if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
        raise ValueError(
            "DDSC-GPG training requires CPU/CUDA autocast to be disabled"
        )
    try:
        device_type = torch.device(args.device).type
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"--device is invalid: {args.device!r}") from exc
    if device_type not in {"cpu", "cuda"}:
        raise ValueError(
            "exact DDSC continuation currently supports only CPU and CUDA"
        )
    positive = {
        "eps": args.eps,
        "batch_size": args.batch_size,
        "n_iters": args.n_iters,
        "epochs": args.epochs,
        "lr": args.lr,
        "save_every": args.save_every,
        "decoder_width": args.decoder_width,
        "worker_timeout_seconds": args.worker_timeout_seconds,
    }
    for name, value in positive.items():
        if not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"--{name} must be finite and positive")
    if (
        args.sample_per_class < 0
        or args.max_batches_per_epoch < 0
        or args.num_workers < 0
    ):
        raise ValueError(
            "sample_per_class, max_batches_per_epoch, and num_workers "
            "must be non-negative"
        )
    if args.decoder_num_blocks < 0:
        raise ValueError("decoder_num_blocks must be non-negative")
    if args.generator_mode == "legacy":
        changed_decoder_options = {
            name: getattr(args, name)
            for name, default in ISOLATED_DECODER_DEFAULTS.items()
            if getattr(args, name) != default
        }
        if changed_decoder_options:
            raise ValueError(
                "decoder_width/decoder_num_blocks/decoder_upsample_backend "
                "configure only the isolated generator; legacy mode requires "
                f"their defaults, got {changed_decoder_options}"
            )
    if args.ddsc_warmup_epochs < 0:
        raise ValueError("ddsc_warmup_epochs must be non-negative")
    if not 0.0 < args.ddsc_target_density <= 1.0:
        raise ValueError("ddsc_target_density must be in (0, 1]")
    if args.target < -1 or args.target >= 1000:
        raise ValueError("--target must be -1 or an ImageNet class in [0, 999]")
    if not math.isfinite(args.layer1_dropout_p) or not (
        0.0 <= args.layer1_dropout_p < 1.0
    ):
        raise ValueError("layer1_dropout_p must be finite and in [0, 1)")
    if args.layer1_dropout_mode != "off":
        if args.model_type != "res50":
            raise ValueError(
                "layer1 frequency-channel dropout requires --model_type res50"
            )
        if args.layer1_dropout_p <= 0.0:
            raise ValueError(
                "layer1_dropout_p must be positive when layer1 dropout is enabled"
            )
    for name in ("layer1_dropout_channel_ratio", "layer1_dropout_hf_ratio"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be finite and in (0, 1]")
    if args.layer1_dropout_eot_samples <= 0:
        raise ValueError("layer1_dropout_eot_samples must be positive")
    loss_multipliers = (args.lam_1, args.lam_2, args.lam_3)
    if not all(math.isfinite(value) and value >= 0.0 for value in loss_multipliers):
        raise ValueError("loss multipliers must be finite and non-negative")
    if args.load_CP == "Continue" and not args.CP_path:
        raise ValueError("--CP_path is required when --load_CP Continue")


def controller_config_from_args(args: argparse.Namespace) -> DDSCControllerConfig:
    gain = (
        args.lam_1
        if args.ddsc_restoring_gain is None
        else args.ddsc_restoring_gain
    )
    return validate_controller_config(
        DDSCControllerConfig(
            ema_decay=float(args.ddsc_ema_decay),
            mass=float(args.ddsc_mass),
            damping=float(args.ddsc_damping),
            restoring_gain=float(gain),
            integration_dt=float(args.ddsc_dt),
            lambda1_init=float(args.lam_1),
            lambda1_min=float(args.ddsc_lambda1_min),
            lambda1_max=(
                None
                if args.ddsc_lambda1_max is None
                else float(args.ddsc_lambda1_max)
            ),
        )
    )


def _normalize_checkpoint_train_args(
    stored_args: Mapping[str, Any],
    *,
    checkpoint_format: str,
) -> dict[str, Any]:
    """Normalize supported checkpoints into the current v8 argument schema."""

    if not isinstance(stored_args, Mapping):
        raise ValueError("checkpoint train_args must be a mapping")
    normalized = dict(stored_args)
    normalized.setdefault("max_batches_per_epoch", 0)
    dropout_keys = set(LAYER1_DROPOUT_DEFAULTS)
    present = dropout_keys.intersection(normalized)
    if checkpoint_format == CHECKPOINT_FORMAT:
        if present != dropout_keys:
            missing = sorted(dropout_keys - present)
            raise ValueError(
                "v8 checkpoint layer1-dropout arguments are incomplete: "
                f"missing={missing}"
            )
        if "generator_mode" not in normalized:
            raise ValueError("v8 checkpoint generator_mode is missing")
        generator_type_for_mode(normalized["generator_mode"])
        return normalized
    if checkpoint_format == DROPOUT_TRAINING_CHECKPOINT_FORMAT:
        if present != dropout_keys:
            missing = sorted(dropout_keys - present)
            raise ValueError(
                "v7 checkpoint layer1-dropout arguments are incomplete: "
                f"missing={missing}"
            )
        if "generator_mode" in normalized:
            raise ValueError("v7 checkpoint must not contain generator_mode")
        normalized.setdefault("train_csv", "")
        normalized["generator_mode"] = "isolated"
        return normalized
    if checkpoint_format == LEGACY_TRAINING_CHECKPOINT_FORMAT:
        if present:
            raise ValueError(
                "legacy v6 checkpoints must not contain layer1-dropout "
                f"arguments: present={sorted(present)}"
            )
        if "generator_mode" in normalized:
            raise ValueError("legacy v6 checkpoint must not contain generator_mode")
        normalized.update(LAYER1_DROPOUT_DEFAULTS)
        normalized.setdefault("train_csv", "")
        normalized["generator_mode"] = "isolated"
        return normalized
    raise ValueError(f"unsupported training checkpoint format: {checkpoint_format!r}")


def experiment_fingerprint(
    args: argparse.Namespace,
    controller_config: DDSCControllerConfig,
) -> str:
    contract = {key: getattr(args, key) for key in RESUME_EXACT_ARGS}
    contract["train_dir"] = str(Path(args.train_dir).expanduser().resolve())
    if args.train_csv:
        contract["train_csv"] = str(Path(args.train_csv).expanduser().resolve())
    contract["controller_config"] = asdict(controller_config)
    canonical = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:12]


def file_fingerprint(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as checkpoint_file:
            while True:
                chunk = checkpoint_file.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"cannot fingerprint checkpoint {path!s}: {exc}") from exc
    return digest.hexdigest()[:12]


def runtime_contract(device: torch.device) -> dict[str, Any]:
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(
            "exact DDSC continuation currently supports only CPU and CUDA"
        )
    cuda_index: int | None = None
    cuda_name: str | None = None
    cuda_capability: list[int] | None = None
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA runtime contract requested without CUDA")
        cuda_index = (
            torch.cuda.current_device() if device.index is None else device.index
        )
        cuda_name = torch.cuda.get_device_name(cuda_index)
        cuda_capability = list(torch.cuda.get_device_capability(cuda_index))
    return {
        "schema": 3,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "pillow": str(PILLOW_VERSION),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_type": device.type,
        "cuda_device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
        "cuda_index": cuda_index,
        "cuda_name": cuda_name,
        "cuda_capability": cuda_capability,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "mkldnn_deterministic": torch.backends.mkldnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_algorithms_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
    }


def lambda1_for_epoch(
    epoch: int,
    warmup_epochs: int,
    state: DDSCControllerState,
) -> float:
    if type(epoch) is not int or epoch < 0:
        raise ValueError("epoch must be a non-negative plain integer")
    if type(warmup_epochs) is not int or warmup_epochs < 0:
        raise ValueError("warmup_epochs must be a non-negative plain integer")
    return 0.0 if epoch < warmup_epochs else float(state.lambda1)


def update_controller_after_epoch(
    *,
    epoch: int,
    warmup_epochs: int,
    state: DDSCControllerState,
    config: DDSCControllerConfig,
    observed_k: float,
    target_k: int,
) -> tuple[DDSCControllerState, dict[str, float | bool | int] | None]:
    """Update after an epoch so the returned lambda applies next epoch."""

    lambda1_for_epoch(epoch, warmup_epochs, state)
    if epoch < warmup_epochs:
        return state, None
    return ddsc_controller_transition(
        state,
        config,
        observed_k=observed_k,
        target_k=target_k,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def capture_rng_state(
    data_loader_generator: torch.Generator,
    *,
    include_cuda: bool,
) -> dict[str, Any]:
    """Capture every stochastic stream used after an epoch boundary."""

    if type(include_cuda) is not bool:
        raise TypeError("include_cuda must be boolean")
    if include_cuda and not torch.cuda.is_available():
        raise RuntimeError("cannot capture CUDA RNG state because CUDA is unavailable")
    python_version, python_internal, python_gauss = random.getstate()
    numpy_name, numpy_keys, numpy_pos, numpy_has_gauss, numpy_cached = (
        np.random.get_state()
    )
    return {
        "schema": 1,
        "python": {
            "version": int(python_version),
            "internal": [int(value) for value in python_internal],
            "gauss_next": (
                None if python_gauss is None else float(python_gauss)
            ),
        },
        "numpy": {
            "bit_generator": str(numpy_name),
            "keys": torch.tensor(
                numpy_keys.astype(np.int64),
                dtype=torch.int64,
            ),
            "position": int(numpy_pos),
            "has_gauss": int(numpy_has_gauss),
            "cached_gaussian": float(numpy_cached),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda_all": [
            state.clone() for state in torch.cuda.get_rng_state_all()
        ]
        if include_cuda
        else [],
        "data_loader": data_loader_generator.get_state().clone(),
    }


def restore_rng_state(
    payload: Mapping[str, Any],
    data_loader_generator: torch.Generator,
    *,
    require_cuda: bool = False,
) -> None:
    """Restore stochastic streams immediately before creating the next iterator.

    This restores random draws and sample order.  It does not claim bitwise
    CUDA equivalence for operators whose backward kernels are nondeterministic.
    """

    expected = {
        "schema",
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda_all",
        "data_loader",
    }
    if set(payload) != expected:
        raise ValueError(
            "RNG state keys mismatch: "
            f"missing={sorted(expected - set(payload))}, "
            f"unexpected={sorted(set(payload) - expected)}"
        )
    if type(require_cuda) is not bool:
        raise TypeError("require_cuda must be boolean")
    if data_loader_generator.device.type != "cpu":
        raise ValueError("DataLoader RNG generator must be on CPU")
    if type(payload["schema"]) is not int or payload["schema"] != 1:
        raise ValueError("unsupported RNG state schema")
    python_state = payload["python"]
    numpy_state = payload["numpy"]
    if not isinstance(python_state, Mapping):
        raise ValueError("Python RNG state must be a mapping")
    python_expected = {"version", "internal", "gauss_next"}
    if set(python_state) != python_expected:
        raise ValueError("Python RNG state keys mismatch")
    if type(python_state["version"]) is not int:
        raise ValueError("Python RNG version must be a plain integer")
    python_internal = python_state["internal"]
    if not isinstance(python_internal, list) or not python_internal or not all(
        type(value) is int for value in python_internal
    ):
        raise ValueError("Python RNG internal state must be an integer list")
    python_gauss = python_state["gauss_next"]
    if python_gauss is not None and (
        not isinstance(python_gauss, (int, float))
        or isinstance(python_gauss, bool)
        or not math.isfinite(float(python_gauss))
    ):
        raise ValueError("Python RNG Gaussian cache must be numeric or None")
    if not isinstance(numpy_state, Mapping):
        raise ValueError("NumPy RNG state must be a mapping")
    numpy_expected = {
        "bit_generator",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }
    if set(numpy_state) != numpy_expected:
        raise ValueError("NumPy RNG state keys mismatch")
    if not isinstance(numpy_state["bit_generator"], str):
        raise ValueError("NumPy bit_generator must be a string")
    if type(numpy_state["position"]) is not int:
        raise ValueError("NumPy RNG position must be a plain integer")
    if type(numpy_state["has_gauss"]) is not int or numpy_state[
        "has_gauss"
    ] not in (0, 1):
        raise ValueError("NumPy has_gauss must be 0 or 1")
    if not isinstance(numpy_state["cached_gaussian"], (int, float)) or isinstance(
        numpy_state["cached_gaussian"], bool
    ):
        raise ValueError("NumPy cached Gaussian must be numeric")
    numpy_keys = numpy_state["keys"]
    torch_cpu_state = payload["torch_cpu"]
    data_loader_state = payload["data_loader"]
    cuda_states = payload["torch_cuda_all"]
    if not isinstance(numpy_keys, torch.Tensor):
        raise ValueError("NumPy RNG keys must be a tensor")
    if numpy_keys.ndim != 1 or numpy_keys.dtype != torch.int64:
        raise ValueError("NumPy RNG keys must be a one-dimensional int64 tensor")
    if numpy_keys.numel() == 0 or bool(
        torch.any((numpy_keys < 0) | (numpy_keys > 0xFFFFFFFF))
    ):
        raise ValueError("NumPy RNG keys contain values outside uint32")
    if not math.isfinite(float(numpy_state["cached_gaussian"])):
        raise ValueError("NumPy cached Gaussian must be finite")
    if not isinstance(torch_cpu_state, torch.Tensor):
        raise ValueError("Torch CPU RNG state must be a tensor")
    if not isinstance(data_loader_state, torch.Tensor):
        raise ValueError("DataLoader RNG state must be a tensor")
    if not isinstance(cuda_states, (list, tuple)) or not all(
        isinstance(state, torch.Tensor) for state in cuda_states
    ):
        raise ValueError("Torch CUDA RNG state must be a tensor sequence")
    if require_cuda and not cuda_states:
        raise ValueError("CUDA resume checkpoint is missing CUDA RNG state")
    current_cuda_states: list[torch.Tensor] = []
    if cuda_states:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "CUDA device count differs from the checkpoint RNG state"
            )
        current_cuda_states = torch.cuda.get_rng_state_all()
        if any(
            saved.dtype != current.dtype or saved.numel() != current.numel()
            for saved, current in zip(cuda_states, current_cuda_states)
        ):
            raise ValueError("CUDA RNG tensor schema differs from this runtime")

    decoded_python_state = (
        python_state["version"],
        tuple(python_internal),
        None if python_gauss is None else float(python_gauss),
    )
    decoded_numpy_state = (
        str(numpy_state["bit_generator"]),
        numpy_keys.detach().cpu().numpy().astype(np.uint32, copy=False),
        int(numpy_state["position"]),
        int(numpy_state["has_gauss"]),
        float(numpy_state["cached_gaussian"]),
    )
    try:
        # Validate all non-CUDA payloads without mutating process-global state.
        random_probe = random.Random()
        random_probe.setstate(decoded_python_state)
        numpy_probe = np.random.RandomState()
        numpy_probe.set_state(decoded_numpy_state)
        torch_probe = torch.Generator(device="cpu")
        torch_probe.set_state(torch_cpu_state.detach().cpu())
        data_loader_probe = torch.Generator(device="cpu")
        data_loader_probe.set_state(data_loader_state.detach().cpu())
        decoded_cuda_states = [
            state.detach().cpu().clone() for state in cuda_states
        ]
        for device_index, cuda_state in enumerate(decoded_cuda_states):
            cuda_probe = torch.Generator(device=f"cuda:{device_index}")
            cuda_probe.set_state(cuda_state)
    except Exception as exc:
        raise ValueError(f"invalid RNG state in checkpoint: {exc}") from exc

    previous_python_state = random.getstate()
    previous_numpy_state = np.random.get_state()
    previous_torch_cpu_state = torch.get_rng_state().clone()
    previous_cuda_states = [state.clone() for state in current_cuda_states]
    previous_data_loader_state = data_loader_generator.get_state().clone()
    try:
        random.setstate(decoded_python_state)
        np.random.set_state(decoded_numpy_state)
        torch.set_rng_state(torch_cpu_state.detach().cpu())
        if cuda_states:
            torch.cuda.set_rng_state_all(decoded_cuda_states)
        data_loader_generator.set_state(data_loader_state.detach().cpu())
    except Exception as exc:
        rollback_failures: list[str] = []
        rollback_actions = (
            ("Python", lambda: random.setstate(previous_python_state)),
            ("NumPy", lambda: np.random.set_state(previous_numpy_state)),
            ("Torch CPU", lambda: torch.set_rng_state(previous_torch_cpu_state)),
            (
                "Torch CUDA",
                lambda: torch.cuda.set_rng_state_all(previous_cuda_states),
            ),
            (
                "DataLoader",
                lambda: data_loader_generator.set_state(
                    previous_data_loader_state
                ),
            ),
        )
        for stream_name, rollback in rollback_actions:
            if stream_name == "Torch CUDA" and not previous_cuda_states:
                continue
            try:
                rollback()
            except Exception:
                rollback_failures.append(stream_name)
        if rollback_failures:
            raise ValueError(
                "failed to restore RNG state and rollback streams: "
                f"{rollback_failures}"
            ) from exc
        raise ValueError(f"failed to restore RNG state: {exc}") from exc


def normalize_for_classifier(image: torch.Tensor) -> torch.Tensor:
    mean = image.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (image - mean) / std


def build_attack_objective_model(
    clean_model: torch.nn.Module,
    args: argparse.Namespace,
) -> torch.nn.Module:
    """Return the clean model unchanged or an isolated stochastic wrapper."""

    if args.layer1_dropout_mode == "off":
        return clean_model
    if args.model_type != "res50":
        raise ValueError("layer1 dropout attack objective requires model_type res50")
    return IsolatedResNet50Layer1ChannelDropoutEOT(
        clean_model,
        drop_probability=args.layer1_dropout_p,
        channel_ratio=args.layer1_dropout_channel_ratio,
        low_frequency_ratio=args.layer1_dropout_hf_ratio,
        eot_samples=args.layer1_dropout_eot_samples,
        eot_reduction=args.layer1_dropout_eot_reduction,
    )


def hard_spatial_support(continuous_mask: torch.Tensor) -> torch.Tensor:
    """Return one hard active-pixel count per sample for a Bx1xHxW mask."""

    if continuous_mask.ndim != 4 or continuous_mask.shape[1] != 1:
        raise ValueError("continuous_mask must have shape Bx1xHxW")
    return (continuous_mask.detach() >= 0.5).flatten(1).sum(dim=1)


def cw_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    kappa: float = 0.0,
    targeted: bool = False,
) -> torch.Tensor:
    target = target.long()
    target_one_hot = F.one_hot(target, num_classes=logits.shape[1]).to(logits.dtype)
    real = torch.sum(target_one_hot * logits, dim=1)
    other = logits.masked_fill(target_one_hot.bool(), float("-inf")).max(dim=1).values
    margin = other - real if targeted else real - other
    return torch.clamp(margin, min=float(kappa)).sum()


def attack_pgd(
    model: torch.nn.Module,
    image: torch.Tensor,
    labels: torch.Tensor,
    *,
    eps: float,
    alpha: float,
    n_iters: int,
) -> torch.Tensor:
    if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
        raise ValueError("attack_pgd requires CPU/CUDA autocast to be disabled")
    with torch.inference_mode(False), torch.enable_grad():
        delta = torch.empty_like(image).uniform_(-eps, eps)
        delta = torch.clamp(delta, -image, 1.0 - image)
        for _ in range(n_iters):
            delta.requires_grad_(True)
            loss, _ = attack_model_loss_and_logits(
                model,
                normalize_for_classifier(image + delta),
                lambda logits: F.cross_entropy(logits, labels),
            )
            gradient = torch.autograd.grad(loss, delta, only_inputs=True)[0]
            delta = delta.detach() + alpha * gradient.sign()
            delta = torch.clamp(delta, min=-eps, max=eps)
            delta = torch.clamp(delta, -image, 1.0 - image).detach()
        return delta


def build_attack_model(model_type: str) -> torch.nn.Module:
    if model_type == "res50":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        return torchvision.models.resnet50(weights=weights)
    weights = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
    return torchvision.models.inception_v3(weights=weights)


def build_encoder_backbone(
    model_type: str,
    attack_model: torch.nn.Module,
    *,
    continuing: bool,
) -> torch.nn.Module:
    if model_type == "res50":
        return attack_model
    # A continuation checkpoint replaces the complete prefix state, so a
    # random scaffold avoids an unnecessary ResNet download in this case.
    weights = None
    if not continuing:
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    return torchvision.models.resnet50(weights=weights)


class NIPS2017Dataset(torch.utils.data.Dataset):
    """NIPS2017 images with the official one-based ImageNet labels."""

    def __init__(
        self,
        image_root: str | os.PathLike[str],
        csv_path: str | os.PathLike[str],
        transform: transforms.Compose,
    ) -> None:
        self.root = str(Path(image_root).expanduser().resolve())
        self.csv_path = str(Path(csv_path).expanduser().resolve())
        self.transform = transform
        self.samples: list[tuple[str, int]] = []
        self.targets: list[int] = []

        required_columns = {"ImageId", "TrueLabel"}
        with Path(self.csv_path).open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not required_columns.issubset(
                reader.fieldnames
            ):
                raise ValueError(
                    "NIPS2017 CSV must contain ImageId and TrueLabel columns"
                )
            for row_number, row in enumerate(reader, start=2):
                image_id = (row.get("ImageId") or "").strip()
                if not image_id:
                    raise ValueError(
                        f"NIPS2017 CSV row {row_number} has no ImageId"
                    )
                try:
                    label = int(row["TrueLabel"]) - 1
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"NIPS2017 CSV row {row_number} has an invalid TrueLabel"
                    ) from exc
                if not 0 <= label < 1000:
                    raise ValueError(
                        f"NIPS2017 CSV row {row_number} TrueLabel is outside [1, 1000]"
                    )
                image_path = Path(self.root) / f"{image_id}.png"
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"NIPS2017 image listed in CSV is missing: {image_path}"
                    )
                self.samples.append((str(image_path), label))
                self.targets.append(label)
        if not self.samples:
            raise ValueError("NIPS2017 CSV does not contain any samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            sample = self.transform(image.convert("RGB"))
        return sample, label


def build_train_set(
    train_dir: str,
    transform: transforms.Compose,
    *,
    train_csv: str = "",
    sample_per_class: int,
    seed: int,
) -> datasets.ImageFolder | NIPS2017Dataset | Subset:
    train_set: datasets.ImageFolder | NIPS2017Dataset
    if train_csv:
        train_set = NIPS2017Dataset(train_dir, train_csv, transform)
    else:
        train_set = datasets.ImageFolder(train_dir, transform)
    if sample_per_class <= 0:
        return train_set

    class_to_indices: defaultdict[int, list[int]] = defaultdict(list)
    for index, (_, class_index) in enumerate(train_set.samples):
        class_to_indices[class_index].append(index)
    rng = random.Random(seed)
    selected_indices: list[int] = []
    for class_index in sorted(class_to_indices):
        indices = class_to_indices[class_index]
        rng.shuffle(indices)
        selected_indices.extend(indices[:sample_per_class])
    rng.shuffle(selected_indices)
    LOGGER.info(
        "sample_per_class=%d selected %d/%d samples across %d classes",
        sample_per_class,
        len(selected_indices),
        len(train_set),
        len(class_to_indices),
    )
    return Subset(train_set, selected_indices)


def build_training_data_loader(
    train_set: torch.utils.data.Dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    sampler_generator: torch.Generator,
    worker_timeout_seconds: float,
) -> torch.utils.data.DataLoader:
    """Build the map-style training loader without coupling its RNG streams.

    A persistent DataLoader consumes its worker ``base_seed`` only when the
    worker pool is first created, whereas a resumed process creates a new pool.
    Sharing that generator with ``RandomSampler`` would therefore change the
    resumed sample order.  The sampler owns the checkpointed generator and the
    worker pool receives a private clone.  The current ImageFolder transform is
    deterministic, so worker-local Python/NumPy/Torch RNG streams do not affect
    samples.
    """

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("DataLoader batch_size must be a positive integer")
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("DataLoader num_workers must be a non-negative integer")
    if type(pin_memory) is not bool:
        raise ValueError("DataLoader pin_memory must be boolean")
    if sampler_generator.device.type != "cpu":
        raise ValueError("DataLoader sampler generator must be on CPU")
    if (
        not isinstance(worker_timeout_seconds, (int, float))
        or isinstance(worker_timeout_seconds, bool)
        or not math.isfinite(float(worker_timeout_seconds))
        or worker_timeout_seconds <= 0
    ):
        raise ValueError("DataLoader worker timeout must be finite and positive")

    worker_seed_generator = torch.Generator(device="cpu")
    worker_seed_generator.set_state(sampler_generator.get_state())
    sampler = torch.utils.data.RandomSampler(
        train_set,
        generator=sampler_generator,
    )
    return torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=worker_seed_generator,
        persistent_workers=num_workers > 0,
        timeout=(float(worker_timeout_seconds) if num_workers > 0 else 0.0),
    )


def _consume_epoch_base_seed(sampler_generator: torch.Generator) -> int:
    """Preserve the pre-persistent-loader sampler RNG transition per epoch.

    Before persistent workers were enabled, DataLoader consumed one int64
    ``base_seed`` from the same generator immediately before RandomSampler.
    Keeping that transition preserves existing v6 checkpoint continuation and
    makes uninterrupted and newly constructed resumed loaders agree.
    """

    if sampler_generator.device.type != "cpu":
        raise ValueError("DataLoader sampler generator must be on CPU")
    return int(
        torch.empty((), dtype=torch.int64, device="cpu")
        .random_(generator=sampler_generator)
        .item()
    )


def _iter_training_batches(
    train_loader: torch.utils.data.DataLoader,
    *,
    num_workers: int,
    worker_timeout_seconds: float,
) -> Iterator[Any]:
    """Fetch batches and turn opaque Windows worker failures into diagnostics."""

    def raise_worker_failure(exc: Exception) -> None:
        if num_workers <= 0:
            raise exc
        raise RuntimeError(
            "DataLoader worker pipeline failed or exceeded the "
            f"{worker_timeout_seconds:g}-second batch timeout "
            f"(num_workers={num_workers}). On Windows, check for corrupt "
            "images, native image-codec crashes, antivirus/storage stalls, "
            "and memory or page-file pressure. Temporarily using "
            "--num_workers 0 can expose the source exception. "
            f"Original {type(exc).__name__}: {exc}"
        ) from exc

    try:
        loader_iterator = iter(train_loader)
    except Exception as exc:
        raise_worker_failure(exc)
        raise AssertionError("unreachable")
    while True:
        try:
            yield next(loader_iterator)
        except StopIteration:
            return
        except Exception as exc:
            raise_worker_failure(exc)
            raise AssertionError("unreachable")


def dataset_contract(
    train_set: datasets.ImageFolder | NIPS2017Dataset | Subset,
) -> dict[str, Any]:
    """Fingerprint the ordered path/label manifest used by this run.

    This detects additions, removals, renames, relabeling, and subset changes
    without reading every ImageNet image during startup.  It intentionally is
    not a content hash of the image bytes.
    """

    if isinstance(train_set, Subset):
        base_dataset = train_set.dataset
        sample_indices = train_set.indices
    else:
        base_dataset = train_set
        sample_indices = range(len(train_set))
    if not isinstance(base_dataset, (datasets.ImageFolder, NIPS2017Dataset)):
        raise TypeError(
            "dataset_contract requires an ImageFolder, NIPS2017Dataset, or Subset"
        )

    root = Path(base_dataset.root)
    digest = hashlib.sha256()
    sample_count = 0
    for raw_index in sample_indices:
        index = int(raw_index)
        sample_path, class_index = base_dataset.samples[index]
        try:
            relative_path = Path(sample_path).relative_to(root)
        except ValueError:
            relative_path = Path(os.path.relpath(sample_path, root))
        encoded_path = relative_path.as_posix().encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "little", signed=False))
        digest.update(encoded_path)
        digest.update(int(class_index).to_bytes(8, "little", signed=True))
        sample_count += 1
    return {
        "schema": 1,
        "kind": "ordered_relative_path_and_label_sha256",
        "sample_count": sample_count,
        "sha256": digest.hexdigest(),
    }


def validate_dataset_contract(
    stored: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> None:
    expected_keys = {"schema", "kind", "sample_count", "sha256"}
    if set(stored) != expected_keys or set(actual) != expected_keys:
        raise ValueError("dataset contract keys mismatch")
    if not _values_equal_exact(stored, actual):
        raise ValueError(
            "training dataset path/label manifest differs from the checkpoint"
        )


def validate_module_state_dict_finite(
    state_dict: Mapping[str, Any],
    *,
    field_name: str,
) -> None:
    """Reject malformed or non-finite generator parameters and buffers."""

    if not isinstance(field_name, str) or not field_name:
        raise ValueError("module state field_name must be a nonempty string")
    if not state_dict:
        raise ValueError(f"{field_name} must not be empty")
    for name, value in state_dict.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{field_name} contains an invalid state key")
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{field_name}.{name} must be a tensor")
        if value.layout != torch.strided:
            raise ValueError(f"{field_name}.{name} must be dense")
        if (value.is_floating_point() or value.is_complex()) and not (
            torch.isfinite(value).all().item()
        ):
            raise ValueError(f"{field_name}.{name} contains non-finite values")
        if name.endswith("running_var"):
            if not value.is_floating_point() or torch.any(value < 0).item():
                raise ValueError(
                    f"{field_name}.{name} must be a non-negative floating tensor"
                )
        if name.endswith("num_batches_tracked"):
            if (
                value.shape != torch.Size([])
                or value.dtype != torch.int64
                or value.item() < 0
            ):
                raise ValueError(
                    f"{field_name}.{name} must be a non-negative int64 scalar"
                )


def _validate_module_state_schema_exact(
    stored_state: Mapping[str, Any],
    expected_state: Mapping[str, torch.Tensor],
    *,
    field_name: str,
) -> None:
    """Validate state keys and schemas before PyTorch version migration runs."""

    if list(stored_state) != list(expected_state):
        raise ValueError(f"{field_name} state keys/order differ from the model")
    stored_metadata = getattr(stored_state, "_metadata", None)
    expected_metadata = getattr(expected_state, "_metadata", None)
    if not _values_equal_exact(stored_metadata, expected_metadata):
        raise ValueError(f"{field_name} module metadata differs from the model")
    for name, expected_tensor in expected_state.items():
        stored_tensor = stored_state[name]
        if not isinstance(stored_tensor, torch.Tensor):
            raise ValueError(f"{field_name}.{name} must be a tensor")
        if (
            stored_tensor.shape != expected_tensor.shape
            or stored_tensor.dtype != expected_tensor.dtype
            or stored_tensor.layout != expected_tensor.layout
        ):
            raise ValueError(
                f"{field_name}.{name} tensor schema differs from the model"
            )


def _values_equal_exact(left: Any, right: Any) -> bool:
    """Compare checkpoint values without Python's cross-type equality."""

    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        if len(left) != len(right):
            return False
        return all(
            _values_equal_exact(left_key, right_key)
            and _values_equal_exact(left_value, right_value)
            for (left_key, left_value), (right_key, right_value) in zip(
                left.items(), right.items()
            )
        )
    if type(left) in {list, tuple}:
        return len(left) == len(right) and all(
            _values_equal_exact(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    if left is None or type(left) in {bool, int, float, str}:
        return bool(left == right)
    return False


def attack_model_contract(
    attack_model: torch.nn.Module,
    *,
    model_type: str,
) -> dict[str, Any]:
    if model_type not in {"res50", "incv3"}:
        raise ValueError("attack model_type must be res50 or incv3")
    state = attack_model.state_dict()
    validate_module_state_dict_finite(
        state,
        field_name="attack_model_state_dict",
    )
    return {
        "schema": 1,
        "model_type": model_type,
        "architecture": "resnet50" if model_type == "res50" else "inception_v3",
        "weights_enum": "IMAGENET1K_V1",
        "state_sha256": module_state_sha256(attack_model),
    }


def validate_attack_model_contract(
    contract: Mapping[str, Any],
    *,
    expected_model_type: str,
    actual_contract: Mapping[str, Any] | None = None,
) -> None:
    if expected_model_type not in {"res50", "incv3"}:
        raise ValueError("expected attack model_type must be res50 or incv3")
    expected_keys = {
        "schema",
        "model_type",
        "architecture",
        "weights_enum",
        "state_sha256",
    }
    if set(contract) != expected_keys:
        raise ValueError("attack-model contract keys mismatch")
    expected_architecture = (
        "resnet50" if expected_model_type == "res50" else "inception_v3"
    )
    if (
        type(contract["schema"]) is not int
        or contract["schema"] != 1
        or contract["model_type"] != expected_model_type
        or contract["architecture"] != expected_architecture
        or contract["weights_enum"] != "IMAGENET1K_V1"
    ):
        raise ValueError("attack-model contract metadata is incompatible")
    digest = contract["state_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("attack-model state_sha256 is invalid")
    if actual_contract is not None and not _values_equal_exact(
        contract,
        actual_contract,
    ):
        raise ValueError(
            "pretrained attack-model state differs from the training checkpoint"
        )


def _load_checkpoint_envelope(
    path: str | os.PathLike[str],
    *,
    expected_format: str,
) -> Mapping[str, Any]:
    _, payload = _load_checkpoint_envelope_with_format(
        path,
        expected_formats=(expected_format,),
    )
    return payload


def _load_checkpoint_envelope_with_format(
    path: str | os.PathLike[str],
    *,
    expected_formats: Sequence[str],
) -> tuple[str, Mapping[str, Any]]:
    # torch.load inherits the caller's inference mode when materializing
    # tensors.  Public loaders must return ordinary CPU tensors regardless of
    # the embedding application's ambient autograd state.
    with torch.inference_mode(False), torch.no_grad():
        envelope = torch.load(path, map_location="cpu", weights_only=True)
    if type(envelope) is not tuple or len(envelope) != 2:
        raise ValueError("DDSC checkpoint must use the fail-closed tuple envelope")
    format_name, payload = envelope
    if not isinstance(format_name, str):
        raise ValueError("DDSC checkpoint format marker must be a string")
    if format_name not in expected_formats:
        raise ValueError(
            f"checkpoint format mismatch: expected={tuple(expected_formats)!r}, "
            f"actual={format_name!r}"
        )
    if not isinstance(payload, Mapping):
        raise ValueError("DDSC checkpoint payload must be a mapping")
    _validate_loaded_checkpoint_tensors(payload)
    return format_name, payload


def _validate_loaded_checkpoint_tensors(
    value: Any,
    *,
    path: str = "checkpoint payload",
    active_containers: set[int] | None = None,
) -> None:
    """Reject storage-less, non-CPU, or ambient-mode checkpoint tensors."""

    if isinstance(value, torch.Tensor):
        if type(value) is not torch.Tensor:
            raise ValueError(f"{path} must contain plain tensors")
        if vars(value):
            raise ValueError(f"{path} tensor must not carry custom attributes")
        if value.is_meta or value.device.type != "cpu":
            raise ValueError(f"{path} must be an ordinary CPU tensor")
        if value.is_quantized or value.is_nested:
            raise ValueError(f"{path} must be a non-quantized, non-nested tensor")
        if value.layout != torch.strided:
            raise ValueError(f"{path} must be a strided tensor")
        if value.requires_grad or torch.is_inference(value):
            raise ValueError(f"{path} must be a detached non-inference tensor")
        return

    if not isinstance(value, (Mapping, list, tuple)):
        return
    if active_containers is None:
        active_containers = set()
    identity = id(value)
    if identity in active_containers:
        raise ValueError(f"{path} must not contain cyclic containers")
    active_containers.add(identity)
    try:
        if isinstance(value, Mapping):
            for index, (key, child) in enumerate(value.items()):
                _validate_loaded_checkpoint_tensors(
                    key,
                    path=f"{path}.key[{index}]",
                    active_containers=active_containers,
                )
                _validate_loaded_checkpoint_tensors(
                    child,
                    path=f"{path}.value[{index}]",
                    active_containers=active_containers,
                )
            if isinstance(value, OrderedDict):
                attributes = vars(value)
                unexpected_attributes = set(attributes) - {"_metadata"}
                if unexpected_attributes:
                    raise ValueError(
                        f"{path} OrderedDict has unsupported attributes: "
                        f"{sorted(unexpected_attributes)}"
                    )
                if "_metadata" in attributes:
                    _validate_loaded_checkpoint_tensors(
                        attributes["_metadata"],
                        path=f"{path}._metadata",
                        active_containers=active_containers,
                    )
            return
        else:
            children = enumerate(value)
        for key, child in children:
            _validate_loaded_checkpoint_tensors(
                child,
                path=f"{path}[{key!r}]",
                active_containers=active_containers,
            )
    finally:
        active_containers.remove(identity)


def _require_plain_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be a plain integer")
    return value


def load_training_checkpoint(path: str | os.PathLike[str]) -> Mapping[str, Any]:
    checkpoint_format, payload = _load_checkpoint_envelope_with_format(
        path,
        expected_formats=(
            CHECKPOINT_FORMAT,
            DROPOUT_TRAINING_CHECKPOINT_FORMAT,
            LEGACY_TRAINING_CHECKPOINT_FORMAT,
        ),
    )
    required = {
        "kind",
        "checkpoint_boundary",
        "generator_type",
        "generator_state_dict",
        "attack_model_contract",
        "optimizer_state_dict",
        "optimizer_spec",
        "controller_config",
        "controller_state",
        "completed_epoch",
        "next_epoch",
        "train_args",
        "architecture",
        "rng_state",
        "dataset_contract",
        "runtime_contract",
    }
    if set(payload) != required:
        raise ValueError(
            "training checkpoint keys mismatch: "
            f"missing={sorted(required - set(payload))}, "
            f"unexpected={sorted(set(payload) - required)}"
        )
    if payload["kind"] != "training":
        raise ValueError("checkpoint kind is not training")
    if payload["checkpoint_boundary"] != "post_epoch_post_controller_update":
        raise ValueError("unsupported training checkpoint boundary")
    completed_epoch = _require_plain_int(
        payload["completed_epoch"], "completed_epoch"
    )
    next_epoch = _require_plain_int(payload["next_epoch"], "next_epoch")
    if completed_epoch < 0 or next_epoch != completed_epoch + 1:
        raise ValueError("checkpoint epoch boundary is inconsistent")
    for key in (
        "generator_state_dict",
        "attack_model_contract",
        "optimizer_state_dict",
        "optimizer_spec",
        "controller_config",
        "controller_state",
        "train_args",
        "architecture",
        "rng_state",
        "dataset_contract",
        "runtime_contract",
    ):
        if not isinstance(payload[key], Mapping):
            raise ValueError(f"checkpoint field {key} must be a mapping")
    payload = dict(payload)
    payload["train_args"] = _normalize_checkpoint_train_args(
        payload["train_args"],
        checkpoint_format=checkpoint_format,
    )
    expected_generator_type = generator_type_for_mode(
        payload["train_args"]["generator_mode"]
    )
    if payload["generator_type"] != expected_generator_type:
        raise ValueError(
            "checkpoint generator_type is inconsistent with generator_mode"
        )
    if payload["architecture"].get("generator_type") != expected_generator_type:
        raise ValueError(
            "checkpoint architecture generator_type is inconsistent"
        )
    if (
        payload["train_args"]["generator_mode"] == "legacy"
        and _legacy_feature_guidance_contract(payload["architecture"])
        != "trainable"
    ):
        raise ValueError(
            "legacy training checkpoint does not use trainable feature guidance; "
            "restart legacy training under the current objective"
        )
    required_train_args = set(RESUME_EXACT_ARGS) | {"epochs"}
    missing_train_args = required_train_args - set(payload["train_args"])
    if missing_train_args:
        raise ValueError(
            "training checkpoint train_args is incomplete: "
            f"missing={sorted(missing_train_args)}"
        )
    validate_module_state_dict_finite(
        payload["generator_state_dict"],
        field_name="generator_state_dict",
    )
    validate_attack_model_contract(
        payload["attack_model_contract"],
        expected_model_type=payload["train_args"]["model_type"],
    )
    return payload


def load_inference_checkpoint(path: str | os.PathLike[str]) -> Mapping[str, Any]:
    """Load only the self-describing DDSC-GPG inference format."""

    checkpoint_format, payload = _load_checkpoint_envelope_with_format(
        path,
        expected_formats=(
            INFERENCE_CHECKPOINT_FORMAT,
            PREVIOUS_INFERENCE_CHECKPOINT_FORMAT,
        ),
    )
    required = {
        "kind",
        "generator_type",
        "model_type",
        "target",
        "eps_pixels",
        "image_size",
        "completed_epoch",
        "architecture",
        "generator_state_dict",
    }
    if set(payload) != required:
        raise ValueError(
            "inference checkpoint keys mismatch: "
            f"missing={sorted(required - set(payload))}, "
            f"unexpected={sorted(set(payload) - required)}"
        )
    if payload["kind"] != "inference":
        raise ValueError("checkpoint kind is not inference")
    generator_mode = generator_mode_for_type(payload["generator_type"])
    if (
        checkpoint_format == PREVIOUS_INFERENCE_CHECKPOINT_FORMAT
        and generator_mode != "isolated"
    ):
        raise ValueError("v2 inference checkpoints support only isolated mode")
    if payload["model_type"] not in {"res50", "incv3"}:
        raise ValueError("checkpoint model_type is unsupported")
    target = _require_plain_int(payload["target"], "target")
    if target < -1 or target >= 1000:
        raise ValueError("checkpoint target is outside the ImageNet class range")
    expected_size = 299 if payload["model_type"] == "incv3" else 224
    if _require_plain_int(payload["image_size"], "image_size") != expected_size:
        raise ValueError("checkpoint image_size is inconsistent with model_type")
    completed_epoch = _require_plain_int(
        payload["completed_epoch"], "completed_epoch"
    )
    if completed_epoch < 0:
        raise ValueError("completed_epoch must be non-negative")
    if not isinstance(payload["eps_pixels"], (int, float)) or isinstance(
        payload["eps_pixels"], bool
    ):
        raise ValueError("eps_pixels must be numeric")
    if not math.isfinite(float(payload["eps_pixels"])) or payload["eps_pixels"] <= 0:
        raise ValueError("eps_pixels must be finite and positive")
    if not isinstance(payload["architecture"], Mapping) or not isinstance(
        payload["generator_state_dict"], Mapping
    ):
        raise ValueError("inference architecture/state must be mappings")
    if (
        payload["architecture"].get("generator_type")
        != payload["generator_type"]
    ):
        raise ValueError(
            "inference architecture generator_type is inconsistent"
        )
    if (
        generator_mode == "legacy"
        and _legacy_feature_guidance_contract(payload["architecture"])
        not in {"trainable", "frozen_unused", "unversioned_unused"}
    ):
        raise ValueError("legacy inference feature-guidance metadata is incompatible")
    validate_module_state_dict_finite(
        payload["generator_state_dict"],
        field_name="generator_state_dict",
    )
    return payload


def build_generator_from_inference_checkpoint(
    path: str | os.PathLike[str],
) -> tuple[torch.nn.Module, Mapping[str, Any]]:
    """Construct a CPU-float32 generator and load every state key strictly."""

    payload = load_inference_checkpoint(path)
    architecture = payload["architecture"]
    generator_type = payload["generator_type"]
    generator_mode = generator_mode_for_type(generator_type)
    if architecture.get("generator_type") != generator_type:
        raise ValueError("inference architecture generator_type is incompatible")
    encoder = architecture.get("encoder")
    decoder = architecture.get("decoder")
    if not isinstance(encoder, Mapping) or not isinstance(decoder, Mapping):
        raise ValueError("inference architecture is incomplete")
    expected_crop = (
        "legacy_remove_top_row_and_right_column"
        if payload["model_type"] == "incv3"
        else "none"
    )
    if architecture.get("crop_policy") != expected_crop:
        raise ValueError("inference crop policy is incompatible")
    # The scaffold is fully overwritten below.  Force CPU/float32 construction
    # so host default tensor settings cannot touch accelerator RNG or placement.
    with torch.inference_mode(False):
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            if generator_mode == "isolated":
                if (
                    encoder.get("architecture") != "resnet50"
                    or encoder.get("weights_enum") != "IMAGENET1K_V1"
                    or encoder.get("stage") != "layer1"
                    or encoder.get("frozen") is not True
                ):
                    raise ValueError("inference encoder metadata is incompatible")
                if decoder.get("variant") != "shared_lite":
                    raise ValueError("inference decoder variant is incompatible")
                generator = DDSCGPGGenerator(
                    torchvision.models.resnet50(weights=None),
                    inception=payload["model_type"] == "incv3",
                    decoder_width=_require_plain_int(
                        decoder.get("width"), "decoder.width"
                    ),
                    decoder_num_blocks=_require_plain_int(
                        decoder.get("num_blocks"), "decoder.num_blocks"
                    ),
                    decoder_upsample_backend=str(decoder.get("upsample_backend")),
                ).float()
            else:
                if (
                    encoder.get("variant") != "legacy_learned_dual"
                    or encoder.get("frozen") is not False
                    or decoder.get("variant") != "legacy_dual_head"
                ):
                    raise ValueError(
                        "legacy inference architecture metadata is incompatible"
                    )
                generator = LegacyGPGGenerator(
                    inception=payload["model_type"] == "incv3",
                    eps=float(payload["eps_pixels"]) / 255.0,
                    evaluate=True,
                ).float()
        _validate_module_state_schema_exact(
            payload["generator_state_dict"],
            generator.state_dict(),
            field_name="generator_state_dict",
        )
        generator.load_state_dict(payload["generator_state_dict"], strict=True)
        validate_module_state_dict_finite(
            generator.state_dict(),
            field_name="loaded_generator_state_dict",
        )
        generator.eval()
        architecture_matches = _values_equal_exact(
            generator.architecture_metadata(),
            architecture,
        )
        legacy_inference_compatible = (
            generator_mode == "legacy"
            and _legacy_inference_architecture_matches_current(
                architecture,
                generator.architecture_metadata(),
            )
        )
        if not architecture_matches and not legacy_inference_compatible:
            raise ValueError(
                "reconstructed inference architecture does not match metadata"
            )
    return generator, payload


def validate_resume_metadata(
    payload: Mapping[str, Any],
    args: argparse.Namespace,
    controller_config: DDSCControllerConfig,
) -> None:
    stored_args = _normalize_checkpoint_train_args(
        payload["train_args"],
        checkpoint_format=CHECKPOINT_FORMAT,
    )
    for key in RESUME_EXACT_ARGS:
        stored_value = stored_args.get(key)
        requested_value = getattr(args, key)
        if key in {"train_dir", "train_csv"} and requested_value:
            if not isinstance(stored_value, (str, os.PathLike)):
                raise ValueError(f"resume {key} must be path-like")
            stored_value = str(Path(stored_value).expanduser().resolve())
            requested_value = str(Path(requested_value).expanduser().resolve())
        if not _values_equal_exact(stored_value, requested_value):
            raise ValueError(
                f"resume argument {key} differs from checkpoint: "
                f"stored={stored_value!r}, requested={requested_value!r}"
            )
    validate_attack_model_contract(
        payload["attack_model_contract"],
        expected_model_type=args.model_type,
    )
    config_fields = set(DDSCControllerConfig.__dataclass_fields__)
    if set(payload["controller_config"]) != config_fields:
        raise ValueError("resume controller configuration keys differ")
    try:
        stored_controller_config = validate_controller_config(
            DDSCControllerConfig(**dict(payload["controller_config"]))
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid stored controller configuration: {exc}") from exc
    if not _values_equal_exact(
        asdict(stored_controller_config),
        asdict(controller_config),
    ):
        raise ValueError("resume controller configuration differs from checkpoint")
    if not _values_equal_exact(
        payload["runtime_contract"],
        runtime_contract(torch.device(args.device)),
    ):
        raise ValueError("resume runtime contract differs from checkpoint")
    next_epoch = _require_plain_int(payload["next_epoch"], "next_epoch")
    stored_epochs = _require_plain_int(stored_args.get("epochs"), "train_args.epochs")
    if next_epoch > stored_epochs:
        raise ValueError("checkpoint next_epoch exceeds its stored epoch budget")
    state = load_controller_state(payload["controller_state"], controller_config)
    expected_updates = max(0, next_epoch - args.ddsc_warmup_epochs)
    if state.update_index != expected_updates:
        raise ValueError(
            "controller update_index is inconsistent with the epoch boundary"
        )
    if expected_updates == 0 and state != initial_controller_state(controller_config):
        raise ValueError("warm-up checkpoint contains a modified controller state")
    image_size = 299 if args.model_type == "incv3" else 224
    target_k = max(1, round(args.ddsc_target_density * image_size * image_size))
    if state.support_ema is not None:
        if state.support_ema > image_size * image_size:
            raise ValueError("controller support_ema exceeds the spatial support")
        expected_error = (state.support_ema - target_k) / target_k
        if not math.isclose(state.error, expected_error, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("controller error is inconsistent with support_ema")
    if state.lambda1 == controller_config.lambda1_min and state.velocity < 0.0:
        raise ValueError("controller has outward velocity at its lower bound")
    if (
        controller_config.lambda1_max is not None
        and state.lambda1 == controller_config.lambda1_max
        and state.velocity > 0.0
    ):
        raise ValueError("controller has outward velocity at its upper bound")
    validate_optimizer_state_dict(
        payload["optimizer_state_dict"],
        payload["optimizer_spec"],
        expected_lr=args.lr,
        expected_step=expected_optimizer_step(
            next_epoch=next_epoch,
            dataset_manifest=payload["dataset_contract"],
            batch_size=args.batch_size,
            max_batches_per_epoch=args.max_batches_per_epoch,
        ),
    )


def expected_optimizer_step(
    *,
    next_epoch: int,
    dataset_manifest: Mapping[str, Any],
    batch_size: int,
    max_batches_per_epoch: int = 0,
) -> int:
    if type(next_epoch) is not int or next_epoch <= 0:
        raise ValueError("next_epoch must be a positive plain integer")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive plain integer")
    if type(max_batches_per_epoch) is not int or max_batches_per_epoch < 0:
        raise ValueError(
            "max_batches_per_epoch must be a non-negative plain integer"
        )
    sample_count = dataset_manifest.get("sample_count")
    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError("dataset sample_count must be a positive plain integer")
    batches_per_epoch = math.ceil(sample_count / batch_size)
    if max_batches_per_epoch > 0:
        batches_per_epoch = min(batches_per_epoch, max_batches_per_epoch)
    return next_epoch * batches_per_epoch


def epoch_timing_metrics(
    *,
    elapsed_seconds: float,
    processed_batches: int,
    processed_samples: int,
) -> dict[str, float]:
    """Return stable epoch throughput metrics after synchronized training."""

    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0.0:
        raise ValueError("elapsed_seconds must be finite and positive")
    if type(processed_batches) is not int or processed_batches <= 0:
        raise ValueError("processed_batches must be a positive plain integer")
    if type(processed_samples) is not int or processed_samples <= 0:
        raise ValueError("processed_samples must be a positive plain integer")
    return {
        "seconds": float(elapsed_seconds),
        "batches_per_second": processed_batches / elapsed_seconds,
        "images_per_second": processed_samples / elapsed_seconds,
    }


def expected_adam_group_options(expected_lr: float) -> dict[str, Any]:
    """Return this PyTorch runtime's complete default Adam option contract."""

    dummy = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    optimizer = optim.Adam(
        [dummy],
        lr=expected_lr,
        betas=(0.5, 0.999),
    )
    group = dict(optimizer.state_dict()["param_groups"][0])
    group.pop("params")
    return group


def optimizer_spec_for_generator(
    net_g: torch.nn.Module,
    *,
    expected_lr: float,
) -> dict[str, Any]:
    named_parameters = [
        (name, parameter)
        for name, parameter in net_g.named_parameters()
        if parameter.requires_grad
    ]
    if not named_parameters:
        raise ValueError("generator has no trainable parameters")
    return {
        "schema": 2,
        "type": "Adam",
        "group_options": expected_adam_group_options(expected_lr),
        "parameter_names": [name for name, _ in named_parameters],
        "parameter_shapes": [
            list(parameter.shape) for _, parameter in named_parameters
        ],
        "parameter_dtypes": [
            str(parameter.dtype) for _, parameter in named_parameters
        ],
    }


def validate_optimizer_spec(
    spec: Mapping[str, Any],
    *,
    expected_lr: float,
    expected_parameters: Sequence[tuple[str, torch.nn.Parameter]] | None = None,
) -> tuple[list[str], list[list[int]], list[str]]:
    expected_keys = {
        "schema",
        "type",
        "group_options",
        "parameter_names",
        "parameter_shapes",
        "parameter_dtypes",
    }
    if set(spec) != expected_keys:
        raise ValueError("checkpoint optimizer specification keys mismatch")
    if type(spec["schema"]) is not int or spec["schema"] != 2:
        raise ValueError("unsupported optimizer specification schema")
    if spec["type"] != "Adam":
        raise ValueError("checkpoint optimizer type must be Adam")
    group_options = spec["group_options"]
    if not isinstance(group_options, Mapping):
        raise ValueError("optimizer group_options must be a mapping")
    expected_group_options = expected_adam_group_options(expected_lr)
    if not _values_equal_exact(group_options, expected_group_options):
        raise ValueError(
            "optimizer specification group options differ from this runtime"
        )

    names = spec["parameter_names"]
    shapes = spec["parameter_shapes"]
    dtypes = spec["parameter_dtypes"]
    if not isinstance(names, list) or not names or not all(
        isinstance(name, str) and name for name in names
    ):
        raise ValueError("optimizer parameter_names must be a nonempty string list")
    if len(set(names)) != len(names):
        raise ValueError("optimizer parameter_names must be unique")
    if not isinstance(shapes, list) or len(shapes) != len(names):
        raise ValueError("optimizer parameter_shapes length mismatch")
    normalized_shapes: list[list[int]] = []
    for shape in shapes:
        if not isinstance(shape, list) or not all(
            type(dimension) is int and dimension >= 0 for dimension in shape
        ):
            raise ValueError("optimizer parameter shape must contain plain integers")
        normalized_shapes.append(list(shape))
    if not isinstance(dtypes, list) or len(dtypes) != len(names) or not all(
        isinstance(dtype, str) and dtype for dtype in dtypes
    ):
        raise ValueError("optimizer parameter_dtypes length/type mismatch")

    if expected_parameters is not None:
        actual_names = [name for name, _ in expected_parameters]
        actual_shapes = [list(parameter.shape) for _, parameter in expected_parameters]
        actual_dtypes = [str(parameter.dtype) for _, parameter in expected_parameters]
        if names != actual_names:
            raise ValueError("optimizer parameter name/order differs from the model")
        if normalized_shapes != actual_shapes:
            raise ValueError("optimizer parameter shapes differ from the model")
        if dtypes != actual_dtypes:
            raise ValueError("optimizer parameter dtypes differ from the model")
    return list(names), normalized_shapes, list(dtypes)


def validate_optimizer_state_dict(
    payload: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    expected_lr: float,
    expected_step: int,
    expected_parameters: Sequence[tuple[str, torch.nn.Parameter]] | None = None,
) -> None:
    """Validate the complete Adam state before it can affect a continuation."""

    if type(expected_step) is not int or expected_step <= 0:
        raise ValueError("expected optimizer step must be a positive plain integer")
    names, shapes, dtypes = validate_optimizer_spec(
        spec,
        expected_lr=expected_lr,
        expected_parameters=expected_parameters,
    )
    if set(payload) != {"state", "param_groups"}:
        raise ValueError("checkpoint optimizer state_dict keys mismatch")
    state = payload["state"]
    groups = payload["param_groups"]
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint optimizer state must be a mapping")
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError("checkpoint optimizer must contain one parameter group")
    group = groups[0]
    if not isinstance(group, Mapping):
        raise ValueError("checkpoint optimizer parameter group must be a mapping")
    validate_optimizer_param_group(group, expected_lr=expected_lr)

    parameter_ids = group["params"]
    expected_ids = list(range(len(names)))
    if not _values_equal_exact(parameter_ids, expected_ids):
        raise ValueError(
            "optimizer parameter IDs must be unique and in model parameter order"
        )
    if not _values_equal_exact(list(state), expected_ids):
        raise ValueError("optimizer state must cover every trainable parameter")

    optimizer_storages: dict[tuple[str, int | None, int, int], str] = {}

    def register_optimizer_storage(tensor: torch.Tensor, label: str) -> None:
        if not tensor.is_contiguous():
            raise ValueError(f"optimizer {label} must be contiguous")
        if tensor.storage_offset() != 0:
            raise ValueError(
                f"optimizer {label} must have zero storage_offset"
            )
        storage = tensor.untyped_storage()
        logical_nbytes = tensor.numel() * tensor.element_size()
        if storage.nbytes() != logical_nbytes:
            raise ValueError(
                f"optimizer {label} must own an exact-size storage"
            )
        storage_key = (
            tensor.device.type,
            tensor.device.index,
            storage.data_ptr(),
            storage.nbytes(),
        )
        previous = optimizer_storages.get(storage_key)
        if previous is not None:
            raise ValueError(
                f"optimizer {label} shares storage with {previous}"
            )
        optimizer_storages[storage_key] = label

    for parameter_id, shape, dtype in zip(expected_ids, shapes, dtypes):
        entry = state[parameter_id]
        if not isinstance(entry, Mapping) or set(entry) != {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }:
            raise ValueError(
                f"optimizer state {parameter_id} must contain exact Adam moments"
            )
        step = entry["step"]
        if isinstance(step, torch.Tensor):
            if (
                step.layout != torch.strided
                or step.shape != torch.Size([])
                or step.dtype != torch.float32
                or not torch.isfinite(step).all().item()
            ):
                raise ValueError(f"optimizer step {parameter_id} is invalid")
            step_value = float(step.item())
        else:
            raise ValueError(
                f"optimizer step {parameter_id} must be a scalar float32 tensor"
            )
        register_optimizer_storage(step, f"step {parameter_id}")
        if step_value != float(expected_step):
            raise ValueError(
                f"optimizer step {parameter_id} differs from the epoch boundary"
            )

        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = entry[moment_name]
            if not isinstance(moment, torch.Tensor):
                raise ValueError(
                    f"optimizer {moment_name} {parameter_id} must be a tensor"
                )
            if moment.layout != torch.strided:
                raise ValueError(
                    f"optimizer {moment_name} {parameter_id} must be dense"
                )
            if list(moment.shape) != shape:
                raise ValueError(
                    f"optimizer {moment_name} {parameter_id} shape mismatch"
                )
            if str(moment.dtype) != dtype:
                raise ValueError(
                    f"optimizer {moment_name} {parameter_id} dtype mismatch"
                )
            if not torch.isfinite(moment).all().item():
                raise ValueError(
                    f"optimizer {moment_name} {parameter_id} contains non-finite values"
                )
            register_optimizer_storage(
                moment,
                f"{moment_name} {parameter_id}",
            )
        if torch.any(entry["exp_avg_sq"] < 0).item():
            raise ValueError(
                f"optimizer exp_avg_sq {parameter_id} contains negative values"
            )


def validate_optimizer_param_group(
    group: Mapping[str, Any],
    *,
    expected_lr: float,
) -> None:
    """Reject optimizer hyperparameter drift hidden inside a checkpoint."""

    expected = expected_adam_group_options(expected_lr)
    if set(group) != set(expected) | {"params"}:
        raise ValueError("checkpoint optimizer parameter-group keys mismatch")
    for key, expected_value in expected.items():
        if key not in group or not _values_equal_exact(
            group[key],
            expected_value,
        ):
            raise ValueError(
                f"checkpoint optimizer {key} differs from the resume contract: "
                f"stored={group.get(key)!r}, expected={expected_value!r}"
            )


def save_epoch_checkpoints(
    *,
    output_path: Path,
    epoch: int,
    net_g: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    controller_config: DDSCControllerConfig,
    controller_state: DDSCControllerState,
    args: argparse.Namespace,
    data_loader_generator: torch.Generator,
    attack_model_manifest: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    runtime_manifest: Mapping[str, Any],
) -> tuple[Path, Path]:
    inference_path = output_path / (
        f"DDSC_GPG_{args.model_type}_epoch_{epoch:04d}.inference.pth"
    )
    training_path = output_path / (
        f"DDSC_GPG_{args.model_type}_epoch_{epoch:04d}.train.pth"
    )
    generator_state = net_g.state_dict()
    validate_module_state_dict_finite(
        generator_state,
        field_name="generator_state_dict",
    )
    validate_attack_model_contract(
        attack_model_manifest,
        expected_model_type=args.model_type,
    )
    architecture = net_g.architecture_metadata()
    generator_type = getattr(net_g, "generator_type", None)
    generator_mode_for_type(generator_type)
    if architecture.get("generator_type") != generator_type:
        raise ValueError(
            "generator architecture metadata does not match generator_type"
        )
    optimizer_state = optimizer.state_dict()
    optimizer_spec = optimizer_spec_for_generator(
        net_g,
        expected_lr=args.lr,
    )
    validate_optimizer_state_dict(
        optimizer_state,
        optimizer_spec,
        expected_lr=args.lr,
        expected_step=expected_optimizer_step(
            next_epoch=epoch + 1,
            dataset_manifest=dataset_manifest,
            batch_size=args.batch_size,
            max_batches_per_epoch=args.max_batches_per_epoch,
        ),
        expected_parameters=[
            (name, parameter)
            for name, parameter in net_g.named_parameters()
            if parameter.requires_grad
        ],
    )
    rng_state = capture_rng_state(
        data_loader_generator,
        include_cuda=torch.device(args.device).type == "cuda",
    )
    training_payload = (
        CHECKPOINT_FORMAT,
        {
            "kind": "training",
            "checkpoint_boundary": "post_epoch_post_controller_update",
            "generator_type": generator_type,
            "generator_state_dict": generator_state,
            "attack_model_contract": dict(attack_model_manifest),
            "optimizer_state_dict": optimizer_state,
            "optimizer_spec": optimizer_spec,
            "controller_config": asdict(controller_config),
            "controller_state": asdict(controller_state),
            "completed_epoch": epoch,
            "next_epoch": epoch + 1,
            "train_args": vars(args).copy(),
            "architecture": architecture,
            "rng_state": rng_state,
            "dataset_contract": dict(dataset_manifest),
            "runtime_contract": dict(runtime_manifest),
        },
    )
    inference_payload = (
        INFERENCE_CHECKPOINT_FORMAT,
        {
            "kind": "inference",
            "generator_type": generator_type,
            "model_type": args.model_type,
            "target": args.target,
            "eps_pixels": args.eps,
            "image_size": 299 if args.model_type == "incv3" else 224,
            "completed_epoch": epoch,
            "architecture": architecture,
            "generator_state_dict": generator_state,
        },
    )
    _atomic_torch_save(training_payload, training_path)
    _atomic_torch_save(inference_payload, inference_path)
    return inference_path, training_path


def _atomic_torch_save(payload: Any, path: Path) -> None:
    # Do not repeat the potentially long checkpoint filename here.  The default
    # experiment directory is already long enough that doing so can exceed the
    # legacy Windows MAX_PATH limit even when the final path itself is valid.
    temporary_path = path.with_name(f"tmp_{uuid.uuid4().hex}.pth")
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _close_training_log_handlers() -> None:
    owned_handlers = [
        handler
        for handler in LOGGER.handlers
        if getattr(handler, "_ddsc_gpg_owned", False)
    ]
    if not owned_handlers:
        return
    previous_level = getattr(owned_handlers[0], "_ddsc_previous_level")
    previous_propagate = getattr(owned_handlers[0], "_ddsc_previous_propagate")
    try:
        for handler in owned_handlers:
            LOGGER.removeHandler(handler)
            stream = getattr(handler, "stream", None)
            try:
                handler.flush()
            except Exception:
                pass
            try:
                handler.close()
            except Exception:
                # FileHandler.close() can call flush() again.  Preserve the
                # training exception while still releasing the OS file handle.
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass
                handler.stream = None
                try:
                    logging.Handler.close(handler)
                except Exception:
                    pass
    finally:
        LOGGER.setLevel(previous_level)
        LOGGER.propagate = previous_propagate


def configure_logging(
    output_path: Path,
    *,
    continuing: bool,
) -> logging.FileHandler:
    _close_training_log_handlers()
    output_path.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(
        output_path / "train_info.log",
        mode="a" if continuing else "w",
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] - %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
        )
    )
    setattr(handler, "_ddsc_gpg_owned", True)
    setattr(handler, "_ddsc_previous_level", LOGGER.level)
    setattr(handler, "_ddsc_previous_propagate", LOGGER.propagate)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    LOGGER.addHandler(handler)
    return handler


def _claim_output_directory(output_path: Path) -> None:
    """Atomically reserve one output directory for exactly one trainer."""

    try:
        output_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError(
            "refusing to write a run into an existing output directory: "
            f"{output_path}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"failed to create output directory {output_path}: {exc}"
        ) from exc


def _run_training_impl(args: argparse.Namespace) -> Path:
    validate_args(args)
    controller_config = controller_config_from_args(args)
    seed_everything(args.seed)
    torch.set_num_threads(3)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    current_runtime_contract = runtime_contract(device)

    if args.model_type == "res50":
        scale_size, image_size = 256, 224
    else:
        scale_size, image_size = 300, 299
    target_k = max(1, round(args.ddsc_target_density * image_size * image_size))
    config_fingerprint = experiment_fingerprint(args, controller_config)

    resume_payload: Mapping[str, Any] | None = None
    lineage_fingerprint = "root"
    if args.load_CP == "Continue":
        resume_payload = load_training_checkpoint(args.CP_path)
        validate_resume_metadata(resume_payload, args, controller_config)
        lineage_fingerprint = file_fingerprint(args.CP_path)

    generator_descriptor = (
        f"isolated_resnet50_layer1_shared_lite_"
        f"{args.decoder_upsample_backend}"
        if args.generator_mode == "isolated"
        else "legacy_dual_encoder"
    )
    output_name = (
        f"DDSC_GPG_{args.model_type}_tar_{args.target}_eps_{args.eps}_"
        f"Load_{args.load_CP}_lam1init_{args.lam_1}_targetdens_"
        f"{args.ddsc_target_density}_{generator_descriptor}_"
        f"cfg_{config_fingerprint}_"
        f"from_{lineage_fingerprint}"
    )
    output_path = Path(args.out_dir) / output_name
    if output_path.exists():
        raise ValueError(
            "refusing to write a run into an existing output directory: "
            f"{output_path}"
        )

    clean_attack_model = build_attack_model(args.model_type)
    current_attack_model_contract = attack_model_contract(
        clean_attack_model,
        model_type=args.model_type,
    )
    if resume_payload is not None:
        validate_attack_model_contract(
            resume_payload["attack_model_contract"],
            expected_model_type=args.model_type,
            actual_contract=current_attack_model_contract,
        )
    if args.generator_mode == "isolated":
        encoder_backbone = build_encoder_backbone(
            args.model_type,
            clean_attack_model,
            continuing=resume_payload is not None,
        )
        net_g: torch.nn.Module = DDSCGPGGenerator(
            encoder_backbone,
            inception=args.model_type == "incv3",
            decoder_width=args.decoder_width,
            decoder_num_blocks=args.decoder_num_blocks,
            decoder_upsample_backend=args.decoder_upsample_backend,
        )
        if encoder_backbone is not clean_attack_model:
            del encoder_backbone
    else:
        net_g = LegacyGPGGenerator(
            inception=args.model_type == "incv3",
            eps=args.eps / 255.0,
            evaluate=False,
        )

    clean_attack_model.requires_grad_(False)
    clean_attack_model.eval().to(device)
    attack_objective_model = build_attack_objective_model(
        clean_attack_model,
        args,
    )
    attack_objective_model.eval()
    net_g.to(device)
    optimizer = optim.Adam(
        list(net_g.trainable_parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )
    controller_state = initial_controller_state(controller_config)
    start_epoch = 0
    if resume_payload is not None:
        resume_expected_step = expected_optimizer_step(
            next_epoch=_require_plain_int(
                resume_payload["next_epoch"], "next_epoch"
            ),
            dataset_manifest=resume_payload["dataset_contract"],
            batch_size=args.batch_size,
            max_batches_per_epoch=args.max_batches_per_epoch,
        )
        validate_optimizer_state_dict(
            resume_payload["optimizer_state_dict"],
            resume_payload["optimizer_spec"],
            expected_lr=args.lr,
            expected_step=resume_expected_step,
            expected_parameters=[
                (name, parameter)
                for name, parameter in net_g.named_parameters()
                if parameter.requires_grad
            ],
        )
        _validate_module_state_schema_exact(
            resume_payload["generator_state_dict"],
            net_g.state_dict(),
            field_name="generator_state_dict",
        )
        net_g.load_state_dict(resume_payload["generator_state_dict"], strict=True)
        validate_module_state_dict_finite(
            net_g.state_dict(),
            field_name="loaded_generator_state_dict",
        )
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        validate_optimizer_state_dict(
            optimizer.state_dict(),
            resume_payload["optimizer_spec"],
            expected_lr=args.lr,
            expected_step=resume_expected_step,
            expected_parameters=[
                (name, parameter)
                for name, parameter in net_g.named_parameters()
                if parameter.requires_grad
            ],
        )
        controller_state = load_controller_state(
            resume_payload["controller_state"],
            controller_config,
        )
        start_epoch = _require_plain_int(
            resume_payload["next_epoch"], "next_epoch"
        )
        if not _values_equal_exact(
            net_g.architecture_metadata(),
            resume_payload["architecture"],
        ):
            raise ValueError("restored generator architecture differs from checkpoint")
    if start_epoch >= args.epochs:
        raise ValueError(
            f"checkpoint next_epoch={start_epoch} leaves no work for epochs={args.epochs}"
        )

    trainable_parameters = parameter_count(net_g, trainable_only=True)
    total_generator_parameters = parameter_count(net_g)
    reduction = 1.0 - trainable_parameters / LEGACY_GPG_PARAMETER_COUNT

    data_transform = transforms.Compose(
        [
            transforms.Resize(scale_size, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    train_set = build_train_set(
        args.train_dir,
        data_transform,
        train_csv=args.train_csv,
        sample_per_class=args.sample_per_class,
        seed=args.seed,
    )
    dataset_manifest = dataset_contract(train_set)
    if resume_payload is not None:
        validate_dataset_contract(
            resume_payload["dataset_contract"],
            dataset_manifest,
        )
    data_loader_generator = torch.Generator(device="cpu")
    data_loader_generator.manual_seed(args.seed)
    if resume_payload is not None:
        restore_rng_state(
            resume_payload["rng_state"],
            data_loader_generator,
            require_cuda=device.type == "cuda",
        )
    train_loader = build_training_data_loader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        sampler_generator=data_loader_generator,
        worker_timeout_seconds=args.worker_timeout_seconds,
    )
    train_size = len(train_set)
    report_interval = max(1, 1000 // args.batch_size)

    # Do not create run artifacts until every fallible model, checkpoint, and
    # dataset preflight has succeeded.  A startup error can then be retried
    # without leaving a log-only directory that blocks the same configuration.
    _claim_output_directory(output_path)
    try:
        configure_logging(output_path, continuing=args.load_CP == "Continue")
    except BaseException:
        # FileHandler construction normally fails before creating a file.  Only
        # remove the directory if it is still empty and still ours.
        try:
            output_path.rmdir()
        except OSError:
            pass
        raise
    LOGGER.info("args=%s", args)
    LOGGER.info("controller_config=%s", controller_config)
    LOGGER.info("target_k=%d image_size=%d", target_k, image_size)
    if args.epochs <= args.ddsc_warmup_epochs:
        LOGGER.warning(
            "epoch budget ends during lambda1 warm-up; no controlled epoch runs"
        )
    elif args.epochs == args.ddsc_warmup_epochs + 1:
        LOGGER.warning(
            "the final epoch uses lambda1_init; the first feedback-adjusted "
            "lambda1 is saved but never applied in this run"
        )
    LOGGER.info(
        "generator_mode=%s generator_type=%s trainable_params=%d total_params=%d "
        "legacy_trainable_params=%d trainable_reduction=%.4f",
        args.generator_mode,
        net_g.generator_type,
        trainable_parameters,
        total_generator_parameters,
        LEGACY_GPG_PARAMETER_COUNT,
        reduction,
    )
    if args.generator_mode == "isolated":
        LOGGER.info(
            "legacy feature guidance disabled: frozen feature distance has no "
            "decoder gradient; pixel-space PGD guidance remains enabled"
        )
    else:
        LOGGER.info(
            "legacy dual-encoder generator selected; DDSC uses both pixel-space "
            "PGD guidance and the original trainable feature-guidance branch"
        )
    LOGGER.info(
        "attack_objective layer1_dropout_mode=%s p=%.6g channel_ratio=%.6g "
        "hf_ratio=%.6g stochastic_members=%d clean_members=1 "
        "eot_reduction=%s isolated_from_clean_labels_and_generator=True",
        args.layer1_dropout_mode,
        args.layer1_dropout_p,
        args.layer1_dropout_channel_ratio,
        args.layer1_dropout_hf_ratio,
        args.layer1_dropout_eot_samples,
        args.layer1_dropout_eot_reduction,
    )
    if args.layer1_dropout_mode != "off":
        LOGGER.warning(
            "layer1 dropout expands the ResNet suffix batch by %d x; reduce "
            "batch_size or EOT samples if memory is insufficient",
            args.layer1_dropout_eot_samples + 1,
        )
    LOGGER.info("dataset_contract=%s", dataset_manifest)
    LOGGER.info(
        "data_loader num_workers=%d persistent_workers=%s "
        "worker_timeout_seconds=%.6g sampler_rng=checkpointed_separate",
        args.num_workers,
        train_loader.persistent_workers,
        args.worker_timeout_seconds,
    )

    for epoch in range(start_epoch, args.epochs):
        lambda1_applied = lambda1_for_epoch(
            epoch,
            args.ddsc_warmup_epochs,
            controller_state,
        )
        fool_count_epoch = 0
        support_sum = 0.0
        support_samples = 0
        window_fool_count = 0
        window_samples = 0
        last_metrics: dict[str, float] = {}

        # Match the historical one-base-seed-per-epoch sampler transition while
        # the private worker generator handles persistent-worker seeding.
        _consume_epoch_base_seed(data_loader_generator)
        epoch_batch_total = len(train_loader)
        if args.max_batches_per_epoch > 0:
            epoch_batch_total = min(
                epoch_batch_total,
                args.max_batches_per_epoch,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_start_time = time.perf_counter()
        training_batches = _iter_training_batches(
            train_loader,
            num_workers=args.num_workers,
            worker_timeout_seconds=args.worker_timeout_seconds,
        )
        limited_training_batches = islice(training_batches, epoch_batch_total)
        processed_batches = 0
        for batch_index, (image, ground_truth) in enumerate(
            tqdm(limited_training_batches, total=epoch_batch_total)
        ):
            image = image.to(device, non_blocking=True)
            ground_truth = ground_truth.to(device, non_blocking=True)
            with torch.no_grad():
                clean_logits = clean_attack_model(normalize_for_classifier(image))
                clean_prediction = clean_logits.argmax(dim=-1)
            if args.target == -1:
                attack_label = clean_prediction
                reference_prediction = clean_prediction
            else:
                attack_label = torch.full_like(ground_truth, args.target)
                reference_prediction = attack_label

            if args.pb == "half" and args.eps > 128:
                grad_eps = args.eps / 2.0
            else:
                grad_eps = float(args.eps)
            grad_alpha = grad_eps / args.n_iters
            grad_delta = attack_pgd(
                attack_objective_model,
                image,
                ground_truth,
                eps=grad_eps / 255.0,
                alpha=grad_alpha / 255.0,
                n_iters=args.n_iters,
            )

            net_g.train()
            optimizer.zero_grad(set_to_none=True)
            if args.generator_mode == "legacy":
                adv, adv_inf, adv_0, adv_00, feature_guidance_loss = net_g(
                    image,
                    args.eps / 255.0,
                    image + grad_delta,
                )
            else:
                adv, adv_inf, adv_0, adv_00 = net_g(
                    image,
                    args.eps / 255.0,
                )
                feature_guidance_loss = adv.new_zeros(())
            grad_guided_loss = torch.sum((adv_inf - grad_delta) ** 2)
            loss_adv, adv_logits = attack_model_loss_and_logits(
                attack_objective_model,
                normalize_for_classifier(adv),
                lambda logits: cw_loss(
                    logits,
                    attack_label,
                    targeted=args.target != -1,
                ),
            )
            adv_prediction = adv_logits.detach().argmax(dim=-1)
            if args.target == -1:
                batch_fool_count = int(
                    (adv_prediction != reference_prediction).sum().item()
                )
            else:
                batch_fool_count = int(
                    (adv_prediction == reference_prediction).sum().item()
                )

            loss_spa = torch.norm(adv_0, p=1)
            binary_mask = (adv_00 >= 0.5).to(dtype=adv_00.dtype)
            loss_qua = torch.sum((binary_mask - adv_00) ** 2)
            loss = (
                loss_adv
                + lambda1_applied * loss_spa
                + args.lam_2 * loss_qua
                + args.lam_3 * grad_guided_loss
                + args.lam_3 * feature_guidance_loss
            )
            loss.backward()
            optimizer.step()

            batch_size_actual = image.shape[0]
            processed_batches += 1
            batch_support = hard_spatial_support(adv_00)
            support_sum += float(batch_support.sum().item())
            support_samples += batch_size_actual
            fool_count_epoch += batch_fool_count
            window_fool_count += batch_fool_count
            window_samples += batch_size_actual

            should_measure = (
                batch_index % report_interval == 0 or batch_index % 2000 == 0
            )
            if should_measure:
                perturbation = binary_mask.detach() * adv_inf.detach()
                last_metrics = {
                    "loss": float(loss.detach()),
                    "adv_loss": float(loss_adv.detach()),
                    "spa_weighted": float((lambda1_applied * loss_spa).detach()),
                    "qua_weighted": float((args.lam_2 * loss_qua).detach()),
                    "l0": float(batch_support.to(dtype=torch.float64).mean().item()),
                    "l1": float(
                        (torch.norm(perturbation, p=1) / batch_size_actual).item()
                    ),
                    "l2": float(
                        (torch.norm(perturbation, p=2) / batch_size_actual).item()
                    ),
                    "linf": float(torch.norm(perturbation, p=float("inf")).item()),
                    "fool_rate_window": window_fool_count / max(1, window_samples),
                }
                window_fool_count = 0
                window_samples = 0

            if batch_index % 2000 == 0 and last_metrics:
                LOGGER.info(
                    "epoch=%d progress=%d/%d lambda1=%.9g l0=%.2f l1=%.2f "
                    "l2=%.2f linf=%.6f loss=%.3f adv=%.3f spa=%.3f "
                    "qua=%.3f FR=%.4f",
                    epoch,
                    batch_index,
                    epoch_batch_total,
                    lambda1_applied,
                    last_metrics["l0"],
                    last_metrics["l1"],
                    last_metrics["l2"],
                    last_metrics["linf"],
                    last_metrics["loss"],
                    last_metrics["adv_loss"],
                    last_metrics["spa_weighted"],
                    last_metrics["qua_weighted"],
                    last_metrics["fool_rate_window"],
                )

            if batch_index in (100, 10000, 20000, 30000):
                vutils.save_image(
                    vutils.make_grid(adv.detach(), normalize=True, scale_each=True),
                    output_path / f"ep{epoch}_adv{batch_index}.png",
                )
                vutils.save_image(
                    vutils.make_grid(
                        adv.detach() - image,
                        normalize=True,
                        scale_each=True,
                    ),
                    output_path / f"ep{epoch}_noise{batch_index}.png",
                )

        if support_samples == 0:
            raise RuntimeError("training loader produced no samples")
        if processed_batches != epoch_batch_total:
            raise RuntimeError(
                "training loader ended before the configured epoch batch count: "
                f"processed={processed_batches}, expected={epoch_batch_total}"
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timing = epoch_timing_metrics(
            elapsed_seconds=time.perf_counter() - epoch_start_time,
            processed_batches=processed_batches,
            processed_samples=support_samples,
        )
        observed_k = support_sum / support_samples
        fool_rate_epoch = fool_count_epoch / support_samples
        print(
            f"running:{epoch} | FR-{args.model_type}:{fool_rate_epoch:.6f} | "
            f"lambda1:{lambda1_applied:.9g} | support:{observed_k:.3f}/"
            f"{target_k}"
        )
        print(
            f"timing:{epoch} | batches:{processed_batches} | "
            f"samples:{support_samples} | seconds:{timing['seconds']:.3f} | "
            f"batches/s:{timing['batches_per_second']:.3f} | "
            f"images/s:{timing['images_per_second']:.3f}"
        )
        LOGGER.info(
            "epoch_timing epoch=%d batches=%d samples=%d seconds=%.9g "
            "batches_per_second=%.9g images_per_second=%.9g",
            epoch,
            processed_batches,
            support_samples,
            timing["seconds"],
            timing["batches_per_second"],
            timing["images_per_second"],
        )
        LOGGER.info(
            "epoch_summary epoch=%d FR=%.9g lambda1_applied=%.9g "
            "observed_k=%.9g target_k=%d",
            epoch,
            fool_rate_epoch,
            lambda1_applied,
            observed_k,
            target_k,
        )

        controller_state, diagnostics = update_controller_after_epoch(
            epoch=epoch,
            warmup_epochs=args.ddsc_warmup_epochs,
            state=controller_state,
            config=controller_config,
            observed_k=observed_k,
            target_k=target_k,
        )
        if diagnostics is not None:
            LOGGER.info("controller_update=%s", diagnostics)
        else:
            LOGGER.info(
                "controller_update=skipped warmup_epoch=%d/%d",
                epoch + 1,
                args.ddsc_warmup_epochs,
            )

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            inference_path, training_path = save_epoch_checkpoints(
                output_path=output_path,
                epoch=epoch,
                net_g=net_g,
                optimizer=optimizer,
                controller_config=controller_config,
                controller_state=controller_state,
                args=args,
                data_loader_generator=data_loader_generator,
                attack_model_manifest=current_attack_model_contract,
                dataset_manifest=dataset_manifest,
                runtime_manifest=current_runtime_contract,
            )
            LOGGER.info(
                "saved inference_checkpoint=%s training_checkpoint=%s",
                inference_path,
                training_path,
            )

    return output_path


def run_training(args: argparse.Namespace) -> Path:
    """Run one in-process trainer and always release its private log file."""

    if not _RUN_TRAINING_LOCK.acquire(blocking=False):
        raise RuntimeError(
            "another DDSC-GPG run is already active in this Python process"
        )
    try:
        try:
            with torch.inference_mode(False), torch.enable_grad():
                return _run_training_impl(args)
        finally:
            _close_training_log_handlers()
    finally:
        _RUN_TRAINING_LOCK.release()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output_path = run_training(args)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"output_path={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_FORMAT",
    "DEFAULT_WORKER_TIMEOUT_SECONDS",
    "DROPOUT_TRAINING_CHECKPOINT_FORMAT",
    "GENERATOR_MODES",
    "GENERATOR_TYPES_BY_MODE",
    "INFERENCE_CHECKPOINT_FORMAT",
    "ISOLATED_GENERATOR_TYPE",
    "LAYER1_DROPOUT_DEFAULTS",
    "LEGACY_GENERATOR_TYPE",
    "LegacyGPGGenerator",
    "LEGACY_TRAINING_CHECKPOINT_FORMAT",
    "NIPS2017Dataset",
    "PREVIOUS_INFERENCE_CHECKPOINT_FORMAT",
    "DDSCControllerConfig",
    "DDSCControllerState",
    "IsolatedResNet50Layer1ChannelDropoutEOT",
    "attack_model_contract",
    "attack_model_loss_and_logits",
    "attack_pgd",
    "build_attack_model",
    "build_attack_objective_model",
    "build_generator_from_inference_checkpoint",
    "build_parser",
    "build_training_data_loader",
    "capture_rng_state",
    "controller_config_from_args",
    "cw_loss",
    "ddsc_controller_transition",
    "dataset_contract",
    "epoch_timing_metrics",
    "experiment_fingerprint",
    "expected_adam_group_options",
    "expected_optimizer_step",
    "file_fingerprint",
    "generator_mode_for_type",
    "generator_type_for_mode",
    "hard_spatial_support",
    "initial_controller_state",
    "lambda1_for_epoch",
    "load_controller_state",
    "load_inference_checkpoint",
    "load_training_checkpoint",
    "main",
    "normalize_for_classifier",
    "optimizer_spec_for_generator",
    "restore_rng_state",
    "runtime_contract",
    "run_training",
    "save_epoch_checkpoints",
    "seed_everything",
    "update_controller_after_epoch",
    "validate_controller_config",
    "validate_controller_state",
    "validate_args",
    "validate_attack_model_contract",
    "validate_dataset_contract",
    "validate_module_state_dict_finite",
    "validate_optimizer_param_group",
    "validate_optimizer_spec",
    "validate_optimizer_state_dict",
    "validate_resume_metadata",
]
