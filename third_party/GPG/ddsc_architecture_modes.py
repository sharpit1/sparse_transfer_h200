"""Architecture adapters for DDSC sparse-generator training.

The adapters instantiate the generator classes from the vendored GPG, TSAA,
and EGS-TSSA directories directly.  They standardize construction, metadata,
and forward calls without wrapping the modules, so parameter names and tensor
schemas remain identical to the source implementations.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from collections.abc import Iterator, Mapping
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from torch import nn
from torchvision.transforms import InterpolationMode

try:
    from .generators_ddsc_gpg import (
        DDSCGPGGenerator,
        GENERATOR_TYPE,
        SPLIT_GENERATOR_TYPE,
        generator_type_for_decoder_mode,
        parameter_count,
    )
except ImportError:
    from generators_ddsc_gpg import (  # type: ignore[no-redef]
        DDSCGPGGenerator,
        GENERATOR_TYPE,
        SPLIT_GENERATOR_TYPE,
        generator_type_for_decoder_mode,
        parameter_count,
    )


ARCHITECTURE_MODES = ("simple", "gpg", "tsaa", "egs_tsaa")
ARCHITECTURE_MODE_ALIASES = {"egs_tssa": "egs_tsaa"}
GPG_GENERATOR_TYPE = "ddsc_original_gpg_v1"
TSAA_GENERATOR_TYPE = "ddsc_original_tsaa_v1"
EGS_TSAA_GENERATOR_TYPE = "ddsc_original_egs_tssa_v1"
SUPPORTED_GENERATOR_TYPES = frozenset(
    {
        GENERATOR_TYPE,
        SPLIT_GENERATOR_TYPE,
        GPG_GENERATOR_TYPE,
        TSAA_GENERATOR_TYPE,
        EGS_TSAA_GENERATOR_TYPE,
    }
)

_GPG_ROOT = Path(__file__).resolve().parent
_THIRD_PARTY_ROOT = _GPG_ROOT.parent
_SOURCE_PATHS = {
    ("gpg", False): _GPG_ROOT / "generators_modify.py",
    ("gpg", True): _GPG_ROOT / "generators_modify.py",
    ("tsaa", False): _THIRD_PARTY_ROOT / "TSAA" / "generators.py",
    ("tsaa", True): _THIRD_PARTY_ROOT / "TSAA" / "generators.py",
    ("egs_tsaa", False): _THIRD_PARTY_ROOT / "EGS-TSSA" / "train_generators.py",
    ("egs_tsaa", True): _THIRD_PARTY_ROOT / "EGS-TSSA" / "test_generators.py",
}
_EGS_UTILS_PATH = _THIRD_PARTY_ROOT / "EGS-TSSA" / "egs_model_utils.py"
_ATTACK_MODEL_CONTRACT_KEYS = (
    "schema",
    "model_type",
    "architecture",
    "weights_enum",
    "state_sha256",
)
_EGS_CONDITIONER_CONTRACT_KEYS = (
    "schema",
    "kind",
    "model_type",
    "attack_model_contract",
    "egs_model_utils",
    "torch_version",
    "torchvision_version",
    "cam_layer",
    "backward_contract",
    "resize",
    "image_size",
    "filter_size",
    "stride",
    "topk_fraction",
    "box_count",
    "requested_topk_box_count",
    "selected_topk_box_count",
    "box_index_shape",
    "box_index_sha256",
)


def canonical_architecture_mode(mode: str) -> str:
    """Return the public architecture name, accepting the EGS-TSSA spelling."""

    normalized = str(mode).strip().lower().replace("-", "_")
    normalized = ARCHITECTURE_MODE_ALIASES.get(normalized, normalized)
    if normalized not in ARCHITECTURE_MODES:
        raise ValueError(
            "architecture_mode must be simple, gpg, tsaa, egs_tsaa, or egs_tssa"
        )
    return normalized


def generator_type_for_architecture_mode(
    architecture_mode: str,
    decoder_mode: str = "shared",
) -> str:
    mode = canonical_architecture_mode(architecture_mode)
    if mode == "simple":
        return generator_type_for_decoder_mode(decoder_mode)
    if decoder_mode != "shared":
        raise ValueError("decoder_mode is configurable only for architecture_mode simple")
    return {
        "gpg": GPG_GENERATOR_TYPE,
        "tsaa": TSAA_GENERATOR_TYPE,
        "egs_tsaa": EGS_TSAA_GENERATOR_TYPE,
    }[mode]


def architecture_mode_from_generator_type(generator_type: str) -> str:
    if generator_type in {GENERATOR_TYPE, SPLIT_GENERATOR_TYPE}:
        return "simple"
    mapping = {
        GPG_GENERATOR_TYPE: "gpg",
        TSAA_GENERATOR_TYPE: "tsaa",
        EGS_TSAA_GENERATOR_TYPE: "egs_tsaa",
    }
    try:
        return mapping[generator_type]
    except KeyError as exc:
        raise ValueError("checkpoint generator_type is incompatible") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_source_path(path: Path) -> str:
    return path.resolve().relative_to(_THIRD_PARTY_ROOT.parent).as_posix()


@lru_cache(maxsize=None)
def _load_source_module(label: str, path_text: str) -> ModuleType:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"architecture source file is missing: {path}")
    module_name = f"_ddsc_arch_{label}_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load architecture source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _source_module(mode: str, *, inference: bool) -> ModuleType:
    path = _SOURCE_PATHS[(mode, inference)]
    return _load_source_module(f"{mode}_{'eval' if inference else 'train'}", str(path))


def build_original_generator(
    architecture_mode: str,
    *,
    inception: bool,
    eps: float,
    inference: bool,
) -> nn.Module:
    """Instantiate one original full generator without a parameter wrapper."""

    mode = canonical_architecture_mode(architecture_mode)
    if mode == "simple":
        raise ValueError("simple generators require a ResNet encoder backbone")
    module = _source_module(mode, inference=inference)
    generator_class = getattr(module, "GeneratorResnet", None)
    if not isinstance(generator_class, type) or not issubclass(generator_class, nn.Module):
        raise ValueError(f"{mode} source does not expose GeneratorResnet")
    kwargs: dict[str, Any] = {
        "inception": bool(inception),
        "eps": float(eps),
        "evaluate": bool(inference),
    }
    if mode == "tsaa":
        kwargs["data_dim"] = "high"
    return generator_class(**kwargs)


def iter_generator_trainable_parameters(
    generator: nn.Module,
    architecture_mode: str,
) -> Iterator[nn.Parameter]:
    mode = canonical_architecture_mode(architecture_mode)
    if mode == "simple" and hasattr(generator, "trainable_parameters"):
        return iter(generator.trainable_parameters())
    return (parameter for parameter in generator.parameters() if parameter.requires_grad)


def _state_schema_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        record = f"{name}\0{tensor.dtype}\0{tuple(tensor.shape)}\n".encode("utf-8")
        digest.update(record)
    return digest.hexdigest()


def generator_architecture_metadata(
    generator: nn.Module,
    architecture_mode: str,
) -> dict[str, Any]:
    mode = canonical_architecture_mode(architecture_mode)
    if mode == "simple":
        if not hasattr(generator, "architecture_metadata"):
            raise ValueError("simple generator lacks architecture_metadata")
        return dict(generator.architecture_metadata())

    training_source = _SOURCE_PATHS[(mode, False)]
    inference_source = _SOURCE_PATHS[(mode, True)]
    expected_count = {
        "gpg": 8_592_516,
        "tsaa": 8_213_572,
        "egs_tsaa": 8_213_572,
    }[mode]
    actual_count = parameter_count(generator)
    if actual_count != expected_count:
        raise ValueError(
            f"{mode} generator parameter count differs from its source contract: "
            f"expected={expected_count}, actual={actual_count}"
        )
    metadata = {
        "generator_type": generator_type_for_architecture_mode(mode),
        "architecture_mode": mode,
        "source_method": "EGS-TSSA" if mode == "egs_tsaa" else mode.upper(),
        "class_name": "GeneratorResnet",
        "training_source": {
            "path": _relative_source_path(training_source),
            "sha256": _sha256_file(training_source),
        },
        "inference_source": {
            "path": _relative_source_path(inference_source),
            "sha256": _sha256_file(inference_source),
        },
        "state_schema_sha256": _state_schema_sha256(generator),
        "state_entry_count": len(generator.state_dict()),
        "parameter_count": actual_count,
        "trainable_parameter_count": parameter_count(generator, trainable_only=True),
        "inception_crop": bool(getattr(generator, "inception", False)),
        "eps": float(getattr(generator, "eps")),
        "residual_blocks": 6,
        "residual_dropout_probability": 0.5,
        "independent_perturbation_and_mask_decoders": True,
        "mask_training": "source_stochastic_soft_or_detached_hard",
    }
    if mode == "egs_tsaa":
        metadata["egs_tk"] = float(getattr(generator, "ddsc_egs_tk", 0.6))
    return metadata


def forward_generator_training(
    generator: nn.Module,
    architecture_mode: str,
    image: torch.Tensor,
    eps: float,
    *,
    pgd_delta: torch.Tensor | None = None,
    structured_mask: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    """Run a source-specific training forward and standardize auxiliary terms."""

    mode = canonical_architecture_mode(architecture_mode)
    auxiliary: dict[str, torch.Tensor] = {}
    if mode == "simple":
        adv, adv_inf, adv_0, adv_00 = generator(image, eps)
    elif mode == "gpg":
        if pgd_delta is None:
            raise ValueError("gpg mode requires a PGD guidance delta")
        adv, adv_inf, adv_0, adv_00, feature_guidance = generator(
            image,
            eps,
            image + pgd_delta,
        )
        auxiliary["feature_guidance"] = feature_guidance
    elif mode == "tsaa":
        if image.device.type != "cuda":
            raise ValueError("the original TSAA training generator requires CUDA")
        # The vendored source creates its stochastic mask on CPU and then uses
        # an argument-free .cuda().  Scope the current CUDA device to the
        # input so explicit devices such as cuda:1 preserve that exact RNG and
        # copy contract without modifying the source file or its checkpointed
        # SHA-256.
        with torch.cuda.device(image.device):
            adv, adv_inf, adv_0, adv_00 = generator(image)
    else:
        if structured_mask is None:
            raise ValueError("egs_tsaa mode requires a structured mask")
        if image.device.type != "cuda":
            raise ValueError(
                "the original EGS-TSSA training generator requires CUDA"
            )
        if structured_mask.device != image.device:
            raise ValueError("EGS-TSSA structured mask must share the image device")
        # See the TSAA branch above: EGS-TSSA has the same source-level
        # CPU-random -> current-CUDA stochastic-mask contract.
        with torch.cuda.device(image.device):
            adv, adv_inf, adv_0, adv_00, grad_image = generator(
                image,
                structured_mask,
            )
        auxiliary["structured_image"] = grad_image
    return adv, adv_inf, adv_0, adv_00, auxiliary


def forward_generator_inference(
    generator: nn.Module,
    architecture_mode: str,
    image: torch.Tensor,
    eps: float,
    *,
    structured_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a hard-mask inference generator with its source call signature."""

    mode = canonical_architecture_mode(architecture_mode)
    if mode == "simple":
        return generator(image, eps)
    stored_eps = float(getattr(generator, "eps"))
    if not math.isclose(stored_eps, float(eps), rel_tol=0.0, abs_tol=0.0):
        raise ValueError("inference eps differs from the generator constructor eps")
    if mode in {"gpg", "tsaa"}:
        outputs = generator(image)
    else:
        if structured_mask is None:
            raise ValueError("egs_tsaa inference requires a structured mask")
        outputs = generator(image, structured_mask)
    return outputs[0], outputs[1], outputs[2], outputs[3]


def legacy_cw_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    kappa: float = -0.0,
    targeted: bool = False,
) -> torch.Tensor:
    """Exact GPG/TSAA/EGS-TSSA 1000-class CW batch-sum expression."""

    if logits.ndim != 2 or logits.shape[1] != 1000:
        raise ValueError("legacy CW loss requires Bx1000 logits")
    target = target.long()
    target_one_hot = F.one_hot(target, num_classes=1000).to(logits.dtype)
    real = torch.sum(target_one_hot * logits, dim=1)
    other = torch.max(
        (1.0 - target_one_hot) * logits - target_one_hot * 10000.0,
        dim=1,
    ).values
    floor = torch.zeros_like(other).fill_(float(kappa))
    margin = other - real if targeted else real - other
    return torch.sum(torch.maximum(margin, floor))


def quantization_loss(
    architecture_mode: str,
    continuous_mask: torch.Tensor,
    *,
    structured_mask: torch.Tensor | None = None,
    egs_smooth_loss: str = "soft",
) -> torch.Tensor:
    """Return the original mode-specific mask quantization/smoothness loss."""

    mode = canonical_architecture_mode(architecture_mode)
    hard = (continuous_mask >= 0.5).to(dtype=continuous_mask.dtype)
    if mode != "egs_tsaa":
        return torch.sum((hard - continuous_mask) ** 2)
    if structured_mask is None:
        raise ValueError("egs_tsaa quantization requires a structured mask")
    structured = structured_mask.to(dtype=continuous_mask.dtype)
    structured_hard = hard * structured
    if egs_smooth_loss == "soft":
        reference = continuous_mask
    elif egs_smooth_loss == "hard":
        reference = continuous_mask * structured
    else:
        raise ValueError("egs_smooth_loss must be soft or hard")
    return torch.sum((structured_hard - reference) ** 2)


def controller_support_mask(
    architecture_mode: str,
    continuous_mask: torch.Tensor,
    *,
    structured_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the deployed hard spatial support used by the DDSC controller."""

    mode = canonical_architecture_mode(architecture_mode)
    hard = (continuous_mask.detach() >= 0.5).to(continuous_mask.dtype)
    if mode == "egs_tsaa":
        if structured_mask is None:
            raise ValueError("egs_tsaa support requires a structured mask")
        hard = hard * (structured_mask.detach() > 0).to(hard.dtype)
    return hard


def egs_lambda2_for_epoch(
    epoch: int,
    *,
    stage1_lambda2: float,
    stage2_start_epoch: int,
    stage2_lambda2: float,
) -> float:
    if stage2_start_epoch >= 0 and epoch >= stage2_start_epoch:
        return float(stage2_lambda2)
    return float(stage1_lambda2)


def _make_egs_box_index(image_size: int, filter_size: int) -> torch.Tensor:
    indices: list[list[int]] = []
    stride = filter_size
    side = (image_size - filter_size) // stride + 1
    for row in range(side):
        for column in range(side):
            box: list[int] = []
            for offset in range(filter_size):
                start = (
                    row * stride * image_size
                    + column * stride
                    + offset * image_size
                )
                box.extend(range(start, start + filter_size))
            indices.append(box)
    return torch.tensor(indices, dtype=torch.long, device="cpu")


def _egs_box_index_sha256(box_index: torch.Tensor) -> str:
    if (
        box_index.device.type != "cpu"
        or box_index.dtype != torch.long
        or box_index.ndim != 2
    ):
        raise ValueError("EGS box index must be a two-dimensional CPU int64 tensor")
    digest = hashlib.sha256()
    digest.update(b"dtype=torch.int64\n")
    digest.update(
        f"shape={box_index.shape[0]},{box_index.shape[1]}\n".encode("ascii")
    )
    for row in box_index.tolist():
        digest.update((",".join(str(value) for value in row) + "\n").encode("ascii"))
    return digest.hexdigest()


def _canonical_attack_model_contract(
    contract: Mapping[str, Any],
    *,
    model_type: str,
) -> dict[str, Any]:
    if not isinstance(contract, Mapping):
        raise ValueError("EGS attack_model_contract must be a mapping")
    if set(contract) != set(_ATTACK_MODEL_CONTRACT_KEYS):
        raise ValueError("EGS attack_model_contract keys mismatch")
    expected_architecture = "resnet50" if model_type == "res50" else "inception_v3"
    if (
        type(contract["schema"]) is not int
        or contract["schema"] != 1
        or contract["model_type"] != model_type
        or contract["architecture"] != expected_architecture
        or contract["weights_enum"] != "IMAGENET1K_V1"
    ):
        raise ValueError("EGS attack_model_contract metadata is incompatible")
    state_digest = contract["state_sha256"]
    if (
        not isinstance(state_digest, str)
        or len(state_digest) != 64
        or any(character not in "0123456789abcdef" for character in state_digest)
    ):
        raise ValueError("EGS attack-model state_sha256 is invalid")
    return {
        "schema": 1,
        "model_type": model_type,
        "architecture": expected_architecture,
        "weights_enum": "IMAGENET1K_V1",
        "state_sha256": state_digest,
    }


def _contract_values_equal_exact(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        if len(left) != len(right):
            return False
        return all(
            _contract_values_equal_exact(left_key, right_key)
            and _contract_values_equal_exact(left_value, right_value)
            for (left_key, left_value), (right_key, right_value) in zip(
                left.items(), right.items()
            )
        )
    if type(left) in {list, tuple}:
        return len(left) == len(right) and all(
            _contract_values_equal_exact(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    if left is None or type(left) in {bool, int, float, str}:
        return bool(left == right)
    return False


def egs_conditioner_contract(
    *,
    model_type: str,
    image_size: int,
    topk_fraction: float,
    attack_model_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact EGS Grad-CAM/box-selection provenance contract."""

    if model_type not in {"res50", "incv3"}:
        raise ValueError("EGS conditioner model_type must be res50 or incv3")
    if type(image_size) is not int:
        raise ValueError("EGS conditioner image_size must be a plain integer")
    expected_size = 224 if model_type == "res50" else 299
    if image_size != expected_size:
        raise ValueError("EGS conditioner image_size is inconsistent with model_type")
    if (
        type(topk_fraction) not in {int, float}
        or not math.isfinite(float(topk_fraction))
        or not 0.0 < float(topk_fraction) <= 1.0
    ):
        raise ValueError("EGS conditioner topk_fraction must be finite and in (0, 1]")

    topk_fraction = float(topk_fraction)
    filter_size = 8 if model_type == "res50" else 13
    box_index = _make_egs_box_index(image_size, filter_size)
    box_count = int(box_index.shape[0])
    requested_box_count = int(
        ((image_size / filter_size) ** 2) * topk_fraction
    )
    selected_box_count = min(requested_box_count, box_count)
    if selected_box_count <= 0:
        raise ValueError("EGS conditioner top-k configuration selects no boxes")

    return {
        "schema": 1,
        "kind": "egs_tssa_conditioner",
        "model_type": model_type,
        "attack_model_contract": _canonical_attack_model_contract(
            attack_model_contract,
            model_type=model_type,
        ),
        "egs_model_utils": {
            "path": _relative_source_path(_EGS_UTILS_PATH),
            "sha256": _sha256_file(_EGS_UTILS_PATH),
        },
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "cam_layer": "layer4[-1]" if model_type == "res50" else "Mixed_7c",
        "backward_contract": (
            "torch.autograd.backward(logits, torch.ones_like(logits))"
        ),
        "resize": {
            "size": [image_size, image_size],
            "interpolation": "bilinear",
            "antialias": True,
        },
        "image_size": image_size,
        "filter_size": filter_size,
        "stride": filter_size,
        "topk_fraction": topk_fraction,
        "box_count": box_count,
        "requested_topk_box_count": requested_box_count,
        "selected_topk_box_count": selected_box_count,
        "box_index_shape": [int(value) for value in box_index.shape],
        "box_index_sha256": _egs_box_index_sha256(box_index),
    }


def _validated_egs_conditioner_contract(
    contract: Mapping[str, Any],
    *,
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(contract, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    if list(contract) != list(_EGS_CONDITIONER_CONTRACT_KEYS):
        raise ValueError(f"{field_name} keys/order mismatch")
    try:
        expected = egs_conditioner_contract(
            model_type=contract["model_type"],
            image_size=contract["image_size"],
            topk_fraction=contract["topk_fraction"],
            attack_model_contract=contract["attack_model_contract"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} metadata is invalid: {exc}") from exc
    if not _contract_values_equal_exact(contract, expected):
        raise ValueError(f"{field_name} metadata differs from the current EGS contract")
    return expected


def validate_egs_conditioner_contract(
    stored_contract: Mapping[str, Any],
    *,
    actual_contract: Mapping[str, Any] | None = None,
) -> None:
    """Validate stored EGS metadata and optionally compare a live exact contract."""

    _validated_egs_conditioner_contract(
        stored_contract,
        field_name="stored EGS conditioner contract",
    )
    if actual_contract is not None:
        _validated_egs_conditioner_contract(
            actual_contract,
            field_name="actual EGS conditioner contract",
        )
        if not _contract_values_equal_exact(stored_contract, actual_contract):
            raise ValueError(
                "EGS conditioner contract differs from the training checkpoint"
            )


class EGSStructuredMask:
    """Capture the source EGS-TSSA Grad-CAM and construct its box top-k mask."""

    def __init__(
        self,
        model: nn.Module,
        *,
        model_type: str,
        image_size: int,
        topk_fraction: float,
    ) -> None:
        if model_type not in {"res50", "incv3"}:
            raise ValueError("EGS structured masks currently support res50 or incv3")
        if not math.isfinite(topk_fraction) or not 0.0 < topk_fraction <= 1.0:
            raise ValueError("EGS topk_fraction must be in (0, 1]")
        expected_size = 224 if model_type == "res50" else 299
        if image_size != expected_size:
            raise ValueError("EGS image_size is inconsistent with model_type")
        self.model = model
        self.model_type = model_type
        self.image_size = image_size
        self.filter_size = 8 if model_type == "res50" else 13
        self.topk_fraction = float(topk_fraction)
        self._features: list[torch.Tensor] | None = None
        self._grads: list[torch.Tensor] | None = None
        self._utils = _load_source_module("egs_utils", str(_EGS_UTILS_PATH))
        self._cam_layer = (
            model.layer4[-1] if model_type == "res50" else model.Mixed_7c
        )
        self._forward_handle: Any | None = None
        self._backward_handle: Any | None = None
        self._box_index = self._make_box_index()

    def _forward_hook(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: Any,
    ) -> None:
        self._features = [
            part.detach()
            for part in self._utils.feature_to_spatial_parts(output, self.model_type)
        ]

    def _backward_hook(
        self,
        _module: nn.Module,
        _grad_inputs: tuple[torch.Tensor | None, ...],
        grad_outputs: tuple[torch.Tensor | None, ...],
    ) -> None:
        tensors = tuple(part for part in grad_outputs if part is not None)
        self._grads = [
            part.clone().detach()
            for part in self._utils.feature_to_spatial_parts(tensors, self.model_type)
        ]

    def _register_hooks(self) -> None:
        if self._forward_handle is not None or self._backward_handle is not None:
            raise RuntimeError("EGS CAM hooks are already registered")
        self._forward_handle = self._cam_layer.register_forward_hook(
            self._forward_hook
        )
        try:
            self._backward_handle = self._cam_layer.register_full_backward_hook(
                self._backward_hook
            )
        except BaseException:
            try:
                self._remove_hooks()
            finally:
                self.clear()
            raise

    def _remove_hooks(self) -> None:
        forward_handle = self._forward_handle
        backward_handle = self._backward_handle
        self._forward_handle = None
        self._backward_handle = None
        if forward_handle is not None:
            try:
                forward_handle.remove()
            finally:
                if backward_handle is not None:
                    backward_handle.remove()
        elif backward_handle is not None:
            backward_handle.remove()

    def _make_box_index(self) -> torch.Tensor:
        return _make_egs_box_index(self.image_size, self.filter_size)

    @property
    def maximum_density(self) -> float:
        box_count = self._box_index.shape[0]
        selected = int(
            ((self.image_size / self.filter_size) ** 2) * self.topk_fraction
        )
        return min(selected, box_count) / float(box_count)

    def clear(self) -> None:
        self._features = None
        self._grads = None

    def mask_from_captured(self, *, batch_size: int, device: torch.device) -> torch.Tensor:
        if self._features is None or self._grads is None:
            raise RuntimeError("EGS feature/gradient hooks did not capture a complete CAM")
        cam = self._utils.cam_from_feature_and_grad(self._features, self._grads)
        resized = TF.resize(
            cam.unsqueeze(1),
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ).reshape(batch_size, 1, self.image_size, self.image_size)
        index = self._box_index.to(device=device).unsqueeze(0).expand(batch_size, -1, -1)
        flat_grad = resized.reshape(batch_size, -1)
        offsets = (
            torch.arange(batch_size, device=device).view(batch_size, 1, 1)
            * flat_grad.shape[1]
        )
        boxes = torch.take(flat_grad, index + offsets)
        norms = torch.norm(boxes, dim=-1)
        requested_k = int(
            ((self.image_size / self.filter_size) ** 2) * self.topk_fraction
        )
        k = min(requested_k, norms.shape[-1])
        if k <= 0:
            raise ValueError("EGS top-k configuration selected no spatial boxes")
        top_indices = torch.topk(norms, k=k, dim=-1).indices
        box_mask = torch.zeros_like(norms)
        box_mask.scatter_(1, top_indices, 1.0)
        box_size = self.filter_size * self.filter_size
        pixel_mask = box_mask.unsqueeze(-1).expand(-1, -1, box_size).reshape(
            batch_size, -1
        )
        flat_mask = torch.zeros_like(flat_grad)
        flat_mask.scatter_(1, index.reshape(batch_size, -1), pixel_mask)
        self.clear()
        return flat_mask.reshape_as(resized)

    def clean_logits_and_mask(
        self,
        normalized_image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Match EGS-TSSA's backward(ones_like(all logits)) CAM contract."""

        self.clear()
        self._register_hooks()
        try:
            with torch.inference_mode(False), torch.enable_grad():
                model_input = normalized_image.detach().requires_grad_(True)
                logits = self.model(model_input)
                torch.autograd.backward(logits, torch.ones_like(logits))
                mask = self.mask_from_captured(
                    batch_size=normalized_image.shape[0],
                    device=normalized_image.device,
                )
            return logits.detach(), mask.detach()
        finally:
            try:
                self._remove_hooks()
            finally:
                self.clear()

    def close(self) -> None:
        try:
            self._remove_hooks()
        finally:
            self.clear()


__all__ = [
    "ARCHITECTURE_MODES",
    "EGSStructuredMask",
    "EGS_TSAA_GENERATOR_TYPE",
    "GPG_GENERATOR_TYPE",
    "SUPPORTED_GENERATOR_TYPES",
    "TSAA_GENERATOR_TYPE",
    "architecture_mode_from_generator_type",
    "build_original_generator",
    "canonical_architecture_mode",
    "controller_support_mask",
    "egs_conditioner_contract",
    "egs_lambda2_for_epoch",
    "forward_generator_inference",
    "forward_generator_training",
    "generator_architecture_metadata",
    "generator_type_for_architecture_mode",
    "iter_generator_trainable_parameters",
    "legacy_cw_loss",
    "quantization_loss",
    "validate_egs_conditioner_contract",
]
