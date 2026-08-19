"""Isolated parameter-reduced generators for DDSC-GPG training.

The generator keeps GPG's perturbation and stochastic soft/hard mask outputs,
but replaces both learned image encoders with a frozen ImageNet ResNet-50 V1
prefix (conv1 -> bn1 -> relu -> maxpool -> layer1).  The shared variant uses
one upsampling trunk for both outputs; the split variant keeps the adapter and
residual body shared but learns independent perturbation and mask upsampling
trunks.

The legacy feature-alignment loss is intentionally not implemented.  Once the
encoder is frozen, a distance between clean and PGD-image encoder features is
constant with respect to the decoder and therefore cannot train it.  The GPG
pixel-space PGD guidance loss remains in ``DDSC_GPG_train.py``.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GENERATOR_TYPE = "ddsc_gpg_resnet50_layer1_shared_lite_v1"
SPLIT_GENERATOR_TYPE = "ddsc_gpg_resnet50_layer1_split_lite_v1"


def module_state_sha256(module: nn.Module) -> str:
    """Hash names, tensor schemas, and bytes for artifact provenance."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        name_bytes = name.encode("utf-8")
        digest.update(len(name_bytes).to_bytes(8, "little", signed=False))
        digest.update(name_bytes)
        schema = f"{tensor.dtype}:{tuple(tensor.shape)}".encode("ascii")
        digest.update(len(schema).to_bytes(8, "little", signed=False))
        digest.update(schema)
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FrozenResNet50Layer1(nn.Module):
    """Frozen ResNet-50 V1 prefix ending at ``layer1``.

    A private copy is used so generator checkpoints are self-contained and
    calling ``train()`` on the attack classifier cannot mutate encoder buffers.
    """

    def __init__(self, resnet50: nn.Module) -> None:
        super().__init__()
        required = ("conv1", "bn1", "relu", "maxpool", "layer1")
        missing = [name for name in required if not hasattr(resnet50, name)]
        if missing:
            raise ValueError(
                "resnet50 does not expose the required layer1 prefix: "
                f"missing={missing}"
            )

        self.conv1 = copy.deepcopy(resnet50.conv1)
        self.bn1 = copy.deepcopy(resnet50.bn1)
        self.relu = copy.deepcopy(resnet50.relu)
        self.maxpool = copy.deepcopy(resnet50.maxpool)
        self.layer1 = copy.deepcopy(resnet50.layer1)
        self.register_buffer(
            "normalization_mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "normalization_std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
        )

        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenResNet50Layer1":
        # Frozen BatchNorm buffers must never update.
        super().train(False)
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape Bx3xHxW")
        mean = self.normalization_mean.to(dtype=image.dtype)
        std = self.normalization_std.to(dtype=image.dtype)
        feature = (image - mean) / std
        feature = self.conv1(feature)
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)
        return self.layer1(feature)


class DepthwiseResidualBlock(nn.Module):
    """Low-parameter residual block used by the shared decoder."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return feature + self.block(feature)


def _make_adapter(input_channels: int, width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_channels, width, kernel_size=1, bias=False),
        nn.BatchNorm2d(width),
        nn.ReLU(inplace=True),
    )


def _make_upsample_block(
    input_channels: int,
    output_channels: int,
    upsample_backend: str,
) -> nn.Sequential:
    if upsample_backend == "nearest_conv":
        modules: tuple[nn.Module, ...] = (
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )
    elif upsample_backend == "transpose":
        modules = (
            nn.ConvTranspose2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
        )
    else:  # pragma: no cover - public constructors validate this first
        raise ValueError("unsupported upsample backend")
    return nn.Sequential(
        *modules,
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
    )


class SharedLiteGPGDecoder(nn.Module):
    """Decode a 256-channel ResNet layer1 feature into GPG's two heads."""

    def __init__(
        self,
        *,
        input_channels: int = 256,
        width: int = 128,
        num_blocks: int = 3,
        upsample_backend: str = "transpose",
    ) -> None:
        super().__init__()
        if input_channels <= 0 or width <= 0 or num_blocks < 0:
            raise ValueError(
                "input_channels/width must be positive and num_blocks non-negative"
            )
        if upsample_backend not in {"nearest_conv", "transpose"}:
            raise ValueError(
                "upsample_backend must be nearest_conv or transpose"
            )
        self.input_channels = int(input_channels)
        self.width = int(width)
        self.num_blocks = int(num_blocks)
        self.upsample_backend = upsample_backend
        middle_channels = max(1, width // 2)
        output_channels = max(1, width // 4)
        self.trunk_channels = (self.width, middle_channels, output_channels)

        self.adapter = nn.Sequential(
            nn.Conv2d(input_channels, width, kernel_size=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.resblocks = nn.Sequential(
            *(DepthwiseResidualBlock(width) for _ in range(num_blocks))
        )
        if upsample_backend == "nearest_conv":
            upsample1: tuple[nn.Module, ...] = (
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(
                    width,
                    middle_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
            )
            upsample2: tuple[nn.Module, ...] = (
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(
                    middle_channels,
                    output_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
            )
        else:
            upsample1 = (
                nn.ConvTranspose2d(
                    width,
                    middle_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                ),
            )
            upsample2 = (
                nn.ConvTranspose2d(
                    middle_channels,
                    output_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                ),
            )
        self.upsample1 = nn.Sequential(
            *upsample1,
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
        )
        self.upsample2 = nn.Sequential(
            *upsample2,
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.perturbation_head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(output_channels, 3, kernel_size=7),
        )
        self.mask_head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(output_channels, 1, kernel_size=7),
        )

    def forward(
        self, encoder_feature: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decoded = self.adapter(encoder_feature)
        decoded = self.resblocks(decoded)
        decoded = self.upsample1(decoded)
        decoded = self.upsample2(decoded)
        return self.perturbation_head(decoded), self.mask_head(decoded)

    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "variant": "shared_lite",
            "input_channels": self.input_channels,
            "width": self.width,
            "num_blocks": self.num_blocks,
            "upsample_backend": self.upsample_backend,
            "shared_upsample_trunk": True,
            "trunk_channels": list(self.trunk_channels),
        }


class SplitLiteGPGDecoder(nn.Module):
    """Share the adapter/body and split both learnable upsampling stages."""

    def __init__(
        self,
        *,
        input_channels: int = 256,
        width: int = 128,
        num_blocks: int = 3,
        upsample_backend: str = "transpose",
    ) -> None:
        super().__init__()
        if input_channels <= 0 or width <= 0 or num_blocks < 0:
            raise ValueError(
                "input_channels/width must be positive and num_blocks non-negative"
            )
        if upsample_backend not in {"nearest_conv", "transpose"}:
            raise ValueError(
                "upsample_backend must be nearest_conv or transpose"
            )
        self.input_channels = int(input_channels)
        self.width = int(width)
        self.num_blocks = int(num_blocks)
        self.upsample_backend = upsample_backend
        middle_channels = max(1, width // 2)
        output_channels = max(1, width // 4)
        self.trunk_channels = (self.width, middle_channels, output_channels)

        self.adapter = _make_adapter(input_channels, width)
        self.resblocks = nn.Sequential(
            *(DepthwiseResidualBlock(width) for _ in range(num_blocks))
        )
        self.perturbation_upsample1 = _make_upsample_block(
            width, middle_channels, upsample_backend
        )
        self.perturbation_upsample2 = _make_upsample_block(
            middle_channels, output_channels, upsample_backend
        )
        self.mask_upsample1 = _make_upsample_block(
            width, middle_channels, upsample_backend
        )
        self.mask_upsample2 = _make_upsample_block(
            middle_channels, output_channels, upsample_backend
        )
        self.perturbation_head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(output_channels, 3, kernel_size=7),
        )
        self.mask_head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(output_channels, 1, kernel_size=7),
        )

    def forward(
        self, encoder_feature: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.adapter(encoder_feature)
        shared = self.resblocks(shared)
        perturbation = self.perturbation_upsample1(shared)
        perturbation = self.perturbation_upsample2(perturbation)
        mask = self.mask_upsample1(shared)
        mask = self.mask_upsample2(mask)
        return self.perturbation_head(perturbation), self.mask_head(mask)

    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "variant": "split_lite",
            "input_channels": self.input_channels,
            "width": self.width,
            "num_blocks": self.num_blocks,
            "upsample_backend": self.upsample_backend,
            "shared_upsample_trunk": False,
            "split_point": "after_shared_residual_body",
            "trunk_channels": list(self.trunk_channels),
        }


class DDSCGPGGenerator(nn.Module):
    """Frozen ResNet layer1 encoder plus a trainable GPG shared-lite decoder."""

    generator_type = GENERATOR_TYPE
    decoder_class = SharedLiteGPGDecoder

    def __init__(
        self,
        resnet50: nn.Module,
        *,
        inception: bool = False,
        decoder_width: int = 128,
        decoder_num_blocks: int = 3,
        decoder_upsample_backend: str = "transpose",
    ) -> None:
        super().__init__()
        self.encoder = FrozenResNet50Layer1(resnet50)
        self.decoder = self.decoder_class(
            width=decoder_width,
            num_blocks=decoder_num_blocks,
            upsample_backend=decoder_upsample_backend,
        )
        self.inception = bool(inception)

    def train(self, mode: bool = True) -> "DDSCGPGGenerator":
        super().train(mode)
        self.encoder.eval()
        return self

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return (
            parameter
            for parameter in self.decoder.parameters()
            if parameter.requires_grad
        )

    def forward(
        self,
        image: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not torch.is_floating_point(image):
            raise TypeError("image must be floating point")
        perturb_budget = float(eps)
        if perturb_budget < 0.0:
            raise ValueError("eps must be non-negative")

        with torch.no_grad():
            encoder_feature = self.encoder(image).detach()
        perturbation_logits, mask_logits = self.decoder(encoder_feature)

        # ResNet layer1 is 1/4 scale.  A 299x299 image yields 75x75 features
        # and therefore a 300x300 decode.  Match GPG's ConstantPad2d
        # ((0, -1, -1, 0), 0): remove the first row and last column.  Outside
        # the explicit Inception path, unexpected spatial drift is an error.
        height, width = image.shape[-2:]
        expected_decoder_size = (
            (height + 1, width + 1) if self.inception else (height, width)
        )
        if perturbation_logits.shape[-2:] != expected_decoder_size:
            raise ValueError(
                "decoder perturbation size does not match the configured crop policy"
            )
        if mask_logits.shape[-2:] != expected_decoder_size:
            raise ValueError(
                "decoder mask size does not match the configured crop policy"
            )
        if self.inception:
            perturbation_logits = perturbation_logits[..., 1 : height + 1, :width]
            mask_logits = mask_logits[..., 1 : height + 1, :width]
        if perturbation_logits.shape != image.shape:
            raise ValueError("decoder perturbation shape does not match input")
        if mask_logits.shape != (image.shape[0], 1, height, width):
            raise ValueError("decoder mask shape does not match input")

        adv_inf = perturb_budget * torch.tanh(perturbation_logits)
        adv_00 = (torch.tanh(mask_logits) + 1.0) * 0.5
        hard_mask = (adv_00 >= 0.5).to(dtype=adv_00.dtype)
        if self.training:
            use_soft = torch.rand_like(adv_00) < 0.5
            adv_0 = torch.where(use_soft, adv_00, hard_mask.detach())
        else:
            adv_0 = hard_mask.detach()
        adv = torch.clamp(image + adv_inf * adv_0, min=0.0, max=1.0)
        return adv, adv_inf, adv_0, adv_00

    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "generator_type": self.generator_type,
            "encoder": {
                "architecture": "resnet50",
                "weights_enum": "IMAGENET1K_V1",
                "stage": "layer1",
                "prefix": ["conv1", "bn1", "relu", "maxpool", "layer1"],
                "frozen": True,
                "state_sha256": module_state_sha256(self.encoder),
            },
            "decoder": self.decoder.architecture_metadata(),
            "mask_training": "gpg_stochastic_soft_or_detached_hard",
            "legacy_feature_guidance": "disabled_for_frozen_encoder",
            "crop_policy": (
                "legacy_remove_top_row_and_right_column"
                if self.inception
                else "none"
            ),
        }


class DDSCSplitGPGGenerator(DDSCGPGGenerator):
    """Frozen ResNet plus shared adapter/body and split output trunks."""

    generator_type = SPLIT_GENERATOR_TYPE
    decoder_class = SplitLiteGPGDecoder


def parameter_count(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Return a deterministic parameter count for logging and tests."""

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


__all__ = [
    "DDSCGPGGenerator",
    "DDSCSplitGPGGenerator",
    "DepthwiseResidualBlock",
    "FrozenResNet50Layer1",
    "GENERATOR_TYPE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "SharedLiteGPGDecoder",
    "SPLIT_GENERATOR_TYPE",
    "SplitLiteGPGDecoder",
    "module_state_sha256",
    "parameter_count",
]
