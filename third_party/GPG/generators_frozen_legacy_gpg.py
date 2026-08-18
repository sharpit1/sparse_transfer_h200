"""Frozen ResNet-50 layer1 encoder with the unchanged legacy GPG back end.

This generator is the encoder-only control between the original learned
dual-encoder legacy generator and the parameter-reduced isolated generator.
It preserves the legacy six full residual blocks and independent perturbation
and mask decoders while replacing only the learned image encoder with a frozen
ImageNet ResNet-50 V1 prefix.  No channel adapter is inserted: both encoder
interfaces are 256 channels at one-quarter input resolution.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn

try:
    from .generators_ddsc_gpg import FrozenResNet50Layer1, module_state_sha256
    from .generators_legacy_gpg import NGF, LegacyGPGGenerator, ResidualBlock
except ImportError:
    from generators_ddsc_gpg import (  # type: ignore[no-redef]
        FrozenResNet50Layer1,
        module_state_sha256,
    )
    from generators_legacy_gpg import (  # type: ignore[no-redef]
        NGF,
        LegacyGPGGenerator,
        ResidualBlock,
    )


FROZEN_LEGACY_GENERATOR_TYPE = "gpg_frozen_resnet50_layer1_legacy_decoder_v1"


class FrozenResNetLegacyGPGGenerator(nn.Module):
    """Frozen ResNet layer1 encoder followed by the exact legacy GPG back end."""

    generator_type = FROZEN_LEGACY_GENERATOR_TYPE

    def __init__(
        self,
        resnet50: nn.Module,
        *,
        inception: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = FrozenResNet50Layer1(resnet50)
        self.inception = bool(inception)

        self.resblock1 = ResidualBlock(NGF * 4)
        self.resblock2 = ResidualBlock(NGF * 4)
        self.resblock3 = ResidualBlock(NGF * 4)
        self.resblock4 = ResidualBlock(NGF * 4)
        self.resblock5 = ResidualBlock(NGF * 4)
        self.resblock6 = ResidualBlock(NGF * 4)

        self.upsampl_inf1 = LegacyGPGGenerator._upsample_block(
            NGF * 4,
            NGF * 2,
        )
        self.upsampl_inf2 = LegacyGPGGenerator._upsample_block(NGF * 2, NGF)
        self.blockf_inf = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(NGF, 3, kernel_size=7, padding=0),
        )

        self.upsampl_01 = LegacyGPGGenerator._upsample_block(NGF * 4, NGF * 2)
        self.upsampl_02 = LegacyGPGGenerator._upsample_block(NGF * 2, NGF)
        self.blockf_0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(NGF, 1, kernel_size=7, padding=0),
        )
        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)

    def train(self, mode: bool = True) -> "FrozenResNetLegacyGPGGenerator":
        super().train(mode)
        self.encoder.eval()
        return self

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return (
            parameter for parameter in self.parameters() if parameter.requires_grad
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
            code = self.encoder(image).detach()
        code = self.resblock1(code)
        code = self.resblock2(code)
        code = self.resblock3(code)
        code = self.resblock4(code)
        code = self.resblock5(code)
        code = self.resblock6(code)

        perturbation_logits = self.blockf_inf(
            self.upsampl_inf2(self.upsampl_inf1(code))
        )
        mask_logits = self.blockf_0(self.upsampl_02(self.upsampl_01(code)))
        if self.inception:
            perturbation_logits = self.crop(perturbation_logits)
            mask_logits = self.crop(mask_logits)
        if perturbation_logits.shape != image.shape:
            raise ValueError("decoder perturbation shape does not match input")
        if mask_logits.shape != (
            image.shape[0],
            1,
            image.shape[-2],
            image.shape[-1],
        ):
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
                "output_channels": NGF * 4,
                "adapter": "none",
                "frozen": True,
                "state_sha256": module_state_sha256(self.encoder),
            },
            "decoder": {
                "variant": "legacy_dual_head",
                "base_channels": NGF,
                "input_channels": NGF * 4,
                "residual_blocks": 6,
                "residual_block_type": "legacy_full",
                "upsample_backend": "transpose",
                "shared_upsample_trunk": False,
            },
            "mask_training": "gpg_stochastic_soft_or_detached_hard",
            "legacy_feature_guidance": "disabled_for_frozen_encoder",
            "crop_policy": (
                "legacy_remove_top_row_and_right_column"
                if self.inception
                else "none"
            ),
        }


__all__ = [
    "FROZEN_LEGACY_GENERATOR_TYPE",
    "FrozenResNetLegacyGPGGenerator",
]
