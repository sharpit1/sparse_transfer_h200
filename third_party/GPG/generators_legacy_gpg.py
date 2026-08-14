"""Legacy dual-encoder GPG generator adapted to the DDSC checkpoint API.

The learnable module names intentionally match the original GPG
``GeneratorResnet`` implementation so legacy state dictionaries retain their
established key layout.  DDSC legacy training supplies ``grad_AE`` and uses
both the pixel-space PGD guidance loss and the fifth feature-guidance output.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn


NGF = 64
LEGACY_GENERATOR_TYPE = "gpg_legacy_dual_encoder_v1"


class LegacyGPGGenerator(nn.Module):
    """Original learned dual-encoder GPG architecture."""

    generator_type = LEGACY_GENERATOR_TYPE

    def __init__(
        self,
        *,
        inception: bool = False,
        eps: float = 1.0,
        evaluate: bool = False,
    ) -> None:
        super().__init__()
        self.inception = bool(inception)
        self.eps = float(eps)
        self.evaluate = bool(evaluate)

        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, NGF, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(NGF),
            nn.ReLU(True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(
                NGF,
                NGF * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(NGF * 2),
            nn.ReLU(True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(
                NGF * 2,
                NGF * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(NGF * 4),
            nn.ReLU(True),
        )

        self.resblock1 = ResidualBlock(NGF * 4)
        self.resblock2 = ResidualBlock(NGF * 4)
        self.resblock3 = ResidualBlock(NGF * 4)
        self.resblock4 = ResidualBlock(NGF * 4)
        self.resblock5 = ResidualBlock(NGF * 4)
        self.resblock6 = ResidualBlock(NGF * 4)

        self.Grad_block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, NGF, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(NGF),
            nn.ReLU(True),
        )
        self.Grad_block2 = nn.Sequential(
            nn.Conv2d(
                NGF,
                NGF * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(NGF * 2),
            nn.ReLU(True),
        )
        self.Grad_block3 = nn.Sequential(
            nn.Conv2d(
                NGF * 2,
                NGF * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(NGF * 4),
            nn.ReLU(True),
        )
        self.upsampl_inf1 = self._upsample_block(NGF * 4, NGF * 2)
        self.upsampl_inf2 = self._upsample_block(NGF * 2, NGF)
        self.blockf_inf = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(NGF, 3, kernel_size=7, padding=0),
        )

        self.upsampl_01 = self._upsample_block(NGF * 4, NGF * 2)
        self.upsampl_02 = self._upsample_block(NGF * 2, NGF)
        self.blockf_0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(NGF, 1, kernel_size=7, padding=0),
        )
        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)

    @staticmethod
    def _upsample_block(input_channels: int, output_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(True),
        )

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return (
            parameter for parameter in self.parameters() if parameter.requires_grad
        )

    def _clean_encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.block3(self.block2(self.block1(image)))
        code = self.resblock1(feature)
        code = self.resblock2(code)
        code = self.resblock3(code)
        code = self.resblock4(code)
        code = self.resblock5(code)
        code = self.resblock6(code)
        return feature, code

    def _gradient_encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.Grad_block3(self.Grad_block2(self.Grad_block1(image)))

    def forward(
        self,
        image: torch.Tensor,
        eps: float | torch.Tensor | None = None,
        grad_AE: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        # Preserve both historical calls: netG(x, eps, x_adv) and netG(x, x_adv).
        if torch.is_tensor(eps) and grad_AE is None:
            grad_AE = eps
            eps = None
        if not torch.is_floating_point(image):
            raise TypeError("image must be floating point")
        perturb_budget = self.eps if eps is None else float(eps)
        if perturb_budget < 0.0:
            raise ValueError("eps must be non-negative")

        clean_feature, code = self._clean_encode(image)
        feature_guidance_loss = None
        if grad_AE is not None:
            feature_guidance_loss = torch.norm(
                clean_feature - self._gradient_encode(grad_AE),
                p=2,
            )

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
        if self.evaluate or not self.training:
            adv_0 = hard_mask.detach()
        else:
            use_soft = torch.rand_like(adv_00) < 0.5
            adv_0 = torch.where(use_soft, adv_00, hard_mask.detach())
        adv = torch.clamp(image + adv_inf * adv_0, min=0.0, max=1.0)

        outputs: tuple[torch.Tensor, ...] = (adv, adv_inf, adv_0, adv_00)
        if feature_guidance_loss is not None:
            outputs += (feature_guidance_loss,)
        return outputs

    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "generator_type": self.generator_type,
            "encoder": {
                "variant": "legacy_learned_dual",
                "base_channels": NGF,
                "downsample_stages": 2,
                "gradient_branch": {
                    "present": True,
                    "frozen": False,
                    "used_by_ddsc": True,
                },
                "frozen": False,
            },
            "decoder": {
                "variant": "legacy_dual_head",
                "base_channels": NGF,
                "residual_blocks": 6,
                "upsample_backend": "transpose",
                "shared_upsample_trunk": False,
            },
            "mask_training": "gpg_stochastic_soft_or_detached_hard",
            "legacy_feature_guidance": "enabled_trainable_by_ddsc",
            "crop_policy": (
                "legacy_remove_top_row_and_right_column"
                if self.inception
                else "none"
            ),
        }


class ResidualBlock(nn.Module):
    def __init__(self, num_filters: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                num_filters,
                num_filters,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                num_filters,
                num_filters,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(num_filters),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return feature + self.block(feature)


# Compatibility alias matching the historical module API.
GeneratorResnet = LegacyGPGGenerator


__all__ = [
    "GeneratorResnet",
    "LEGACY_GENERATOR_TYPE",
    "LegacyGPGGenerator",
    "ResidualBlock",
]
