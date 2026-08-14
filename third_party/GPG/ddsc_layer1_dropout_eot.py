"""Side-effect-free layer1 frequency-channel dropout for DDSC objectives.

The wrapper intentionally owns no trainable state.  It reuses a frozen,
dropout-free ResNet as its base model and is meant only for attack-objective
evaluation (PGD guidance and adversarial loss), never for clean labels or the
generator encoder.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn


_LOW_FREQUENCY_MASK_CACHE: dict[tuple[object, ...], Tensor] = {}


def clear_low_frequency_mask_cache() -> None:
    """Clear the deterministic FFT-mask cache (primarily for tests)."""

    _LOW_FREQUENCY_MASK_CACHE.clear()


def centered_low_frequency_mask(
    height: int,
    width: int,
    ratio: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return the unshifted rectangular low-frequency mask used by SPGD."""

    key = (height, width, float(ratio), device, dtype)
    cached = _LOW_FREQUENCY_MASK_CACHE.get(key)
    if cached is not None:
        return cached

    keep_h = max(1, int(round(height * ratio)))
    keep_w = max(1, int(round(width * ratio)))
    y0 = (height - keep_h) // 2
    x0 = (width - keep_w) // 2
    mask = torch.zeros((height, width), device=device, dtype=dtype)
    mask[y0 : y0 + keep_h, x0 : x0 + keep_w] = 1.0
    mask = torch.fft.ifftshift(mask, dim=(-2, -1))
    mask = mask.view(1, 1, height, width)
    _LOW_FREQUENCY_MASK_CACHE[key] = mask
    return mask


def high_frequency_projection(feature: Tensor, low_frequency_ratio: float) -> Tensor:
    """Project a layer1 feature map onto the configured high frequencies."""

    values = feature.float()
    height, width = values.shape[-2:]
    spectrum = torch.fft.fft2(values, dim=(-2, -1))
    low_mask = centered_low_frequency_mask(
        height,
        width,
        low_frequency_ratio,
        values.device,
        values.real.dtype,
    )
    return torch.fft.ifft2(
        spectrum * (1.0 - low_mask),
        dim=(-2, -1),
    ).real


class IsolatedResNet50Layer1ChannelDropoutEOT(nn.Module):
    """Clean-inclusive EOT wrapper that drops full ResNet layer1 channels."""

    _REQUIRED_RESNET_ATTRIBUTES = (
        "conv1",
        "bn1",
        "relu",
        "maxpool",
        "layer1",
        "layer2",
        "layer3",
        "layer4",
        "avgpool",
        "fc",
    )

    def __init__(
        self,
        base_model: nn.Module,
        *,
        drop_probability: float,
        channel_ratio: float,
        low_frequency_ratio: float,
        eot_samples: int,
        eot_reduction: str,
    ) -> None:
        super().__init__()
        missing = [
            name
            for name in self._REQUIRED_RESNET_ATTRIBUTES
            if not hasattr(base_model, name)
        ]
        if missing:
            raise ValueError(
                "layer1 dropout requires a torchvision-style ResNet-50; "
                f"missing={missing}"
            )
        if not math.isfinite(drop_probability) or not 0.0 < drop_probability < 1.0:
            raise ValueError("drop_probability must be finite and in (0, 1)")
        for name, value in (
            ("channel_ratio", channel_ratio),
            ("low_frequency_ratio", low_frequency_ratio),
        ):
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        if isinstance(eot_samples, bool) or not isinstance(eot_samples, int):
            raise ValueError("eot_samples must be a positive integer")
        if eot_samples <= 0:
            raise ValueError("eot_samples must be a positive integer")
        if eot_reduction not in {"logits", "loss"}:
            raise ValueError("eot_reduction must be logits or loss")

        self.base_model = base_model
        self.drop_probability = float(drop_probability)
        self.channel_ratio = float(channel_ratio)
        self.low_frequency_ratio = float(low_frequency_ratio)
        self.eot_samples = eot_samples
        self.eot_reduction = eot_reduction
        self.base_model.eval()

    def train(self, mode: bool = True) -> "IsolatedResNet50Layer1ChannelDropoutEOT":
        # The wrapper is stochastic by construction and does not use the
        # module training flag.  Keep the complete frozen classifier in eval,
        # including any nested BatchNorm layers in residual blocks.
        nn.Module.train(self, False)
        return self

    def _forward_to_layer1(self, image: Tensor) -> Tensor:
        feature = self.base_model.conv1(image)
        feature = self.base_model.bn1(feature)
        feature = self.base_model.relu(feature)
        feature = self.base_model.maxpool(feature)
        return self.base_model.layer1(feature)

    def _forward_from_layer1(self, feature: Tensor) -> Tensor:
        feature = self.base_model.layer2(feature)
        feature = self.base_model.layer3(feature)
        feature = self.base_model.layer4(feature)
        feature = self.base_model.avgpool(feature)
        return self.base_model.fc(torch.flatten(feature, 1))

    def _channel_energy(self, feature: Tensor) -> Tensor:
        with torch.no_grad():
            high = high_frequency_projection(
                feature.detach(),
                self.low_frequency_ratio,
            )
            return torch.linalg.vector_norm(high, ord=2, dim=(-2, -1))

    def _dropout_members(self, feature: Tensor, channel_energy: Tensor) -> Tensor:
        batch, channels, _, _ = feature.shape
        keep_probability = 1.0 - self.drop_probability
        with torch.no_grad():
            eligible_count = max(1, int(round(channels * self.channel_ratio)))
            # Stable sorting makes the otherwise unspecified top-k tie case
            # reproducible: equal-energy channels prefer the lower index.
            eligible_indices = torch.argsort(
                channel_energy,
                dim=1,
                descending=True,
                stable=True,
            )[:, :eligible_count]
            eligible = torch.zeros(
                (batch, channels),
                device=feature.device,
                dtype=feature.dtype,
            )
            eligible.scatter_(1, eligible_indices, 1.0)
            keep = (
                torch.rand(
                    (self.eot_samples, batch, channels),
                    device=feature.device,
                )
                < keep_probability
            ).to(feature.dtype)
            eligible = eligible.unsqueeze(0)
            mask = 1.0 - eligible + eligible * keep / keep_probability
            mask = mask.unsqueeze(-1).unsqueeze(-1)
        return feature.unsqueeze(0) * mask

    def forward_members(self, image: Tensor) -> tuple[Tensor, ...]:
        clean_feature = self._forward_to_layer1(image)
        channel_energy = self._channel_energy(clean_feature)
        dropped_features = self._dropout_members(clean_feature, channel_energy)
        feature_members = torch.cat(
            (clean_feature.unsqueeze(0), dropped_features),
            dim=0,
        )
        member_count, batch = feature_members.shape[:2]
        logits = self._forward_from_layer1(feature_members.flatten(0, 1))
        return logits.reshape(member_count, batch, -1).unbind(0)

    def forward(self, image: Tensor) -> Tensor:
        member_logits = self.forward_members(image)
        return torch.stack(member_logits, dim=0).mean(dim=0)


def attack_model_uses_loss_average(model: nn.Module) -> bool:
    """Return whether an EOT wrapper requests member-loss averaging."""

    return (
        hasattr(model, "forward_members")
        and getattr(model, "eot_reduction", "logits") == "loss"
    )


def attack_model_loss_and_logits(
    model: nn.Module,
    normalized_input: Tensor,
    loss_fn: Callable[[Tensor], Tensor],
) -> tuple[Tensor, Tensor]:
    """Apply either loss(mean logits) or mean(member losses), as configured."""

    if attack_model_uses_loss_average(model):
        member_logits = model.forward_members(normalized_input)
        loss = sum(loss_fn(logits) for logits in member_logits) / float(
            len(member_logits)
        )
        mean_logits = torch.stack(member_logits, dim=0).mean(dim=0)
        return loss, mean_logits
    logits = model(normalized_input)
    return loss_fn(logits), logits


__all__ = [
    "IsolatedResNet50Layer1ChannelDropoutEOT",
    "attack_model_loss_and_logits",
    "attack_model_uses_loss_average",
    "centered_low_frequency_mask",
    "clear_low_frequency_mask_cache",
    "high_frequency_projection",
]
