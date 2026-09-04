"""Online per-image clean-ranked layer1 high-frequency energy objective."""

from __future__ import annotations

import hashlib
import math
import os
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

try:
    from ddsc_layer1_dropout_eot import high_frequency_projection
except ImportError:
    from third_party.GPG.ddsc_layer1_dropout_eot import (  # type: ignore[no-redef]
        high_frequency_projection,
    )


CALIBRATION_SCHEMA = "per_image_layer1_hf_online_calibration_v2"
MEASUREMENT_CONTRACT = "dropout_fft_ifft_real_mean_square_v1"
DEFAULT_CHANNEL_RATIO = 0.30
DEFAULT_LOW_FREQUENCY_RATIO = 0.35
DEFAULT_RIDGE_FRACTION = 1.0e-3
DEFAULT_DENOMINATOR_EPS = 1.0e-12
PROGRESS_INTERVAL = 128_000


def _require_ratio(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be finite and in (0, 1]")
    return value


def _require_plain_int(name: str, value: Any, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_sha256(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a SHA-256 string")
    value = value.strip().lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must contain exactly 64 lowercase hex characters")
    return value


def selected_channel_count(channel_count: int, channel_ratio: float) -> int:
    channel_count = _require_plain_int("channel_count", channel_count, positive=True)
    return max(1, int(round(channel_count * _require_ratio("channel_ratio", channel_ratio))))


def dropout_style_high_frequency_energy(
    feature: Tensor,
    low_frequency_ratio: float,
) -> Tensor:
    """Return BxC mean-square energy after the existing dropout HF operator."""

    if feature.ndim != 4 or any(size <= 0 for size in feature.shape):
        raise ValueError("feature must have non-empty shape BxCxHxW")
    high = high_frequency_projection(
        feature,
        _require_ratio("low_frequency_ratio", low_frequency_ratio),
    )
    return high.square().mean(dim=(-2, -1))


class IndexedDataset(Dataset):
    """Add a stable post-subset index for the per-image cache."""

    def __init__(self, dataset: Sequence[Any]) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Any, Any, int]:
        item = self.dataset[index]
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("training dataset must return exactly (image, label)")
        return item[0], item[1], index


@dataclass(frozen=True)
class PerImageLayer1HFCalibration:
    schema: str
    measurement_contract: str
    source_model_sha256: str
    dataset_sha256: str
    dataset_size: int
    low_frequency_ratio: float
    channel_ratio: float
    ridge_fraction: float
    channel_count: int
    selected_channels: Tensor
    selected_clean_energy: Tensor

    @property
    def selected_count(self) -> int:
        return selected_channel_count(self.channel_count, self.channel_ratio)

    def validate(self) -> "PerImageLayer1HFCalibration":
        if self.schema != CALIBRATION_SCHEMA:
            raise ValueError(f"unsupported layer1 HF calibration schema: {self.schema!r}")
        if self.measurement_contract != MEASUREMENT_CONTRACT:
            raise ValueError("layer1 HF measurement contract mismatch")
        _require_sha256("source_model_sha256", self.source_model_sha256)
        _require_sha256("dataset_sha256", self.dataset_sha256)
        _require_plain_int("dataset_size", self.dataset_size, positive=True)
        _require_ratio("low_frequency_ratio", self.low_frequency_ratio)
        _require_ratio("channel_ratio", self.channel_ratio)
        if not math.isfinite(self.ridge_fraction) or self.ridge_fraction < 0.0:
            raise ValueError("ridge_fraction must be finite and non-negative")
        _require_plain_int("channel_count", self.channel_count, positive=True)
        expected_shape = (self.dataset_size, self.selected_count)
        if self.selected_channels.device.type != "cpu":
            raise ValueError("selected_channels cache must remain on CPU")
        if self.selected_channels.dtype != torch.int16:
            raise ValueError("selected_channels cache must use torch.int16")
        if tuple(self.selected_channels.shape) != expected_shape:
            raise ValueError("selected_channels cache shape mismatch")
        if self.selected_clean_energy.device.type != "cpu":
            raise ValueError("selected_clean_energy cache must remain on CPU")
        if self.selected_clean_energy.dtype != torch.float32:
            raise ValueError("selected_clean_energy cache must use torch.float32")
        if tuple(self.selected_clean_energy.shape) != expected_shape:
            raise ValueError("selected_clean_energy cache shape mismatch")
        if int(self.selected_channels.min()) < 0 or int(self.selected_channels.max()) >= self.channel_count:
            raise ValueError("selected channel index is outside the channel range")
        if not bool(torch.isfinite(self.selected_clean_energy).all()):
            raise ValueError("selected clean energy must be finite")
        if bool((self.selected_clean_energy < 0.0).any()):
            raise ValueError("selected clean energy must be non-negative")
        return self

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "measurement_contract": self.measurement_contract,
            "source_model_sha256": self.source_model_sha256,
            "dataset_sha256": self.dataset_sha256,
            "dataset_size": self.dataset_size,
            "low_frequency_ratio": self.low_frequency_ratio,
            "channel_ratio": self.channel_ratio,
            "ridge_fraction": self.ridge_fraction,
            "channel_count": self.channel_count,
            "selected_channels": self.selected_channels,
            "selected_clean_energy": self.selected_clean_energy,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PerImageLayer1HFCalibration":
        required = {
            "schema",
            "measurement_contract",
            "source_model_sha256",
            "dataset_sha256",
            "dataset_size",
            "low_frequency_ratio",
            "channel_ratio",
            "ridge_fraction",
            "channel_count",
            "selected_channels",
            "selected_clean_energy",
        }
        if set(payload) != required:
            raise ValueError("layer1 HF calibration keys mismatch")
        return cls(
            schema=payload["schema"],
            measurement_contract=payload["measurement_contract"],
            source_model_sha256=payload["source_model_sha256"],
            dataset_sha256=payload["dataset_sha256"],
            dataset_size=_require_plain_int("dataset_size", payload["dataset_size"]),
            low_frequency_ratio=float(payload["low_frequency_ratio"]),
            channel_ratio=float(payload["channel_ratio"]),
            ridge_fraction=float(payload["ridge_fraction"]),
            channel_count=_require_plain_int("channel_count", payload["channel_count"]),
            selected_channels=payload["selected_channels"],
            selected_clean_energy=payload["selected_clean_energy"],
        ).validate()


class OnlinePerImageLayer1HighFrequencyEnergy(nn.Module):
    """Record clean profiles on first use and reward the same batch's adv energy."""

    def __init__(
        self,
        *,
        source_model_sha256: str,
        dataset_sha256: str,
        dataset_size: int,
        channel_ratio: float = DEFAULT_CHANNEL_RATIO,
        low_frequency_ratio: float = DEFAULT_LOW_FREQUENCY_RATIO,
        ridge_fraction: float = DEFAULT_RIDGE_FRACTION,
        calibration: PerImageLayer1HFCalibration | None = None,
        denominator_eps: float = DEFAULT_DENOMINATOR_EPS,
    ) -> None:
        super().__init__()
        self.source_model_sha256 = _require_sha256(
            "source_model_sha256", source_model_sha256
        )
        self.dataset_sha256 = _require_sha256("dataset_sha256", dataset_sha256)
        self.dataset_size = _require_plain_int("dataset_size", dataset_size, positive=True)
        self.channel_ratio = _require_ratio("channel_ratio", channel_ratio)
        self.low_frequency_ratio = _require_ratio(
            "low_frequency_ratio", low_frequency_ratio
        )
        if not math.isfinite(ridge_fraction) or ridge_fraction < 0.0:
            raise ValueError("ridge_fraction must be finite and non-negative")
        self.ridge_fraction = float(ridge_fraction)
        if not math.isfinite(denominator_eps) or denominator_eps <= 0.0:
            raise ValueError("denominator_eps must be finite and positive")
        self.denominator_eps = float(denominator_eps)
        self.channel_count: int | None = None
        self.selected_channels: Tensor | None = None
        self.selected_clean_energy: Tensor | None = None
        self.observed = torch.zeros(self.dataset_size, dtype=torch.bool)
        self.observed_count = 0
        self.next_progress = PROGRESS_INTERVAL

        if calibration is not None:
            calibration.validate()
            expected = (
                calibration.source_model_sha256 == self.source_model_sha256
                and calibration.dataset_sha256 == self.dataset_sha256
                and calibration.dataset_size == self.dataset_size
                and calibration.channel_ratio == self.channel_ratio
                and calibration.low_frequency_ratio == self.low_frequency_ratio
                and calibration.ridge_fraction == self.ridge_fraction
            )
            if not expected:
                raise ValueError("loaded layer1 HF calibration contract mismatch")
            self.channel_count = calibration.channel_count
            self.selected_channels = calibration.selected_channels
            self.selected_clean_energy = calibration.selected_clean_energy
            self.observed.fill_(True)
            self.observed_count = self.dataset_size

    def _indices(self, sample_indices: Tensor, batch_size: int) -> Tensor:
        if not isinstance(sample_indices, Tensor) or sample_indices.ndim != 1:
            raise ValueError("sample_indices must be a one-dimensional tensor")
        if sample_indices.numel() != batch_size:
            raise ValueError("sample_indices length differs from batch size")
        indices = sample_indices.detach().to(device="cpu", dtype=torch.long)
        if int(indices.min()) < 0 or int(indices.max()) >= self.dataset_size:
            raise ValueError("sample index is outside the calibration cache")
        if indices.unique().numel() != indices.numel():
            raise ValueError("sample_indices must be unique within a training batch")
        return indices

    def needs_clean_record(self, sample_indices: Tensor) -> bool:
        indices = self._indices(sample_indices, sample_indices.numel())
        flags = self.observed.index_select(0, indices)
        if bool(flags.any()) and not bool(flags.all()):
            raise RuntimeError("training batch mixes calibrated and uncalibrated samples")
        return not bool(flags.all())

    @torch.no_grad()
    def record_clean(self, clean_layer1_feature: Tensor, sample_indices: Tensor) -> None:
        if clean_layer1_feature.ndim != 4:
            raise ValueError("clean layer1 feature must have shape BxCxHxW")
        batch_size, channel_count, _height, _width = clean_layer1_feature.shape
        indices = self._indices(sample_indices, batch_size)
        if bool(self.observed.index_select(0, indices).any()):
            raise RuntimeError("clean profile for a sample may be recorded only once")
        energy = dropout_style_high_frequency_energy(
            clean_layer1_feature.detach(), self.low_frequency_ratio
        )
        if self.channel_count is None:
            if channel_count > torch.iinfo(torch.int16).max:
                raise ValueError("layer1 channel count exceeds int16 cache capacity")
            self.channel_count = channel_count
            top_count = selected_channel_count(channel_count, self.channel_ratio)
            self.selected_channels = torch.empty(
                (self.dataset_size, top_count), dtype=torch.int16
            )
            self.selected_clean_energy = torch.empty(
                (self.dataset_size, top_count), dtype=torch.float32
            )
        if channel_count != self.channel_count:
            raise ValueError("layer1 channel count changed during online calibration")
        assert self.selected_channels is not None
        assert self.selected_clean_energy is not None
        order = torch.argsort(energy, dim=1, descending=True, stable=True)
        top_indices = order[:, : self.selected_channels.shape[1]]
        top_energy = energy.gather(1, top_indices)
        self.selected_channels.index_copy_(
            0, indices, top_indices.to(device="cpu", dtype=torch.int16)
        )
        self.selected_clean_energy.index_copy_(
            0, indices, top_energy.to(device="cpu", dtype=torch.float32)
        )
        self.observed.index_fill_(0, indices, True)
        self.observed_count += batch_size
        if self.observed_count >= self.next_progress or self.observed_count == self.dataset_size:
            print(
                "LAYER1_HF_ONLINE_PROGRESS "
                f"recorded={self.observed_count} total={self.dataset_size}",
                flush=True,
            )
            while self.next_progress <= self.observed_count:
                self.next_progress += PROGRESS_INTERVAL

    def export_calibration(self) -> PerImageLayer1HFCalibration:
        if self.observed_count != self.dataset_size or not bool(self.observed.all()):
            raise RuntimeError(
                "online layer1 HF cache is incomplete: "
                f"recorded={self.observed_count} total={self.dataset_size}"
            )
        assert self.channel_count is not None
        assert self.selected_channels is not None
        assert self.selected_clean_energy is not None
        return PerImageLayer1HFCalibration(
            schema=CALIBRATION_SCHEMA,
            measurement_contract=MEASUREMENT_CONTRACT,
            source_model_sha256=self.source_model_sha256,
            dataset_sha256=self.dataset_sha256,
            dataset_size=self.dataset_size,
            low_frequency_ratio=self.low_frequency_ratio,
            channel_ratio=self.channel_ratio,
            ridge_fraction=self.ridge_fraction,
            channel_count=self.channel_count,
            selected_channels=self.selected_channels,
            selected_clean_energy=self.selected_clean_energy,
        ).validate()

    def forward(self, adversarial_layer1_feature: Tensor, sample_indices: Tensor) -> Tensor:
        if adversarial_layer1_feature.ndim != 4:
            raise ValueError("adversarial layer1 feature must have shape BxCxHxW")
        batch_size, channels, height, width = adversarial_layer1_feature.shape
        indices = self._indices(sample_indices, batch_size)
        if not bool(self.observed.index_select(0, indices).all()):
            raise RuntimeError("clean profile must be recorded before adversarial reward")
        if channels != self.channel_count:
            raise ValueError("adversarial layer1 channel count differs from calibration")
        assert self.selected_channels is not None
        assert self.selected_clean_energy is not None
        channel_indices = self.selected_channels.index_select(0, indices).to(
            device=adversarial_layer1_feature.device, dtype=torch.long
        )
        clean_energy = self.selected_clean_energy.index_select(0, indices).to(
            device=adversarial_layer1_feature.device
        )
        gather_indices = channel_indices.view(batch_size, -1, 1, 1).expand(
            -1, -1, height, width
        )
        selected_feature = adversarial_layer1_feature.gather(1, gather_indices)
        adversarial_energy = dropout_style_high_frequency_energy(
            selected_feature, self.low_frequency_ratio
        )
        ridge = (self.ridge_fraction * clean_energy.median(dim=1).values).clamp_min(
            self.denominator_eps
        )
        return torch.log1p(
            adversarial_energy / (clean_energy + ridge.unsqueeze(1))
        ).mean()


def save_calibration_cache(
    calibration: PerImageLayer1HFCalibration,
    path: str | os.PathLike[str],
) -> str:
    calibration = calibration.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"tmp_{uuid.uuid4().hex}.pth")
    try:
        torch.save((CALIBRATION_SCHEMA, calibration.to_payload()), temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return calibration_cache_sha256(destination)


def calibration_cache_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_calibration_cache(
    path: str | os.PathLike[str],
    *,
    source_model_sha256: str,
    dataset_sha256: str,
    dataset_size: int,
    channel_ratio: float,
    low_frequency_ratio: float,
    ridge_fraction: float,
) -> PerImageLayer1HFCalibration:
    envelope = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(envelope, tuple) or len(envelope) != 2:
        raise ValueError("layer1 HF calibration cache envelope is invalid")
    schema, payload = envelope
    if schema != CALIBRATION_SCHEMA or not isinstance(payload, Mapping):
        raise ValueError("layer1 HF calibration cache schema is invalid")
    calibration = PerImageLayer1HFCalibration.from_payload(payload)
    expected = {
        "source_model_sha256": _require_sha256(
            "source_model_sha256", source_model_sha256
        ),
        "dataset_sha256": _require_sha256("dataset_sha256", dataset_sha256),
        "dataset_size": _require_plain_int("dataset_size", dataset_size, positive=True),
        "channel_ratio": float(channel_ratio),
        "low_frequency_ratio": float(low_frequency_ratio),
        "ridge_fraction": float(ridge_fraction),
    }
    mismatches = {
        key: (getattr(calibration, key), value)
        for key, value in expected.items()
        if getattr(calibration, key) != value
    }
    if mismatches:
        raise ValueError(f"layer1 HF calibration contract mismatch: {mismatches}")
    return calibration


@contextmanager
def capture_resnet_layer1_features(model: nn.Module) -> Iterator[list[Tensor]]:
    layer1 = getattr(model, "layer1", None)
    if not isinstance(layer1, nn.Module):
        raise ValueError("layer1 HF loss requires a torchvision-style ResNet-50")
    captured: list[Tensor] = []

    def capture(
        _module: nn.Module,
        _inputs: tuple[Tensor, ...],
        output: Any,
    ) -> None:
        if not isinstance(output, Tensor):
            raise TypeError("ResNet layer1 must return a tensor")
        captured.append(output)

    handle = layer1.register_forward_hook(capture)
    try:
        yield captured
    finally:
        handle.remove()


def captured_resnet_layer1_batch(
    captured: Sequence[Tensor],
    *,
    batch_size: int,
) -> Tensor:
    if len(captured) != 1:
        raise RuntimeError("layer1 HF loss expected exactly one source-ResNet forward")
    feature = captured[0]
    if feature.ndim != 4 or feature.shape[0] < batch_size:
        raise RuntimeError("captured source-ResNet layer1 feature has an invalid shape")
    if feature.shape[0] % batch_size != 0:
        raise RuntimeError("captured layer1 batch is incompatible with the input batch")
    return feature[:batch_size]


__all__ = [
    "CALIBRATION_SCHEMA",
    "DEFAULT_CHANNEL_RATIO",
    "DEFAULT_DENOMINATOR_EPS",
    "DEFAULT_LOW_FREQUENCY_RATIO",
    "DEFAULT_RIDGE_FRACTION",
    "IndexedDataset",
    "MEASUREMENT_CONTRACT",
    "OnlinePerImageLayer1HighFrequencyEnergy",
    "PerImageLayer1HFCalibration",
    "calibration_cache_sha256",
    "capture_resnet_layer1_features",
    "captured_resnet_layer1_batch",
    "dropout_style_high_frequency_energy",
    "load_calibration_cache",
    "save_calibration_cache",
    "selected_channel_count",
]
