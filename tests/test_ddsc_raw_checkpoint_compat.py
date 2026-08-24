from __future__ import annotations

import importlib.util
import os
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import torchvision


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HOME", str(PROJECT_ROOT / ".cache" / "home"))
os.environ.setdefault("USERPROFILE", os.environ["HOME"])

EVALUATOR_PATH = (
    PROJECT_ROOT / "third_party" / "GPG" / "tools" / "evaluate_ddsc_gpg_all_model_t.py"
)
SPEC = importlib.util.spec_from_file_location(
    "ddsc_raw_checkpoint_compat_under_test",
    EVALUATOR_PATH,
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - importlib guard
    raise RuntimeError(f"cannot load evaluator module: {EVALUATOR_PATH}")
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def _raw_args(
    *,
    architecture_mode: str | None,
    model_type: str = "res50",
    eps_pixels: float = 10.0,
    gpg_generator_mode: str = "auto",
) -> Namespace:
    return Namespace(
        architecture_mode=architecture_mode,
        model_type=model_type,
        eps_pixels=eps_pixels,
        target=None,
        completed_epoch=None,
        egs_tsaa_tk=None,
        gpg_generator_mode=gpg_generator_mode,
    )


@pytest.mark.parametrize("architecture_mode", ["gpg", "tsaa", "egs_tsaa"])
def test_raw_source_state_dict_loads_strictly(
    tmp_path: Path,
    architecture_mode: str,
) -> None:
    training = EVALUATOR.build_original_generator(
        architecture_mode,
        inception=False,
        eps=10.0 / 255.0,
        inference=False,
    )
    checkpoint = tmp_path / f"{architecture_mode}.pth"
    torch.save(training.state_dict(), checkpoint)

    requested_mode = None if architecture_mode == "gpg" else architecture_mode
    loaded, payload = EVALUATOR.build_generator_from_compatible_checkpoint(
        checkpoint,
        _raw_args(architecture_mode=requested_mode),
    )

    assert payload["kind"] == "raw_state_dict"
    assert payload["model_type"] == "res50"
    assert payload["eps_pixels"] == 10.0
    assert payload["image_size"] == 224
    assert payload["target"] == -1
    assert payload["conditioner_contract"] is None
    assert payload["architecture"]["architecture_mode"] == architecture_mode
    assert payload["architecture"]["raw_checkpoint_wrapper"] == "root"
    assert list(loaded.state_dict()) == list(training.state_dict())
    for name, tensor in training.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor)


def test_raw_tsaa_egs_schema_requires_explicit_architecture(tmp_path: Path) -> None:
    training = EVALUATOR.build_original_generator(
        "tsaa",
        inception=False,
        eps=10.0 / 255.0,
        inference=False,
    )
    checkpoint = tmp_path / "ambiguous.pth"
    torch.save(training.state_dict(), checkpoint)

    with pytest.raises(ValueError, match="same state schema"):
        EVALUATOR.build_generator_from_compatible_checkpoint(
            checkpoint,
            _raw_args(architecture_mode=None),
        )


def test_raw_gpg_isolated_state_dict_is_detected(tmp_path: Path) -> None:
    training = EVALUATOR.GPGGeneratorResnet(
        inception=False,
        eps=10.0 / 255.0,
        evaluate=False,
        encoder_mode="isolated",
        encoder_backbone=torchvision.models.resnet50(weights=None),
    )
    checkpoint = tmp_path / "gpg_isolated.pth"
    torch.save(training.state_dict(), checkpoint)

    loaded, payload = EVALUATOR.build_generator_from_compatible_checkpoint(
        checkpoint,
        _raw_args(architecture_mode=None),
    )

    assert loaded.encoder_mode == "isolated"
    assert payload["architecture"]["encoder_mode"] == "isolated"
    assert list(loaded.state_dict()) == list(training.state_dict())


def test_inception_source_transform_and_victim_resize_contract() -> None:
    contract = EVALUATOR.input_transform_contract(299)

    assert contract["ordered_steps"][0]["size"] == 300
    assert contract["ordered_steps"][1]["size"] == 299
    assert contract["victim_prediction_resize"] == {
        "incv3_size": [299, 299],
        "other_model_size": [224, 224],
        "mode": "bilinear",
        "align_corners": False,
    }


def test_prediction_resizes_each_victim_to_its_native_input() -> None:
    class CaptureShape(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shape: tuple[int, ...] | None = None

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            self.shape = tuple(images.shape)
            return images.new_zeros((images.shape[0], 1000))

    resnet = CaptureShape()
    inception = CaptureShape()
    EVALUATOR.predict(resnet, "res50", torch.rand(1, 3, 299, 299))
    EVALUATOR.predict(inception, "incv3", torch.rand(1, 3, 224, 224))

    assert resnet.shape == (1, 3, 224, 224)
    assert inception.shape == (1, 3, 299, 299)


def test_raw_checkpoint_requires_noninferable_model_type_and_eps(
    tmp_path: Path,
) -> None:
    training = EVALUATOR.build_original_generator(
        "gpg",
        inception=False,
        eps=10.0 / 255.0,
        inference=False,
    )
    checkpoint = tmp_path / "gpg.pth"
    torch.save(training.state_dict(), checkpoint)

    args = _raw_args(architecture_mode=None)
    args.model_type = None
    with pytest.raises(ValueError, match="--model-type"):
        EVALUATOR.build_generator_from_compatible_checkpoint(checkpoint, args)

    args.model_type = "res50"
    args.eps_pixels = None
    with pytest.raises(ValueError, match="--eps-pixels"):
        EVALUATOR.build_generator_from_compatible_checkpoint(checkpoint, args)


def test_existing_ddsc_inference_tuple_still_loads(tmp_path: Path) -> None:
    generator = EVALUATOR.build_original_generator(
        "tsaa",
        inception=False,
        eps=10.0 / 255.0,
        inference=True,
    )
    payload = {
        "kind": "inference",
        "generator_type": EVALUATOR.generator_type_for_architecture_mode("tsaa"),
        "model_type": "res50",
        "target": -1,
        "eps_pixels": 10.0,
        "image_size": 224,
        "completed_epoch": 0,
        "architecture": EVALUATOR.generator_architecture_metadata(
            generator,
            "tsaa",
        ),
        "conditioner_contract": None,
        "generator_state_dict": generator.state_dict(),
    }
    checkpoint = tmp_path / "ddsc.inference.pth"
    torch.save(("ddsc_gpg_inference_v3", payload), checkpoint)

    loaded, loaded_payload = EVALUATOR.build_generator_from_compatible_checkpoint(
        checkpoint,
        _raw_args(architecture_mode=None, model_type=None, eps_pixels=None),
    )

    assert loaded_payload["kind"] == "inference"
    assert loaded_payload["generator_type"] == payload["generator_type"]
    assert list(loaded.state_dict()) == list(generator.state_dict())
