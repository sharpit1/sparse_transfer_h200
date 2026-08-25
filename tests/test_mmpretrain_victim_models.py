from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HOME", str(PROJECT_ROOT / ".cache" / "home"))
os.environ.setdefault("USERPROFILE", os.environ["HOME"])

EVALUATOR_PATH = (
    PROJECT_ROOT / "third_party" / "GPG" / "tools" / "evaluate_ddsc_gpg_all_model_t.py"
)
SPEC = importlib.util.spec_from_file_location(
    "ddsc_mmpretrain_victim_models_under_test",
    EVALUATOR_PATH,
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - importlib guard
    raise RuntimeError(f"cannot load evaluator module: {EVALUATOR_PATH}")
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


EXPECTED_MODEL_KEYS = (
    "mm_deit_small_4xb256_in1k",
    "mm_tnt_small_p16_3rdparty_in1k",
    "mm_swin_tiny_16xb64_in1k",
    "mm_twins_pcpvt_small_3rdparty_8xb128_in1k",
    "mm_vit_base_p16_32xb128_mae_in1k",
)


def test_detailed_mmpretrain_keywords_are_registered_without_short_aliases() -> None:
    assert tuple(EVALUATOR.MMPRETRAIN_MODEL_SPECS) == EXPECTED_MODEL_KEYS
    assert tuple(EVALUATOR.MODEL_ORDER[-5:]) == EXPECTED_MODEL_KEYS
    assert len(EVALUATOR.MODEL_ORDER) == 23
    assert not {"deit", "tnt", "twins"} & set(EVALUATOR.MODEL_ORDER)
    assert set(EVALUATOR.MODEL_ORDER) == set(EVALUATOR.MODEL_IMPLEMENTATIONS)


def test_checkpoint_specs_have_full_hashes_and_correct_vit_asset() -> None:
    for model_name, spec in EVALUATOR.MMPRETRAIN_MODEL_SPECS.items():
        assert model_name.startswith("mm_")
        assert len(spec["checkpoint_sha256"]) == 64
        short_hash = Path(spec["checkpoint_filename"]).stem.rsplit("-", 1)[-1]
        assert spec["checkpoint_sha256"].startswith(short_hash)
    vit = EVALUATOR.MMPRETRAIN_MODEL_SPECS[
        "mm_vit_base_p16_32xb128_mae_in1k"
    ]
    assert vit["model_name"] == "vit-base-p16_32xb128-mae_in1k"
    assert vit["checkpoint_filename"].startswith("vit-base-p16_")
    assert "twins" not in vit["checkpoint_filename"]


def test_verified_local_checkpoint_is_passed_to_get_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = EXPECTED_MODEL_KEYS[0]
    spec = EVALUATOR.MMPRETRAIN_MODEL_SPECS[name]
    checkpoint = tmp_path / spec["checkpoint_filename"]
    checkpoint.write_bytes(b"test checkpoint")
    monkeypatch.setattr(
        EVALUATOR,
        "sha256_file",
        lambda path: spec["checkpoint_sha256"],
    )
    monkeypatch.setattr(
        EVALUATOR,
        "_mmpretrain_distribution_versions",
        lambda: dict(EVALUATOR.MMPRETRAIN_DISTRIBUTIONS),
    )

    calls: list[tuple[str, str, str]] = []
    fake_mmpretrain = types.ModuleType("mmpretrain")

    def fake_get_model(
        model_name: str,
        *,
        pretrained: str,
        device: str,
    ) -> torch.nn.Module:
        calls.append((model_name, pretrained, device))
        return torch.nn.Linear(2, 2)

    fake_mmpretrain.get_model = fake_get_model
    monkeypatch.setitem(sys.modules, "mmpretrain", fake_mmpretrain)

    model = EVALUATOR.build_mmpretrain_model(name, tmp_path)

    assert isinstance(model, torch.nn.Module)
    assert calls == [(spec["model_name"], str(checkpoint), "cpu")]


def test_mmpretrain_checkpoint_hash_mismatch_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = EXPECTED_MODEL_KEYS[1]
    spec = EVALUATOR.MMPRETRAIN_MODEL_SPECS[name]
    checkpoint = tmp_path / spec["checkpoint_filename"]
    checkpoint.write_bytes(b"wrong checkpoint")
    monkeypatch.setattr(EVALUATOR, "sha256_file", lambda path: "0" * 64)

    with pytest.raises(RuntimeError, match="checkpoint hash mismatch"):
        EVALUATOR.build_mmpretrain_model(name, tmp_path)


def test_mmpretrain_contract_records_selected_models_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        EVALUATOR,
        "_mmpretrain_distribution_versions",
        lambda: dict(EVALUATOR.MMPRETRAIN_DISTRIBUTIONS),
    )
    selected = ["res50", EXPECTED_MODEL_KEYS[2], EXPECTED_MODEL_KEYS[4]]

    contract = EVALUATOR.mmpretrain_victim_contract(selected, tmp_path)

    assert contract["packages"] == EVALUATOR.MMPRETRAIN_DISTRIBUTIONS
    assert [entry["name"] for entry in contract["models"]] == [
        EXPECTED_MODEL_KEYS[2],
        EXPECTED_MODEL_KEYS[4],
    ]
