from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from torchvision import datasets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HOME", str(PROJECT_ROOT / ".cache" / "home"))
os.environ.setdefault("USERPROFILE", os.environ["HOME"])

EVALUATOR_PATH = (
    PROJECT_ROOT / "third_party" / "GPG" / "tools" / "evaluate_ddsc_gpg_all_model_t.py"
)
SPEC = importlib.util.spec_from_file_location(
    "ddsc_all_model_progress_contract_under_test",
    EVALUATOR_PATH,
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - importlib guard
    raise RuntimeError(f"cannot load evaluator module: {EVALUATOR_PATH}")
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def _victim_state_contract(sha256: str = "0" * 64) -> dict[str, object]:
    return {
        "schema": 1,
        "kind": "sorted_key_dtype_shape_and_tensor_bytes_sha256",
        "tensor_count": 1,
        "total_tensor_bytes": 4,
        "byte_order": sys.byteorder,
        "sha256": sha256,
    }


def _victim_execution_contract(
    *,
    fused_attn: bool = False,
) -> dict[str, object]:
    root = torch.nn.Module()
    attention = torch.nn.Identity()
    attention.fused_attn = fused_attn
    root.add_module("attention", attention)
    return EVALUATOR.victim_model_execution_contract(root)


def _evaluation_contract(
    *,
    dataset_n: int = 4,
    batch_size: int = 2,
    victim_sha256: str = "0" * 64,
) -> dict[str, object]:
    model_name = "res50"
    return {
        "schema": EVALUATOR.EVALUATION_CONTRACT_SCHEMA,
        "framework_and_device": EVALUATOR.framework_device_contract(
            torch.device("cpu")
        ),
        "dataset": {
            "ordered_manifest": {
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
            },
            "image_loader": EVALUATOR.image_loader_contract(),
        },
        "models": [
            {
                "name": model_name,
                "implementation": EVALUATOR.MODEL_IMPLEMENTATIONS[model_name],
                "state_dict": _victim_state_contract(victim_sha256),
                "execution": _victim_execution_contract(),
            }
        ],
        "metrics": EVALUATOR.metrics_contract([model_name]),
        "data_loader": {"batch_size": batch_size},
        "input_transform": EVALUATOR.input_transform_contract(),
    }


def _progress_state(
    contract: dict[str, object],
    *,
    next_index: int = 2,
    complete: bool = False,
) -> dict[str, object]:
    model_metrics = {"res50": EVALUATOR.empty_model_metrics()}
    model_metrics["res50"]["n"] = next_index
    perturbation_metrics = EVALUATOR.empty_perturbation_metrics()
    perturbation_metrics["n"] = next_index
    return EVALUATOR.progress_state_payload(
        contract,
        next_index,
        model_metrics,
        perturbation_metrics,
        complete=complete,
    )


def _validate(
    state: dict[str, object],
    contract: dict[str, object],
    *,
    dataset_n: int = 4,
):
    return EVALUATOR.validate_progress_state(
        state,
        expected_contract=contract,
        dataset_n=dataset_n,
        model_names=["res50"],
    )


def test_toy_victim_state_hash_is_deterministic_and_detects_mutation() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 3),
            torch.nn.BatchNorm1d(3),
        ).to(device="cpu", dtype=torch.float32)
    model.eval()

    first = EVALUATOR.victim_model_state_contract(model)
    second = EVALUATOR.victim_model_state_contract(model)
    assert first == second
    assert first["tensor_count"] > 0
    assert first["total_tensor_bytes"] > 0
    assert len(first["sha256"]) == 64

    with torch.no_grad():
        model[0].weight[0, 0].add_(1.0)
    mutated = EVALUATOR.victim_model_state_contract(model)
    assert mutated["sha256"] != first["sha256"]


def test_identical_victim_state_with_different_fused_attention_is_distinguished() -> (
    None
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        unfused = torch.nn.Module()
        unfused.add_module("projection", torch.nn.Linear(4, 3))
        attention = torch.nn.Identity()
        attention.fused_attn = False
        unfused.add_module("attention", attention)
    fused = copy.deepcopy(unfused)
    fused.attention.fused_attn = True

    assert EVALUATOR.victim_model_state_contract(
        unfused
    ) == EVALUATOR.victim_model_state_contract(fused)
    unfused_execution = EVALUATOR.victim_model_execution_contract(unfused)
    fused_execution = EVALUATOR.victim_model_execution_contract(fused)
    assert unfused_execution["entries"] == [{"name": "attention", "fused_attn": False}]
    assert fused_execution["entries"] == [{"name": "attention", "fused_attn": True}]
    assert unfused_execution["sha256"] != fused_execution["sha256"]


def test_victim_execution_contract_rejects_non_boolean_fused_attention() -> None:
    model = torch.nn.Identity()
    model.fused_attn = 1

    with pytest.raises(ValueError, match="plain boolean"):
        EVALUATOR.victim_model_execution_contract(model)


def test_ordered_imagefolder_manifest_detects_order_label_content_and_size_changes(
    tmp_path: Path,
) -> None:
    first_class = tmp_path / "class_a"
    second_class = tmp_path / "class_b"
    first_class.mkdir()
    second_class.mkdir()
    first_image = first_class / "a.jpg"
    second_image = second_class / "b.jpg"
    first_image.write_bytes(b"first")
    second_image.write_bytes(b"second")

    dataset = datasets.ImageFolder(str(tmp_path))
    original_samples = list(dataset.samples)
    baseline = EVALUATOR.ordered_dataset_manifest(dataset, tmp_path, 2)
    assert baseline["schema"] == 2
    assert baseline["kind"] == "ordered_imagefolder_prefix_with_content_sha256"
    assert baseline["record_fields"][-1] == "content_sha256"

    dataset.samples = list(reversed(original_samples))
    reordered = EVALUATOR.ordered_dataset_manifest(dataset, tmp_path, 2)
    assert reordered["ordered_records_sha256"] != baseline["ordered_records_sha256"]

    dataset.samples = list(original_samples)
    sample_path, original_label = dataset.samples[0]
    dataset.samples[0] = (sample_path, 1 - original_label)
    relabelled = EVALUATOR.ordered_dataset_manifest(dataset, tmp_path, 2)
    assert relabelled["ordered_records_sha256"] != baseline["ordered_records_sha256"]

    dataset.samples = list(original_samples)
    original_size = first_image.stat().st_size
    first_image.write_bytes(b"FIRST")
    assert first_image.stat().st_size == original_size
    content_changed = EVALUATOR.ordered_dataset_manifest(dataset, tmp_path, 2)
    assert (
        content_changed["ordered_records_sha256"] != baseline["ordered_records_sha256"]
    )

    first_image.write_bytes(b"first")
    restored = EVALUATOR.ordered_dataset_manifest(dataset, tmp_path, 2)
    assert restored["ordered_records_sha256"] == baseline["ordered_records_sha256"]

    second_image.write_bytes(b"second-with-a-different-size")
    resized = EVALUATOR.ordered_dataset_manifest(dataset, tmp_path, 2)
    assert resized["ordered_records_sha256"] != baseline["ordered_records_sha256"]


def test_evaluation_dataset_forces_pil_loader_and_bilinear_resize(
    tmp_path: Path,
) -> None:
    class_dir = tmp_path / "class_a"
    class_dir.mkdir()
    (class_dir / "a.jpg").write_bytes(b"sample")
    original_backend = EVALUATOR.torchvision.get_image_backend()
    try:
        EVALUATOR.torchvision.set_image_backend("accimage")
        dataset = EVALUATOR.build_evaluation_dataset(tmp_path)
        loader_contract = EVALUATOR.image_loader_contract()
        resize = dataset.transform.transforms[0]

        assert dataset.loader is EVALUATOR.pil_loader
        assert resize.interpolation is EVALUATOR.InterpolationMode.BILINEAR
        assert resize.antialias is True
        assert loader_contract["forced_backend"] == "PIL"
        assert loader_contract["torchvision_image_backend"] == "accimage"
        assert loader_contract["callable"].endswith(".pil_loader")
    finally:
        EVALUATOR.torchvision.set_image_backend(original_backend)


def test_valid_progress_survives_json_round_trip() -> None:
    contract = _evaluation_contract()
    state = _progress_state(contract)
    loaded = json.loads(json.dumps(state, allow_nan=False))

    next_index, model_metrics, perturbation_metrics = _validate(loaded, contract)
    assert next_index == 2
    assert model_metrics["res50"]["n"] == 2
    assert perturbation_metrics["n"] == 2

    complete_state = _progress_state(contract, next_index=4, complete=True)
    complete_loaded = json.loads(json.dumps(complete_state, allow_nan=False))
    assert _validate(complete_loaded, contract)[0] == 4


def test_bool_cannot_spoof_integer_in_saved_contract() -> None:
    contract = _evaluation_contract(batch_size=1)
    state = copy.deepcopy(_progress_state(contract))
    state["contract"]["data_loader"]["batch_size"] = True

    with pytest.raises(ValueError, match="contract differs"):
        _validate(state, contract)


@pytest.mark.parametrize(
    ("next_index", "message"),
    [
        (True, "plain integer"),
        (-1, "outside the dataset range"),
        (5, "outside the dataset range"),
        (1, "batch boundary"),
    ],
)
def test_next_index_must_be_in_range_and_on_a_batch_boundary(
    next_index: int,
    message: str,
) -> None:
    contract = _evaluation_contract(dataset_n=4, batch_size=2)
    state = _progress_state(contract, next_index=next_index)

    with pytest.raises(ValueError, match=message):
        _validate(state, contract)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"n": 1}, "n mismatch"),
        ({"clean_correct": 3}, "count exceeds n"),
        ({"adv_correct_on_clean": 2}, "exceeds adv_correct"),
        ({"attack_success_on_clean": 0}, "attack counts are inconsistent"),
        ({"prediction_flip_count": 0}, "below its implied minimum"),
        ({"prediction_flip_count": 2}, "exceeds its implied maximum"),
    ],
)
def test_model_count_invariants_are_fail_closed(
    updates: dict[str, int],
    message: str,
) -> None:
    contract = _evaluation_contract()
    state = _progress_state(contract)
    state["model_metrics"]["res50"].update(
        {
            "clean_correct": 2,
            "adv_correct": 1,
            "adv_correct_on_clean": 1,
            "attack_success_on_clean": 1,
            "prediction_flip_count": 1,
        }
    )
    state["model_metrics"]["res50"].update(updates)

    with pytest.raises(ValueError, match=message):
        _validate(state, contract)


def test_nan_perturbation_accumulator_is_rejected() -> None:
    contract = _evaluation_contract()
    state = _progress_state(contract)
    state["perturbation_metrics"]["applied_l1_sum"] = float("nan")

    with pytest.raises(ValueError, match="finite/nonnegative"):
        _validate(state, contract)


def test_saved_victim_sha_mismatch_is_rejected() -> None:
    contract = _evaluation_contract(victim_sha256="0" * 64)
    state = copy.deepcopy(_progress_state(contract))
    state["contract"]["models"][0]["state_dict"]["sha256"] = "f" * 64

    with pytest.raises(ValueError, match="contract differs"):
        _validate(state, contract)


def test_saved_victim_execution_contract_mutation_is_rejected() -> None:
    contract = _evaluation_contract()
    state = copy.deepcopy(_progress_state(contract))
    saved_execution = state["contract"]["models"][0]["execution"]
    assert EVALUATOR._victim_execution_contract_is_valid(saved_execution)
    saved_execution["entries"][0]["fused_attn"] = True
    assert not EVALUATOR._victim_execution_contract_is_valid(saved_execution)

    with pytest.raises(ValueError, match="contract differs"):
        _validate(state, contract)


def test_expected_victim_execution_contract_type_is_validated() -> None:
    contract = _evaluation_contract()
    contract["models"][0]["execution"]["entries"][0]["fused_attn"] = 0
    state = _progress_state(contract)

    with pytest.raises(ValueError, match="model order is inconsistent"):
        _validate(state, contract)


def test_framework_contract_records_stable_runtime_policy_types() -> None:
    framework = EVALUATOR.framework_device_contract(torch.device("cpu"))
    assert json.loads(json.dumps(framework, allow_nan=False)) == framework
    for key in (
        "deterministic_algorithms",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cudnn_allow_tf32",
        "cuda_matmul_allow_tf32",
    ):
        assert type(framework[key]) is bool

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
        record = framework[key]
        assert set(record) == {"available", "value"}
        assert type(record["available"]) is bool
        assert (
            (type(record["value"]) is bool)
            if record["available"]
            else (record["value"] is None)
        )

    precision = framework["float32_matmul_precision"]
    assert set(precision) == {"available", "value"}
    assert type(precision["available"]) is bool
    assert (
        (isinstance(precision["value"], str) and precision["value"])
        if precision["available"]
        else precision["value"] is None
    )

    assert set(framework["sdpa"]) == {
        "flash",
        "memory_efficient",
        "math",
        "cudnn",
        "fp16_bf16_reduction_math",
    }
    for record in framework["sdpa"].values():
        assert set(record) == {"available", "value"}
        assert type(record["available"]) is bool
        assert (
            (type(record["value"]) is bool)
            if record["available"]
            else (record["value"] is None)
        )

    assert set(framework["autocast"]) == {"cpu", "cuda", "cache_enabled"}
    cache_policy = framework["autocast"]["cache_enabled"]
    assert set(cache_policy) == {"available", "value"}
    assert type(cache_policy["available"]) is bool
    assert (
        type(cache_policy["value"]) is bool
        if cache_policy["available"]
        else cache_policy["value"] is None
    )
    for device_type in ("cpu", "cuda"):
        device_policy = framework["autocast"][device_type]
        assert set(device_policy) == {"enabled", "dtype"}
        assert type(device_policy["enabled"]["available"]) is bool
        assert type(device_policy["dtype"]["available"]) is bool
        if device_policy["enabled"]["available"]:
            assert type(device_policy["enabled"]["value"]) is bool
        else:
            assert device_policy["enabled"]["value"] is None
        if device_policy["dtype"]["available"]:
            assert isinstance(device_policy["dtype"]["value"], str)
        else:
            assert device_policy["dtype"]["value"] is None


def test_framework_float32_policy_change_changes_contract() -> None:
    original_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        high = EVALUATOR.framework_device_contract(torch.device("cpu"))
        torch.set_float32_matmul_precision("medium")
        medium = EVALUATOR.framework_device_contract(torch.device("cpu"))
    finally:
        torch.set_float32_matmul_precision(original_precision)

    assert high["float32_matmul_precision"] == {
        "available": True,
        "value": "high",
    }
    assert medium["float32_matmul_precision"] == {
        "available": True,
        "value": "medium",
    }
    assert high != medium


def test_framework_cuda_low_precision_reduction_changes_contract() -> None:
    matmul = torch.backends.cuda.matmul
    cuda_backend = torch.backends.cuda
    required_attributes = (
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
    )
    if any(not hasattr(matmul, name) for name in required_attributes):
        pytest.skip("CUDA reduced-precision matmul policy is unavailable")
    math_sdp_getter = getattr(
        cuda_backend,
        "fp16_bf16_reduction_math_sdp_allowed",
        None,
    )
    math_sdp_setter = getattr(
        cuda_backend,
        "allow_fp16_bf16_reduction_math_sdp",
        None,
    )
    if not callable(math_sdp_getter) or not callable(math_sdp_setter):
        pytest.skip("CUDA reduced-precision math-SDPA policy is unavailable")

    original_fp16 = matmul.allow_fp16_reduced_precision_reduction
    original_bf16 = matmul.allow_bf16_reduced_precision_reduction
    original_math_sdp = math_sdp_getter()
    try:
        matmul.allow_fp16_reduced_precision_reduction = False
        matmul.allow_bf16_reduced_precision_reduction = False
        math_sdp_setter(False)
        disabled = EVALUATOR.framework_device_contract(torch.device("cpu"))

        matmul.allow_fp16_reduced_precision_reduction = True
        matmul.allow_bf16_reduced_precision_reduction = True
        math_sdp_setter(True)
        enabled = EVALUATOR.framework_device_contract(torch.device("cpu"))
    finally:
        matmul.allow_fp16_reduced_precision_reduction = original_fp16
        matmul.allow_bf16_reduced_precision_reduction = original_bf16
        math_sdp_setter(original_math_sdp)

    for contract, expected in ((disabled, False), (enabled, True)):
        assert contract["cuda_matmul_allow_fp16_reduced_precision_reduction"] == {
            "available": True,
            "value": expected,
        }
        assert contract["cuda_matmul_allow_bf16_reduced_precision_reduction"] == {
            "available": True,
            "value": expected,
        }
        assert contract["sdpa"]["fp16_bf16_reduction_math"] == {
            "available": True,
            "value": expected,
        }
    assert disabled != enabled


def test_saved_framework_runtime_policy_mutation_is_rejected() -> None:
    contract = _evaluation_contract()
    state = copy.deepcopy(_progress_state(contract))
    precision = state["contract"]["framework_and_device"]["float32_matmul_precision"]
    assert precision["available"] is True
    precision["value"] = "medium" if precision["value"] != "medium" else "high"

    with pytest.raises(ValueError, match="contract differs"):
        _validate(state, contract)


def test_framework_runtime_policy_bool_spoof_is_rejected() -> None:
    contract = _evaluation_contract()
    contract["framework_and_device"]["timm_use_fused_attn"] = {
        "available": True,
        "value": 1,
    }
    state = _progress_state(contract)

    with pytest.raises(ValueError, match="timm_use_fused_attn"):
        _validate(state, contract)


def test_schema_v2_progress_cannot_resume_silently() -> None:
    contract = _evaluation_contract()
    state = _progress_state(contract)
    state["schema"] = 2

    with pytest.raises(ValueError, match="unsupported saved progress schema: 2"):
        _validate(state, contract)


def test_schema_v2_evaluation_contract_is_rejected() -> None:
    contract = _evaluation_contract()
    contract["schema"] = 2
    state = _progress_state(contract)

    with pytest.raises(ValueError, match="evaluation contract schema is invalid"):
        _validate(state, contract)


def test_no_resume_invalidates_stale_state_then_writes_valid_zero_state(
    tmp_path: Path,
) -> None:
    contract = _evaluation_contract()
    state_path = tmp_path / "progress_state.json"
    EVALUATOR.atomic_json(state_path, _progress_state(contract))

    EVALUATOR.invalidate_progress_state_for_fresh_run(state_path)
    with state_path.open("r", encoding="utf-8") as handle:
        tombstone = json.load(handle)
    assert tombstone == {
        "schema": EVALUATOR.PROGRESS_STATE_SCHEMA,
        "status": "invalidated",
        "reason": "fresh --no-resume setup has not completed",
    }
    with pytest.raises(ValueError, match="saved progress keys mismatch"):
        _validate(tombstone, contract)

    model_metrics, perturbation_metrics = EVALUATOR.initialize_fresh_progress_state(
        state_path,
        contract=contract,
        dataset_n=4,
        model_names=["res50"],
    )
    with state_path.open("r", encoding="utf-8") as handle:
        initialized = json.load(handle)
    next_index, loaded_models, loaded_perturbation = _validate(initialized, contract)
    assert next_index == 0
    assert loaded_models == model_metrics
    assert loaded_perturbation == perturbation_metrics
    assert all(value == 0 for value in loaded_models["res50"].values())
    assert all(value == 0 for value in loaded_perturbation.values())


def test_no_resume_invalidates_before_device_setup_can_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "results"
    out_dir.mkdir()
    state_path = out_dir / "progress_state.json"
    state_path.write_text('{"stale": true}', encoding="utf-8")
    args = Namespace(
        checkpoint=str(tmp_path / "missing-checkpoint.pt"),
        imagenet_val_root=str(tmp_path / "missing-imagenet"),
        out_dir=str(out_dir),
        models=["res50"],
        batch_size=2,
        num_workers=0,
        samples=0,
        seed=1,
        device="cuda",
        state_every_batches=1,
        resume=False,
    )
    monkeypatch.setattr(EVALUATOR, "parse_args", lambda: args)
    monkeypatch.setattr(EVALUATOR, "seed_everything", lambda _seed: None)
    monkeypatch.setattr(EVALUATOR.torch, "set_num_threads", lambda _count: None)
    monkeypatch.setattr(EVALUATOR.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested but is unavailable"):
        EVALUATOR.main()

    with state_path.open("r", encoding="utf-8") as handle:
        invalidated = json.load(handle)
    assert invalidated["status"] == "invalidated"
    assert invalidated["schema"] == EVALUATOR.PROGRESS_STATE_SCHEMA
    assert "stale" not in invalidated
