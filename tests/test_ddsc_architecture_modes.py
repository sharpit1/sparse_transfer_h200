import copy
import inspect
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn as nn


ROOT = pathlib.Path(__file__).resolve().parents[1]
GPG_DIR = ROOT / "third_party" / "GPG"
if str(GPG_DIR) not in sys.path:
    sys.path.insert(0, str(GPG_DIR))

import DDSC_GPG_train as ddsc_train  # noqa: E402
from DDSC_GPG_train_training_reuse import (  # noqa: E402
    build_training_reuse_source,
)
from ddsc_architecture_modes import (  # noqa: E402
    EGSStructuredMask,
    build_original_generator,
    controller_support_mask,
    egs_conditioner_contract,
    forward_generator_inference,
    forward_generator_training,
    generator_architecture_metadata,
    generator_type_for_architecture_mode,
    legacy_cw_loss,
    quantization_loss,
    validate_egs_conditioner_contract,
)
from generators_ddsc_gpg import parameter_count  # noqa: E402


class _TinyEGSResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.bn1 = nn.Identity()
        self.relu = nn.ReLU()
        self.maxpool = nn.Identity()
        self.layer1 = nn.Conv2d(4, 4, kernel_size=1, bias=False)
        self.layer2 = nn.Conv2d(4, 4, kernel_size=1, bias=False)
        self.layer3 = nn.Conv2d(4, 4, kernel_size=1, bias=False)
        self.layer4 = nn.Sequential(nn.Conv2d(4, 4, kernel_size=1, bias=False))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, 1000, bias=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feature = self.conv1(image)
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)
        feature = self.layer1(feature)
        feature = self.layer2(feature)
        feature = self.layer3(feature)
        feature = self.layer4(feature)
        return self.fc(torch.flatten(self.avgpool(feature), 1))


class ArchitectureModeContractTests(unittest.TestCase):
    def test_training_reuse_wrapper_reuses_adv00_and_keeps_previous_forward(
        self,
    ) -> None:
        source = build_training_reuse_source()
        compile(source, "DDSC_GPG_train.py", "exec")
        self.assertEqual(source.count("current_temporal_mask = adv_00"), 1)
        self.assertNotIn("with generator_deployment_mask_mode(net_g):", source)
        self.assertEqual(
            source.count("previous_adv_00 = forward_generator_training("),
            1,
        )
        self.assertIn("_CM-training-adv00", source)

    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def test_parser_exposes_all_modes_and_canonicalizes_egs_tssa_alias(self) -> None:
        parser = ddsc_train.build_parser()
        for mode in ("simple", "gpg", "tsaa", "egs_tsaa"):
            device = "cuda" if mode in {"tsaa", "egs_tsaa"} else "cpu"
            args = parser.parse_args(["--device", device, "--architecture_mode", mode])
            ddsc_train.validate_args(args)
            self.assertEqual(args.architecture_mode, mode)
        alias = parser.parse_args(
            ["--device", "cuda", "--architecture_mode", "egs_tssa"]
        )
        ddsc_train.validate_args(alias)
        self.assertEqual(alias.architecture_mode, "egs_tsaa")

    def test_temporal_intersection_options_cover_all_architecture_modes(self) -> None:
        parser = ddsc_train.build_parser()
        for mode in ("simple", "gpg", "tsaa", "egs_tsaa"):
            device = "cuda" if mode in {"tsaa", "egs_tsaa"} else "cpu"
            args = parser.parse_args(
                [
                    "--device",
                    device,
                    "--architecture_mode",
                    mode,
                    "--intersection_reg_mode",
                    "normalized_l2",
                    "--intersection_reg_lambda",
                    "0.25",
                ]
            )
            ddsc_train.validate_args(args)
            self.assertEqual(args.intersection_reg_mode, "normalized_l2")
            self.assertEqual(args.intersection_reg_lambda, 0.25)

        fixed = parser.parse_args(
            [
                "--intersection_reg_mode",
                "fixed",
                "--intersection_reg_lambda",
                "0.25",
            ]
        )
        ddsc_train.validate_args(fixed)
        self.assertEqual(fixed.intersection_reg_mode, "fixed")

        invalid_enabled = parser.parse_args(
            ["--intersection_reg_mode", "normalized_l2"]
        )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            ddsc_train.validate_args(invalid_enabled)
        invalid_disabled = parser.parse_args(["--intersection_reg_lambda", "0.1"])
        with self.assertRaisesRegex(ValueError, "must be zero"):
            ddsc_train.validate_args(invalid_disabled)
        invalid_eps = parser.parse_args(["--intersection_reg_eps", "0"])
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            ddsc_train.validate_args(invalid_eps)

    def test_intersection_regularization_starts_two_epochs_after_warmup(
        self,
    ) -> None:
        self.assertEqual(ddsc_train.INTERSECTION_REGULARIZATION_DELAY_EPOCHS, 2)
        for epoch in range(4):
            self.assertFalse(
                ddsc_train.intersection_regularization_active(
                    epoch,
                    warmup_epochs=2,
                )
            )
        self.assertTrue(
            ddsc_train.intersection_regularization_active(
                4,
                warmup_epochs=2,
            )
        )
        self.assertTrue(
            ddsc_train.intersection_regularization_active(
                5,
                warmup_epochs=2,
            )
        )
        with self.assertRaisesRegex(ValueError, "epoch must be"):
            ddsc_train.intersection_regularization_active(-1, warmup_epochs=2)
        with self.assertRaisesRegex(ValueError, "warmup_epochs must be"):
            ddsc_train.intersection_regularization_active(0, warmup_epochs=-1)

    def test_v9_checkpoint_args_default_temporal_intersection_to_off(self) -> None:
        train_args = vars(ddsc_train.build_parser().parse_args([]))
        for key in ddsc_train.INTERSECTION_REGULARIZATION_DEFAULTS:
            train_args.pop(key)
        normalized = ddsc_train._normalize_checkpoint_train_args(
            train_args,
            checkpoint_format=ddsc_train.PRE_INTERSECTION_TRAINING_CHECKPOINT_FORMAT,
        )
        for key, value in ddsc_train.INTERSECTION_REGULARIZATION_DEFAULTS.items():
            self.assertEqual(normalized[key], value)

        with self.assertRaisesRegex(ValueError, "v11.*incomplete"):
            ddsc_train._normalize_checkpoint_train_args(
                train_args,
                checkpoint_format=ddsc_train.CHECKPOINT_FORMAT,
            )
        ambiguous_v9 = dict(normalized)
        with self.assertRaisesRegex(ValueError, "must not contain v10\\+"):
            ddsc_train._normalize_checkpoint_train_args(
                ambiguous_v9,
                checkpoint_format=(
                    ddsc_train.PRE_INTERSECTION_TRAINING_CHECKPOINT_FORMAT
                ),
            )

    def test_normalized_temporal_intersection_has_exploratory_gradient(self) -> None:
        current = torch.tensor(
            [[[[0.8, 0.2]]]], dtype=torch.float64, requires_grad=True
        )
        previous = torch.tensor(
            [[[[1.0, 0.0]]]], dtype=torch.float64, requires_grad=True
        )
        loss = ddsc_train.normalized_temporal_intersection_loss(
            current,
            previous,
            eps=1.0e-12,
        )
        loss.backward()
        self.assertAlmostEqual(float(loss), 0.64 / 0.68, places=6)
        self.assertGreater(float(current.grad[0, 0, 0, 0]), 0.0)
        self.assertLess(float(current.grad[0, 0, 0, 1]), 0.0)
        self.assertIsNone(previous.grad)

    def test_temporal_intersection_uses_hard_previous_support_and_batch_sum(
        self,
    ) -> None:
        current = torch.tensor(
            [
                [[[0.8, 0.2]]],
                [[[0.8, 0.2]]],
            ],
            requires_grad=True,
        )
        previous = torch.tensor(
            [
                [[[0.51, 0.49]]],
                [[[0.99, 0.01]]],
            ],
            requires_grad=True,
        )
        loss = ddsc_train.normalized_temporal_intersection_loss(
            current,
            previous,
            eps=1.0e-12,
        )
        self.assertAlmostEqual(float(loss), 2.0 * 0.64 / 0.68, places=6)
        loss.backward()
        self.assertTrue(torch.all(current.grad[:, 0, 0, 0] > 0.0))
        self.assertTrue(torch.all(current.grad[:, 0, 0, 1] < 0.0))
        self.assertIsNone(previous.grad)

        half_current = current.detach().half().requires_grad_(True)
        half_loss = ddsc_train.normalized_temporal_intersection_loss(
            half_current,
            previous.detach().half(),
            eps=1.0e-12,
        )
        self.assertEqual(half_loss.dtype, torch.float32)
        half_loss.backward()
        self.assertTrue(torch.isfinite(half_current.grad).all())

    def test_fixed_temporal_intersection_uses_previous_support_denominator(
        self,
    ) -> None:
        current = torch.tensor(
            [
                [[[0.8, 0.4, 0.6]]],
                [[[0.7, 0.5, 0.3]]],
            ],
            requires_grad=True,
        )
        previous = torch.tensor(
            [
                [[[0.9, 0.8, 0.0]]],
                [[[0.0, 0.0, 0.0]]],
            ],
            requires_grad=True,
        )
        loss = ddsc_train.fixed_temporal_intersection_loss(
            current,
            previous,
            eps=1.0e-12,
        )
        self.assertAlmostEqual(float(loss), (0.64 + 0.16) / 2.0, places=6)
        loss.backward()
        self.assertGreater(float(current.grad[0, 0, 0, 0]), 0.0)
        self.assertGreater(float(current.grad[0, 0, 0, 1]), 0.0)
        self.assertEqual(float(current.grad[0, 0, 0, 2]), 0.0)
        self.assertTrue(torch.equal(current.grad[1], torch.zeros_like(current.grad[1])))
        self.assertIsNone(previous.grad)

    def test_hard_temporal_intersection_metrics_and_logging_contract(self) -> None:
        current = torch.tensor(
            [
                [[[0.6, 0.49], [0.5, 0.1]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
            ],
            requires_grad=True,
        )
        previous = torch.tensor(
            [
                [[[0.7, 0.8], [0.4, 0.1]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
            ]
        )
        metrics = ddsc_train.hard_temporal_intersection_metrics(
            current,
            previous,
            threshold=0.5,
        )
        self.assertEqual(metrics["intersection_count"].tolist(), [1.0, 0.0])
        self.assertEqual(metrics["density"].tolist(), [0.25, 0.0])
        self.assertEqual(metrics["rprev_percent"].tolist(), [50.0, 0.0])
        self.assertEqual(metrics["rcurr_percent"].tolist(), [50.0, 0.0])
        self.assertAlmostEqual(
            metrics["jaccard_percent"][0].item(),
            100.0 / 3.0,
        )
        self.assertEqual(metrics["jaccard_percent"][1].item(), 0.0)
        self.assertTrue(
            all(not value.requires_grad for value in metrics.values())
        )
        training_source = inspect.getsource(ddsc_train._run_training_impl)
        self.assertIn("HARD_OVERLAP_BATCH", training_source)
        self.assertIn("HARD_OVERLAP_EPOCH", training_source)
        self.assertIn("empty_denominator=record_as_zero", training_source)

    def test_temporal_overlap_uses_deployed_egs_support(self) -> None:
        continuous = torch.tensor(
            [[[[0.9, 0.8], [0.7, 0.6]]]], requires_grad=True
        )
        structured = torch.tensor(
            [[[[1.0, 0.0], [0.0, 1.0]]]], requires_grad=True
        )
        self.assertIs(
            ddsc_train.temporal_overlap_mask("gpg", continuous),
            continuous,
        )
        overlap = ddsc_train.temporal_overlap_mask(
            "egs_tsaa",
            continuous,
            structured_mask=structured,
        )
        self.assertTrue(torch.equal(overlap, continuous * structured))
        overlap.sum().backward()
        self.assertTrue(torch.equal(continuous.grad, structured.detach()))
        self.assertIsNone(structured.grad)

    def test_frozen_generator_snapshot_is_independent_and_gradient_free(self) -> None:
        generator = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Dropout(p=0.5),
            nn.Conv2d(1, 1, kernel_size=1),
        ).train()
        generator[0].eval()
        generator.evaluate = False
        expected_training = tuple(module.training for module in generator.modules())
        with ddsc_train.generator_deployment_mask_mode(generator):
            self.assertTrue(all(not module.training for module in generator.modules()))
            self.assertTrue(generator.evaluate)
            self.assertTrue(
                all(parameter.requires_grad for parameter in generator.parameters())
            )
        self.assertEqual(
            tuple(module.training for module in generator.modules()),
            expected_training,
        )
        self.assertFalse(generator.evaluate)

        snapshot = ddsc_train.frozen_generator_snapshot(generator)
        self.assertIsNot(snapshot, generator)
        self.assertFalse(snapshot.training)
        self.assertTrue(snapshot.evaluate)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in snapshot.parameters())
        )
        for original, frozen in zip(
            generator.state_dict().values(), snapshot.state_dict().values()
        ):
            self.assertTrue(torch.equal(original, frozen))
            self.assertNotEqual(original.data_ptr(), frozen.data_ptr())

    def test_non_simple_frequency_dropout_requires_member_loss_average(self) -> None:
        parser = ddsc_train.build_parser()
        invalid = parser.parse_args(
            [
                "--device",
                "cuda",
                "--architecture_mode",
                "tsaa",
                "--layer1_dropout_mode",
                "frequency_channel",
                "--layer1_dropout_p",
                "0.2",
            ]
        )
        with self.assertRaisesRegex(ValueError, "eot_reduction=loss"):
            ddsc_train.validate_args(invalid)
        valid = parser.parse_args(
            [
                "--device",
                "cuda",
                "--architecture_mode",
                "tsaa",
                "--layer1_dropout_mode",
                "frequency_channel",
                "--layer1_dropout_p",
                "0.2",
                "--layer1_dropout_eot_reduction",
                "loss",
            ]
        )
        ddsc_train.validate_args(valid)

    def test_original_generator_parameter_and_state_contracts(self) -> None:
        expected = {
            "gpg": (8_592_516, 136),
            "tsaa": (8_213_572, 118),
            "egs_tsaa": (8_213_572, 118),
        }
        for mode, (expected_parameters, expected_entries) in expected.items():
            with self.subTest(mode=mode):
                training = build_original_generator(
                    mode, inception=False, eps=10.0 / 255.0, inference=False
                )
                inference = build_original_generator(
                    mode, inception=False, eps=10.0 / 255.0, inference=True
                )
                self.assertEqual(parameter_count(training), expected_parameters)
                self.assertEqual(len(training.state_dict()), expected_entries)
                self.assertEqual(
                    list(training.state_dict()), list(inference.state_dict())
                )
                for train_tensor, inference_tensor in zip(
                    training.state_dict().values(), inference.state_dict().values()
                ):
                    self.assertEqual(train_tensor.shape, inference_tensor.shape)
                    self.assertEqual(train_tensor.dtype, inference_tensor.dtype)
                inference.load_state_dict(training.state_dict(), strict=True)
                residual_dropouts = [
                    module
                    for module in training.modules()
                    if isinstance(module, nn.Dropout)
                ]
                self.assertEqual(len(residual_dropouts), 6)
                self.assertTrue(
                    all(module.p == 0.5 for module in residual_dropouts)
                )

    def test_mode_forward_contracts_and_gpg_feature_guidance(self) -> None:
        image = torch.rand(1, 3, 32, 32)
        delta = torch.zeros_like(image)
        structured = torch.ones(1, 1, 32, 32)
        torch.manual_seed(9)
        gpg = build_original_generator(
            "gpg", inception=False, eps=0.1, inference=False
        ).train()
        adv, adv_inf, adv_0, adv_00, auxiliary = forward_generator_training(
            gpg,
            "gpg",
            image,
            0.1,
            pgd_delta=delta,
        )
        self.assertEqual(adv.shape, image.shape)
        self.assertEqual(adv_inf.shape, image.shape)
        self.assertEqual(adv_0.shape, structured.shape)
        self.assertEqual(adv_00.shape, structured.shape)
        self.assertTrue(torch.isfinite(adv).all())
        self.assertIn("feature_guidance", auxiliary)
        self.assertEqual(auxiliary["feature_guidance"].ndim, 0)

        training_flags = tuple(module.training for module in gpg.modules())
        rng_before = torch.get_rng_state().clone()
        with ddsc_train.generator_deployment_mask_mode(gpg):
            deployment_mask = forward_generator_training(
                gpg,
                "gpg",
                image,
                0.1,
                pgd_delta=delta,
            )[3]
            self.assertTrue(gpg.evaluate)
            self.assertTrue(deployment_mask.requires_grad)
        self.assertEqual(
            tuple(module.training for module in gpg.modules()),
            training_flags,
        )
        self.assertFalse(gpg.evaluate)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))

        # The unmodified TSAA/EGS-TSSA training sources hard-code CUDA for the
        # stochastic branch. Their evaluate=True source classes are device-safe.
        for mode in ("tsaa", "egs_tsaa"):
            with self.subTest(mode=mode):
                generator = build_original_generator(
                    mode, inception=False, eps=0.1, inference=True
                ).eval()
                adv, adv_inf, adv_0, adv_00 = forward_generator_inference(
                    generator,
                    mode,
                    image,
                    0.1,
                    structured_mask=structured if mode == "egs_tsaa" else None,
                )
                self.assertEqual(adv.shape, image.shape)
                self.assertEqual(adv_inf.shape, image.shape)
                self.assertEqual(adv_0.shape, structured.shape)
                self.assertEqual(adv_00.shape, structured.shape)
                self.assertTrue(torch.isfinite(adv).all())

    def test_legacy_cw_and_mode_specific_quantization_match_source_formulas(self) -> None:
        logits = torch.linspace(-2.0, 3.0, 2000).reshape(2, 1000)
        target = torch.tensor([0, 999])
        one_hot = torch.nn.functional.one_hot(target, 1000).to(logits.dtype)
        real = torch.sum(one_hot * logits, dim=1)
        other = torch.max((1 - one_hot) * logits - one_hot * 10000, dim=1).values
        expected_untargeted = torch.sum(torch.maximum(real - other, torch.zeros_like(real)))
        expected_targeted = torch.sum(torch.maximum(other - real, torch.zeros_like(real)))
        self.assertEqual(legacy_cw_loss(logits, target), expected_untargeted)
        self.assertEqual(
            legacy_cw_loss(logits, target, targeted=True), expected_targeted
        )

        continuous = torch.tensor([[[[0.2, 0.8], [0.6, 0.4]]]])
        structured = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        hard = (continuous >= 0.5).to(continuous.dtype)
        self.assertEqual(
            quantization_loss("tsaa", continuous),
            torch.sum((hard - continuous) ** 2),
        )
        structured_hard = hard * structured
        self.assertEqual(
            quantization_loss(
                "egs_tsaa",
                continuous,
                structured_mask=structured,
                egs_smooth_loss="soft",
            ),
            torch.sum((structured_hard - continuous) ** 2),
        )
        self.assertEqual(
            quantization_loss(
                "egs_tsaa",
                continuous,
                structured_mask=structured,
                egs_smooth_loss="hard",
            ),
            torch.sum((structured_hard - continuous * structured) ** 2),
        )

    def test_ddsc_changes_only_sparse_multiplier(self) -> None:
        adversarial = torch.tensor(3.0)
        sparse = torch.tensor(7.0)
        quantization = torch.tensor(11.0)
        lambda2 = 0.2
        lambda_a, lambda_b = 0.1, 0.4
        adv_inf = torch.tensor([[[[1.0, 2.0]]]])
        pgd_delta = torch.tensor([[[[0.5, 1.5]]]])
        feature_guidance = torch.tensor(13.0)
        lambda3 = 0.3
        for mode in ("simple", "gpg", "tsaa", "egs_tsaa"):
            with self.subTest(mode=mode):
                guided = mode in {"simple", "gpg"}
                kwargs = {
                    "adv_inf": adv_inf if guided else None,
                    "pgd_delta": pgd_delta if guided else None,
                    "feature_guidance": (
                        feature_guidance if mode == "gpg" else None
                    ),
                }
                loss_a, guidance_a = ddsc_train.assemble_mode_loss(
                    mode,
                    adversarial_loss=adversarial,
                    sparse_loss=sparse,
                    quantization_loss_value=quantization,
                    lambda1=lambda_a,
                    lambda2=lambda2,
                    lambda3=lambda3,
                    **kwargs,
                )
                loss_b, guidance_b = ddsc_train.assemble_mode_loss(
                    mode,
                    adversarial_loss=adversarial,
                    sparse_loss=sparse,
                    quantization_loss_value=quantization,
                    lambda1=lambda_b,
                    lambda2=lambda2,
                    lambda3=lambda3,
                    **kwargs,
                )
                pixel = torch.sum((adv_inf - pgd_delta) ** 2)
                expected_guidance = adversarial.new_zeros(())
                if guided:
                    expected_guidance = lambda3 * pixel
                if mode == "gpg":
                    expected_guidance = (
                        expected_guidance + lambda3 * feature_guidance
                    )
                expected_a = (
                    adversarial
                    + lambda_a * sparse
                    + lambda2 * quantization
                    + expected_guidance
                )
                self.assertEqual(loss_a, expected_a)
                self.assertEqual(guidance_a, expected_guidance)
                self.assertEqual(guidance_b, expected_guidance)
                self.assertTrue(
                    torch.allclose(
                        loss_b - loss_a,
                        (lambda_b - lambda_a) * sparse,
                    )
                )

    def test_training_device_resolution_is_concrete_and_fail_closed(self) -> None:
        self.assertEqual(ddsc_train.resolve_training_device("cpu"), torch.device("cpu"))
        with (
            mock.patch.object(ddsc_train.torch.cuda, "is_available", return_value=True),
            mock.patch.object(ddsc_train.torch.cuda, "current_device", return_value=1),
            mock.patch.object(ddsc_train.torch.cuda, "device_count", return_value=2),
        ):
            self.assertEqual(
                ddsc_train.resolve_training_device("cuda"),
                torch.device("cuda:1"),
            )
            self.assertEqual(
                ddsc_train.resolve_training_device("cuda:0"),
                torch.device("cuda:0"),
            )
            with self.assertRaisesRegex(ValueError, "device index is unavailable"):
                ddsc_train.resolve_training_device("cuda:2")
        with mock.patch.object(
            ddsc_train.torch.cuda, "is_available", return_value=False
        ):
            with self.assertRaisesRegex(RuntimeError, "not available"):
                ddsc_train.resolve_training_device("cuda")

    def test_tsaa_and_egs_training_forward_scope_the_requested_cuda_device(self) -> None:
        image = mock.Mock(device=torch.device("cuda:1"))
        outputs = tuple(torch.tensor(float(index)) for index in range(5))
        with mock.patch.object(torch.cuda, "device") as cuda_device:
            tsaa = mock.Mock(return_value=outputs[:4])
            forward_generator_training(tsaa, "tsaa", image, 0.1)
            cuda_device.assert_called_once_with(torch.device("cuda:1"))

        structured = mock.Mock(device=torch.device("cuda:1"))
        with mock.patch.object(torch.cuda, "device") as cuda_device:
            egs = mock.Mock(return_value=outputs)
            forward_generator_training(
                egs,
                "egs_tsaa",
                image,
                0.1,
                structured_mask=structured,
            )
            cuda_device.assert_called_once_with(torch.device("cuda:1"))

    def test_egs_controller_support_matches_structured_deployment_mask(self) -> None:
        continuous = torch.tensor([[[[0.9, 0.9], [0.1, 0.8]]]])
        structured = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])
        expected = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        self.assertTrue(
            torch.equal(
                controller_support_mask(
                    "egs_tsaa", continuous, structured_mask=structured
                ),
                expected,
            )
        )

    def test_egs_cam_uses_all_logits_and_exact_box_density(self) -> None:
        torch.manual_seed(3)
        model = _TinyEGSResNet().eval().requires_grad_(False)
        conditioner = EGSStructuredMask(
            model,
            model_type="res50",
            image_size=224,
            topk_fraction=0.6,
        )
        try:
            logits, mask = conditioner.clean_logits_and_mask(
                torch.rand(1, 3, 224, 224)
            )
        finally:
            conditioner.close()
        self.assertEqual(logits.shape, (1, 1000))
        self.assertEqual(mask.shape, (1, 1, 224, 224))
        self.assertTrue(torch.equal(mask, (mask > 0).to(mask.dtype)))
        expected_boxes = int((224 / 8) ** 2 * 0.6)
        self.assertEqual(int(mask.sum().item()), expected_boxes * 8 * 8)

    def test_egs_cam_hooks_are_transient_on_success_and_failure(self) -> None:
        torch.manual_seed(4)
        model = _TinyEGSResNet().eval().requires_grad_(False)
        cam_layer = model.layer4[-1]
        before = (len(cam_layer._forward_hooks), len(cam_layer._backward_hooks))
        conditioner = EGSStructuredMask(
            model,
            model_type="res50",
            image_size=224,
            topk_fraction=0.6,
        )
        conditioner.clean_logits_and_mask(torch.rand(1, 3, 224, 224))
        self.assertEqual(
            (len(cam_layer._forward_hooks), len(cam_layer._backward_hooks)),
            before,
        )
        with mock.patch.object(
            conditioner,
            "mask_from_captured",
            side_effect=RuntimeError("injected CAM failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected CAM failure"):
                conditioner.clean_logits_and_mask(torch.rand(1, 3, 224, 224))
        self.assertEqual(
            (len(cam_layer._forward_hooks), len(cam_layer._backward_hooks)),
            before,
        )
        conditioner.close()
        conditioner.close()

    def test_egs_conditioner_contract_rejects_live_state_mismatch(self) -> None:
        attack_contract = {
            "schema": 1,
            "model_type": "res50",
            "architecture": "resnet50",
            "weights_enum": "IMAGENET1K_V1",
            "state_sha256": "1" * 64,
        }
        contract = egs_conditioner_contract(
            model_type="res50",
            image_size=224,
            topk_fraction=0.6,
            attack_model_contract=attack_contract,
        )
        validate_egs_conditioner_contract(contract, actual_contract=contract)
        changed = copy.deepcopy(contract)
        changed["attack_model_contract"]["state_sha256"] = "2" * 64
        with self.assertRaisesRegex(ValueError, "differs from the training checkpoint"):
            validate_egs_conditioner_contract(
                contract,
                actual_contract=changed,
            )

    def test_egs_conditioner_contract_ignores_ambient_default_device(self) -> None:
        attack_contract = {
            "schema": 1,
            "model_type": "res50",
            "architecture": "resnet50",
            "weights_enum": "IMAGENET1K_V1",
            "state_sha256": "1" * 64,
        }
        previous_device = torch.get_default_device()
        try:
            torch.set_default_device("meta")
            contract = egs_conditioner_contract(
                model_type="res50",
                image_size=224,
                topk_fraction=0.6,
                attack_model_contract=attack_contract,
            )
        finally:
            torch.set_default_device(previous_device)
        self.assertEqual(contract["box_index_shape"], [784, 64])
        self.assertEqual(len(contract["box_index_sha256"]), 64)

    def test_non_simple_inference_checkpoint_reconstructs_source_eval_class(self) -> None:
        for mode in ("gpg", "tsaa", "egs_tsaa"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                training = build_original_generator(
                    mode, inception=False, eps=10.0 / 255.0, inference=False
                )
                if mode == "egs_tsaa":
                    setattr(training, "ddsc_egs_tk", 0.6)
                architecture = generator_architecture_metadata(training, mode)
                conditioner_contract = None
                if mode == "egs_tsaa":
                    conditioner_contract = egs_conditioner_contract(
                        model_type="res50",
                        image_size=224,
                        topk_fraction=0.6,
                        attack_model_contract={
                            "schema": 1,
                            "model_type": "res50",
                            "architecture": "resnet50",
                            "weights_enum": "IMAGENET1K_V1",
                            "state_sha256": "1" * 64,
                        },
                    )
                payload = {
                    "kind": "inference",
                    "generator_type": generator_type_for_architecture_mode(mode),
                    "model_type": "res50",
                    "target": -1,
                    "eps_pixels": 10,
                    "image_size": 224,
                    "completed_epoch": 0,
                    "architecture": architecture,
                    "conditioner_contract": conditioner_contract,
                    "generator_state_dict": copy.deepcopy(training.state_dict()),
                }
                checkpoint = pathlib.Path(temporary) / f"{mode}.pth"
                torch.save((ddsc_train.INFERENCE_CHECKPOINT_FORMAT, payload), checkpoint)
                reconstructed, loaded = ddsc_train.build_generator_from_inference_checkpoint(
                    checkpoint
                )
                self.assertEqual(loaded["generator_type"], payload["generator_type"])
                self.assertTrue(bool(getattr(reconstructed, "evaluate")))
                image = torch.rand(1, 3, 32, 32)
                structured = (
                    torch.ones(1, 1, 32, 32) if mode == "egs_tsaa" else None
                )
                with torch.no_grad():
                    _, _, mask, _ = forward_generator_inference(
                        reconstructed,
                        mode,
                        image,
                        10.0 / 255.0,
                        structured_mask=structured,
                    )
                self.assertTrue(torch.equal(mask, (mask > 0).to(mask.dtype)))
                if mode == "gpg":
                    legacy_payload = copy.deepcopy(payload)
                    legacy_payload.pop("conditioner_contract")
                    legacy_checkpoint = (
                        pathlib.Path(temporary) / "gpg_legacy_v2.pth"
                    )
                    torch.save(
                        (
                            ddsc_train.LEGACY_INFERENCE_CHECKPOINT_FORMAT,
                            legacy_payload,
                        ),
                        legacy_checkpoint,
                    )
                    migrated = ddsc_train.load_inference_checkpoint(
                        legacy_checkpoint
                    )
                    self.assertIsNone(migrated["conditioner_contract"])

    def test_legacy_v2_egs_inference_checkpoint_is_rejected(self) -> None:
        training = build_original_generator(
            "egs_tsaa", inception=False, eps=10.0 / 255.0, inference=False
        )
        setattr(training, "ddsc_egs_tk", 0.6)
        payload = {
            "kind": "inference",
            "generator_type": generator_type_for_architecture_mode("egs_tsaa"),
            "model_type": "res50",
            "target": -1,
            "eps_pixels": 10,
            "image_size": 224,
            "completed_epoch": 0,
            "architecture": generator_architecture_metadata(training, "egs_tsaa"),
            "generator_state_dict": copy.deepcopy(training.state_dict()),
        }
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = pathlib.Path(temporary) / "legacy_egs.pth"
            torch.save(
                (ddsc_train.LEGACY_INFERENCE_CHECKPOINT_FORMAT, payload),
                checkpoint,
            )
            with self.assertRaisesRegex(ValueError, "lacks the required"):
                ddsc_train.load_inference_checkpoint(checkpoint)

    def test_v3_egs_inference_cross_field_tampering_is_rejected(self) -> None:
        attack_contract = {
            "schema": 1,
            "model_type": "res50",
            "architecture": "resnet50",
            "weights_enum": "IMAGENET1K_V1",
            "state_sha256": "1" * 64,
        }
        payload = {
            "kind": "inference",
            "generator_type": generator_type_for_architecture_mode("egs_tsaa"),
            "model_type": "res50",
            "target": -1,
            "eps_pixels": 10,
            "image_size": 224,
            "completed_epoch": 0,
            "architecture": {"egs_tk": 0.6},
            "conditioner_contract": egs_conditioner_contract(
                model_type="res50",
                image_size=224,
                topk_fraction=0.6,
                attack_model_contract=attack_contract,
            ),
            "generator_state_dict": {"dummy": torch.zeros(1)},
        }
        mutations = []

        changed_tk = copy.deepcopy(payload)
        changed_tk["architecture"]["egs_tk"] = 0.5
        mutations.append(("architecture_tk", changed_tk, "topk_fraction differs"))

        changed_model = copy.deepcopy(payload)
        changed_model["model_type"] = "incv3"
        changed_model["image_size"] = 299
        mutations.append(("model_type", changed_model, "model_type differs"))

        changed_conditioner_size = copy.deepcopy(payload)
        changed_conditioner_size["conditioner_contract"]["image_size"] = 299
        mutations.append(
            ("conditioner_image_size", changed_conditioner_size, "image_size differs")
        )

        changed_classifier = copy.deepcopy(payload)
        changed_classifier["conditioner_contract"]["attack_model_contract"][
            "model_type"
        ] = "incv3"
        mutations.append(
            (
                "classifier_model_type",
                changed_classifier,
                "attack-model contract metadata",
            )
        )

        with tempfile.TemporaryDirectory() as temporary:
            valid_path = pathlib.Path(temporary) / "valid.pth"
            torch.save((ddsc_train.INFERENCE_CHECKPOINT_FORMAT, payload), valid_path)
            loaded = ddsc_train.load_inference_checkpoint(valid_path)
            self.assertEqual(loaded["architecture"]["egs_tk"], 0.6)
            for name, changed, error in mutations:
                with self.subTest(field=name):
                    checkpoint = pathlib.Path(temporary) / f"{name}.pth"
                    torch.save(
                        (ddsc_train.INFERENCE_CHECKPOINT_FORMAT, changed),
                        checkpoint,
                    )
                    with self.assertRaisesRegex(ValueError, error):
                        ddsc_train.load_inference_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
