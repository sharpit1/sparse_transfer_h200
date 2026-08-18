from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path


try:
    import torch
    import torchvision

    from third_party.GPG.DDSC_GPG_train import (
        CHECKPOINT_FORMAT,
        INFERENCE_CHECKPOINT_FORMAT,
        ISOLATED_DECODER_DEFAULTS,
        PREVIOUS_TRAINING_CHECKPOINT_FORMAT,
        build_generator_from_inference_checkpoint,
        build_parser,
        controller_config_from_args,
        epoch_timing_metrics,
        expected_optimizer_step,
        initial_controller_state,
        load_training_checkpoint,
        optimizer_spec_for_generator,
        runtime_contract,
        save_epoch_checkpoints,
        validate_args,
        validate_optimizer_state_dict,
        validate_resume_metadata,
    )
    from third_party.GPG.generators_ddsc_gpg import (
        DDSCGPGGenerator,
        DDSCSplitGPGGenerator,
        SPLIT_GENERATOR_TYPE,
    )
    from third_party.GPG.generators_legacy_gpg import (
        LEGACY_GENERATOR_TYPE,
        LegacyGPGGenerator,
    )
    from third_party.GPG.generators_frozen_legacy_gpg import (
        FROZEN_LEGACY_GENERATOR_TYPE,
        FrozenResNetLegacyGPGGenerator,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - optional local runtime
    torch = None
    torchvision = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(IMPORT_ERROR is not None, f"optional dependency missing: {IMPORT_ERROR}")
class DDSCGPGGeneratorModeTests(unittest.TestCase):
    def test_legacy_generator_preserves_original_parameter_contract(self) -> None:
        generator = LegacyGPGGenerator()

        self.assertEqual(generator.generator_type, LEGACY_GENERATOR_TYPE)
        self.assertEqual(sum(p.numel() for p in generator.parameters()), 8_592_516)
        self.assertEqual(len(generator.state_dict()), 136)
        self.assertIn("block1.1.weight", generator.state_dict())
        self.assertIn("Grad_block1.1.weight", generator.state_dict())
        self.assertIn("upsampl_inf1.0.weight", generator.state_dict())
        self.assertIn("upsampl_01.0.weight", generator.state_dict())

    def test_frozen_legacy_changes_only_the_encoder_interface(self) -> None:
        backbone = torchvision.models.resnet50(weights=None)
        generator = FrozenResNetLegacyGPGGenerator(backbone)
        legacy = LegacyGPGGenerator()

        self.assertEqual(
            generator.generator_type,
            FROZEN_LEGACY_GENERATOR_TYPE,
        )
        self.assertEqual(
            sum(p.numel() for p in generator.parameters()),
            8_059_972,
        )
        self.assertEqual(
            sum(p.numel() for p in generator.parameters() if p.requires_grad),
            7_834_628,
        )
        self.assertTrue(
            all(not p.requires_grad for p in generator.encoder.parameters())
        )
        self.assertFalse(
            any("adapter" in name for name, _ in generator.named_modules())
        )
        self.assertFalse(
            any(
                name.startswith("block1")
                for name, _ in generator.named_parameters()
            )
        )
        self.assertFalse(
            any(
                name.startswith("Grad_block")
                for name, _ in generator.named_parameters()
            )
        )

        legacy_back_end_prefixes = ("resblock", "upsampl", "blockf")
        frozen_back_end = {
            name: tuple(tensor.shape)
            for name, tensor in generator.state_dict().items()
            if name.startswith(legacy_back_end_prefixes)
        }
        legacy_back_end = {
            name: tuple(tensor.shape)
            for name, tensor in legacy.state_dict().items()
            if name.startswith(legacy_back_end_prefixes)
        }
        self.assertEqual(frozen_back_end, legacy_back_end)
        self.assertEqual(
            generator.architecture_metadata()["encoder"]["adapter"],
            "none",
        )
        self.assertEqual(
            generator.architecture_metadata()["decoder"]["residual_blocks"],
            6,
        )

    def test_cli_selects_generator_mode_and_rejects_ignored_decoder_knobs(
        self,
    ) -> None:
        parser = build_parser()
        self.assertEqual(parser.parse_args([]).generator_mode, "isolated")
        self.assertFalse(parser.parse_args([]).adapter_feature_guidance)
        self.assertEqual(parser.parse_args([]).max_batches_per_epoch, 0)

        split_args = parser.parse_args(
            [
                "--generator_mode",
                "isolated_split",
                "--adapter_feature_guidance",
            ]
        )
        validate_args(split_args)

        legacy_args = parser.parse_args(["--generator_mode", "legacy"])
        validate_args(legacy_args)
        legacy_args.decoder_width = ISOLATED_DECODER_DEFAULTS["decoder_width"] + 1
        with self.assertRaisesRegex(ValueError, "configure only the isolated"):
            validate_args(legacy_args)

        legacy_guidance_args = parser.parse_args(
            ["--generator_mode", "legacy", "--adapter_feature_guidance"]
        )
        with self.assertRaisesRegex(ValueError, "requires isolated"):
            validate_args(legacy_guidance_args)

        frozen_legacy_args = parser.parse_args(
            ["--generator_mode", "frozen_legacy"]
        )
        validate_args(frozen_legacy_args)
        frozen_legacy_args.decoder_num_blocks += 1
        with self.assertRaisesRegex(ValueError, "configure only the isolated"):
            validate_args(frozen_legacy_args)

        invalid_batch_limit_args = parser.parse_args(
            ["--max_batches_per_epoch", "-1"]
        )
        with self.assertRaisesRegex(ValueError, "max_batches_per_epoch"):
            validate_args(invalid_batch_limit_args)

    def test_batch_limit_controls_optimizer_steps_and_timing_metrics(self) -> None:
        dataset_manifest = {"sample_count": 1_281_167}

        self.assertEqual(
            expected_optimizer_step(
                next_epoch=15,
                dataset_manifest=dataset_manifest,
                batch_size=16,
                max_batches_per_epoch=16,
            ),
            240,
        )
        self.assertEqual(
            expected_optimizer_step(
                next_epoch=1,
                dataset_manifest=dataset_manifest,
                batch_size=16,
                max_batches_per_epoch=0,
            ),
            80_073,
        )
        self.assertEqual(
            epoch_timing_metrics(
                elapsed_seconds=8.0,
                processed_batches=16,
                processed_samples=256,
            ),
            {
                "seconds": 8.0,
                "batches_per_second": 2.0,
                "images_per_second": 32.0,
            },
        )

    def test_all_modes_implement_the_ddsc_four_output_forward_contract(self) -> None:
        generators = (
            DDSCGPGGenerator(torchvision.models.resnet50(weights=None)),
            DDSCSplitGPGGenerator(
                torchvision.models.resnet50(weights=None)
            ),
            FrozenResNetLegacyGPGGenerator(
                torchvision.models.resnet50(weights=None)
            ),
            LegacyGPGGenerator(),
        )
        image = torch.rand(1, 3, 32, 32)
        for generator in generators:
            with self.subTest(generator_type=generator.generator_type):
                generator.train()
                adv, adv_inf, adv_0, adv_00 = generator(image, 10 / 255.0)
                self.assertEqual(tuple(adv.shape), (1, 3, 32, 32))
                self.assertEqual(tuple(adv_inf.shape), (1, 3, 32, 32))
                self.assertEqual(tuple(adv_0.shape), (1, 1, 32, 32))
                self.assertEqual(tuple(adv_00.shape), (1, 1, 32, 32))

    def test_isolated_split_duplicates_only_the_upsampling_trunk(self) -> None:
        generator = DDSCSplitGPGGenerator(
            torchvision.models.resnet50(weights=None)
        )

        self.assertEqual(generator.generator_type, SPLIT_GENERATOR_TYPE)
        self.assertEqual(
            sum(p.numel() for p in generator.parameters() if p.requires_grad),
            278_148,
        )
        self.assertFalse(
            generator.decoder.architecture_metadata()["shared_upsample_trunk"]
        )
        perturbation_parameter = next(
            generator.decoder.perturbation_upsample1.parameters()
        )
        mask_parameter = next(generator.decoder.mask_upsample1.parameters())
        self.assertNotEqual(
            perturbation_parameter.untyped_storage().data_ptr(),
            mask_parameter.untyped_storage().data_ptr(),
        )
        self.assertIsNot(
            generator.decoder.perturbation_upsample1,
            generator.decoder.mask_upsample1,
        )

    def test_adapter_feature_guidance_trains_both_adapters(self) -> None:
        for generator_class, expected_parameters in (
            (DDSCGPGGenerator, 218_820),
            (DDSCSplitGPGGenerator, 311_172),
        ):
            generator = generator_class(
                torchvision.models.resnet50(weights=None),
                adapter_feature_guidance=True,
            )
            image = torch.rand(1, 3, 32, 32)
            pgd_image = torch.rand_like(image)

            with self.subTest(generator_type=generator.generator_type):
                with self.assertRaisesRegex(ValueError, "requires grad_AE"):
                    generator(image, 10 / 255.0)
                outputs = generator(image, 10 / 255.0, pgd_image)
                self.assertEqual(len(outputs), 5)
                feature_guidance = outputs[-1]
                feature_guidance.backward()
                self.assertEqual(
                    sum(
                        p.numel()
                        for p in generator.parameters()
                        if p.requires_grad
                    ),
                    expected_parameters,
                )
                self.assertTrue(
                    all(
                        parameter.grad is not None
                        for parameter in generator.decoder.adapter.parameters()
                    )
                )
                self.assertIsNotNone(generator.pgd_adapter)
                self.assertTrue(
                    all(
                        parameter.grad is not None
                        for parameter in generator.pgd_adapter.parameters()
                    )
                )
                self.assertTrue(
                    all(
                        parameter.grad is None
                        for parameter in generator.encoder.parameters()
                    )
                )
                self.assertEqual(
                    generator.architecture_metadata()[
                        "adapter_feature_guidance"
                    ]["location"],
                    "post_adapter_pre_residual_body",
                )

    def test_isolated_split_routes_branch_gradients_independently(self) -> None:
        generator = DDSCSplitGPGGenerator(
            torchvision.models.resnet50(weights=None)
        )
        generator.train()
        image = torch.rand(1, 3, 32, 32)

        _, adv_inf, _, _ = generator(image, 10 / 255.0)
        adv_inf.mean().backward()
        self.assertTrue(
            all(
                parameter.grad is not None
                for parameter in generator.decoder.perturbation_upsample1.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in generator.decoder.mask_upsample1.parameters()
            )
        )

        generator.zero_grad(set_to_none=True)
        _, _, _, adv_00 = generator(image, 10 / 255.0)
        adv_00.mean().backward()
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in generator.decoder.perturbation_upsample1.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.grad is not None
                for parameter in generator.decoder.mask_upsample1.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.grad is not None
                for parameter in generator.decoder.adapter.parameters()
            )
        )

    def test_split_adapter_guidance_training_checkpoint_round_trip(self) -> None:
        generator = DDSCSplitGPGGenerator(
            torchvision.models.resnet50(weights=None),
            adapter_feature_guidance=True,
        )
        learning_rate = 2.25e-5
        optimizer = torch.optim.Adam(
            list(generator.trainable_parameters()),
            lr=learning_rate,
            betas=(0.5, 0.999),
        )
        image = torch.rand(1, 3, 32, 32)
        pgd_image = torch.rand_like(image)
        adv, adv_inf, _, adv_00, feature_guidance = generator(
            image,
            10 / 255.0,
            pgd_image,
        )
        (
            adv.mean()
            + adv_inf.square().mean()
            + adv_00.mean()
            + feature_guidance
        ).backward()
        optimizer.step()

        args = build_parser().parse_args(
            [
                "--generator_mode",
                "isolated_split",
                "--adapter_feature_guidance",
                "--device",
                "cpu",
                "--epochs",
                "2",
                "--max_batches_per_epoch",
                "1",
            ]
        )
        controller_config = controller_config_from_args(args)
        attack_model_manifest = {
            "schema": 1,
            "model_type": "res50",
            "architecture": "resnet50",
            "weights_enum": "IMAGENET1K_V1",
            "state_sha256": "0" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            inference_path, training_path = save_epoch_checkpoints(
                output_path=Path(directory),
                epoch=0,
                net_g=generator,
                optimizer=optimizer,
                controller_config=controller_config,
                controller_state=initial_controller_state(controller_config),
                args=args,
                data_loader_generator=torch.Generator().manual_seed(0),
                attack_model_manifest=attack_model_manifest,
                dataset_manifest={"sample_count": 17},
                runtime_manifest=runtime_contract(torch.device("cpu")),
            )

            training_payload = load_training_checkpoint(training_path)
            validate_resume_metadata(training_payload, args, controller_config)
            restored, inference_payload = (
                build_generator_from_inference_checkpoint(inference_path)
            )

        self.assertEqual(restored.generator_type, SPLIT_GENERATOR_TYPE)
        self.assertTrue(restored.adapter_feature_guidance)
        with torch.no_grad():
            self.assertEqual(
                len(restored(torch.rand(1, 3, 32, 32), 10 / 255.0)),
                4,
            )
        self.assertEqual(
            inference_payload["architecture"],
            generator.architecture_metadata(),
        )

    def test_inference_checkpoint_round_trip_supports_all_modes(self) -> None:
        generators = (
            DDSCGPGGenerator(torchvision.models.resnet50(weights=None)),
            DDSCGPGGenerator(
                torchvision.models.resnet50(weights=None),
                adapter_feature_guidance=True,
            ),
            DDSCSplitGPGGenerator(
                torchvision.models.resnet50(weights=None)
            ),
            DDSCSplitGPGGenerator(
                torchvision.models.resnet50(weights=None),
                adapter_feature_guidance=True,
            ),
            FrozenResNetLegacyGPGGenerator(
                torchvision.models.resnet50(weights=None)
            ),
            LegacyGPGGenerator(eps=10 / 255.0),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for generator in generators:
                with self.subTest(generator_type=generator.generator_type):
                    payload = {
                        "kind": "inference",
                        "generator_type": generator.generator_type,
                        "model_type": "res50",
                        "target": -1,
                        "eps_pixels": 10,
                        "image_size": 224,
                        "completed_epoch": 0,
                        "architecture": generator.architecture_metadata(),
                        "generator_state_dict": generator.state_dict(),
                    }
                    path = root / f"{generator.generator_type}.pth"
                    torch.save((INFERENCE_CHECKPOINT_FORMAT, payload), path)

                    restored, loaded = build_generator_from_inference_checkpoint(path)

                    self.assertEqual(
                        restored.generator_type,
                        generator.generator_type,
                    )
                    self.assertEqual(
                        list(restored.state_dict()),
                        list(generator.state_dict()),
                    )
                    self.assertEqual(
                        restored.architecture_metadata(),
                        loaded["architecture"],
                    )

    def test_historical_frozen_gradient_legacy_inference_checkpoint_remains_readable(
        self,
    ) -> None:
        generator = LegacyGPGGenerator(eps=10 / 255.0)
        architecture = copy.deepcopy(generator.architecture_metadata())
        architecture["encoder"]["gradient_branch"] = {
            "present": True,
            "frozen": True,
            "used_by_ddsc": False,
        }
        architecture["legacy_feature_guidance"] = (
            "preserved_frozen_unused_by_ddsc"
        )
        payload = {
            "kind": "inference",
            "generator_type": generator.generator_type,
            "model_type": "res50",
            "target": -1,
            "eps_pixels": 10,
            "image_size": 224,
            "completed_epoch": 0,
            "architecture": architecture,
            "generator_state_dict": generator.state_dict(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen_legacy.inference.pth"
            torch.save((INFERENCE_CHECKPOINT_FORMAT, payload), path)

            restored, loaded = build_generator_from_inference_checkpoint(path)

            corrupted_payload = copy.deepcopy(payload)
            corrupted_payload["architecture"]["decoder"]["base_channels"] += 1
            corrupted_path = Path(directory) / "corrupted_legacy.inference.pth"
            torch.save(
                (INFERENCE_CHECKPOINT_FORMAT, corrupted_payload),
                corrupted_path,
            )
            with self.assertRaisesRegex(
                ValueError,
                "reconstructed inference architecture",
            ):
                build_generator_from_inference_checkpoint(corrupted_path)

        self.assertEqual(restored.generator_type, generator.generator_type)
        self.assertEqual(list(restored.state_dict()), list(generator.state_dict()))
        self.assertEqual(loaded["architecture"], architecture)

    def test_historical_frozen_gradient_legacy_training_checkpoint_requires_restart(
        self,
    ) -> None:
        generator = LegacyGPGGenerator()
        architecture = copy.deepcopy(generator.architecture_metadata())
        architecture["encoder"]["gradient_branch"] = {
            "present": True,
            "frozen": True,
            "used_by_ddsc": False,
        }
        architecture["legacy_feature_guidance"] = (
            "preserved_frozen_unused_by_ddsc"
        )
        train_args = vars(
            build_parser().parse_args(["--generator_mode", "legacy"])
        )
        payload = {
            "kind": "training",
            "checkpoint_boundary": "post_epoch_post_controller_update",
            "generator_type": generator.generator_type,
            "generator_state_dict": generator.state_dict(),
            "attack_model_contract": {},
            "optimizer_state_dict": {},
            "optimizer_spec": {},
            "controller_config": {},
            "controller_state": {},
            "completed_epoch": 0,
            "next_epoch": 1,
            "train_args": train_args,
            "architecture": architecture,
            "rng_state": {},
            "dataset_contract": {},
            "runtime_contract": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen_legacy.train.pth"
            torch.save((CHECKPOINT_FORMAT, payload), path)

            with self.assertRaisesRegex(ValueError, "restart legacy training"):
                load_training_checkpoint(path)

    def test_legacy_feature_guidance_trains_and_round_trips_checkpoints(self) -> None:
        generator = LegacyGPGGenerator()
        learning_rate = 2.25e-5
        optimizer = torch.optim.Adam(
            list(generator.trainable_parameters()),
            lr=learning_rate,
            betas=(0.5, 0.999),
        )
        image = torch.rand(1, 3, 32, 32)
        gradient_adversarial_image = torch.rand_like(image)

        adv, adv_inf, _, adv_00, feature_guidance = generator(
            image,
            10 / 255.0,
            gradient_adversarial_image,
        )
        (
            adv.mean()
            + adv_inf.square().mean()
            + adv_00.mean()
            + feature_guidance
        ).backward()
        gradient_encoder_parameters = [
            parameter
            for name, parameter in generator.named_parameters()
            if name.startswith("Grad_block")
        ]
        self.assertTrue(gradient_encoder_parameters)
        self.assertTrue(
            all(parameter.requires_grad for parameter in gradient_encoder_parameters)
        )
        self.assertTrue(
            all(parameter.grad is not None for parameter in gradient_encoder_parameters)
        )
        optimizer.step()

        expected_parameters = [
            (name, parameter)
            for name, parameter in generator.named_parameters()
            if parameter.requires_grad
        ]
        validate_optimizer_state_dict(
            optimizer.state_dict(),
            optimizer_spec_for_generator(
                generator,
                expected_lr=learning_rate,
            ),
            expected_lr=learning_rate,
            expected_step=1,
            expected_parameters=expected_parameters,
        )

        args = build_parser().parse_args(
            [
                "--generator_mode",
                "legacy",
                "--device",
                "cpu",
                "--epochs",
                "2",
                "--max_batches_per_epoch",
                "1",
            ]
        )
        controller_config = controller_config_from_args(args)
        attack_model_manifest = {
            "schema": 1,
            "model_type": "res50",
            "architecture": "resnet50",
            "weights_enum": "IMAGENET1K_V1",
            "state_sha256": "0" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            inference_path, training_path = save_epoch_checkpoints(
                output_path=Path(directory),
                epoch=0,
                net_g=generator,
                optimizer=optimizer,
                controller_config=controller_config,
                controller_state=initial_controller_state(controller_config),
                args=args,
                data_loader_generator=torch.Generator().manual_seed(0),
                attack_model_manifest=attack_model_manifest,
                dataset_manifest={"sample_count": 17},
                runtime_manifest=runtime_contract(torch.device("cpu")),
            )

            training_payload = load_training_checkpoint(training_path)
            validate_resume_metadata(training_payload, args, controller_config)
            checkpoint_format, previous_payload = torch.load(
                training_path,
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(checkpoint_format, CHECKPOINT_FORMAT)
            previous_payload = copy.deepcopy(previous_payload)
            previous_payload["train_args"].pop("adapter_feature_guidance")
            previous_path = Path(directory) / "previous_v8.train.pth"
            torch.save(
                (PREVIOUS_TRAINING_CHECKPOINT_FORMAT, previous_payload),
                previous_path,
            )
            normalized_previous = load_training_checkpoint(previous_path)
            self.assertFalse(
                normalized_previous["train_args"]["adapter_feature_guidance"]
            )
            restored, inference_payload = (
                build_generator_from_inference_checkpoint(inference_path)
            )

        self.assertEqual(restored.generator_type, generator.generator_type)
        self.assertEqual(
            inference_payload["architecture"],
            generator.architecture_metadata(),
        )


if __name__ == "__main__":
    unittest.main()
