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
        build_generator_from_inference_checkpoint,
        build_parser,
        controller_config_from_args,
        initial_controller_state,
        load_training_checkpoint,
        optimizer_spec_for_generator,
        runtime_contract,
        save_epoch_checkpoints,
        validate_args,
        validate_optimizer_state_dict,
        validate_resume_metadata,
    )
    from third_party.GPG.generators_ddsc_gpg import DDSCGPGGenerator
    from third_party.GPG.generators_legacy_gpg import (
        LEGACY_GENERATOR_TYPE,
        LegacyGPGGenerator,
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

    def test_cli_selects_generator_mode_and_rejects_ignored_decoder_knobs(
        self,
    ) -> None:
        parser = build_parser()
        self.assertEqual(parser.parse_args([]).generator_mode, "isolated")

        legacy_args = parser.parse_args(["--generator_mode", "legacy"])
        validate_args(legacy_args)
        legacy_args.decoder_width = ISOLATED_DECODER_DEFAULTS["decoder_width"] + 1
        with self.assertRaisesRegex(ValueError, "configure only the isolated"):
            validate_args(legacy_args)

    def test_both_modes_implement_the_ddsc_four_output_forward_contract(self) -> None:
        generators = (
            DDSCGPGGenerator(torchvision.models.resnet50(weights=None)),
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

    def test_inference_checkpoint_round_trip_supports_both_modes(self) -> None:
        generators = (
            DDSCGPGGenerator(torchvision.models.resnet50(weights=None)),
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

    def test_frozen_legacy_inference_checkpoint_remains_readable(self) -> None:
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

    def test_frozen_legacy_training_checkpoint_requires_restart(self) -> None:
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
                dataset_manifest={"sample_count": 1},
                runtime_manifest=runtime_contract(torch.device("cpu")),
            )

            training_payload = load_training_checkpoint(training_path)
            validate_resume_metadata(training_payload, args, controller_config)
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
