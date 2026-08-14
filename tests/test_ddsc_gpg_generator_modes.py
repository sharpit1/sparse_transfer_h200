from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


try:
    import torch
    import torchvision

    from third_party.GPG.DDSC_GPG_train import (
        INFERENCE_CHECKPOINT_FORMAT,
        ISOLATED_DECODER_DEFAULTS,
        build_generator_from_inference_checkpoint,
        build_parser,
        optimizer_spec_for_generator,
        validate_args,
        validate_optimizer_state_dict,
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

    def test_legacy_natural_adam_step_satisfies_checkpoint_contract(self) -> None:
        generator = LegacyGPGGenerator()
        learning_rate = 2.25e-5
        optimizer = torch.optim.Adam(
            list(generator.trainable_parameters()),
            lr=learning_rate,
            betas=(0.5, 0.999),
        )
        image = torch.rand(1, 3, 32, 32)

        adv, adv_inf, _, adv_00 = generator(image, 10 / 255.0)
        (adv.mean() + adv_inf.square().mean() + adv_00.mean()).backward()
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


if __name__ == "__main__":
    unittest.main()
