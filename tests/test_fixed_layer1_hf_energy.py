import pathlib
import sys
import unittest

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
GPG_DIR = ROOT / "third_party" / "GPG"
if str(GPG_DIR) not in sys.path:
    sys.path.insert(0, str(GPG_DIR))

from fixed_layer1_hf_energy import (  # noqa: E402
    HIGH_FREQUENCY_CHANGE_REWARD,
    OnlinePerImageLayer1HighFrequencyEnergy,
)


class Layer1HighFrequencyChangeEnergyTests(unittest.TestCase):
    def _objective(self) -> OnlinePerImageLayer1HighFrequencyEnergy:
        return OnlinePerImageLayer1HighFrequencyEnergy(
            source_model_sha256="0" * 64,
            dataset_sha256="1" * 64,
            dataset_size=1,
            channel_ratio=0.5,
            low_frequency_ratio=0.5,
            ridge_fraction=1.0e-3,
            reward_mode=HIGH_FREQUENCY_CHANGE_REWARD,
        )

    @staticmethod
    def _clean_feature() -> torch.Tensor:
        coordinates = torch.arange(8)
        checkerboard = ((coordinates[:, None] + coordinates[None, :]) % 2) * 2 - 1
        amplitudes = torch.tensor([4.0, 3.0, 2.0, 1.0]).view(1, 4, 1, 1)
        return amplitudes * checkerboard.to(torch.float32).view(1, 1, 8, 8)

    def test_change_reward_is_finite_when_features_are_identical(self) -> None:
        objective = self._objective()
        clean = self._clean_feature()
        indices = torch.tensor([0])
        objective.record_clean(clean, indices)

        adversarial = clean.clone().requires_grad_(True)
        reward = objective(
            adversarial,
            indices,
            clean_layer1_feature=clean,
        )

        self.assertAlmostEqual(
            float(reward.detach()),
            float(torch.log(torch.tensor(objective.denominator_eps))),
            places=5,
        )
        reward.backward()
        self.assertTrue(torch.isfinite(adversarial.grad).all())

    def test_change_reward_responds_to_hf_displacement_not_dc_offset(self) -> None:
        objective = self._objective()
        clean = self._clean_feature()
        indices = torch.tensor([0])
        objective.record_clean(clean, indices)

        high_frequency_change = clean.clone()
        high_frequency_change[:, 0] += self._clean_feature()[:, 0] * 0.5
        high_reward = objective(
            high_frequency_change,
            indices,
            clean_layer1_feature=clean,
        )

        dc_offset = clean + 0.5
        dc_reward = objective(
            dc_offset,
            indices,
            clean_layer1_feature=clean,
        )

        floor_reward = float(torch.log(torch.tensor(objective.denominator_eps)))
        self.assertGreater(float(high_reward), floor_reward)
        self.assertAlmostEqual(float(dc_reward), floor_reward, places=5)

    def test_change_reward_requires_matching_clean_feature(self) -> None:
        objective = self._objective()
        clean = self._clean_feature()
        indices = torch.tensor([0])
        objective.record_clean(clean, indices)

        with self.assertRaisesRegex(ValueError, "clean layer1 feature is required"):
            objective(clean, indices)


if __name__ == "__main__":
    unittest.main()
