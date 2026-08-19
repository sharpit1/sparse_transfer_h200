from __future__ import annotations

import csv
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "nips2017_smoke"


class NIPS2017SmokeTests(unittest.TestCase):
    def test_inventory_attribution_and_labels(self) -> None:
        with (DATA_ROOT / "images.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 16)
        self.assertEqual(rows[0]["ImageId"], "0c7ac4a8c9dfa802")
        self.assertEqual(rows[0]["TrueLabel"], "306")
        for row in rows:
            self.assertEqual(
                row["License"],
                "https://creativecommons.org/licenses/by/2.0/",
            )
            self.assertTrue(row["Author"])
            self.assertTrue(row["OriginalLandingURL"])
            self.assertTrue(
                (DATA_ROOT / "images" / f"{row['ImageId']}.png").is_file()
            )

    def test_dataset_uses_official_true_labels(self) -> None:
        try:
            import torch
            from torchvision import transforms

            from third_party.GPG.DDSC_GPG_train import (
                NIPS2017Dataset,
                dataset_contract,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional runtime dependency is not installed: {exc}")

        dataset = NIPS2017Dataset(
            DATA_ROOT / "images",
            DATA_ROOT / "images.csv",
            transforms.Compose([transforms.ToTensor()]),
        )

        image, label = dataset[0]
        self.assertEqual(len(dataset), 16)
        self.assertEqual(label, 305)
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(tuple(image.shape), (3, 299, 299))
        self.assertEqual(dataset_contract(dataset)["sample_count"], 16)

    def test_all_images_are_decodable_rgb_nips_crops(self) -> None:
        from PIL import Image

        image_paths = sorted((DATA_ROOT / "images").glob("*.png"))
        self.assertEqual(len(image_paths), 16)
        for image_path in image_paths:
            with Image.open(image_path) as image:
                image.load()
                self.assertEqual(image.format, "PNG", image_path.name)
                self.assertEqual(image.mode, "RGB", image_path.name)
                self.assertEqual(image.size, (299, 299), image_path.name)

    def test_launchers_fix_batch_size_and_damping_variants(self) -> None:
        common = (ROOT / "scripts" / "run_nips2017_smoke.sh").read_text(
            encoding="utf-8"
        )
        damping_025 = (
            ROOT / "scripts" / "run_nips2017_smoke_damping_025.sh"
        ).read_text(encoding="utf-8")
        damping_050 = (
            ROOT / "scripts" / "run_nips2017_smoke_damping_050.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("--batch_size 16", common)
        self.assertIn("--epochs 5", common)
        self.assertIn("--ddsc_warmup_epochs 2", common)
        self.assertIn("/app/output/nips2017_smoke", common)
        self.assertIn("0.25", damping_025)
        self.assertIn("0.5", damping_050)

    def test_imagenet_launchers_fix_full_training_protocol(self) -> None:
        common = (ROOT / "scripts" / "run_imagenet_train.sh").read_text(
            encoding="utf-8"
        )
        damping_025 = (
            ROOT / "scripts" / "run_imagenet_train_damping_025.sh"
        ).read_text(encoding="utf-8")
        damping_050 = (
            ROOT / "scripts" / "run_imagenet_train_damping_050.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("--batch_size 16", common)
        self.assertIn('train_epochs="${TRAIN_EPOCHS:-15}"', common)
        self.assertIn('max_batches_per_epoch="${MAX_BATCHES_PER_EPOCH:-0}"', common)
        self.assertIn('ddsc_ema_decay="${DDSC_EMA_DECAY:-0.0}"', common)
        self.assertIn('--ddsc_ema_decay "$ddsc_ema_decay"', common)
        self.assertIn(
            'layer1_dropout_mode="${LAYER1_DROPOUT_MODE:-off}"',
            common,
        )
        self.assertIn('layer1_dropout_p="${LAYER1_DROPOUT_P:-0.4}"', common)
        self.assertIn(
            'layer1_dropout_channel_ratio="${LAYER1_DROPOUT_CHANNEL_RATIO:-0.3}"',
            common,
        )
        self.assertIn(
            'layer1_dropout_hf_ratio="${LAYER1_DROPOUT_HF_RATIO:-0.35}"',
            common,
        )
        self.assertIn(
            'layer1_dropout_eot_samples="${LAYER1_DROPOUT_EOT_SAMPLES:-4}"',
            common,
        )
        self.assertIn(
            'layer1_dropout_eot_reduction="${LAYER1_DROPOUT_EOT_REDUCTION:-logits}"',
            common,
        )
        self.assertIn('--layer1_dropout_mode "$layer1_dropout_mode"', common)
        self.assertIn('--layer1_dropout_p "$layer1_dropout_p"', common)
        self.assertIn(
            '--layer1_dropout_channel_ratio "$layer1_dropout_channel_ratio"',
            common,
        )
        self.assertIn(
            '--layer1_dropout_hf_ratio "$layer1_dropout_hf_ratio"',
            common,
        )
        self.assertIn(
            '--layer1_dropout_eot_samples "$layer1_dropout_eot_samples"',
            common,
        )
        self.assertIn(
            '--layer1_dropout_eot_reduction "$layer1_dropout_eot_reduction"',
            common,
        )
        self.assertIn("--ddsc_warmup_epochs 2", common)
        self.assertIn("--num_workers 8", common)
        self.assertNotIn("--train_csv", common)
        self.assertIn("1000", common)
        self.assertIn("/app/data/ImageNet-2012/train", common)
        self.assertIn("/app/output/sharpit1", common)
        self.assertIn("0.25", damping_025)
        self.assertIn("0.5", damping_050)

    def test_imagenet_timing_smoke_processes_exactly_16_batches(self) -> None:
        common = (ROOT / "scripts" / "run_imagenet_train.sh").read_text(
            encoding="utf-8"
        )
        timing_smoke = (
            ROOT / "scripts" / "run_imagenet_smoke_16_batches_damping_025.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('--epochs "$train_epochs"', common)
        self.assertIn(
            '--max_batches_per_epoch "$max_batches_per_epoch"',
            common,
        )
        self.assertIn("TRAIN_EPOCHS=1", timing_smoke)
        self.assertIn("MAX_BATCHES_PER_EPOCH=16", timing_smoke)
        self.assertIn("imagenet_smoke_total_seconds=", timing_smoke)


if __name__ == "__main__":
    unittest.main()
