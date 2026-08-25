from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_PATH = (
    ROOT / "third_party" / "GPG" / "tools" / "bootstrap_transfer_eval.py"
)
SPEC = importlib.util.spec_from_file_location("bootstrap_transfer_eval", BOOTSTRAP_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - test setup contract
    raise RuntimeError(f"cannot load bootstrap module: {BOOTSTRAP_PATH}")
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


def _write_fake_distribution(
    root: Path,
    *,
    distribution: str,
    version: str,
) -> None:
    normalized = distribution.replace("-", "_")
    metadata_dir = root / f"{normalized}-{version}.dist-info"
    metadata_dir.mkdir()
    (metadata_dir / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        f"Name: {distribution}\n"
        f"Version: {version}\n",
        encoding="utf-8",
    )


class GPGTransferBootstrapTests(unittest.TestCase):
    def test_pip_commands_never_resolve_pytorch(self) -> None:
        target = Path("/tmp/eval-deps")
        wheelhouse = Path("/tmp/wheels")
        support, timm = bootstrap.pip_install_commands(
            "python",
            target,
            wheelhouse,
        )

        self.assertNotIn("--no-deps", support)
        self.assertEqual(
            tuple(support[-len(bootstrap.SUPPORT_REQUIREMENTS) :]),
            bootstrap.SUPPORT_REQUIREMENTS,
        )
        self.assertIn("--no-index", support)
        self.assertIn(str(wheelhouse), support)
        self.assertIn("--no-deps", timm)
        self.assertEqual(timm[-1], bootstrap.TIMM_REQUIREMENT)
        for command in (support, timm):
            self.assertIn("--ignore-installed", command)
            self.assertFalse(any(item.startswith("torch==") for item in command))
            self.assertFalse(
                any(item.startswith("torchvision==") for item in command)
            )

    def test_exact_dependency_probe_accepts_isolated_packages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "timm").mkdir()
            (root / "timm" / "__init__.py").write_text(
                "from . import layers\n",
                encoding="utf-8",
            )
            (root / "timm" / "layers.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "huggingface_hub").mkdir()
            (root / "huggingface_hub" / "__init__.py").write_text(
                "def hf_hub_download(*args, **kwargs): return None\n",
                encoding="utf-8",
            )
            (root / "safetensors").mkdir()
            (root / "safetensors" / "__init__.py").write_text("", encoding="utf-8")
            (root / "safetensors" / "torch.py").write_text(
                "VALUE = 1\n",
                encoding="utf-8",
            )
            (root / "yaml.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "mmpretrain").mkdir()
            (root / "mmpretrain" / "__init__.py").write_text(
                "def get_model(*args, **kwargs): return None\n",
                encoding="utf-8",
            )
            for package in ("mmcv", "mmengine"):
                (root / package).mkdir()
                (root / package / "__init__.py").write_text(
                    "VALUE = 1\n",
                    encoding="utf-8",
                )
            for distribution, version in bootstrap.EXPECTED_DISTRIBUTIONS.items():
                _write_fake_distribution(
                    root,
                    distribution=distribution,
                    version=version,
                )

            previous = os.environ.get("PYTHONPATH", "")
            pythonpath = str(root) + (os.pathsep + previous if previous else "")
            with mock.patch.dict(os.environ, {"PYTHONPATH": pythonpath}):
                ready, details = bootstrap.dependency_probe(sys.executable)

        self.assertTrue(ready, details)
        self.assertIn('"timm": "1.0.28"', details)

    def test_auto_install_disabled_fails_before_pip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "deps"
            with mock.patch.object(
                bootstrap,
                "dependency_probe",
                return_value=(False, "missing timm"),
            ), mock.patch.object(bootstrap.subprocess, "run") as run:
                with self.assertRaisesRegex(RuntimeError, "AUTO_INSTALL_EVAL_DEPS=0"):
                    bootstrap.ensure_dependencies(
                        python_executable=sys.executable,
                        target=target,
                        auto_install=False,
                        wheelhouse=None,
                    )
            run.assert_not_called()

    def test_existing_asset_requires_exact_hash(self) -> None:
        payload = b"verified-openmmlab-test-payload"
        expected = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "vit.pth"
            destination.write_bytes(payload)
            bootstrap.ensure_downloaded_file(
                destination=destination,
                url="https://invalid.example/unused",
                expected_sha256=expected,
                auto_download=False,
            )
            self.assertEqual(destination.read_bytes(), payload)

            destination.write_bytes(b"wrong")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                bootstrap.ensure_downloaded_file(
                    destination=destination,
                    url="https://invalid.example/unused",
                    expected_sha256=expected,
                    auto_download=True,
                )
            self.assertEqual(destination.read_bytes(), b"wrong")

    def test_mmpretrain_asset_catalog_has_full_matching_hashes(self) -> None:
        self.assertEqual(len(bootstrap.MMPRETRAIN_CHECKPOINTS), 5)
        for model_name, spec in bootstrap.MMPRETRAIN_CHECKPOINTS.items():
            self.assertTrue(model_name.startswith("mm_"))
            self.assertEqual(len(spec["sha256"]), 64)
            short_hash = Path(spec["filename"]).stem.rsplit("-", 1)[-1]
            self.assertTrue(spec["sha256"].startswith(short_hash))
            self.assertTrue(spec["url"].endswith(spec["filename"]))

    def test_existing_asset_symlink_is_rejected(self) -> None:
        payload = b"verified-openmmlab-test-payload"
        expected = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pth"
            destination = root / "vit.pth"
            source.write_bytes(payload)
            try:
                destination.symlink_to(source)
            except OSError as exc:  # Windows may not grant symlink privileges.
                self.skipTest(f"symlinks are unavailable: {exc}")

            with self.assertRaisesRegex(RuntimeError, "symbolic link"):
                bootstrap.ensure_downloaded_file(
                    destination=destination,
                    url="https://invalid.example/unused",
                    expected_sha256=expected,
                    auto_download=True,
                )
            self.assertEqual(source.read_bytes(), payload)

    def test_download_uses_temporary_file_and_verified_replace(self) -> None:
        payload = b"downloaded-openmmlab-test-payload"
        expected = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pth"
            destination = root / "nested" / "vit.pth"
            source.write_bytes(payload)
            bootstrap.ensure_downloaded_file(
                destination=destination,
                url=source.resolve().as_uri(),
                expected_sha256=expected,
                auto_download=True,
                attempts=1,
            )

            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(bootstrap.sha256_file(destination), expected)
            self.assertEqual(list(destination.parent.glob("*.part")), [])

    def test_eval_requirements_and_launcher_contract(self) -> None:
        requirements = (ROOT / "requirements-transfer-eval.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("timm==1.0.28", requirements)
        self.assertIn("huggingface-hub==1.24.0", requirements)
        self.assertIn("safetensors==0.8.0", requirements)
        self.assertIn("PyYAML==6.0.3", requirements)
        self.assertIn("mmpretrain==1.2.0", requirements)
        self.assertIn("mmengine==0.10.7", requirements)
        self.assertIn("mmcv-lite==2.2.0", requirements)
        package_lines = [
            line.strip()
            for line in requirements.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertFalse(any(line.lower().startswith("torch==") for line in package_lines))
        self.assertFalse(
            any(line.lower().startswith("torchvision==") for line in package_lines)
        )

        launcher = (ROOT / "scripts" / "gpg_ddsc_damping025.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('auto_install_eval_deps="${AUTO_INSTALL_EVAL_DEPS:-1}"', launcher)
        self.assertIn(
            'auto_download_eval_assets="${AUTO_DOWNLOAD_EVAL_ASSETS:-1}"',
            launcher,
        )
        self.assertIn('--deps-dir "${eval_deps_dir}"', launcher)
        self.assertIn('--vit-checkpoint "${openmmlab_vit_checkpoint}"', launcher)
        self.assertIn(
            '--mmpretrain-checkpoint-dir "${mmpretrain_checkpoint_dir}"',
            launcher,
        )
        self.assertIn("mm_deit_small_4xb256_in1k", launcher)
        self.assertIn("mm_vit_base_p16_32xb128_mae_in1k", launcher)
        self.assertIn('PYTHONPATH="${eval_pythonpath}"', launcher)
        self.assertLess(launcher.index("eval_bootstrap_args=("), launcher.index("trainer_command=("))


if __name__ == "__main__":
    unittest.main()
