"""Bootstrap isolated transfer-evaluation dependencies and the ViT asset.

The H200 image already supplies its CUDA-matched torch and torchvision build.
This helper never installs those packages.  Evaluation-only dependencies are
placed in a repository-local ``--target`` directory, and timm is installed
with ``--no-deps`` so pip cannot traverse into the PyTorch dependency graph.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Iterator, Sequence


TIMM_REQUIREMENT = "timm==1.0.28"
SUPPORT_REQUIREMENTS = (
    "huggingface-hub==1.24.0",
    "safetensors==0.8.0",
    "PyYAML==6.0.3",
)
EXPECTED_DISTRIBUTIONS = {
    "timm": "1.0.28",
    "huggingface-hub": "1.24.0",
    "safetensors": "0.8.0",
    "PyYAML": "6.0.3",
}
FORBIDDEN_TARGET_PREFIXES = ("torch", "torchvision")
OPENMMLAB_MAE_VIT_URL = (
    "https://download.openmmlab.com/mmclassification/v0/vit/"
    "vit-base-p16_pt-32xb128-mae_in1k_20220623-4c544545.pth"
)
OPENMMLAB_MAE_VIT_SHA256 = (
    "4c544545d50657b87c62ca2de1a5da8d5f12abfc80e386bfb9a37dfbdf5b3e08"
)
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
DOWNLOAD_PROGRESS_BYTES = 64 * 1024 * 1024


DEPENDENCY_PROBE = r"""
import importlib.metadata
import json

expected = {
    "timm": "1.0.28",
    "huggingface-hub": "1.24.0",
    "safetensors": "0.8.0",
    "PyYAML": "6.0.3",
}
actual = {name: importlib.metadata.version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"evaluation dependency versions differ: {actual!r}")

import timm
from timm import layers as timm_layers
from huggingface_hub import hf_hub_download
import safetensors.torch
import yaml

if not callable(hf_hub_download):
    raise SystemExit("huggingface_hub.hf_hub_download is not callable")
if timm_layers is None or safetensors.torch is None or yaml is None:
    raise SystemExit("evaluation dependency API probe failed")
print(json.dumps(actual, sort_keys=True))
"""

CORE_RUNTIME_PROBE = r"""
import json
from pathlib import Path

import torch
import torchvision

print(json.dumps({
    "python": {
        "executable": str(Path(__import__("sys").executable).resolve()),
        "version": __import__("platform").python_version(),
    },
    "torch": {
        "version": torch.__version__,
        "module": str(Path(torch.__file__).resolve()),
    },
    "torchvision": {
        "version": torchvision.__version__,
        "module": str(Path(torchvision.__file__).resolve()),
    },
}, sort_keys=True))
"""


def _run_probe(
    python_executable: str,
    source: str,
) -> tuple[bool, str]:
    completed = subprocess.run(
        [python_executable, "-c", source],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    )
    return completed.returncode == 0, completed.stdout.strip()


def dependency_probe(python_executable: str) -> tuple[bool, str]:
    """Return whether exact dependency versions and required APIs import."""

    return _run_probe(python_executable, DEPENDENCY_PROBE)


def core_runtime_snapshot(python_executable: str) -> dict[str, object]:
    """Capture the CUDA framework identity that pip must not change."""

    ok, output = _run_probe(python_executable, CORE_RUNTIME_PROBE)
    if not ok:
        raise RuntimeError(f"cannot inspect torch/torchvision runtime: {output}")
    try:
        snapshot = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"runtime probe returned invalid JSON: {output}") from exc
    if not isinstance(snapshot, dict):
        raise RuntimeError("runtime probe did not return an object")
    return snapshot


def pip_install_commands(
    python_executable: str,
    target: Path,
    wheelhouse: Path | None,
) -> tuple[list[str], list[str]]:
    """Build commands that cannot resolve or install torch/torchvision."""

    common = [
        python_executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--quiet",
        "--progress-bar",
        "off",
        "--ignore-installed",
        "--only-binary=:all:",
        "--upgrade",
        "--target",
        str(target),
    ]
    if wheelhouse is not None:
        common.extend(("--no-index", "--find-links", str(wheelhouse)))
    support = [*common, *SUPPORT_REQUIREMENTS]
    timm = [*common, "--no-deps", TIMM_REQUIREMENT]
    return support, timm


def _assert_target_has_no_pytorch(target: Path) -> None:
    if not target.exists():
        return
    offenders = sorted(
        path.name
        for path in target.iterdir()
        if path.name.lower().startswith(FORBIDDEN_TARGET_PREFIXES)
    )
    if offenders:
        raise RuntimeError(
            "evaluation dependency target contains forbidden PyTorch entries: "
            + ", ".join(offenders)
        )


@contextlib.contextmanager
def advisory_lock(path: Path) -> Iterator[None]:
    """Serialize bootstrap work on POSIX; use a no-op lock elsewhere."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            import fcntl  # POSIX-only, including the H200 Linux runtime.
        except ImportError:  # pragma: no cover - Windows test/development host
            yield
            return
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def ensure_dependencies(
    *,
    python_executable: str,
    target: Path,
    auto_install: bool,
    wheelhouse: Path | None,
) -> None:
    """Ensure exact evaluator dependencies without changing the H200 runtime."""

    if sys.version_info < (3, 10):
        raise RuntimeError(
            "transfer evaluation requires Python 3.10 or newer; "
            f"current={platform.python_version()}"
        )
    target = target.resolve()
    lock_path = target.parent / f".{target.name}.bootstrap.lock"
    with advisory_lock(lock_path):
        _assert_target_has_no_pytorch(target)
        ready, details = dependency_probe(python_executable)
        if ready:
            print(f"eval_dependencies=ready {details}", flush=True)
            return
        if not auto_install:
            raise RuntimeError(
                "evaluation dependencies are missing or incompatible and "
                "AUTO_INSTALL_EVAL_DEPS=0; probe output: "
                + details
            )
        if wheelhouse is not None and not wheelhouse.is_dir():
            raise RuntimeError(f"EVAL_DEPS_WHEELHOUSE is not a directory: {wheelhouse}")

        target.mkdir(parents=True, exist_ok=True)
        runtime_before = core_runtime_snapshot(python_executable)
        support_command, timm_command = pip_install_commands(
            python_executable,
            target,
            wheelhouse,
        )
        print(f"eval_dependencies=installing target={target}", flush=True)
        try:
            subprocess.run(support_command, check=True, env=os.environ.copy())
            subprocess.run(timm_command, check=True, env=os.environ.copy())
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "evaluation dependency installation failed; provide network "
                "access or EVAL_DEPS_WHEELHOUSE"
            ) from exc

        _assert_target_has_no_pytorch(target)
        runtime_after = core_runtime_snapshot(python_executable)
        if runtime_after != runtime_before:
            raise RuntimeError(
                "torch/torchvision runtime changed during evaluator bootstrap: "
                f"before={runtime_before!r}, after={runtime_after!r}"
            )
        ready, details = dependency_probe(python_executable)
        if not ready:
            raise RuntimeError(
                "evaluation dependencies still fail validation after installation: "
                + details
            )
        print(f"eval_dependencies=installed {details}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_once(url: str, destination: Path) -> tuple[str, int]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "sparse-transfer-h200-bootstrap/1"},
    )
    digest = hashlib.sha256()
    received = 0
    next_progress = DOWNLOAD_PROGRESS_BYTES
    with urllib.request.urlopen(request, timeout=60) as response:
        with destination.open("xb") as output:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                if received >= next_progress:
                    print(
                        f"openmmlab_vit_downloaded_bytes={received}",
                        flush=True,
                    )
                    next_progress += DOWNLOAD_PROGRESS_BYTES
            output.flush()
            os.fsync(output.fileno())
    return digest.hexdigest(), received


def ensure_downloaded_file(
    *,
    destination: Path,
    url: str,
    expected_sha256: str,
    auto_download: bool,
    attempts: int = 3,
) -> None:
    """Verify or atomically download one immutable external artifact."""

    destination = destination.expanduser()
    if destination.is_symlink():
        raise RuntimeError(
            "OpenMMLab ViT checkpoint path must not be a symbolic link: "
            f"{destination}"
        )
    destination = destination.absolute()
    if destination.is_file():
        actual = sha256_file(destination)
        if actual == expected_sha256:
            print(f"openmmlab_vit=ready path={destination}", flush=True)
            return
        raise RuntimeError(
            "existing OpenMMLab ViT checkpoint hash mismatch; refusing to "
            f"overwrite it: expected={expected_sha256}, actual={actual}, "
            f"path={destination}"
        )
    elif destination.exists():
        raise RuntimeError(
            "OpenMMLab ViT checkpoint path exists but is not a regular file: "
            f"{destination}"
        )
    if not auto_download:
        raise RuntimeError(
            "OpenMMLab ViT checkpoint is missing and "
            "AUTO_DOWNLOAD_EVAL_ASSETS=0: "
            f"{destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_key = hashlib.sha256(str(destination).encode("utf-8")).hexdigest()[:20]
    lock_path = Path(tempfile.gettempdir()) / f"ddsc-transfer-{lock_key}.lock"
    with advisory_lock(lock_path):
        if destination.is_symlink() or destination.exists():
            if destination.is_symlink() or not destination.is_file():
                raise RuntimeError(
                    "OpenMMLab ViT checkpoint path became unsafe while waiting "
                    f"for the download lock: {destination}"
                )
            actual = sha256_file(destination)
            if actual == expected_sha256:
                print(f"openmmlab_vit=ready path={destination}", flush=True)
                return
            raise RuntimeError(
                "existing OpenMMLab ViT checkpoint hash mismatch; refusing to "
                f"overwrite it: expected={expected_sha256}, actual={actual}, "
                f"path={destination}"
            )
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            temporary = destination.with_name(
                f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.part"
            )
            try:
                print(
                    f"openmmlab_vit=downloading attempt={attempt}/{attempts} "
                    f"path={destination}",
                    flush=True,
                )
                actual, received = _download_once(url, temporary)
                if actual != expected_sha256:
                    raise RuntimeError(
                        "downloaded OpenMMLab ViT hash mismatch: "
                        f"expected={expected_sha256}, actual={actual}, "
                        f"bytes={received}"
                    )
                if destination.is_symlink() or destination.exists():
                    if destination.is_symlink() or not destination.is_file():
                        raise RuntimeError(
                            "OpenMMLab ViT checkpoint path became unsafe during "
                            f"download: {destination}"
                        )
                    installed_hash = sha256_file(destination)
                    if installed_hash != expected_sha256:
                        raise RuntimeError(
                            "another process created a mismatched OpenMMLab ViT "
                            f"checkpoint; refusing to overwrite it: {destination}"
                        )
                    temporary.unlink()
                else:
                    os.replace(temporary, destination)
                print(
                    f"openmmlab_vit=downloaded bytes={received} "
                    f"sha256={actual} path={destination}",
                    flush=True,
                )
                return
            except Exception as exc:
                last_error = exc
                temporary.unlink(missing_ok=True)
                if attempt < attempts:
                    time.sleep(float(2 ** (attempt - 1)))
        raise RuntimeError(
            f"failed to download the OpenMMLab ViT checkpoint after {attempts} "
            f"attempts: {last_error}"
        ) from last_error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap DDSC-GPG transfer-evaluation dependencies and assets"
    )
    parser.add_argument("--deps-dir", required=True, type=Path)
    parser.add_argument(
        "--auto-install-deps",
        choices=("0", "1"),
        default="1",
    )
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--require-vit", action="store_true")
    parser.add_argument("--vit-checkpoint", type=Path)
    parser.add_argument(
        "--auto-download-assets",
        choices=("0", "1"),
        default="1",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        ensure_dependencies(
            python_executable=sys.executable,
            target=args.deps_dir,
            auto_install=args.auto_install_deps == "1",
            wheelhouse=args.wheelhouse,
        )
        if args.require_vit:
            if args.vit_checkpoint is None:
                raise RuntimeError("--require-vit requires --vit-checkpoint")
            ensure_downloaded_file(
                destination=args.vit_checkpoint,
                url=OPENMMLAB_MAE_VIT_URL,
                expected_sha256=OPENMMLAB_MAE_VIT_SHA256,
                auto_download=args.auto_download_assets == "1",
            )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: transfer evaluation bootstrap failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_DISTRIBUTIONS",
    "OPENMMLAB_MAE_VIT_SHA256",
    "OPENMMLAB_MAE_VIT_URL",
    "SUPPORT_REQUIREMENTS",
    "TIMM_REQUIREMENT",
    "core_runtime_snapshot",
    "dependency_probe",
    "ensure_dependencies",
    "ensure_downloaded_file",
    "main",
    "pip_install_commands",
    "sha256_file",
]
