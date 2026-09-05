"""Run DDSC-GPG while reusing the current generator's training ``adv_00``.

The canonical trainer remains unchanged.  This fail-closed wrapper removes
only the second forward through the current generator; the frozen
previous-epoch generator forward is retained for temporal regularization.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


BASE_TRAINER = Path(__file__).resolve().with_name("DDSC_GPG_train.py")
BASE_SHA256 = "b2ae3134f2f99f3ce06222f62efcad8e0c1798f2307d80d6a2384c71d5eadddf"


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"DDSC-GPG training-reuse patch {label!r} expected one marker, "
            f"found {count}"
        )
    return source.replace(old, new, 1)


def build_training_reuse_source() -> str:
    raw = BASE_TRAINER.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != BASE_SHA256:
        raise RuntimeError(
            "DDSC_GPG_train.py hash mismatch: "
            f"expected {BASE_SHA256}, observed {digest}"
        )
    source = raw.decode("utf-8")

    source = replace_once(
        source,
        "                with torch.random.fork_rng(devices=fork_devices):\n"
        "                    with torch.set_grad_enabled(intersection_active):\n"
        "                        with generator_deployment_mask_mode(net_g):\n"
        "                            current_temporal_mask = forward_generator_training(\n"
        "                                net_g,\n"
        "                                args.architecture_mode,\n"
        "                                image,\n"
        "                                args.eps / 255.0,\n"
        "                                pgd_delta=grad_delta,\n"
        "                                structured_mask=structured_mask,\n"
        "                            )[3]\n",
        "                current_temporal_mask = adv_00\n",
        "reuse current-generator training output",
    )
    source = replace_once(
        source,
        "        f\"from_{lineage_fingerprint}\"\n"
        "    )\n",
        "        f\"from_{lineage_fingerprint}\"\n"
        "        \"_CM-training-adv00\"\n"
        "    )\n",
        "collision-free output suffix",
    )
    source = replace_once(
        source,
        "        \"comparison=current_and_frozen_previous_deployment_mode \"\n",
        "        \"comparison=training_adv_00_and_frozen_previous_deployment_mode \"\n"
        "        \"current_generator_extra_forward=False \"\n",
        "current-mask provenance",
    )

    return source


def main() -> None:
    source = build_training_reuse_source()
    execution_globals = {
        "__name__": "__main__",
        "__file__": str(BASE_TRAINER),
        "__package__": None,
    }
    exec(compile(source, str(BASE_TRAINER), "exec"), execution_globals)


if __name__ == "__main__":
    main()
