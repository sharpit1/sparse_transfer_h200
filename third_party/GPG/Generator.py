"""Evaluation-compatible entry point for the dual-mode GPG generator.

The implementation lives in ``generators_modify.py`` so training and
evaluation cannot drift.  The wrapper preserves the historical evaluation
default (hard masks) while retaining the exact module/state-dict names created
by the shared base class.
"""

from generators_modify import (
    GENERATOR_ENCODER_MODES,
    FrozenResNet50Layer1,
    GeneratorResnet as _TrainingGeneratorResnet,
    ResidualBlock,
)


class GeneratorResnet(_TrainingGeneratorResnet):
    def __init__(
        self,
        inception=False,
        eps=1.0,
        evaluate=True,
        encoder_mode="legacy",
        encoder_backbone=None,
    ):
        super(GeneratorResnet, self).__init__(
            inception=inception,
            eps=eps,
            evaluate=evaluate,
            encoder_mode=encoder_mode,
            encoder_backbone=encoder_backbone,
        )


__all__ = [
    "GENERATOR_ENCODER_MODES",
    "FrozenResNet50Layer1",
    "GeneratorResnet",
    "ResidualBlock",
]
