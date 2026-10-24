"""Comprehensive tests for ``zero_keras.legacy`` and ``zero_keras.optimizers.legacy``."""

from __future__ import annotations

from zero_keras import legacy, optimizers
from zero_keras.optimizers import legacy as opt_legacy


def test_legacy() -> None:
    """Verify legacy saving and legacy optimizer aliases."""
    assert legacy.saving.deserialize_keras_object("test") == "test"

    class Dummy:
        """Dummy class for legacy serialization testing."""

    assert legacy.saving.serialize_keras_object(Dummy()) == {"class_name": "Dummy"}

    assert opt_legacy.SGD is optimizers.SGD
    assert opt_legacy.Adagrad is optimizers.Adagrad
    assert opt_legacy.Adam is optimizers.Adam
    assert opt_legacy.Ftrl is optimizers.Ftrl
    assert opt_legacy.Optimizer is optimizers.Optimizer
    assert opt_legacy.RMSprop is optimizers.RMSprop
