"""Module docstring."""

from __future__ import annotations
from zero_keras.types import TensorLike

import pytest
from ml_switcheroo_compiler.core import config

from zero_keras.core_layers import KerasTensor


def test_kerastensor_bool_eager_no_data() -> TensorLike:
    """Function docstring."""
    t = KerasTensor((2,))
    t.data = None

    old_eager = config.eager_mode if hasattr(config, "eager_mode") else False
    config.eager_mode = True
    try:
        with pytest.raises(TypeError):
            bool(t)
    finally:
        config.eager_mode = old_eager
