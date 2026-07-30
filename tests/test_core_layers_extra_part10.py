"""Module docstring."""

import pytest
from ml_switcheroo_compiler.core import config

from zero_keras.core_layers import KerasTensor


def test_kerastensor_bool_eager_no_data():
    """Function docstring."""
    t = KerasTensor((2,))
    t.data = None

    old_eager = getattr(config, "eager_mode", False)
    config.eager_mode = True
    try:
        with pytest.raises(TypeError):
            bool(t)
    finally:
        config.eager_mode = old_eager
