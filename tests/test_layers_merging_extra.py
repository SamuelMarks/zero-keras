"""Module `test_layers_merging_extra` for zero_keras."""

from __future__ import annotations
from zero_keras.types import TensorLike
import numpy as np
import pytest


def test_merging_layers() -> TensorLike:
    """Execute ``test_merging_layers``."""
    from zero_keras.layers.merging import (
        Add,
        Average,
        Concatenate,
        Maximum,
        Minimum,
        Multiply,
        Subtract,
        add,
        average,
        concatenate,
        maximum,
        minimum,
        multiply,
        subtract,
    )

    a = np.ones((2, 3))
    b = np.ones((2, 3))

    res = Add()([a, b])
    res = add([a, b])

    res = Subtract()([a, b])
    res = subtract([a, b])

    res = Multiply()([a, b])
    res = multiply([a, b])

    res = Average()([a, b])
    res = average([a, b])

    res = Maximum()([a, b])
    res = maximum([a, b])

    res = Minimum()([a, b])
    res = minimum([a, b])

    res = Concatenate(axis=-1)([a, b])
    res = concatenate([a, b], axis=-1)


def test_dot_merging() -> TensorLike:
    """Execute ``test_dot_merging``."""
    from zero_keras.layers.merging import Dot, dot

    a = np.ones((2, 4))
    b = np.ones((2, 4))

    # 2D case to hit expanding dims logic and pure dot
    res = Dot(axes=1)([a, b])

    # 3D with standard axes
    a3 = np.ones((2, 3, 4))
    b3 = np.ones((2, 3, 4))
    res3 = Dot(axes=2)([a3, b3])

    # normalize
    res_norm = dot([a, b], axes=1, normalize=True)


if __name__ == "__main__":
    pytest.main(["-v", "test_layers_merging_extra.py"])
