"""Module `test_layers_merging_extra` for zero_keras."""

from __future__ import annotations

import pytest

from zero_keras import ops


def test_merging_layers() -> None:
    """Test ``merging_layers`` behavior and edge cases in ``test_layers_merging_extra``."""
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

    a = ops.ones((2, 3))
    b = ops.ones((2, 3))

    _res = Add()([a, b])
    _res = add([a, b])

    _res = Subtract()([a, b])
    _res = subtract([a, b])

    _res = Multiply()([a, b])
    _res = multiply([a, b])

    _res = Average()([a, b])
    _res = average([a, b])

    _res = Maximum()([a, b])
    _res = maximum([a, b])

    _res = Minimum()([a, b])
    _res = minimum([a, b])

    _res = Concatenate(axis=-1)([a, b])
    _res = concatenate([a, b], axis=-1)


def test_dot_merging() -> None:
    """Test ``dot_merging`` behavior and edge cases in ``test_layers_merging_extra``."""
    from zero_keras.layers.merging import Dot, dot

    a = ops.ones((2, 4))
    b = ops.ones((2, 4))

    _res = Dot(axes=1)([a, b])

    a3 = ops.ones((2, 3, 4))
    b3 = ops.ones((2, 3, 4))
    _res3 = Dot(axes=2)([a3, b3])

    _res_norm = dot([a, b], axes=1, normalize=True)


if __name__ == "__main__":
    pytest.main(["-v", "test_layers_merging_extra.py"])
