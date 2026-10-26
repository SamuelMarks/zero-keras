"""Module `test_layers_activations_extra` for zero_keras."""

from __future__ import annotations

import pytest

from zero_keras import ops

from .conftest import to_flat_floats


def test_activations_layer() -> None:
    """Test ``activations_layer`` behavior and edge cases in ``test_layers_activations_extra``."""
    from zero_keras.layers.activations import (
        ELU,
        Activation,
        LeakyReLU,
        PReLU,
        ReLU,
        Softmax,
    )

    layer = Activation("relu")
    res = layer(ops.convert_to_tensor([-1.0, 1.0]))
    assert to_flat_floats(res)[1] == 1.0

    layer_get_config = layer.get_config()
    assert (
        layer_get_config.get(
            "activation",
            layer.activation.__name__
            if hasattr(layer.activation, "__name__")
            else str(layer.activation),
        )
        == "relu"
    )

    layer = PReLU(
        alpha_initializer="zeros",
        alpha_regularizer="l2",
        alpha_constraint="max_norm",
        shared_axes=[1, 2],
    )
    layer.build((2, 3, 3, 4))
    res = layer(ops.ones((2, 3, 3, 4)))

    layer_get_config = layer.get_config()
    assert layer_get_config.get("shared_axes", layer.shared_axes) == [1, 2]

    layer = ELU(alpha=2.0)
    res = layer(ops.convert_to_tensor([-1.0, 1.0]))

    layer_get_config = layer.get_config()
    assert layer_get_config.get("alpha", layer.alpha) == 2.0

    layer = ReLU(max_value=1.0, negative_slope=0.1, threshold=0.0)
    res = layer(ops.convert_to_tensor([-1.0, 2.0]))

    layer_get_config = layer.get_config()
    assert layer_get_config.get("max_value", layer.max_value) == 1.0

    layer = LeakyReLU(negative_slope=0.2)
    res = layer(ops.convert_to_tensor([-1.0, 1.0]))

    layer_get_config = layer.get_config()
    assert layer_get_config.get("negative_slope", layer.negative_slope) == 0.2

    layer = Softmax(axis=-1)
    res = layer(ops.convert_to_tensor([[1.0, 1.0]]))

    layer_get_config = layer.get_config()
    assert layer_get_config.get("axis", layer.axis) == -1


if __name__ == "__main__":
    pytest.main(["-v", "test_layers_activations_extra.py"])
