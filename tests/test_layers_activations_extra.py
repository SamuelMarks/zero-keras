"""Module `test_layers_activations_extra` for zero_keras."""

from __future__ import annotations
from zero_keras.types import TensorLike
import numpy as np
import pytest


def test_activations_layer() -> TensorLike:
    """Execute ``test_activations_layer``."""
    from zero_keras.layers.activations import (
        ELU,
        Activation,
        LeakyReLU,
        PReLU,
        ReLU,
        Softmax,
    )

    # test base Activation
    layer = Activation("relu")
    res = layer(np.array([-1.0, 1.0]))
    assert np.array(res)[1] == 1.0

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

    # test PReLU
    layer = PReLU(
        alpha_initializer="zeros",
        alpha_regularizer="l2",
        alpha_constraint="max_norm",
        shared_axes=[1, 2],
    )
    layer.build((2, 3, 3, 4))
    res = layer(np.ones((2, 3, 3, 4)))

    layer_get_config = layer.get_config()
    assert layer_get_config.get("shared_axes", layer.shared_axes) == [1, 2]

    # test ELU
    layer = ELU(alpha=2.0)
    res = layer(np.array([-1.0, 1.0]))

    layer_get_config = layer.get_config()
    assert layer_get_config.get("alpha", layer.alpha) == 2.0

    # test ReLU
    layer = ReLU(max_value=1.0, negative_slope=0.1, threshold=0.0)
    res = layer(np.array([-1.0, 2.0]))

    layer_get_config = layer.get_config()
    assert layer_get_config.get("max_value", layer.max_value) == 1.0

    # test LeakyReLU
    layer = LeakyReLU(negative_slope=0.2)
    res = layer(np.array([-1.0, 1.0]))

    layer_get_config = layer.get_config()
    assert layer_get_config.get("negative_slope", layer.negative_slope) == 0.2

    # test Softmax
    layer = Softmax(axis=-1)
    res = layer(np.array([[1.0, 1.0]]))

    layer_get_config = layer.get_config()
    assert layer_get_config.get("axis", layer.axis) == -1


if __name__ == "__main__":
    pytest.main(["-v", "test_layers_activations_extra.py"])
