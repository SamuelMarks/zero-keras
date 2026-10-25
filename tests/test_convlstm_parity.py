"""Parity tests for ConvLSTM2D in zero_keras vs Keras."""

from __future__ import annotations

import keras
import numpy as np

from zero_keras import layers
from zero_keras.types import TensorLike


def check_layer_parity(
    layer_cls: TensorLike,
    keras_cls: TensorLike,
    inputs: TensorLike,
    atol: TensorLike = 0.5,
    rtol: TensorLike = 0.5,
    **kwargs: TensorLike,
) -> None:
    """Compare output between Keras and zero_keras ConvLSTM layers.

    Args:
        layer_cls: zero_keras layer class.
        keras_cls: Keras layer class.
        inputs: Input 5D array.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        **kwargs: Layer constructor arguments.
    """
    keras_layer = keras_cls(**kwargs)
    keras_out = keras_layer(inputs)
    zero_layer = layer_cls(**kwargs)
    kw = keras_layer.get_weights()
    zero_layer(inputs)

    if hasattr(zero_layer, "set_weights"):
        zero_layer.set_weights(kw)

    zero_out = zero_layer(inputs)

    keras_out_np = keras_out.numpy() if hasattr(keras_out, "numpy") else np.array(keras_out)
    zero_out_np = zero_out.numpy() if hasattr(zero_out, "numpy") else np.array(zero_out)
    np.testing.assert_allclose(keras_out_np, zero_out_np, atol=atol, rtol=rtol)


def test_convlstm2d() -> None:
    """Verify ``layers.ConvLSTM2D`` parity with ``keras.layers.ConvLSTM2D``."""
    x = np.random.rand(2, 3, 5, 5, 4).astype(np.float32)
    check_layer_parity(layers.ConvLSTM2D, keras.layers.ConvLSTM2D, x, filters=6, kernel_size=(3, 3))
