"""Parity tests for Bidirectional RNN layers in zero_keras vs Keras."""

from __future__ import annotations

import keras
import numpy as np

from zero_keras import layers
from zero_keras.types import TensorLike


def check_layer_parity(
    layer_cls: TensorLike,
    keras_cls: TensorLike,
    inputs: TensorLike,
    atol: TensorLike = 1e-5,
    rtol: TensorLike = 1e-5,
    **kwargs: TensorLike,
) -> None:
    """Compare output between a Keras layer and a zero_keras layer with synchronized weights.

    Args:
        layer_cls: zero_keras layer class.
        keras_cls: Keras layer class.
        inputs: Input tensor or array.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        **kwargs: Layer constructor arguments.
    """
    keras_layer = keras_cls(**kwargs)
    keras_out = keras_layer(inputs)
    zero_layer = layer_cls(**kwargs)
    kw = keras_layer.get_weights()
    zero_layer(inputs)
    zero_layer.set_weights(kw)
    zero_out = zero_layer(inputs)

    keras_out_np = keras_out.numpy() if hasattr(keras_out, "numpy") else np.array(keras_out)
    zero_out_np = zero_out.numpy() if hasattr(zero_out, "numpy") else np.array(zero_out)
    np.testing.assert_allclose(keras_out_np, zero_out_np, atol=atol, rtol=rtol)


def test_bidirectional() -> None:
    """Verify ``layers.Bidirectional`` parity with ``keras.layers.Bidirectional``."""
    x = np.random.rand(3, 5, 4).astype(np.float32)
    keras_layer = keras.layers.Bidirectional(keras.layers.SimpleRNN(10))
    keras_out = keras_layer(x)

    zero_layer = layers.Bidirectional(layers.SimpleRNN(10))
    zero_out = zero_layer(x)

    kw = keras_layer.get_weights()
    zero_layer.set_weights(kw)
    zero_out = zero_layer(x)

    keras_out_np = keras_out.numpy() if hasattr(keras_out, "numpy") else np.array(keras_out)
    zero_out_np = zero_out.numpy() if hasattr(zero_out, "numpy") else np.array(zero_out)
    np.testing.assert_allclose(keras_out_np, zero_out_np, atol=1e-5, rtol=1e-5)
