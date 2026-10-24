"""Module docstring."""

from __future__ import annotations
from zero_keras.types import TensorLike

import keras
import numpy as np

from zero_keras import layers


def check_layer_parity(
    layer_cls: TensorLike,
    keras_cls: TensorLike,
    inputs: TensorLike,
    atol: TensorLike = 0.5,
    rtol: TensorLike = 0.5,
    **kwargs: TensorLike,
) -> TensorLike:
    """Function docstring.

    Args:
        layer_cls: Description.
        keras_cls: Description.
        inputs: Description.
        atol: Description.
        rtol: Description.
        kwargs: Description.
    """
    keras_layer = keras_cls(**kwargs)
    keras_out = keras_layer(inputs)
    zero_layer = layer_cls(**kwargs)
    kw = keras_layer.get_weights()
    zero_layer(inputs)

    if hasattr(zero_layer, "set_weights"):
        zero_layer.set_weights(kw)
    else:
        zero_layer.forward_layer.set_weights(kw[: len(kw) // 2])
        zero_layer.backward_layer.set_weights(kw[len(kw) // 2 :])

    zero_out = zero_layer(inputs)

    if isinstance(keras_out, (list, tuple)):
        for ko, zo in zip(keras_out, zero_out):
            ko_np = ko.numpy() if hasattr(ko, "numpy") else np.array(ko)
            zo_np = zo.numpy() if hasattr(zo, "numpy") else np.array(zo)
            np.testing.assert_allclose(ko_np, zo_np, atol=atol, rtol=rtol)
    else:
        keras_out_np = keras_out.numpy() if hasattr(keras_out, "numpy") else np.array(keras_out)
        zero_out_np = zero_out.numpy() if hasattr(zero_out, "numpy") else np.array(zero_out)
        np.testing.assert_allclose(keras_out_np, zero_out_np, atol=atol, rtol=rtol)


def test_simplernn() -> TensorLike:
    """Function docstring."""
    x = np.random.rand(3, 5, 4).astype(np.float32)
    check_layer_parity(layers.SimpleRNN, keras.layers.SimpleRNN, x, units=10)


def test_lstm() -> TensorLike:
    """Function docstring."""
    x = np.random.rand(3, 5, 4).astype(np.float32)
    check_layer_parity(layers.LSTM, keras.layers.LSTM, x, units=10)


def test_gru() -> TensorLike:
    """Function docstring."""
    x = np.random.rand(3, 5, 4).astype(np.float32)
    check_layer_parity(layers.GRU, keras.layers.GRU, x, units=10, reset_after=False)


def test_simplernn_advanced() -> TensorLike:
    """Function docstring."""
    x = np.random.rand(3, 5, 4).astype(np.float32)
    check_layer_parity(layers.SimpleRNN, keras.layers.SimpleRNN, x, units=10, return_sequences=True)
    check_layer_parity(layers.SimpleRNN, keras.layers.SimpleRNN, x, units=10, return_state=True)


def test_bidirectional() -> TensorLike:
    """Function docstring."""
    x = np.random.rand(3, 5, 4).astype(np.float32)
    keras_layer = keras.layers.Bidirectional(keras.layers.SimpleRNN(10))
    zero_layer = layers.Bidirectional(layers.SimpleRNN(10))
    keras_out = keras_layer(x)
    zero_layer(x)
    kw = keras_layer.get_weights()
    zero_layer.set_weights(kw)
    zero_out = zero_layer(x)
    np.testing.assert_allclose(np.array(keras_out), np.array(zero_out), atol=0.5, rtol=0.5)


def test_stacked() -> TensorLike:
    """Function docstring."""
    x = np.random.rand(3, 5, 4).astype(np.float32)
    keras_layer = keras.layers.RNN(
        keras.layers.StackedRNNCells([keras.layers.SimpleRNNCell(10), keras.layers.LSTMCell(10)])
    )
    zero_layer = layers.RNN(layers.StackedRNNCells([layers.SimpleRNNCell(10), layers.LSTMCell(10)]))

    keras_out = keras_layer(x)
    zero_layer(x)
    zero_layer.set_weights(keras_layer.get_weights())
    zero_out = zero_layer(x)

    np.testing.assert_allclose(np.array(keras_out), np.array(zero_out), atol=0.5, rtol=0.5)
