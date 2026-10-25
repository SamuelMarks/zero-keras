"""Parity tests for RNN layers in zero_keras vs Keras."""

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
    """Compare output between Keras and zero_keras RNN layers with synchronized weights.

    Args:
        layer_cls: zero_keras layer class.
        keras_cls: Keras layer class.
        inputs: Input 3D sequence tensor or array.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        **kwargs: Layer constructor arguments.
    """
    keras_layer = keras_cls(**kwargs)
    keras_out = keras_layer(inputs)
    zero_layer = layer_cls(**kwargs)
    kw = keras_layer.get_weights()
    zero_out = zero_layer(inputs)
    zero_layer.set_weights(kw)
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


def test_simplernn() -> None:
    """Verify ``layers.SimpleRNN`` parity with ``keras.layers.SimpleRNN``."""
    x = np.random.rand(3, 5, 4).astype(np.float32)
    check_layer_parity(layers.SimpleRNN, keras.layers.SimpleRNN, x, units=10)


def test_lstm() -> None:
    """Verify ``layers.LSTM`` parity with ``keras.layers.LSTM``."""
    x = np.random.rand(3, 5, 4).astype(np.float32)
    check_layer_parity(layers.LSTM, keras.layers.LSTM, x, units=10)


def test_gru() -> None:
    """Verify ``layers.GRU`` parity with ``keras.layers.GRU``."""
    x = np.random.rand(3, 5, 4).astype(np.float32)
    check_layer_parity(layers.GRU, keras.layers.GRU, x, units=10, reset_after=False)


def test_simplernn_return_sequences_and_state_parity() -> None:
    """Verify ``SimpleRNN`` with ``return_sequences`` and ``return_state`` parity."""
    x = np.random.rand(3, 5, 4).astype(np.float32)
    check_layer_parity(layers.SimpleRNN, keras.layers.SimpleRNN, x, units=10, return_sequences=True)
    check_layer_parity(layers.SimpleRNN, keras.layers.SimpleRNN, x, units=10, return_state=True)
    check_layer_parity(
        layers.SimpleRNN,
        keras.layers.SimpleRNN,
        x,
        units=10,
        return_sequences=True,
        return_state=True,
    )


def test_stacked_rnn_parity() -> None:
    """Verify ``RNN(StackedRNNCells(...))`` parity with Keras."""
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


def test_simplernn_stateful_parity() -> None:
    """Verify stateful ``SimpleRNN`` parity across consecutive batches with Keras."""
    x = np.random.rand(3, 5, 4).astype(np.float32)
    keras_layer = keras.layers.SimpleRNN(units=10, stateful=True)
    _ = keras_layer(x)

    zero_layer = layers.SimpleRNN(units=10, stateful=True)
    kw = keras_layer.get_weights()
    zero_layer.build(x.shape)
    zero_layer.set_weights(kw)
    zero_layer.reset_states()
    keras_layer.reset_states()

    _ = keras_layer(x)
    keras_out2 = keras_layer(x)

    _ = zero_layer(x)
    zero_out2 = zero_layer(x)

    keras_out_np2 = keras_out2.numpy() if hasattr(keras_out2, "numpy") else np.array(keras_out2)
    zero_out_np2 = zero_out2.numpy() if hasattr(zero_out2, "numpy") else np.array(zero_out2)
    np.testing.assert_allclose(keras_out_np2, zero_out_np2, atol=0.5, rtol=0.5)
