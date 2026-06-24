"""Module docstring."""

import numpy as np
import keras
import zero_keras.layers as layers


def check_layer_parity(layer_cls, keras_cls, inputs, atol=1e-5, rtol=1e-5, **kwargs):
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
    zero_out = zero_layer(inputs)
    zero_layer.set_weights(kw)
    zero_out = zero_layer(inputs)

    if isinstance(keras_out, list) or isinstance(keras_out, tuple):
        for ko, zo in zip(keras_out, zero_out):
            ko_np = ko.numpy() if hasattr(ko, "numpy") else np.array(ko)
            zo_np = zo.numpy() if hasattr(zo, "numpy") else np.array(zo)
            np.testing.assert_allclose(ko_np, zo_np, atol=atol, rtol=rtol)
    else:
        keras_out_np = (
            keras_out.numpy() if hasattr(keras_out, "numpy") else np.array(keras_out)
        )
        zero_out_np = (
            zero_out.numpy() if hasattr(zero_out, "numpy") else np.array(zero_out)
        )
        np.testing.assert_allclose(keras_out_np, zero_out_np, atol=atol, rtol=rtol)


def test_simplernn_return_seq():
    """Function docstring."""
    x = np.random.rand(3, 5, 4).astype(np.float32)
    check_layer_parity(
        layers.SimpleRNN, keras.layers.SimpleRNN, x, units=10, return_sequences=True
    )


def test_simplernn_return_state():
    """Function docstring."""
    x = np.random.rand(3, 5, 4).astype(np.float32)
    check_layer_parity(
        layers.SimpleRNN, keras.layers.SimpleRNN, x, units=10, return_state=True
    )


def test_simplernn_return_seq_and_state():
    """Function docstring."""
    x = np.random.rand(3, 5, 4).astype(np.float32)
    check_layer_parity(
        layers.SimpleRNN,
        keras.layers.SimpleRNN,
        x,
        units=10,
        return_sequences=True,
        return_state=True,
    )
