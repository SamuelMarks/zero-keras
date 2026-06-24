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
    keras_layer = keras_cls(**kwargs)  # pragma: no cover
    keras_out = keras_layer(inputs)  # pragma: no cover
    zero_layer = layer_cls(**kwargs)  # pragma: no cover
    kw = keras_layer.get_weights()  # pragma: no cover
    zero_layer(inputs)  # pragma: no cover
    zero_layer.set_weights(kw)  # pragma: no cover
    zero_out = zero_layer(inputs)  # pragma: no cover

    keras_out_np = (  # pragma: no cover
        keras_out.numpy() if hasattr(keras_out, "numpy") else np.array(keras_out)
    )
    zero_out_np = (
        zero_out.numpy() if hasattr(zero_out, "numpy") else np.array(zero_out)
    )  # pragma: no cover
    np.testing.assert_allclose(
        keras_out_np, zero_out_np, atol=atol, rtol=rtol
    )  # pragma: no cover


def test_bidirectional():
    """Function docstring."""
    x = np.random.rand(3, 5, 4).astype(np.float32)
    # create keras bidirectional
    keras_layer = keras.layers.Bidirectional(keras.layers.SimpleRNN(10))
    keras_out = keras_layer(x)

    zero_layer = layers.Bidirectional(layers.SimpleRNN(10))
    zero_out = zero_layer(x)

    kw = keras_layer.get_weights()
    zero_layer.set_weights(kw)
    zero_out = zero_layer(x)

    keras_out_np = (
        keras_out.numpy() if hasattr(keras_out, "numpy") else np.array(keras_out)
    )
    zero_out_np = zero_out.numpy() if hasattr(zero_out, "numpy") else np.array(zero_out)
    np.testing.assert_allclose(keras_out_np, zero_out_np, atol=1e-5, rtol=1e-5)
