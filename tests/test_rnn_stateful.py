"""Module docstring."""

import numpy as np
import keras
import zero_keras.layers as layers


def check_layer_parity(layer_cls, keras_cls, inputs, atol=0.5, rtol=0.5, **kwargs):
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
    _ = keras_layer(inputs)
    keras_out2 = keras_layer(inputs)

    zero_layer = layer_cls(**kwargs)
    kw = keras_layer.get_weights()

    # Let's manually build it, set weights, and reset states.
    zero_layer.build(inputs.shape)
    zero_layer.set_weights(kw)
    zero_layer.reset_states()
    keras_layer.reset_states()

    _ = keras_layer(inputs)
    keras_out2 = keras_layer(inputs)

    _ = zero_layer(inputs)
    zero_out2 = zero_layer(inputs)

    keras_out_np2 = (
        keras_out2.numpy() if hasattr(keras_out2, "numpy") else np.array(keras_out2)
    )
    zero_out_np2 = (
        zero_out2.numpy() if hasattr(zero_out2, "numpy") else np.array(zero_out2)
    )
    np.testing.assert_allclose(keras_out_np2, zero_out_np2, atol=atol, rtol=rtol)


def test_simplernn_stateful():
    """Function docstring."""
    x = np.random.rand(3, 5, 4).astype(np.float32)
    check_layer_parity(
        layers.SimpleRNN, keras.layers.SimpleRNN, x, units=10, stateful=True
    )
