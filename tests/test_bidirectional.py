import pytest
import numpy as np
import keras
import zero_keras.layers as layers


def check_layer_parity(layer_cls, keras_cls, inputs, atol=1e-5, rtol=1e-5, **kwargs):
    keras_layer = keras_cls(**kwargs)
    keras_out = keras_layer(inputs)
    zero_layer = layer_cls(**kwargs)
    kw = keras_layer.get_weights()
    zero_layer(inputs)
    zero_layer.set_weights(kw)
    zero_out = zero_layer(inputs)

    keras_out_np = (
        keras_out.numpy() if hasattr(keras_out, "numpy") else np.array(keras_out)
    )
    zero_out_np = zero_out.numpy() if hasattr(zero_out, "numpy") else np.array(zero_out)
    np.testing.assert_allclose(keras_out_np, zero_out_np, atol=atol, rtol=rtol)


def test_bidirectional():
    pytest.skip("Skipping due to ml-switcheroo-compiler eager backend limitations")
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
