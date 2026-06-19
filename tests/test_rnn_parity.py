import numpy as np
import keras
import zero_keras.layers as layers


def check_layer_parity(layer_cls, keras_cls, inputs, atol=0.5, rtol=0.5, **kwargs):
    keras_layer = keras_cls(**kwargs)
    keras_out = keras_layer(inputs)
    zero_layer = layer_cls(**kwargs)
    kw = keras_layer.get_weights()
    zero_out = zero_layer(inputs)
    zero_layer.set_weights(kw)
    zero_out = zero_layer(inputs)

    keras_out_np = (
        keras_out.numpy() if hasattr(keras_out, "numpy") else np.array(keras_out)
    )
    zero_out_np = zero_out.numpy() if hasattr(zero_out, "numpy") else np.array(zero_out)

    np.testing.assert_allclose(keras_out_np, zero_out_np, atol=atol, rtol=rtol)


def test_simplernn():
    x = np.random.rand(3, 5, 4).astype(np.float32)
    check_layer_parity(layers.SimpleRNN, keras.layers.SimpleRNN, x, units=10)


def test_lstm():
    x = np.random.rand(3, 5, 4).astype(np.float32)
    check_layer_parity(layers.LSTM, keras.layers.LSTM, x, units=10)


def test_gru():
    x = np.random.rand(3, 5, 4).astype(np.float32)
    check_layer_parity(layers.GRU, keras.layers.GRU, x, units=10, reset_after=False)
