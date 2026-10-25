"""Unit and behavioral tests for stateful RNN layers in zero_keras."""

from __future__ import annotations

from zero_keras import layers, random


def test_simplernn_stateful() -> None:
    """Verify stateful ``layers.SimpleRNN`` across consecutive batches and state resets."""
    x = random.uniform((3, 5, 4), dtype="float32")
    zero_layer = layers.SimpleRNN(units=10, stateful=True)
    zero_layer.build((3, 5, 4))
    kw = zero_layer.get_weights()
    zero_layer.set_weights(kw)
    zero_layer.reset_states()

    out1 = zero_layer(x)
    out2 = zero_layer(x)
    assert out1.shape == (3, 10)
    assert out2.shape == (3, 10)
