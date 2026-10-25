"""Unit and behavioral tests for Bidirectional layers in zero_keras."""

from __future__ import annotations

from zero_keras import layers, random


def test_bidirectional() -> None:
    """Verify ``layers.Bidirectional`` forward pass and weight synchronization."""
    x = random.uniform((3, 5, 4), dtype="float32")
    zero_layer = layers.Bidirectional(layers.SimpleRNN(10))
    zero_out = zero_layer(x)
    assert zero_out.shape == (3, 20)

    kw = zero_layer.get_weights()
    zero_layer.set_weights(kw)
    zero_out2 = zero_layer(x)
    assert zero_out2.shape == (3, 20)
