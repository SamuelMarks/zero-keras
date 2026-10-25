"""Unit and behavioral tests for ConvLSTM layers in zero_keras."""

from __future__ import annotations

from zero_keras import layers, random


def test_convlstm2d() -> None:
    """Verify ``layers.ConvLSTM2D`` forward pass and weight getters/setters."""
    x = random.uniform((2, 3, 5, 5, 4), dtype="float32")
    layer = layers.ConvLSTM2D(filters=6, kernel_size=(3, 3))
    out = layer(x)
    assert out.shape == (2, 3, 3, 6)
    kw = layer.get_weights()
    layer.set_weights(kw)
    out2 = layer(x)
    assert out2.shape == (2, 3, 3, 6)
