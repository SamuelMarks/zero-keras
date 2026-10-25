"""Unit and behavioral tests for advanced SimpleRNN modes in zero_keras."""

from __future__ import annotations

from zero_keras import layers, random


def test_simplernn_return_seq() -> None:
    """Verify ``SimpleRNN`` with ``return_sequences=True``."""
    x = random.uniform((3, 5, 4), dtype="float32")
    layer = layers.SimpleRNN(units=10, return_sequences=True)
    out = layer(x)
    assert out.shape == (3, 5, 10)
    layer.set_weights(layer.get_weights())
    assert layer(x).shape == (3, 5, 10)


def test_simplernn_return_state() -> None:
    """Verify ``SimpleRNN`` with ``return_state=True``."""
    x = random.uniform((3, 5, 4), dtype="float32")
    layer = layers.SimpleRNN(units=10, return_state=True)
    out, state = layer(x)
    assert out.shape == (3, 10)
    assert state.shape == (3, 10)


def test_simplernn_return_seq_and_state() -> None:
    """Verify ``SimpleRNN`` with ``return_sequences=True`` and ``return_state=True``."""
    x = random.uniform((3, 5, 4), dtype="float32")
    layer = layers.SimpleRNN(units=10, return_sequences=True, return_state=True)
    out, state = layer(x)
    assert out.shape == (3, 5, 10)
    assert state.shape == (3, 10)
