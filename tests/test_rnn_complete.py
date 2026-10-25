"""Unit and behavioral tests for complete RNN layer family in zero_keras."""

from __future__ import annotations

from zero_keras import layers, random


def test_simplernn() -> None:
    """Verify ``layers.SimpleRNN`` forward pass and weight synchronization."""
    x = random.uniform((3, 5, 4), dtype="float32")
    layer = layers.SimpleRNN(units=10)
    out = layer(x)
    assert out.shape == (3, 10)
    layer.set_weights(layer.get_weights())
    assert layer(x).shape == (3, 10)


def test_lstm() -> None:
    """Verify ``layers.LSTM`` forward pass and weight synchronization."""
    x = random.uniform((3, 5, 4), dtype="float32")
    layer = layers.LSTM(units=10)
    out = layer(x)
    assert out.shape == (3, 10)
    layer.set_weights(layer.get_weights())
    assert layer(x).shape == (3, 10)


def test_gru() -> None:
    """Verify ``layers.GRU`` forward pass and weight synchronization."""
    x = random.uniform((3, 5, 4), dtype="float32")
    layer = layers.GRU(units=10, reset_after=False)
    out = layer(x)
    assert out.shape == (3, 10)
    layer.set_weights(layer.get_weights())
    assert layer(x).shape == (3, 10)


def test_simplernn_advanced() -> None:
    """Verify ``layers.SimpleRNN`` with ``return_sequences`` and ``return_state``."""
    x = random.uniform((3, 5, 4), dtype="float32")
    l_seq = layers.SimpleRNN(units=10, return_sequences=True)
    assert l_seq(x).shape == (3, 5, 10)
    l_state = layers.SimpleRNN(units=10, return_state=True)
    out, state = l_state(x)
    assert out.shape == (3, 10)
    assert state.shape == (3, 10)


def test_bidirectional() -> None:
    """Verify ``layers.Bidirectional(SimpleRNN)`` forward pass."""
    x = random.uniform((3, 5, 4), dtype="float32")
    zero_layer = layers.Bidirectional(layers.SimpleRNN(10))
    out = zero_layer(x)
    assert out.shape == (3, 20)
    zero_layer.set_weights(zero_layer.get_weights())
    assert zero_layer(x).shape == (3, 20)


def test_stacked() -> None:
    """Verify ``layers.RNN(StackedRNNCells([SimpleRNNCell, LSTMCell]))`` forward pass."""
    x = random.uniform((3, 5, 4), dtype="float32")
    zero_layer = layers.RNN(layers.StackedRNNCells([layers.SimpleRNNCell(10), layers.LSTMCell(10)]))
    out = zero_layer(x)
    assert out.shape == (3, 10)
    zero_layer.set_weights(zero_layer.get_weights())
    assert zero_layer(x).shape == (3, 10)
