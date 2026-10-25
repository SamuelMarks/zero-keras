"""Unit and behavioral tests for bidirectional backward in zero_keras."""

from __future__ import annotations

from zero_keras import layers, random


def test_bidirectional_backward_layer() -> None:
    """Verify or execute `test bidirectional backward layer`."""
    l1 = layers.SimpleRNN(4)
    l2 = layers.SimpleRNN(4, go_backwards=True)
    b = layers.Bidirectional(l1, backward_layer=l2)
    b(random.uniform((2, 3, 2)))
