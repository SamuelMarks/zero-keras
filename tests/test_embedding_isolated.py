"""Module `test_embedding_isolated` for zero_keras."""

from __future__ import annotations

from zero_keras import layers, random


def test_embedding_isolated() -> None:
    """Test ``embedding_isolated`` behavior and edge cases in ``test_embedding_isolated``."""
    layer = layers.Embedding(input_dim=10, output_dim=4)
    x = random.randint((2, 5), 0, 10)
    out = layer(x)
    assert out.shape == (2, 5, 4)


def test_embedding_built_twice() -> None:
    """Test ``embedding_built_twice`` behavior and edge cases in ``test_embedding_isolated``."""
    layer = layers.Embedding(input_dim=10, output_dim=4)
    layer.build((2, 5))
    layer.build((2, 5))
    assert layer.built
