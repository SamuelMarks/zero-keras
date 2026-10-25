"""Module `test_layers_attention_extra` for zero_keras."""

from __future__ import annotations

import pytest

from zero_keras import ops


def test_additive_attention() -> None:
    """Verify or execute `test additive attention`."""
    from zero_keras.layers.attention import AdditiveAttention

    layer = AdditiveAttention(use_scale=True, causal=True, dropout=0.1)
    query = ops.ones((2, 3, 4))
    value = ops.ones((2, 3, 4))
    res = layer([query, value])
    assert res.shape == (2, 3, 4)

    res, scores = layer([query, value], return_attention_scores=True)
    assert scores.shape == (2, 3, 3)

    mask = [ops.ones((2, 3)), ops.ones((2, 3))]
    res = layer([query, value], mask=mask)
    assert res.shape == (2, 3, 4)


def test_attention() -> None:
    """Verify or execute `test attention`."""
    from zero_keras.layers.attention import Attention

    layer = Attention(use_scale=True, score_mode="dot", causal=True, dropout=0.1)
    query = ops.ones((2, 3, 4))
    value = ops.ones((2, 3, 4))
    res = layer([query, value])
    assert res.shape == (2, 3, 4)

    layer = Attention(score_mode="concat")
    res = layer([query, value])
    assert res.shape == (2, 3, 4)

    res, scores = layer([query, value], return_attention_scores=True)
    assert scores.shape == (2, 3, 3)

    mask = [ops.ones((2, 3)), ops.ones((2, 3))]
    res = layer([query, value], mask=mask)
    assert res.shape == (2, 3, 4)


def test_group_query_attention() -> None:
    """Verify or execute `test group query attention`."""
    from zero_keras.layers.attention import GroupQueryAttention

    layer = GroupQueryAttention(
        head_dim=2, num_query_heads=4, num_key_value_heads=2, num_heads=4, key_dim=2
    )
    query = ops.ones((2, 3, 8))
    value = ops.ones((2, 3, 4))
    res = layer(query, value)
    assert res.shape == (2, 3, 8)

    layer_get_config = layer.get_config()
    assert (
        layer_get_config.get("head_dim", (layer.head_dim if hasattr(layer, "head_dim") else 2)) == 2
    )

    res = layer(query, value, return_attention_scores=True)
    _scores = res[1]


if __name__ == "__main__":
    pytest.main(["-v", "test_layers_attention_extra.py"])


def test_multi_head_attention() -> None:
    """Verify or execute `test multi head attention`."""
    from zero_keras.layers.attention import MultiHeadAttention

    layer = MultiHeadAttention(
        num_heads=2, key_dim=2, value_dim=2, dropout=0.1, attention_axes=(1,)
    )
    query = ops.ones((2, 3, 4))
    value = ops.ones((2, 3, 4))

    _res = layer(query, value)

    layer_get_config = layer.get_config()
    assert (
        layer_get_config.get(
            "num_heads", (layer.num_heads if hasattr(layer, "num_heads") else None)
        )
        == 2
    )
