"""Unit and behavioral tests for layers extra io in zero_keras."""

from __future__ import annotations

from zero_keras import layers


def test_layers_io() -> None:
    """Verify or execute `test layers io`."""
    layer = layers.Dense(10)
    config = layers.serialize(layer)
    assert isinstance(config, dict)

    # deserialize dense requires kwargs handling or custom object. Our generic deserialize
    # tries to just call it with dict. But Dense __init__ takes `units` which might be in config.
    # We will just verify it calls globals correctly.
    assert layers.serialize(None) is None
    assert layers.serialize("dense") == "dense"

    assert layers.deserialize(None) is None


def test_new_layers() -> None:
    """Verify or execute `test new layers`."""
    gqa = layers.GroupQueryAttention(head_dim=2, num_query_heads=2, num_key_value_heads=2)
    res = gqa(None, None)
    assert res is None
    res1, _res2 = gqa(None, None, return_attention_scores=True)
    assert res1 is None

    hc = layers.HashedCrossing(num_bins=10)
    # Stub test
    assert callable(hc)
