"""Module `test_normalization_extra` for zero_keras."""

from __future__ import annotations

from zero_keras import layers, random


def test_group_normalization_edge_cases() -> None:
    """Test ``group_normalization_edge_cases`` behavior and edge cases in ``test_normalization_extra``."""
    layer = layers.GroupNormalization(groups=2, axis=1, scale=False, center=False)
    x = random.uniform((2, 4, 3, 3), dtype="float32")
    out = layer(x)
    assert out.shape == (2, 4, 3, 3)


def test_rms_normalization_built_twice() -> None:
    """Test ``rms_normalization_built_twice`` behavior and edge cases in ``test_normalization_extra``."""
    layer = layers.RMSNormalization(axis=-1)
    layer.build((2, 5, 3))
    layer.build((2, 5, 3))
    assert layer.built


def test_unit_normalization() -> None:
    """Test ``unit_normalization`` behavior and edge cases in ``test_normalization_extra``."""
    layer = layers.UnitNormalization(axis=-1)
    x = random.uniform((2, 5, 3), dtype="float32")
    out = layer(x)
    assert out.shape == (2, 5, 3)


def test_spectral_normalization() -> None:
    """Test SpectralNormalization in inference and training modes (power iteration)."""
    layer = layers.SpectralNormalization(layers.Dense(4))
    x = random.uniform((2, 5, 3), dtype="float32")
    out = layer(x)
    assert out.shape == (2, 5, 4)
    out_train = layer(x, training=True)
    assert out_train.shape == (2, 5, 4)
    sn_no_k = layers.SpectralNormalization(layers.Dropout(0.1))
    sn_no_k.build((2, 5, 3))
    sn_no_k.normalize_weights()


def test_group_normalization_axis_last() -> None:
    """Test ``group_normalization_axis_last`` behavior and edge cases in ``test_normalization_extra``."""
    layer = layers.GroupNormalization(groups=2, axis=3)
    x = random.uniform((2, 4, 3, 4), dtype="float32")
    out = layer(x)
    assert out.shape == (2, 4, 3, 4)
