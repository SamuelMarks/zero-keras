"""Module `test_normalization_extra` for zero_keras."""

from __future__ import annotations
from zero_keras.types import TensorLike
import numpy as np

from zero_keras import layers


def test_group_normalization_edge_cases() -> TensorLike:
    """Execute ``test_group_normalization_edge_cases``."""
    layer = layers.GroupNormalization(groups=2, axis=1, scale=False, center=False)
    x = np.random.rand(2, 4, 3, 3).astype(np.float32)
    out = layer(x)
    assert out.shape == (2, 4, 3, 3)


def test_rms_normalization_built_twice() -> TensorLike:
    """Execute ``test_rms_normalization_built_twice``."""
    layer = layers.RMSNormalization(axis=-1)
    layer.build((2, 5, 3))
    layer.build((2, 5, 3))
    assert layer.built


def test_unit_normalization() -> TensorLike:
    """Execute ``test_unit_normalization``."""
    layer = layers.UnitNormalization(axis=-1)
    x = np.random.rand(2, 5, 3).astype(np.float32)
    out = layer(x)
    assert out.shape == (2, 5, 3)


def test_spectral_normalization() -> TensorLike:
    """Execute ``test_spectral_normalization``."""
    layer = layers.SpectralNormalization(layers.Dense(4))
    x = np.random.rand(2, 5, 3).astype(np.float32)
    out = layer(x)
    assert out.shape == (2, 5, 4)


def test_group_normalization_axis_last() -> TensorLike:
    """Execute ``test_group_normalization_axis_last``."""
    layer = layers.GroupNormalization(groups=2, axis=3)
    x = np.random.rand(2, 4, 3, 4).astype(np.float32)
    out = layer(x)
    assert out.shape == (2, 4, 3, 4)
