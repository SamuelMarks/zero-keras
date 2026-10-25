"""Unit and behavioral tests for spatial dropout in zero_keras."""

from __future__ import annotations

from zero_keras import layers, ops


def test_spatial_dropout1d() -> None:
    """Verify or execute `test spatial dropout1d`."""
    layer = layers.SpatialDropout1D(0.5, data_format="channels_last")
    x = ops.ones((2, 10, 4))
    out = layer(x, training=True)
    assert out is not None

    layer_first = layers.SpatialDropout1D(0.5, data_format="channels_first")
    out = layer_first(x, training=True)
    assert out is not None


def test_spatial_dropout2d() -> None:
    """Verify or execute `test spatial dropout2d`."""
    layer = layers.SpatialDropout2D(0.5, data_format="channels_last")
    x = ops.ones((2, 10, 10, 4))
    out = layer(x, training=True)
    assert out is not None

    layer_first = layers.SpatialDropout2D(0.5, data_format="channels_first")
    out = layer_first(x, training=True)
    assert out is not None


def test_spatial_dropout3d() -> None:
    """Verify or execute `test spatial dropout3d`."""
    layer = layers.SpatialDropout3D(0.5, data_format="channels_last")
    x = ops.ones((2, 10, 10, 10, 4))
    out = layer(x, training=True)
    assert out is not None

    layer_first = layers.SpatialDropout3D(0.5, data_format="channels_first")
    out = layer_first(x, training=True)
    assert out is not None


def test_alpha_dropout() -> None:
    """Verify or execute `test alpha dropout`."""
    layer = layers.AlphaDropout(0.5, noise_shape=(2, 1, 4))
    x = ops.ones((2, 10, 4))
    out = layer(x, training=True)
    assert out is not None
