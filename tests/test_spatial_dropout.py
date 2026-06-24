"""Module docstring."""

import numpy as np
from zero_keras import layers


def test_spatial_dropout1d():
    """Function docstring."""
    layer = layers.SpatialDropout1D(0.5, data_format="channels_last")
    x = np.ones((2, 10, 4))
    out = layer(x, training=True)
    assert out is not None

    layer_first = layers.SpatialDropout1D(0.5, data_format="channels_first")
    out = layer_first(x, training=True)
    assert out is not None


def test_spatial_dropout2d():
    """Function docstring."""
    layer = layers.SpatialDropout2D(0.5, data_format="channels_last")
    x = np.ones((2, 10, 10, 4))
    out = layer(x, training=True)
    assert out is not None

    layer_first = layers.SpatialDropout2D(0.5, data_format="channels_first")
    out = layer_first(x, training=True)
    assert out is not None


def test_spatial_dropout3d():
    """Function docstring."""
    layer = layers.SpatialDropout3D(0.5, data_format="channels_last")
    x = np.ones((2, 10, 10, 10, 4))
    out = layer(x, training=True)
    assert out is not None

    layer_first = layers.SpatialDropout3D(0.5, data_format="channels_first")
    out = layer_first(x, training=True)
    assert out is not None


def test_alpha_dropout():
    """Function docstring."""
    layer = layers.AlphaDropout(0.5, noise_shape=(2, 1, 4))
    x = np.ones((2, 10, 4))
    out = layer(x, training=True)
    assert out is not None
