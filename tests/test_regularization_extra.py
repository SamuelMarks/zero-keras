"""Module `test_regularization_extra` for zero_keras."""

from __future__ import annotations

from zero_keras import layers, random


def test_activity_regularization_l1_l2() -> None:
    """Test ``activity_regularization_l1_l2`` behavior and edge cases in ``test_regularization_extra``."""
    layer = layers.ActivityRegularization(l1=0.1, l2=0.2)
    x = random.uniform((2, 5), dtype="float32")
    out = layer(x)
    assert out.shape == (2, 5)


def test_gaussian_dropout() -> None:
    """Test ``gaussian_dropout`` behavior and edge cases in ``test_regularization_extra``."""
    layer = layers.GaussianDropout(rate=0.2)
    x = random.uniform((2, 5), dtype="float32")
    out = layer(x)
    assert out.shape == (2, 5)


def test_gaussian_noise() -> None:
    """Test ``gaussian_noise`` behavior and edge cases in ``test_regularization_extra``."""
    layer = layers.GaussianNoise(stddev=0.1)
    x = random.uniform((2, 5), dtype="float32")
    out = layer(x)
    assert out.shape == (2, 5)


def test_spatial_dropout_not_training() -> None:
    """Test ``spatial_dropout_not_training`` behavior and edge cases in ``test_regularization_extra``."""
    layer1d = layers.SpatialDropout1D(rate=0.2)
    x1d = random.uniform((2, 10, 3), dtype="float32")
    assert layer1d(x1d, training=False).shape == (2, 10, 3)

    layer2d = layers.SpatialDropout2D(rate=0.2)
    x2d = random.uniform((2, 10, 10, 3), dtype="float32")
    assert layer2d(x2d, training=False).shape == (2, 10, 10, 3)

    layer3d = layers.SpatialDropout3D(rate=0.2)
    x3d = random.uniform((2, 10, 10, 10, 3), dtype="float32")
    assert layer3d(x3d, training=False).shape == (2, 10, 10, 10, 3)


def test_spatial_dropout_training() -> None:
    """Test ``spatial_dropout_training`` behavior and edge cases in ``test_regularization_extra``."""
    layer1d = layers.SpatialDropout1D(rate=0.2)
    x1d = random.uniform((2, 10, 3), dtype="float32")
    layer1d(x1d, training=True)

    layer2d = layers.SpatialDropout2D(rate=0.2)
    x2d = random.uniform((2, 10, 10, 3), dtype="float32")
    layer2d(x2d, training=True)

    layer3d = layers.SpatialDropout3D(rate=0.2)
    x3d = random.uniform((2, 10, 10, 10, 3), dtype="float32")
    layer3d(x3d, training=True)
