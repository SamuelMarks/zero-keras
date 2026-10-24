"""Module `test_regularization_extra` for zero_keras."""

from __future__ import annotations
from zero_keras.types import TensorLike
import numpy as np

from zero_keras import layers


def test_activity_regularization_l1_l2() -> TensorLike:
    """Execute ``test_activity_regularization_l1_l2``."""
    layer = layers.ActivityRegularization(l1=0.1, l2=0.2)
    x = np.random.rand(2, 5).astype(np.float32)
    out = layer(x)
    assert out.shape == (2, 5)


def test_gaussian_dropout() -> TensorLike:
    """Execute ``test_gaussian_dropout``."""
    layer = layers.GaussianDropout(rate=0.2)
    x = np.random.rand(2, 5).astype(np.float32)
    out = layer(x)
    assert out.shape == (2, 5)


def test_gaussian_noise() -> TensorLike:
    """Execute ``test_gaussian_noise``."""
    layer = layers.GaussianNoise(stddev=0.1)
    x = np.random.rand(2, 5).astype(np.float32)
    out = layer(x)
    assert out.shape == (2, 5)


def test_spatial_dropout_not_training() -> TensorLike:
    """Execute ``test_spatial_dropout_not_training``."""
    layer1d = layers.SpatialDropout1D(rate=0.2)
    x1d = np.random.rand(2, 10, 3).astype(np.float32)
    assert layer1d(x1d, training=False).shape == (2, 10, 3)

    layer2d = layers.SpatialDropout2D(rate=0.2)
    x2d = np.random.rand(2, 10, 10, 3).astype(np.float32)
    assert layer2d(x2d, training=False).shape == (2, 10, 10, 3)

    layer3d = layers.SpatialDropout3D(rate=0.2)
    x3d = np.random.rand(2, 10, 10, 10, 3).astype(np.float32)
    assert layer3d(x3d, training=False).shape == (2, 10, 10, 10, 3)


def test_spatial_dropout_training() -> TensorLike:
    """Execute ``test_spatial_dropout_training``."""
    layer1d = layers.SpatialDropout1D(rate=0.2)
    x1d = np.random.rand(2, 10, 3).astype(np.float32)
    layer1d(x1d, training=True)

    layer2d = layers.SpatialDropout2D(rate=0.2)
    x2d = np.random.rand(2, 10, 10, 3).astype(np.float32)
    layer2d(x2d, training=True)

    layer3d = layers.SpatialDropout3D(rate=0.2)
    x3d = np.random.rand(2, 10, 10, 10, 3).astype(np.float32)
    layer3d(x3d, training=True)
