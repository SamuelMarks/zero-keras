"""Parity tests for activation layers in zero_keras vs Keras."""

from __future__ import annotations

import keras
import numpy as np
import pytest

from zero_keras import layers
from zero_keras.types import TensorLike

from .utils import assert_allclose_keras_zero, set_seed


@pytest.fixture(autouse=True)
def _set_seed() -> None:
    """Seed random number generators before each parity test."""
    set_seed(42)


def get_x() -> TensorLike:
    """Return a 2D float32 sample array for activation layer parity tests.

    Returns:
        TensorLike: Sample 2D input array.
    """
    return np.array([[-3.0, -1.0, 0.0], [0.5, 1.0, 3.0]], dtype=np.float32)


def test_layer_Activation() -> None:
    """Verify ``layers.Activation`` parity with ``keras.layers.Activation``."""
    x = get_x()
    keras_layer = keras.layers.Activation("relu")
    zero_layer = layers.Activation("relu")
    assert_allclose_keras_zero(keras_layer(x), zero_layer(x))

    keras_layer = keras.layers.Activation("swish")
    zero_layer = layers.Activation("swish")
    assert_allclose_keras_zero(keras_layer(x), zero_layer(x))


def test_layer_ELU() -> None:
    """Verify ``layers.ELU`` parity with ``keras.layers.ELU``."""
    x = get_x()
    keras_layer = keras.layers.ELU(alpha=0.5)
    zero_layer = layers.ELU(alpha=0.5)
    assert_allclose_keras_zero(keras_layer(x), zero_layer(x))


def test_layer_LeakyReLU() -> None:
    """Verify ``layers.LeakyReLU`` parity with ``keras.layers.LeakyReLU``."""
    x = get_x()
    keras_layer = keras.layers.LeakyReLU(negative_slope=0.2)
    zero_layer = layers.LeakyReLU(negative_slope=0.2)
    assert_allclose_keras_zero(keras_layer(x), zero_layer(x))


def test_layer_PReLU() -> None:
    """Verify ``layers.PReLU`` parity with ``keras.layers.PReLU``."""
    x = get_x()
    keras_layer = keras.layers.PReLU(alpha_initializer="ones")
    zero_layer = layers.PReLU(alpha_initializer="ones")
    assert_allclose_keras_zero(keras_layer(x), zero_layer(x))

    keras_layer2 = keras.layers.PReLU()
    zero_layer2 = layers.PReLU()
    assert_allclose_keras_zero(keras_layer2(x), zero_layer2(x))


def test_layer_ReLU() -> None:
    """Verify ``layers.ReLU`` parity with ``keras.layers.ReLU``."""
    x = get_x()
    keras_layer = keras.layers.ReLU(max_value=1.0, negative_slope=0.1, threshold=0.5)
    zero_layer = layers.ReLU(max_value=1.0, negative_slope=0.1, threshold=0.5)
    assert_allclose_keras_zero(keras_layer(x), zero_layer(x))


def test_layer_Softmax() -> None:
    """Verify ``layers.Softmax`` parity with ``keras.layers.Softmax``."""
    x = get_x()
    keras_layer = keras.layers.Softmax(axis=-1)
    zero_layer = layers.Softmax(axis=-1)
    assert_allclose_keras_zero(keras_layer(x), zero_layer(x))

    keras_layer2 = keras.layers.Softmax(axis=0)
    zero_layer2 = layers.Softmax(axis=0)
    assert_allclose_keras_zero(keras_layer2(x), zero_layer2(x))
