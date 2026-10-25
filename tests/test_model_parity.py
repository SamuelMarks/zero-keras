"""Tests for zero_keras model parity with Keras."""

from __future__ import annotations

import keras
import numpy as np
import pytest

from zero_keras import core_layers

from .utils import assert_allclose_keras_zero, set_seed


@pytest.fixture(autouse=True)
def _set_seed() -> None:
    """Seed random number generators before each parity test."""
    set_seed(42)


def test_keras_tensor_parity() -> None:
    """Verify ``core_layers.Input`` metadata parity with ``keras.Input``."""
    keras_input = keras.Input(shape=(10,), name="test_in")
    zero_input = core_layers.Input(shape=(10,), name="test_in")

    assert keras_input.shape == zero_input.shape
    assert zero_input.name == "test_in"


def test_model_Sequential_parity() -> None:
    """Verify ``Sequential`` identity pass parity with ``keras.Sequential``."""
    k_seq = keras.Sequential([keras.layers.Identity()])
    z_seq = core_layers.Sequential()
    z_seq.add(core_layers.Layer())

    x = np.random.rand(2, 5).astype(np.float32)
    assert_allclose_keras_zero(k_seq(x), z_seq(x))
