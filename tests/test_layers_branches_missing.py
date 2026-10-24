"""Module docstring."""

from __future__ import annotations
from zero_keras.types import TensorLike

import numpy as np
import pytest
from ml_switcheroo_compiler.core import config

from zero_keras import core_layers, layers


def test_dense_activation_none() -> TensorLike:
    """Function docstring."""
    l = layers.Dense(10, activation=None)
    l(np.random.rand(2, 5))


def test_activity_reg_zeros() -> TensorLike:
    """Function docstring."""
    l = layers.ActivityRegularization(l1=0.0, l2=0.0)
    l(np.random.rand(2, 5))


def test_conv_activation_none() -> TensorLike:
    """Function docstring."""
    # Conv1D
    l = layers.Conv1D(2, 2, activation=None)
    l(np.random.rand(2, 5, 3))

    # Conv1DTranspose
    l = layers.Conv1DTranspose(2, 2, activation=None)
    l(np.random.rand(2, 5, 3))

    # Conv2D
    l = layers.Conv2D(2, 2, activation=None)
    l(np.random.rand(2, 5, 5, 3))

    # Conv2DTranspose
    l = layers.Conv2DTranspose(2, 2, activation=None)
    l(np.random.rand(2, 5, 5, 3))

    # Conv3D
    l = layers.Conv3D(2, 2, activation=None)
    l(np.random.rand(2, 5, 5, 5, 3))

    # Conv3DTranspose
    l = layers.Conv3DTranspose(2, 2, activation=None)
    l(np.random.rand(2, 5, 5, 5, 3))


def test_kerastensor_bool_eager_false() -> TensorLike:
    """Function docstring."""
    t = core_layers.KerasTensor((2,))
    t.data = True

    old_eager = config.eager_mode if hasattr(config, "eager_mode") else False
    config.eager_mode = False
    try:
        with pytest.raises(TypeError):
            bool(t)
    finally:
        config.eager_mode = old_eager
