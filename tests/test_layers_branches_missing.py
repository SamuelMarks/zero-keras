import pytest
import numpy as np
import zero_keras.layers as layers
import zero_keras.core_layers as core_layers
import ml_switcheroo_compiler.core.config as config


def test_dense_activation_none():
    l = layers.Dense(10, activation=None)
    l(np.random.rand(2, 5))


def test_activity_reg_zeros():
    l = layers.ActivityRegularization(l1=0.0, l2=0.0)
    l(np.random.rand(2, 5))


def test_conv_activation_none():
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


def test_kerastensor_bool_eager_false():
    t = core_layers.KerasTensor((2,))
    t.data = True

    old_eager = getattr(config, "eager_mode", False)
    config.eager_mode = False
    try:
        with pytest.raises(TypeError):
            bool(t)
    finally:
        config.eager_mode = old_eager
