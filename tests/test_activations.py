"""Tests for zero_keras activations."""

import numpy as np
import pytest
import keras

from zero_keras import activations
from .utils import assert_allclose_keras_zero, set_seed


@pytest.fixture(autouse=True)
def _set_seed():
    set_seed(42)


def get_x():
    return np.array([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0], dtype=np.float32)


def get_x_glu():
    return np.array([[-1.0, 1.0, 2.0, -2.0], [0.0, 2.0, -1.0, 3.0]], dtype=np.float32)


def test_activation_celu():
    x = get_x()
    assert_allclose_keras_zero(keras.activations.celu(x), activations.celu(x))
    assert_allclose_keras_zero(
        keras.activations.celu(x, alpha=0.5), activations.celu(x, alpha=0.5)
    )


def test_activation_elu():
    x = get_x()
    assert_allclose_keras_zero(keras.activations.elu(x), activations.elu(x))
    assert_allclose_keras_zero(
        keras.activations.elu(x, alpha=0.5), activations.elu(x, alpha=0.5)
    )


def test_activation_exponential():
    x = get_x()
    assert_allclose_keras_zero(
        keras.activations.exponential(x), activations.exponential(x)
    )


def test_activation_gelu():
    x = get_x()
    assert_allclose_keras_zero(keras.activations.gelu(x), activations.gelu(x))
    assert_allclose_keras_zero(
        keras.activations.gelu(x, approximate=True),
        activations.gelu(x, approximate=True),
    )


def test_activation_glu():
    x = get_x_glu()
    assert_allclose_keras_zero(keras.activations.glu(x), activations.glu(x))
    assert_allclose_keras_zero(
        keras.activations.glu(x, axis=1), activations.glu(x, axis=1)
    )


def test_activation_hard_shrink():
    x = get_x()
    assert_allclose_keras_zero(
        keras.activations.hard_shrink(x), activations.hard_shrink(x)
    )
    assert_allclose_keras_zero(
        keras.activations.hard_shrink(x, threshold=0.7),
        activations.hard_shrink(x, threshold=0.7),
    )


def test_activation_hard_sigmoid():
    x = get_x()
    assert_allclose_keras_zero(
        keras.activations.hard_sigmoid(x), activations.hard_sigmoid(x)
    )


def test_activation_hard_silu():
    x = get_x()
    assert_allclose_keras_zero(keras.activations.hard_silu(x), activations.hard_silu(x))


def test_activation_hard_swish():
    x = get_x()
    assert_allclose_keras_zero(
        keras.activations.hard_swish(x), activations.hard_swish(x)
    )


def test_activation_hard_tanh():
    x = get_x()
    assert_allclose_keras_zero(keras.activations.hard_tanh(x), activations.hard_tanh(x))


def test_activation_leaky_relu():
    x = get_x()
    assert_allclose_keras_zero(
        keras.activations.leaky_relu(x), activations.leaky_relu(x)
    )
    assert_allclose_keras_zero(
        keras.activations.leaky_relu(x, negative_slope=0.3),
        activations.leaky_relu(x, negative_slope=0.3),
    )


def test_activation_linear():
    x = get_x()
    assert_allclose_keras_zero(keras.activations.linear(x), activations.linear(x))


def test_activation_log_sigmoid():
    x = get_x()
    assert_allclose_keras_zero(
        keras.activations.log_sigmoid(x), activations.log_sigmoid(x)
    )


def test_activation_log_softmax():
    x = get_x_glu()
    assert_allclose_keras_zero(
        keras.activations.log_softmax(x), activations.log_softmax(x)
    )
    assert_allclose_keras_zero(
        keras.activations.log_softmax(x, axis=0), activations.log_softmax(x, axis=0)
    )


def test_activation_mish():
    x = get_x()
    assert_allclose_keras_zero(keras.activations.mish(x), activations.mish(x))


def test_activation_relu():
    x = get_x()
    assert_allclose_keras_zero(keras.activations.relu(x), activations.relu(x))
    assert_allclose_keras_zero(
        keras.activations.relu(x, negative_slope=0.1),
        activations.relu(x, negative_slope=0.1),
    )
    assert_allclose_keras_zero(
        keras.activations.relu(x, max_value=1.0), activations.relu(x, max_value=1.0)
    )
    assert_allclose_keras_zero(
        keras.activations.relu(x, threshold=0.5), activations.relu(x, threshold=0.5)
    )


def test_activation_relu6():
    x = get_x()
    assert_allclose_keras_zero(keras.activations.relu6(x), activations.relu6(x))


def test_activation_selu():
    x = get_x()
    assert_allclose_keras_zero(keras.activations.selu(x), activations.selu(x))


def test_activation_sigmoid():
    x = get_x()
    assert_allclose_keras_zero(keras.activations.sigmoid(x), activations.sigmoid(x))


def test_activation_silu():
    x = get_x()
    assert_allclose_keras_zero(keras.activations.silu(x), activations.silu(x))


def test_activation_soft_shrink():
    x = get_x()
    assert_allclose_keras_zero(
        keras.activations.soft_shrink(x), activations.soft_shrink(x)
    )
    assert_allclose_keras_zero(
        keras.activations.soft_shrink(x, threshold=0.2),
        activations.soft_shrink(x, threshold=0.2),
    )


def test_activation_softmax():
    x = get_x_glu()
    assert_allclose_keras_zero(keras.activations.softmax(x), activations.softmax(x))
    assert_allclose_keras_zero(
        keras.activations.softmax(x, axis=0), activations.softmax(x, axis=0)
    )


def test_activation_softplus():
    x = get_x()
    assert_allclose_keras_zero(keras.activations.softplus(x), activations.softplus(x))


def test_activation_softsign():
    x = get_x()
    assert_allclose_keras_zero(keras.activations.softsign(x), activations.softsign(x))


def test_activation_sparse_plus():
    x = get_x()
    assert_allclose_keras_zero(
        keras.activations.sparse_plus(x), activations.sparse_plus(x)
    )


def test_activation_sparse_sigmoid():
    x = get_x()
    assert_allclose_keras_zero(
        keras.activations.sparse_sigmoid(x), activations.sparse_sigmoid(x)
    )


def test_activation_sparsemax():
    x = get_x_glu()
    assert_allclose_keras_zero(keras.activations.sparsemax(x), activations.sparsemax(x))
    assert_allclose_keras_zero(
        keras.activations.sparsemax(x, axis=0), activations.sparsemax(x, axis=0)
    )


def test_activation_squareplus():
    x = get_x()
    assert_allclose_keras_zero(
        keras.activations.squareplus(x), activations.squareplus(x)
    )
    assert_allclose_keras_zero(
        keras.activations.squareplus(x, b=3), activations.squareplus(x, b=3)
    )


def test_activation_swish():
    x = get_x()
    assert_allclose_keras_zero(keras.activations.swish(x), activations.swish(x))


def test_activation_tanh():
    x = get_x()
    assert_allclose_keras_zero(keras.activations.tanh(x), activations.tanh(x))


def test_activation_tanh_shrink():
    x = get_x()
    assert_allclose_keras_zero(
        keras.activations.tanh_shrink(x), activations.tanh_shrink(x)
    )


def test_activation_threshold():
    x = get_x()
    assert_allclose_keras_zero(
        keras.activations.threshold(x, threshold=0.5, default_value=-1.0),
        activations.threshold(x, threshold=0.5, default_value=-1.0),
    )


def test_deserialize():
    assert activations.deserialize("relu") == activations.relu
    assert activations.deserialize({"class_name": "relu"}) == {"class_name": "relu"}


def test_serialize():
    assert activations.serialize(activations.relu) == "relu"
    assert activations.serialize("relu") == "relu"


def test_get():
    assert activations.get(None) == activations.linear
    assert activations.get("relu") == activations.relu
    assert activations.get("nonexistent") == activations.linear
    assert activations.get(activations.celu) == activations.celu
    assert activations.get(123) == activations.linear
