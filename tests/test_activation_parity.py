"""Tests for zero_keras activations parity with Keras."""

from __future__ import annotations

import keras
import numpy as np
import pytest

from zero_keras import activations
from zero_keras.types import TensorLike

from .utils import assert_allclose_keras_zero, set_seed


@pytest.fixture(autouse=True)
def _set_seed() -> None:
    """Seed random number generators before each parity test."""
    set_seed(42)


def get_x() -> TensorLike:
    """Return a 1D float32 sample array for activation parity tests.

    Returns:
        TensorLike: Sample 1D input array.
    """
    return np.array([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0], dtype=np.float32)


def get_x_glu() -> TensorLike:
    """Return a 2D float32 sample array for GLU and softmax parity tests.

    Returns:
        TensorLike: Sample 2D input array.
    """
    return np.array([[-1.0, 1.0, 2.0, -2.0], [0.0, 2.0, -1.0, 3.0]], dtype=np.float32)


def test_activation_celu() -> None:
    """Verify ``celu`` parity with ``keras.activations.celu``."""
    x = get_x()
    assert_allclose_keras_zero(keras.activations.celu(x), activations.celu(x))
    assert_allclose_keras_zero(keras.activations.celu(x, alpha=0.5), activations.celu(x, alpha=0.5))


def test_activation_elu() -> None:
    """Verify ``elu`` parity with ``keras.activations.elu``."""
    x = get_x()
    assert_allclose_keras_zero(keras.activations.elu(x), activations.elu(x))
    assert_allclose_keras_zero(keras.activations.elu(x, alpha=0.5), activations.elu(x, alpha=0.5))


def test_activation_exponential() -> None:
    """Verify ``exponential`` parity with ``keras.activations.exponential``."""
    x = get_x()
    assert_allclose_keras_zero(keras.activations.exponential(x), activations.exponential(x))


def test_activation_gelu() -> None:
    """Verify ``gelu`` parity with ``keras.activations.gelu``."""
    x = get_x()
    assert_allclose_keras_zero(keras.activations.gelu(x), activations.gelu(x))
    assert_allclose_keras_zero(
        keras.activations.gelu(x, approximate=True),
        activations.gelu(x, approximate=True),
    )


def test_activation_glu() -> None:
    """Verify ``glu`` parity with ``keras.activations.glu``."""
    x = get_x_glu()
    assert_allclose_keras_zero(keras.activations.glu(x), activations.glu(x))
    assert_allclose_keras_zero(keras.activations.glu(x, axis=1), activations.glu(x, axis=1))


def test_activation_hard_shrink() -> None:
    """Verify ``hard_shrink`` parity with ``keras.activations.hard_shrink``."""
    x = get_x()
    assert_allclose_keras_zero(keras.activations.hard_shrink(x), activations.hard_shrink(x))
    assert_allclose_keras_zero(
        keras.activations.hard_shrink(x, threshold=0.7),
        activations.hard_shrink(x, threshold=0.7),
    )


def test_activation_hard_sigmoid() -> None:
    """Verify ``hard_sigmoid`` parity with ``keras.activations.hard_sigmoid``."""
    x = get_x()
    assert_allclose_keras_zero(keras.activations.hard_sigmoid(x), activations.hard_sigmoid(x))


def test_activation_hard_silu() -> None:
    """Verify ``hard_silu`` parity with ``keras.activations.hard_silu``."""
    x = get_x()
    assert_allclose_keras_zero(keras.activations.hard_silu(x), activations.hard_silu(x))


def test_activation_hard_swish() -> None:
    """Verify ``hard_swish`` parity with ``keras.activations.hard_swish``."""
    x = get_x()
    assert_allclose_keras_zero(keras.activations.hard_swish(x), activations.hard_swish(x))


def test_activation_hard_tanh() -> None:
    """Verify ``hard_tanh`` parity with ``keras.activations.hard_tanh``."""
    x = get_x()
    assert_allclose_keras_zero(keras.activations.hard_tanh(x), activations.hard_tanh(x))


def test_activation_leaky_relu() -> None:
    """Verify ``leaky_relu`` parity with ``keras.activations.leaky_relu``."""
    x = get_x()
    assert_allclose_keras_zero(keras.activations.leaky_relu(x), activations.leaky_relu(x))
    assert_allclose_keras_zero(
        keras.activations.leaky_relu(x, negative_slope=0.3),
        activations.leaky_relu(x, negative_slope=0.3),
    )


def test_activation_linear() -> None:
    """Verify ``linear`` parity with ``keras.activations.linear``."""
    x = get_x()
    assert_allclose_keras_zero(keras.activations.linear(x), activations.linear(x))


def test_activation_log_sigmoid() -> None:
    """Verify ``log_sigmoid`` parity with ``keras.activations.log_sigmoid``."""
    x = get_x()
    assert_allclose_keras_zero(keras.activations.log_sigmoid(x), activations.log_sigmoid(x))


def test_activation_log_softmax() -> None:
    """Verify ``log_softmax`` parity with ``keras.activations.log_softmax``."""
    x = get_x_glu()
    assert_allclose_keras_zero(keras.activations.log_softmax(x), activations.log_softmax(x))
    assert_allclose_keras_zero(
        keras.activations.log_softmax(x, axis=0), activations.log_softmax(x, axis=0)
    )


def test_activation_mish() -> None:
    """Verify ``mish`` parity with ``keras.activations.mish``."""
    x = get_x()
    assert_allclose_keras_zero(keras.activations.mish(x), activations.mish(x))


def test_activation_relu() -> None:
    """Verify ``relu`` parity with ``keras.activations.relu``."""
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


def test_activation_relu6() -> None:
    """Verify ``relu6`` parity with ``keras.activations.relu6``."""
    x = get_x()
    assert_allclose_keras_zero(keras.activations.relu6(x), activations.relu6(x))


def test_activation_selu() -> None:
    """Verify ``selu`` parity with ``keras.activations.selu``."""
    x = get_x()
    assert_allclose_keras_zero(keras.activations.selu(x), activations.selu(x))


def test_activation_sigmoid() -> None:
    """Verify ``sigmoid`` parity with ``keras.activations.sigmoid``."""
    x = get_x()
    assert_allclose_keras_zero(keras.activations.sigmoid(x), activations.sigmoid(x))


def test_activation_silu() -> None:
    """Verify ``silu`` parity with ``keras.activations.silu``."""
    x = get_x()
    assert_allclose_keras_zero(keras.activations.silu(x), activations.silu(x))


def test_activation_soft_shrink() -> None:
    """Verify ``soft_shrink`` parity with ``keras.activations.soft_shrink``."""
    x = get_x()
    assert_allclose_keras_zero(keras.activations.soft_shrink(x), activations.soft_shrink(x))
    assert_allclose_keras_zero(
        keras.activations.soft_shrink(x, threshold=0.2),
        activations.soft_shrink(x, threshold=0.2),
    )


def test_activation_softmax() -> None:
    """Verify ``softmax`` parity with ``keras.activations.softmax``."""
    x = get_x_glu()
    assert_allclose_keras_zero(keras.activations.softmax(x), activations.softmax(x))
    assert_allclose_keras_zero(keras.activations.softmax(x, axis=0), activations.softmax(x, axis=0))


def test_activation_softplus() -> None:
    """Verify ``softplus`` parity with ``keras.activations.softplus``."""
    x = get_x()
    assert_allclose_keras_zero(keras.activations.softplus(x), activations.softplus(x))


def test_activation_softsign() -> None:
    """Verify ``softsign`` parity with ``keras.activations.softsign``."""
    x = get_x()
    assert_allclose_keras_zero(keras.activations.softsign(x), activations.softsign(x))


def test_activation_sparse_plus() -> None:
    """Verify ``sparse_plus`` parity with ``keras.activations.sparse_plus``."""
    x = get_x()
    assert_allclose_keras_zero(keras.activations.sparse_plus(x), activations.sparse_plus(x))


def test_activation_sparse_sigmoid() -> None:
    """Verify ``sparse_sigmoid`` parity with ``keras.activations.sparse_sigmoid``."""
    x = get_x()
    assert_allclose_keras_zero(keras.activations.sparse_sigmoid(x), activations.sparse_sigmoid(x))


def test_activation_sparsemax() -> None:
    """Verify ``sparsemax`` parity with ``keras.activations.sparsemax``."""
    x = get_x_glu()
    assert_allclose_keras_zero(keras.activations.sparsemax(x), activations.sparsemax(x))
    assert_allclose_keras_zero(
        keras.activations.sparsemax(x, axis=0), activations.sparsemax(x, axis=0)
    )


def test_activation_squareplus() -> None:
    """Verify ``squareplus`` parity with ``keras.activations.squareplus``."""
    x = get_x()
    assert_allclose_keras_zero(keras.activations.squareplus(x), activations.squareplus(x))
    assert_allclose_keras_zero(keras.activations.squareplus(x, b=3), activations.squareplus(x, b=3))


def test_activation_swish() -> None:
    """Verify ``swish`` parity with ``keras.activations.swish``."""
    x = get_x()
    assert_allclose_keras_zero(keras.activations.swish(x), activations.swish(x))


def test_activation_tanh() -> None:
    """Verify ``tanh`` parity with ``keras.activations.tanh``."""
    x = get_x()
    assert_allclose_keras_zero(keras.activations.tanh(x), activations.tanh(x))


def test_activation_tanh_shrink() -> None:
    """Verify ``tanh_shrink`` parity with ``keras.activations.tanh_shrink``."""
    x = get_x()
    assert_allclose_keras_zero(keras.activations.tanh_shrink(x), activations.tanh_shrink(x))


def test_activation_threshold() -> None:
    """Verify ``threshold`` parity with ``keras.activations.threshold``."""
    x = get_x()
    assert_allclose_keras_zero(
        keras.activations.threshold(x, threshold=0.5, default_value=-1.0),
        activations.threshold(x, threshold=0.5, default_value=-1.0),
    )
