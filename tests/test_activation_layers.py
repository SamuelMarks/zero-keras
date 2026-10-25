"""Unit and behavioral tests for activation layers in zero_keras."""

from __future__ import annotations

from zero_keras import activations, layers, ops
from zero_keras.types import TensorLike

from .conftest import assert_allclose_zero


def get_x() -> TensorLike:
    """Return a 2D float32 sample tensor for activation layer unit tests.

    Returns:
        TensorLike: Sample 2D input tensor.
    """
    return ops.convert_to_tensor([[-3.0, -1.0, 0.0], [0.5, 1.0, 3.0]], dtype="float32")


def test_layer_Activation() -> None:
    """Verify ``layers.Activation`` matches functional ``activations``."""
    x = get_x()
    zero_layer = layers.Activation("relu")
    assert_allclose_zero(zero_layer(x), activations.relu(x))

    zero_layer_swish = layers.Activation("swish")
    assert_allclose_zero(zero_layer_swish(x), activations.swish(x))


def test_layer_ELU() -> None:
    """Verify ``layers.ELU`` matches ``activations.elu``."""
    x = get_x()
    zero_layer = layers.ELU(alpha=0.5)
    assert_allclose_zero(zero_layer(x), activations.elu(x, alpha=0.5))


def test_layer_LeakyReLU() -> None:
    """Verify ``layers.LeakyReLU`` matches ``activations.leaky_relu``."""
    x = get_x()
    zero_layer = layers.LeakyReLU(negative_slope=0.2)
    assert_allclose_zero(zero_layer(x), activations.leaky_relu(x, negative_slope=0.2))


def test_layer_PReLU() -> None:
    """Verify ``layers.PReLU`` with ones and default zeros alpha initializer."""
    x = get_x()
    zero_layer = layers.PReLU(alpha_initializer="ones")
    assert_allclose_zero(zero_layer(x), x)

    zero_layer2 = layers.PReLU()
    assert_allclose_zero(zero_layer2(x), activations.relu(x))


def test_layer_ReLU() -> None:
    """Verify ``layers.ReLU`` matches ``activations.relu`` with parameters."""
    x = get_x()
    zero_layer = layers.ReLU(max_value=1.0, negative_slope=0.1, threshold=0.5)
    assert_allclose_zero(
        zero_layer(x),
        activations.relu(x, max_value=1.0, negative_slope=0.1, threshold=0.5),
    )


def test_layer_Softmax() -> None:
    """Verify ``layers.Softmax`` matches ``activations.softmax`` along axes."""
    x = get_x()
    zero_layer = layers.Softmax(axis=-1)
    assert_allclose_zero(zero_layer(x), activations.softmax(x, axis=-1))

    zero_layer2 = layers.Softmax(axis=0)
    assert_allclose_zero(zero_layer2(x), activations.softmax(x, axis=0))
