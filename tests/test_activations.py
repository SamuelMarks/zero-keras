"""Unit and behavioral tests for zero_keras.activations without third-party dependencies."""

from __future__ import annotations

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

from zero_keras import activations, ops
from zero_keras.types import TensorLike

from .conftest import assert_allclose_zero


def get_x() -> TensorLike:
    """Return a 1D float32 sample tensor for activation tests.

    Returns:
        TensorLike: Sample 1D input tensor.
    """
    return ops.convert_to_tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0], dtype="float32")


def get_x_glu() -> TensorLike:
    """Return a 2D float32 sample tensor for GLU and softmax tests.

    Returns:
        TensorLike: Sample 2D input tensor.
    """
    return ops.convert_to_tensor([[-1.0, 1.0, 2.0, -2.0], [0.0, 2.0, -1.0, 3.0]], dtype="float32")


def test_all_activations_unit() -> None:
    """Execute all activation functions in ``zero_keras.activations`` on eager tensors."""
    x = get_x()
    x_2d = get_x_glu()

    assert activations.celu(x).shape == (7,)
    assert activations.celu(x, alpha=0.5).shape == (7,)
    assert activations.elu(x).shape == (7,)
    assert activations.elu(x, alpha=0.5).shape == (7,)
    assert activations.exponential(x).shape == (7,)
    assert activations.gelu(x).shape == (7,)
    assert activations.gelu(x, approximate=True).shape == (7,)
    assert activations.glu(x_2d).shape == (2, 2)
    assert activations.glu(x_2d, axis=1).shape == (2, 2)
    assert activations.hard_shrink(x).shape == (7,)
    assert activations.hard_shrink(x, threshold=0.7).shape == (7,)
    assert activations.hard_sigmoid(x).shape == (7,)
    assert activations.hard_silu(x).shape == (7,)
    assert activations.hard_swish(x).shape == (7,)
    assert activations.hard_tanh(x).shape == (7,)
    assert activations.leaky_relu(x).shape == (7,)
    assert activations.leaky_relu(x, negative_slope=0.3).shape == (7,)
    assert_allclose_zero(activations.linear(x), x)
    assert activations.log_sigmoid(x).shape == (7,)
    assert activations.log_softmax(x_2d).shape == (2, 4)
    assert activations.log_softmax(x_2d, axis=0).shape == (2, 4)
    assert activations.mish(x).shape == (7,)
    assert activations.relu(x).shape == (7,)
    assert activations.relu(x, negative_slope=0.1).shape == (7,)
    assert activations.relu(x, max_value=1.0).shape == (7,)
    assert activations.relu(x, threshold=0.5).shape == (7,)
    assert activations.relu6(x).shape == (7,)
    assert activations.selu(x).shape == (7,)
    assert activations.sigmoid(x).shape == (7,)
    assert activations.silu(x).shape == (7,)
    assert activations.soft_shrink(x).shape == (7,)
    assert activations.soft_shrink(x, threshold=0.2).shape == (7,)
    assert activations.softmax(x_2d).shape == (2, 4)
    assert activations.softmax(x_2d, axis=0).shape == (2, 4)
    assert activations.softplus(x).shape == (7,)
    assert activations.softsign(x).shape == (7,)
    assert activations.sparse_plus(x).shape == (7,)
    assert activations.sparse_sigmoid(x).shape == (7,)
    assert activations.sparsemax(x_2d).shape == (2, 4)
    assert activations.sparsemax(x_2d, axis=0).shape == (2, 4)
    assert activations.squareplus(x).shape == (7,)
    assert activations.squareplus(x, b=3).shape == (7,)
    assert activations.swish(x).shape == (7,)
    assert activations.tanh(x).shape == (7,)
    assert activations.tanh_shrink(x).shape == (7,)
    assert activations.threshold(x, threshold=0.5, default_value=-1.0).shape == (7,)


def test_deserialize() -> None:
    """Verify ``activations.deserialize`` for string and dict configs."""
    assert activations.deserialize("relu") == activations.relu
    assert activations.deserialize({"class_name": "relu"}) == {"class_name": "relu"}


def test_serialize() -> None:
    """Verify ``activations.serialize`` for callables and strings."""
    assert activations.serialize(activations.relu) == "relu"
    assert activations.serialize("relu") == "relu"


def test_get() -> None:
    """Verify ``activations.get`` across None, valid names, unknown names, and callables."""
    assert activations.get(None) == activations.linear
    assert activations.get("relu") == activations.relu
    assert activations.get("nonexistent") == activations.linear
    assert activations.get(activations.celu) == activations.celu
    assert activations.get(123) == activations.linear


def test_activations_coverage() -> None:
    """Cover internal helper branches in ``zero_keras.activations``."""

    class MockKerasTensor:
        """Stub with ``data`` attribute."""

        data = "mock_data"

    x1 = MockKerasTensor()
    activations._to_tensor(x1)

    config = TensorConfig(shape=(), dtype=DType.Float32, device="cpu")
    x2 = Tensor("mock", config)
    activations._to_tensor(x2)

    class MockTensorWithUnderTensor:
        """Stub with ``_tensor`` attribute."""

        _tensor = "mock_tensor"

    x3 = MockTensorWithUnderTensor()
    activations._to_tensor(x3)

    class MockDataWithId:
        """Stub data object with symbolic ``id``."""

        id = "123"

    class MockBackendTensor:
        """Stub backend tensor with symbolic data."""

        data = MockDataWithId()
        shape = ()
        dtype = "float32"

    activations._wrap(MockBackendTensor())

    assert activations.get(None) == activations.linear
    assert activations.get("not_an_activation") == activations.linear
    assert activations.serialize("relu") == "relu"

    class MockActivationWithoutName:
        """Callable stub without ``__name__``."""

    assert activations.serialize(MockActivationWithoutName()) == "MockActivationWithoutName"
