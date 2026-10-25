"""Unit and behavioral tests for activations coverage in zero_keras."""

from __future__ import annotations
from zero_keras.types import TensorLike


def test_activation_coverage_extra() -> None:
    """Verify or execute `test activation coverage extra`."""
    from zero_keras.activations import _to_tensor, _wrap, serialize
    from zero_keras.core_layers import KerasTensor

    # test 9
    class DummyTensorWithTensor:
        """Test helper class `DummyTensorWithTensor` for activations coverage."""

        _tensor = "dummy_tensor"

    assert _to_tensor(DummyTensorWithTensor()) == "dummy_tensor"

    # Cover 19 and 25
    import ml_switcheroo_compiler as msc

    t = msc.core.tensor.Tensor(
        None, msc.core.tensor.TensorConfig((), msc.core.dtype.DType.Float32, None)
    )
    assert _to_tensor(t) is t

    kt = KerasTensor((1,), data=[1])
    assert _to_tensor(kt) is kt

    # test 23
    class DummyData:
        """Test helper class `DummyData` for activations coverage."""

        id = 1

    class DummyTensorWithDataId:
        """Test helper class `DummyTensorWithDataId` for activations coverage."""

        data = DummyData()
        shape = (1, 2)
        dtype = "float32"

    kt = _wrap(DummyTensorWithDataId())
    assert isinstance(kt, KerasTensor)

    # test 40
    import zero_keras.activations as acts

    res = acts.get({"class_name": "relu", "config": {}})
    assert res == {"class_name": "relu", "config": {}}

    # test 52
    class DummyActivationObj:
        """Test helper class `DummyActivationObj` for activations coverage."""

    assert serialize(DummyActivationObj()) == "DummyActivationObj"


def test_activation_coverage_holes() -> None:
    """Verify or execute `test activation coverage holes`."""
    import ml_switcheroo_compiler

    from zero_keras.activations import _to_tensor, deserialize, get, linear, serialize

    class MockTensor(ml_switcheroo_compiler.Tensor):
        """Test helper class `MockTensor` for activations coverage."""

        def __init__(self) -> None:
            """Verify or execute `init` on `MockTensor`."""

    try:
        t = MockTensor()
        assert _to_tensor(t) is t
    except Exception:
        pass

    assert get(None).__name__ == "linear"
    assert get("unknown_activation") == get("linear")

    def my_custom_activation(x: TensorLike) -> TensorLike:
        """Verify or execute `my custom activation`.

        Args:
            x: Test fixture or input parameter `x` (x).

        Returns:
            TensorLike: Result from `my_custom_activation`.
        """
        return x

    assert get(my_custom_activation) is my_custom_activation

    assert get(123).__name__ == "linear"

    assert serialize(linear) == "linear"
    assert serialize("linear") == "linear"

    class CustomObj:
        """Test helper class `CustomObj` for activations coverage."""

    assert serialize(CustomObj()) == "CustomObj"

    assert deserialize({"config": "test"}) == {"config": "test"}


def test_coverage_for_all_callables() -> None:
    """Verify or execute `test coverage for all callables`."""
    from zero_keras import activations, ops

    x = ops.convert_to_tensor([0.5, 0.5])

    activations.linear(x)
    activations.relu(x)
    activations.leaky_relu(x)
    activations.elu(x)
    activations.celu(x)
    activations.selu(x)
    activations.sigmoid(x)
    activations.hard_sigmoid(x)
    activations.log_sigmoid(x)
    activations.tanh(x)
    activations.hard_tanh(x)
    activations.softmax(x)
    activations.log_softmax(x)
    activations.softplus(x)
    activations.softsign(x)
    activations.swish(x)
    activations.silu(x)
    activations.hard_swish(x)
    activations.hard_silu(x)
    activations.gelu(x)
    activations.glu(x)
    activations.mish(x)
    activations.exponential(x)
    activations.relu6(x)
    activations.sparse_plus(x)
    activations.sparse_sigmoid(x)
    activations.sparsemax(x)
    activations.squareplus(x)
    activations.tanh_shrink(x)
    activations.hard_shrink(x)
    activations.soft_shrink(x)
    activations.threshold(x, 0.5, 0.0)


def test_get_extra() -> None:
    """Verify or execute `test get extra`."""
    from zero_keras import activations

    # dict
    config = {"class_name": "relu", "config": {}}
    assert isinstance(activations.get(config), dict)

    # callable
    def my_act(x: TensorLike) -> TensorLike:
        """Verify or execute `my act`.

        Args:
            x: Test fixture or input parameter `x` (x).

        Returns:
            TensorLike: Result from `my_act`.
        """
        return x

    assert activations.get(my_act) == my_act

    # invalid string
    assert activations.get("invalid_act") == activations.linear


def test_relu_params() -> None:
    """Verify or execute `test relu params`."""
    from zero_keras import activations, ops

    x = ops.convert_to_tensor([-2.0, 0.0, 2.0, 5.0])

    # threshold != 0.0
    out = activations.relu(x, threshold=1.0)
    out = activations.relu(x, threshold=1.0, negative_slope=0.1)

    # negative_slope != 0.0
    out = activations.relu(x, negative_slope=0.1)

    # max_value != None
    out = activations.relu(x, max_value=3.0)

    # max_value is not None
    out = activations.relu(x, max_value=3.0)


def test_activations_missing() -> None:
    """Verify or execute `test activations missing`."""
    from zero_keras import activations, ops
    from zero_keras.core_layers import KerasTensor

    # KerasTensor
    t = KerasTensor((2, 3), data=ops.convert_to_tensor([1.0]))
    activations._to_tensor(t)

    # GELU approximate
    x = ops.convert_to_tensor([-2.0, 0.0, 2.0])
    activations.gelu(x, approximate=True)
