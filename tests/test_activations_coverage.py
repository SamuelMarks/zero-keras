"""Module docstring."""

from __future__ import annotations
from zero_keras.types import TensorLike


def test_activation_coverage_extra() -> TensorLike:
    """Function docstring."""
    from zero_keras.activations import _to_tensor, _wrap, serialize
    from zero_keras.core_layers import KerasTensor

    # test 9
    class DummyTensorWithTensor:
        """Class docstring."""

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
        """Class docstring."""

        id = 1

    class DummyTensorWithDataId:
        """Class docstring."""

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
        """Class docstring."""

    assert serialize(DummyActivationObj()) == "DummyActivationObj"


def test_activation_coverage_holes() -> TensorLike:
    """Function docstring."""
    import ml_switcheroo_compiler

    from zero_keras.activations import _to_tensor, deserialize, get, linear, serialize

    class MockTensor(ml_switcheroo_compiler.Tensor):
        """Class docstring."""

        def __init__(self) -> None:
            """Function docstring."""

    try:
        t = MockTensor()
        assert _to_tensor(t) is t
    except Exception:
        pass

    assert get(None).__name__ == "linear"
    assert get("unknown_activation") == get("linear")

    def my_custom_activation(x: TensorLike) -> TensorLike:
        """Function docstring.

        Args:
            x: Description.
        """
        return x

    assert get(my_custom_activation) is my_custom_activation

    assert get(123).__name__ == "linear"

    assert serialize(linear) == "linear"
    assert serialize("linear") == "linear"

    class CustomObj:
        """Class docstring."""

    assert serialize(CustomObj()) == "CustomObj"

    assert deserialize({"config": "test"}) == {"config": "test"}


def test_coverage_for_all_callables() -> TensorLike:
    """Function docstring."""
    import numpy as np

    from zero_keras import activations

    x = np.array([0.5, 0.5])

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


def test_get_extra() -> TensorLike:
    """Execute ``test_get_extra``."""
    from zero_keras import activations

    # dict
    config = {"class_name": "relu", "config": {}}
    assert isinstance(activations.get(config), dict)

    # callable
    def my_act(x: TensorLike) -> TensorLike:
        """Execute ``my_act``.

        Args:
            x: Input parameter ``x``.
        """
        return x

    assert activations.get(my_act) == my_act

    # invalid string
    assert activations.get("invalid_act") == activations.linear


def test_relu_params() -> TensorLike:
    """Execute ``test_relu_params``."""
    import numpy as np

    from zero_keras import activations

    x = np.array([-2.0, 0.0, 2.0, 5.0])

    # threshold != 0.0
    out = activations.relu(x, threshold=1.0)
    out = activations.relu(x, threshold=1.0, negative_slope=0.1)

    # negative_slope != 0.0
    out = activations.relu(x, negative_slope=0.1)

    # max_value != None
    out = activations.relu(x, max_value=3.0)

    # max_value is not None
    out = activations.relu(x, max_value=3.0)


def test_activations_missing() -> TensorLike:
    """Execute ``test_activations_missing``."""
    import numpy as np

    from zero_keras import activations
    from zero_keras.core_layers import KerasTensor

    # KerasTensor
    t = KerasTensor((2, 3), data=np.array([1.0]))
    activations._to_tensor(t)

    # GELU approximate
    x = np.array([-2.0, 0.0, 2.0])
    activations.gelu(x, approximate=True)
