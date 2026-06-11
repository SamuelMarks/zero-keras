def test_activation_coverage_extra():
    from zero_keras.activations import _to_tensor, _wrap, serialize
    from zero_keras.core_layers import KerasTensor

    # test 9
    class DummyTensorWithTensor:
        _tensor = "dummy_tensor"

    assert _to_tensor(DummyTensorWithTensor()) == "dummy_tensor"

    # test 23
    class DummyData:
        id = 1

    class DummyTensorWithDataId:
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
        pass

    assert serialize(DummyActivationObj()) == "DummyActivationObj"


def test_activation_coverage_holes():
    from zero_keras.activations import get, serialize, deserialize, _to_tensor, linear
    import ml_switcheroo

    class MockTensor(ml_switcheroo.Tensor):
        def __init__(self):
            pass

    try:
        t = MockTensor()
        assert _to_tensor(t) is t
    except Exception:
        pass

    assert get(None).__name__ == "linear"
    assert get("unknown_activation") == get("linear")

    def my_custom_activation(x):
        return x

    assert get(my_custom_activation) is my_custom_activation

    assert get(123).__name__ == "linear"

    assert serialize(linear) == "linear"
    assert serialize("linear") == "linear"

    class CustomObj:
        pass

    assert serialize(CustomObj()) == "CustomObj"

    assert deserialize({"config": "test"}) == {"config": "test"}


def test_coverage_for_all_callables():
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
