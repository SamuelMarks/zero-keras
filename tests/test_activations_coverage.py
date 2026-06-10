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
