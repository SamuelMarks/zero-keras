def test_distribution_methods():
    from zero_keras.distribution import (
        DataParallel,
        DeviceMesh,
        LayoutMap,
        ModelParallel,
        TensorLayout,
    )

    dp = DataParallel()
    assert dp.batch_dim_name is None
    assert dp.device_mesh is None
    assert dp.distribute_dataset("dataset") == "dataset"
    assert dp.get_data_layout((1,)) is None
    assert dp.get_tensor_layout((1,)) is None
    assert dp.get_variable_layout(None) is None
    with dp.scope():
        pass

    dm = DeviceMesh()
    assert dm.axis_names is None
    assert dm.backend_mesh is None
    assert dm.devices is None
    assert dm.shape is None

    lm = LayoutMap()
    assert list(iter(lm)) == []
    assert len(lm) == 0
    lm.clear()
    assert dm.devices is None  # wait dm has no device_mesh
    assert lm.device_mesh is None
    assert list(lm.items()) == []
    assert list(lm.keys()) == []
    assert lm.pop("k") is None
    assert lm.popitem() is None
    assert lm.setdefault("k") is None
    lm.update({})
    assert list(lm.values()) == []

    mp = ModelParallel()
    assert mp.batch_dim_name is None
    assert mp.device_mesh is None
    assert mp.distribute_dataset("dataset") == "dataset"
    assert mp.get_data_layout((1,)) is None
    assert mp.get_tensor_layout((1,)) is None
    assert mp.get_variable_layout(None) is None
    with mp.scope():
        pass

    tl = TensorLayout()
    assert tl.axes is None
    assert tl.backend_layout is None
    assert tl.device_mesh is None
