def test_initializers_coverage():
    from zero_keras.initializers import (
        _get_dtype,
        _wrap,
        Orthogonal,
        VarianceScaling,
        get,
    )
    from ml_switcheroo.core.dtype import DType
    from zero_keras.core_layers import KerasTensor

    # 54, 61
    assert _get_dtype(DType.Float32) == DType.Float32
    assert _get_dtype("doesnotexist") == DType.Float32

    # 69
    class DummyData:
        id = 1

    class DummyTensorWithDataId:
        data = DummyData()
        shape = (1, 2)
        dtype = "float32"

    kt = _wrap(DummyTensorWithDataId())
    assert isinstance(kt, KerasTensor)

    # 165
    ortho = Orthogonal()
    res = ortho(shape=(2, 4))
    assert res.shape == (2, 4)

    # 255
    vs = VarianceScaling()
    res = vs(shape=(2, 2, 3, 4))
    assert res.shape == (2, 2, 3, 4)

    # 258
    res = vs(shape=(5,))
    assert res.shape == (5,)
    res = vs(shape=())
    assert res.shape == ()

    # 277-279
    vs_untrunc = VarianceScaling(distribution="untruncated_normal")
    res = vs_untrunc(shape=(2, 2))
    assert res.shape == (2, 2)

    # 477, 497
    assert get(None).__class__.__name__ == "GlorotUniform"

    class DummyObj:
        pass

    obj = DummyObj()
    assert get(obj) is obj
