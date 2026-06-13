def test_remaining_coverage():
    from zero_keras.core_layers import KerasTensor, Layer
    import zero_keras.optimizers as optimizers
    import numpy as np

    # test KerasTensor magic methods
    kt = KerasTensor((2, 2), "float32")
    assert (kt - kt).shape == (2, 2)
    assert (kt * kt).shape == (2, 2)
    assert (kt / kt).shape == (2, 2)
    assert (kt**2).shape == (2, 2)

    # 34
    res = kt == 1
    assert res is not None  # just to make sure it returns the tensor
    # 51
    assert kt[0, 0] == 0.0

    kt_data = KerasTensor((2, 2), "float32", data=np.array([[1, 2], [3, 4]]))
    assert np.array_equal((kt_data == 1), np.array([[True, False], [False, False]]))
    assert kt_data[0, 0] == 1

    # 43
    arr_copy = kt_data.__array__(copy=True)
    assert np.array_equal(arr_copy, np.array([[1, 2], [3, 4]]))
    assert arr_copy is not kt_data.data

    assert np.array_equal(kt_data.__array__(copy=False), np.array([[1, 2], [3, 4]]))

    # test Layer.name
    layer = Layer()
    assert layer.name == "layer"
    layer._name = "my_layer"
    assert layer.name == "my_layer"

    # test missing optimizers init
    optimizers.SGD()
    optimizers.Adagrad()
    optimizers.Nadam()


def test_keras_tensor_data_none():
    from zero_keras.core_layers import KerasTensor

    kt = KerasTensor((2, 2))
    kt.data = None

    # 34
    res = kt == 1
    assert res is not None

    # 43
    arr = kt.__array__(copy=True)
    assert arr is not None

    # 51
    val = kt[0, 0]
    assert val == 0.0


def test_remaining_core_and_opts():
    from zero_keras.core_layers import KerasTensor
    import zero_keras.optimizers as optimizers
    import numpy as np

    kt_data = KerasTensor((2, 2), "float32", data=np.array([[1, 2], [3, 4]]))
    arr_none = kt_data.__array__(copy=None)
    assert np.array_equal(arr_none, np.array([[1, 2], [3, 4]]))

    optimizers.Adafactor()
    optimizers.Lamb()
    optimizers.Muon()
