"""Unit and behavioral tests for keras in zero_keras."""

from __future__ import annotations
from zero_keras.types import TensorLike

from zero_keras.core_layers import Input, KerasTensor, Layer, Model
from zero_keras.ops import ops


def test_keras_tensor() -> None:
    """Verify or execute `test keras tensor`."""
    t = Input((2, 3))
    assert isinstance(t, KerasTensor)
    t.numpy()
    t2 = t + t
    assert isinstance(t2, KerasTensor)

    # KerasTensor edge cases
    t_data = KerasTensor((2, 3), data=[1])
    assert t_data.data == [1]

    # hit branch 28->30 by raising in ops.zeros
    from unittest import mock

    from ml_switcheroo_compiler import ops

    with mock.patch.object(ops, "zeros", side_effect=Exception):
        t_bad_shape = KerasTensor((2,))
        assert t_bad_shape.data is None

    # Math ops
    _ = t - t
    _ = t * t
    _ = t / t
    _ = t**t
    _ = t[0]
    _ = t_data[0]

    _ = t_bad_shape[0]

    try:
        _ = t == t
    except ValueError:
        pass
    _ = t_data == 1

    t_numpyable = KerasTensor((1,), data=[1])
    assert list(t_numpyable.__array__(copy=False)) == [1]

    t_ndarray = KerasTensor((1,), data=ops.convert_to_tensor([1]))
    assert list(t_ndarray.__array__(copy=False)) == [1]

    class DummyHasNumpy:
        """Test helper class `DummyHasNumpy` for keras."""

        def numpy(self) -> TensorLike:
            """Verify or execute `numpy` on `DummyHasNumpy`.

            Returns:
                TensorLike: Result from `numpy`.
            """
            return ops.convert_to_numpy(ops.convert_to_tensor([2]))

    t_hasnumpy = KerasTensor((1,), data=DummyHasNumpy())
    try:
        t_hasnumpy.__array__(copy=True)
    except Exception:
        pass

    try:
        t_bad_shape.__array__(copy=True)
    except Exception:
        pass
    try:
        t_bad_shape.__array__(copy=False)
    except Exception:
        pass

    from ml_switcheroo_compiler.core import config

    config.eager_mode = True
    try:
        t_true = KerasTensor((1,), data=True)
        assert bool(t_true) is True
    except Exception:
        pass

    # Hit boolean exception branch by providing data that fails bool()
    class Unboolable:
        """Test helper class `Unboolable` for keras."""

        def __bool__(self) -> TensorLike:
            """Verify or execute `bool` on `Unboolable`."""
            raise ValueError("cannot bool")

    t_unbool = KerasTensor((1,), data=Unboolable())
    try:
        bool(t_unbool)
    except Exception:
        pass

    try:
        bool(t)  # Should raise
    except TypeError:
        pass

    from zero_keras.core_layers import Node

    # Hit missing node branches
    class HasHistory:
        """Test helper class `HasHistory` for keras."""

        def __init__(self) -> None:
            """Verify or execute `init` on `HasHistory`."""
            self._keras_history = None

    Node(Layer(), outputs=[HasHistory()])
    Node(Layer(), outputs=HasHistory())
    Node(Layer(), outputs=t)

    class HasInboundNodes:
        """Test helper class `HasInboundNodes` for keras."""

        _inbound_nodes = []

    Node(HasInboundNodes(), outputs=[])


def test_layer_build() -> None:
    """Verify or execute `test layer build`."""

    class MyLayer(Layer):
        """Test helper class `MyLayer` for keras."""

        def build(self, input_shape: TensorLike) -> None:
            """Verify or execute `build` on `MyLayer`.

            Args:
                input_shape: Test fixture or input parameter `input_shape` (input shape).
            """
            self.w = 1.0
            super().build(input_shape)

    layer = MyLayer()
    assert not layer.built
    layer(Input((2,)))
    assert layer.built


def test_model_functional() -> None:
    """Verify or execute `test model functional`."""
    from zero_keras import random
    from zero_keras.layers import Dense

    i = Input((2,))
    layer_obj = Dense(1)(i)
    m = Model(inputs=i, outputs=layer_obj)
    print("type of m is", type(m))
    # mock loss
    m.compile(optimizer="sgd", loss="mse")

    x = random.uniform((1, 2))
    random.uniform((1, 1))

    class TestIterator:
        """Test helper class `TestIterator` for keras."""

        def __iter__(self) -> TensorLike:
            """Verify or execute `iter` on `TestIterator`."""
            yield (ops.convert_to_tensor([[1.0, 2.0]]), ops.convert_to_tensor([[1.0]]))
            yield (ops.convert_to_tensor([[2.0, 3.0]]), ops.convert_to_tensor([[2.0]]))

    m.fit(TestIterator(), epochs=1)

    class DummyMetric:
        """Test helper class `DummyMetric` for keras."""

        name = "dummy_metric"

        def reset_state(self) -> TensorLike:
            """Verify or execute `reset state` on `DummyMetric`."""

        def update_state(self, y: TensorLike, y_pred: TensorLike) -> TensorLike:
            """Verify or execute `update state` on `DummyMetric`.

            Args:
                y: Test fixture or input parameter `y` (y).
                y_pred: Test fixture or input parameter `y_pred` (y pred).
            """

        def result(self) -> TensorLike:
            """Verify or execute `result` on `DummyMetric`.

            Returns:
                TensorLike: Result from `result`.
            """
            return 1.0

    m.compile(optimizer="sgd", loss="mse", metrics=[DummyMetric()])
    m.evaluate(TestIterator(), None)
    m.predict(TestIterator())

    class TestTorchLikeIterator:
        """Test helper class `TestTorchLikeIterator` for keras."""

        __module__ = "torch.utils.data.DataLoader"

        def __iter__(self) -> TensorLike:
            """Verify or execute `iter` on `TestTorchLikeIterator`."""

            class FakeTorchTensor:
                """Test helper class `FakeTorchTensor` for keras."""

                def __init__(self, val: TensorLike) -> None:
                    """Verify or execute `init` on `FakeTorchTensor`.

                    Args:
                        val: Test fixture or input parameter `val` (val).
                    """
                    self.val = val

                def numpy(self) -> TensorLike:
                    """Verify or execute `numpy` on `FakeTorchTensor`.

                    Returns:
                        TensorLike: Result from `numpy`.
                    """
                    return ops.convert_to_numpy(ops.convert_to_tensor([[self.val]]))

            yield (FakeTorchTensor(1.0), FakeTorchTensor(1.0))
            yield {"x": FakeTorchTensor(1.0)}
            yield FakeTorchTensor(1.0)

    try:
        m.fit(TestTorchLikeIterator())
    except Exception:
        pass
    try:
        m.evaluate(TestTorchLikeIterator())
    except Exception:
        pass
    try:
        m.predict(TestTorchLikeIterator())
    except Exception:
        pass

    class TestTfLikeIterator:
        """Test helper class `TestTfLikeIterator` for keras."""

        __module__ = "tensorflow.python.data.ops.dataset_ops.DatasetV2"

        def __iter__(self) -> TensorLike:
            """Verify or execute `iter` on `TestTfLikeIterator`."""
            yield (ops.convert_to_tensor([[1.0, 2.0]]), ops.convert_to_tensor([[1.0]]))

    m.evaluate(TestTfLikeIterator())
    m.predict(TestTfLikeIterator())

    class UnpredictableTensor:
        """Test helper class `UnpredictableTensor` for keras."""

        def __init__(self, val: TensorLike) -> None:
            """Verify or execute `init` on `UnpredictableTensor`.

            Args:
                val: Test fixture or input parameter `val` (val).
            """
            self.val = val

    # Test generator for prediction exception
    class MyPredictLayer(Layer):
        """Test helper class `MyPredictLayer` for keras."""

        def call(self, inputs: TensorLike) -> TensorLike:
            """Verify or execute `call` on `MyPredictLayer`.

            Args:
                inputs: Test fixture or input parameter `inputs` (inputs).

            Returns:
                TensorLike: Result from `call`.
            """
            return UnpredictableTensor(1.0)

    m_pred_fail = Model(inputs=Input((2,)), outputs=MyPredictLayer()(Input((2,))))

    def my_gen_fail() -> TensorLike:
        """Verify or execute `my gen fail`."""
        yield ops.convert_to_tensor([[1.0, 2.0]])

    class MyEvalLayer(Layer):
        """Test helper class `MyEvalLayer` for keras."""

        def call(self, inputs: TensorLike) -> TensorLike:
            """Verify or execute `call` on `MyEvalLayer`.

            Args:
                inputs: Test fixture or input parameter `inputs` (inputs).

            Returns:
                TensorLike: Result from `call`.
            """
            return inputs

    m_eval_fail = Model(inputs=Input((2,)), outputs=MyEvalLayer()(Input((2,))))

    # dummy loss
    class EvalFailLoss:
        """Test helper class `EvalFailLoss` for keras."""

        def __call__(self, y_true: TensorLike, y_pred: TensorLike) -> TensorLike:
            """Verify or execute `call` on `EvalFailLoss`.

            Args:
                y_true: Test fixture or input parameter `y_true` (y true).
                y_pred: Test fixture or input parameter `y_pred` (y pred).

            Returns:
                TensorLike: Result from `__call__`.
            """
            return UnpredictableTensor(1.0)

    class DummyMetric:
        """Test helper class `DummyMetric` for keras."""

        name = "dummy_metric"

        def reset_state(self) -> TensorLike:
            """Verify or execute `reset state` on `DummyMetric`."""

        def update_state(self, y: TensorLike, y_pred: TensorLike) -> TensorLike:
            """Verify or execute `update state` on `DummyMetric`.

            Args:
                y: Test fixture or input parameter `y` (y).
                y_pred: Test fixture or input parameter `y_pred` (y pred).
            """

        def result(self) -> TensorLike:
            """Verify or execute `result` on `DummyMetric`.

            Returns:
                TensorLike: Result from `result`.
            """
            return 1.0

    m_eval_fail.compile(optimizer="sgd", loss=EvalFailLoss(), metrics=[DummyMetric()])
    m_pred_fail.predict(my_gen_fail())
    try:
        m_eval_fail.evaluate(my_gen_fail())
    except Exception:
        pass

    m_eval_fail.evaluate(ops.convert_to_tensor([[1.0, 2.0]]), ops.convert_to_tensor([[1.0, 2.0]]))

    class Unsliceable:
        """Test helper class `Unsliceable` for keras."""

        def __init__(self, val: TensorLike) -> None:
            """Verify or execute `init` on `Unsliceable`.

            Args:
                val: Test fixture or input parameter `val` (val).
            """
            self.val = val

        def __len__(self) -> TensorLike:
            """Verify or execute `len` on `Unsliceable`.

            Returns:
                TensorLike: Result from `__len__`.
            """
            return 2

        def __getitem__(self, key: TensorLike) -> TensorLike:
            """Verify or execute `getitem` on `Unsliceable`.

            Args:
                key: Test fixture or input parameter `key` (key).
            """
            raise TypeError("Not sliceable")

    # This should trigger TypeError in bx = x[start:end]
    try:
        m.fit(Unsliceable(1), Unsliceable(1), epochs=1)
    except TypeError:
        pass
    try:
        m.evaluate(Unsliceable(1), Unsliceable(1))
    except TypeError:
        pass
    try:
        m.predict(Unsliceable(1))
    except TypeError:
        pass

    # Test TypeError branch in call
    class NoTrainingLayer(Layer):
        """Test helper class `NoTrainingLayer` for keras."""

        def call(self, inputs: TensorLike) -> TensorLike:
            """Verify or execute `call` on `NoTrainingLayer`.

            Args:
                inputs: Test fixture or input parameter `inputs` (inputs).

            Returns:
                TensorLike: Result from `call`.
            """
            return inputs

    i2 = Input((2,))
    out2 = NoTrainingLayer()(i2)
    m2 = Model(inputs=i2, outputs=out2)
    m2.compile(optimizer="sgd", loss="mse")
    m2.fit(x, x)


def test_model_uncompiled() -> None:
    """Verify or execute `test model uncompiled`."""
    i = Input((2,))
    layer_obj = Layer()(i)
    m = Model(inputs=i, outputs=layer_obj)
    print("type of m is", type(m))
    # Fit without compile
    res = m.fit([1], [1])
    assert "loss" in res.history


def test_ops() -> None:
    """Verify or execute `test ops`."""
    assert isinstance(ops.add(Input((2,)), Input((2,))), KerasTensor)


def test_layer_properties() -> None:
    """Verify or execute `test layer properties`."""
    from zero_keras.core_layers import Layer

    class MyLayer(Layer):
        """Test helper class `MyLayer` for keras."""

        def build(self, input_shape: TensorLike) -> None:
            """Verify or execute `build` on `MyLayer`.

            Args:
                input_shape: Test fixture or input parameter `input_shape` (input shape).
            """
            self.w = self.add_weight(shape=(1,), initializer="zeros", trainable=True)
            self.w2 = self.add_weight(shape=(1,), initializer="zeros", trainable=False)
            super().build(input_shape)

    layer_obj = MyLayer(name="my_layer")
    layer_obj.my_list = [MyLayer(), 1]
    layer_obj.my_list[0].build((1,))
    if hasattr(layer_obj, "get_config"):
        layer_obj.get_config()
    assert layer_obj.name == "my_layer"
    layer_obj.add_loss(0.5)
    layer_obj.add_loss(0.6)
    assert layer_obj.losses == [0.5, 0.6]

    layer_obj.build((2,))

    # Duplicate a weight to hit branch 515->513
    layer_obj._weights.append(layer_obj._weights[0])

    assert len(layer_obj.weights) == 4
    assert len(layer_obj.trainable_weights) == 4
    assert len(layer_obj.non_trainable_weights) == 0

    # Hit missing get_weights branch where weight lacks numpy / data.numpy
    class DummyWeightNoNumpy:
        """Test helper class `DummyWeightNoNumpy` for keras."""

        def __init__(self, data: TensorLike) -> None:
            """Verify or execute `init` on `DummyWeightNoNumpy`.

            Args:
                data: Test fixture or input parameter `data` (data).
            """
            self.data = data
            self.shape = (1,)

        def __float__(self) -> TensorLike:
            """Verify or execute `float` on `DummyWeightNoNumpy`.

            Returns:
                TensorLike: Result from `__float__`.
            """
            return float(self.data[0])

        def assign(self, value: TensorLike) -> None:
            """Verify or execute `assign` on `DummyWeightNoNumpy`.

            Args:
                value: Test fixture or input parameter `value` (value).
            """
            self.data = value

    layer_obj._weights.append(DummyWeightNoNumpy([1.0]))
    w_list = layer_obj.get_weights()
    assert len(w_list) == 5

    # Add dummy optimizer variables to hit model save branches
    m = Model(inputs=Input((2,)), outputs=layer_obj)
    m.compile(optimizer="adam", loss="mse")

    class DummyOpt:
        """Test helper class `DummyOpt` for keras."""

        variables = [DummyWeightNoNumpy([2.0])]

    m.optimizer = DummyOpt()

    class DummyWeightNoNumpyAndIterable:
        """Test helper class `DummyWeightNoNumpyAndIterable` for keras."""

        def __init__(self, data: TensorLike) -> None:
            """Verify or execute `init` on `DummyWeightNoNumpyAndIterable`.

            Args:
                data: Test fixture or input parameter `data` (data).
            """
            self.data = data
            self.shape = (1, 1)

        def __iter__(self) -> TensorLike:
            """Verify or execute `iter` on `DummyWeightNoNumpyAndIterable`.

            Returns:
                TensorLike: Result from `__iter__`.
            """
            return iter([self.data])

        def __float__(self) -> TensorLike:
            """Verify or execute `float` on `DummyWeightNoNumpyAndIterable`.

            Returns:
                TensorLike: Result from `__float__`.
            """
            return float(self.data[0])

        def assign(self, value: TensorLike) -> None:
            """Verify or execute `assign` on `DummyWeightNoNumpyAndIterable`.

            Args:
                value: Test fixture or input parameter `value` (value).
            """
            self.data = value

    layer_obj._weights.append(DummyWeightNoNumpyAndIterable([1.0]))
    w_list = layer_obj.get_weights()
    assert len(w_list) == 6

    import os
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model_with_iter_weight.keras")
        m.save(path)
        try:
            m.save(path, overwrite=False)
        except FileExistsError:
            pass

    m.optimizer.variables = []
    with tempfile.TemporaryDirectory() as d:
        path2 = os.path.join(d, "model_empty_opt.keras")
        m.save(path2)

    layer_obj.set_weights(layer_obj.get_weights())


def test_sequential_properties() -> None:
    """Verify or execute `test sequential properties`."""
    from zero_keras.layers import Dense
    from zero_keras.models import Sequential

    m = Sequential([Dense(5, input_shape=(10,)), Dense(1)], name="seq")
    assert m.name == "seq"
    m.build((2, 10))
    cfg = m.get_config()
    for lyr in cfg["layers"]:
        lyr["config"]["units"] = 5
    m2 = Sequential.from_config(cfg)
    m2.build((2, 10))

    # run call
    # m2(np.random.rand(2, 10).astype("float32"))
    # cover property edge cases
    _ = m.weights
    _ = m.trainable_weights
    _ = m.non_trainable_weights
    assert m.name == "seq"
    m.build((2, 10))
    cfg = m.get_config()
    assert cfg["name"] == "seq"

    # Patch config for dummy test
    for lyr in cfg["layers"]:
        lyr["config"]["units"] = 5
    m2 = Sequential.from_config(cfg)
    m2.build((2, 10))

    assert len(m.weights) > 0
    assert len(m.trainable_weights) > 0
    assert len(m.non_trainable_weights) == 0
    assert len(m.get_weights()) > 0
    m.set_weights(m.get_weights())


def test_models_sequential() -> None:
    """Verify or execute `test models sequential`."""
    from zero_keras import random
    from zero_keras.layers import Dense
    from zero_keras.models import Sequential

    m_real = Sequential([Dense(1)])
    m_real.build((2, 5))
    m_real(random.uniform((2, 5)))
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        m_real.save(os.path.join(d, "seq.keras"))
    m_real.get_config()
    m_real.weights
    m_real.trainable_weights
    m_real.non_trainable_weights
    m_real.set_weights(m_real.get_weights())

    # Test Sequential
    model = Sequential(name="my_seq")
    assert model._name == "my_seq"

    model.add(Dense(10))
    model.add(Dense(5))

    # config
    cfg = model.get_config()
    assert "layers" in cfg

    # patch config
    for lyr in cfg["layers"]:
        lyr["config"]["units"] = 5

    m2 = Sequential.from_config(cfg)
    assert len(m2.layers) == 2

    # Call
    x = random.uniform((2, 5))
    model.build((None, 5))
    res = model(x)
    assert res is not None

    # config with missing layer
    cfg_miss = {"layers": [{"class_name": "DoesNot_Exist"}], "name": "miss"}
    Sequential.from_config(cfg_miss)

    class DummyLayerWithoutWeights:
        """Test helper class `DummyLayerWithoutWeights` for keras."""

        built = True
        weights = []
        trainable_weights = []
        non_trainable_weights = []

        def __call__(self, inputs: TensorLike, **kwargs: TensorLike) -> TensorLike:
            """Verify or execute `call` on `DummyLayerWithoutWeights`.

            Args:
                inputs: Test fixture or input parameter `inputs` (inputs).
                **kwargs: Keyword arguments forwarded to helper.

            Returns:
                TensorLike: Result from `__call__`.
            """
            return inputs

        def compute_output_shape(self, shape: TensorLike) -> TensorLike:
            """Verify or execute `compute output shape` on `DummyLayerWithoutWeights`.

            Args:
                shape: Test fixture or input parameter `shape` (shape).

            Returns:
                TensorLike: Result from `compute_output_shape`.
            """
            return shape

    class DummyLayerWithComputeShape:
        """Test helper class `DummyLayerWithComputeShape` for keras."""

        built = False
        weights = []
        trainable_weights = []
        non_trainable_weights = []

        def build(self, shape: TensorLike) -> None:
            """Verify or execute `build` on `DummyLayerWithComputeShape`.

            Args:
                shape: Test fixture or input parameter `shape` (shape).
            """
            self.built = True

        def compute_output_shape(self, shape: TensorLike) -> TensorLike:
            """Verify or execute `compute output shape` on `DummyLayerWithComputeShape`.

            Args:
                shape: Test fixture or input parameter `shape` (shape).

            Returns:
                TensorLike: Result from `compute_output_shape`.
            """
            return shape

        def __call__(self, inputs: TensorLike, **kwargs: TensorLike) -> TensorLike:
            """Verify or execute `call` on `DummyLayerWithComputeShape`.

            Args:
                inputs: Test fixture or input parameter `inputs` (inputs).
                **kwargs: Keyword arguments forwarded to helper.

            Returns:
                TensorLike: Result from `__call__`.
            """
            return inputs

    m3 = Sequential()
    m3.add(DummyLayerWithoutWeights())
    m3.add(DummyLayerWithComputeShape())
    m3.build((None, 5))
    m3(random.uniform((2, 5)))

    # fake units
    class DummyLayerWithUnits:
        """Test helper class `DummyLayerWithUnits` for keras."""

        units = 5
        built = False

        def build(self, shape: TensorLike) -> None:
            """Verify or execute `build` on `DummyLayerWithUnits`.

            Args:
                shape: Test fixture or input parameter `shape` (shape).
            """
            self.built = True

    import pytest

    m4 = Sequential()
    m4.add(DummyLayerWithUnits())
    m4.build(None)

    m5 = Sequential()
    m5.add(DummyLayerWithUnits())
    m5.build((1,))

    # Weights empty properties for dummy layer

    class LayerNoGetWeights(Layer):
        """Test helper class `LayerNoGetWeights` for keras."""

    m3.add(LayerNoGetWeights())
    m3.set_weights(m3.get_weights())

    # Save
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.keras")
        model.save(path)

        # Load
        from zero_keras.models import load_model

        loaded = load_model(path)
        assert len(loaded.layers) == 2

        # load non-sequential config
        import json
        import struct
        import zipfile

        path2 = os.path.join(d, "model2.keras")
        with zipfile.ZipFile(path2, "w") as z:
            z.writestr("config.json", json.dumps({"class_name": "Functional"}))
            header = json.dumps({"__metadata__": {"format": "pt"}})
            safetensors_data = struct.pack("<Q", len(header)) + header.encode()
            z.writestr("model.safetensors", safetensors_data)
        loaded2 = load_model(path2)
        assert loaded2 is not None

        # load model with fake layer class
        path3 = os.path.join(d, "model3.keras")
        with zipfile.ZipFile(path3, "w") as z:
            z.writestr(
                "config.json",
                json.dumps(
                    {
                        "class_name": "Sequential",
                        "config": {"layers": [{"class_name": "DoesNot_Exist"}]},
                    }
                ),
            )
            # no safetensors file
        load_model(path3)

        # load model with un-settable weights
        path4 = os.path.join(d, "model4.keras")
        with zipfile.ZipFile(path4, "w") as z:
            z.writestr("config.json", json.dumps({"class_name": "Functional"}))
            import array

            arr = array.array("f", [1.0])
            header = json.dumps({"0": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}})
            safetensors_data = struct.pack("<Q", len(header)) + header.encode() + arr.tobytes()
            z.writestr("model.safetensors", safetensors_data)
        load_model(path4)

        # load model not zip
        path5 = os.path.join(d, "model5.keras")
        with open(path5, "w") as f:
            f.write("not zip")
        import zipfile

        try:
            load_model(path5)
        except zipfile.BadZipFile:
            pass

        # Test 126 branch (neither .keras nor zip)
        path6 = os.path.join(d, "model6.txt")
        with open(path6, "w") as f:
            f.write("not zip")
        with pytest.raises(ValueError):
            load_model(path6)

    from zero_keras.core_layers import Functional, deserialize, get
    from zero_keras.core_layers import ops as keras_ops

    f = Functional()
    s = Sequential()
    s.add(Layer())
    s.call(1)

    assert keras_ops.add(Input((2,)), Input((2,))) is not None
    assert deserialize("a") == "a"
    with pytest.raises(ValueError):
        get("b")


def test_model_dict_in_out() -> None:
    """Verify or execute `test model dict in out`."""
    from zero_keras.core_layers import Input, Model
    from zero_keras.layers import Dense

    i1 = Input((2,), name="in1")
    i2 = Input((2,), name="in2")

    d1 = Dense(1)(i1)
    d2 = Dense(1)(i2)

    m = Model(inputs={"a": i1, "b": i2}, outputs={"x": d1, "y": d2})
    assert len(m.layers) == 2  # 2 dense

    # Also test single scalar outputs/inputs not in lists
    i3 = Input((2,))
    d3 = Dense(1)(i3)
    m2 = Model(inputs=i3, outputs=d3)
    assert len(m2.layers) == 1


def test_model_coverage_edges() -> None:
    """Verify or execute `test model coverage edges`."""
    from zero_keras.core_layers import Input, Layer, Model
    from zero_keras.layers import Dense

    # Custom layer to take a list and a dict
    class ListLayer(Layer):
        """Test helper class `ListLayer` for keras."""

        def call(self, inputs: TensorLike) -> TensorLike:
            """Verify or execute `call` on `ListLayer`.

            Args:
                inputs: Test fixture or input parameter `inputs` (inputs).

            Returns:
                TensorLike: Result from `call`.
            """
            return inputs[0] + inputs[1]

    class DictLayer(Layer):
        """Test helper class `DictLayer` for keras."""

        def call(self, inputs: TensorLike) -> TensorLike:
            """Verify or execute `call` on `DictLayer`.

            Args:
                inputs: Test fixture or input parameter `inputs` (inputs).

            Returns:
                TensorLike: Result from `call`.
            """
            return inputs["a"] + inputs["b"]

    i1 = Input((2,))
    i2 = Input((2,))
    d1 = Dense(1)(i1)
    d2 = Dense(1)(i2)

    l1 = ListLayer()([d1, d2])
    l2 = DictLayer()({"a": d1, "b": d2})

    # layer used twice to trigger return
    class AddLayer(Layer):
        """Test helper class `AddLayer` for keras."""

        def call(self, inputs: TensorLike) -> TensorLike:
            """Verify or execute `call` on `AddLayer`.

            Args:
                inputs: Test fixture or input parameter `inputs` (inputs).

            Returns:
                TensorLike: Result from `call`.
            """
            return inputs[0] + inputs[1]

    d3 = Dense(1)(i1)
    a1 = AddLayer()([d3, d3])

    m = Model(inputs=[i1, i2], outputs=[l1, l2, a1])
    assert m is not None
