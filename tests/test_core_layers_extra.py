import numpy as np
from zero_keras.core_layers import KerasTensor, Layer, Model, Sequential
from unittest.mock import patch
import os


def test_keras_tensor_eq_no_data():
    tensor = KerasTensor(shape=(2, 2))
    other = KerasTensor(shape=(2, 2))
    result = tensor == other
    assert result is not None


def test_keras_tensor_array_branches():
    tensor = KerasTensor(shape=(2,), data=np.array([1, 2]))
    np.testing.assert_array_equal(tensor.__array__(copy=False), np.array([1, 2]))

    tensor2 = KerasTensor(shape=(2,), data=[3, 4])
    np.testing.assert_array_equal(tensor2.__array__(copy=False), np.array([3, 4]))

    tensor3 = KerasTensor(shape=(2,), data=[5, 6])
    np.testing.assert_array_equal(tensor3.__array__(copy=True), np.array([5, 6]))


def test_layer_set_weights_fallback():
    class CustomLayer(Layer):
        @property
        def weights(self):
            class W:
                def __setitem__(self, key, value):
                    self.val = value

            if not hasattr(self, "w"):
                self.w = W()
            return [self.w]

    layer = CustomLayer()
    layer.set_weights([[3, 4]])
    assert layer.weights[0].val == [3, 4]


def test_model_compute_loss_reg():
    class MyModel(Model):
        @property
        def losses(self):
            return [0.5]

    model = MyModel()
    x = KerasTensor(shape=(1,), data=np.array([1.0]))
    loss = model.compute_loss(x, x, x)
    assert float(np.asarray(loss)) == 0.5


def test_model_steps_type_error_fallback():
    class SimpleModel(Model):
        def call(self, inputs):
            return inputs * 2

    model = SimpleModel()
    model.compile(optimizer="sgd", loss="mse")
    x = np.array([1.0, 2.0])
    y = np.array([2.0, 4.0])

    metrics = model.train_step((x, y))
    assert "loss" in metrics

    metrics = model.test_step((x, y))
    assert "loss" in metrics

    preds = model.predict_step((x,))
    np.testing.assert_array_equal(np.asarray(preds), np.array([2.0, 4.0]))


def test_model_is_iterator_torch():
    model = Model()

    class FakeDataloader:
        pass

    FakeDataloader.__module__ = "torch.utils.data"
    assert model._is_iterator(FakeDataloader())


def test_model_fit_batch_size_none():
    model = Sequential([Layer()])
    model.compile("sgd", "mse")
    x = np.ones((33, 1))
    y = np.ones((33, 1))
    history = model.fit(x, y, batch_size=None, epochs=1, verbose=0)
    assert len(history.history["loss"]) == 1


def test_model_fit_eval_predict_unsliceable():
    model = Sequential([Layer()])
    model.compile("sgd", "mse")

    def gen():
        yield np.array([[1.0]]), np.array([[1.0]])

    model.fit(gen(), epochs=1, verbose=0)
    model.evaluate(gen(), verbose=0)

    def gen_x():
        yield np.array([[1.0]])

    model.predict(gen_x(), verbose=0)


def test_model_predict_numpy_fallback():
    class SimpleModel(Model):
        def call(self, inputs):
            class Pred:
                def numpy(self):
                    return np.array([[1.0]])

            return Pred()

    model = SimpleModel()
    model.compile("sgd", "mse")
    res = model.predict(np.array([[1.0]]), verbose=0)
    assert res is not None


def test_flatten_in_save():
    class NestedLayer(Layer):
        @property
        def weights(self):
            return [[[1.0, 2.0]]]

        @property
        def _trainable_weights(self):
            return self.weights

        @property
        def _non_trainable_weights(self):
            return []

    model = Sequential([NestedLayer()])
    model.save("test_flatten.keras")
    if os.path.exists("test_flatten.keras"):
        os.remove("test_flatten.keras")


def test_predict_concatenate_arrays_fallback():
    class Pred:
        def numpy(self):
            return [1.0]  # returns list, which is not a numpy array

    class SimpleModel(Model):
        def call(self, inputs):
            return Pred()

    model = SimpleModel()
    model.compile("sgd", "mse")
    # two batches so len(all_preds) == 2
    res = model.predict(np.array([[1.0], [2.0]]), batch_size=1, verbose=0)
    assert isinstance(res, list)
    assert len(res) == 2


def test_fit_exception_loss():
    class BrokenLossModel(Model):
        def call(self, inputs):
            return inputs

        def train_step(self, data):
            return {"loss": "not_a_float_or_array"}

        def test_step(self, data):
            return {"loss": "not_a_float"}

    model = BrokenLossModel()
    model.compile("sgd", "mse")
    x = np.array([[1.0]])
    y = np.array([[1.0]])
    model.fit(x, y, epochs=1, verbose=0)
    model.evaluate(x, y, verbose=0)


def test_predict_exception_data():
    class BrokenPredModel(Model):
        def call(self, inputs):
            class Pred:
                @property
                def data(self):
                    raise ValueError("Fail data")

            return Pred()

    model = BrokenPredModel()
    model.compile("sgd", "mse")
    x = np.array([[1.0]])

    res = model.predict(x, verbose=0)
    assert res is not None


def test_sequential_add():
    model = Sequential()
    model.add(Layer())
    assert len(model.layers) == 1


def test_ops_add():
    from zero_keras.core_layers import ops

    tensor = KerasTensor(shape=(2, 2))
    res = ops.add(tensor, tensor)
    assert res.shape == (2, 2)


def test_deserialize():
    from zero_keras.core_layers import deserialize

    assert deserialize({"class_name": "Dense"}) == {"class_name": "Dense"}


def test_get():
    from zero_keras.core_layers import get as layer_get

    assert layer_get("dense") == "dense"


def test_predict_concatenate_arrays_success():
    class SimpleModel(Model):
        def call(self, inputs):
            return inputs * 2

    model = SimpleModel()
    model.compile("sgd", "mse")
    res = model.predict(np.array([[1.0], [2.0]]), batch_size=1, verbose=0)
    assert np.allclose(res, np.array([[2.0], [4.0]]))


@patch("ml_switcheroo_compiler.ops.ones")
def test_keras_tensor_eq_no_data2(mock_ones):
    mock_ones.return_value = "mock_ones"
    tensor = KerasTensor(shape=None)
    other = KerasTensor(shape=None)
    result = tensor == other
    assert result == "mock_ones"


@patch("ml_switcheroo_compiler.ops.asarray")
def test_keras_tensor_array_numpy_attr(mock_asarray):
    class MockArr:
        def numpy(self):
            return np.array([5, 6])

    mock_asarray.return_value = MockArr()
    tensor = KerasTensor(shape=(2,), data=[5, 6])
    res = tensor.__array__(copy=True)
    np.testing.assert_array_equal(res, np.array([5, 6]))
