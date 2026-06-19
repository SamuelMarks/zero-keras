import numpy as np
from zero_keras.core_layers import Model, Layer, KerasTensor, Sequential
from unittest.mock import patch
import os
import ml_switcheroo_compiler.ops


def test_keras_tensor_eq_no_data():
    tensor = KerasTensor(shape=(2, 2))
    other = KerasTensor(shape=(2, 2))
    assert (tensor == other) is not None


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
    assert float(np.asarray(model.compute_loss(x, x, x))) == 0.5


def test_model_steps_type_error_fallback():
    class SimpleModel(Model):
        def call(self, inputs):
            return inputs * 2

    model = SimpleModel()
    model.compile(optimizer="sgd", loss="mse")
    x, y = np.array([1.0, 2.0]), np.array([2.0, 4.0])
    assert "loss" in model.train_step((x, y))
    assert "loss" in model.test_step((x, y))
    np.testing.assert_array_equal(
        np.asarray(model.predict_step((x,))), np.array([2.0, 4.0])
    )


def test_model_is_iterator_torch():
    model = Model()

    class FakeDataloader:
        pass

    FakeDataloader.__module__ = "torch.utils.data"
    assert model._is_iterator(FakeDataloader())


def test_model_fit_batch_size_none():
    model = Sequential([Layer()])
    model.compile("sgd", "mse")
    assert (
        len(
            model.fit(
                np.ones((33, 1)), np.ones((33, 1)), batch_size=None, epochs=1, verbose=0
            ).history["loss"]
        )
        == 1
    )


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
    assert model.predict(np.array([[1.0]]), verbose=0) is not None


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
            return [1.0]

    class SimpleModel(Model):
        def call(self, inputs):
            return Pred()

    model = SimpleModel()
    model.compile("sgd", "mse")
    res = model.predict(np.array([[1.0], [2.0]]), batch_size=1, verbose=0)
    assert isinstance(res, list) and len(res) == 2


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
    model.fit(np.array([[1.0]]), np.array([[1.0]]), epochs=1, verbose=0)
    model.evaluate(np.array([[1.0]]), np.array([[1.0]]), verbose=0)


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
    assert model.predict(np.array([[1.0]]), verbose=0) is not None


def test_sequential_add():
    model = Sequential()
    model.add(Layer())
    assert len(model.layers) == 1


def test_ops_add():
    from zero_keras.core_layers import ops

    assert ops.add(KerasTensor(shape=(2, 2)), None).shape == (2, 2)


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
    assert np.allclose(
        model.predict(np.array([[1.0], [2.0]]), batch_size=1, verbose=0),
        np.array([[2.0], [4.0]]),
    )


@patch("ml_switcheroo_compiler.ops.ones")
def test_keras_tensor_eq_no_data2(mock_ones):
    mock_ones.return_value = "mock_ones"
    assert (KerasTensor(shape=None) == KerasTensor(shape=None)) == "mock_ones"


def test_model_train_step_apply_gradients():
    class SimpleModel(Model):
        def call(self, inputs):
            return inputs

    model = SimpleModel()
    model.w = KerasTensor((1,), data=np.array([1.0]))
    model.w.id = "w_id"
    model._trainable_weights = [model.w]

    class Opt:
        def apply_gradients(self, grads):
            pass

    model.optimizer = Opt()

    def fake_get_grads(*args, **kwargs):
        return {"w_id": 1.0}

    original = getattr(ml_switcheroo_compiler.ops, "get_gradients", None)
    ml_switcheroo_compiler.ops.get_gradients = fake_get_grads
    try:
        model.train_step((np.array([1.0]), np.array([1.0])))
    finally:
        if original is None:
            del ml_switcheroo_compiler.ops.get_gradients
        else:
            ml_switcheroo_compiler.ops.get_gradients = original


def test_b_loss_fallback_train():
    class BadLossModel(Model):
        def call(self, inputs):
            return inputs

        def train_step(self, data):
            class Bad:
                def __float__(self):
                    raise ValueError

            return {"loss": Bad()}

        def test_step(self, data):
            class Bad:
                def __float__(self):
                    raise ValueError

            return {"loss": Bad()}

    model = BadLossModel()
    model.compile("sgd", "mse")
    model.fit(np.array([[1.0]]), np.array([[1.0]]), epochs=1, verbose=0)


def test_bx_by_iterator_branch_885():
    model = Sequential([Layer()])
    model.compile("sgd", "mse")

    def gen():
        yield np.array([1.0]), np.array([1.0])

    model.evaluate(gen(), steps=1, verbose=0)


def test_predict_1165():
    model = Sequential([Layer()])
    model.compile("sgd", "mse")

    def gen():
        yield np.array([1.0])

    model.predict(gen(), steps=1, verbose=0)


@patch("ml_switcheroo_compiler.ops.asarray")
def test_predict_1153(mock_asarray):
    class ThrowingModel(Model):
        def predict_step(self, data):
            return "throw"

    m = ThrowingModel()
    m.compile("sgd", "mse")

    class Thrower:
        @property
        def data(self):
            raise ValueError

    mock_asarray.return_value = Thrower()
    assert m.predict(np.array([1.0]), verbose=0) == "throw"
