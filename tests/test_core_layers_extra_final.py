import numpy as np
import os
from zero_keras.core_layers import Model, Layer, KerasTensor, Sequential
from unittest.mock import patch
import ml_switcheroo_compiler.ops


def test_model_train_step_apply_gradients():
    class SimpleModel(Model):
        def call(self, inputs):
            return inputs

    model = SimpleModel()
    model.w = KerasTensor((1,), data=np.array([1.0]))
    model.w.id = "w_id"
    model._weights = [model.w]

    class Opt:
        def apply_gradients(self, grads):
            self.applied = True

    model.optimizer = Opt()

    def fake_get_grads(*args, **kwargs):
        return {"w_id": 1.0}

    original = getattr(ml_switcheroo_compiler.ops, "get_gradients", None)
    ml_switcheroo_compiler.ops.get_gradients = fake_get_grads
    try:
        model.train_step((np.array([1.0]), np.array([1.0])))
        assert getattr(model.optimizer, "applied", False)
    finally:
        if original is None:
            del ml_switcheroo_compiler.ops.get_gradients
        else:
            ml_switcheroo_compiler.ops.get_gradients = original


def test_866_867_fit_iterator_bad_loss():
    class BadLossModel(Model):
        def call(self, inputs):
            return inputs

        def train_step(self, data):
            class Bad:
                def __float__(self):
                    raise ValueError

            return {"loss": Bad()}

    model = BadLossModel()
    model.compile("sgd", "mse")

    def gen():
        yield np.array([1.0]), np.array([1.0])

    model.fit(gen(), steps_per_epoch=1, verbose=0)


def test_885_1097_fit_evaluate_else_bx_by():
    model = Sequential([Layer()])
    model.compile("sgd", "mse")
    # Float doesn't have __getitem__, triggering else: bx, by = x, y
    model.fit(1.0, 1.0, epochs=1, verbose=0)
    model.evaluate(1.0, 1.0, verbose=0)


def test_1165_predict_else_bx():
    model = Sequential([Layer()])
    model.compile("sgd", "mse")
    model.predict(1.0, verbose=0)


def test_946_to_bytes():
    class MockTensorNumpy:
        def numpy(self):
            return np.array([1.0])

    class NestedLayer(Layer):
        @property
        def weights(self):
            return [MockTensorNumpy()]

        @property
        def _trainable_weights(self):
            return self.weights

        @property
        def _non_trainable_weights(self):
            return []

    model = Sequential([NestedLayer()])
    model.save("test_946.keras")
    if os.path.exists("test_946.keras"):
        os.remove("test_946.keras")


def test_948_to_bytes():
    class MockTensorData:
        class Data:
            def numpy(self):
                return np.array([1.0])

        data = property(lambda self: self.Data())

    class NestedLayer(Layer):
        @property
        def weights(self):
            return [MockTensorData()]

        @property
        def _trainable_weights(self):
            return self.weights

        @property
        def _non_trainable_weights(self):
            return []

    model = Sequential([NestedLayer()])
    model.save("test_948.keras")
    if os.path.exists("test_948.keras"):
        os.remove("test_948.keras")


def test_1147_predict_iterator_numpy():
    class NumpyModel(Model):
        def predict_step(self, data):
            class Preds:
                def numpy(self):
                    return np.array([1.0])

            return Preds()

    m = NumpyModel()
    m.compile("sgd", "mse")

    def gen():
        yield np.array([1.0])

    m.predict(gen(), steps=1, verbose=0)


@patch("ml_switcheroo_compiler.ops.asarray")
def test_1153_predict_iterator_exception(mock_asarray):
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

    def gen():
        yield np.array([1.0])

    res = m.predict(gen(), steps=1, verbose=0)
    assert res == "throw"
