"""Module docstring."""

import numpy as np
import os
from zero_keras.core_layers import Model, Layer, KerasTensor, Sequential
from unittest.mock import patch
import ml_switcheroo_compiler.ops


def test_model_train_step_apply_gradients():
    """Function docstring."""

    class SimpleModel(Model):
        """Class docstring."""

        def call(self, inputs):
            """Function docstring.

            Args:
                inputs: Description.
            """
            return inputs

    model = SimpleModel()
    model.w = KerasTensor((1,), data=np.array([1.0]))
    model.w.id = "w_id"
    model._weights = [model.w]

    class Opt:
        """Class docstring."""

        def apply_gradients(self, grads):
            """Function docstring.

            Args:
                grads: Description.
            """
            self.applied = True

    model.optimizer = Opt()

    def fake_get_grads(*args, **kwargs):
        """Function docstring.

        Args:
            args: Description.
            kwargs: Description.
        """
        return {"w_id": 1.0}  # pragma: no cover

    original = getattr(ml_switcheroo_compiler.ops, "get_gradients", None)
    ml_switcheroo_compiler.ops.get_gradients = fake_get_grads
    try:
        model.train_step((np.array([1.0]), np.array([1.0])))
        assert getattr(model.optimizer, "applied", False)
    finally:
        if original is None:
            del ml_switcheroo_compiler.ops.get_gradients
        else:
            ml_switcheroo_compiler.ops.get_gradients = original  # pragma: no cover


def test_866_867_fit_iterator_bad_loss():
    """Function docstring."""

    class BadLossModel(Model):
        """Class docstring."""

        def call(self, inputs):
            """Function docstring.

            Args:
                inputs: Description.
            """
            return inputs  # pragma: no cover

        def train_step(self, data):
            """Function docstring.

            Args:
                data: Description.
            """

            class Bad:
                """Class docstring."""

                def __float__(self):
                    """Function docstring."""
                    raise ValueError

            return {"loss": Bad()}

    model = BadLossModel()
    model.compile("sgd", "mse")

    def gen():
        """Function docstring."""
        yield np.array([1.0]), np.array([1.0])

    model.fit(gen(), steps_per_epoch=1, verbose=0)


def test_885_1097_fit_evaluate_else_bx_by():
    """Function docstring."""
    model = Sequential([Layer()])
    # compile with a custom loss that works on scalars
    model.compile("sgd", loss=lambda y_true, y_pred: y_true - y_pred)
    # Float doesn't have __getitem__, triggering else: bx, by = x, y
    model.fit(1.0, 1.0, epochs=1, verbose=0)
    model.evaluate(1.0, 1.0, verbose=0)


def test_1165_predict_else_bx():
    """Function docstring."""
    model = Sequential([Layer()])
    model.compile("sgd", "mse")
    model.predict(1.0, verbose=0)


def test_946_to_bytes():
    """Function docstring."""

    class MockTensorNumpy:
        """Class docstring."""

        def numpy(self):
            """Function docstring."""
            return np.array([1.0])

    class NestedLayer(Layer):
        """Class docstring."""

        @property
        def weights(self):
            """Function docstring."""
            return [MockTensorNumpy()]

        @property
        def _trainable_weights(self):
            """Function docstring."""
            return self.weights  # pragma: no cover

        @property
        def _non_trainable_weights(self):
            """Function docstring."""
            return []  # pragma: no cover

    model = Sequential([NestedLayer()])
    model.save("test_946.keras")
    if os.path.exists("test_946.keras"):
        os.remove("test_946.keras")


def test_948_to_bytes():
    """Function docstring."""

    class MockTensorData:
        """Class docstring."""

        class Data:
            """Class docstring."""

            def numpy(self):
                """Function docstring."""
                return np.array([1.0])

        data = property(lambda self: self.Data())

    class NestedLayer(Layer):
        """Class docstring."""

        @property
        def weights(self):
            """Function docstring."""
            return [MockTensorData()]

        @property
        def _trainable_weights(self):
            """Function docstring."""
            return self.weights  # pragma: no cover

        @property
        def _non_trainable_weights(self):
            """Function docstring."""
            return []  # pragma: no cover

    model = Sequential([NestedLayer()])
    model.save("test_948.keras")
    if os.path.exists("test_948.keras"):
        os.remove("test_948.keras")


def test_1147_predict_iterator_numpy():
    """Function docstring."""

    class NumpyModel(Model):
        """Class docstring."""

        def predict_step(self, data):
            """Function docstring.

            Args:
                data: Description.
            """

            class Preds:
                """Class docstring."""

                def numpy(self):
                    """Function docstring."""
                    return np.array([1.0])

            return Preds()

    m = NumpyModel()
    m.compile("sgd", "mse")

    def gen():
        """Function docstring."""
        yield np.array([1.0])

    m.predict(gen(), steps=1, verbose=0)


@patch("ml_switcheroo_compiler.ops.asarray")
def test_1153_predict_iterator_exception(mock_asarray):
    """Function docstring.

    Args:
        mock_asarray: Description.
    """

    class ThrowingModel(Model):
        """Class docstring."""

        def predict_step(self, data):
            """Function docstring.

            Args:
                data: Description.
            """
            return "throw"

    m = ThrowingModel()
    m.compile("sgd", "mse")

    class Thrower:
        """Class docstring."""

        @property
        def data(self):
            """Function docstring."""
            raise ValueError

    mock_asarray.return_value = Thrower()

    def gen():
        """Function docstring."""
        yield np.array([1.0])

    res = m.predict(gen(), steps=1, verbose=0)
    assert res == "throw"
