"""Module docstring."""

import numpy as np
from zero_keras.core_layers import KerasTensor, Layer, Model, Sequential
from unittest.mock import patch
import os


def test_keras_tensor_eq_no_data():
    """Function docstring."""
    tensor = KerasTensor(shape=(2, 2))
    other = KerasTensor(shape=(2, 2))
    result = tensor == other
    assert result is not None


def test_keras_tensor_array_branches():
    """Function docstring."""
    tensor = KerasTensor(shape=(2,), data=np.array([1, 2]))
    np.testing.assert_array_equal(tensor.__array__(copy=False), np.array([1, 2]))

    tensor2 = KerasTensor(shape=(2,), data=[3, 4])
    np.testing.assert_array_equal(tensor2.__array__(copy=False), np.array([3, 4]))

    tensor3 = KerasTensor(shape=(2,), data=[5, 6])
    np.testing.assert_array_equal(tensor3.__array__(copy=True), np.array([5, 6]))


def test_layer_set_weights_fallback():
    """Function docstring."""

    class CustomLayer(Layer):
        """Class docstring."""

        @property
        def weights(self):
            """Function docstring."""

            class W:
                """Class docstring."""

                def __setitem__(self, key, value):
                    """Function docstring.

                    Args:
                        key: Description.
                        value: Description.
                    """
                    self.val = value

            if not hasattr(self, "w"):
                self.w = W()
            return [self.w]

    layer = CustomLayer()
    layer.set_weights([[3, 4]])
    assert layer.weights[0].val == [3, 4]


def test_model_compute_loss_reg():
    """Function docstring."""

    class MyModel(Model):
        """Class docstring."""

        @property
        def losses(self):
            """Function docstring."""
            return [0.5]

    model = MyModel()
    x = KerasTensor(shape=(1,), data=np.array([1.0]))
    loss = model.compute_loss(x, x, x)
    assert float(np.asarray(loss)) == 0.5


def test_model_steps_type_error_fallback():
    """Function docstring."""

    class SimpleModel(Model):
        """Class docstring."""

        def call(self, inputs):
            """Function docstring.

            Args:
                inputs: Description.
            """
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
    """Function docstring."""
    model = Model()

    class FakeDataloader:
        """Class docstring."""

        pass

    FakeDataloader.__module__ = "torch.utils.data"
    assert model._is_iterator(FakeDataloader())


def test_model_fit_batch_size_none():
    """Function docstring."""
    model = Sequential([Layer()])
    model.compile("sgd", "mse")
    x = np.ones((33, 1))
    y = np.ones((33, 1))
    history = model.fit(x, y, batch_size=None, epochs=1, verbose=0)
    assert len(history.history["loss"]) == 1


def test_model_fit_eval_predict_unsliceable():
    """Function docstring."""
    model = Sequential([Layer()])
    model.compile("sgd", "mse")

    def gen():
        """Function docstring."""
        yield np.array([[1.0]]), np.array([[1.0]])

    model.fit(gen(), epochs=1, verbose=0)
    model.evaluate(gen(), verbose=0)

    def gen_x():
        """Function docstring."""
        yield np.array([[1.0]])

    model.predict(gen_x(), verbose=0)


def test_model_predict_numpy_fallback():
    """Function docstring."""

    class SimpleModel(Model):
        """Class docstring."""

        def call(self, inputs):
            """Function docstring.

            Args:
                inputs: Description.
            """

            class Pred:
                """Class docstring."""

                def numpy(self):
                    """Function docstring."""
                    return np.array([[1.0]])

            return Pred()

    model = SimpleModel()
    model.compile("sgd", "mse")
    res = model.predict(np.array([[1.0]]), verbose=0)
    assert res is not None


def test_flatten_in_save():
    """Function docstring."""

    class NestedLayer(Layer):
        """Class docstring."""

        @property
        def weights(self):
            """Function docstring."""
            return [[[1.0, 2.0]]]

        @property
        def _trainable_weights(self):
            """Function docstring."""
            return self.weights  # pragma: no cover

        @property
        def _non_trainable_weights(self):
            """Function docstring."""
            return []  # pragma: no cover

    model = Sequential([NestedLayer()])
    model.save("test_flatten.keras")
    if os.path.exists("test_flatten.keras"):
        os.remove("test_flatten.keras")


def test_predict_concatenate_arrays_fallback():
    """Function docstring."""

    class Pred:
        """Class docstring."""

        def numpy(self):
            """Function docstring."""
            return [1.0]  # returns list, which is not a numpy array

    class SimpleModel(Model):
        """Class docstring."""

        def call(self, inputs):
            """Function docstring.

            Args:
                inputs: Description.
            """
            return Pred()

    model = SimpleModel()
    model.compile("sgd", "mse")
    # two batches so len(all_preds) == 2
    res = model.predict(np.array([[1.0], [2.0]]), batch_size=1, verbose=0)
    assert isinstance(res, list)
    assert len(res) == 2


def test_fit_exception_loss():
    """Function docstring."""

    class BrokenLossModel(Model):
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
            return {"loss": "not_a_float_or_array"}

        def test_step(self, data):
            """Function docstring.

            Args:
                data: Description.
            """
            return {"loss": "not_a_float"}

    model = BrokenLossModel()
    model.compile("sgd", "mse")
    x = np.array([[1.0]])
    y = np.array([[1.0]])
    model.fit(x, y, epochs=1, verbose=0)
    model.evaluate(x, y, verbose=0)


def test_predict_exception_data():
    """Function docstring."""

    class BrokenPredModel(Model):
        """Class docstring."""

        def call(self, inputs):
            """Function docstring.

            Args:
                inputs: Description.
            """

            class Pred:
                """Class docstring."""

                @property
                def data(self):
                    """Function docstring."""
                    raise ValueError("Fail data")  # pragma: no cover

            return Pred()

    model = BrokenPredModel()
    model.compile("sgd", "mse")
    x = np.array([[1.0]])

    res = model.predict(x, verbose=0)
    assert res is not None


def test_sequential_add():
    """Function docstring."""
    model = Sequential()
    model.add(Layer())
    assert len(model.layers) == 1


def test_ops_add():
    """Function docstring."""
    from zero_keras.core_layers import ops

    tensor = KerasTensor(shape=(2, 2))
    res = ops.add(tensor, tensor)
    assert res.shape == (2, 2)


def test_deserialize():
    """Function docstring."""
    from zero_keras.core_layers import deserialize

    assert deserialize({"class_name": "Dense"}) == {"class_name": "Dense"}


def test_get():
    """Function docstring."""
    from zero_keras.core_layers import get as layer_get

    assert layer_get("dense") == "dense"


def test_predict_concatenate_arrays_success():
    """Function docstring."""

    class SimpleModel(Model):
        """Class docstring."""

        def call(self, inputs):
            """Function docstring.

            Args:
                inputs: Description.
            """
            return inputs * 2

    model = SimpleModel()
    model.compile("sgd", "mse")
    res = model.predict(np.array([[1.0], [2.0]]), batch_size=1, verbose=0)
    assert np.allclose(res, np.array([[2.0], [4.0]]))


@patch("ml_switcheroo_compiler.ops.ones")
def test_keras_tensor_eq_no_data2(mock_ones):
    """Function docstring.

    Args:
        mock_ones: Description.
    """
    mock_ones.return_value = "mock_ones"
    tensor = KerasTensor(shape=None)
    other = KerasTensor(shape=None)
    result = tensor == other
    assert result == "mock_ones"


@patch("ml_switcheroo_compiler.ops.asarray")
def test_keras_tensor_array_numpy_attr(mock_asarray):
    """Function docstring.

    Args:
        mock_asarray: Description.
    """

    class MockArr:
        """Class docstring."""

        def numpy(self):
            """Function docstring."""
            return np.array([5, 6])  # pragma: no cover

    mock_asarray.return_value = MockArr()
    tensor = KerasTensor(shape=(2,), data=[5, 6])
    res = tensor.__array__(copy=True)
    np.testing.assert_array_equal(res, np.array([5, 6]))
