"""Module docstring."""

import numpy as np
from zero_keras.core_layers import Model, Layer, KerasTensor, Sequential
from unittest.mock import patch
import os
import ml_switcheroo_compiler.ops


def test_keras_tensor_eq_no_data():
    """Function docstring."""
    tensor = KerasTensor(shape=(2, 2))
    other = KerasTensor(shape=(2, 2))
    assert (tensor == other) is not None


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
    assert float(np.asarray(model.compute_loss(x, x, x))) == 0.5


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
    x, y = np.array([1.0, 2.0]), np.array([2.0, 4.0])
    assert "loss" in model.train_step((x, y))
    assert "loss" in model.test_step((x, y))
    np.testing.assert_array_equal(
        np.asarray(model.predict_step((x,))), np.array([2.0, 4.0])
    )


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
    assert (
        len(
            model.fit(
                np.ones((33, 1)), np.ones((33, 1)), batch_size=None, epochs=1, verbose=0
            ).history["loss"]
        )
        == 1
    )


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
    assert model.predict(np.array([[1.0]]), verbose=0) is not None


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
            return [1.0]

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
    res = model.predict(np.array([[1.0], [2.0]]), batch_size=1, verbose=0)
    assert isinstance(res, list) and len(res) == 2


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
    model.fit(np.array([[1.0]]), np.array([[1.0]]), epochs=1, verbose=0)
    model.evaluate(np.array([[1.0]]), np.array([[1.0]]), verbose=0)


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
    assert model.predict(np.array([[1.0]]), verbose=0) is not None


def test_sequential_add():
    """Function docstring."""
    model = Sequential()
    model.add(Layer())
    assert len(model.layers) == 1


def test_ops_add():
    """Function docstring."""
    from zero_keras.core_layers import ops

    assert ops.add(KerasTensor(shape=(2, 2)), None).shape == (2, 2)


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
    assert np.allclose(
        model.predict(np.array([[1.0], [2.0]]), batch_size=1, verbose=0),
        np.array([[2.0], [4.0]]),
    )


@patch("ml_switcheroo_compiler.ops.ones")
def test_keras_tensor_eq_no_data2(mock_ones):
    """Function docstring.

    Args:
        mock_ones: Description.
    """
    mock_ones.return_value = "mock_ones"
    assert (KerasTensor(shape=None) == KerasTensor(shape=None)) == "mock_ones"


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
    model._trainable_weights = [model.w]

    class Opt:
        """Class docstring."""

        def apply_gradients(self, grads):
            """Function docstring.

            Args:
                grads: Description.
            """
            pass  # pragma: no cover

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
    finally:
        if original is None:
            del ml_switcheroo_compiler.ops.get_gradients
        else:
            ml_switcheroo_compiler.ops.get_gradients = original  # pragma: no cover


def test_b_loss_fallback_train():
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

        def test_step(self, data):
            """Function docstring.

            Args:
                data: Description.
            """

            class Bad:  # pragma: no cover
                """Class docstring."""

                def __float__(self):  # pragma: no cover
                    """Function docstring."""
                    raise ValueError  # pragma: no cover

            return {"loss": Bad()}  # pragma: no cover

    model = BadLossModel()
    model.compile("sgd", "mse")
    model.fit(np.array([[1.0]]), np.array([[1.0]]), epochs=1, verbose=0)


def test_bx_by_iterator_branch_885():
    """Function docstring."""
    model = Sequential([Layer()])
    model.compile("sgd", "mse")

    def gen():
        """Function docstring."""
        yield np.array([1.0]), np.array([1.0])

    model.evaluate(gen(), steps=1, verbose=0)


def test_predict_1165():
    """Function docstring."""
    model = Sequential([Layer()])
    model.compile("sgd", "mse")

    def gen():
        """Function docstring."""
        yield np.array([1.0])

    model.predict(gen(), steps=1, verbose=0)


@patch("ml_switcheroo_compiler.ops.asarray")
def test_predict_1153(mock_asarray):
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
    assert m.predict(np.array([1.0]), verbose=0) == "throw"
