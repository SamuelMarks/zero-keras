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


def test_function_methods():
    from zero_keras.core_layers import Function

    fn = Function()
    assert fn.call(1) is None
    assert fn.compute_output_shape((1, 2)) == (1, 2)
    assert fn.compute_output_spec(1) is None
    new_fn = Function.from_config({})
    assert new_fn.get_config() == {}
    assert fn.input is None
    assert fn.inputs == []
    assert fn.operations == []
    assert fn.output is None
    assert fn.outputs == []
    assert fn.quantized_call(1) is None
    assert fn.symbolic_call(1) is None


def test_input_spec_methods():
    from zero_keras.core_layers import InputSpec

    spec = InputSpec(name="test")
    config = spec.get_config()
    assert config["name"] == "test"
    new_spec = InputSpec.from_config(config)
    assert new_spec.name == "test"


def test_keras_tensor_methods():
    from zero_keras.core_layers import KerasTensor
    import pytest

    t = KerasTensor(shape=(1, 2, 1), dtype="float32")
    with pytest.raises(TypeError):
        iter(t)
    assert t.ndim == 3
    assert not t.ragged
    assert not t.sparse
    t2 = t.reshape((2, 1))
    assert t2.shape == (2, 1)
    t3 = t.squeeze()
    assert t3.shape == (2,)
    t4 = KerasTensor(shape=None)
    t5 = t4.squeeze()
    assert t5.shape is None


def test_layer_methods_extra():
    from zero_keras.core_layers import Layer

    layer = Layer(name="test_layer")
    layer.add_metric(1.0, name="metric")
    var = layer.add_variable(shape=(1,), initializer="zeros")
    assert var.shape == (1,)
    layer.build_from_config({"input_shape": (1,)})
    assert layer.compute_dtype is None
    assert layer.compute_mask(None, mask=True) is True
    assert layer.compute_output_shape((1,)) == (1,)
    assert layer.compute_output_spec(1) is None
    assert layer.count_params() == 0
    assert layer.dtype is None
    assert layer.dtype_policy is None
    assert layer.get_build_config() == {}
    assert layer.get_config()["name"] == "test_layer"
    assert layer.input is None
    assert layer.input_dtype is None
    assert layer.input_spec is None
    layer.load_own_variables({})
    assert layer.metrics == []
    assert layer.metrics_variables == []
    assert layer.non_trainable_variables == []
    assert layer.output is None
    assert layer.path == "test_layer"
    assert layer.quantization_mode is None
    layer.quantize("int8")
    layer.quantized_build((1,))
    assert layer.quantized_call(1) == 1
    assert layer.rematerialized_call(1) == 1
    layer.save_own_variables({})
    assert layer.stateless_call([], [], 1) == 1
    assert not layer.supports_masking
    layer.supports_masking = True
    assert layer.supports_masking

    assert layer.symbolic_call(1) == 1

    assert layer.trainable is True
    layer.trainable = False
    assert layer.trainable is False
    assert layer.trainable_variables == []
    assert layer.variable_dtype is None
    assert layer.variables == []


def test_model_methods_extra():
    from zero_keras.core_layers import Model

    model = Model()
    model.compile_from_config({})
    assert model.compiled_loss is None
    assert model.compiled_metrics is None
    assert model.distribute_reduction_method is None
    assert model.distribute_strategy is None
    model.export("test")
    assert model.get_compile_config() == {}
    assert model.get_layer() is None
    assert model.get_metrics_result() == {}
    assert model.get_state_tree() == {}
    model.jit_compile()
    assert model.layers == []
    assert model.loss is None
    assert model.make_predict_function() is None
    assert model.make_test_function() is None
    assert model.make_train_function() is None
    assert model.metrics_names == []
    assert model.predict_on_batch(1) == 1
    model.reset_metrics()
    assert not model.run_eagerly
    model.run_eagerly = True
    assert model.run_eagerly
    model.set_state_tree({})
    assert model.stateless_compute_loss() is None
    model.summary()
    assert model.test_on_batch() is None
    assert isinstance(model.to_json(), str)
    assert model.train_on_batch() is None


def test_operation_methods_extra():
    from zero_keras.core_layers import Operation

    op = Operation()
    assert op.call() is None
    assert op.compute_output_spec() is None
    new_op = Operation.from_config({})
    assert new_op.get_config() == {}
    assert op.input is None
    assert op.output is None
    assert op.quantized_call() is None
    assert op.symbolic_call() is None


def test_quantizer_regularizer_methods():
    from zero_keras.core_layers import Quantizer
    from zero_keras.regularizers import Regularizer

    q = Quantizer()
    assert q.get_config() == {}
    q2 = Quantizer.from_config({})
    assert isinstance(q2, Quantizer)

    r = Regularizer()
    assert r.get_config() == {}
    r2 = Regularizer.from_config({})
    assert isinstance(r2, Regularizer)


def test_stateless_variable_methods():
    from zero_keras.core_layers import StatelessScope, Variable

    with StatelessScope() as scope:
        scope.add_loss(0.1)
        scope.add_update(1)

    var = Variable("zeros")
    assert scope.get_current_value(var) is None

    assert var[0] is None
    var.aggregation = "mean"
    assert var.aggregation == "mean"
    var.assign(1)
    assert var.value == 1
    var.assign_add(1)
    var.assign_sub(1)
    var.constraint = "none"
    assert var.constraint == "none"
    var.handle = 1
    assert var.handle == 1
    assert var.ndim == 0
    assert var.numpy() == 1
    var.overwrite_with_gradient(1)
    assert var.path == var.name
    var.regularizer = "l2"
    assert var.regularizer == "l2"
    var.synchronization = "auto"
    assert var.synchronization == "auto"


def test_sequential_methods_extra():
    from zero_keras.core_layers import Sequential

    seq = Sequential()
    seq.add_metric(1)
    seq.add_variable((1,), "zeros")
    seq.build_from_config({"input_shape": (1,)})
    seq.compile_from_config({})
    assert seq.compiled_loss is None
    assert seq.compiled_metrics is None
    assert seq.compute_dtype is None
    assert seq.compute_mask(1, mask=1) == 1
    assert seq.compute_output_shape((1,)) == (1,)
    assert seq.compute_output_spec(1) is None
    assert seq.count_params() == 0
    assert seq.distribute_reduction_method is None
    assert seq.distribute_strategy is None
    assert seq.dtype is None
    assert seq.dtype_policy is None
    seq.export("path")
    assert seq.get_build_config() == {}
    assert seq.get_compile_config() == {}
    assert seq.get_layer() is None
    assert seq.get_metrics_result() == {}
    assert seq.get_state_tree() == {}
    assert seq.input is None
    assert seq.input_dtype is None
    assert seq.input_spec is None
    seq.jit_compile()
    assert seq.layers == []
    seq.load_own_variables({})
    assert seq.loss is None
    assert seq.make_predict_function() is None
    assert seq.make_test_function() is None
    assert seq.make_train_function() is None
    assert seq.metrics == []
    assert seq.metrics_names == []
    assert seq.metrics_variables == []
    assert seq.non_trainable_variables == []
    assert seq.output is None
    assert seq.path == seq.name
    assert seq.predict_on_batch(1) == 1
    assert seq.quantization_mode is None
    seq.quantize("int8")
    seq.quantized_build((1,))
    assert seq.quantized_call(1) == 1
    assert seq.rematerialized_call(1) == 1
    seq.reset_metrics()
    assert not seq.run_eagerly
    seq.save_own_variables({})
    seq.set_state_tree({})
    assert seq.stateless_call([], [], 1) == 1
    assert seq.stateless_compute_loss(1) is None
    seq.summary()
    assert not seq.supports_masking
    assert seq.symbolic_call(1) == 1
    assert seq.test_on_batch() is None
    assert isinstance(seq.to_json(), str)
    assert seq.train_on_batch() is None
    assert seq.trainable
    assert seq.trainable_variables == []
    assert seq.variable_dtype is None
    assert seq.variables == []
