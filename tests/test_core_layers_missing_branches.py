"""Module docstring."""

from zero_keras.core_layers import TensorShape, Node, Functional, Input
from zero_keras.layers import Dense, Layer


def test_tensorshape_iterable():
    """Function docstring."""
    ts = TensorShape((1, 2))
    assert ts.dims == [1, 2]


def test_node_outputs_dict_or_no_keras_history():
    """Function docstring."""
    layer = Dense(1)
    node = Node(layer, (), {}, outputs={"out": 1})
    assert len(layer._inbound_nodes) == 1


def test_functional_reused_layer():
    """Function docstring."""
    inp = Input(shape=(10,))
    d = Dense(10)
    x = d(inp)
    y = d(x)
    model = Functional(inputs=inp, outputs=y)
    assert len(model.layers) == 1


def test_functional_empty_inputs():
    """Function docstring."""
    # Cover 1339->exit (add_input with non-KerasTensor)
    model = Functional(inputs=["not_a_tensor"], outputs=[])
    model.call(["val"])


class DummyLayerListOutput(Layer):
    """Class docstring."""

    def call(self, inputs):
        """Function docstring.

        Args:
            inputs: Description.
        """
        return [inputs, "not_a_tensor"]

    def compute_output_shape(self, input_shape):
        """Function docstring.

        Args:
            input_shape: Description.
        """
        return [input_shape, None]  # pragma: no cover


class DummyLayerDictOutput(Layer):
    """Class docstring."""

    def call(self, inputs):
        """Function docstring.

        Args:
            inputs: Description.
        """
        return {"a": inputs, "b": "not_a_tensor"}

    def compute_output_shape(self, input_shape):
        """Function docstring.

        Args:
            input_shape: Description.
        """
        return {"a": input_shape, "b": None}  # pragma: no cover


class DummyLayerScalarOutput(Layer):
    """Class docstring."""

    def call(self, inputs):
        """Function docstring.

        Args:
            inputs: Description.
        """
        return inputs

    def compute_output_shape(self, input_shape):
        """Function docstring.

        Args:
            input_shape: Description.
        """
        return input_shape  # pragma: no cover


def test_functional_node_outputs_not_keras_tensor():
    """Function docstring."""
    inp = Input(shape=(10,))

    # 1370->1369
    l1 = DummyLayerListOutput()(inp)
    # 1374->1373
    l2 = DummyLayerDictOutput()(inp)

    # For 1377->1352, node.outputs must NOT be a KerasTensor, list/tuple, or dict.
    # Wait, if it's not a KerasTensor, how can we trace it?
    # We can manually set the node's outputs!
    l3_tensor = DummyLayerScalarOutput()(inp)
    node3 = l3_tensor._keras_history
    node3.outputs = "not_a_tensor"

    model = Functional(
        inputs=inp, outputs={"l1": l1[0], "l2": l2["a"], "l3": l3_tensor}
    )
    import numpy as np

    x = np.ones((1, 10))
    out = model(x)
