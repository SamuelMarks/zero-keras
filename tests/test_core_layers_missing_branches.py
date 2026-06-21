from zero_keras.core_layers import TensorShape, Node, Functional, Input
from zero_keras.layers import Dense, Layer


def test_tensorshape_iterable():
    ts = TensorShape((1, 2))
    assert ts.dims == [1, 2]


def test_node_outputs_dict_or_no_keras_history():
    layer = Dense(1)
    node = Node(layer, (), {}, outputs={"out": 1})
    assert len(layer._inbound_nodes) == 1


def test_functional_reused_layer():
    inp = Input(shape=(10,))
    d = Dense(10)
    x = d(inp)
    y = d(x)
    model = Functional(inputs=inp, outputs=y)
    assert len(model.layers) == 1


def test_functional_empty_inputs():
    # Cover 1339->exit (add_input with non-KerasTensor)
    model = Functional(inputs=["not_a_tensor"], outputs=[])
    model.call(["val"])


class DummyLayerListOutput(Layer):
    def call(self, inputs):
        return [inputs, "not_a_tensor"]

    def compute_output_shape(self, input_shape):
        return [input_shape, None]


class DummyLayerDictOutput(Layer):
    def call(self, inputs):
        return {"a": inputs, "b": "not_a_tensor"}

    def compute_output_shape(self, input_shape):
        return {"a": input_shape, "b": None}


class DummyLayerScalarOutput(Layer):
    def call(self, inputs):
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape


def test_functional_node_outputs_not_keras_tensor():
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
