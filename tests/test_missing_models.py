"""Module docstring."""

import numpy as np
from zero_keras import models
from zero_keras import layers
from zero_keras.core_layers import Input, Functional


def test_clone_model_functional():
    """Function docstring."""
    inp1 = Input(shape=(10,), name="in1")
    inp2 = Input(shape=(10,), name="in2")

    d1 = layers.Dense(5, name="d1")(inp1)
    d2 = layers.Dense(5, name="d2")(inp2)

    concat = layers.Concatenate(name="concat")([d1, d2])
    out = layers.Dense(1, name="out")(concat)

    model = Functional(inputs=[inp1, inp2], outputs=out, name="test_func")

    cloned = models.clone_model(model)
    assert cloned.name == "test_func"
    assert len(cloned.inputs) == 2
    assert isinstance(cloned.outputs, list) or hasattr(cloned.outputs, "shape")

    # Check outputs work
    x1 = np.ones((2, 10), dtype="float32")
    x2 = np.ones((2, 10), dtype="float32")
    y_orig = model([x1, x2])
    y_clone = cloned([x1, x2])


def test_model_from_json_functional():
    """Function docstring."""
    json_str = """
    {
        "class_name": "Functional",
        "config": {
            "name": "model1",
            "layers": [
                {"class_name": "InputLayer", "config": {"name": "input1", "batch_input_shape": [null, 10], "dtype": "float32"}, "name": "input1", "inbound_nodes": []},
                {"class_name": "Dense", "config": {"units": 5, "name": "dense1"}, "name": "dense1", "inbound_nodes": [[["input1", 0, 0, {}]]]}
            ],
            "input_layers": [["input1", 0, 0]],
            "output_layers": [["dense1", 0, 0]]
        }
    }
    """
    model = models.model_from_json(json_str)
    assert model.name == "model1"
    x = np.ones((2, 10), dtype="float32")
    y = model(x)
    assert y.shape == (2, 5)
