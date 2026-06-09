"""Tests for zero_keras activations."""

import numpy as np
from zero_keras import activations


def test_activations():
    x = np.array([-1.0, 0.0, 1.0, 2.0])

    assert activations.celu(x).shape == x.shape
    assert activations.elu(x).shape == x.shape
    assert activations.exponential(x).shape == x.shape
    assert activations.gelu(x).shape == x.shape
    assert activations.gelu(x, approximate=True).shape == x.shape

    # glu needs an even sized dimension along axis
    x_glu = np.array([[-1.0, 1.0], [0.0, 2.0]])
    assert activations.glu(x_glu, axis=-1).shape == (2, 1)

    assert activations.hard_shrink(x).shape == x.shape
    assert activations.hard_sigmoid(x).shape == x.shape
    assert activations.hard_silu(x).shape == x.shape
    assert activations.hard_swish(x).shape == x.shape
    assert activations.hard_tanh(x).shape == x.shape
    assert activations.leaky_relu(x).shape == x.shape
    assert activations.linear(x).shape == x.shape
    assert activations.log_sigmoid(x).shape == x.shape
    assert activations.log_softmax(x).shape == x.shape
    assert activations.mish(x).shape == x.shape
    assert activations.relu(x, max_value=1.0).shape == x.shape
    assert activations.relu6(x).shape == x.shape
    assert activations.selu(x).shape == x.shape
    assert activations.sigmoid(x).shape == x.shape
    assert activations.silu(x).shape == x.shape
    assert activations.soft_shrink(x).shape == x.shape
    assert activations.softmax(x).shape == x.shape
    assert activations.softplus(x).shape == x.shape
    assert activations.softsign(x).shape == x.shape
    assert activations.sparse_plus(x).shape == x.shape
    assert activations.sparse_sigmoid(x).shape == x.shape
    assert activations.sparsemax(x).shape == x.shape
    assert activations.sparsemax(np.array([[1.0, 2.0], [3.0, 4.0]])).shape == (2, 2)
    assert activations.squareplus(x).shape == x.shape
    assert activations.swish(x).shape == x.shape
    assert activations.tanh(x).shape == x.shape
    assert activations.tanh_shrink(x).shape == x.shape
    assert activations.threshold(x, 0.5, 0.0).shape == x.shape


def test_deserialize():
    assert activations.deserialize("relu") == activations.relu
    assert activations.deserialize({"class_name": "relu"}) == {"class_name": "relu"}


def test_serialize():
    assert activations.serialize(activations.relu) == "relu"
    assert activations.serialize("relu") == "relu"


def test_get():
    assert activations.get(None) == activations.linear
    assert activations.get("relu") == activations.relu
    assert activations.get("nonexistent") == activations.linear
    assert activations.get(activations.celu) == activations.celu
    assert activations.get(123) == activations.linear
