"""Tests for zero_keras model parity with Keras."""

import numpy as np
import pytest
import keras
from zero_keras import core_layers
from .utils import set_seed


@pytest.fixture(autouse=True)
def _set_seed():
    set_seed(42)


def test_keras_tensor_parity():
    # KerasTensor is just a metadata holder in keras. Check shape/dtype match
    keras_input = keras.Input(shape=(10,), name="test_in")
    zero_input = core_layers.Input(shape=(10,), name="test_in")

    assert keras_input.shape[1:] == zero_input.shape  # Keras prepends None for batch
    assert zero_input.name == "test_in"


def test_model_Sequential():
    # Test our basic mocked architecture
    z_seq = core_layers.Sequential()
    z_seq.add(core_layers.Layer())

    assert len(z_seq.layers) == 1

    x = np.random.rand(2, 5)
    res = z_seq(x)
    assert res.shape == x.shape


def test_model_Functional():
    # Test our basic mocked architecture
    z_in = core_layers.Input(shape=(5,))
    z_out = core_layers.Layer()(z_in)
    z_func = core_layers.Functional(inputs=z_in, outputs=z_out)

    x = np.random.rand(2, 5)
    res = z_func(x)
    # the dummy layer just passes through
    assert getattr(res, "shape", None) == z_in.shape or res.shape == x.shape


def test_model_Model_API():
    m = core_layers.Model()
    m.compile(optimizer="sgd", loss="mse")
    assert m._compiled

    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])

    res = m.fit(x, y, epochs=2)
    assert len(res.history["loss"]) == 2

    res = m.evaluate(x, y)
    assert "loss" in res

    res = m.predict(x)
    assert res.shape[0] == 3


def test_get_and_deserialize():
    # Basic structural mock
    assert core_layers.get("test") == "test"
    assert core_layers.deserialize({"class_name": "Test"}) == {"class_name": "Test"}


def test_load_model_safe_stub():
    # just cover the fallback stub
    from zero_keras.models import load_model

    m = load_model("nonexistent.h5")
    assert m is not None
