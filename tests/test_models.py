"""Comprehensive unit and branch coverage tests for zero_keras.models."""

from __future__ import annotations

import json
import pytest

from zero_keras import core_layers, models, ops, optimizers, random
from zero_keras.core_layers import Layer, Model
from zero_keras.types import TensorLike


class DummyLayer(Layer):
    """Test helper class `DummyLayer` for models."""

    def get_config(self) -> TensorLike:
        """Return the serializable configuration dictionary for ``DummyLayer``.

        Returns:
            TensorLike: Result from `get_config`.
        """
        return {"name": "dummy"}

    @classmethod
    def from_config(cls, config: TensorLike) -> TensorLike:
        """Reconstruct a ``DummyLayer`` instance from its configuration dictionary.

        Args:
            config: Serialized configuration dictionary.

        Returns:
            TensorLike: Result from `from_config`.
        """
        _ = config
        return cls()


def test_clone_model() -> None:
    """Test ``clone_model`` behavior and edge cases in ``test_models``."""
    seq = models.Sequential()
    seq.add(DummyLayer())

    cloned = models.clone_model(seq)
    assert isinstance(cloned, models.Sequential)
    assert len(cloned.layers) == 1

    base = Model()
    assert models.clone_model(base) is base

    with pytest.raises(ValueError):
        models.clone_model("not a model")


def test_save_model(tmp_path: TensorLike) -> None:
    """Test ``save_model`` behavior and edge cases in ``test_models``.

    Args:
        tmp_path: Pytest temporary directory ``Path`` fixture.
    """
    seq = models.Sequential()
    seq.add(DummyLayer())

    fp = tmp_path / "model.keras"
    models.save_model(seq, fp)
    assert fp.exists()

    base = Model()
    fp2 = tmp_path / "model2.keras"
    models.save_model(base, fp2)
    assert fp2.exists()


def test_model_from_json() -> None:
    """Test ``model_from_json`` behavior and edge cases in ``test_models``."""
    seq_config = {
        "class_name": "Sequential",
        "config": {
            "name": "seq1",
            "layers": [{"class_name": "Dense", "config": {"units": 10}}],
        },
    }
    model = models.model_from_json(json.dumps(seq_config))
    assert isinstance(model, models.Sequential)
    assert (model._name if hasattr(model, "_name") else None) == "seq1"

    base_config = {"class_name": "Model", "config": {"name": "model1"}}
    base = models.model_from_json(json.dumps(base_config))
    assert isinstance(base, Model)


def test_optimizers_deserialize_fallback() -> None:
    """Test ``optimizers_deserialize_fallback`` behavior and edge cases in ``test_models``."""
    assert optimizers.deserialize(123) == 123
    assert optimizers.deserialize({"class_name": "UnknownOpt"}) == {"class_name": "UnknownOpt"}


def test_models_clone_function_fallback() -> None:
    """Test ``models_clone_function_fallback`` behavior and edge cases in ``test_models``."""

    class MockLayer:
        """Test helper class `MockLayer` for missing coverage 1."""

    mock_layer = MockLayer()
    model = models.Sequential([mock_layer])
    cloned = models.clone_model(model)
    assert cloned.layers[0] is mock_layer


def test_models_save_model_fallback() -> None:
    """Test ``models_save_model_fallback`` behavior and edge cases in ``test_models``."""

    class MockModel:
        """Test helper class `MockModel` for missing coverage 1."""

    mock_model = MockModel()
    with pytest.raises(Exception):
        models.save_model(mock_model, "fake_path.keras")


def test_models_model_from_json_builtins() -> None:
    """Test ``models_model_from_json_builtins`` behavior and edge cases in ``test_models``."""
    config = {
        "class_name": "Functional",
        "config": {
            "name": "test",
            "layers": [
                {
                    "name": "input_1",
                    "class_name": "InputLayer",
                    "config": {"batch_input_shape": [None, 10], "dtype": "float32"},
                },
                {"name": "dict_layer", "class_name": "dict", "config": {}},
            ],
            "input_layers": [["input_1", 0, 0]],
            "output_layers": [["input_1", 0, 0]],
        },
    }
    json_str_with_dict = json.dumps(config)
    m = models.model_from_json(json_str_with_dict, custom_objects={"dict": dict})
    assert m.name == "test"


def test_models_model_from_json_multiple_connections() -> None:
    """Test ``models_model_from_json_multiple_connections`` behavior and edge cases in ``test_models``."""
    config = {
        "class_name": "Functional",
        "config": {
            "name": "test",
            "layers": [
                {
                    "name": "input_1",
                    "class_name": "InputLayer",
                    "config": {"batch_input_shape": [None, 10], "dtype": "float32"},
                    "inbound_nodes": [],
                },
                {
                    "name": "input_2",
                    "class_name": "InputLayer",
                    "config": {"batch_input_shape": [None, 10], "dtype": "float32"},
                    "inbound_nodes": [],
                },
                {
                    "name": "concat",
                    "class_name": "Concatenate",
                    "config": {},
                    "inbound_nodes": [[["input_1", 0, 0, {}], ["input_2", 0, 0, {}]]],
                },
            ],
            "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]],
            "output_layers": [["concat", 0, 0]],
        },
    }
    json_str = json.dumps(config)

    m = models.model_from_json(json_str)
    assert m.name == "test"


def test_models_model_from_json_fallback() -> None:
    """Test ``models_model_from_json_fallback`` behavior and edge cases in ``test_models``."""
    json_str = '{"class_name": "UnknownModel", "config": {}}'
    m = models.model_from_json(json_str)
    assert type(m).__name__ == "Model"


def test_model_list_outputs() -> None:
    """Test ``model_list_outputs`` behavior and edge cases in ``test_models``."""
    inp = core_layers.Input(shape=(2,))

    class ListOutputLayer(core_layers.Layer):
        """Test helper class `ListOutputLayer` for missing coverage 5."""

        def call(self, x: TensorLike) -> TensorLike:
            """Execute the forward computation of ``ListOutputLayer``.

            Args:
                x: Input tensor or feature array.

            Returns:
                TensorLike: Result from `call`.
            """
            return [x, x]

    layer_out = ListOutputLayer()(inp)

    m = models.Model(inputs=inp, outputs=layer_out)
    res = m(ops.zeros((1, 2)))
    assert isinstance(res, list)


def test_model_dict_outputs() -> None:
    """Test ``model_dict_outputs`` behavior and edge cases in ``test_models``."""
    inp = core_layers.Input(shape=(2,))

    class DictOutputLayer(core_layers.Layer):
        """Test helper class `DictOutputLayer` for missing coverage 5."""

        def call(self, x: TensorLike) -> TensorLike:
            """Execute the forward computation of ``DictOutputLayer``.

            Args:
                x: Input tensor or feature array.

            Returns:
                TensorLike: Result from `call`.
            """
            return {"a": x, "b": x}

    layer_out = DictOutputLayer()(inp)

    m = models.Model(inputs=inp, outputs=layer_out)
    res = m(ops.zeros((1, 2)))
    assert isinstance(res, dict)
    assert "a" in res


def test_model_sequential_functional_and_deserialization_unit() -> None:
    """Verify Sequential, Functional, Model training API, and deserialization without numpy."""
    z_seq = core_layers.Sequential()
    z_seq.add(core_layers.Layer())
    assert len(z_seq.layers) == 1

    x = random.uniform((2, 5))
    res = z_seq(x)
    assert res.shape == x.shape

    z_in = core_layers.Input(shape=(5,))
    z_out = core_layers.Layer()(z_in)
    z_func = core_layers.Functional(inputs=z_in, outputs=z_out)
    res_f = z_func(x)
    assert (
        res_f.shape if hasattr(res_f, "shape") else None
    ) == z_in.shape or res_f.shape == x.shape

    m = core_layers.Model()
    m.compile(optimizer="sgd", loss="mse")
    assert m._compiled

    x_1d = ops.convert_to_tensor([1.0, 2.0, 3.0])
    y_1d = ops.convert_to_tensor([1.0, 2.0, 3.0])
    hist = m.fit(x_1d, y_1d, epochs=2)
    assert len(hist.history["loss"]) == 2
    ev = m.evaluate(x_1d, y_1d)
    assert "loss" in ev
    pred = m.predict(x_1d)
    assert pred.shape[0] == 3

    from zero_keras.core_layers import deserialize
    from zero_keras.layers import Dense

    assert core_layers.get("Dense") == Dense
    cfg = {"class_name": "Dense", "config": {"units": 10}}
    obj = core_layers.get(cfg)
    assert isinstance(obj, Dense)
    assert obj.units == 10

    with pytest.raises(ValueError, match="safe_mode is True"):
        deserialize({"class_name": "Lambda", "config": {}}, safe_mode=True)

    with pytest.raises(ValueError):
        core_layers.get("unknown_layer_test")

    with pytest.raises(ValueError, match="Legacy HDF5 format is not supported"):
        models.load_model("nonexistent.h5")

    with pytest.raises(ValueError, match="File format not supported"):
        models.load_model("nonexistent.unknown")
