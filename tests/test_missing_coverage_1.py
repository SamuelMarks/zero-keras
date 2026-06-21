import pytest
from zero_keras import optimizers, models
import json


def test_optimizers_deserialize_fallback():
    # If the identifier is a float or unhandled dict
    assert optimizers.deserialize(123) == 123
    assert optimizers.deserialize({"class_name": "UnknownOpt"}) == {
        "class_name": "UnknownOpt"
    }


def test_models_clone_function_fallback():
    # Model cloning where a layer has no get_config
    class MockLayer:
        pass

    mock_layer = MockLayer()
    model = models.Sequential([mock_layer])
    # The default clone function should return the layer itself
    cloned = models.clone_model(model)
    assert cloned.layers[0] is mock_layer


def test_models_save_model_fallback():
    class MockModel:
        pass

    mock_model = MockModel()
    with pytest.raises(Exception):
        models.save_model(mock_model, "fake_path.keras")


def test_models_model_from_json_builtins():
    # test falling back to builtins
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
    with pytest.raises(AttributeError):
        models.model_from_json(json_str_with_dict)


def test_models_model_from_json_multiple_connections():
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


def test_models_model_from_json_fallback():
    json_str = '{"class_name": "UnknownModel", "config": {}}'
    m = models.model_from_json(json_str)
    assert type(m).__name__ == "Model"
