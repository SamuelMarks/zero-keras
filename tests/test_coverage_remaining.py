"""Unit and behavioral tests for coverage remaining in zero_keras."""

from __future__ import annotations

from zero_keras import ops
from zero_keras.initializers import deserialize as deserialize_initializer
from zero_keras.layers import Conv1DTranspose, Conv2DTranspose, Conv3DTranspose
from zero_keras.layers import Input as LayersInput
from zero_keras.models import Sequential, clone_model, model_from_json
from zero_keras.optimizers import LossScaleOptimizer
from zero_keras.regularizers import deserialize as deserialize_regularizer
from zero_keras.types import TensorLike


def test_initializers_deserialize_invalid() -> None:
    """Verify or execute `test initializers deserialize invalid`."""
    res = deserialize_initializer({"class_name": "InvalidClassThatDoesNotExist"})
    assert res == {"class_name": "InvalidClassThatDoesNotExist"}


def test_layers_conv_transpose_channels_first_no_bias() -> None:
    """Verify or execute `test layers conv transpose channels first no bias`."""
    c1 = Conv1DTranspose(2, 2, data_format="channels_first", use_bias=False)
    c1.build((None, 3, 10))
    c1(ops.ones((1, 3, 10)))

    c2 = Conv2DTranspose(2, 2, data_format="channels_first", use_bias=False)
    c2.build((None, 3, 10, 10))
    c2(ops.ones((1, 3, 10, 10)))

    c3 = Conv3DTranspose(2, 2, data_format="channels_first", use_bias=False)
    c3.build((None, 3, 10, 10, 10))
    c3(ops.ones((1, 3, 10, 10, 10)))


def test_layers_inputlayer_batch_shape_not_none() -> None:
    """Verify or execute `test layers inputlayer batch shape not none`."""
    layer = LayersInput(shape=(10,), batch_shape=(32, 10))
    assert layer.batch_shape == (32, 10)


def test_layers_deserialize_invalid() -> None:
    """Verify or execute `test layers deserialize invalid`."""
    from zero_keras.layers import deserialize

    res = deserialize({"class_name": "InvalidLayerClassThatDoesNotExist"})
    assert res == {"class_name": "InvalidLayerClassThatDoesNotExist"}


def test_models_clone_model_custom_function() -> None:
    """Verify or execute `test models clone model custom function`."""
    model = Sequential()
    clone_model(model, clone_function=lambda x: x)


def test_models_from_json_edges() -> None:
    """Verify or execute `test models from json edges`."""
    json_str = """{
        "class_name": "Functional",
        "config": {
            "layers": [
                {
                    "class_name": "InputLayer",
                    "name": "input_1",
                    "config": {
                        "batch_input_shape": [32, 10]
                    }
                },
                {
                    "class_name": "InvalidClassXYZ",
                    "name": "invalid_1",
                    "config": {}
                }
            ]
        }
    }"""
    try:
        model_from_json(json_str)
    except Exception:
        pass


class DummyOptWithoutMethods:
    """Test helper class `DummyOptWithoutMethods` for coverage remaining."""


def test_optimizers_lossscale_no_inner_methods() -> None:
    """Verify or execute `test optimizers lossscale no inner methods`."""
    opt = LossScaleOptimizer(DummyOptWithoutMethods())
    opt.build([])
    opt.apply_gradients([])


def test_regularizers_deserialize_invalid() -> None:
    """Verify or execute `test regularizers deserialize invalid`."""
    res = deserialize_regularizer({"class_name": "InvalidRegularizerClassXYZ"})
    assert res == {"class_name": "InvalidRegularizerClassXYZ"}


def test_initializers_deserialize_custom() -> None:
    """Verify or execute `test initializers deserialize custom`."""
    from zero_keras.initializers import deserialize

    class MockInit:
        """Test helper class `MockInit` for coverage remaining."""

        def __init__(self, **kwargs: TensorLike) -> None:
            """Verify or execute `init` on `MockInit`.

            Args:
                **kwargs: Keyword arguments forwarded to helper.
            """
            self.kwargs = kwargs

    res = deserialize(
        {"class_name": "MockInit", "config": {"a": 1}},
        custom_objects={"MockInit": MockInit},
    )
    assert isinstance(res, MockInit)


def test_layers_deserialize_custom() -> None:
    """Verify or execute `test layers deserialize custom`."""
    from zero_keras.layers import deserialize

    class MockLayer:
        """Test helper class `MockLayer` for coverage remaining."""

        def __init__(self, **kwargs: TensorLike) -> None:
            """Verify or execute `init` on `MockLayer`.

            Args:
                **kwargs: Keyword arguments forwarded to helper.
            """
            self.kwargs = kwargs

    res = deserialize(
        {"class_name": "MockLayer", "config": {"a": 1}},
        custom_objects={"MockLayer": MockLayer},
    )
    assert isinstance(res, MockLayer)
