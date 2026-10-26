"""Module `test_layers_core_extra` for zero_keras."""

from __future__ import annotations

import pytest

from zero_keras import ops


def test_dense() -> None:
    """Test ``dense`` behavior and edge cases in ``test_layers_core_extra``."""
    from zero_keras.layers.core import Dense

    layer = Dense(32, activation="relu", use_bias=True)
    layer.build((10, 20))
    _res = layer(ops.ones((10, 20)))
    config = layer.get_config()
    assert config.get("units", layer.units) == 32


def test_dropout() -> None:
    """Test ``dropout`` behavior and edge cases in ``test_layers_core_extra``."""
    from zero_keras.layers.core import Dropout

    layer = Dropout(0.5, noise_shape=(None, 1, 1), seed=42)
    _res = layer(ops.ones((10, 20)))
    config = layer.get_config()
    assert config.get("rate", layer.rate) == 0.5


def test_flatten() -> None:
    """Test ``flatten`` behavior and edge cases in ``test_layers_core_extra``."""
    from zero_keras.layers.core import Flatten

    layer = Flatten(data_format="channels_first")
    _res = layer(ops.ones((10, 3, 20, 20)))
    config = layer.get_config()
    assert config.get("data_format", layer.data_format) == "channels_first"


def test_reshape() -> None:
    """Test ``reshape`` behavior and edge cases in ``test_layers_core_extra``."""
    from zero_keras.layers.core import Reshape

    layer = Reshape((10, 2))
    _res = layer(ops.ones((2, 20)))
    config = layer.get_config()
    assert config.get("target_shape", layer.target_shape) == (10, 2)


def test_permute() -> None:
    """Test ``permute`` behavior and edge cases in ``test_layers_core_extra``."""
    from zero_keras.layers.core import Permute

    layer = Permute((2, 1))
    _res = layer(ops.ones((10, 20, 30)))
    config = layer.get_config()
    assert config.get("dims", layer.dims) == (2, 1)


def test_repeat_vector() -> None:
    """Test ``repeat_vector`` behavior and edge cases in ``test_layers_core_extra``."""
    from zero_keras.layers.core import RepeatVector

    layer = RepeatVector(3)
    _res = layer(ops.ones((10, 20)))
    config = layer.get_config()
    assert config.get("n", layer.n) == 3


def test_masking() -> None:
    """Test ``masking`` behavior and edge cases in ``test_layers_core_extra``."""
    from zero_keras.layers.core import Masking

    layer = Masking(mask_value=0.0)
    _res = layer(ops.ones((10, 20)))
    config = layer.get_config()
    assert config.get("mask_value", layer.mask_value) == 0.0


def test_lambda() -> None:
    """Test ``lambda`` behavior and edge cases in ``test_layers_core_extra``."""
    from zero_keras.layers.core import Lambda

    layer = Lambda(lambda x: x)
    _res = layer(ops.ones((10, 20)))


def test_identity() -> None:
    """Test ``identity`` behavior and edge cases in ``test_layers_core_extra``."""
    from zero_keras.layers.core import Identity

    layer = Identity()
    _res = layer(ops.ones((10, 20)))
    _config = layer.get_config()


if __name__ == "__main__":
    pytest.main(["-v", "test_layers_core_extra.py"])


def test_input_layer() -> None:
    """Test ``input_layer`` behavior and edge cases in ``test_layers_core_extra``."""
    from zero_keras.layers.core import InputLayer

    layer = InputLayer(input_shape=(10,), batch_size=2)
    assert layer.batch_input_shape == (2, 10)
    layer2 = InputLayer(input_shape=(10,))
    assert layer2.batch_input_shape == (None, 10)
    layer3 = InputLayer(batch_input_shape=(3, 10))
    assert layer3.batch_input_shape == (3, 10)


def test_wrapper_layer() -> None:
    """Test ``wrapper_layer`` behavior and edge cases in ``test_layers_core_extra``."""
    from zero_keras.layers.core import Dense, Wrapper

    dense = Dense(32)
    wrapper = Wrapper(dense)
    wrapper.build((10, 20))
    _res = wrapper(ops.ones((10, 20)))
    config = wrapper.get_config()
    assert isinstance(config, dict)
