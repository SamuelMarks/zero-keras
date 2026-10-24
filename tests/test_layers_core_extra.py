"""Module `test_layers_core_extra` for zero_keras."""

from __future__ import annotations
from zero_keras.types import TensorLike
import numpy as np
import pytest


def test_dense() -> TensorLike:
    """Execute ``test_dense``."""
    from zero_keras.layers.core import Dense

    layer = Dense(32, activation="relu", use_bias=True)
    layer.build((10, 20))
    res = layer(np.ones((10, 20)))
    config = layer.get_config()
    assert config.get("units", layer.units) == 32


def test_dropout() -> TensorLike:
    """Execute ``test_dropout``."""
    from zero_keras.layers.core import Dropout

    layer = Dropout(0.5, noise_shape=(None, 1, 1), seed=42)
    res = layer(np.ones((10, 20)))
    config = layer.get_config()
    assert config.get("rate", layer.rate) == 0.5


def test_flatten() -> TensorLike:
    """Execute ``test_flatten``."""
    from zero_keras.layers.core import Flatten

    layer = Flatten(data_format="channels_first")
    res = layer(np.ones((10, 3, 20, 20)))
    config = layer.get_config()
    assert config.get("data_format", layer.data_format) == "channels_first"


def test_reshape() -> TensorLike:
    """Execute ``test_reshape``."""
    from zero_keras.layers.core import Reshape

    layer = Reshape((10, 2))
    res = layer(np.ones((2, 20)))
    config = layer.get_config()
    assert config.get("target_shape", layer.target_shape) == (10, 2)


def test_permute() -> TensorLike:
    """Execute ``test_permute``."""
    from zero_keras.layers.core import Permute

    layer = Permute((2, 1))
    res = layer(np.ones((10, 20, 30)))
    config = layer.get_config()
    assert config.get("dims", layer.dims) == (2, 1)


def test_repeat_vector() -> TensorLike:
    """Execute ``test_repeat_vector``."""
    from zero_keras.layers.core import RepeatVector

    layer = RepeatVector(3)
    res = layer(np.ones((10, 20)))
    config = layer.get_config()
    assert config.get("n", layer.n) == 3


def test_masking() -> TensorLike:
    """Execute ``test_masking``."""
    from zero_keras.layers.core import Masking

    layer = Masking(mask_value=0.0)
    res = layer(np.ones((10, 20)))
    config = layer.get_config()
    assert config.get("mask_value", layer.mask_value) == 0.0


def test_lambda() -> TensorLike:
    """Execute ``test_lambda``."""
    from zero_keras.layers.core import Lambda

    layer = Lambda(lambda x: x)
    res = layer(np.ones((10, 20)))


def test_identity() -> TensorLike:
    """Execute ``test_identity``."""
    from zero_keras.layers.core import Identity

    layer = Identity()
    res = layer(np.ones((10, 20)))
    config = layer.get_config()


if __name__ == "__main__":
    pytest.main(["-v", "test_layers_core_extra.py"])


def test_input_layer() -> TensorLike:
    """Execute ``test_input_layer``."""
    from zero_keras.layers.core import InputLayer

    layer = InputLayer(input_shape=(10,), batch_size=2)
    assert layer.batch_input_shape == (2, 10)
    layer2 = InputLayer(input_shape=(10,))
    assert layer2.batch_input_shape == (None, 10)
    layer3 = InputLayer(batch_input_shape=(3, 10))
    assert layer3.batch_input_shape == (3, 10)


def test_wrapper_layer() -> TensorLike:
    """Execute ``test_wrapper_layer``."""
    from zero_keras.layers.core import Dense, Wrapper

    dense = Dense(32)
    wrapper = Wrapper(dense)
    wrapper.build((10, 20))
    res = wrapper(np.ones((10, 20)))
    config = wrapper.get_config()
    assert isinstance(config, dict)
