"""Module `test_misc_extra` for zero_keras."""

from __future__ import annotations
from zero_keras.types import TensorLike
import numpy as np

from zero_keras import layers
from zero_keras.models import Sequential


def test_timedistributed_built_twice() -> TensorLike:
    """Execute ``test_timedistributed_built_twice``."""
    layer = layers.TimeDistributed(layers.Dense(4))
    layer.build((2, 5, 3))
    layer.build((2, 5, 3))
    assert layer.built


def test_sequential_call_with_layers() -> TensorLike:
    """Execute ``test_sequential_call_with_layers``."""
    layer = Sequential([layers.Dense(4)])
    x = np.random.rand(2, 3).astype(np.float32)
    out = layer(x)
    assert out.shape == (2, 4)


def test_deserialize_custom_objects() -> TensorLike:
    """Execute ``test_deserialize_custom_objects``."""

    class MyCustomLayer(layers.Layer):
        """Keras ``MyCustomLayer`` class."""

        def __init__(self, **kwargs: TensorLike) -> None:
            """Execute ``__init__``.

            Args:
                kwargs: Input parameter ``kwargs``.
            """
            super().__init__(**kwargs)

    config = {"class_name": "MyCustomLayer", "config": {"name": "custom"}}

    obj = layers.deserialize(config, custom_objects={"MyCustomLayer": MyCustomLayer})
    assert isinstance(obj, MyCustomLayer)


def test_deserialize_str() -> TensorLike:
    """Execute ``test_deserialize_str``."""
    from zero_keras.layers import misc

    class DummyMiscLayer(layers.Layer):
        """Keras ``DummyMiscLayer`` class."""

        def __init__(self, **kwargs: TensorLike) -> None:
            """Execute ``__init__``.

            Args:
                kwargs: Input parameter ``kwargs``.
            """
            super().__init__(**kwargs)

    misc.DummyMiscLayer = DummyMiscLayer

    # Str matching global
    obj = layers.deserialize("DummyMiscLayer")
    assert isinstance(obj, DummyMiscLayer)
    # Str not matching
    obj2 = layers.deserialize("UnknownLayer")
    assert obj2 == "UnknownLayer"


def test_deserialize_dict_fallback() -> TensorLike:
    """Execute ``test_deserialize_dict_fallback``."""
    from zero_keras.layers import misc

    class DummyMiscLayer2(layers.Layer):
        """Keras ``DummyMiscLayer2`` class."""

        def __init__(self, units: TensorLike = 10, **kwargs: TensorLike) -> None:
            """Execute ``__init__``.

            Args:
                units: Input parameter ``units``.
                kwargs: Input parameter ``kwargs``.
            """
            self.units = units
            super().__init__(**kwargs)

    misc.DummyMiscLayer2 = DummyMiscLayer2

    # Dict not matching class and no custom_objects
    config = {"class_name": "UnknownLayer", "config": {}}
    obj = layers.deserialize(config)
    assert obj == config

    # Dict matching global class
    config2 = {"class_name": "DummyMiscLayer2", "config": {"units": 4}}
    obj2 = layers.deserialize(config2)
    assert isinstance(obj2, DummyMiscLayer2)
    assert obj2.units == 4


def test_deserialize_other() -> TensorLike:
    # Not str, not dict
    """Execute ``test_deserialize_other``."""
    obj = layers.deserialize(123)
    assert obj == 123
