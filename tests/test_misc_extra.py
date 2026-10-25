"""Module `test_misc_extra` for zero_keras."""

from __future__ import annotations

from zero_keras import layers, random
from zero_keras.models import Sequential
from zero_keras.types import TensorLike


def test_timedistributed_built_twice() -> None:
    """Verify or execute `test timedistributed built twice`."""
    layer = layers.TimeDistributed(layers.Dense(4))
    layer.build((2, 5, 3))
    layer.build((2, 5, 3))
    assert layer.built


def test_sequential_call_with_layers() -> None:
    """Verify or execute `test sequential call with layers`."""
    layer = Sequential([layers.Dense(4)])
    x = random.uniform((2, 3), dtype="float32")
    out = layer(x)
    assert out.shape == (2, 4)


def test_deserialize_custom_objects() -> None:
    """Verify or execute `test deserialize custom objects`."""

    class MyCustomLayer(layers.Layer):
        """Keras ``MyCustomLayer`` class."""

        def __init__(self, **kwargs: TensorLike) -> None:
            """Verify or execute `init` on `MyCustomLayer`.

            Args:
                **kwargs: Keyword arguments forwarded to helper.
            """
            super().__init__(**kwargs)

    config = {"class_name": "MyCustomLayer", "config": {"name": "custom"}}

    obj = layers.deserialize(config, custom_objects={"MyCustomLayer": MyCustomLayer})
    assert isinstance(obj, MyCustomLayer)


def test_deserialize_str() -> None:
    """Verify or execute `test deserialize str`."""
    from zero_keras.layers import misc

    class DummyMiscLayer(layers.Layer):
        """Keras ``DummyMiscLayer`` class."""

        def __init__(self, **kwargs: TensorLike) -> None:
            """Verify or execute `init` on `DummyMiscLayer`.

            Args:
                **kwargs: Keyword arguments forwarded to helper.
            """
            super().__init__(**kwargs)

    misc._MISC_LAYER_REGISTRY["DummyMiscLayer"] = DummyMiscLayer

    obj = layers.deserialize("DummyMiscLayer")
    assert isinstance(obj, DummyMiscLayer)
    obj2 = layers.deserialize("UnknownLayer")
    assert obj2 == "UnknownLayer"


def test_deserialize_dict_fallback() -> None:
    """Verify or execute `test deserialize dict fallback`."""
    from zero_keras.layers import misc

    class DummyMiscLayer2(layers.Layer):
        """Keras ``DummyMiscLayer2`` class."""

        def __init__(self, units: TensorLike = 10, **kwargs: TensorLike) -> None:
            """Verify or execute `init` on `DummyMiscLayer2`.

            Args:
                units: Test fixture or input parameter `units` (units).
                **kwargs: Keyword arguments forwarded to helper.
            """
            self.units = units
            super().__init__(**kwargs)

    misc._MISC_LAYER_REGISTRY["DummyMiscLayer2"] = DummyMiscLayer2

    config = {"class_name": "UnknownLayer", "config": {}}
    obj = layers.deserialize(config)
    assert obj == config

    config2 = {"class_name": "DummyMiscLayer2", "config": {"units": 4}}
    obj2 = layers.deserialize(config2)
    assert isinstance(obj2, DummyMiscLayer2)
    assert obj2.units == 4


def test_deserialize_other() -> None:
    """Verify or execute `test deserialize other`."""
    obj = layers.deserialize(123)
    assert obj == 123
