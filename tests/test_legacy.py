"""Comprehensive tests for ``zero_keras.legacy`` and ``zero_keras.optimizers.legacy``."""

from __future__ import annotations

from zero_keras import legacy, optimizers
from zero_keras.optimizers import legacy as opt_legacy


def test_legacy() -> None:
    """Verify legacy saving and legacy optimizer aliases."""
    assert legacy.saving.deserialize_keras_object("test") == "test"

    class Dummy:
        """Dummy class for legacy serialization testing."""

    assert legacy.saving.serialize_keras_object(Dummy()) == {"class_name": "Dummy"}

    from zero_keras import layers
    from zero_keras.saving import registry

    dense = layers.Dense(units=4)
    serialized = legacy.saving.serialize_keras_object(dense)
    assert serialized["class_name"] == "Dense"
    assert "config" in serialized
    reconstructed = legacy.saving.deserialize_keras_object(serialized)
    assert isinstance(reconstructed, layers.Dense)
    assert reconstructed.units == 4

    nested_wrapped = legacy.saving.deserialize_keras_object(
        {
            "class_name": "TimeDistributed",
            "config": {
                "layer": {"class_name": "Dense", "config": {"units": 2}},
            },
        }
    )
    assert isinstance(nested_wrapped, layers.TimeDistributed)

    registry.register_class("custom_cat", "CustomRegistered", layers.Identity)
    assert registry.lookup_class("custom_cat", "CustomRegistered") is layers.Identity
    assert registry.lookup_class("other_cat", "CustomRegistered") is layers.Identity
    assert registry.lookup_class("other_cat", "NonExistentClass") is None

    assert isinstance(legacy.saving.deserialize_keras_object("Identity"), layers.Identity)
    assert legacy.saving.deserialize_keras_object(None) is None
    assert legacy.saving.deserialize_keras_object({"no_class_name": 1}) == {"no_class_name": 1}
    assert (
        legacy.saving.deserialize_keras_object(
            "CustomStr", custom_objects={"CustomStr": lambda: "str_res"}
        )
        == "str_res"
    )
    assert (
        legacy.saving.deserialize_keras_object(
            "CustomNonCall", custom_objects={"CustomNonCall": 123}
        )
        == "CustomNonCall"
    )
    assert (
        legacy.saving.deserialize_keras_object(
            {"class_name": "CustomFn", "config": {}},
            custom_objects={"CustomFn": lambda **kw: "fn_result"},
        )
        == "fn_result"
    )
    assert legacy.saving.deserialize_keras_object(
        {"class_name": "CustomNonCall", "config": {}},
        custom_objects={"CustomNonCall": 123},
    ) == {"class_name": "CustomNonCall", "config": {}}

    class CustomCallableOnly:
        """Callable constructor without from_config."""

        def __init__(self, **kwargs: str | int | float | bool | None) -> None:
            """Initialize CustomCallableOnly.

            Args:
                **kwargs: Keyword arguments.
            """
            self.kwargs = kwargs

    registry.register_class("layers", "CustomCallableOnly", CustomCallableOnly)
    assert isinstance(
        legacy.saving.deserialize_keras_object({"class_name": "CustomCallableOnly", "config": {}}),
        CustomCallableOnly,
    )
    assert isinstance(
        legacy.saving.deserialize_keras_object(
            {"class_name": "Identity", "config": {}},
            custom_objects={"Identity": layers.Identity},
        ),
        layers.Identity,
    )

    assert opt_legacy.SGD is optimizers.SGD
    assert opt_legacy.Adagrad is optimizers.Adagrad
    assert opt_legacy.Adam is optimizers.Adam
    assert opt_legacy.Ftrl is optimizers.Ftrl
    assert opt_legacy.Optimizer is optimizers.Optimizer
    assert opt_legacy.RMSprop is optimizers.RMSprop
