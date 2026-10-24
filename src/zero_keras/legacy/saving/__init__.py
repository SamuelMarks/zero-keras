"""Legacy saving and serialization utilities for zero_keras.legacy.saving."""

from __future__ import annotations

from zero_keras.types import ConfigDict, ConfigValue


def deserialize_keras_object(
    identifier: ConfigValue,
    custom_objects: dict[str, ConfigValue] | None = None,
    **kwargs: ConfigValue,
) -> ConfigValue:
    """Deserialize a legacy Keras object specification.

    Args:
        identifier: Serialized config or class identifier.
        custom_objects: Optional dictionary of custom classes.
        **kwargs: Additional deserialization options.

    Returns:
        ConfigValue: Deserialized object or identifier.
    """
    _ = custom_objects, kwargs
    return identifier


def serialize_keras_object(obj: ConfigValue) -> ConfigDict:
    """Serialize an object into a legacy class-name dictionary.

    Args:
        obj: Object instance to serialize.

    Returns:
        ConfigDict: Dictionary containing ``class_name``.
    """
    return {"class_name": obj.__class__.__name__}


__all__ = ["deserialize_keras_object", "serialize_keras_object"]
