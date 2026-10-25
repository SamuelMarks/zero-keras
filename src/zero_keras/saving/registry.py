"""Centralized strongly typed serialization registry for zero_keras.

Provides explicit dictionary registries and helper functions for serializing
and deserializing Keras objects without relying on ``globals()`` or dunder
reflection hacks.
"""

from __future__ import annotations

from typing import Protocol, Union, runtime_checkable
from zero_keras.types import (
    ConfigDict,
    ConfigValue,
    KerasSerializable,
    SupportsGetConfig,
    TensorLike,
)


@runtime_checkable
class ConstructorFn(Protocol):
    """Protocol for registered class constructors and factory callables."""

    def __call__(
        self, *args: TensorLike, **kwargs: ConfigValue
    ) -> Union[TensorLike, SupportsGetConfig]:
        """Instantiate or evaluate the registered object.

        Args:
            *args: Positional constructor arguments.
            **kwargs: Keyword constructor arguments.

        Returns:
            Union[TensorLike, SupportsGetConfig]: Constructed instance.
        """


_CATEGORY_REGISTRIES: dict[str, dict[str, ConstructorFn]] = {}


def register_class(category: str, name: str, constructor: ConstructorFn) -> None:
    """Register a class or factory function in a named category registry.

    Args:
        category: Category key (e.g., ``"losses"``, ``"metrics"``, ``"layers"``).
        name: Class or alias name to register.
        constructor: Class or callable constructor.
    """
    bucket = _CATEGORY_REGISTRIES.setdefault(category, {})
    bucket[name] = constructor


def register_classes(category: str, mapping: dict[str, ConstructorFn]) -> None:
    """Bulk-register a mapping of class names to constructors in ``category``.

    Args:
        category: Category key (e.g., ``"losses"``, ``"metrics"``, ``"layers"``).
        mapping: Dictionary mapping class names to constructors.
    """
    bucket = _CATEGORY_REGISTRIES.setdefault(category, {})
    bucket.update(mapping)


def lookup_class(category: str, name: str) -> ConstructorFn | None:
    """Look up a registered constructor by category and class name.

    Args:
        category: Category key to search.
        name: Class or alias name to resolve.

    Returns:
        ConstructorFn | None: Registered constructor or ``None`` if not found.
    """
    bucket = _CATEGORY_REGISTRIES.get(category)
    if bucket is not None and name in bucket:
        return bucket[name]
    for other_bucket in _CATEGORY_REGISTRIES.values():
        if name in other_bucket:
            return other_bucket[name]
    return None


def get_object_class_name(obj: TensorLike | SupportsGetConfig | ConfigValue) -> str:
    """Return the runtime type name of ``obj`` using ``type(obj).__name__``.

    Args:
        obj: Target instance whose class name is requested.

    Returns:
        str: Unqualified class name string.
    """
    return type(obj).__name__


def serialize_with_config(obj: TensorLike | SupportsGetConfig | ConfigValue) -> ConfigDict:
    """Serialize an object into a ``{"class_name": ..., "config": ...}`` dictionary.

    Args:
        obj: Object instance to serialize.

    Returns:
        ConfigDict: Dictionary containing ``class_name`` and ``config`` (when supported).
    """
    class_name = get_object_class_name(obj)
    if isinstance(obj, SupportsGetConfig):
        return {
            "class_name": class_name,
            "config": obj.get_config(),
        }
    return {"class_name": class_name}


def deserialize_from_registry(
    category: str,
    config: ConfigValue,
    custom_objects: dict[str, ConfigValue] | None = None,
) -> TensorLike | SupportsGetConfig | ConfigValue:
    """Deserialize an object configuration using a named category registry.

    Args:
        category: Category key to search first.
        config: Serialized configuration dictionary or string identifier.
        custom_objects: Optional custom object mapping.

    Returns:
        TensorLike | SupportsGetConfig | ConfigValue: Reconstructed instance or original ``config``.
    """
    if config is None:
        return None
    if isinstance(config, str):
        cls = lookup_class(category, config)
        if cls is not None:
            return cls()
        if custom_objects and config in custom_objects:
            custom_entry = custom_objects[config]
            if callable(custom_entry):
                return custom_entry()
        return config
    if isinstance(config, dict):
        class_name_raw = config.get("class_name")
        if not isinstance(class_name_raw, str):
            return config
        conf_raw = config.get("config", {})
        conf: ConfigDict = dict(conf_raw) if isinstance(conf_raw, dict) else {}

        for k, v in list(conf.items()):
            if isinstance(v, dict) and "class_name" in v and "config" in v:
                conf[k] = deserialize_from_registry(category, v, custom_objects=custom_objects)

        if custom_objects and class_name_raw in custom_objects:
            custom_fn = custom_objects[class_name_raw]
            if isinstance(custom_fn, type) and issubclass(custom_fn, KerasSerializable):
                return custom_fn.from_config(conf)
            if callable(custom_fn):
                return custom_fn(**conf)

        cls = lookup_class(category, class_name_raw)
        if cls is not None:
            if isinstance(cls, type) and issubclass(cls, KerasSerializable):
                return cls.from_config(conf)
            return cls(**conf)

    return config


__all__ = [
    "ConstructorFn",
    "deserialize_from_registry",
    "get_object_class_name",
    "lookup_class",
    "register_class",
    "register_classes",
    "serialize_with_config",
]
