"""dtype_policies API."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Dict, Optional

from zero_keras import ops
from zero_keras.core_layers import DTypePolicy, FloatDTypePolicy


class DTypePolicyMap:
    """DTypePolicyMap docstring."""

    def __init__(self, default_policy: Any = None) -> None:
        """docstring."""
        self.default_policy = default_policy
        self._policies: dict[str, Any] = {}
        self._name: str | None = None
        self._compute_dtype: str = "float32"
        self._quantization_mode: str | None = None
        self._variable_dtype: str = "float32"

    def __setitem__(self, key: str, value: Any) -> None:
        """__setitem__ docstring.

        Args:
            key: Key.
            value: Value.
        """
        self._policies[key] = value

    def __getitem__(self, key: str) -> Any:
        """__getitem__ docstring.

        Args:
            key: Key.
        """
        return self._policies.get(key, self.default_policy)

    def __iter__(self) -> Iterator[str]:
        """__iter__ docstring."""
        return iter(self._policies)

    def __len__(self) -> int:
        """__len__ docstring."""
        return len(self._policies)

    def clear(self) -> None:
        """clear docstring."""
        self._policies.clear()

    @property
    def compute_dtype(self) -> str:
        """compute_dtype docstring."""
        return getattr(self, "_compute_dtype", "float32")

    def convert_input(
        self, x: Any, dtype: str | None = None, exact: bool = False
    ) -> Any:
        """convert_input docstring.

        Args:
            x: Input.
            dtype: Dtype.
            exact: Exact.
        """
        if dtype is not None:
            return ops.cast(x, dtype)
        return ops.cast(x, self.compute_dtype)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DTypePolicyMap:
        """from_config docstring.

        Args:
            config: Config.
        """
        return cls(**config)

    def get(self, key: str, default: Any = None) -> Any:
        """get docstring.

        Args:
            key: Key.
            default: Default.
        """
        if key in self._policies:
            return self._policies[key]
        if self.default_policy is not None:
            return self.default_policy
        return default

    def get_config(self) -> dict[str, Any]:
        """get_config docstring."""
        return {"name": self.name}

    def items(self) -> Any:
        """items docstring."""
        return self._policies.items()

    def keys(self) -> Any:
        """keys docstring."""
        return self._policies.keys()

    @property
    def name(self) -> str | None:
        """name docstring."""
        return getattr(self, "_name", None)

    @name.setter
    def name(self, value: str) -> None:
        """name setter docstring."""
        self._name = value

    def pop(self, key: str, default: Any = None) -> Any:
        """pop docstring.

        Args:
            key: Key.
            default: Default.
        """
        return self._policies.pop(key, default)

    def popitem(self) -> Any:
        """popitem docstring."""
        return self._policies.popitem()

    @property
    def quantization_mode(self) -> str | None:
        """quantization_mode docstring."""
        return getattr(self, "_quantization_mode", None)

    def setdefault(self, key: str, default: Any = None) -> Any:
        """setdefault docstring.

        Args:
            key: Key.
            default: Default.
        """
        return self._policies.setdefault(key, default)

    def update(self, other: dict[str, Any]) -> None:
        """update docstring.

        Args:
            other: Other dict to update from.
        """
        self._policies.update(other)

    def values(self) -> Any:
        """values docstring."""
        return self._policies.values()

    @property
    def variable_dtype(self) -> str:
        """variable_dtype docstring."""
        return getattr(self, "_variable_dtype", "float32")


class QuantizedDTypePolicy(DTypePolicy):
    """QuantizedDTypePolicy docstring."""

    @property
    def compute_dtype(self) -> str:
        """compute_dtype docstring."""
        return getattr(self, "_compute_dtype", "float32")

    def convert_input(
        self, x: Any, dtype: str | None = None, exact: bool = False
    ) -> Any:
        """convert_input docstring.

        Args:
            x: Input.
            dtype: Dtype.
            exact: Exact.
        """
        if dtype is not None:
            return ops.cast(x, dtype)
        return ops.cast(x, self.compute_dtype)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> QuantizedDTypePolicy:
        """from_config docstring.

        Args:
            config: Config.
        """
        return cls(**config)

    def get_config(self) -> dict[str, Any]:
        """get_config docstring."""
        return {"name": self.name}

    @property
    def name(self) -> str | None:
        """name docstring."""
        return getattr(self, "_name", None)

    @property
    def quantization_mode(self) -> str | None:
        """quantization_mode docstring."""
        return getattr(self, "_quantization_mode", None)

    @property
    def variable_dtype(self) -> str:
        """variable_dtype docstring."""
        return getattr(self, "_variable_dtype", "float32")


class QuantizedFloat8DTypePolicy(QuantizedDTypePolicy):
    """QuantizedFloat8DTypePolicy docstring."""

    @property
    def amax_history_length(self) -> int | None:
        """amax_history_length docstring."""
        return getattr(self, "_amax_history_length", None)

    @property
    def default_amax_history_length(self) -> int | None:
        """default_amax_history_length docstring."""
        return getattr(self, "_default_amax_history_length", None)


def deserialize(config: Any, custom_objects: Any | None = None) -> Any:
    """deserialize docstring."""
    return config


def get(identifier: Any) -> Any:
    """get docstring."""
    return identifier


def serialize(policy: Any) -> Any:
    """serialize docstring."""
    return policy


__all__ = [
    "DTypePolicy",
    "DTypePolicyMap",
    "FloatDTypePolicy",
    "QuantizedDTypePolicy",
    "QuantizedFloat8DTypePolicy",
    "deserialize",
    "get",
    "serialize",
]
