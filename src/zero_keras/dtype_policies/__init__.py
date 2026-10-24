"""DType policies and quantization policy mappings for zero_keras."""

from __future__ import annotations

from collections.abc import ItemsView, Iterator, KeysView, ValuesView
from typing import Union

from zero_keras import ops
from zero_keras.types import ConfigDict, ConfigValue, DTypeLike, TensorLike


def _parse_policy_dtypes(name: str | None) -> tuple[str | None, str | None, str | None]:
    """Parse ``(compute_dtype, variable_dtype, quantization_mode)`` from a policy name.

    Args:
        name: Policy name string (e.g., ``"mixed_float16"``, ``"float32"``, ``"int8_from_float32"``).

    Returns:
        tuple[str | None, str | None, str | None]: Parsed compute dtype, variable dtype,
            and quantization mode.
    """
    if name is None:
        return None, None, None
    norm = name.lower().strip()
    if norm == "mixed_float16":
        return "float16", "float32", None
    if norm == "mixed_bfloat16":
        return "bfloat16", "float32", None
    if norm in ("float16", "float32", "float64", "bfloat16", "int32", "int64", "uint8", "bool"):
        return norm, norm, None
    if "_from_" in norm:
        q_mode, source_policy = norm.split("_from_", 1)
        src_comp, src_var, _ = _parse_policy_dtypes(source_policy)
        return src_comp or "float32", src_var or "float32", q_mode
    if norm in ("int8", "int4", "float8", "gptq", "awq"):
        return "float32", "float32", norm
    return None, None, None


class DTypePolicy:
    """Base dtype policy specifying compute and variable storage dtypes."""

    def __init__(self, name: str | None = None) -> None:
        """Initialize a DTypePolicy.

        Args:
            name: Optional policy name (e.g. ``"float32"``, ``"mixed_float16"``).
        """
        self._name: str | None = name
        comp, var, q_mode = _parse_policy_dtypes(name)
        self._compute_dtype: str | None = comp
        self._variable_dtype: str | None = var
        self._quantization_mode: str | None = q_mode

    @property
    def name(self) -> str | None:
        """Return the policy name.

        Returns:
            str | None: Policy name string or None.
        """
        return self._name

    @name.setter
    def name(self, value: str | None) -> None:
        """Set the policy name and update derived dtypes.

        Args:
            value: New policy name string.
        """
        self._name = value
        comp, var, q_mode = _parse_policy_dtypes(value)
        if comp is not None:
            self._compute_dtype = comp
            self._variable_dtype = var
            self._quantization_mode = q_mode

    @property
    def compute_dtype(self) -> str | None:
        """Return the compute dtype string.

        Returns:
            str | None: Compute dtype or None.
        """
        return self._compute_dtype

    @property
    def variable_dtype(self) -> str | None:
        """Return the variable storage dtype string.

        Returns:
            str | None: Variable dtype or None.
        """
        return self._variable_dtype

    @property
    def quantization_mode(self) -> str | None:
        """Return the quantization mode, if any.

        Returns:
            str | None: Quantization mode or None.
        """
        return self._quantization_mode

    def convert_input(
        self,
        x: TensorLike,
        dtype: DTypeLike = None,
        exact: bool = False,
    ) -> TensorLike:
        """Convert input ``x`` according to ``dtype`` or ``self.compute_dtype``.

        Args:
            x: Input tensor or scalar.
            dtype: Optional explicit target dtype.
            exact: Whether conversion must be exact.

        Returns:
            TensorLike: Converted tensor or original scalar.
        """
        _ = exact
        if isinstance(x, (int, float, bool, str)):
            return x
        target = str(dtype) if dtype is not None else self._compute_dtype
        if target is not None:
            return ops.cast(x, target)
        return x

    def get_config(self) -> ConfigDict:
        """Return serializable configuration dictionary.

        Returns:
            ConfigDict: Configuration dictionary containing ``name``.
        """
        return {"name": self._name}

    @classmethod
    def from_config(cls, config: ConfigDict) -> DTypePolicy:
        """Instantiate a DTypePolicy from its configuration dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            DTypePolicy: Reconstructed policy instance.
        """
        name_val = config.get("name")
        name = str(name_val) if isinstance(name_val, str) else None
        return cls(name=name)


class FloatDTypePolicy(DTypePolicy):
    """Floating-point dtype policy (defaulting to ``"float32"`` compute and storage)."""

    def __init__(self, name: str | None = None) -> None:
        """Initialize a FloatDTypePolicy.

        Args:
            name: Optional policy name.
        """
        super().__init__(name=name)
        if self._compute_dtype is None:
            self._compute_dtype = "float32"
        if self._variable_dtype is None:
            self._variable_dtype = "float32"

    @property
    def compute_dtype(self) -> str:
        """Return the floating-point compute dtype.

        Returns:
            str: Compute dtype string.
        """
        return self._compute_dtype or "float32"

    @property
    def variable_dtype(self) -> str:
        """Return the floating-point variable storage dtype.

        Returns:
            str: Variable dtype string.
        """
        return self._variable_dtype or "float32"

    @classmethod
    def from_config(cls, config: ConfigDict) -> FloatDTypePolicy:
        """Instantiate a FloatDTypePolicy from its configuration dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            FloatDTypePolicy: Reconstructed policy instance.
        """
        name_val = config.get("name")
        name = str(name_val) if isinstance(name_val, str) else None
        return cls(name=name)


PolicyValue = Union[DTypePolicy, str, ConfigDict, None]


class DTypePolicyMap:
    """Mapping from layer paths/names to ``DTypePolicy`` instances."""

    def __init__(
        self,
        default_policy: PolicyValue = None,
        name: str | None = None,
    ) -> None:
        """Initialize a DTypePolicyMap.

        Args:
            default_policy: Default fallback policy when a key is not present.
            name: Optional name for the policy map.
        """
        self.default_policy: PolicyValue = default_policy
        self._policies: dict[str, PolicyValue] = {}
        self._name: str | None = name
        self._compute_dtype: str = "float32"
        self._quantization_mode: str | None = None
        self._variable_dtype: str = "float32"

    def __setitem__(self, key: str, value: PolicyValue) -> None:
        """Assign ``value`` policy to ``key``.

        Args:
            key: Layer path or name pattern.
            value: DTypePolicy or policy identifier.
        """
        self._policies[key] = value

    def __getitem__(self, key: str) -> PolicyValue:
        """Retrieve policy for ``key``, falling back to ``default_policy``.

        Args:
            key: Layer path or name pattern.

        Returns:
            PolicyValue: Resolved policy or default policy.
        """
        return self._policies.get(key, self.default_policy)

    def __iter__(self) -> Iterator[str]:
        """Iterate over policy map keys.

        Returns:
            Iterator[str]: Key iterator.
        """
        return iter(self._policies)

    def __len__(self) -> int:
        """Return number of explicit entries in the policy map.

        Returns:
            int: Entry count.
        """
        return len(self._policies)

    def clear(self) -> None:
        """Remove all explicit policy entries from the map."""
        self._policies.clear()

    @property
    def compute_dtype(self) -> str:
        """Return the compute dtype string.

        Returns:
            str: Compute dtype (defaults to ``"float32"``).
        """
        return self._compute_dtype

    def convert_input(
        self,
        x: TensorLike,
        dtype: DTypeLike = None,
        exact: bool = False,
    ) -> TensorLike:
        """Cast input ``x`` to ``dtype`` or ``self.compute_dtype``.

        Args:
            x: Input tensor or scalar.
            dtype: Optional explicit target dtype.
            exact: Whether to enforce exact dtype conversion.

        Returns:
            TensorLike: Converted tensor.
        """
        _ = exact
        if dtype is not None:
            return ops.cast(x, str(dtype))
        return ops.cast(x, self.compute_dtype)

    @classmethod
    def from_config(cls, config: dict[str, PolicyValue]) -> DTypePolicyMap:
        """Instantiate a DTypePolicyMap from its configuration dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            DTypePolicyMap: Reconstructed policy map.
        """
        default_policy = config.get("default_policy")
        name_val = config.get("name")
        name = str(name_val) if isinstance(name_val, str) else None
        return cls(default_policy=default_policy, name=name)

    def get(self, key: str, default: PolicyValue = None) -> PolicyValue:
        """Look up ``key`` in the policy map with fallback.

        Args:
            key: Layer name or path.
            default: Fallback value if ``key`` and ``default_policy`` are unset.

        Returns:
            PolicyValue: Matching policy or fallback.
        """
        if key in self._policies:
            return self._policies[key]
        if self.default_policy is not None:
            return self.default_policy
        return default

    def get_config(self) -> ConfigDict:
        """Return serializable configuration dictionary.

        Returns:
            ConfigDict: Configuration dictionary.
        """
        return {"name": self.name}

    def items(self) -> ItemsView[str, PolicyValue]:
        """Return a view of ``(key, policy)`` pairs.

        Returns:
            ItemsView[str, PolicyValue]: Items view.
        """
        return self._policies.items()

    def keys(self) -> KeysView[str]:
        """Return a view of policy map keys.

        Returns:
            KeysView[str]: Keys view.
        """
        return self._policies.keys()

    @property
    def name(self) -> str | None:
        """Return the policy map name.

        Returns:
            str | None: Policy map name.
        """
        return self._name

    @name.setter
    def name(self, value: str | None) -> None:
        """Set the policy map name.

        Args:
            value: New name string.
        """
        self._name = value

    def pop(self, key: str, default: PolicyValue = None) -> PolicyValue:
        """Remove and return the policy at ``key``.

        Args:
            key: Key to remove.
            default: Value to return if ``key`` is not present.

        Returns:
            PolicyValue: Removed policy or ``default``.
        """
        return self._policies.pop(key, default)

    def popitem(self) -> tuple[str, PolicyValue]:
        """Remove and return an arbitrary ``(key, policy)`` pair.

        Returns:
            tuple[str, PolicyValue]: Popped key-value pair.
        """
        return self._policies.popitem()

    @property
    def quantization_mode(self) -> str | None:
        """Return the quantization mode, if any.

        Returns:
            str | None: Quantization mode string or None.
        """
        return self._quantization_mode

    def setdefault(self, key: str, default: PolicyValue = None) -> PolicyValue:
        """Insert ``default`` at ``key`` if absent and return the value.

        Args:
            key: Key to query or set.
            default: Default policy to insert when ``key`` is missing.

        Returns:
            PolicyValue: Policy associated with ``key``.
        """
        return self._policies.setdefault(key, default)

    def update(self, other: dict[str, PolicyValue]) -> None:
        """Update the policy map with key-value pairs from ``other``.

        Args:
            other: Dictionary of policy mappings.
        """
        self._policies.update(other)

    def values(self) -> ValuesView[PolicyValue]:
        """Return a view of policies stored in the map.

        Returns:
            ValuesView[PolicyValue]: Values view.
        """
        return self._policies.values()

    @property
    def variable_dtype(self) -> str:
        """Return the variable storage dtype string.

        Returns:
            str: Variable dtype string.
        """
        return self._variable_dtype


class QuantizedDTypePolicy(DTypePolicy):
    """DType policy for quantized execution modes (e.g., int8, int4, float8)."""

    def __init__(self, name: str | None = None) -> None:
        """Initialize a QuantizedDTypePolicy.

        Args:
            name: Optional policy name.
        """
        super().__init__(name=name)
        if self._compute_dtype is None:
            self._compute_dtype = "float32"
        if self._variable_dtype is None:
            self._variable_dtype = "float32"

    @property
    def compute_dtype(self) -> str:
        """Return the compute dtype for quantized operations.

        Returns:
            str: Compute dtype string.
        """
        return self._compute_dtype or "float32"

    def convert_input(
        self,
        x: TensorLike,
        dtype: DTypeLike = None,
        exact: bool = False,
    ) -> TensorLike:
        """Cast ``x`` to ``dtype`` or ``self.compute_dtype``.

        Args:
            x: Input tensor or scalar.
            dtype: Optional explicit dtype.
            exact: Whether conversion must be exact.

        Returns:
            TensorLike: Cast tensor.
        """
        _ = exact
        if dtype is not None:
            return ops.cast(x, str(dtype))
        return ops.cast(x, self.compute_dtype)

    @classmethod
    def from_config(cls, config: ConfigDict) -> QuantizedDTypePolicy:
        """Instantiate a QuantizedDTypePolicy from config.

        Args:
            config: Configuration dictionary.

        Returns:
            QuantizedDTypePolicy: Reconstructed policy.
        """
        name_val = config.get("name")
        name = str(name_val) if isinstance(name_val, str) else None
        return cls(name=name)

    def get_config(self) -> ConfigDict:
        """Return serializable configuration dictionary.

        Returns:
            ConfigDict: Configuration dictionary.
        """
        return {"name": self.name}

    @property
    def name(self) -> str | None:
        """Return the policy name.

        Returns:
            str | None: Policy name.
        """
        return self._name

    @property
    def quantization_mode(self) -> str | None:
        """Return the quantization mode.

        Returns:
            str | None: Quantization mode string or None.
        """
        return self._quantization_mode

    @property
    def variable_dtype(self) -> str:
        """Return the variable dtype string.

        Returns:
            str: Variable dtype.
        """
        return self._variable_dtype or "float32"


class QuantizedFloat8DTypePolicy(QuantizedDTypePolicy):
    """DType policy for FP8 quantization with amax history tracking."""

    def __init__(
        self,
        name: str | None = None,
        amax_history_length: int | None = None,
    ) -> None:
        """Initialize a QuantizedFloat8DTypePolicy.

        Args:
            name: Optional policy name.
            amax_history_length: Optional length of the amax history window.
        """
        super().__init__(name=name)
        self._amax_history_length: int | None = amax_history_length
        self._default_amax_history_length: int | None = None

    @property
    def amax_history_length(self) -> int | None:
        """Return the configured amax history length.

        Returns:
            int | None: History length or None.
        """
        return self._amax_history_length

    @property
    def default_amax_history_length(self) -> int | None:
        """Return the default amax history length.

        Returns:
            int | None: Default history length or None.
        """
        return self._default_amax_history_length


_POLICY_CLASSES: dict[str, type[DTypePolicy]] = {
    "DTypePolicy": DTypePolicy,
    "Policy": DTypePolicy,
    "FloatDTypePolicy": FloatDTypePolicy,
    "QuantizedDTypePolicy": QuantizedDTypePolicy,
    "QuantizedFloat8DTypePolicy": QuantizedFloat8DTypePolicy,
}


def deserialize(
    config: PolicyValue | DTypePolicyMap,
    custom_objects: dict[str, ConfigValue] | None = None,
) -> PolicyValue | DTypePolicyMap:
    """Deserialize a dtype policy configuration.

    Args:
        config: Serialized policy config or instance.
        custom_objects: Optional custom object mapping.

    Returns:
        PolicyValue | DTypePolicyMap: Deserialized policy.
    """
    _ = custom_objects
    if isinstance(config, dict):
        cls_name = config.get("class_name")
        inner_raw = config.get("config", {})
        inner: ConfigDict = inner_raw if isinstance(inner_raw, dict) else {}
        if cls_name == "DTypePolicyMap":
            return DTypePolicyMap.from_config(inner)
        if isinstance(cls_name, str) and cls_name in _POLICY_CLASSES:
            return _POLICY_CLASSES[cls_name].from_config(inner)
    return config


def get(identifier: PolicyValue | DTypePolicyMap) -> PolicyValue | DTypePolicyMap:
    """Retrieve a dtype policy from ``identifier``.

    Args:
        identifier: Policy instance, name, or dict.

    Returns:
        PolicyValue | DTypePolicyMap: Resolved policy.
    """
    if isinstance(identifier, dict):
        return deserialize(identifier)
    return identifier


def serialize(policy: PolicyValue | DTypePolicyMap) -> PolicyValue | DTypePolicyMap:
    """Serialize a dtype policy instance.

    Args:
        policy: Policy to serialize.

    Returns:
        PolicyValue | DTypePolicyMap: Serialized policy representation.
    """
    if isinstance(policy, QuantizedFloat8DTypePolicy):
        return {"class_name": "QuantizedFloat8DTypePolicy", "config": policy.get_config()}
    if isinstance(policy, QuantizedDTypePolicy):
        return {"class_name": "QuantizedDTypePolicy", "config": policy.get_config()}
    if isinstance(policy, FloatDTypePolicy):
        return {"class_name": "FloatDTypePolicy", "config": policy.get_config()}
    if isinstance(policy, DTypePolicy):
        return {"class_name": "DTypePolicy", "config": policy.get_config()}
    if isinstance(policy, DTypePolicyMap):
        return {"class_name": "DTypePolicyMap", "config": policy.get_config()}
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
