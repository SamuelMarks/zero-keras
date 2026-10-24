"""Weight constraints API for zero_keras."""

from __future__ import annotations

from zero_keras import config as zk_config, ops
from zero_keras.activations import _to_tensor, _wrap
from zero_keras.types import AxisSpec, ConfigDict, ConfigValue, TensorLike


class Constraint:
    """Base class for weight constraints."""

    _registry_name: str = "Constraint"

    @classmethod
    def from_config(cls, config: ConfigDict) -> Constraint:
        """Instantiate a constraint from its configuration dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            Constraint: Reconstructed constraint instance.
        """
        return cls()

    def get_config(self) -> ConfigDict:
        """Return the constraint configuration dictionary.

        Returns:
            ConfigDict: Empty or populated configuration dictionary.
        """
        return {}

    def __call__(self, w: TensorLike) -> TensorLike:
        """Apply the constraint projection to weight tensor ``w``.

        Args:
            w: Input weight tensor.

        Returns:
            TensorLike: Constrained weight tensor.
        """
        return w


class MaxNorm(Constraint):
    """MaxNorm weight constraint."""

    _registry_name: str = "MaxNorm"

    def __init__(self, max_value: float = 2.0, axis: AxisSpec = 0) -> None:
        """Initialize MaxNorm constraint.

        Args:
            max_value: Maximum norm for the incoming weights.
            axis: Axis or axes along which to calculate weight norms.
        """
        self.max_value = float(max_value)
        self.axis: AxisSpec = axis

    def __call__(self, w: TensorLike) -> TensorLike:
        """Constrain weight norms along ``self.axis`` to at most ``self.max_value``.

        Args:
            w: Input weight tensor.

        Returns:
            TensorLike: Projected weight tensor.
        """
        t = _to_tensor(w)
        norms = ops.sqrt(ops.sum(ops.square(t), axis=self.axis, keepdims=True))
        desired = ops.clip(norms, 0.0, self.max_value)
        scaled = ops.multiply(
            t,
            ops.divide(desired, ops.add(zk_config.epsilon(), norms)),
        )
        return _wrap(scaled)

    def get_config(self) -> ConfigDict:
        """Return the MaxNorm configuration dictionary.

        Returns:
            ConfigDict: Configuration dictionary containing ``max_value`` and ``axis``.
        """
        axis_cfg: ConfigValue
        if isinstance(self.axis, list):
            axis_cfg = tuple(self.axis)
        else:
            axis_cfg = self.axis
        return {"max_value": self.max_value, "axis": axis_cfg}

    @classmethod
    def from_config(cls, config: ConfigDict) -> MaxNorm:
        """Instantiate a MaxNorm constraint from its configuration dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            MaxNorm: Reconstructed MaxNorm instance.
        """
        max_val_raw = config.get("max_value", 2.0)
        max_val = float(max_val_raw) if isinstance(max_val_raw, (int, float)) else 2.0
        axis_raw = config.get("axis", 0)
        axis_val: AxisSpec
        if isinstance(axis_raw, int) or axis_raw is None:
            axis_val = axis_raw
        elif isinstance(axis_raw, (tuple, list)):
            axis_val = tuple(int(a) for a in axis_raw if isinstance(a, int))
        else:
            axis_val = 0
        return cls(max_value=max_val, axis=axis_val)


class NonNeg(Constraint):
    """Constrains the weights to be non-negative."""

    _registry_name: str = "NonNeg"

    def __call__(self, w: TensorLike) -> TensorLike:
        """Clamp negative weights in ``w`` to zero.

        Args:
            w: Input weight tensor.

        Returns:
            TensorLike: Non-negative weight tensor.
        """
        t = _to_tensor(w)
        return _wrap(ops.maximum(t, 0.0))


class UnitNorm(Constraint):
    """Constrains the weights incident to each hidden unit to have unit norm."""

    _registry_name: str = "UnitNorm"

    def __init__(self, axis: AxisSpec = 0) -> None:
        """Initialize UnitNorm constraint.

        Args:
            axis: Axis or axes along which to calculate weight norms.
        """
        self.axis: AxisSpec = axis

    def __call__(self, w: TensorLike) -> TensorLike:
        """Normalize ``w`` along ``self.axis`` to unit L2 norm.

        Args:
            w: Input weight tensor.

        Returns:
            TensorLike: Unit-norm projected weight tensor.
        """
        t = _to_tensor(w)
        norms = ops.sqrt(ops.sum(ops.square(t), axis=self.axis, keepdims=True))
        return _wrap(ops.divide(t, ops.add(zk_config.epsilon(), norms)))

    def get_config(self) -> ConfigDict:
        """Return the UnitNorm configuration dictionary.

        Returns:
            ConfigDict: Configuration dictionary containing ``axis``.
        """
        axis_cfg: ConfigValue
        if isinstance(self.axis, list):
            axis_cfg = tuple(self.axis)
        else:
            axis_cfg = self.axis
        return {"axis": axis_cfg}

    @classmethod
    def from_config(cls, config: ConfigDict) -> UnitNorm:
        """Instantiate a UnitNorm constraint from its configuration dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            UnitNorm: Reconstructed UnitNorm instance.
        """
        axis_raw = config.get("axis", 0)
        axis_val: AxisSpec
        if isinstance(axis_raw, int) or axis_raw is None:
            axis_val = axis_raw
        elif isinstance(axis_raw, (tuple, list)):
            axis_val = tuple(int(a) for a in axis_raw if isinstance(a, int))
        else:
            axis_val = 0
        return cls(axis=axis_val)


class MinMaxNorm(Constraint):
    """MinMaxNorm weight constraint."""

    _registry_name: str = "MinMaxNorm"

    def __init__(
        self,
        min_value: float = 0.0,
        max_value: float = 1.0,
        rate: float = 1.0,
        axis: AxisSpec = 0,
    ) -> None:
        """Initialize MinMaxNorm constraint.

        Args:
            min_value: Minimum norm for the incoming weights.
            max_value: Maximum norm for the incoming weights.
            rate: Rate for enforcing the constraint.
            axis: Axis or axes along which to calculate weight norms.
        """
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.rate = float(rate)
        self.axis: AxisSpec = axis

    def __call__(self, w: TensorLike) -> TensorLike:
        """Constrain weight norms between ``self.min_value`` and ``self.max_value``.

        Args:
            w: Input weight tensor.

        Returns:
            TensorLike: Constrained weight tensor.
        """
        t = _to_tensor(w)
        norms = ops.sqrt(ops.sum(ops.square(t), axis=self.axis, keepdims=True))
        clipped = ops.clip(norms, self.min_value, self.max_value)
        desired = ops.add(
            ops.multiply(self.rate, clipped),
            ops.multiply(1.0 - self.rate, norms),
        )
        scaled = ops.multiply(
            t,
            ops.divide(desired, ops.add(zk_config.epsilon(), norms)),
        )
        return _wrap(scaled)

    def get_config(self) -> ConfigDict:
        """Return the MinMaxNorm configuration dictionary.

        Returns:
            ConfigDict: Configuration dictionary.
        """
        axis_cfg: ConfigValue
        if isinstance(self.axis, list):
            axis_cfg = tuple(self.axis)
        else:
            axis_cfg = self.axis
        return {
            "min_value": self.min_value,
            "max_value": self.max_value,
            "rate": self.rate,
            "axis": axis_cfg,
        }

    @classmethod
    def from_config(cls, config: ConfigDict) -> MinMaxNorm:
        """Instantiate a MinMaxNorm constraint from its configuration dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            MinMaxNorm: Reconstructed MinMaxNorm instance.
        """
        min_raw = config.get("min_value", 0.0)
        max_raw = config.get("max_value", 1.0)
        rate_raw = config.get("rate", 1.0)
        axis_raw = config.get("axis", 0)
        axis_val: AxisSpec
        if isinstance(axis_raw, int) or axis_raw is None:
            axis_val = axis_raw
        elif isinstance(axis_raw, (tuple, list)):
            axis_val = tuple(int(a) for a in axis_raw if isinstance(a, int))
        else:
            axis_val = 0
        return cls(
            min_value=float(min_raw) if isinstance(min_raw, (int, float)) else 0.0,
            max_value=float(max_raw) if isinstance(max_raw, (int, float)) else 1.0,
            rate=float(rate_raw) if isinstance(rate_raw, (int, float)) else 1.0,
            axis=axis_val,
        )


def max_norm(max_value: float = 2.0, axis: AxisSpec = 0) -> MaxNorm:
    """Create a ``MaxNorm`` constraint instance.

    Args:
        max_value: Maximum norm value.
        axis: Norm calculation axis.

    Returns:
        MaxNorm: Configured MaxNorm instance.
    """
    return MaxNorm(max_value=max_value, axis=axis)


def min_max_norm(
    min_value: float = 0.0,
    max_value: float = 1.0,
    rate: float = 1.0,
    axis: AxisSpec = 0,
) -> MinMaxNorm:
    """Create a ``MinMaxNorm`` constraint instance.

    Args:
        min_value: Minimum norm value.
        max_value: Maximum norm value.
        rate: Constraint interpolation rate.
        axis: Norm calculation axis.

    Returns:
        MinMaxNorm: Configured MinMaxNorm instance.
    """
    return MinMaxNorm(min_value=min_value, max_value=max_value, rate=rate, axis=axis)


def non_neg() -> NonNeg:
    """Create a ``NonNeg`` constraint instance.

    Returns:
        NonNeg: Configured NonNeg instance.
    """
    return NonNeg()


def unit_norm(axis: AxisSpec = 0) -> UnitNorm:
    """Create a ``UnitNorm`` constraint instance.

    Args:
        axis: Norm calculation axis.

    Returns:
        UnitNorm: Configured UnitNorm instance.
    """
    return UnitNorm(axis=axis)


_CONSTRAINT_CLASSES: dict[str, type[Constraint]] = {
    "Constraint": Constraint,
    "MaxNorm": MaxNorm,
    "max_norm": MaxNorm,
    "MinMaxNorm": MinMaxNorm,
    "min_max_norm": MinMaxNorm,
    "NonNeg": NonNeg,
    "non_neg": NonNeg,
    "UnitNorm": UnitNorm,
    "unit_norm": UnitNorm,
}


def serialize(
    constraint: Constraint | str | ConfigDict | None,
) -> ConfigDict | str | None:
    """Serialize a constraint instance.

    Args:
        constraint: Constraint to serialize.

    Returns:
        ConfigDict | str | None: Serialized constraint representation.
    """
    if constraint is None:
        return None
    if isinstance(constraint, Constraint):
        return {
            "class_name": constraint._registry_name,
            "config": constraint.get_config(),
        }
    return constraint


def deserialize(
    config: Constraint | str | ConfigDict | None,
    custom_objects: dict[str, ConfigValue] | None = None,
) -> Constraint | str | ConfigDict | None:
    """Deserialize a constraint configuration.

    Args:
        config: Serialized constraint specification.
        custom_objects: Optional custom object mapping.

    Returns:
        Constraint | str | ConfigDict | None: Deserialized constraint.
    """
    if config is None or isinstance(config, Constraint):
        return config
    if isinstance(config, str):
        if custom_objects and config in custom_objects:
            custom_obj = custom_objects[config]
            if callable(custom_obj):
                res = custom_obj()
                if isinstance(res, Constraint):
                    return res
        cls = _CONSTRAINT_CLASSES.get(config)
        if cls is not None:
            return cls()
        return config
    if isinstance(config, dict):
        class_name = config.get("class_name")
        conf_raw = config.get("config", {})
        conf: ConfigDict = conf_raw if isinstance(conf_raw, dict) else {}
        if isinstance(class_name, str):
            if custom_objects and class_name in custom_objects:
                custom_cls = custom_objects[class_name]
                if callable(custom_cls):
                    res = custom_cls(**conf)
                    if isinstance(res, Constraint):
                        return res
            cls = _CONSTRAINT_CLASSES.get(class_name)
            if cls is not None:
                return cls.from_config(conf)
    return config


def get(
    identifier: Constraint | str | ConfigDict | None,
) -> Constraint | str | ConfigDict | None:
    """Retrieve a constraint instance by identifier.

    Args:
        identifier: Constraint instance, name, or config dict.

    Returns:
        Constraint | str | ConfigDict | None: Resolved constraint.
    """
    if identifier is None or isinstance(identifier, Constraint):
        return identifier
    if isinstance(identifier, (str, dict)):
        return deserialize(identifier)
    return identifier


__all__ = [
    "Constraint",
    "MaxNorm",
    "MinMaxNorm",
    "NonNeg",
    "UnitNorm",
    "deserialize",
    "get",
    "max_norm",
    "min_max_norm",
    "non_neg",
    "serialize",
    "unit_norm",
]
