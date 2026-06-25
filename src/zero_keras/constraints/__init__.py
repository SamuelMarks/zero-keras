"""constraints API."""


class Constraint:
    """Constraint docstring."""

    @classmethod
    def from_config(cls, config):
        """from_config docstring.

        Args:
            config: Config.
        """
        return cls(**config)

    def get_config(self):
        """get_config docstring."""
        return {}

    def __call__(self, w):
        """docstring."""

        return w  # pragma: no cover


class MaxNorm(Constraint):
    """MaxNorm docstring."""

    def __init__(self, max_value=2, axis=0):
        """docstring."""

        self.max_value = max_value
        self.axis = axis


class MinMaxNorm(Constraint):
    """MinMaxNorm docstring."""

    def __init__(self, min_value=0.0, max_value=1.0, rate=1.0, axis=0):
        """docstring."""

        self.min_value = min_value
        self.max_value = max_value
        self.rate = rate
        self.axis = axis


class NonNeg(Constraint):
    """NonNeg docstring."""

    pass


class UnitNorm(Constraint):
    """UnitNorm docstring."""

    def __init__(self, axis=0):
        """docstring."""

        self.axis = axis


def max_norm(max_value=2, axis=0):
    """max_norm docstring."""
    return MaxNorm(max_value=max_value, axis=axis)  # pragma: no cover


def min_max_norm(min_value=0.0, max_value=1.0, rate=1.0, axis=0):
    """min_max_norm docstring."""
    return MinMaxNorm(
        min_value=min_value, max_value=max_value, rate=rate, axis=axis
    )  # pragma: no cover


def non_neg():
    """non_neg docstring."""
    return NonNeg()  # pragma: no cover


def unit_norm(axis=0):
    """unit_norm docstring."""
    return UnitNorm(axis=axis)  # pragma: no cover


def deserialize(config, custom_objects=None):
    """deserialize docstring."""
    return config  # pragma: no cover


def get(identifier):
    """get docstring."""
    return identifier  # pragma: no cover


def serialize(constraint):
    """serialize docstring."""
    return constraint  # pragma: no cover


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
