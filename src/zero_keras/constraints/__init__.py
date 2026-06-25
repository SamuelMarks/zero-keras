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
        return w


class MaxNorm(Constraint):
    """MaxNorm docstring."""

    def __init__(self, max_value=2, axis=0):
        self.max_value = max_value
        self.axis = axis


class MinMaxNorm(Constraint):
    """MinMaxNorm docstring."""

    def __init__(self, min_value=0.0, max_value=1.0, rate=1.0, axis=0):
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
        self.axis = axis


def max_norm(max_value=2, axis=0):
    """max_norm docstring."""
    return MaxNorm(max_value=max_value, axis=axis)


def min_max_norm(min_value=0.0, max_value=1.0, rate=1.0, axis=0):
    """min_max_norm docstring."""
    return MinMaxNorm(min_value=min_value, max_value=max_value, rate=rate, axis=axis)


def non_neg():
    """non_neg docstring."""
    return NonNeg()


def unit_norm(axis=0):
    """unit_norm docstring."""
    return UnitNorm(axis=axis)


def deserialize(config, custom_objects=None):
    """deserialize docstring."""
    return config


def get(identifier):
    """get docstring."""
    return identifier


def serialize(constraint):
    """serialize docstring."""
    return constraint


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
