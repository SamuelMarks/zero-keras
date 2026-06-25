"""mixed_precision API."""

from zero_keras.core_layers import DTypePolicy
from zero_keras.optimizers import LossScaleOptimizer


class Policy(DTypePolicy):
    """Policy docstring."""

    pass


def dtype_policy():
    """dtype_policy docstring."""
    pass


def global_policy():
    """global_policy docstring."""
    pass


def set_dtype_policy(policy):
    """set_dtype_policy docstring."""
    pass


def set_global_policy(policy):
    """set_global_policy docstring."""
    pass


__all__ = [
    "DTypePolicy",
    "LossScaleOptimizer",
    "Policy",
    "dtype_policy",
    "global_policy",
    "set_dtype_policy",
    "set_global_policy",
]
