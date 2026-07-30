"""mixed_precision API."""

from typing import Any

from zero_keras.core_layers import DTypePolicy
from zero_keras.optimizers import LossScaleOptimizer


class Policy(DTypePolicy):
    """Policy docstring."""


_GLOBAL_POLICY: Any = None


def dtype_policy() -> Any:
    """dtype_policy docstring."""
    return _GLOBAL_POLICY


def global_policy() -> Any:
    """global_policy docstring."""
    return _GLOBAL_POLICY


def set_dtype_policy(policy: Any) -> None:
    """set_dtype_policy docstring.

    Args:
        policy: Policy.
    """
    global _GLOBAL_POLICY
    _GLOBAL_POLICY = policy


def set_global_policy(policy: Any) -> None:
    """set_global_policy docstring.

    Args:
        policy: Policy.
    """
    global _GLOBAL_POLICY
    _GLOBAL_POLICY = policy


__all__ = [
    "DTypePolicy",
    "LossScaleOptimizer",
    "Policy",
    "dtype_policy",
    "global_policy",
    "set_dtype_policy",
    "set_global_policy",
]
