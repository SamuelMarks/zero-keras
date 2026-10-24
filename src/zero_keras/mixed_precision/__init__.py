"""Mixed precision policy management API for zero_keras."""

from __future__ import annotations

from zero_keras import config as zk_config
from zero_keras.dtype_policies import DTypePolicy
from zero_keras.optimizers import LossScaleOptimizer


class Policy(DTypePolicy):
    """Deprecated alias for ``DTypePolicy`` in mixed_precision."""


def dtype_policy() -> DTypePolicy | str | None:
    """Return the active global dtype policy.

    Returns:
        DTypePolicy | str | None: Current global dtype policy.
    """
    return zk_config.dtype_policy()


def global_policy() -> DTypePolicy | str | None:
    """Return the active global mixed-precision policy.

    Returns:
        DTypePolicy | str | None: Current global policy.
    """
    return zk_config.dtype_policy()


def set_dtype_policy(policy: DTypePolicy | str | None) -> None:
    """Set the global dtype policy.

    Args:
        policy: Policy instance or string name.
    """
    zk_config.set_dtype_policy(policy)


def set_global_policy(policy: DTypePolicy | str | None) -> None:
    """Set the global mixed-precision policy.

    Args:
        policy: Policy instance or string name.
    """
    zk_config.set_dtype_policy(policy)


__all__ = [
    "DTypePolicy",
    "LossScaleOptimizer",
    "Policy",
    "dtype_policy",
    "global_policy",
    "set_dtype_policy",
    "set_global_policy",
]
