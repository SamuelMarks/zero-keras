"""Comprehensive tests for ``zero_keras.mixed_precision``."""

from __future__ import annotations

from zero_keras import mixed_precision


def test_mixed_precision() -> None:
    """Verify mixed precision policy getters and setters synchronize with global config."""
    mixed_precision.set_global_policy(None)
    assert mixed_precision.dtype_policy() is None
    assert mixed_precision.global_policy() is None

    policy1 = mixed_precision.Policy("mixed_float16")
    assert policy1.compute_dtype == "float16"
    assert policy1.variable_dtype == "float32"
    mixed_precision.set_dtype_policy(policy1)
    assert mixed_precision.dtype_policy() == policy1
    assert mixed_precision.global_policy() == policy1

    policy2 = mixed_precision.Policy("mixed_bfloat16")
    mixed_precision.set_global_policy(policy2)
    assert mixed_precision.global_policy() == policy2
    assert mixed_precision.dtype_policy() == policy2

    # Restore default None state
    mixed_precision.set_global_policy(None)
