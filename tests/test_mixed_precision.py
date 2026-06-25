from zero_keras import mixed_precision


def test_mixed_precision():
    """Test mixed_precision module."""
    mixed_precision.DTypePolicy()
    mixed_precision.Policy()
    mixed_precision.dtype_policy()
    mixed_precision.global_policy()
    mixed_precision.set_dtype_policy("test")
    mixed_precision.set_global_policy("test")

    # LossScaleOptimizer is already tested elsewhere or is just a class reference
    assert hasattr(mixed_precision, "LossScaleOptimizer")
