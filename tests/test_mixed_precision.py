from zero_keras import mixed_precision


def test_mixed_precision():
    """Test mixed precision module."""
    # Test defaults
    assert mixed_precision.dtype_policy() is None
    assert mixed_precision.global_policy() is None

    # Test setters/getters
    policy1 = mixed_precision.Policy("test_policy_1")
    mixed_precision.set_dtype_policy(policy1)
    assert mixed_precision.dtype_policy() == policy1
    assert mixed_precision.global_policy() == policy1

    policy2 = mixed_precision.Policy("test_policy_2")
    mixed_precision.set_global_policy(policy2)
    assert mixed_precision.global_policy() == policy2
    assert mixed_precision.dtype_policy() == policy2
