from zero_keras import dtype_policies


def test_dtype_policies():
    """Test dtype_policies module."""
    dtype_policies.DTypePolicy()
    dtype_policies.DTypePolicyMap()
    dtype_policies.FloatDTypePolicy()
    dtype_policies.QuantizedDTypePolicy()
    dtype_policies.QuantizedFloat8DTypePolicy()

    dtype_policies.deserialize("test")
    dtype_policies.get("test")
    dtype_policies.serialize("test")


def test_dtype_policy_methods():
    """Test methods of DTypePolicy."""
    policy = dtype_policies.DTypePolicy(name="test_policy")
    assert policy.compute_dtype is None
    assert policy.convert_input(1) == 1
    assert policy.convert_input(1, dtype="float32", exact=True) == 1
    assert policy.get_config() == {"name": "test_policy"}
    new_policy = dtype_policies.DTypePolicy.from_config({"name": "test_policy_new"})
    assert new_policy.name == "test_policy_new"
    assert policy.quantization_mode is None
    assert policy.variable_dtype is None


def test_float_dtype_policy_methods():
    """Test methods of FloatDTypePolicy."""
    policy = dtype_policies.FloatDTypePolicy(name="float_test")
    assert policy.compute_dtype is None
    assert policy.convert_input(1) == 1
    assert policy.convert_input(1, dtype="float32", exact=True) == 1
    assert policy.name == "float_test"
    policy.name = "new_float_test"
    assert policy.name == "new_float_test"
    assert policy.get_config() == {"name": "new_float_test"}
    new_policy = dtype_policies.FloatDTypePolicy.from_config(
        {"name": "float_from_config"}
    )
    assert new_policy.name == "float_from_config"
    assert policy.quantization_mode is None
    assert policy.variable_dtype is None


def test_dtype_policies_methods():
    from zero_keras.dtype_policies import (
        DTypePolicyMap,
        QuantizedDTypePolicy,
        QuantizedFloat8DTypePolicy,
    )

    d = DTypePolicyMap()
    assert d["key"] is None
    assert list(iter(d)) == []
    assert len(d) == 0
    d.clear()
    assert d.compute_dtype is None
    assert d.convert_input(1) == 1
    assert d.get("key") is None
    assert d.get_config() == {}
    d2 = DTypePolicyMap.from_config({})
    assert list(d.items()) == []
    assert list(d.keys()) == []
    assert d.name is None
    assert d.pop("key") is None
    assert d.popitem() is None
    assert d.quantization_mode is None
    assert d.setdefault("key") is None
    d.update({})
    assert list(d.values()) == []
    assert d.variable_dtype is None

    q = QuantizedDTypePolicy()
    assert q.compute_dtype is None
    assert q.convert_input(1) == 1
    assert q.get_config() == {}
    q2 = QuantizedDTypePolicy.from_config({})
    assert q.name is None
    assert q.quantization_mode is None
    assert q.variable_dtype is None

    qf = QuantizedFloat8DTypePolicy()
    assert qf.amax_history_length is None
    assert qf.default_amax_history_length is None
