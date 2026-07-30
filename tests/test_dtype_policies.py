from zero_keras import dtype_policies, ops


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
    assert getattr(policy, "variable_dtype", None) in (None, "float32")


def test_float_dtype_policy_methods():
    """Test methods of FloatDTypePolicy."""
    policy = dtype_policies.FloatDTypePolicy(name="float_test")
    assert policy.compute_dtype == "float32"
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
    assert getattr(policy, "variable_dtype", None) in (None, "float32")


def test_dtype_policies_methods():
    from zero_keras.dtype_policies import (
        DTypePolicyMap,
        QuantizedDTypePolicy,
        QuantizedFloat8DTypePolicy,
    )

    d = DTypePolicyMap(default_policy="default")
    assert d["key"] == "default"
    d["key"] = "test"
    assert d["key"] == "test"
    assert list(iter(d)) == ["key"]
    assert len(d) == 1

    assert list(d.items()) == [("key", "test")]
    assert list(d.keys()) == ["key"]
    assert list(d.values()) == ["test"]

    d.clear()
    assert len(d) == 0

    assert d.compute_dtype == "float32"
    # use ops.convert_to_numpy
    assert float(ops.convert_to_numpy(d.convert_input(1.0))) == 1.0
    assert float(ops.convert_to_numpy(d.convert_input(1.0, dtype="float32"))) == 1.0

    assert d.get("missing") == "default"
    d["key"] = "test"
    assert d.get("key") == "test"

    d.name = "test_map"
    assert d.name == "test_map"
    assert d.get_config() == {"name": "test_map"}

    d2 = DTypePolicyMap.from_config({"default_policy": "default2"})
    assert d2.default_policy == "default2"

    assert d.pop("key") == "test"
    assert d.pop("missing", "fb") == "fb"

    d["k1"] = "v1"
    assert d.popitem() == ("k1", "v1")

    assert d.quantization_mode is None
    assert d.setdefault("k2", "v2") == "v2"
    assert d["k2"] == "v2"

    d.update({"k3": "v3"})
    assert d["k3"] == "v3"

    assert d.variable_dtype == "float32"

    q = QuantizedDTypePolicy()
    assert q.compute_dtype == "float32"
    assert float(ops.convert_to_numpy(q.convert_input(1.0))) == 1.0
    assert float(ops.convert_to_numpy(q.convert_input(1.0, dtype="float32"))) == 1.0
    assert q.get_config() == {"name": None}
    q2 = QuantizedDTypePolicy.from_config({})
    assert q.name is None
    assert q.quantization_mode is None
    assert q.variable_dtype == "float32"

    qf = QuantizedFloat8DTypePolicy()
    assert qf.amax_history_length is None
    assert qf.default_amax_history_length is None
