"""Comprehensive tests for ``zero_keras.dtype_policies``."""

from __future__ import annotations

import numpy as np

from zero_keras import dtype_policies, ops


def test_dtype_policies() -> None:
    """Verify policy instantiation, parsing, and serialization round-trips."""
    p_base = dtype_policies.DTypePolicy()
    assert p_base.name is None
    assert p_base.compute_dtype is None
    assert p_base.variable_dtype is None

    p_mixed16 = dtype_policies.DTypePolicy("mixed_float16")
    assert p_mixed16.compute_dtype == "float16"
    assert p_mixed16.variable_dtype == "float32"

    p_mixed_bf16 = dtype_policies.DTypePolicy("mixed_bfloat16")
    assert p_mixed_bf16.compute_dtype == "bfloat16"
    assert p_mixed_bf16.variable_dtype == "float32"

    p_f32 = dtype_policies.DTypePolicy("float32")
    assert p_f32.compute_dtype == "float32"
    assert p_f32.variable_dtype == "float32"

    p_qfrom = dtype_policies.DTypePolicy("int8_from_mixed_float16")
    assert p_qfrom.compute_dtype == "float16"
    assert p_qfrom.variable_dtype == "float32"
    assert p_qfrom.quantization_mode == "int8"

    p_qfrom_unk = dtype_policies.DTypePolicy("int8_from_custom")
    assert p_qfrom_unk.compute_dtype == "float32"
    assert p_qfrom_unk.variable_dtype == "float32"

    p_qshort = dtype_policies.DTypePolicy("int8")
    assert p_qshort.quantization_mode == "int8"
    p_qshort.name = "int4"
    assert p_qshort.quantization_mode == "int4"

    # Serialization / Deserialization / Get
    assert dtype_policies.deserialize("test") == "test"
    assert dtype_policies.get("test") == "test"
    assert dtype_policies.serialize("test") == "test"

    for inst in [
        p_mixed16,
        dtype_policies.FloatDTypePolicy("float32"),
        dtype_policies.QuantizedDTypePolicy("int8"),
        dtype_policies.QuantizedFloat8DTypePolicy("float8"),
        dtype_policies.DTypePolicyMap(name="map1"),
    ]:
        ser = dtype_policies.serialize(inst)
        assert isinstance(ser, dict)
        deser = dtype_policies.get(ser)
        assert deser is not None

    assert dtype_policies.deserialize({"class_name": "UnknownPolicy"}) == {
        "class_name": "UnknownPolicy"
    }


def test_dtype_policy_methods() -> None:
    """Test methods of DTypePolicy."""
    policy = dtype_policies.DTypePolicy(name="test_policy")
    assert policy.compute_dtype is None
    assert policy.convert_input(1) == 1
    assert policy.convert_input(1, dtype="float32", exact=True) == 1
    t = ops.convert_to_tensor([1.0, 2.0], dtype="float32")
    assert policy.convert_input(t) is t
    t_cast = policy.convert_input(t, dtype="float16")
    assert ops.dtype(t_cast) == "float16"

    assert policy.get_config() == {"name": "test_policy"}
    new_policy = dtype_policies.DTypePolicy.from_config({"name": "test_policy_new"})
    assert new_policy.name == "test_policy_new"
    assert policy.quantization_mode is None
    assert policy.variable_dtype is None


def test_float_dtype_policy_methods() -> None:
    """Test methods of FloatDTypePolicy."""
    policy = dtype_policies.FloatDTypePolicy(name="float_test")
    assert policy.compute_dtype == "float32"
    assert policy.variable_dtype == "float32"
    assert policy.convert_input(1) == 1
    assert policy.convert_input(1, dtype="float32", exact=True) == 1
    assert policy.name == "float_test"
    policy.name = "custom_unknown_name"
    assert policy.name == "custom_unknown_name"
    policy.name = "float16"
    assert policy.name == "float16"
    assert policy.compute_dtype == "float16"
    assert policy.variable_dtype == "float16"
    assert policy.get_config() == {"name": "float16"}
    new_policy = dtype_policies.FloatDTypePolicy.from_config({"name": "float_from_config"})
    assert new_policy.name == "float_from_config"
    assert policy.quantization_mode is None


def test_dtype_policies_methods() -> None:
    """Test DTypePolicyMap, QuantizedDTypePolicy, and QuantizedFloat8DTypePolicy."""
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
    np.testing.assert_allclose(ops.convert_to_numpy(d.convert_input(1.0)), 1.0)
    np.testing.assert_allclose(ops.convert_to_numpy(d.convert_input(1.0, dtype="float32")), 1.0)

    assert d.get("missing") == "default"
    d_no_def = DTypePolicyMap()
    assert d_no_def.get("missing", "fallback") == "fallback"
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
    np.testing.assert_allclose(ops.convert_to_numpy(q.convert_input(1.0)), 1.0)
    np.testing.assert_allclose(ops.convert_to_numpy(q.convert_input(1.0, dtype="float32")), 1.0)
    assert q.get_config() == {"name": None}
    q2 = QuantizedDTypePolicy.from_config({})
    assert q2.name is None
    assert q.quantization_mode is None
    assert q.variable_dtype == "float32"

    qf = QuantizedFloat8DTypePolicy()
    assert qf.amax_history_length is None
    assert qf.default_amax_history_length is None
