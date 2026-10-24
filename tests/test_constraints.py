"""Comprehensive tests for ``zero_keras.constraints``."""

from __future__ import annotations

import numpy as np

from zero_keras import constraints, ops
from zero_keras.types import TensorLike


def test_constraints_methods() -> None:
    """Verify projection, config, and serialization across all weight constraints."""
    from zero_keras.constraints import (
        Constraint,
        MaxNorm,
        MinMaxNorm,
        NonNeg,
        UnitNorm,
    )

    base = Constraint()
    assert base.get_config() == {}
    base_from = Constraint.from_config({})
    assert isinstance(base_from, Constraint)
    w = ops.convert_to_tensor([[3.0, -4.0], [-1.0, 2.0]], dtype="float32")
    np.testing.assert_allclose(ops.convert_to_numpy(base(w)), [[3.0, -4.0], [-1.0, 2.0]])

    # NonNeg clamps negative entries to 0
    nn = NonNeg()
    assert nn.get_config() == {}
    assert isinstance(NonNeg.from_config({}), NonNeg)
    nn_out = ops.convert_to_numpy(nn(w))
    np.testing.assert_allclose(nn_out, [[3.0, 0.0], [0.0, 2.0]])

    # UnitNorm normalizes along axis
    un = UnitNorm(axis=1)
    assert un.get_config() == {"axis": 1}
    un_list = UnitNorm(axis=[0, 1])
    assert un_list.get_config() == {"axis": (0, 1)}
    un_from_int = UnitNorm.from_config({"axis": 1})
    assert un_from_int.axis == 1
    un_from_tuple = UnitNorm.from_config({"axis": [0, 1]})
    assert un_from_tuple.axis == (0, 1)
    un_from_invalid = UnitNorm.from_config({"axis": "bad"})
    assert un_from_invalid.axis == 0
    un_out = ops.convert_to_numpy(un(ops.convert_to_tensor([[3.0, 4.0]], dtype="float32")))
    np.testing.assert_allclose(un_out, [[0.6, 0.8]], rtol=1e-5, atol=1e-5)

    # MaxNorm scales down vectors exceeding max_value
    mn = MaxNorm(max_value=2.5, axis=1)
    assert mn.get_config() == {"max_value": 2.5, "axis": 1}
    mn_list = MaxNorm(max_value=2.0, axis=[0, 1])
    assert mn_list.get_config() == {"max_value": 2.0, "axis": (0, 1)}
    mn_from = MaxNorm.from_config({"max_value": 2.5, "axis": [1]})
    assert mn_from.max_value == 2.5
    assert mn_from.axis == (1,)
    mn_from_bad = MaxNorm.from_config({"max_value": "bad", "axis": "bad"})
    assert mn_from_bad.max_value == 2.0
    assert mn_from_bad.axis == 0
    mn_out = ops.convert_to_numpy(mn(ops.convert_to_tensor([[3.0, 4.0]], dtype="float32")))
    np.testing.assert_allclose(mn_out, [[1.5, 2.0]], rtol=1e-5, atol=1e-5)

    # MinMaxNorm clamps norm between min_value and max_value
    mmn = MinMaxNorm(min_value=1.0, max_value=2.5, rate=1.0, axis=1)
    assert mmn.get_config() == {
        "min_value": 1.0,
        "max_value": 2.5,
        "rate": 1.0,
        "axis": 1,
    }
    mmn_list = MinMaxNorm(axis=[0, 1])
    assert mmn_list.get_config()["axis"] == (0, 1)
    mmn_from_int = MinMaxNorm.from_config({"axis": 1})
    assert mmn_from_int.axis == 1
    mmn_from = MinMaxNorm.from_config(
        {"min_value": 0.5, "max_value": 1.5, "rate": 0.5, "axis": [0]}
    )
    assert mmn_from.min_value == 0.5
    assert mmn_from.axis == (0,)
    mmn_from_bad = MinMaxNorm.from_config(
        {"min_value": "x", "max_value": "y", "rate": "z", "axis": "w"}
    )
    assert mmn_from_bad.min_value == 0.0
    assert mmn_from_bad.axis == 0
    mmn_out = ops.convert_to_numpy(mmn(ops.convert_to_tensor([[3.0, 4.0]], dtype="float32")))
    np.testing.assert_allclose(mmn_out, [[1.5, 2.0]], rtol=1e-5, atol=1e-5)

    # Functional aliases and serialization/deserialization
    assert isinstance(constraints.max_norm(3.0), MaxNorm)
    assert isinstance(constraints.min_max_norm(0.1, 2.0), MinMaxNorm)
    assert isinstance(constraints.non_neg(), NonNeg)
    assert isinstance(constraints.unit_norm(), UnitNorm)

    assert constraints.serialize(None) is None
    assert constraints.serialize("max_norm") == "max_norm"
    ser = constraints.serialize(mn)
    assert ser == {"class_name": "MaxNorm", "config": {"max_value": 2.5, "axis": 1}}

    assert constraints.deserialize(None) is None
    assert constraints.deserialize(mn) is mn
    deser = constraints.deserialize(ser)
    assert isinstance(deser, MaxNorm)
    assert deser.max_value == 2.5

    assert isinstance(constraints.deserialize("non_neg"), NonNeg)
    assert constraints.deserialize("unknown_constraint") == "unknown_constraint"
    assert constraints.deserialize({"class_name": "UnknownConstraint"}) == {
        "class_name": "UnknownConstraint"
    }
    assert constraints.deserialize({"not_a_class": 1}) == {"not_a_class": 1}
    assert constraints.deserialize(123) == 123  # type: ignore[arg-type]

    class CustomConstraint(Constraint):
        """Custom test constraint."""

    def make_non_constraint(**kwargs: TensorLike) -> str:
        """Return a non-constraint string for fallback coverage.

        Args:
            **kwargs: Ignored keyword arguments.

        Returns:
            str: Dummy string.
        """
        _ = kwargs
        return "not_a_constraint"

    custom_res = constraints.deserialize("custom_c", custom_objects={"custom_c": CustomConstraint})
    assert isinstance(custom_res, CustomConstraint)
    assert (
        constraints.deserialize("custom_bad", custom_objects={"custom_bad": make_non_constraint})
        == "custom_bad"
    )
    assert constraints.deserialize("custom_int", custom_objects={"custom_int": 123}) == "custom_int"

    custom_dict_res = constraints.deserialize(
        {"class_name": "CustomConstraint", "config": {}},
        custom_objects={"CustomConstraint": CustomConstraint},
    )
    assert isinstance(custom_dict_res, CustomConstraint)
    assert constraints.deserialize(
        {"class_name": "BadCustom", "config": {}},
        custom_objects={"BadCustom": make_non_constraint},
    ) == {"class_name": "BadCustom", "config": {}}
    assert constraints.deserialize(
        {"class_name": "IntCustom", "config": {}},
        custom_objects={"IntCustom": 123},
    ) == {"class_name": "IntCustom", "config": {}}

    assert constraints.get(None) is None
    assert constraints.get(mn) is mn
    assert isinstance(constraints.get("unit_norm"), UnitNorm)
    assert isinstance(constraints.get(ser), MaxNorm)
    assert constraints.get(123) == 123  # type: ignore[arg-type]
