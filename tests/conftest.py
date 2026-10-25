"""Pytest configuration and backend fixtures for zero_keras tests."""

from __future__ import annotations

import math
import os

from ml_switcheroo_compiler.backends.numpy.eager import numpy_eager_registry
import ml_switcheroo_compiler.ops as msc_ops
import pytest

from zero_keras import ops
from zero_keras.types import TensorLike

os.environ["KERAS_BACKEND"] = "numpy"


def to_flat_floats(x: TensorLike) -> list[float]:
    """Flatten any tensor, array, scalar, or nested sequence into a list of Python floats.

    Args:
        x: Input tensor, scalar, or nested sequence.

    Returns:
        list[float]: Flattened list of float values.
    """
    if isinstance(x, (int, float, bool)):
        return [float(x)]
    if isinstance(x, (list, tuple)):
        out: list[float] = []
        for item in x:
            out.extend(to_flat_floats(item))
        return out
    if hasattr(x, "tolist"):
        return to_flat_floats(x.tolist())
    if hasattr(x, "numpy") and callable(x.numpy):
        return to_flat_floats(x.numpy())
    if hasattr(x, "data") and x.data is not x:
        return to_flat_floats(x.data)
    t = ops.convert_to_numpy(ops.convert_to_tensor(x))
    if hasattr(t, "tolist"):
        return to_flat_floats(t.tolist())
    return [float(t)]


def is_close_zero(
    actual: TensorLike,
    expected: TensorLike,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> bool:
    """Return whether ``actual`` and ``expected`` are element-wise close within tolerances.

    Args:
        actual: Actual tensor, scalar, or sequence.
        expected: Expected tensor, scalar, or sequence.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        bool: True if all elements are within tolerance.
    """
    flat_a = to_flat_floats(actual)
    flat_b = to_flat_floats(expected)
    if len(flat_a) != len(flat_b):
        if len(flat_a) == 1:
            flat_a = flat_a * len(flat_b)
        elif len(flat_b) == 1:
            flat_b = flat_b * len(flat_a)
        else:
            return False
    for a_val, b_val in zip(flat_a, flat_b):
        if math.isnan(a_val) and math.isnan(b_val):
            continue
        if not math.isclose(a_val, b_val, rel_tol=rtol, abs_tol=atol):
            return False
    return True


def assert_allclose_zero(
    actual: TensorLike,
    expected: TensorLike,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    """Assert that ``actual`` and ``expected`` are element-wise close within tolerances.

    Args:
        actual: Actual tensor, scalar, or sequence.
        expected: Expected tensor, scalar, or sequence.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
    """
    assert is_close_zero(actual, expected, rtol=rtol, atol=atol), (
        f"Arrays not close: actual={to_flat_floats(actual)} vs expected={to_flat_floats(expected)}"
    )


def assert_array_equal_zero(actual: TensorLike, expected: TensorLike) -> None:
    """Assert that ``actual`` and ``expected`` have identical element values.

    Args:
        actual: Actual tensor, scalar, or sequence.
        expected: Expected tensor, scalar, or sequence.
    """
    flat_a = to_flat_floats(actual)
    flat_b = to_flat_floats(expected)
    assert flat_a == flat_b, f"Arrays not equal: actual={flat_a} vs expected={flat_b}"


@pytest.fixture(autouse=True)
def fallback_registry_more() -> None:
    """Ensure ``RandomNormal`` and ``StopGradient`` are registered in the eager registry."""
    if "RandomNormal" not in numpy_eager_registry._registry:
        numpy_eager_registry.register("RandomNormal")(
            lambda np_mod, *args, **kwargs: np_mod.random.normal(
                size=kwargs.get("shape", args[0] if args else (1,))
            )
        )
    if "StopGradient" not in numpy_eager_registry._registry:
        numpy_eager_registry.register("StopGradient")(lambda np_mod, x, *args, **kwargs: x)


@pytest.fixture(autouse=True)
def fallback_registry_concat2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bridge ``dim`` and ``axis`` keyword arguments in ``ml_switcheroo_compiler.ops.concatenate``.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """

    def mock_concat_eager(
        np_mod: TensorLike,
        x: TensorLike,
        dim: int | None = None,
        axis: int | None = None,
        *args: TensorLike,
        **kwargs: TensorLike,
    ) -> TensorLike:
        """Concatenate ``x`` along ``dim`` or ``axis`` in the eager registry.

        Args:
            np_mod: Backend module reference.
            x: Sequence of arrays to concatenate.
            dim: Concatenation dimension.
            axis: Concatenation axis.
            *args: Extra positional arguments.
            **kwargs: Extra keyword arguments.

        Returns:
            TensorLike: Concatenated array.
        """
        _ = args, kwargs
        ax = dim if dim is not None else (axis if axis is not None else 0)
        return np_mod.concatenate(x, axis=ax)

    numpy_eager_registry.register("Concatenate")(mock_concat_eager)

    orig_concat = msc_ops.concatenate

    def mock_concat(
        arrays: TensorLike,
        dim: int | None = None,
        axis: int | None = None,
        **kwargs: TensorLike,
    ) -> TensorLike:
        """Concatenate ``arrays`` along ``dim`` or ``axis``.

        Args:
            arrays: Sequence of tensors or arrays.
            dim: Optional dimension argument.
            axis: Optional axis argument.
            **kwargs: Extra keyword arguments.

        Returns:
            TensorLike: Concatenated tensor.
        """
        _ = kwargs
        ax = dim if dim is not None else (axis if axis is not None else 0)
        tensors = [ops.convert_to_tensor(a) for a in arrays]
        return orig_concat(tensors, axis=ax)

    monkeypatch.setattr(msc_ops, "concatenate", mock_concat, raising=False)


@pytest.fixture(autouse=True)
def mock_one_hot_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure ``ml_switcheroo_compiler.ops.one_hot`` supports ``axis`` and ``dtype``.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """

    def mock_one_hot(
        indices: TensorLike,
        num_classes: int,
        axis: int = -1,
        dtype: str | None = "float32",
        **kwargs: TensorLike,
    ) -> TensorLike:
        """Compute one-hot encoding for ``indices`` with ``num_classes`` depth.

        Args:
            indices: Integer index tensor or array.
            num_classes: Number of classes (one-hot depth).
            axis: Axis along which one-hot representation is placed.
            dtype: Output data type string.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: One-hot encoded tensor.
        """
        _ = kwargs
        target_dtype = dtype if dtype is not None else "float32"
        idx_t = ops.cast(ops.convert_to_tensor(indices), "int32")
        eye_t = ops.eye(num_classes, dtype=target_dtype)
        res = ops.take(eye_t, idx_t, axis=0)
        if axis != -1:
            res = ops.swapaxes(res, axis, -1)
        return res

    monkeypatch.setattr(msc_ops, "one_hot", mock_one_hot, raising=False)
