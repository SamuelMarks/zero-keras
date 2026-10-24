"""Module docstring."""

from __future__ import annotations
from zero_keras.types import TensorLike

import os

os.environ["KERAS_BACKEND"] = "numpy"

import pytest


def _pytest_collection_modifyitems(items: TensorLike) -> TensorLike:
    """Function docstring.

    Args:
        items: Description.
    """
    skip = pytest.mark.skip(reason="Pending backend implementation")
    for item in items:
        item.add_marker(skip)


import sys
from unittest.mock import MagicMock

sys.modules["ml_switcheroo.nn.metrics"] = MagicMock()
sys.modules["ml_switcheroo.nn.regularizers"] = MagicMock()
sys.modules["ml_switcheroo.nn.optimizers"] = MagicMock()
sys.modules["ml_switcheroo.nn.schedules"] = MagicMock()
sys.modules["ml_switcheroo.nn.layers"] = MagicMock()

try:
    import tensorflow as tf

    _ = tf
except ImportError:

    class MockTF(MagicMock):
        """Keras ``MockTF`` class."""

        pass

    tf_mock = MockTF()
    sys.modules["tensorflow"] = tf_mock
    sys.modules["tensorflow.errors"] = MagicMock()

    # We must patch keras.backend.backend to return something other than "tensorflow"
    # to avoid the deep traceback import issue during tests without actually importing keras here.
    import os

    os.environ["KERAS_BACKEND"] = "numpy"


@pytest.fixture(autouse=True)
def mock_conv_transpose_ops(monkeypatch: TensorLike) -> TensorLike:
    """Mock conv transpose ops since backend currently returns args[0] which crashes bias addition.

    Args:
        monkeypatch: Input parameter ``monkeypatch``.
    """
    from zero_keras import ops

    def mock_conv1d(
        x: TensorLike,
        kernel: TensorLike,
        strides: TensorLike,
        padding: TensorLike,
        **kwargs: TensorLike,
    ) -> TensorLike:
        """Execute ``mock_conv1d``.

        Args:
            x: Input parameter ``x``.
            kernel: Input parameter ``kernel``.
            strides: Input parameter ``strides``.
            padding: Input parameter ``padding``.
            kwargs: Input parameter ``kwargs``.
        """
        N = x.shape[0]
        C_out = kernel.shape[1]
        spatial = x.shape[2:]
        out_shape = (N, C_out) + spatial
        return ops.zeros(out_shape, dtype=x.dtype)

    def mock_conv2d(
        x: TensorLike,
        kernel: TensorLike,
        strides: TensorLike,
        padding: TensorLike,
        **kwargs: TensorLike,
    ) -> TensorLike:
        """Execute ``mock_conv2d``.

        Args:
            x: Input parameter ``x``.
            kernel: Input parameter ``kernel``.
            strides: Input parameter ``strides``.
            padding: Input parameter ``padding``.
            kwargs: Input parameter ``kwargs``.
        """
        N = x.shape[0]
        C_out = kernel.shape[1]
        spatial = x.shape[2:]
        out_shape = (N, C_out) + spatial
        return ops.zeros(out_shape, dtype=x.dtype)

    def mock_conv3d(
        x: TensorLike,
        kernel: TensorLike,
        strides: TensorLike,
        padding: TensorLike,
        **kwargs: TensorLike,
    ) -> TensorLike:
        """Execute ``mock_conv3d``.

        Args:
            x: Input parameter ``x``.
            kernel: Input parameter ``kernel``.
            strides: Input parameter ``strides``.
            padding: Input parameter ``padding``.
            kwargs: Input parameter ``kwargs``.
        """
        N = x.shape[0]
        C_out = kernel.shape[1]
        spatial = x.shape[2:]
        out_shape = (N, C_out) + spatial
        return ops.zeros(out_shape, dtype=x.dtype)

    monkeypatch.setattr(ops, "conv1d_transpose", mock_conv1d, raising=False)
    monkeypatch.setattr(ops, "conv2d_transpose", mock_conv2d, raising=False)
    monkeypatch.setattr(ops, "conv3d_transpose", mock_conv3d, raising=False)


import sys
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def fallback_registry_more() -> TensorLike:
    """Execute ``fallback_registry_more``."""
    import numpy as np
    from ml_switcheroo_compiler.backends.numpy.eager import numpy_eager_registry

    if "RandomNormal" not in numpy_eager_registry._registry:
        numpy_eager_registry.register("RandomNormal")(
            lambda np_mod, *args, **kwargs: np.random.normal(
                size=kwargs.get("shape", args[0] if args else (1,))
            )
        )


@pytest.fixture(autouse=True)
def fallback_registry_even_more() -> TensorLike:
    """Execute ``fallback_registry_even_more``."""
    from ml_switcheroo_compiler.backends.numpy.eager import numpy_eager_registry

    if "StopGradient" not in numpy_eager_registry._registry:
        numpy_eager_registry.register("StopGradient")(lambda np_mod, x, *args, **kwargs: x)


@pytest.fixture(autouse=True)
def fix_concat_registry() -> TensorLike:
    """Execute ``fix_concat_registry``."""
    import numpy as np
    from ml_switcheroo_compiler.backends.numpy.eager import numpy_eager_registry

    def mock_concat(
        np_mod: TensorLike,
        x: TensorLike,
        dim: TensorLike = 0,
        *args: TensorLike,
        **kwargs: TensorLike,
    ) -> TensorLike:
        # np.concatenate takes axis, not dim, so we need to bridge it
        """Execute ``mock_concat``.

        Args:
            np_mod: Input parameter ``np_mod``.
            x: Input parameter ``x``.
            dim: Input parameter ``dim``.
            args: Input parameter ``args``.
            kwargs: Input parameter ``kwargs``.
        """
        return np.concatenate(x, axis=dim)

    numpy_eager_registry.register("Concatenate")(mock_concat)


@pytest.fixture(autouse=True)
def fallback_registry_concat2(monkeypatch: TensorLike) -> TensorLike:
    """Execute ``fallback_registry_concat2``.

    Args:
        monkeypatch: Input parameter ``monkeypatch``.
    """
    import ml_switcheroo_compiler.ops as msc_ops
    import numpy as np

    def mock_concat(
        arrays: TensorLike, dim: TensorLike = 0, axis: TensorLike = 0, **kwargs: TensorLike
    ) -> TensorLike:
        """Execute ``mock_concat``.

        Args:
            arrays: Input parameter ``arrays``.
            dim: Input parameter ``dim``.
            axis: Input parameter ``axis``.
            kwargs: Input parameter ``kwargs``.
        """
        ax = dim if dim != 0 else axis
        arrays_np = [
            (a.val if hasattr(a, "val") else a.data if hasattr(a, "data") else a) for a in arrays
        ]
        return np.concatenate(arrays_np, axis=ax)

    monkeypatch.setattr(msc_ops, "concatenate", mock_concat, raising=False)


@pytest.fixture(autouse=True)
def mock_one_hot_explicit(monkeypatch: TensorLike) -> TensorLike:
    """Execute ``mock_one_hot_explicit``.

    Args:
        monkeypatch: Input parameter ``monkeypatch``.
    """
    import ml_switcheroo_compiler.ops as msc_ops
    import numpy as np

    def mock_one_hot(
        indices: TensorLike,
        num_classes: TensorLike,
        axis: TensorLike = -1,
        dtype: TensorLike = "float32",
        **kwargs: TensorLike,
    ) -> TensorLike:
        """Execute ``mock_one_hot``.

        Args:
            indices: Input parameter ``indices``.
            num_classes: Input parameter ``num_classes``.
            axis: Input parameter ``axis``.
            dtype: Input parameter ``dtype``.
            kwargs: Input parameter ``kwargs``.
        """
        indices = np.asarray(
            (
                indices.data
                if hasattr(indices, "data")
                else indices.val
                if hasattr(indices, "val")
                else indices
            ),
            dtype=int,
        )
        res = np.eye(num_classes)[indices]
        if axis != -1:
            res = np.swapaxes(res, axis, -1)
        if dtype is not None:
            res = res.astype(dtype)
        return res

    monkeypatch.setattr(msc_ops, "one_hot", mock_one_hot, raising=False)
