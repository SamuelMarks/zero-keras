"""Module docstring."""

from __future__ import annotations
from zero_keras.types import TensorLike

import numpy as np

from zero_keras.layers import FlaxLayer, JaxLayer, TFSMLayer, TorchModuleWrapper


def dummy_module(x: TensorLike, **kwargs: TensorLike) -> TensorLike:
    """Function docstring.

    Args:
        x: Description.
        kwargs: Description.
    """
    return x + 1


def dummy_jax_module(params: TensorLike, x: TensorLike, **kwargs: TensorLike) -> TensorLike:
    """Function docstring.

    Args:
        params: Description.
        x: Description.
        kwargs: Description.
    """
    return x + 1


def dummy_module_tuple(x: TensorLike, **kwargs: TensorLike) -> TensorLike:
    """Function docstring.

    Args:
        x: Description.
        kwargs: Description.
    """
    return (x + 1, x + 2)


def test_torch_module_wrapper() -> TensorLike:
    """Function docstring."""
    layer = TorchModuleWrapper(module=dummy_module)

    layer2 = TorchModuleWrapper(module=dummy_module_tuple)
    res2 = layer2(np.array([1, 2]))
    assert np.array_equal(res2[0].data, [2, 3])
    assert np.array_equal(res2[1].data, [3, 4])


def test_jax_layer() -> TensorLike:
    """Function docstring."""
    layer = JaxLayer(call_fn=dummy_jax_module)


def test_flax_layer() -> TensorLike:
    """Function docstring."""
    layer = FlaxLayer(module=dummy_module)


def test_tfsm_layer() -> TensorLike:
    """Function docstring."""
    layer = TFSMLayer("path")
    # without module set, returns identity

    layer.module = dummy_module
