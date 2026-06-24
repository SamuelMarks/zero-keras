"""Module docstring."""

import numpy as np
from zero_keras.layers import TorchModuleWrapper, JaxLayer, FlaxLayer, TFSMLayer


def dummy_module(x, **kwargs):
    """Function docstring.

    Args:
        x: Description.
        kwargs: Description.
    """
    return x + 1


def dummy_jax_module(params, x, **kwargs):
    """Function docstring.

    Args:
        params: Description.
        x: Description.
        kwargs: Description.
    """
    return x + 1


def dummy_module_tuple(x, **kwargs):
    """Function docstring.

    Args:
        x: Description.
        kwargs: Description.
    """
    return (x + 1, x + 2)


def test_torch_module_wrapper():
    """Function docstring."""
    layer = TorchModuleWrapper(module=dummy_module)
    res = layer(np.array([1, 2]))
    assert np.array_equal(res.data, [2, 3])

    layer2 = TorchModuleWrapper(module=dummy_module_tuple)
    res2 = layer2(np.array([1, 2]))
    assert np.array_equal(res2[0].data, [2, 3])
    assert np.array_equal(res2[1].data, [3, 4])


def test_jax_layer():
    """Function docstring."""
    layer = JaxLayer(call_fn=dummy_jax_module)
    res = layer(np.array([1, 2]))
    assert np.array_equal(res.data, [2, 3])


def test_flax_layer():
    """Function docstring."""
    layer = FlaxLayer(module=dummy_module)
    res = layer(np.array([1, 2]))
    assert np.array_equal(res.data, [2, 3])


def test_tfsm_layer():
    """Function docstring."""
    layer = TFSMLayer("path")
    # without module set, returns identity
    res = layer(np.array([1, 2]))
    assert np.array_equal(res.data, [1, 2])

    layer.module = dummy_module
    res = layer(np.array([1, 2]))
    assert np.array_equal(res.data, [2, 3])
