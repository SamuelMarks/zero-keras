"""Module docstring."""

from __future__ import annotations
from zero_keras.types import TensorLike

import keras
import numpy as np


def assert_allclose_keras_zero(
    keras_out: TensorLike, zero_out: TensorLike, rtol: TensorLike = 1e-5, atol: TensorLike = 1e-5
) -> TensorLike:
    """Assert that the output from Keras and Zero Keras are close.

    Handles converting tf/torch/jax tensors to numpy arrays if necessary.

    Args:
        keras_out: Input parameter ``keras_out``.
        zero_out: Input parameter ``zero_out``.
        rtol: Input parameter ``rtol``.
        atol: Input parameter ``atol``.
    """
    keras_np = np.asarray(keras_out)
    zero_np = np.asarray(zero_out)

    np.testing.assert_allclose(keras_np, zero_np, rtol=rtol, atol=atol)


def set_seed(seed: TensorLike = 42) -> TensorLike:
    """Sets the random seed for reproducible tests across Keras backends and numpy.

    Args:
        seed: Input parameter ``seed``.
    """
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
