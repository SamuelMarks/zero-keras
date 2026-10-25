"""Unit and behavioral tests for utils in zero_keras."""

from __future__ import annotations
from zero_keras.types import TensorLike

import keras
import numpy as np


def assert_allclose_keras_zero(
    keras_out: TensorLike, zero_out: TensorLike, rtol: TensorLike = 1e-5, atol: TensorLike = 1e-5
) -> None:
    """Assert that the output from Keras and Zero Keras are close.

    Args:
        keras_out: Test fixture or input parameter `keras_out` (keras out).
        zero_out: Test fixture or input parameter `zero_out` (zero out).
        rtol: Test fixture or input parameter `rtol` (rtol).
        atol: Test fixture or input parameter `atol` (atol).
    """
    keras_np = np.asarray(keras_out)
    zero_np = np.asarray(zero_out)

    np.testing.assert_allclose(keras_np, zero_np, rtol=rtol, atol=atol)


def set_seed(seed: TensorLike = 42) -> None:
    """Sets the random seed for reproducible tests across Keras backends and numpy.

    Args:
        seed: Test fixture or input parameter `seed` (seed).
    """
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
