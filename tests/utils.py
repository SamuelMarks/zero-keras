"""Module docstring."""

import numpy as np
import keras


def assert_allclose_keras_zero(keras_out, zero_out, rtol=1e-5, atol=1e-5):
    """
    Asserts that the output from Keras and Zero Keras are close.
    Handles converting tf/torch/jax tensors to numpy arrays if necessary.
    """
    keras_np = np.asarray(keras_out)
    zero_np = np.asarray(zero_out)

    np.testing.assert_allclose(keras_np, zero_np, rtol=rtol, atol=atol)


def set_seed(seed=42):
    """
    Sets the random seed for reproducible tests across Keras backends and numpy.
    """
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
