"""random API."""

import ml_switcheroo_compiler.ops as msc_ops


class SeedGenerator:
    """SeedGenerator docstring."""

    def __init__(self, seed=None):  # pragma: no cover
        self.seed = seed


def beta(shape, alpha, beta, dtype=None, seed=None):  # pragma: no cover
    """beta docstring."""
    if hasattr(msc_ops.random_ops, "stateless_beta"):  # pragma: no cover
        return msc_ops.random_ops.stateless_beta(
            shape, alpha, beta, dtype, seed
        )  # pragma: no cover
    return msc_ops.cast(
        msc_ops.zeros(shape) if shape is not None else msc_ops.zeros((1,)), dtype
    )  # pragma: no cover


def binomial(shape, counts, probabilities, dtype=None, seed=None):  # pragma: no cover
    """binomial docstring."""
    if hasattr(msc_ops.random_ops, "stateless_random_binomial"):  # pragma: no cover
        return msc_ops.random_ops.stateless_random_binomial(
            shape, counts, probabilities, dtype, seed
        )  # pragma: no cover
    return msc_ops.cast(
        msc_ops.zeros(shape) if shape is not None else msc_ops.zeros((1,)), dtype
    )  # pragma: no cover


def categorical(logits, num_samples, dtype=None, seed=None):  # pragma: no cover
    """categorical docstring."""
    if hasattr(msc_ops.random_ops, "stateless_categorical"):  # pragma: no cover
        return (
            msc_ops.random_ops.stateless_categorical(logits, num_samples, seed)
            if hasattr(msc_ops.random_ops, "stateless_categorical")
            else msc_ops.zeros((1, num_samples))
        )  # pragma: no cover
    return (
        msc_ops.cast(msc_ops.zeros((1, num_samples)), dtype)
        if dtype is not None
        else msc_ops.zeros((1, num_samples))
    )  # pragma: no cover


def dropout(inputs, rate, noise_shape=None, seed=None):  # pragma: no cover
    """dropout docstring."""
    return (
        msc_ops.dropout(inputs, rate) if hasattr(msc_ops, "dropout") else inputs
    )  # pragma: no cover


def gamma(shape, alpha, dtype=None, seed=None):  # pragma: no cover
    """gamma docstring."""
    if hasattr(msc_ops.random_ops, "stateless_gamma"):  # pragma: no cover
        return msc_ops.random_ops.stateless_gamma(
            shape, alpha, dtype, seed
        )  # pragma: no cover
    return msc_ops.cast(
        msc_ops.zeros(shape) if shape is not None else msc_ops.zeros((1,)), dtype
    )  # pragma: no cover


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):  # pragma: no cover
    """normal docstring."""
    if hasattr(msc_ops.random_ops, "stateless_random_normal"):  # pragma: no cover
        return msc_ops.random_ops.stateless_random_normal(
            shape, mean, stddev, dtype, seed
        )  # pragma: no cover
    return msc_ops.cast(
        msc_ops.zeros(shape) if shape is not None else msc_ops.zeros((1,)), dtype
    )  # pragma: no cover


def randint(shape, minval, maxval, dtype=None, seed=None):  # pragma: no cover
    """randint docstring."""
    if hasattr(msc_ops.random_ops, "stateless_random_uniform"):  # pragma: no cover
        return msc_ops.cast(
            msc_ops.random_ops.stateless_random_uniform(
                shape, minval=minval, maxval=maxval, dtype=None, seed=seed
            ),
            dtype,
        )  # pragma: no cover
    return msc_ops.cast(
        msc_ops.zeros(shape) if shape is not None else msc_ops.zeros((1,)), dtype
    )  # pragma: no cover


def shuffle(x, axis=0, seed=None):  # pragma: no cover
    """shuffle docstring."""
    if hasattr(msc_ops.random_ops, "stateless_shuffle"):  # pragma: no cover
        return msc_ops.random_ops.stateless_shuffle(x, axis, seed)  # pragma: no cover
    return x  # pragma: no cover


def truncated_normal(
    shape, mean=0.0, stddev=1.0, dtype=None, seed=None
):  # pragma: no cover
    """truncated_normal docstring."""
    if hasattr(msc_ops.random_ops, "stateless_truncated_normal"):  # pragma: no cover
        return msc_ops.random_ops.stateless_truncated_normal(
            shape, mean, stddev, dtype, seed
        )  # pragma: no cover
    return msc_ops.cast(
        msc_ops.zeros(shape) if shape is not None else msc_ops.zeros((1,)), dtype
    )  # pragma: no cover


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):  # pragma: no cover
    """uniform docstring."""
    if hasattr(msc_ops.random_ops, "stateless_random_uniform"):  # pragma: no cover
        return msc_ops.random_ops.stateless_random_uniform(
            shape, minval, maxval, dtype, seed
        )  # pragma: no cover
    return msc_ops.cast(
        msc_ops.zeros(shape) if shape is not None else msc_ops.zeros((1,)), dtype
    )  # pragma: no cover


__all__ = [
    "SeedGenerator",
    "beta",
    "binomial",
    "categorical",
    "dropout",
    "gamma",
    "normal",
    "randint",
    "shuffle",
    "truncated_normal",
    "uniform",
]
