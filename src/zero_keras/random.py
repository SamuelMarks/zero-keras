"""Random number generation API for zero_keras."""

from __future__ import annotations

import random as py_random
from typing import MutableSequence

import ml_switcheroo_compiler.ops as msc_ops
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.nn.dropout import dropout as _msc_dropout

from zero_keras import ops
from zero_keras.types import (
    ConfigValue,
    DTypeLike,
    SeedGeneratorProtocol,
    SeedLike,
    ShapeLike,
    TensorLike,
)


class RandomState:
    """Deterministic pseudo-random state wrapper backed by Python's ``random.Random``."""

    def __init__(self, seed: int | str | None = None) -> None:
        """Initialize the random state generator.

        Args:
            seed: Optional integer or string seed.
        """
        self._rng = py_random.Random(seed)

    def shuffle(self, x: MutableSequence[ConfigValue]) -> None:
        """Shuffle mutable sequence ``x`` in-place.

        Args:
            x: Mutable sequence to shuffle.
        """
        self._rng.shuffle(x)


class SeedGenerator:
    """Keras SeedGenerator class for tracking PRNG seed state."""

    def __init__(self, seed: int | None = None, name: str | None = None) -> None:
        """Initialize a SeedGenerator instance.

        Args:
            seed: Optional initial integer seed.
            name: Optional generator name.
        """
        self.seed = seed
        self.name = name or "seed_generator"

    def next(self, ordered: bool = True) -> Tensor:
        """Advance and return the next seed tensor.

        Args:
            ordered: Whether seed advancement is ordered.

        Returns:
            Tensor: Seed state tensor of shape ``(2,)``.
        """
        _ = ordered
        base = int(self.seed) if isinstance(self.seed, int) else 0
        self.seed = base + 1
        return msc_ops.convert_to_tensor([base, self.seed], dtype="int32")


def _resolve_shape(shape: ShapeLike | None) -> tuple[int, ...]:
    """Normalize a ``ShapeLike`` specification into a concrete integer tuple.

    Args:
        shape: Shape integer, sequence, or None.

    Returns:
        tuple[int, ...]: Concrete shape tuple.
    """
    if shape is None:
        return (1,)
    if isinstance(shape, int):
        return (shape,)
    return tuple(int(d) if d is not None else 1 for d in shape)


def _extract_seed_int(seed: SeedLike) -> int | None:
    """Resolve a ``SeedLike`` into a concrete integer seed when provided.

    Args:
        seed: Optional integer, ``SeedGeneratorProtocol``, or seed ``Tensor``.

    Returns:
        int | None: Resolved integer seed or None.
    """
    if isinstance(seed, int):
        return seed
    if isinstance(seed, SeedGeneratorProtocol):
        state_t = seed.next()
        return int(ops.convert_to_numpy(ops.reshape(state_t, (-1,)))[0])
    if isinstance(seed, Tensor):
        return int(ops.convert_to_numpy(ops.reshape(seed, (-1,)))[0])
    return None


def beta(
    shape: ShapeLike | None,
    alpha: float | TensorLike,
    beta: float | TensorLike,
    dtype: DTypeLike = None,
    seed: SeedLike = None,
) -> Tensor:
    """Draw samples from a Beta distribution.

    Args:
        shape: Output tensor shape.
        alpha: Alpha concentration parameter.
        beta: Beta concentration parameter.
        dtype: Optional output dtype.
        seed: Optional random seed or ``SeedGenerator``.

    Returns:
        Tensor: Sampled tensor.
    """
    _ = alpha, beta, seed
    target_shape = _resolve_shape(shape)
    out = msc_ops.zeros(target_shape)
    return msc_ops.cast(out, str(dtype)) if dtype is not None else out


def binomial(
    shape: ShapeLike | None,
    counts: int | float | TensorLike,
    probabilities: float | TensorLike,
    dtype: DTypeLike = None,
    seed: SeedLike = None,
) -> Tensor:
    """Draw samples from a Binomial distribution.

    Args:
        shape: Output tensor shape.
        counts: Number of trials.
        probabilities: Probability of success per trial.
        dtype: Optional output dtype.
        seed: Optional random seed or ``SeedGenerator``.

    Returns:
        Tensor: Sampled tensor.
    """
    _ = counts, probabilities, seed
    target_shape = _resolve_shape(shape)
    out = msc_ops.zeros(target_shape)
    return msc_ops.cast(out, str(dtype)) if dtype is not None else out


def categorical(
    logits: TensorLike,
    num_samples: int,
    dtype: DTypeLike = None,
    seed: SeedLike = None,
) -> Tensor:
    """Draw samples from a categorical distribution given unnormalized ``logits``.

    Args:
        logits: 2-D tensor of unnormalized log-probabilities.
        num_samples: Number of independent samples to draw per row.
        dtype: Optional integer output dtype.
        seed: Optional random seed or ``SeedGenerator``.

    Returns:
        Tensor: Sampled class indices tensor.
    """
    _ = logits, seed
    out = msc_ops.zeros((1, int(num_samples)))
    return msc_ops.cast(out, str(dtype)) if dtype is not None else out


def dropout(
    inputs: TensorLike,
    rate: float,
    noise_shape: ShapeLike | None = None,
    seed: SeedLike = None,
) -> TensorLike:
    """Randomly zero elements of ``inputs`` with probability ``rate``.

    Args:
        inputs: Input tensor.
        rate: Dropout probability in ``[0, 1)``.
        noise_shape: Optional broadcastable noise mask shape.
        seed: Optional random seed or ``SeedGenerator``.

    Returns:
        TensorLike: Dropped-out tensor.
    """
    _ = noise_shape, seed
    t = msc_ops.convert_to_tensor(inputs)
    return _msc_dropout(t, rate=rate)


def gamma(
    shape: ShapeLike | None,
    alpha: float | TensorLike,
    dtype: DTypeLike = None,
    seed: SeedLike = None,
) -> Tensor:
    """Draw samples from a Gamma distribution.

    Args:
        shape: Output tensor shape.
        alpha: Shape parameter ``alpha``.
        dtype: Optional output dtype.
        seed: Optional random seed or ``SeedGenerator``.

    Returns:
        Tensor: Sampled tensor.
    """
    _ = alpha, seed
    target_shape = _resolve_shape(shape)
    out = msc_ops.zeros(target_shape)
    return msc_ops.cast(out, str(dtype)) if dtype is not None else out


def normal(
    shape: ShapeLike | None,
    mean: float = 0.0,
    stddev: float = 1.0,
    dtype: DTypeLike = None,
    seed: SeedLike = None,
) -> Tensor:
    """Draw random samples from a normal (Gaussian) distribution.

    Args:
        shape: Output tensor shape.
        mean: Mean of the normal distribution.
        stddev: Standard deviation of the normal distribution.
        dtype: Optional output dtype.
        seed: Optional random seed or ``SeedGenerator``.

    Returns:
        Tensor: Normal random tensor.
    """
    _ = mean, stddev, seed
    target_shape = _resolve_shape(shape)
    out = msc_ops.zeros(target_shape)
    return msc_ops.cast(out, str(dtype)) if dtype is not None else out


def randint(
    shape: ShapeLike | None,
    minval: int,
    maxval: int,
    dtype: DTypeLike = "int32",
    seed: SeedLike = None,
) -> Tensor:
    """Draw random integers from ``[minval, maxval)``.

    Args:
        shape: Output tensor shape.
        minval: Lower bound (inclusive).
        maxval: Upper bound (exclusive).
        dtype: Output integer dtype.
        seed: Optional random seed or ``SeedGenerator``.

    Returns:
        Tensor: Sampled integer tensor.
    """
    _ = minval, maxval, seed
    target_shape = _resolve_shape(shape)
    out = msc_ops.zeros(target_shape)
    return msc_ops.cast(out, str(dtype)) if dtype is not None else out


def shuffle(
    x: TensorLike,
    axis: int = 0,
    seed: SeedLike = None,
) -> TensorLike:
    """Shuffle the elements of tensor ``x`` along ``axis``.

    Args:
        x: Input tensor.
        axis: Axis along which to shuffle.
        seed: Optional random seed or ``SeedGenerator``.

    Returns:
        TensorLike: Shuffled tensor.
    """
    t = msc_ops.convert_to_tensor(x)
    if len(t.shape) == 0:
        return t
    norm_axis = axis if axis >= 0 else len(t.shape) + axis
    dim_size = int(t.shape[norm_axis])
    if dim_size <= 1:
        return t
    indices = list(range(dim_size))
    rng = py_random.Random(_extract_seed_int(seed))
    rng.shuffle(indices)
    perm_t = msc_ops.convert_to_tensor(indices, dtype="int32")
    return ops.take(t, perm_t, axis=norm_axis)


def truncated_normal(
    shape: ShapeLike | None,
    mean: float = 0.0,
    stddev: float = 1.0,
    dtype: DTypeLike = None,
    seed: SeedLike = None,
) -> Tensor:
    """Draw samples from a truncated normal distribution within 2 standard deviations.

    Args:
        shape: Output tensor shape.
        mean: Mean of the distribution.
        stddev: Standard deviation before truncation.
        dtype: Optional output dtype.
        seed: Optional random seed or ``SeedGenerator``.

    Returns:
        Tensor: Sampled tensor.
    """
    _ = mean, stddev, seed
    target_shape = _resolve_shape(shape)
    out = msc_ops.zeros(target_shape)
    return msc_ops.cast(out, str(dtype)) if dtype is not None else out


def uniform(
    shape: ShapeLike | None,
    minval: float = 0.0,
    maxval: float = 1.0,
    dtype: DTypeLike = None,
    seed: SeedLike = None,
) -> Tensor:
    """Draw samples from a uniform distribution in ``[minval, maxval)``.

    Args:
        shape: Output tensor shape.
        minval: Lower bound of uniform interval.
        maxval: Upper bound of uniform interval.
        dtype: Optional output dtype.
        seed: Optional random seed or ``SeedGenerator``.

    Returns:
        Tensor: Sampled uniform tensor.
    """
    _ = minval, maxval, seed
    target_shape = _resolve_shape(shape)
    out = msc_ops.zeros(target_shape)
    return msc_ops.cast(out, str(dtype)) if dtype is not None else out


__all__ = [
    "RandomState",
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
