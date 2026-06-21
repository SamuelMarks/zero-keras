"""Keras initializers module."""

import math
from typing import Any, Optional
from .activations import _to_tensor

import ml_switcheroo_compiler.ops as ops
import ml_switcheroo_compiler.random as random

__all__ = [
    "Initializer",
    "Constant",
    "Zeros",
    "Ones",
    "Identity",
    "IdentityInitializer",
    "Orthogonal",
    "OrthogonalInitializer",
    "RandomNormal",
    "RandomUniform",
    "TruncatedNormal",
    "VarianceScaling",
    "GlorotNormal",
    "GlorotUniform",
    "HeNormal",
    "HeUniform",
    "LecunNormal",
    "LecunUniform",
    "STFT",
    "STFTInitializer",
    "constant",
    "zeros",
    "ones",
    "identity",
    "orthogonal",
    "random_normal",
    "random_uniform",
    "truncated_normal",
    "variance_scaling",
    "glorot_normal",
    "glorot_uniform",
    "he_normal",
    "he_uniform",
    "lecun_normal",
    "lecun_uniform",
    "stft",
]


def _get_dtype(dtype: Any) -> Any:
    """_get_dtype function.

    Args:
    dtype: Parameter dtype.

    Returns:
    Any: Return value.

    """
    from ml_switcheroo_compiler.core.dtype import DType

    if isinstance(dtype, DType):
        return dtype
    if dtype is None:
        return DType.Float32
    if isinstance(dtype, str):
        for d in DType:
            if d.value == dtype:
                return d
    return DType.Float32


def _wrap(x):
    """Docstring."""
    from zero_keras.core_layers import KerasTensor

    if hasattr(x, "data") and hasattr(x.data, "id"):
        return KerasTensor(x.shape, x.dtype, data=x)
    return x.data if hasattr(x, "data") else x


class Initializer:
    """Initializer base class: all Keras initializers inherit from this class.

    Initializers should implement a `__call__()` method with the following
    signature:

    ```python
    def __call__(self, shape, dtype=None, **kwargs):
        # returns a tensor of shape `shape` and dtype `dtype`
        # containing values drawn from a distribution of your choice.
    ```

    Optionally, you can also implement the method `get_config()` and the class
    method `from_config` in order to support serialization, just like with
    any Keras object.

    Here's a simple example: a random normal initializer.

    ```python
    class ExampleRandomNormal(Initializer):
        def __init__(self, mean, stddev):
            self.mean = mean
            self.stddev = stddev

        def __call__(self, shape, dtype=None, **kwargs):
            return keras.random.normal(
                shape, mean=self.mean, stddev=self.stddev, dtype=dtype
            )

        def get_config(self):  # To support serialization
            return {"mean": self.mean, "stddev": self.stddev}
    ```

    Note that we don't have to implement `from_config()` in the example above
    since the constructor arguments of the class the keys in the config returned
    by `get_config()` are the same. In this case, the default `from_config()`
    works fine.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """Returns a tensor object initialized as specified by the initializer.

        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor.

        """
        dt = _get_dtype(dtype)
        return _wrap(ops.zeros(shape, dtype=dt))


class Constant(Initializer):
    """Initializer that generates tensors with constant values.

    Only scalar values are allowed.
    The constant value provided must be convertible to the dtype requested
    when calling the initializer.

    Examples:
    >>> # Standalone usage:
    >>> initializer = Constant(10.)
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = Constant(10.)
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        value: A Python scalar.

    """

    def __init__(self, value: float = 0.0):
        self.value = value

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """Returns a tensor object initialized as specified by the initializer.

        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor.

        """
        dt = _get_dtype(dtype)
        return _wrap(ops.full(shape, self.value, dtype=dt))


class Zeros(Initializer):
    """Initializer that generates tensors initialized to 0.

    Examples:
    >>> # Standalone usage:
    >>> initializer = Zeros()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = Zeros()
    >>> layer = Dense(units=3, kernel_initializer=initializer)

    """

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """Returns a tensor object initialized as specified by the initializer.

        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. Only numeric or boolean dtypes
                are supported. If not specified, `keras.backend.floatx()`
                is used, which default to `float32` unless you configured it
                otherwise (via `keras.backend.set_floatx(float_dtype)`).

        """
        dt = _get_dtype(dtype)
        return _wrap(ops.zeros(shape, dtype=dt))


class Ones(Initializer):
    """Initializer that generates tensors initialized to 1.

    Also available via the shortcut function `ones`.

    Examples:
    >>> # Standalone usage:
    >>> initializer = Ones()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = Ones()
    >>> layer = Dense(3, kernel_initializer=initializer)

    """

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """Returns a tensor object initialized as specified by the initializer.

        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. Only numeric or boolean dtypes
                are supported. If not specified, `keras.backend.floatx()`
                is used, which default to `float32` unless you configured it
                otherwise (via `keras.backend.set_floatx(float_dtype)`).

        """
        dt = _get_dtype(dtype)
        return _wrap(ops.ones(shape, dtype=dt))


class Identity(Initializer):
    """Initializer that generates the identity matrix.

    Only usable for generating 2D matrices.

    Examples:
    >>> # Standalone usage:
    >>> initializer = Identity()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = Identity()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        gain: Multiplicative factor to apply to the identity matrix.

    """

    def __init__(self, gain: float = 1.0):
        self.gain = gain

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """Returns a tensor object initialized as specified by the initializer.

        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. Only numeric or boolean dtypes
                are supported. If not specified, `keras.backend.floatx()`
                is used, which default to `float32` unless you configured it
                otherwise (via `keras.backend.set_floatx(float_dtype)`).

        """
        if len(shape) != 2:
            raise ValueError(
                "Identity matrix initializer can only be used for 2D matrices."
            )
        dt = _get_dtype(dtype)
        return _wrap(
            ops.multiply(ops.eye(shape[0], shape[1], dtype=dt), _to_tensor(self.gain))
        )


class Orthogonal(Initializer):
    """Initializer that generates an orthogonal matrix.

    If the shape of the tensor to initialize is two-dimensional, it is
    initialized with an orthogonal matrix obtained from the QR decomposition of
    a matrix of random numbers drawn from a normal distribution. If the matrix
    has fewer rows than columns then the output will have orthogonal rows.
    Otherwise, the output will have orthogonal columns.

    If the shape of the tensor to initialize is more than two-dimensional,
    a matrix of shape `(shape[0] * ... * shape[n - 2], shape[n - 1])`
    is initialized, where `n` is the length of the shape vector.
    The matrix is subsequently reshaped to give a tensor of the desired shape.

    Examples:
    >>> # Standalone usage:
    >>> initializer = keras.initializers.Orthogonal()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = keras.initializers.Orthogonal()
    >>> layer = keras.layers.Dense(3, kernel_initializer=initializer)

    Args:
        gain: Multiplicative factor to apply to the orthogonal matrix.
        seed: A Python integer. Used to make the behavior of the initializer
            deterministic.

    Reference:

    - [Saxe et al., 2014](https://openreview.net/forum?id=_wzZwKpTDF_9C)

    """

    def __init__(self, gain: float = 1.0, seed: Optional[int] = None):
        self.gain = gain
        self.seed = seed

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """Returns a tensor object initialized as specified by the initializer.

        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor.

        """
        if len(shape) < 2:
            raise ValueError(
                "The tensor to initialize must be at least two-dimensional"
            )
        dt = _get_dtype(dtype)
        num_rows = 1
        for d in shape[:-1]:
            num_rows *= d
        num_cols = shape[-1]

        flat_shape = (max(num_rows, num_cols), min(num_rows, num_cols))
        key = random.PRNGKey(self.seed if self.seed is not None else 0)
        a = random.normal(key, flat_shape, dtype=dt)
        q, r = ops.linalg.qr(a)

        d = ops.diag(r)
        q = ops.multiply(q, ops.sign(d))

        if num_rows < num_cols:
            q = ops.transpose(q, 0, 1)

        q = ops.reshape(q, shape)
        return _wrap(ops.multiply(q, _to_tensor(self.gain)))


class RandomNormal(Initializer):
    """Random normal initializer.

    Draws samples from a normal distribution for given parameters.

    Examples:
    >>> # Standalone usage:
    >>> initializer = RandomNormal(mean=0.0, stddev=1.0)
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = RandomNormal(mean=0.0, stddev=1.0)
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        mean: A python scalar or a scalar keras tensor. Mean of the random
            values to generate.
        stddev: A python scalar or a scalar keras tensor. Standard deviation of
           the random values to generate.
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    """

    def __init__(
        self, mean: float = 0.0, stddev: float = 0.05, seed: Optional[int] = None
    ):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """Returns a tensor object initialized as specified by the initializer.

        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor.

        """
        dt = _get_dtype(dtype)
        key = random.PRNGKey(self.seed if self.seed is not None else 0)
        res = random.normal(key, shape, dtype=dt)
        return _wrap(
            ops.add(ops.multiply(res, _to_tensor(self.stddev)), _to_tensor(self.mean))
        )


class RandomUniform(Initializer):
    """Random uniform initializer.

    Draws samples from a uniform distribution for given parameters.

    Examples:
    >>> # Standalone usage:
    >>> initializer = RandomUniform(minval=0.0, maxval=1.0)
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = RandomUniform(minval=0.0, maxval=1.0)
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        minval: A python scalar or a scalar keras tensor. Lower bound of the
            range of random values to generate (inclusive).
        maxval: A python scalar or a scalar keras tensor. Upper bound of the
            range of random values to generate (exclusive).
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    """

    def __init__(
        self, minval: float = -0.05, maxval: float = 0.05, seed: Optional[int] = None
    ):
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """Returns a tensor object initialized as specified by the initializer.

        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor.

        """
        dt = _get_dtype(dtype)
        key = random.PRNGKey(self.seed if self.seed is not None else 0)
        res = random.uniform(
            key, shape, dtype=dt, minval=self.minval, maxval=self.maxval
        )
        return _wrap(res)


class TruncatedNormal(Initializer):
    """Initializer that generates a truncated normal distribution.

    The values generated are similar to values from a
    `RandomNormal` initializer, except that values more
    than two standard deviations from the mean are
    discarded and re-drawn.

    Examples:
    >>> # Standalone usage:
    >>> initializer = TruncatedNormal(mean=0., stddev=1.)
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = TruncatedNormal(mean=0., stddev=1.)
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        mean: A python scalar or a scalar keras tensor. Mean of the random
            values to generate.
        stddev: A python scalar or a scalar keras tensor. Standard deviation of
           the random values to generate.
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    """

    def __init__(
        self, mean: float = 0.0, stddev: float = 0.05, seed: Optional[int] = None
    ):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """Returns a tensor object initialized as specified by the initializer.

        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor.

        """
        dt = _get_dtype(dtype)
        key = random.PRNGKey(self.seed if self.seed is not None else 0)
        res = random.truncated_normal(key, -2.0, 2.0, shape, dtype=dt)
        return _wrap(
            ops.add(ops.multiply(res, _to_tensor(self.stddev)), _to_tensor(self.mean))
        )


class VarianceScaling(Initializer):
    """Initializer that adapts its scale to the shape of its input tensors.

    With `distribution="truncated_normal" or "untruncated_normal"`, samples are
    drawn from a truncated/untruncated normal distribution with a mean of zero
    and a standard deviation (after truncation, if used) `stddev = sqrt(scale /
    n)`, where `n` is:

    - number of input units in the weight tensor, if `mode="fan_in"`
    - number of output units, if `mode="fan_out"`
    - average of the numbers of input and output units, if `mode="fan_avg"`

    With `distribution="uniform"`, samples are drawn from a uniform distribution
    within `[-limit, limit]`, where `limit = sqrt(3 * scale / n)`.

    Examples:
    >>> # Standalone usage:
    >>> initializer = VarianceScaling(
        scale=0.1, mode='fan_in', distribution='uniform')
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = VarianceScaling(
        scale=0.1, mode='fan_in', distribution='uniform')
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        scale: Scaling factor (positive float).
        mode: One of `"fan_in"`, `"fan_out"`, `"fan_avg"`.
        distribution: Random distribution to use.
            One of `"truncated_normal"`, `"untruncated_normal"`, or `"uniform"`.
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    """

    def __init__(
        self,
        scale: float = 1.0,
        mode: str = "fan_in",
        distribution: str = "truncated_normal",
        seed: Optional[int] = None,
    ):
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """Returns a tensor object initialized as specified by the initializer.

        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor.

        """
        dt = _get_dtype(dtype)
        receptive_field_size = 1
        for dim in shape[:-2]:
            receptive_field_size *= dim

        if len(shape) < 2:
            fan_in = fan_out = shape[0] if len(shape) == 1 else 1
        else:
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size

        if self.mode == "fan_in":
            scale = self.scale / max(1.0, fan_in)
        elif self.mode == "fan_out":
            scale = self.scale / max(1.0, fan_out)
        else:
            scale = self.scale / max(1.0, (fan_in + fan_out) / 2.0)

        key = random.PRNGKey(self.seed if self.seed is not None else 0)

        if self.distribution == "truncated_normal":
            stddev = math.sqrt(scale) / 0.87962566103423978
            res = random.truncated_normal(key, -2.0, 2.0, shape, dtype=dt)
            res = ops.multiply(res, _to_tensor(stddev))
        elif self.distribution == "untruncated_normal":
            stddev = math.sqrt(scale)
            res = random.normal(key, shape, dtype=dt)
            res = ops.multiply(res, _to_tensor(stddev))
        else:
            limit = math.sqrt(3.0 * scale)
            res = random.uniform(key, shape, dtype=dt, minval=-limit, maxval=limit)

        return _wrap(res)


class GlorotNormal(VarianceScaling):
    """The Glorot normal initializer, also called Xavier normal initializer.

    Draws samples from a truncated normal distribution centered on 0 with
    `stddev = sqrt(2 / (fan_in + fan_out))` where `fan_in` is the number of
    input units in the weight tensor and `fan_out` is the number of output units
    in the weight tensor.

    Examples:
    >>> # Standalone usage:
    >>> initializer = GlorotNormal()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = GlorotNormal()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    Reference:

    - [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)

    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(
            scale=1.0, mode="fan_avg", distribution="truncated_normal", seed=seed
        )
        self.seed = seed


class GlorotUniform(VarianceScaling):
    """The Glorot uniform initializer, also called Xavier uniform initializer.

    Draws samples from a uniform distribution within `[-limit, limit]`, where
    `limit = sqrt(6 / (fan_in + fan_out))` (`fan_in` is the number of input
    units in the weight tensor and `fan_out` is the number of output units).

    Examples:
    >>> # Standalone usage:
    >>> initializer = GlorotUniform()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = GlorotUniform()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    Reference:

    - [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)

    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(scale=1.0, mode="fan_avg", distribution="uniform", seed=seed)
        self.seed = seed


class HeNormal(VarianceScaling):
    """He normal initializer.

    It draws samples from a truncated normal distribution centered on 0 with
    `stddev = sqrt(2 / fan_in)` where `fan_in` is the number of input units in
    the weight tensor.

    Examples:
    >>> # Standalone usage:
    >>> initializer = HeNormal()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = HeNormal()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    Reference:

    - [He et al., 2015](https://arxiv.org/abs/1502.01852)

    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(
            scale=2.0, mode="fan_in", distribution="truncated_normal", seed=seed
        )
        self.seed = seed


class HeUniform(VarianceScaling):
    """He uniform variance scaling initializer.

    Draws samples from a uniform distribution within `[-limit, limit]`, where
    `limit = sqrt(6 / fan_in)` (`fan_in` is the number of input units in the
    weight tensor).

    Examples:
    >>> # Standalone usage:
    >>> initializer = HeUniform()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = HeUniform()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    Reference:

    - [He et al., 2015](https://arxiv.org/abs/1502.01852)

    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(scale=2.0, mode="fan_in", distribution="uniform", seed=seed)
        self.seed = seed


class LecunNormal(VarianceScaling):
    """Lecun normal initializer.

    Initializers allow you to pre-specify an initialization strategy, encoded in
    the Initializer object, without knowing the shape and dtype of the variable
    being initialized.

    Draws samples from a truncated normal distribution centered on 0 with
    `stddev = sqrt(1 / fan_in)` where `fan_in` is the number of input units in
    the weight tensor.

    Examples:
    >>> # Standalone usage:
    >>> initializer = LecunNormal()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = LecunNormal()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    Reference:

    - [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)

    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(
            scale=1.0, mode="fan_in", distribution="truncated_normal", seed=seed
        )
        self.seed = seed


class LecunUniform(VarianceScaling):
    """Lecun uniform initializer.

    Draws samples from a uniform distribution within `[-limit, limit]`, where
    `limit = sqrt(3 / fan_in)` (`fan_in` is the number of input units in the
    weight tensor).

    Examples:
    >>> # Standalone usage:
    >>> initializer = LecunUniform()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = LecunUniform()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    Reference:

    - [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)

    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(scale=1.0, mode="fan_in", distribution="uniform", seed=seed)
        self.seed = seed


class STFT(Initializer):
    """Initializer of Conv kernels for Short-term Fourier Transformation (STFT).

    Since the formula involves complex numbers, this class compute either the
    real or the imaginary components of the final output.

    Additionally, this initializer supports windowing functions across the time
    dimension as commonly used in STFT. Windowing functions from the module
    `scipy.signal.windows` are supported, including the common `hann` and
    `hamming` windowing functions. This layer supports periodic windows and
    scaling-based normalization.

    This is primarily intended for use in the `STFTSpectrogram` layer.

    Examples:
    >>> # Standalone usage:
    >>> initializer = STFTInitializer("real", "hann", "density", False)
    >>> values = initializer(shape=(128, 1, 513))

    Args:
        side: String, `"real"` or `"imag"` deciding if the kernel will compute
            the real side or the imaginary side of the output. Defaults to
            `"real"`.
        window: String for the name of the windowing function in the
            `scipy.signal.windows` module, or array_like for the window values,
            or `None` for no windowing.
        scaling: String, `"density"` or `"spectrum"` for scaling of the window
            for normalization, either L2 or L1 normalization.
            `None` for no scaling.
        periodic: Boolean, if True, the window function will be treated as
            periodic. Defaults to `False`.

    """

    def __init__(
        self,
        side: str = "real",
        window: str = "hann",
        scaling: str = "density",
        periodic: bool = False,
    ):
        pass


class IdentityInitializer(Identity):
    """Initializer that generates the identity matrix.

    Only usable for generating 2D matrices.

    Examples:
    >>> # Standalone usage:
    >>> initializer = Identity()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = Identity()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        gain: Multiplicative factor to apply to the identity matrix.

    """

    pass


class OrthogonalInitializer(Orthogonal):
    """Initializer that generates an orthogonal matrix.

    If the shape of the tensor to initialize is two-dimensional, it is
    initialized with an orthogonal matrix obtained from the QR decomposition of
    a matrix of random numbers drawn from a normal distribution. If the matrix
    has fewer rows than columns then the output will have orthogonal rows.
    Otherwise, the output will have orthogonal columns.

    If the shape of the tensor to initialize is more than two-dimensional,
    a matrix of shape `(shape[0] * ... * shape[n - 2], shape[n - 1])`
    is initialized, where `n` is the length of the shape vector.
    The matrix is subsequently reshaped to give a tensor of the desired shape.

    Examples:
    >>> # Standalone usage:
    >>> initializer = keras.initializers.Orthogonal()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = keras.initializers.Orthogonal()
    >>> layer = keras.layers.Dense(3, kernel_initializer=initializer)

    Args:
        gain: Multiplicative factor to apply to the orthogonal matrix.
        seed: A Python integer. Used to make the behavior of the initializer
            deterministic.

    Reference:

    - [Saxe et al., 2014](https://openreview.net/forum?id=_wzZwKpTDF_9C)

    """

    pass


class STFTInitializer(STFT):
    """Initializer of Conv kernels for Short-term Fourier Transformation (STFT).

    Since the formula involves complex numbers, this class compute either the
    real or the imaginary components of the final output.

    Additionally, this initializer supports windowing functions across the time
    dimension as commonly used in STFT. Windowing functions from the module
    `scipy.signal.windows` are supported, including the common `hann` and
    `hamming` windowing functions. This layer supports periodic windows and
    scaling-based normalization.

    This is primarily intended for use in the `STFTSpectrogram` layer.

    Examples:
    >>> # Standalone usage:
    >>> initializer = STFTInitializer("real", "hann", "density", False)
    >>> values = initializer(shape=(128, 1, 513))

    Args:
        side: String, `"real"` or `"imag"` deciding if the kernel will compute
            the real side or the imaginary side of the output. Defaults to
            `"real"`.
        window: String for the name of the windowing function in the
            `scipy.signal.windows` module, or array_like for the window values,
            or `None` for no windowing.
        scaling: String, `"density"` or `"spectrum"` for scaling of the window
            for normalization, either L2 or L1 normalization.
            `None` for no scaling.
        periodic: Boolean, if True, the window function will be treated as
            periodic. Defaults to `False`.

    """

    pass


class constant(Constant):
    """Initializer that generates tensors with constant values.

    Only scalar values are allowed.
    The constant value provided must be convertible to the dtype requested
    when calling the initializer.

    Examples:
    >>> # Standalone usage:
    >>> initializer = Constant(10.)
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = Constant(10.)
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        value: A Python scalar.

    """

    pass


class zeros(Zeros):
    """Initializer that generates tensors initialized to 0.

    Examples:
    >>> # Standalone usage:
    >>> initializer = Zeros()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = Zeros()
    >>> layer = Dense(units=3, kernel_initializer=initializer)

    """

    pass


class ones(Ones):
    """Initializer that generates tensors initialized to 1.

    Also available via the shortcut function `ones`.

    Examples:
    >>> # Standalone usage:
    >>> initializer = Ones()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = Ones()
    >>> layer = Dense(3, kernel_initializer=initializer)

    """

    pass


class identity(Identity):
    """Initializer that generates the identity matrix.

    Only usable for generating 2D matrices.

    Examples:
    >>> # Standalone usage:
    >>> initializer = Identity()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = Identity()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        gain: Multiplicative factor to apply to the identity matrix.

    """

    pass


class orthogonal(Orthogonal):
    """Initializer that generates an orthogonal matrix.

    If the shape of the tensor to initialize is two-dimensional, it is
    initialized with an orthogonal matrix obtained from the QR decomposition of
    a matrix of random numbers drawn from a normal distribution. If the matrix
    has fewer rows than columns then the output will have orthogonal rows.
    Otherwise, the output will have orthogonal columns.

    If the shape of the tensor to initialize is more than two-dimensional,
    a matrix of shape `(shape[0] * ... * shape[n - 2], shape[n - 1])`
    is initialized, where `n` is the length of the shape vector.
    The matrix is subsequently reshaped to give a tensor of the desired shape.

    Examples:
    >>> # Standalone usage:
    >>> initializer = keras.initializers.Orthogonal()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = keras.initializers.Orthogonal()
    >>> layer = keras.layers.Dense(3, kernel_initializer=initializer)

    Args:
        gain: Multiplicative factor to apply to the orthogonal matrix.
        seed: A Python integer. Used to make the behavior of the initializer
            deterministic.

    Reference:

    - [Saxe et al., 2014](https://openreview.net/forum?id=_wzZwKpTDF_9C)

    """

    pass


class random_normal(RandomNormal):
    """Random normal initializer.

    Draws samples from a normal distribution for given parameters.

    Examples:
    >>> # Standalone usage:
    >>> initializer = RandomNormal(mean=0.0, stddev=1.0)
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = RandomNormal(mean=0.0, stddev=1.0)
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        mean: A python scalar or a scalar keras tensor. Mean of the random
            values to generate.
        stddev: A python scalar or a scalar keras tensor. Standard deviation of
           the random values to generate.
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    """

    pass


class random_uniform(RandomUniform):
    """Random uniform initializer.

    Draws samples from a uniform distribution for given parameters.

    Examples:
    >>> # Standalone usage:
    >>> initializer = RandomUniform(minval=0.0, maxval=1.0)
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = RandomUniform(minval=0.0, maxval=1.0)
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        minval: A python scalar or a scalar keras tensor. Lower bound of the
            range of random values to generate (inclusive).
        maxval: A python scalar or a scalar keras tensor. Upper bound of the
            range of random values to generate (exclusive).
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    """

    pass


class truncated_normal(TruncatedNormal):
    """Initializer that generates a truncated normal distribution.

    The values generated are similar to values from a
    `RandomNormal` initializer, except that values more
    than two standard deviations from the mean are
    discarded and re-drawn.

    Examples:
    >>> # Standalone usage:
    >>> initializer = TruncatedNormal(mean=0., stddev=1.)
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = TruncatedNormal(mean=0., stddev=1.)
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        mean: A python scalar or a scalar keras tensor. Mean of the random
            values to generate.
        stddev: A python scalar or a scalar keras tensor. Standard deviation of
           the random values to generate.
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    """

    pass


class variance_scaling(VarianceScaling):
    """Initializer that adapts its scale to the shape of its input tensors.

    With `distribution="truncated_normal" or "untruncated_normal"`, samples are
    drawn from a truncated/untruncated normal distribution with a mean of zero
    and a standard deviation (after truncation, if used) `stddev = sqrt(scale /
    n)`, where `n` is:

    - number of input units in the weight tensor, if `mode="fan_in"`
    - number of output units, if `mode="fan_out"`
    - average of the numbers of input and output units, if `mode="fan_avg"`

    With `distribution="uniform"`, samples are drawn from a uniform distribution
    within `[-limit, limit]`, where `limit = sqrt(3 * scale / n)`.

    Examples:
    >>> # Standalone usage:
    >>> initializer = VarianceScaling(
        scale=0.1, mode='fan_in', distribution='uniform')
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = VarianceScaling(
        scale=0.1, mode='fan_in', distribution='uniform')
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        scale: Scaling factor (positive float).
        mode: One of `"fan_in"`, `"fan_out"`, `"fan_avg"`.
        distribution: Random distribution to use.
            One of `"truncated_normal"`, `"untruncated_normal"`, or `"uniform"`.
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    """

    pass


class glorot_normal(GlorotNormal):
    """The Glorot normal initializer, also called Xavier normal initializer.

    Draws samples from a truncated normal distribution centered on 0 with
    `stddev = sqrt(2 / (fan_in + fan_out))` where `fan_in` is the number of
    input units in the weight tensor and `fan_out` is the number of output units
    in the weight tensor.

    Examples:
    >>> # Standalone usage:
    >>> initializer = GlorotNormal()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = GlorotNormal()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    Reference:

    - [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)

    """

    pass


class glorot_uniform(GlorotUniform):
    """The Glorot uniform initializer, also called Xavier uniform initializer.

    Draws samples from a uniform distribution within `[-limit, limit]`, where
    `limit = sqrt(6 / (fan_in + fan_out))` (`fan_in` is the number of input
    units in the weight tensor and `fan_out` is the number of output units).

    Examples:
    >>> # Standalone usage:
    >>> initializer = GlorotUniform()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = GlorotUniform()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    Reference:

    - [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)

    """

    pass


class he_normal(HeNormal):
    """He normal initializer.

    It draws samples from a truncated normal distribution centered on 0 with
    `stddev = sqrt(2 / fan_in)` where `fan_in` is the number of input units in
    the weight tensor.

    Examples:
    >>> # Standalone usage:
    >>> initializer = HeNormal()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = HeNormal()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    Reference:

    - [He et al., 2015](https://arxiv.org/abs/1502.01852)

    """

    pass


class he_uniform(HeUniform):
    """He uniform variance scaling initializer.

    Draws samples from a uniform distribution within `[-limit, limit]`, where
    `limit = sqrt(6 / fan_in)` (`fan_in` is the number of input units in the
    weight tensor).

    Examples:
    >>> # Standalone usage:
    >>> initializer = HeUniform()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = HeUniform()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    Reference:

    - [He et al., 2015](https://arxiv.org/abs/1502.01852)

    """

    pass


class lecun_normal(LecunNormal):
    """Lecun normal initializer.

    Initializers allow you to pre-specify an initialization strategy, encoded in
    the Initializer object, without knowing the shape and dtype of the variable
    being initialized.

    Draws samples from a truncated normal distribution centered on 0 with
    `stddev = sqrt(1 / fan_in)` where `fan_in` is the number of input units in
    the weight tensor.

    Examples:
    >>> # Standalone usage:
    >>> initializer = LecunNormal()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = LecunNormal()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    Reference:

    - [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)

    """

    pass


class lecun_uniform(LecunUniform):
    """Lecun uniform initializer.

    Draws samples from a uniform distribution within `[-limit, limit]`, where
    `limit = sqrt(3 / fan_in)` (`fan_in` is the number of input units in the
    weight tensor).

    Examples:
    >>> # Standalone usage:
    >>> initializer = LecunUniform()
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = LecunUniform()
    >>> layer = Dense(3, kernel_initializer=initializer)

    Args:
        seed: A Python integer or instance of
            `keras.backend.SeedGenerator`.
            Used to make the behavior of the initializer
            deterministic. Note that an initializer seeded with an integer
            or `None` (unseeded) will produce the same random values
            across multiple calls. To get different random values
            across multiple calls, use as seed an instance
            of `keras.backend.SeedGenerator`.

    Reference:

    - [Klambauer et al., 2017](https://arxiv.org/abs/1706.02515)

    """

    pass


class stft(STFT):
    """Initializer of Conv kernels for Short-term Fourier Transformation (STFT).

    Since the formula involves complex numbers, this class compute either the
    real or the imaginary components of the final output.

    Additionally, this initializer supports windowing functions across the time
    dimension as commonly used in STFT. Windowing functions from the module
    `scipy.signal.windows` are supported, including the common `hann` and
    `hamming` windowing functions. This layer supports periodic windows and
    scaling-based normalization.

    This is primarily intended for use in the `STFTSpectrogram` layer.

    Examples:
    >>> # Standalone usage:
    >>> initializer = STFTInitializer("real", "hann", "density", False)
    >>> values = initializer(shape=(128, 1, 513))

    Args:
        side: String, `"real"` or `"imag"` deciding if the kernel will compute
            the real side or the imaginary side of the output. Defaults to
            `"real"`.
        window: String for the name of the windowing function in the
            `scipy.signal.windows` module, or array_like for the window values,
            or `None` for no windowing.
        scaling: String, `"density"` or `"spectrum"` for scaling of the window
            for normalization, either L2 or L1 normalization.
            `None` for no scaling.
        periodic: Boolean, if True, the window function will be treated as
            periodic. Defaults to `False`.

    """

    pass


def get(identifier):
    """Retrieves a Keras initializer object via an identifier.

    The `identifier` may be the string name of a initializers function or class
    (case-sensitively).

    >>> identifier = 'Ones'
    >>> keras.initializers.get(identifier)
    <...keras.initializers.initializers.Ones...>

    You can also specify `config` of the initializer to this function by passing
    dict containing `class_name` and `config` as an identifier. Also note that
    the `class_name` must map to a `Initializer` class.

    >>> cfg = {'class_name': 'Ones', 'config': {}}
    >>> keras.initializers.get(cfg)
    <...keras.initializers.initializers.Ones...>

    In the case that the `identifier` is a class, this method will return a new
    instance of the class by its constructor.

    You may also pass a callable function with a signature that includes `shape`
    and `dtype=None` as an identifier.

    >>> fn = lambda shape, dtype=None: ops.ones(shape, dtype)
    >>> keras.initializers.get(fn)
    <function <lambda> at ...>

    Alternatively, you can pass a backend tensor or numpy array as the
    `identifier` to define the initializer values directly. Note that when
    calling the initializer, the specified `shape` argument must be the same as
    the shape of the tensor.

    >>> tensor = ops.ones(shape=(5, 5))
    >>> keras.initializers.get(tensor)
    <function get.<locals>.initialize_fn at ...>

    Args:
        identifier: A string, dict, callable function, or tensor specifying
            the initializer. If a string, it should be the name of an
            initializer. If a dict, it should contain the configuration of an
            initializer. Callable functions or predefined tensors are also
            accepted.

    Returns:
        Initializer instance base on the input identifier.

    """
    if identifier is None:
        return GlorotUniform()
    if isinstance(identifier, str):
        mapping = {
            "glorot_uniform": GlorotUniform(),
            "glorot_normal": GlorotNormal(),
            "zeros": Zeros(),
            "ones": Ones(),
            "random_normal": RandomNormal(),
            "random_uniform": RandomUniform(),
            "truncated_normal": TruncatedNormal(),
            "orthogonal": Orthogonal(),
            "identity": Identity(),
            "variance_scaling": VarianceScaling(),
            "he_normal": HeNormal(),
            "he_uniform": HeUniform(),
            "lecun_normal": LecunNormal(),
            "lecun_uniform": LecunUniform(),
            "constant": Constant(),
        }
        return mapping.get(identifier, GlorotUniform())
    return identifier


def serialize(initializer):
    """Serialize an initializer."""
    if initializer is None:
        return None
    if isinstance(initializer, str):
        return initializer
    return {
        "class_name": initializer.__class__.__name__,
        "config": initializer.get_config()
        if hasattr(initializer, "get_config")
        else {},
    }


def deserialize(config, custom_objects=None):
    """Deserialize an initializer."""
    if config is None:
        return None
    if isinstance(config, str):
        return get(config)
    if isinstance(config, dict):
        class_name = config.get("class_name")
        conf = config.get("config", {})
        cls = globals().get(class_name)
        if cls:
            return cls(**conf)
    return config
