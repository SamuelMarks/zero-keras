"""Keras initializers module."""

import math
from typing import Any, Optional
from .activations import _to_tensor

import ml_switcheroo.ops as ops
import ml_switcheroo.random as random

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
    from ml_switcheroo.core.dtype import DType

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
    """docstring."""
    from zero_keras.core_layers import KerasTensor

    if hasattr(x, "data") and hasattr(x.data, "id"):
        return KerasTensor(x.shape, x.dtype, data=x)
    return x.data if hasattr(x, "data") else x


class Initializer:
    """docstring."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """docstring."""
        pass

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """docstring."""
        dt = _get_dtype(dtype)
        return _wrap(ops.zeros(shape, dtype=dt))


class Constant(Initializer):
    """docstring."""

    def __init__(self, value: float = 0.0):
        """docstring."""
        self.value = value

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """docstring."""
        dt = _get_dtype(dtype)
        return _wrap(ops.full(shape, self.value, dtype=dt))


class Zeros(Initializer):
    """docstring."""

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """docstring."""
        dt = _get_dtype(dtype)
        return _wrap(ops.zeros(shape, dtype=dt))


class Ones(Initializer):
    """docstring."""

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """docstring."""
        dt = _get_dtype(dtype)
        return _wrap(ops.ones(shape, dtype=dt))


class Identity(Initializer):
    """docstring."""

    def __init__(self, gain: float = 1.0):
        """docstring."""
        self.gain = gain

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """docstring."""
        if len(shape) != 2:
            raise ValueError(
                "Identity matrix initializer can only be used for 2D matrices."
            )
        dt = _get_dtype(dtype)
        return _wrap(
            ops.multiply(ops.eye(shape[0], shape[1], dtype=dt), _to_tensor(self.gain))
        )


class Orthogonal(Initializer):
    """docstring."""

    def __init__(self, gain: float = 1.0, seed: Optional[int] = None):
        """docstring."""
        self.gain = gain
        self.seed = seed

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """docstring."""
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
    """docstring."""

    def __init__(
        self, mean: float = 0.0, stddev: float = 0.05, seed: Optional[int] = None
    ):
        """docstring."""
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """docstring."""
        dt = _get_dtype(dtype)
        key = random.PRNGKey(self.seed if self.seed is not None else 0)
        res = random.normal(key, shape, dtype=dt)
        return _wrap(
            ops.add(ops.multiply(res, _to_tensor(self.stddev)), _to_tensor(self.mean))
        )


class RandomUniform(Initializer):
    """docstring."""

    def __init__(
        self, minval: float = -0.05, maxval: float = 0.05, seed: Optional[int] = None
    ):
        """docstring."""
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """docstring."""
        dt = _get_dtype(dtype)
        key = random.PRNGKey(self.seed if self.seed is not None else 0)
        res = random.uniform(
            key, shape, dtype=dt, minval=self.minval, maxval=self.maxval
        )
        return _wrap(res)


class TruncatedNormal(Initializer):
    """docstring."""

    def __init__(
        self, mean: float = 0.0, stddev: float = 0.05, seed: Optional[int] = None
    ):
        """docstring."""
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """docstring."""
        dt = _get_dtype(dtype)
        key = random.PRNGKey(self.seed if self.seed is not None else 0)
        res = random.truncated_normal(key, -2.0, 2.0, shape, dtype=dt)
        return _wrap(
            ops.add(ops.multiply(res, _to_tensor(self.stddev)), _to_tensor(self.mean))
        )


class VarianceScaling(Initializer):
    """docstring."""

    def __init__(
        self,
        scale: float = 1.0,
        mode: str = "fan_in",
        distribution: str = "truncated_normal",
        seed: Optional[int] = None,
    ):
        """docstring."""
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """docstring."""
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
    """docstring."""

    def __init__(self, seed: Optional[int] = None):
        """docstring."""
        super().__init__(
            scale=1.0, mode="fan_avg", distribution="truncated_normal", seed=seed
        )
        self.seed = seed


class GlorotUniform(VarianceScaling):
    """docstring."""

    def __init__(self, seed: Optional[int] = None):
        """docstring."""
        super().__init__(scale=1.0, mode="fan_avg", distribution="uniform", seed=seed)
        self.seed = seed


class HeNormal(VarianceScaling):
    """docstring."""

    def __init__(self, seed: Optional[int] = None):
        """docstring."""
        super().__init__(
            scale=2.0, mode="fan_in", distribution="truncated_normal", seed=seed
        )
        self.seed = seed


class HeUniform(VarianceScaling):
    """docstring."""

    def __init__(self, seed: Optional[int] = None):
        """docstring."""
        super().__init__(scale=2.0, mode="fan_in", distribution="uniform", seed=seed)
        self.seed = seed


class LecunNormal(VarianceScaling):
    """docstring."""

    def __init__(self, seed: Optional[int] = None):
        """docstring."""
        super().__init__(
            scale=1.0, mode="fan_in", distribution="truncated_normal", seed=seed
        )
        self.seed = seed


class LecunUniform(VarianceScaling):
    """docstring."""

    def __init__(self, seed: Optional[int] = None):
        """docstring."""
        super().__init__(scale=1.0, mode="fan_in", distribution="uniform", seed=seed)
        self.seed = seed


class STFT(Initializer):
    """docstring."""

    def __init__(
        self,
        side: str = "real",
        window: str = "hann",
        scaling: str = "density",
        periodic: bool = False,
    ):
        """docstring."""
        pass


class IdentityInitializer(Identity):
    """docstring."""

    pass


class OrthogonalInitializer(Orthogonal):
    """docstring."""

    pass


class STFTInitializer(STFT):
    """docstring."""

    pass


class constant(Constant):
    """docstring."""

    pass


class zeros(Zeros):
    """docstring."""

    pass


class ones(Ones):
    """docstring."""

    pass


class identity(Identity):
    """docstring."""

    pass


class orthogonal(Orthogonal):
    """docstring."""

    pass


class random_normal(RandomNormal):
    """docstring."""

    pass


class random_uniform(RandomUniform):
    """docstring."""

    pass


class truncated_normal(TruncatedNormal):
    """docstring."""

    pass


class variance_scaling(VarianceScaling):
    """docstring."""

    pass


class glorot_normal(GlorotNormal):
    """docstring."""

    pass


class glorot_uniform(GlorotUniform):
    """docstring."""

    pass


class he_normal(HeNormal):
    """docstring."""

    pass


class he_uniform(HeUniform):
    """docstring."""

    pass


class lecun_normal(LecunNormal):
    """docstring."""

    pass


class lecun_uniform(LecunUniform):
    """docstring."""

    pass


class stft(STFT):
    """docstring."""

    pass


def get(identifier):
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
