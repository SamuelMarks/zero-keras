"""Weight initializers and serialization utilities for zero_keras."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Protocol, Union, runtime_checkable

from ml_switcheroo_compiler import random as msc_random
from ml_switcheroo_compiler.core.dtype import DType

from zero_keras import ops
from zero_keras.activations import _to_tensor
from zero_keras.types import (
    ConfigDict,
    ConfigValue,
    DTypeLike,
    SeedLike,
    ShapeLike,
    SupportsCall,
    TensorLike,
)

__all__ = [
    "STFT",
    "Constant",
    "GlorotNormal",
    "GlorotUniform",
    "HeNormal",
    "HeUniform",
    "Identity",
    "IdentityInitializer",
    "Initializer",
    "LecunNormal",
    "LecunUniform",
    "Ones",
    "Orthogonal",
    "OrthogonalInitializer",
    "RandomNormal",
    "RandomUniform",
    "STFTInitializer",
    "TruncatedNormal",
    "VarianceScaling",
    "Zeros",
    "constant",
    "deserialize",
    "get",
    "glorot_normal",
    "glorot_uniform",
    "he_normal",
    "he_uniform",
    "identity",
    "lecun_normal",
    "lecun_uniform",
    "ones",
    "orthogonal",
    "random_normal",
    "random_uniform",
    "serialize",
    "stft",
    "truncated_normal",
    "variance_scaling",
    "zeros",
]


@runtime_checkable
class _HasNodeIdProtocol(Protocol):
    """Protocol for symbolic IR nodes exposing an ``id`` attribute."""

    id: str


@runtime_checkable
class _HasSymbolicDataProtocol(Protocol):
    """Protocol for tensors carrying a ``data`` attribute."""

    shape: tuple[int | None, ...]
    dtype: str
    data: TensorLike


def _get_dtype(dtype: DTypeLike) -> DType:
    """Resolve ``dtype`` into a compiler ``DType`` enum member.

    Args:
        dtype: Optional dtype string or ``DType`` instance.

    Returns:
        DType: Resolved compiler ``DType`` (defaults to ``DType.Float32``).
    """
    if isinstance(dtype, DType):
        return dtype
    if dtype is None:
        return DType.Float32
    if isinstance(dtype, str):
        for d in DType:
            if d.value == dtype:
                return d
    return DType.Float32


def _wrap(x: TensorLike) -> TensorLike:
    """Unwrap eager tensor data or wrap symbolic node outputs into ``KerasTensor``.

    Args:
        x: Tensor output from an initialization op.

    Returns:
        TensorLike: Wrapped ``KerasTensor`` or underlying array data.
    """
    from zero_keras.core_layers import KerasTensor

    if isinstance(x, _HasSymbolicDataProtocol):
        if isinstance(x.data, _HasNodeIdProtocol):
            return KerasTensor(x.shape, x.dtype, data=x)
        return x.data
    return x


def _normalize_shape(shape: ShapeLike) -> tuple[int | None, ...]:
    """Convert ``shape`` into a tuple of optional integers.

    Args:
        shape: Shape integer or sequence of dimensions.

    Returns:
        tuple[int | None, ...]: Normalized shape tuple.
    """
    if isinstance(shape, int):
        return (shape,)
    return tuple(shape)


class Initializer:
    """Initializer base class: all Keras initializers inherit from this class."""

    def __init__(self, *args: ConfigValue, **kwargs: ConfigValue) -> None:
        """Initialize the base Initializer.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        _ = args, kwargs

    def __call__(
        self,
        shape: ShapeLike,
        dtype: DTypeLike = None,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Return a tensor initialized as specified by the initializer.

        Args:
            shape: Shape of the tensor to initialize.
            dtype: Optional dtype of the tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Initialized weight tensor.
        """
        _ = kwargs
        dt = _get_dtype(dtype)
        return _wrap(ops.zeros(_normalize_shape(shape), dtype=dt))

    def clone(self) -> Initializer:
        """Return a copy of the initializer recreated from its configuration.

        Returns:
            Initializer: Cloned initializer instance.
        """
        return type(self).from_config(self.get_config())

    @classmethod
    def from_config(cls, config: ConfigDict) -> Initializer:
        """Instantiate an initializer from a configuration dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            Initializer: Reconstructed initializer instance.
        """
        return cls(**config)

    def get_config(self) -> ConfigDict:
        """Return the initializer configuration dictionary.

        Returns:
            ConfigDict: Serializable configuration dictionary.
        """
        return {}


class Constant(Initializer):
    """Initializer that generates tensors with constant scalar values."""

    def __init__(self, value: float = 0.0) -> None:
        """Initialize Constant initializer.

        Args:
            value: Constant scalar value.
        """
        self.value = value

    def __call__(
        self,
        shape: ShapeLike,
        dtype: DTypeLike = None,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Generate a tensor filled with ``self.value``.

        Args:
            shape: Target tensor shape.
            dtype: Optional target dtype.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Constant tensor.
        """
        _ = kwargs
        dt = _get_dtype(dtype)
        return _wrap(ops.full(_normalize_shape(shape), self.value, dtype=dt))


class Zeros(Initializer):
    """Initializer that generates tensors initialized to 0."""

    def __call__(
        self,
        shape: ShapeLike,
        dtype: DTypeLike = None,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Generate a tensor of zeros.

        Args:
            shape: Target tensor shape.
            dtype: Optional target dtype.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Zero-initialized tensor.
        """
        _ = kwargs
        dt = _get_dtype(dtype)
        return _wrap(ops.zeros(_normalize_shape(shape), dtype=dt))

    def clone(self) -> Zeros:
        """Return a copy of this Zeros initializer.

        Returns:
            Zeros: Cloned Zeros instance.
        """
        return type(self).from_config(self.get_config())

    @classmethod
    def from_config(cls, config: ConfigDict) -> Zeros:
        """Instantiate Zeros from configuration dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            Zeros: Reconstructed Zeros instance.
        """
        return cls(**config)

    def get_config(self) -> ConfigDict:
        """Return serializable configuration dictionary.

        Returns:
            ConfigDict: Empty configuration dictionary.
        """
        return {}


class Ones(Initializer):
    """Initializer that generates tensors initialized to 1."""

    def __call__(
        self,
        shape: ShapeLike,
        dtype: DTypeLike = None,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Generate a tensor of ones.

        Args:
            shape: Target tensor shape.
            dtype: Optional target dtype.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: One-initialized tensor.
        """
        _ = kwargs
        dt = _get_dtype(dtype)
        return _wrap(ops.ones(_normalize_shape(shape), dtype=dt))


class Identity(Initializer):
    """Initializer that generates the 2D identity matrix scaled by ``gain``."""

    def __init__(self, gain: float = 1.0) -> None:
        """Initialize Identity initializer.

        Args:
            gain: Multiplicative factor applied to the identity matrix.
        """
        self.gain = gain

    def __call__(
        self,
        shape: ShapeLike,
        dtype: DTypeLike = None,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Generate a 2D scaled identity matrix.

        Args:
            shape: 2D matrix shape ``(rows, cols)``.
            dtype: Optional target dtype.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Scaled identity matrix tensor.

        Raises:
            ValueError: If ``shape`` does not have rank 2.
        """
        _ = kwargs
        norm_shape = _normalize_shape(shape)
        if len(norm_shape) != 2:
            raise ValueError("Identity matrix initializer can only be used for 2D matrices.")
        dt = _get_dtype(dtype)
        return _wrap(
            ops.multiply(
                ops.eye(norm_shape[0], norm_shape[1], dtype=dt),
                _to_tensor(self.gain),
            )
        )


class Orthogonal(Initializer):
    """Initializer that generates an orthogonal matrix via QR decomposition."""

    def __init__(self, gain: float = 1.0, seed: SeedLike = None) -> None:
        """Initialize Orthogonal initializer.

        Args:
            gain: Multiplicative factor applied to the orthogonal matrix.
            seed: Optional random seed.
        """
        self.gain = gain
        self.seed = seed

    def __call__(
        self,
        shape: ShapeLike,
        dtype: DTypeLike = None,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Generate an orthogonal weight tensor of ``shape``.

        Args:
            shape: Target tensor shape (must have rank >= 2).
            dtype: Optional target dtype.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Orthogonal tensor.

        Raises:
            ValueError: If ``shape`` has rank < 2.
        """
        _ = kwargs
        norm_shape = _normalize_shape(shape)
        if len(norm_shape) < 2:
            raise ValueError("The tensor to initialize must be at least two-dimensional")
        dt = _get_dtype(dtype)
        num_rows = 1
        for d in norm_shape[:-1]:
            num_rows *= int(d or 1)
        num_cols = int(norm_shape[-1] or 1)

        flat_shape = (max(num_rows, num_cols), min(num_rows, num_cols))
        key = str(self.seed if self.seed is not None else 0)
        a = msc_random.normal(key, flat_shape, dtype=dt)
        qr_res = ops.qr(a)
        qr_parts = list(qr_res) if isinstance(qr_res, (list, tuple)) else [a, a]
        q, r = qr_parts[0], qr_parts[1]

        d_diag = ops.diag(r)
        q = ops.multiply(q, ops.sign(d_diag))

        if num_rows < num_cols:
            q = ops.transpose(q, axes=(0, 1))

        concrete_shape = tuple(int(dim or 1) for dim in norm_shape)
        q = ops.reshape(q, concrete_shape)
        return _wrap(ops.multiply(q, _to_tensor(self.gain)))


class RandomNormal(Initializer):
    """Initializer that generates tensors with a normal distribution."""

    def __init__(
        self,
        mean: float = 0.0,
        stddev: float = 0.05,
        seed: SeedLike = None,
    ) -> None:
        """Initialize RandomNormal initializer.

        Args:
            mean: Mean of the normal distribution.
            stddev: Standard deviation of the normal distribution.
            seed: Optional random seed.
        """
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(
        self,
        shape: ShapeLike,
        dtype: DTypeLike = None,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Generate normal random values of ``shape``.

        Args:
            shape: Target tensor shape.
            dtype: Optional target dtype.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Normal-distributed tensor.
        """
        _ = kwargs
        dt = _get_dtype(dtype)
        key = str(self.seed if self.seed is not None else 0)
        concrete_shape = tuple(int(dim or 1) for dim in _normalize_shape(shape))
        res = msc_random.normal(key, concrete_shape, dtype=dt)
        return _wrap(ops.add(ops.multiply(res, _to_tensor(self.stddev)), _to_tensor(self.mean)))


class RandomUniform(Initializer):
    """Initializer that generates tensors with a uniform distribution."""

    def __init__(
        self,
        minval: float = -0.05,
        maxval: float = 0.05,
        seed: SeedLike = None,
    ) -> None:
        """Initialize RandomUniform initializer.

        Args:
            minval: Lower bound of uniform range.
            maxval: Upper bound of uniform range.
            seed: Optional random seed.
        """
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def __call__(
        self,
        shape: ShapeLike,
        dtype: DTypeLike = None,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Generate uniform random values of ``shape``.

        Args:
            shape: Target tensor shape.
            dtype: Optional target dtype.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Uniform-distributed tensor.
        """
        _ = kwargs
        dt = _get_dtype(dtype)
        key = str(self.seed if self.seed is not None else 0)
        concrete_shape = tuple(int(dim or 1) for dim in _normalize_shape(shape))
        res = msc_random.uniform(
            key, concrete_shape, dtype=dt, minval=self.minval, maxval=self.maxval
        )
        return _wrap(res)


class TruncatedNormal(Initializer):
    """Initializer that generates a truncated normal distribution."""

    def __init__(
        self,
        mean: float = 0.0,
        stddev: float = 0.05,
        seed: SeedLike = None,
    ) -> None:
        """Initialize TruncatedNormal initializer.

        Args:
            mean: Mean of the random values to generate.
            stddev: Standard deviation of the random values.
            seed: Optional random seed.
        """
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(
        self,
        shape: ShapeLike,
        dtype: DTypeLike = None,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Generate truncated-normal random values of ``shape``.

        Args:
            shape: Target tensor shape.
            dtype: Optional target dtype.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Truncated-normal tensor.
        """
        _ = kwargs
        dt = _get_dtype(dtype)
        key = str(self.seed if self.seed is not None else 0)
        concrete_shape = tuple(int(dim or 1) for dim in _normalize_shape(shape))
        res = msc_random.truncated_normal(key, -2.0, 2.0, concrete_shape, dtype=dt)
        return _wrap(ops.add(ops.multiply(res, _to_tensor(self.stddev)), _to_tensor(self.mean)))


class VarianceScaling(Initializer):
    """Initializer that adapts its scale to the shape of its input tensors."""

    def __init__(
        self,
        scale: float = 1.0,
        mode: str = "fan_in",
        distribution: str = "truncated_normal",
        seed: SeedLike = None,
        input_axes: int | Sequence[int] | None = None,
        output_axes: int | Sequence[int] | None = None,
    ) -> None:
        """Initialize VarianceScaling initializer.

        Args:
            scale: Scaling factor (positive float).
            mode: One of ``"fan_in"``, ``"fan_out"``, ``"fan_avg"``.
            distribution: One of ``"truncated_normal"``, ``"untruncated_normal"``, or ``"uniform"``.
            seed: Optional random seed.
            input_axes: Optional input axis or axes for fan-in calculation.
            output_axes: Optional output axis or axes for fan-out calculation.
        """
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed
        self.input_axes = input_axes
        self.output_axes = output_axes

    def __call__(
        self,
        shape: ShapeLike,
        dtype: DTypeLike = None,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Generate variance-scaled weights of ``shape``.

        Args:
            shape: Target tensor shape.
            dtype: Optional target dtype.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Variance-scaled weight tensor.
        """
        _ = kwargs
        dt = _get_dtype(dtype)
        norm_shape = _normalize_shape(shape)
        receptive_field_size = 1
        for dim in norm_shape[:-2]:
            receptive_field_size *= int(dim) if dim is not None else 1

        if len(norm_shape) < 2:
            fan_in = fan_out = (
                (int(norm_shape[0]) if norm_shape[0] is not None else 1)
                if len(norm_shape) == 1
                else 1
            )
        else:
            fan_in = (
                int(norm_shape[-2]) if norm_shape[-2] is not None else 1
            ) * receptive_field_size
            fan_out = (
                int(norm_shape[-1]) if norm_shape[-1] is not None else 1
            ) * receptive_field_size

        if self.mode == "fan_in":
            scale = self.scale / max(1.0, float(fan_in))
        elif self.mode == "fan_out":
            scale = self.scale / max(1.0, float(fan_out))
        else:
            scale = self.scale / max(1.0, float(fan_in + fan_out) / 2.0)

        key = str(self.seed if self.seed is not None else 0)
        resolved_shape = tuple(1 if dim is None else int(dim) for dim in norm_shape)

        if self.distribution == "truncated_normal":
            stddev = math.sqrt(scale) / 0.87962566103423978
            res: TensorLike = msc_random.truncated_normal(key, -2.0, 2.0, resolved_shape, dtype=dt)
            res = ops.multiply(res, _to_tensor(stddev))
        elif self.distribution == "untruncated_normal":
            stddev = math.sqrt(scale)
            res = msc_random.normal(key, resolved_shape, dtype=dt)
            res = ops.multiply(res, _to_tensor(stddev))
        else:
            limit = math.sqrt(3.0 * scale)
            res = msc_random.uniform(key, resolved_shape, dtype=dt, minval=-limit, maxval=limit)

        return _wrap(res)


class GlorotNormal(VarianceScaling):
    """The Glorot normal initializer, also called Xavier normal initializer."""

    def __init__(
        self,
        seed: SeedLike = None,
        input_axes: int | Sequence[int] | None = None,
        output_axes: int | Sequence[int] | None = None,
    ) -> None:
        """Initialize GlorotNormal initializer.

        Args:
            seed: Optional random seed.
            input_axes: Optional input axis or axes for fan-in calculation.
            output_axes: Optional output axis or axes for fan-out calculation.
        """
        super().__init__(
            scale=1.0,
            mode="fan_avg",
            distribution="truncated_normal",
            seed=seed,
            input_axes=input_axes,
            output_axes=output_axes,
        )
        self.seed = seed


class GlorotUniform(VarianceScaling):
    """The Glorot uniform initializer, also called Xavier uniform initializer."""

    def __init__(
        self,
        seed: SeedLike = None,
        input_axes: int | Sequence[int] | None = None,
        output_axes: int | Sequence[int] | None = None,
    ) -> None:
        """Initialize GlorotUniform initializer.

        Args:
            seed: Optional random seed.
            input_axes: Optional input axis or axes for fan-in calculation.
            output_axes: Optional output axis or axes for fan-out calculation.
        """
        super().__init__(
            scale=1.0,
            mode="fan_avg",
            distribution="uniform",
            seed=seed,
            input_axes=input_axes,
            output_axes=output_axes,
        )
        self.seed = seed


class HeNormal(VarianceScaling):
    """He normal variance scaling initializer."""

    def __init__(
        self,
        seed: SeedLike = None,
        input_axes: int | Sequence[int] | None = None,
        output_axes: int | Sequence[int] | None = None,
    ) -> None:
        """Initialize HeNormal initializer.

        Args:
            seed: Optional random seed.
            input_axes: Optional input axis or axes for fan-in calculation.
            output_axes: Optional output axis or axes for fan-out calculation.
        """
        super().__init__(
            scale=2.0,
            mode="fan_in",
            distribution="truncated_normal",
            seed=seed,
            input_axes=input_axes,
            output_axes=output_axes,
        )
        self.seed = seed


class HeUniform(VarianceScaling):
    """He uniform variance scaling initializer."""

    def __init__(
        self,
        seed: SeedLike = None,
        input_axes: int | Sequence[int] | None = None,
        output_axes: int | Sequence[int] | None = None,
    ) -> None:
        """Initialize HeUniform initializer.

        Args:
            seed: Optional random seed.
            input_axes: Optional input axis or axes for fan-in calculation.
            output_axes: Optional output axis or axes for fan-out calculation.
        """
        super().__init__(
            scale=2.0,
            mode="fan_in",
            distribution="uniform",
            seed=seed,
            input_axes=input_axes,
            output_axes=output_axes,
        )
        self.seed = seed


class LecunNormal(VarianceScaling):
    """LeCun normal variance scaling initializer."""

    def __init__(
        self,
        seed: SeedLike = None,
        input_axes: int | Sequence[int] | None = None,
        output_axes: int | Sequence[int] | None = None,
    ) -> None:
        """Initialize LecunNormal initializer.

        Args:
            seed: Optional random seed.
            input_axes: Optional input axis or axes for fan-in calculation.
            output_axes: Optional output axis or axes for fan-out calculation.
        """
        super().__init__(
            scale=1.0,
            mode="fan_in",
            distribution="truncated_normal",
            seed=seed,
            input_axes=input_axes,
            output_axes=output_axes,
        )
        self.seed = seed


class LecunUniform(VarianceScaling):
    """LeCun uniform variance scaling initializer."""

    def __init__(
        self,
        seed: SeedLike = None,
        input_axes: int | Sequence[int] | None = None,
        output_axes: int | Sequence[int] | None = None,
    ) -> None:
        """Initialize LecunUniform initializer.

        Args:
            seed: Optional random seed.
            input_axes: Optional input axis or axes for fan-in calculation.
            output_axes: Optional output axis or axes for fan-out calculation.
        """
        super().__init__(
            scale=1.0,
            mode="fan_in",
            distribution="uniform",
            seed=seed,
            input_axes=input_axes,
            output_axes=output_axes,
        )
        self.seed = seed


class STFT(Initializer):
    """Initializer of Conv kernels for Short-Term Fourier Transformation (STFT)."""

    def __init__(
        self,
        side: str = "real",
        window: str | TensorLike | None = "hann",
        scaling: str | None = "density",
        periodic: bool = False,
    ) -> None:
        """Initialize STFT kernel initializer.

        Args:
            side: ``"real"`` or ``"imag"`` component of the STFT basis.
            window: Windowing function name or window tensor.
            scaling: Normalization mode (``"density"``, ``"spectrum"``, or ``None``).
            periodic: Whether the window function is periodic.
        """
        self.side = side
        self.window = window
        self.scaling = scaling
        self.periodic = periodic


class IdentityInitializer(Identity):
    """Alias for ``Identity`` initializer."""


class OrthogonalInitializer(Orthogonal):
    """Alias for ``Orthogonal`` initializer."""


class STFTInitializer(STFT):
    """Alias for ``STFT`` initializer."""


class constant(Constant):
    """Lowercase alias for ``Constant`` initializer."""


class zeros(Zeros):
    """Lowercase alias for ``Zeros`` initializer."""


class ones(Ones):
    """Lowercase alias for ``Ones`` initializer."""


class identity(Identity):
    """Lowercase alias for ``Identity`` initializer."""


class orthogonal(Orthogonal):
    """Lowercase alias for ``Orthogonal`` initializer."""


class random_normal(RandomNormal):
    """Lowercase alias for ``RandomNormal`` initializer."""


class random_uniform(RandomUniform):
    """Lowercase alias for ``RandomUniform`` initializer."""


class truncated_normal(TruncatedNormal):
    """Lowercase alias for ``TruncatedNormal`` initializer."""


class variance_scaling(VarianceScaling):
    """Lowercase alias for ``VarianceScaling`` initializer."""


class glorot_normal(GlorotNormal):
    """Lowercase alias for ``GlorotNormal`` initializer."""


class glorot_uniform(GlorotUniform):
    """Lowercase alias for ``GlorotUniform`` initializer."""


class he_normal(HeNormal):
    """Lowercase alias for ``HeNormal`` initializer."""


class he_uniform(HeUniform):
    """Lowercase alias for ``HeUniform`` initializer."""


class lecun_normal(LecunNormal):
    """Lowercase alias for ``LecunNormal`` initializer."""


class lecun_uniform(LecunUniform):
    """Lowercase alias for ``LecunUniform`` initializer."""


class stft(STFT):
    """Lowercase alias for ``STFT`` initializer."""


_INITIALIZER_CLASSES: dict[str, type[Initializer]] = {
    "Constant": Constant,
    "GlorotNormal": GlorotNormal,
    "GlorotUniform": GlorotUniform,
    "HeNormal": HeNormal,
    "HeUniform": HeUniform,
    "Identity": Identity,
    "IdentityInitializer": IdentityInitializer,
    "Initializer": Initializer,
    "LecunNormal": LecunNormal,
    "LecunUniform": LecunUniform,
    "Ones": Ones,
    "Orthogonal": Orthogonal,
    "OrthogonalInitializer": OrthogonalInitializer,
    "RandomNormal": RandomNormal,
    "RandomUniform": RandomUniform,
    "STFT": STFT,
    "STFTInitializer": STFTInitializer,
    "TruncatedNormal": TruncatedNormal,
    "VarianceScaling": VarianceScaling,
    "Zeros": Zeros,
    "constant": constant,
    "glorot_normal": glorot_normal,
    "glorot_uniform": glorot_uniform,
    "he_normal": he_normal,
    "he_uniform": he_uniform,
    "identity": identity,
    "lecun_normal": lecun_normal,
    "lecun_uniform": lecun_uniform,
    "ones": ones,
    "orthogonal": orthogonal,
    "random_normal": random_normal,
    "random_uniform": random_uniform,
    "stft": stft,
    "truncated_normal": truncated_normal,
    "variance_scaling": variance_scaling,
    "zeros": zeros,
}

_CANONICAL_INITIALIZERS: dict[str, type[Initializer]] = {
    "constant": Constant,
    "glorot_normal": GlorotNormal,
    "glorot_uniform": GlorotUniform,
    "he_normal": HeNormal,
    "he_uniform": HeUniform,
    "identity": Identity,
    "identity_initializer": IdentityInitializer,
    "initializer": Initializer,
    "lecun_normal": LecunNormal,
    "lecun_uniform": LecunUniform,
    "ones": Ones,
    "orthogonal": Orthogonal,
    "orthogonal_initializer": OrthogonalInitializer,
    "random_normal": RandomNormal,
    "random_uniform": RandomUniform,
    "stft": STFT,
    "stft_initializer": STFTInitializer,
    "truncated_normal": TruncatedNormal,
    "variance_scaling": VarianceScaling,
    "zeros": Zeros,
}

_INITIALIZER_NORMALIZED: dict[str, type[Initializer]] = {
    k.replace("_", "").lower(): v for k, v in _CANONICAL_INITIALIZERS.items()
}

InitializerIdentifier = Union[
    Initializer,
    SupportsCall,
    str,
    ConfigDict,
    TensorLike,
    None,
]


def get(identifier: InitializerIdentifier) -> InitializerIdentifier:
    """Retrieve a Keras initializer object via an identifier.

    Args:
        identifier: Initializer instance, string name, config dictionary, or callable.

    Returns:
        InitializerIdentifier: Resolved initializer instance or callable.
    """
    if identifier is None:
        return GlorotUniform()
    if isinstance(identifier, str):
        cls = _CANONICAL_INITIALIZERS.get(
            identifier,
            _INITIALIZER_NORMALIZED.get(identifier.replace("_", "").lower(), GlorotUniform),
        )
        return cls()
    if isinstance(identifier, dict):
        class_name = str(identifier.get("class_name", ""))
        cfg_raw = identifier.get("config", {})
        cfg = cfg_raw if isinstance(cfg_raw, dict) else {}
        cls = _INITIALIZER_NORMALIZED.get(class_name.replace("_", "").lower(), GlorotUniform)
        return cls(**cfg)
    return identifier


def serialize(initializer: InitializerIdentifier) -> ConfigDict | str | None:
    """Serialize an initializer instance to a configuration dictionary.

    Args:
        initializer: Initializer instance or string name.

    Returns:
        ConfigDict | str | None: Serialized initializer representation.
    """
    if initializer is None:
        return None
    if isinstance(initializer, str):
        return initializer
    cfg = initializer.get_config() if isinstance(initializer, Initializer) else {}
    return {
        "class_name": type(initializer).__name__,
        "config": cfg,
    }


def deserialize(
    config: InitializerIdentifier,
    custom_objects: dict[str, ConfigValue] | None = None,
) -> InitializerIdentifier:
    """Deserialize an initializer from its configuration dictionary or string name.

    Args:
        config: Serialized initializer config or string identifier.
        custom_objects: Optional dictionary mapping custom names to classes.

    Returns:
        InitializerIdentifier: Reconstructed initializer instance.
    """
    if config is None:
        return None
    if isinstance(config, str):
        cls = _INITIALIZER_CLASSES.get(config)
        if cls is not None:
            return cls()
        return config
    if isinstance(config, dict):
        class_name = str(config.get("class_name", ""))
        conf_raw = config.get("config", {})
        conf = conf_raw if isinstance(conf_raw, dict) else {}

        cls = _INITIALIZER_CLASSES.get(class_name)
        if cls is not None:
            return cls(**conf)

        if custom_objects and class_name in custom_objects:
            custom_fn = custom_objects[class_name]
            if callable(custom_fn):
                return custom_fn(**conf)

    return config
