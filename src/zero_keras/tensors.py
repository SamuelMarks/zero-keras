"""Core symbolic and mutable tensor abstractions (`KerasTensor`, `Variable`, `KerasVariable`, `InputSpec`, `Input`)."""

from __future__ import annotations

from zero_keras.operations import Node
from zero_keras.types import (
    EagerModeConfigProtocol,
    ShapeCarrierProtocol,
    TensorLike,
)


class TensorShape(tuple):
    """Tuple representation of a tensor's dimensions."""

    def __new__(cls, dims: TensorLike) -> TensorLike:
        """Create a new ``TensorShape`` instance from ``dims``.

        Args:
            dims: Sequence of dimension sizes, integer dimension, or ``None``.

        Returns:
            TensorLike: Constructed ``TensorShape`` tuple.
        """
        if dims is None:
            return super().__new__(cls, ())
        if isinstance(dims, int):
            dims = (dims,)
        return super().__new__(cls, tuple(dims))

    @property
    def dims(self) -> TensorLike:
        """Return the dimensions of the shape as a list.

        Returns:
            TensorLike: List of dimension values.
        """
        return list(self)

    @property
    def rank(self) -> TensorLike:
        """Return the number of dimensions (rank) of the tensor shape.

        Returns:
            TensorLike: Integer rank of the shape.
        """
        return len(self)

    def as_list(self) -> TensorLike:
        """Return the tensor shape dimensions as a Python list.

        Returns:
            TensorLike: List of dimension sizes.
        """
        return list(self)

    def is_fully_defined(self) -> TensorLike:
        """Return whether every dimension in the shape is known (non-None).

        Returns:
            TensorLike: ``True`` if all dimensions are non-None, else ``False``.
        """
        return all(d is not None for d in self)


class KerasTensor:
    """Symbolic tensor representation carrying shape, dtype, and graph history."""

    def __init__(
        self,
        shape: TensorLike,
        dtype: str = "float32",
        name: TensorLike | None = None,
        data: TensorLike | None = None,
        sparse: bool = False,
        ragged: bool = False,
    ) -> None:
        """Initialize a symbolic KerasTensor.

        Args:
            shape: Shape tuple of the symbolic tensor.
            dtype: Data type string. Defaults to ``"float32"``.
            name: Optional symbolic tensor name.
            data: Optional backing eager tensor or array.
            sparse: Whether the symbolic tensor is sparse.
            ragged: Whether the symbolic tensor is ragged.
        """
        self.shape = shape
        self.dtype = dtype
        self.name = name
        self._sparse: bool = bool(sparse)
        self._ragged: bool = bool(ragged)
        self._keras_history: Node | None = None
        if data is not None:
            self.data: TensorLike = data
        else:
            try:
                import math
                import ml_switcheroo_compiler.ops as backend_ops

                if (
                    isinstance(shape, (list, tuple))
                    and all(isinstance(d, int) and d >= 0 for d in shape)
                    and math.prod(shape) <= 64
                ):
                    self.data = backend_ops.zeros(shape)
                else:
                    self.data = None
            except Exception:
                self.data = None

    def __iter__(self) -> TensorLike:
        """Disallow iteration over a symbolic ``KerasTensor``."""
        raise TypeError("KerasTensor is not iterable")

    @property
    def ndim(self) -> int | None:
        """Return the number of dimensions of the symbolic tensor.

        Returns:
            int | None: Number of dimensions, or ``None`` if shape is unknown.
        """
        return len(self.shape) if self.shape is not None else None

    @property
    def ragged(self) -> bool:
        """Return whether the symbolic tensor represents a ragged tensor.

        Returns:
            bool: ``True`` if ragged, else ``False``.
        """
        return self._ragged

    def reshape(self, shape: TensorLike) -> KerasTensor:
        """Return a reshaped symbolic ``KerasTensor``.

        Args:
            shape: Target shape tuple.

        Returns:
            KerasTensor: Reshaped symbolic tensor.
        """
        return KerasTensor(shape=shape, dtype=self.dtype, name=self.name)

    @property
    def sparse(self) -> bool:
        """Return whether the symbolic tensor represents a sparse tensor.

        Returns:
            bool: ``True`` if sparse, else ``False``.
        """
        return self._sparse

    def squeeze(self, axis: TensorLike | None = None) -> KerasTensor:
        """Remove dimensions of size 1 from the tensor shape.

        Args:
            axis: Optional axis or sequence of axes to squeeze.

        Returns:
            KerasTensor: Squeezed symbolic tensor.
        """
        if self.shape is None:
            return KerasTensor(shape=None, dtype=self.dtype, name=self.name)
        new_shape = tuple(
            [
                s
                for i, s in enumerate(self.shape)
                if s != 1
                or (
                    axis is not None
                    and i not in (axis if isinstance(axis, (list, tuple)) else [axis])
                )
            ]
        )
        return KerasTensor(shape=new_shape, dtype=self.dtype, name=self.name)

    def __call__(self, x: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Return ``x`` unchanged when called as a callable.

        Args:
            x: Input value.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: The input ``x``.
        """
        return x

    def __add__(self, other: TensorLike) -> TensorLike:
        """Add ``other`` to this tensor.

        Args:
            other: Right-hand operand.

        Returns:
            TensorLike: Sum tensor or symbolic ``KerasTensor``.
        """
        if self.data is not None:
            import ml_switcheroo_compiler.ops as msc_ops

            from zero_keras.activations import _to_tensor

            return msc_ops.add(_to_tensor(self.data), _to_tensor(other))
        return KerasTensor(self.shape, self.dtype)

    def __sub__(self, other: TensorLike) -> TensorLike:
        """Subtract ``other`` from this tensor.

        Args:
            other: Right-hand operand.

        Returns:
            TensorLike: Difference tensor or symbolic ``KerasTensor``.
        """
        if self.data is not None:
            import ml_switcheroo_compiler.ops as msc_ops

            from zero_keras.activations import _to_tensor

            return msc_ops.subtract(_to_tensor(self.data), _to_tensor(other))
        return KerasTensor(self.shape, self.dtype)

    def __mul__(self, other: TensorLike) -> TensorLike:
        """Multiply this tensor by ``other``.

        Args:
            other: Right-hand operand.

        Returns:
            TensorLike: Product tensor or symbolic ``KerasTensor``.
        """
        if self.data is not None:
            from zero_keras import ops
            from zero_keras.activations import _to_tensor

            return ops.multiply(_to_tensor(self), _to_tensor(other))
        return KerasTensor(self.shape, self.dtype)

    def __rmul__(self, other: TensorLike) -> TensorLike:
        """Multiply ``other`` by this tensor.

        Args:
            other: Left-hand operand.

        Returns:
            TensorLike: Product tensor or symbolic ``KerasTensor``.
        """
        if self.data is not None:
            from zero_keras import ops
            from zero_keras.activations import _to_tensor

            return ops.multiply(_to_tensor(self), _to_tensor(other))
        return KerasTensor(self.shape, self.dtype)

    def __rtruediv__(self, other: TensorLike) -> TensorLike:
        """Divide ``other`` by this tensor.

        Args:
            other: Left-hand numerator operand.

        Returns:
            TensorLike: Quotient tensor or symbolic ``KerasTensor``.
        """
        if self.data is not None:
            from zero_keras import ops
            from zero_keras.activations import _to_tensor

            return ops.true_divide(_to_tensor(self), _to_tensor(other))
        return KerasTensor(self.shape, self.dtype)

    def __neg__(self) -> KerasTensor:
        """Negate this symbolic tensor.

        Returns:
            KerasTensor: Negated symbolic tensor.
        """
        return KerasTensor(self.shape, self.dtype)

    def __radd__(self, other: TensorLike) -> TensorLike:
        """Add this tensor to ``other``.

        Args:
            other: Left-hand operand.

        Returns:
            TensorLike: Sum tensor or symbolic ``KerasTensor``.
        """
        if self.data is not None:
            from zero_keras import ops
            from zero_keras.activations import _to_tensor

            return ops.add(_to_tensor(self), _to_tensor(other))
        return KerasTensor(self.shape, self.dtype)

    def __rsub__(self, other: TensorLike) -> KerasTensor:
        """Subtract this tensor from ``other``.

        Args:
            other: Left-hand operand.

        Returns:
            KerasTensor: Difference symbolic tensor.
        """
        return KerasTensor(self.shape, self.dtype)

    def __truediv__(self, other: TensorLike) -> TensorLike:
        """Divide this tensor by ``other``.

        Args:
            other: Right-hand denominator operand.

        Returns:
            TensorLike: Quotient tensor or symbolic ``KerasTensor``.
        """
        if self.data is not None:
            import ml_switcheroo_compiler.ops as msc_ops

            from zero_keras.activations import _to_tensor

            return msc_ops.true_divide(_to_tensor(self.data), _to_tensor(other))
        return KerasTensor(self.shape, self.dtype)

    def __pow__(self, other: TensorLike) -> KerasTensor:
        """Raise this tensor to the power ``other``.

        Args:
            other: Exponent operand.

        Returns:
            KerasTensor: Symbolic tensor result.
        """
        return KerasTensor(self.shape, self.dtype)

    def __eq__(self, other: TensorLike) -> TensorLike:
        """Compare this tensor elementwise with ``other``.

        Args:
            other: Right-hand operand.

        Returns:
            TensorLike: Equality comparison result.
        """
        if self.data is not None:
            return self.data == other
        safe_shape = (
            tuple(1 if x is None else x for x in self.shape) if self.shape is not None else ()
        )
        import ml_switcheroo_compiler.ops as backend_ops

        return backend_ops.ones(safe_shape)

    def __bool__(self) -> bool:
        """Convert this tensor to a boolean in eager mode or raise ``TypeError``.

        Returns:
            bool: Boolean truth value in eager mode.
        """
        from ml_switcheroo_compiler.core import config

        if (
            config.eager_mode if isinstance(config, EagerModeConfigProtocol) else False
        ) and self.data is not None:
            try:
                return bool(self.data)
            except Exception:
                pass
        raise TypeError(
            "Using a KerasTensor as a Python bool is not allowed. Use ops.cond or ops.where for data-dependent control flow, or enable eager execution."
        )

    def numpy(self) -> TensorLike:
        """Return the backing tensor data.

        Returns:
            TensorLike: Backing data array or ``None``.
        """
        return self.data

    def __array__(
        self, dtype: TensorLike | None = None, copy: TensorLike | None = None
    ) -> TensorLike:
        """Convert this symbolic tensor to a NumPy array.

        Args:
            dtype: Optional target NumPy dtype.
            copy: Optional boolean controlling copy behavior.

        Returns:
            TensorLike: NumPy array representation.
        """
        from zero_keras import ops

        if self.data is not None:
            arr = self.data
        else:
            shape = [d if d is not None else 1 for d in self.shape] if self.shape else [0]
            arr = ops.zeros(shape)

        arr = ops.convert_to_tensor(arr).numpy()

        if dtype is not None:
            arr = arr.astype(dtype, copy=copy if copy is not None else True)
        elif copy:
            arr = arr.copy()

        return arr

    def __getitem__(self, key: TensorLike) -> TensorLike:
        """Index into the backing tensor data.

        Args:
            key: Slice or index key.

        Returns:
            TensorLike: Indexed element or ``0.0`` when unbacked.
        """
        if self.data is not None:
            return self.data[key]
        return 0.0


def Input(
    shape: TensorLike | None = None, name: TensorLike | None = None, **kwargs: TensorLike
) -> TensorLike:
    """Instantiate a symbolic ``KerasTensor`` entry point for a Functional model.

    Args:
        shape: Shape tuple specifying tensor dimensions (excluding batch dimension).
        name: Optional string identifier for the input tensor.
        **kwargs: Additional keyword configuration arguments (e.g., ``batch_size``, ``batch_shape``, ``dtype``).

    Returns:
        TensorLike: Symbolic ``KerasTensor`` representing the model input.
    """
    batch_shape = kwargs.get("batch_shape")
    if batch_shape is None:
        batch_size = kwargs.get("batch_size")
        t_arg = kwargs.get("tensor")
        if shape is not None:
            batch_shape = (batch_size,) + tuple(shape)
        elif t_arg is not None and isinstance(t_arg, ShapeCarrierProtocol):
            batch_shape = tuple(t_arg.shape)
        else:
            batch_shape = (batch_size,)
    kt = KerasTensor(
        batch_shape,
        dtype=kwargs.get("dtype", "float32"),
        name=name,
    )
    kt.batch_shape = batch_shape
    return kt


class InputSpec:
    """Specification of rank, dtype, and shape for layer inputs."""

    def __new__(cls, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Allocate an ``InputSpec`` instance with default fields.

        Args:
            *args: Positional constructor arguments.
            **kwargs: Keyword constructor arguments.

        Returns:
            TensorLike: Allocated ``InputSpec`` instance.
        """
        _ = args, kwargs
        inst = super().__new__(cls)
        inst.dtype = None
        inst.shape = None
        inst.ndim = None
        inst.max_ndim = None
        inst.min_ndim = None
        inst.axes = None
        inst.allow_last_axis_squeeze = False
        inst.name = None
        return inst

    @classmethod
    def from_config(cls, config: TensorLike) -> TensorLike:
        """Instantiate `InputSpec` from its serialized configuration dictionary.

        Args:
            config: Configuration dictionary describing object hyperparameters.

        Returns:
            TensorLike: Computed result for `from_config`.
        """
        return cls(**config)

    def get_config(self) -> TensorLike:
        """Return the serializable configuration dictionary for `InputSpec`.

        Returns:
            TensorLike: Computed result for `get_config`.
        """
        return {
            "dtype": self.dtype,
            "shape": self.shape,
            "ndim": self.ndim,
            "max_ndim": self.max_ndim,
            "min_ndim": self.min_ndim,
            "axes": self.axes,
            "allow_last_axis_squeeze": self.allow_last_axis_squeeze,
            "name": self.name,
        }

    def __init__(
        self,
        dtype: TensorLike | None = None,
        shape: TensorLike | None = None,
        ndim: TensorLike | None = None,
        max_ndim: TensorLike | None = None,
        min_ndim: TensorLike | None = None,
        axes: TensorLike | None = None,
        allow_last_axis_squeeze: bool = False,
        name: TensorLike | None = None,
    ) -> None:
        """Initialize the `InputSpec` instance.

        Args:
            dtype: Expected data type string of the input tensor.
            shape: Expected shape tuple of the input tensor.
            ndim: Expected exact rank (number of dimensions) of the input.
            max_ndim: Maximum allowed rank (number of dimensions) of the input.
            min_ndim: Minimum allowed rank (number of dimensions) of the input.
            axes: Optional mapping from axis index to required dimension size.
            allow_last_axis_squeeze: Whether a trailing dimension of size 1 may be squeezed.
            name: Optional string identifier for the input tensor.
        """
        self.dtype = dtype
        self.shape = shape
        self.ndim = ndim
        self.max_ndim = max_ndim
        self.min_ndim = min_ndim
        self.axes = axes or {}
        self.allow_last_axis_squeeze = allow_last_axis_squeeze
        self.name = name


class Variable:
    """Mutable tensor state variable."""

    def __new__(cls, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Allocate a ``Variable`` instance with default internal attributes.

        Args:
            *args: Positional constructor arguments.
            **kwargs: Keyword constructor arguments.

        Returns:
            TensorLike: Allocated ``Variable`` instance.
        """
        _ = args, kwargs
        inst = super().__new__(cls)
        inst._value = None
        inst._aggregation = None
        inst._constraint = None
        inst._handle = None
        inst._regularizer = None
        inst._synchronization = None
        inst.shape = None
        inst.dtype = None
        inst.initializer = None
        inst.trainable = True
        inst.name = None
        return inst

    def __getitem__(self, key: TensorLike) -> TensorLike:
        """Index or slice the underlying variable tensor with ``key``.

        Args:
            key: Integer index, slice, or tuple of indices/slices.

        Returns:
            TensorLike: Indexed sub-tensor or scalar value.
        """
        return self.value[key] if self.value is not None else None

    @property
    def aggregation(self) -> TensorLike:
        """Return the distributed variable aggregation strategy.

        Returns:
            TensorLike: Aggregation strategy string or ``None``.
        """
        return self._aggregation

    @aggregation.setter
    def aggregation(self, value: TensorLike) -> None:
        """Set the distributed variable aggregation strategy.

        Args:
            value: Aggregation strategy string or ``None``.
        """
        self._aggregation = value

    def assign(self, value: TensorLike) -> None:
        """Assign a new tensor ``value`` to this variable in-place.

        Args:
            value: New tensor or array value to store.
        """
        self.value = value

    def assign_add(self, value: TensorLike) -> None:
        """Add ``value`` to the current variable value in-place.

        Args:
            value: Increment tensor or scalar to add.
        """
        from zero_keras import ops

        self.value = ops.add(self.value, value) if self.value is not None else value

    def assign_sub(self, value: TensorLike) -> None:
        """Subtract ``value`` from the current variable value in-place.

        Args:
            value: Decrement tensor or scalar to subtract.
        """
        from zero_keras import ops

        self.value = (
            ops.subtract(self.value, value) if self.value is not None else ops.negative(value)
        )

    @property
    def constraint(self) -> TensorLike:
        """Return the projection constraint function applied to this variable.

        Returns:
            TensorLike: Constraint callable or ``None``.
        """
        return self._constraint

    @constraint.setter
    def constraint(self, value: TensorLike) -> None:
        """Set the projection constraint function applied to this variable.

        Args:
            value: Constraint callable or ``None``.
        """
        self._constraint = value

    @property
    def handle(self) -> TensorLike:
        """Return the backend resource handle associated with this variable.

        Returns:
            TensorLike: Backend resource handle or ``None``.
        """
        return self._handle

    @handle.setter
    def handle(self, value: TensorLike) -> None:
        """Set the backend resource handle associated with this variable.

        Args:
            value: Backend resource handle object.
        """
        self._handle = value

    @property
    def ndim(self) -> TensorLike:
        """Return the number of dimensions (rank) of this variable.

        Returns:
            TensorLike: Integer number of dimensions.
        """
        return len(self.shape) if self.shape is not None else 0

    def numpy(self) -> TensorLike:
        """Convert the variable's current value to a NumPy array.

        Returns:
            TensorLike: NumPy array copy of the variable value.
        """
        from ml_switcheroo_compiler.serialization import to_numpy

        return to_numpy(self.value)

    def __array__(self, dtype: TensorLike | None = None) -> TensorLike:
        """Return a NumPy array representation for ``np.asarray(variable)``.

        Args:
            dtype: Optional NumPy data type to cast the array to.

        Returns:
            TensorLike: NumPy array representation of the variable.
        """
        from zero_keras import ops

        arr = self.value if self.value is not None else ops.zeros([0])

        res = ops.convert_to_tensor(arr).numpy()

        if dtype is not None:
            return res.astype(dtype)
        return res

    def overwrite_with_gradient(self, gradient: TensorLike) -> None:
        """Directly overwrite the variable value with ``gradient``.

        Args:
            gradient: Gradient tensor used to overwrite the variable.
        """
        self._gradient = gradient
        self.value = gradient

    @property
    def path(self) -> TensorLike:
        """Return the hierarchical path name of this variable.

        Returns:
            TensorLike: Variable path string or ``None``.
        """
        return self.name

    @property
    def regularizer(self) -> TensorLike:
        """Return the regularizer callable attached to this variable.

        Returns:
            TensorLike: Regularizer callable or ``None``.
        """
        return self._regularizer

    @regularizer.setter
    def regularizer(self, value: TensorLike) -> None:
        """Set the regularizer callable attached to this variable.

        Args:
            value: Regularizer callable or ``None``.
        """
        self._regularizer = value

    @property
    def synchronization(self) -> TensorLike:
        """Return the distributed synchronization policy for this variable.

        Returns:
            TensorLike: Synchronization policy identifier or ``None``.
        """
        return self._synchronization

    @synchronization.setter
    def synchronization(self, value: TensorLike) -> None:
        """Set the distributed synchronization policy for this variable.

        Args:
            value: Synchronization policy identifier or ``None``.
        """
        self._synchronization = value

    @property
    def value(self) -> TensorLike:
        """Return the current tensor value of this variable, initializing lazily if needed.

        Returns:
            TensorLike: Current underlying tensor value.
        """
        if self._value is not None:
            return self._value
        if self.shape is not None and self.initializer is not None:
            return self._tensor
        return None

    @value.setter
    def value(self, val: TensorLike) -> None:
        """Set the underlying tensor value of this variable.

        Args:
            val: New tensor or array value.
        """
        self._value = val

    def __add__(self, other: TensorLike) -> TensorLike:
        """Return the elementwise sum ``self + other``.

        Args:
            other: Right-hand tensor or scalar operand.

        Returns:
            TensorLike: Sum tensor.
        """
        from zero_keras import ops

        return ops.add(self.value, other)

    def __radd__(self, other: TensorLike) -> TensorLike:
        """Return the reflected elementwise sum ``other + self``.

        Args:
            other: Left-hand tensor or scalar operand.

        Returns:
            TensorLike: Sum tensor.
        """
        from zero_keras import ops

        return ops.add(other, self.value)

    def __sub__(self, other: TensorLike) -> TensorLike:
        """Return the elementwise difference ``self - other``.

        Args:
            other: Right-hand tensor or scalar operand.

        Returns:
            TensorLike: Difference tensor.
        """
        from zero_keras import ops

        return ops.subtract(self.value, other)

    def __rsub__(self, other: TensorLike) -> TensorLike:
        """Return the reflected elementwise difference ``other - self``.

        Args:
            other: Left-hand tensor or scalar operand.

        Returns:
            TensorLike: Difference tensor.
        """
        from zero_keras import ops

        return ops.subtract(other, self.value)

    def __mul__(self, other: TensorLike) -> TensorLike:
        """Return the elementwise product ``self * other``.

        Args:
            other: Right-hand tensor or scalar operand.

        Returns:
            TensorLike: Product tensor.
        """
        from zero_keras import ops

        return ops.multiply(self.value, other)

    def __rmul__(self, other: TensorLike) -> TensorLike:
        """Return the reflected elementwise product ``other * self``.

        Args:
            other: Left-hand tensor or scalar operand.

        Returns:
            TensorLike: Product tensor.
        """
        from zero_keras import ops

        return ops.multiply(other, self.value)

    def __truediv__(self, other: TensorLike) -> TensorLike:
        """Return the elementwise quotient ``self / other``.

        Args:
            other: Right-hand tensor or scalar divisor.

        Returns:
            TensorLike: Quotient tensor.
        """
        from zero_keras import ops

        return ops.divide(self.value, other)

    def __rtruediv__(self, other: TensorLike) -> TensorLike:
        """Return the reflected elementwise quotient ``other / self``.

        Args:
            other: Left-hand tensor or scalar dividend.

        Returns:
            TensorLike: Quotient tensor.
        """
        from zero_keras import ops

        return ops.divide(other, self.value)

    def __init__(
        self,
        initializer: TensorLike,
        shape: TensorLike | None = None,
        dtype: TensorLike | None = None,
        trainable: bool = True,
        name: TensorLike | None = None,
    ) -> None:
        """Initialize a Keras Variable.

        Args:
            initializer: Initializer name, callable, or initial tensor value.
            shape: Shape tuple of the variable.
            dtype: Data type of the variable.
            trainable: Whether the variable is trainable.
            name: Optional variable name.
        """
        self.initializer = initializer
        self.shape = shape
        self.dtype = dtype
        self.trainable = trainable
        self.name = name
        self._value: TensorLike | None = None

    @property
    def _tensor(self) -> TensorLike:
        """Materialize and return the backend ``Tensor`` backing this variable.

        Returns:
            TensorLike: Materialized backend ``Tensor``.
        """
        from zero_keras.activations import _to_tensor

        if self._value is None and self.shape is not None:
            from zero_keras.initializers import get as get_initializer

            init_fn = get_initializer(self.initializer)
            self._value = init_fn(shape=self.shape, dtype=self.dtype or "float32")
        if self._value is not None:
            return _to_tensor(self._value)
        from zero_keras import ops

        return _to_tensor(ops.zeros((), dtype=self.dtype or "float32"))


KerasVariable = Variable


def is_keras_tensor(x: TensorLike) -> TensorLike:
    """Return whether `x` is a symbolic `KerasTensor`.

    Args:
        x: Input tensor or array.

    Returns:
        TensorLike: Computed result for `is_keras_tensor`.
    """
    return isinstance(x, KerasTensor)


__all__ = [
    "Input",
    "InputSpec",
    "KerasTensor",
    "KerasVariable",
    "TensorShape",
    "Variable",
    "is_keras_tensor",
]
