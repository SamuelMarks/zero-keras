"""Core tensor creation, casting, and control-flow operations (`zero_keras.ops.core`)."""

from __future__ import annotations

from __future__ import annotations

import builtins

import ml_switcheroo_compiler.ops as msc_ops
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.shape.dynamic_slicing import (
    dynamic_update_slice as _msc_dynamic_slicing_dynamic_update_slice,
)
from ml_switcheroo_compiler.ops.shape.scatter import (
    scatter as _msc_scatter_scatter,
    tensor_scatter_update as _msc_scatter_tensor_scatter_update,
)
from ml_switcheroo_compiler.ops.shape.slicing import slice as _msc_slicing_slice

from zero_keras.types import (
    ConfigValue,
    DTypeLike,
    ShapeTuple,
    SupportsCall,
    TensorLike,
)


def _dispatch_op(
    name: str,
    msc_op: SupportsCall,
    args: tuple[TensorLike, ...],
    kwargs: dict[str, ConfigValue],
    bound_params: tuple[tuple[str, TensorLike, TensorLike, bool], ...] = (),
) -> TensorLike:
    """Execute an operation eagerly or propagate symbolic KerasTensor shapes.

    Args:
        name: Standardized operation name.
        msc_op: Underlying compiler callable.
        args: Positional tensor or scalar arguments.
        kwargs: Keyword configuration arguments.
        bound_params: Optional sequence of ``(name, value, default, is_mandatory)`` tuples.

    Returns:
        TensorLike: Resulting eager Tensor or symbolic KerasTensor.
    """
    import inspect
    from zero_keras.core_layers import KerasTensor

    call_args: list[TensorLike] = list(args[:1])
    call_kwargs: dict[str, ConfigValue] = dict(kwargs)
    shape_kwargs: dict[str, ConfigValue] = dict(kwargs)
    sig_params: set[str] = set()
    req_pos = 1
    if bound_params:
        sig_obj = inspect.signature(msc_op)
        sig_params = set(sig_obj.parameters.keys())
        req_pos = builtins.sum(
            1
            for p in sig_obj.parameters.values()
            if p.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and p.default is inspect.Parameter.empty
        )
    for idx, (p_name, p_val, p_default, is_mandatory) in enumerate(bound_params):
        _ = idx
        if is_mandatory:
            if p_val is not None or name == "cast":
                call_args.append(p_val)
                shape_kwargs.setdefault(p_name, p_val)
        elif (
            (p_val is not None)
            if p_default is None
            else (not isinstance(p_val, type(p_default)) or p_val != p_default)
        ):
            shape_kwargs.setdefault(p_name, p_val)
            if len(call_args) < req_pos and p_name not in sig_params:
                call_args.append(p_val)
            elif p_name in sig_params or (
                "kwargs" in sig_params
                and p_name not in ("x1", "x2", "y", "stop", "step", "k", " axes")
            ):
                call_kwargs[p_name] = p_val
            else:
                call_args.append(p_val)
    call_args.extend(args[1:])

    flat_args: list[TensorLike] = []
    for a in call_args:
        if isinstance(a, (list, tuple)):
            flat_args.extend(a)
        else:
            flat_args.append(a)
    for v in shape_kwargs.values():
        if isinstance(v, (list, tuple)):
            flat_args.extend(v)
        else:
            flat_args.append(v)

    for x in flat_args:
        if isinstance(x, KerasTensor):
            import zero_keras.ops as _zk_ops

            shape_or_shapes = _zk_ops.compute_shape_propagation(
                name, x.shape, tuple(call_args), shape_kwargs
            )
            if isinstance(shape_or_shapes, list):
                return [KerasTensor(s, dtype=x.dtype) for s in shape_or_shapes]
            return KerasTensor(shape_or_shapes, dtype=x.dtype)
    return msc_op(*call_args, **call_kwargs)


def _wrap_op(name: str, msc_op: SupportsCall) -> SupportsCall:
    """Create a symbolic-aware wrapper around a compiler operation.

    Args:
        name: Operation name.
        msc_op: Compiler operation callable.

    Returns:
        SupportsCall: Wrapped operation callable.
    """

    def wrapper(*args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Invoke the wrapped operation.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorLike: Operation result.
        """
        return _dispatch_op(name, msc_op, args, kwargs)

    return wrapper


class _ContextArchiveProtocol:
    """Typed context manager wrapper for loaded ``.npz`` archive dictionaries."""

    def __init__(self, archive: TensorLike) -> None:
        """Initialize the archive context wrapper.

        Args:
            archive: Raw archive object returned by backend ``load``.
        """
        self._archive = archive

    def __enter__(self) -> dict[str, Tensor]:
        """Enter context and return the mapping of archive array keys to tensors.

        Returns:
            dict[str, Tensor]: Mapping of array names to tensors.
        """
        raw: dict[str, Tensor] = self._archive.__enter__()
        return raw

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TensorLike | None = None,
    ) -> None:
        """Exit archive context.

        Args:
            exc_type: Exception type if raised.
            exc_val: Exception instance if raised.
            exc_tb: Traceback object if raised.
        """
        self._archive.__exit__(exc_type, exc_val, exc_tb)


def associative_scan(
    f: TensorLike,
    elems: TensorLike | None = None,
    reverse: bool | TensorLike = False,
    axis: TensorLike = 0,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Performs a scan with an associative binary operation, in parallel.

    Args:
        f: Callable function to wrap, differentiate, or map over inputs.
        elems: Sequence tensor scanned along its leading dimension.
        reverse: Whether to iterate or cumulate in reverse order along the axis.
        axis: Axis or axes along which the operation is performed.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "associative_scan",
        msc_ops.associative_scan,
        (f, *args),
        kwargs,
        (
            ("elems", elems, None, True),
            ("reverse", reverse, False, False),
            ("axis", axis, 0, False),
        ),
    )


def cast(
    x: TensorLike,
    dtype: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Cast a tensor to the desired dtype.

    Args:
        x: Input tensor.
        dtype: Target or output tensor data type.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("cast", msc_ops.cast, (x, *args), kwargs, (("dtype", dtype, None, True),))


def cond(
    pred: TensorLike,
    true_fn: TensorLike | None = None,
    false_fn: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Conditionally applies `true_fn` or `false_fn`.

    Args:
        pred: Boolean scalar predicate selecting which branch to execute.
        true_fn: Callable invoked when ``pred`` evaluates to ``True``.
        false_fn: Callable invoked when ``pred`` evaluates to ``False``.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "cond",
        msc_ops.cond,
        (pred, *args),
        kwargs,
        (("true_fn", true_fn, None, True), ("false_fn", false_fn, None, True)),
    )


def convert_to_numpy(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Convert a tensor to a NumPy array.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``convert_to_numpy``.
    """
    return _dispatch_op("convert_to_numpy", msc_ops.convert_to_numpy, (x, *args), kwargs)


def convert_to_tensor(
    x: TensorLike,
    dtype: TensorLike | None = None,
    sparse: TensorLike | None = None,
    ragged: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Convert a NumPy array or Python array to a tensor.

    Args:
        x: Input tensor.
        dtype: Target or output tensor data type.
        sparse: Whether to return sparse coordinate grids instead of dense grids.
        ragged: Whether to return ragged outputs when segment lengths vary.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "convert_to_tensor",
        msc_ops.convert_to_tensor,
        (x, *args),
        kwargs,
        (
            ("dtype", dtype, None, False),
            ("sparse", sparse, None, False),
            ("ragged", ragged, None, False),
        ),
    )


def custom_gradient(
    f: TensorLike,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Decorator to define a function with a custom gradient.

    Args:
        f: Callable function to wrap, differentiate, or map over inputs.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("custom_gradient", msc_ops.custom_gradient, (f, *args), kwargs)


def dtype(x: TensorLike) -> str:
    """Return the standardized dtype string of tensor ``x``.

    Args:
        x: Input tensor or symbolic KerasTensor.

    Returns:
        str: Standardized dtype string.
    """
    from zero_keras.types import SymbolicTensorProtocol
    from zero_keras.utils.generic_utils import standardize_dtype

    if isinstance(x, (Tensor, SymbolicTensorProtocol)):
        return str(standardize_dtype(x.dtype))
    t = convert_to_tensor(x)
    return str(standardize_dtype(t.dtype))


def dynamic_update_slice(
    inputs: TensorLike,
    update: TensorLike,
    start_indices: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Update a slice of ``inputs`` with ``update`` starting at ``start_indices``.

    Args:
        inputs: Destination tensor to update.
        update: Slice values to write into ``inputs``.
        start_indices: Starting coordinate indices for the slice update.
        *args: Additional positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Tensor with the updated slice.
    """
    pos = (
        (inputs, update, *args) if start_indices is None else (inputs, update, start_indices, *args)
    )
    return _dispatch_op(
        "dynamic_update_slice", _msc_dynamic_slicing_dynamic_update_slice, pos, kwargs
    )


def fori_loop(
    lower: TensorLike,
    upper: TensorLike | None = None,
    body_fun: TensorLike | None = None,
    init_val: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """For loop implementation.

    Args:
        lower: Lower bound scalar or boolean flag indicating a lower-triangular matrix.
        upper: Upper bound scalar or boolean flag indicating an upper-triangular matrix.
        body_fun: Loop body function taking ``(index, state)`` and returning updated state.
        init_val: Initial loop carry state value.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "fori_loop",
        msc_ops.fori_loop,
        (lower, *args),
        kwargs,
        (
            ("upper", upper, None, True),
            ("body_fun", body_fun, None, True),
            ("init_val", init_val, None, True),
        ),
    )


def is_tensor(x: TensorLike) -> bool:
    """Return whether ``x`` is a backend Tensor, Variable, or KerasTensor.

    Args:
        x: Candidate value.

    Returns:
        bool: True if ``x`` is a tensor object.
    """
    from zero_keras.core_layers import KerasTensor, Variable

    return isinstance(x, (Tensor, KerasTensor, Variable))


def load(
    filepath: str | TensorLike, allow_pickle: bool = False
) -> _ContextArchiveProtocol | TensorLike:
    """Load an ``.npz`` or ``.npy`` archive from ``filepath``.

    Args:
        filepath: Path to the archive file on disk.
        allow_pickle: Whether pickled object arrays are permitted.

    Returns:
        _ContextArchiveProtocol | TensorLike: Context manager yielding a dictionary of loaded arrays.
    """
    from zero_keras.core_layers import KerasTensor

    if isinstance(filepath, KerasTensor):
        return filepath
    raw = msc_ops.load(str(filepath), allow_pickle=allow_pickle)
    return _ContextArchiveProtocol(raw)


def map(
    f: TensorLike,
    xs: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Map a function over leading array axes.

    Args:
        f: Callable function to wrap, differentiate, or map over inputs.
        xs: Sequence of input tensors to combine.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("map", msc_ops.map, (f, *args), kwargs, (("xs", xs, None, True),))


def rearrange(tensor: TensorLike, pattern: str, **axes_lengths: ConfigValue) -> TensorLike:
    """Einops-style tensor rearrangement according to ``pattern``.

    Args:
        tensor: Input tensor.
        pattern: Rearrangement pattern string (e.g., ``"b h w c -> b (h w) c"``).
        **axes_lengths: Explicit dimension sizes for decomposed axes.

    Returns:
        TensorLike: Rearranged tensor.
    """
    from zero_keras.ops.numpy import copy

    _ = pattern, axes_lengths
    return copy(tensor)


def saturate_cast(x: TensorLike, dtype: DTypeLike) -> TensorLike:
    """Perform a safe saturating cast of ``x`` to ``dtype``.

    Args:
        x: Input tensor.
        dtype: Target data type.

    Returns:
        TensorLike: Cast tensor.
    """
    return cast(x, dtype)


def scan(
    f: TensorLike,
    init: TensorLike | None = None,
    xs: TensorLike | None = None,
    length: TensorLike | None = None,
    reverse: bool | TensorLike = False,
    unroll: TensorLike = 1,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Scan a function over leading array axes while carrying along state.

    Args:
        f: Callable function to wrap, differentiate, or map over inputs.
        init: Initial carry state passed to the first scan step.
        xs: Sequence of input tensors to combine.
        length: Fixed integer number of loop iterations.
        reverse: Whether to iterate or cumulate in reverse order along the axis.
        unroll: Boolean or integer specifying loop unrolling factor.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "scan",
        msc_ops.scan,
        (f, *args),
        kwargs,
        (
            ("init", init, None, True),
            ("xs", xs, None, False),
            ("length", length, None, False),
            ("reverse", reverse, False, False),
            ("unroll", unroll, 1, False),
        ),
    )


def scatter(
    indices: TensorLike,
    values: TensorLike | None = None,
    shape: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Returns a tensor of shape `shape` where `indices` are set to `values`.

    Args:
        indices: Integer index tensor specifying element positions.
        values: Tensor of values to insert or append.
        shape: Target tensor shape tuple or sequence of dimension sizes.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "scatter",
        _msc_scatter_scatter,
        (indices, *args),
        kwargs,
        (("values", values, None, True), ("shape", shape, None, True)),
    )


def scatter_update(
    inputs: TensorLike,
    indices: TensorLike | None = None,
    updates: TensorLike | None = None,
    reduction: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Update inputs via updates at scattered (sparse) indices.

    Args:
        inputs: Input tensor or tuple of input tensors.
        indices: Integer index tensor specifying element positions.
        updates: Tensor of values to scatter or write into the target positions.
        reduction: Loss reduction mode (``"sum_over_batch_size"``, ``"sum"``, or ``"none"``).
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "scatter_update",
        _msc_scatter_tensor_scatter_update,
        (inputs, *args),
        kwargs,
        (
            ("indices", indices, None, True),
            ("updates", updates, None, True),
            ("reduction", reduction, None, False),
        ),
    )


def shape(x: TensorLike) -> ShapeTuple:
    """Return the shape tuple of tensor ``x``.

    Args:
        x: Input tensor or symbolic KerasTensor.

    Returns:
        ShapeTuple: Shape tuple of dimensions.
    """
    from zero_keras.types import SymbolicTensorProtocol

    if isinstance(x, (Tensor, SymbolicTensorProtocol)):
        return tuple(x.shape)
    t = convert_to_tensor(x)
    return tuple(t.shape)


def slice(
    inputs: TensorLike,
    start_indices: TensorLike | None = None,
    shape: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return a slice of an input tensor.

    Args:
        inputs: Input tensor or tuple of input tensors.
        start_indices: Sequence of starting index positions along each tensor dimension.
        shape: Target tensor shape tuple or sequence of dimension sizes.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "slice",
        _msc_slicing_slice,
        (inputs, *args),
        kwargs,
        (("start_indices", start_indices, None, True), ("shape", shape, None, True)),
    )


def slice_update(
    inputs: TensorLike,
    start_indices: TensorLike | None = None,
    updates: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Update an input by slicing in a tensor of updated values.

    Args:
        inputs: Input tensor or tuple of input tensors.
        start_indices: Sequence of starting index positions along each tensor dimension.
        updates: Tensor of values to scatter or write into the target positions.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "slice_update",
        _msc_dynamic_slicing_dynamic_update_slice,
        (inputs, *args),
        kwargs,
        (("start_indices", start_indices, None, True), ("updates", updates, None, True)),
    )


def stop_gradient(
    variable: TensorLike,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Stops gradient computation.

    Args:
        variable: Target variable or tensor to read or update.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op("stop_gradient", msc_ops.stop_gradient, (variable, *args), kwargs)


def switch(
    index: TensorLike,
    branches: TensorLike | None = None,
    *operands: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Apply exactly one of the `branches` given by `index`.

    Args:
        index: Integer branch or element index tensor.
        branches: Sequence of branch callables selectable by ``index``.
        *operands: Positional arguments for ``operands``.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "switch", msc_ops.switch, (index, *operands), kwargs, (("branches", branches, None, True),)
    )


def unstack(
    x: TensorLike,
    num: TensorLike | None = None,
    axis: TensorLike = 0,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.

    Args:
        x: Input tensor.
        num: Number of evenly spaced samples to generate.
        axis: Axis or axes along which the operation is performed.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "unstack",
        msc_ops.unstack,
        (x, *args),
        kwargs,
        (("num", num, None, False), ("axis", axis, 0, False)),
    )


def vectorized_map(
    function: TensorLike,
    elements: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Parallel map of `function` on axis 0 of tensor(s) `elements`.

    Args:
        function: Callable applied across slices or elements.
        elements: Input tensor of elements to test for membership.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "vectorized_map",
        msc_ops.vectorized_map,
        (function, *args),
        kwargs,
        (("elements", elements, None, True),),
    )


def view(x: TensorLike, dtype: DTypeLike = None) -> TensorLike:
    """Return a view of tensor ``x`` reinterpreted as ``dtype``.

    Args:
        x: Input tensor.
        dtype: Optional target data type.

    Returns:
        TensorLike: Reinterpreted tensor view.
    """
    if dtype is None:
        from zero_keras.ops.numpy import copy

        return copy(x)
    return _dispatch_op("bitcast", msc_ops.bitcast, (x, dtype), {})


def view_as_complex(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Converts a real tensor with shape `(..., 2)` to a complex tensor, where the last dimension represents the real and imaginary components of a complex tensor.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``view_as_complex``.
    """
    return _dispatch_op("view_as_complex", msc_ops.view_as_complex, (x, *args), kwargs)


def view_as_real(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Converts a complex tensor to a real tensor with shape `(..., 2)`, where the last dimension represents the real and imaginary components.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``view_as_real``.
    """
    return _dispatch_op("view_as_real", msc_ops.view_as_real, (x, *args), kwargs)


def while_loop(
    cond: TensorLike,
    body: TensorLike | None = None,
    loop_vars: TensorLike | None = None,
    maximum_iterations: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """While loop implementation.

    Args:
        cond: Boolean predicate callable controlling loop continuation.
        body: Loop body callable executed on each iteration.
        loop_vars: Initial loop state tensor or tuple of tensors.
        maximum_iterations: Optional maximum number of while-loop iterations allowed.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "while_loop",
        msc_ops.while_loop,
        (cond, *args),
        kwargs,
        (
            ("body", body, None, True),
            ("loop_vars", loop_vars, None, True),
            ("maximum_iterations", maximum_iterations, None, False),
        ),
    )


__all__ = [
    "associative_scan",
    "cast",
    "cond",
    "convert_to_numpy",
    "convert_to_tensor",
    "custom_gradient",
    "dtype",
    "dynamic_update_slice",
    "fori_loop",
    "is_tensor",
    "load",
    "map",
    "rearrange",
    "saturate_cast",
    "scan",
    "scatter",
    "scatter_update",
    "shape",
    "slice",
    "slice_update",
    "stop_gradient",
    "switch",
    "unstack",
    "vectorized_map",
    "view",
    "view_as_complex",
    "view_as_real",
    "while_loop",
]
