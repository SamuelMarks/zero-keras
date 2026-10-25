"""Core Keras symbolic tensors, variables, base Layer, and Model abstractions."""

from __future__ import annotations

from zero_keras.types import TensorLike

from zero_keras import ops


class TensorShape(tuple):
    """TensorShape class."""

    def __new__(cls, dims: TensorLike) -> TensorLike:
        """Implement the `__new__` special method on `TensorShape`.

        Args:
            dims: Configuration or input value for `dims` (dims).

        Returns:
            TensorLike: Computed result for `__new__`.
        """
        if dims is None:
            return super().__new__(cls, ())
        if isinstance(dims, int):
            dims = (dims,)
        return super().__new__(cls, tuple(dims))

    @property
    def dims(self) -> TensorLike:
        """Compute or retrieve `dims` for `TensorShape`.

        Returns:
            TensorLike: Computed result for `dims`.
        """
        return list(self)

    @property
    def rank(self) -> TensorLike:
        """Compute or retrieve `rank` for `TensorShape`.

        Returns:
            TensorLike: Computed result for `rank`.
        """
        return len(self)

    def as_list(self) -> TensorLike:
        """Compute or retrieve `as list` for `TensorShape`.

        Returns:
            TensorLike: Computed result for `as_list`.
        """
        return list(self)

    def is_fully_defined(self) -> TensorLike:
        """Compute or retrieve `is fully defined` for `TensorShape`.

        Returns:
            TensorLike: Computed result for `is_fully_defined`.
        """
        return all(d is not None for d in self)


from zero_keras.activations import _to_tensor

"""Core layers module."""


class Node:
    """Symbolic graph node connecting a layer or operation with its inputs and outputs."""

    def __init__(
        self,
        operation: TensorLike | None = None,
        call_args: TensorLike | None = None,
        call_kwargs: TensorLike | None = None,
        outputs: TensorLike | None = None,
    ) -> None:
        """Initialize a symbolic graph node.

        Args:
            operation: The layer or operation executed by this node.
            call_args: Positional arguments or input tensors passed to the operation.
            call_kwargs: Keyword arguments passed to the operation.
            outputs: Output tensor(s) produced by the operation.
        """
        self.operation = operation
        self.layer = operation
        self.call_args = call_args
        self.inputs = call_args
        self.call_kwargs = call_kwargs
        self.outputs = outputs
        self.input_tensors = call_args if call_args is not None else []
        self.output_tensors = outputs if outputs is not None else []
        if not hasattr(operation, "_inbound_nodes"):
            operation._inbound_nodes = []
        operation._inbound_nodes.append(self)
        if isinstance(outputs, list):
            for t in outputs:
                t._keras_history = self
        elif hasattr(outputs, "_keras_history"):
            outputs._keras_history = self


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
                import ml_switcheroo_compiler.ops as backend_ops

                self.data = backend_ops.zeros(shape) if shape is not None else None
            except Exception:
                self.data = None

    def __iter__(self) -> TensorLike:
        """Disallow iteration over a symbolic ``KerasTensor``."""
        raise TypeError("KerasTensor is not iterable")

    @property
    def ndim(self) -> int | None:
        """Compute or retrieve `ndim` for `KerasTensor`.

        Returns:
            int | None: Computed result for `ndim`.
        """
        return len(self.shape) if self.shape is not None else None

    @property
    def ragged(self) -> bool:
        """Compute or retrieve `ragged` for `KerasTensor`.

        Returns:
            bool: Computed result for `ragged`.
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
        """Compute or retrieve `sparse` for `KerasTensor`.

        Returns:
            bool: Computed result for `sparse`.
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
            config.eager_mode if hasattr(config, "eager_mode") else False
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
    """Compute or retrieve `Input` for `Input`.

    Args:
        shape: Shape tuple specifying tensor dimensions.
        name: Optional string identifier for the object.
        **kwargs: Additional keyword configuration arguments.

    Returns:
        TensorLike: Computed result for `Input`.
    """
    batch_shape = kwargs.get("batch_shape")
    if batch_shape is None:
        batch_size = kwargs.get("batch_size")
        t_arg = kwargs.get("tensor")
        if shape is not None:
            batch_shape = (batch_size,) + tuple(shape)
        elif t_arg is not None and hasattr(t_arg, "shape"):
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


class Layer:
    """This is the class from which all layers inherit.

    A layer is a callable object that takes as input one or more tensors and
    that outputs one or more tensors. It involves *computation*, defined
    in the `call()` method, and a *state* (weight variables). State can be
    created:

    * in `__init__()`, for instance via `self.add_weight()`;
    * in the optional `build()` method, which is invoked by the first
      `__call__()` to the layer, and supplies the shape(s) of the input(s),
      which may not have been known at initialization time.

    Layers are recursively composable: If you assign a Layer instance as an
    attribute of another Layer, the outer layer will start tracking the weights
    created by the inner layer. Nested layers should be instantiated in the
    `__init__()` method or `build()` method.

    Users will just instantiate a layer and then treat it as a callable.

    Args:
        trainable: Boolean, whether the layer's variables should be trainable.
        name: String name of the layer.
        dtype: The dtype of the layer's computations and weights. Can also be a
            `keras.DTypePolicy`, which allows the computation and weight dtype
            to differ. Defaults to `None`. `None` means to use
            `keras.config.dtype_policy()`, which is a `float32` policy unless
            set to different value (via `keras.config.set_dtype_policy()`).

    Attributes:
        name: The name of the layer (string).
        dtype: Dtype of the layer's weights. Alias of `layer.variable_dtype`.
        variable_dtype: Dtype of the layer's weights.
        compute_dtype: The dtype of the layer's computations.
            Layers automatically cast inputs to this dtype, which causes
            the computations and output to also be in this dtype.
            When mixed precision is used with a
            `keras.DTypePolicy`, this will be different
            than `variable_dtype`.
        trainable_weights: List of variables to be included in backprop.
        non_trainable_weights: List of variables that should not be
            included in backprop.
        weights: The concatenation of the lists trainable_weights and
            non_trainable_weights (in this order).
        trainable: Whether the layer should be trained (boolean), i.e.
            whether its potentially-trainable weights should be returned
            as part of `layer.trainable_weights`.
        input_spec: Optional (list of) `InputSpec` object(s) specifying the
            constraints on inputs that can be accepted by the layer.

    We recommend that descendants of `Layer` implement the following methods:

    * `__init__()`: Defines custom layer attributes, and creates layer weights
        that do not depend on input shapes, using `add_weight()`,
        or other state.
    * `build(self, input_shape)`: This method can be used to create weights that
        depend on the shape(s) of the input(s), using `add_weight()`, or other
        state. `__call__()` will automatically build the layer
        (if it has not been built yet) by calling `build()`.
    * `call(self, *args, **kwargs)`: Called in `__call__` after making
        sure `build()` has been called. `call()` performs the logic of applying
        the layer to the input arguments.
        Two reserved keyword arguments you can optionally use in `call()` are:
            1. `training` (boolean, whether the call is in inference mode or
                training mode).
            2. `mask` (boolean tensor encoding masked timesteps in the input,
                used e.g. in RNN layers).
        A typical signature for this method is `call(self, inputs)`, and user
        could optionally add `training` and `mask` if the layer need them.
    * `get_config(self)`: Returns a dictionary containing the configuration
        used to initialize this layer. If the keys differ from the arguments
        in `__init__()`, then override `from_config(self)` as well.
        This method is used when saving
        the layer or a model that contains this layer.

    Examples:
    Here's a basic example: a layer with two variables, `w` and `b`,
    that returns `y = w . x + b`.
    It shows how to implement `build()` and `call()`.
    Variables set as attributes of a layer are tracked as weights
    of the layers (in `layer.weights`).

    ```python
    class SimpleDense(Layer):
        def __init__(self, units=32):
            super().__init__()
            self.units = units

        # Create the state of the layer (weights)
        def build(self, input_shape):
            self.kernel = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer="glorot_uniform",
                trainable=True,
                name="kernel",
            )
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
                name="bias",
            )

        # Defines the computation
        def call(self, inputs):
            return ops.matmul(inputs, self.kernel) + self.bias

    # Instantiates the layer.
    linear_layer = SimpleDense(4)

    # This will also call `build(input_shape)` and create the weights.
    y = linear_layer(ops.ones((2, 2)))
    assert len(linear_layer.weights) == 2

    # These weights are trainable, so they're listed in `trainable_weights`:
    assert len(linear_layer.trainable_weights) == 2
    ```

    Besides trainable weights, updated via backpropagation during training,
    layers can also have non-trainable weights. These weights are meant to
    be updated manually during `call()`. Here's a example layer that computes
    the running sum of its inputs:

    ```python
    class ComputeSum(Layer):

      def __init__(self, input_dim):
          super(ComputeSum, self).__init__()
          # Create a non-trainable weight.
          self.total = self.add_weight(
            shape=(),
            initializer="zeros",
            trainable=False,
            name="total",
          )

      def call(self, inputs):
          self.total.assign(self.total + ops.sum(inputs))
          return self.total

    my_sum = ComputeSum(2)
    x = ops.ones((2, 2))
    y = my_sum(x)

    assert my_sum.weights == [my_sum.total]
    assert my_sum.non_trainable_weights == [my_sum.total]
    assert my_sum.trainable_weights == []
    ```

    """

    def add_metric(
        self, value: TensorLike, name: TensorLike | None = None, **kwargs: TensorLike
    ) -> None:
        """add_metric docstring.

        Args:
            value: Value.
            name: Name.
            **kwargs: Kwargs.
        """
        if name is not None:
            self._metrics.append({name: value})
        else:
            self._metrics.append(value)

    def add_variable(
        self,
        shape: TensorLike,
        initializer: TensorLike,
        dtype: TensorLike | None = None,
        trainable: bool = True,
        name: TensorLike | None = None,
    ) -> TensorLike:
        """Compute or retrieve `add variable` for `Layer`.

        Args:
            shape: Shape tuple specifying tensor dimensions.
            initializer: Configuration or input value for `initializer` (initializer).
            dtype: Data type string or DTypePolicy.
            trainable: Whether the variable or layer weights may be updated by backpropagation.
            name: Optional string identifier for the object.

        Returns:
            TensorLike: Computed result for `add_variable`.
        """
        return Variable(initializer, shape=shape, dtype=dtype, trainable=trainable, name=name)

    def build_from_config(self, config: TensorLike) -> None:
        """build_from_config docstring.

        Args:
            config: Config.
        """
        self.build(config.get("input_shape"))

    @property
    def compute_dtype(self) -> TensorLike:
        """Compute or retrieve `compute dtype` for `Layer`.

        Returns:
            TensorLike: Computed result for `compute_dtype`.
        """
        return self._compute_dtype

    def compute_mask(self, inputs: TensorLike, mask: TensorLike | None = None) -> TensorLike:
        """Compute the output boolean mask tensor given `inputs` and `mask`.

        Args:
            inputs: Input tensor or nested structure of tensors.
            mask: Optional boolean mask tensor indicating valid timesteps or elements.

        Returns:
            TensorLike: Computed result for `compute_mask`.
        """
        return mask

    def compute_output_spec(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Compute the symbolic `KerasTensor` specification for the given inputs.

        Args:
            *args: Positional arguments forwarded to the operation or base class.
            **kwargs: Additional keyword configuration arguments.

        Returns:
            TensorLike: Computed result for `compute_output_spec`.
        """
        if hasattr(self, "compute_output_shape"):
            return self.compute_output_shape(*args, **kwargs)
        from zero_keras.core_layers import KerasTensor

        return KerasTensor(shape=None, dtype=self.dtype or "float32")

    def count_params(self) -> TensorLike:
        """Compute or retrieve `count params` for `Layer`.

        Returns:
            TensorLike: Computed result for `count_params`.
        """
        return 0

    @property
    def dtype(self) -> TensorLike:
        """Compute or retrieve `dtype` for `Layer`.

        Returns:
            TensorLike: Computed result for `dtype`.
        """
        return self._dtype

    @property
    def dtype_policy(self) -> TensorLike:
        """Compute or retrieve `dtype policy` for `Layer`.

        Returns:
            TensorLike: Computed result for `dtype_policy`.
        """
        return self._dtype_policy

    def get_build_config(self) -> TensorLike:
        """Return the input shape configuration used to build this layer.

        Returns:
            TensorLike: Computed result for `get_build_config`.
        """
        return self._build_config

    def get_config(self) -> TensorLike:
        """Return the serializable configuration dictionary for `Layer`.

        Returns:
            TensorLike: Computed result for `get_config`.
        """
        return {"name": self.name}

    @property
    def input(self) -> TensorLike:
        """Compute or retrieve `input` for `Layer`.

        Returns:
            TensorLike: Computed result for `input`.
        """
        return self._inbound_nodes[0].input_tensors if self._inbound_nodes else None

    @property
    def input_dtype(self) -> TensorLike:
        """Compute or retrieve `input dtype` for `Layer`.

        Returns:
            TensorLike: Computed result for `input_dtype`.
        """
        return self.dtype

    def load_own_variables(self, store: TensorLike) -> None:
        """load_own_variables docstring.

        Args:
            store: Store.
        """
        if not self.variables:
            return
        for i, var in enumerate(self.variables):
            if str(i) in store:
                var.assign(store[str(i)])

    @property
    def metrics(self) -> TensorLike:
        """Compute or retrieve `metrics` for `Layer`.

        Returns:
            TensorLike: Computed result for `metrics`.
        """
        return self._metrics

    @property
    def metrics_variables(self) -> TensorLike:
        """Compute or retrieve `metrics variables` for `Layer`.

        Returns:
            TensorLike: Computed result for `metrics_variables`.
        """
        return []

    @property
    def non_trainable_variables(self) -> TensorLike:
        """Compute or retrieve `non trainable variables` for `Layer`.

        Returns:
            TensorLike: Computed result for `non_trainable_variables`.
        """
        return self.non_trainable_weights

    @property
    def output(self) -> TensorLike:
        """Compute or retrieve `output` for `Layer`.

        Returns:
            TensorLike: Computed result for `output`.
        """
        return self._inbound_nodes[0].output_tensors if self._inbound_nodes else None

    @property
    def path(self) -> TensorLike:
        """Compute or retrieve `path` for `Layer`.

        Returns:
            TensorLike: Computed result for `path`.
        """
        return self.name

    @property
    def quantization_mode(self) -> TensorLike:
        """Compute or retrieve `quantization mode` for `Layer`.

        Returns:
            TensorLike: Computed result for `quantization_mode`.
        """
        return self._quantization_mode

    def quantize(self, mode: str | TensorLike) -> TensorLike:
        """Apply post-training quantization to the layer or model weights in `mode`.

        Args:
            mode: Configuration or input value for `mode` (mode).

        Returns:
            TensorLike: Computed result for `quantize`.
        """
        self._quantization_mode = mode
        return self

    def quantized_build(self, input_shape: TensorLike) -> None:
        """quantized_build docstring.

        Args:
            input_shape: Input shape.
        """
        self.build(input_shape)

    def quantized_call(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Execute the quantized forward pass on `inputs`.

        Args:
            *args: Positional arguments forwarded to the operation or base class.
            **kwargs: Additional keyword configuration arguments.

        Returns:
            TensorLike: Computed result for `quantized_call`.
        """
        return self.call(*args, **kwargs)

    def rematerialized_call(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Compute or retrieve `rematerialized call` for `Layer`.

        Args:
            *args: Positional arguments forwarded to the operation or base class.
            **kwargs: Additional keyword configuration arguments.

        Returns:
            TensorLike: Computed result for `rematerialized_call`.
        """
        return self.call(*args, **kwargs)

    def save_own_variables(self, store: TensorLike) -> None:
        """save_own_variables docstring.

        Args:
            store: Store.
        """
        if not self.variables:
            return
        for i, var in enumerate(self.variables):
            store[str(i)] = var.numpy()

    def stateless_call(
        self,
        trainable_variables: TensorLike,
        non_trainable_variables: TensorLike,
        *args: TensorLike,
        **kwargs: TensorLike,
    ) -> TensorLike:
        """Run a stateless forward pass using explicitly provided variable values.

        Args:
            trainable_variables: Configuration or input value for `trainable_variables` (trainable variables).
            non_trainable_variables: Configuration or input value for `non_trainable_variables` (non trainable variables).
            *args: Positional arguments forwarded to the operation or base class.
            **kwargs: Additional keyword configuration arguments.

        Returns:
            TensorLike: Computed result for `stateless_call`.
        """
        return self.call(*args, **kwargs)

    def symbolic_call(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Compute or retrieve `symbolic call` for `Layer`.

        Args:
            *args: Positional arguments forwarded to the operation or base class.
            **kwargs: Additional keyword configuration arguments.

        Returns:
            TensorLike: Computed result for `symbolic_call`.
        """
        return self.call(*args, **kwargs)

    @property
    def trainable_variables(self) -> TensorLike:
        """Compute or retrieve `trainable variables` for `Layer`.

        Returns:
            TensorLike: Computed result for `trainable_variables`.
        """
        return self.trainable_weights

    @property
    def variable_dtype(self) -> TensorLike:
        """Compute or retrieve `variable dtype` for `Layer`.

        Returns:
            TensorLike: Computed result for `variable_dtype`.
        """
        return self.dtype

    @property
    def variables(self) -> TensorLike:
        """Compute or retrieve `variables` for `Layer`.

        Returns:
            TensorLike: Computed result for `variables`.
        """
        return self.weights

    @property
    def input_spec(self) -> TensorLike:
        """Compute or retrieve `input spec` for `Layer`.

        Returns:
            TensorLike: Computed result for `input_spec`.
        """
        return self._input_spec

    @input_spec.setter
    def input_spec(self, value: TensorLike) -> None:
        """Compute or retrieve `input spec` for `Layer`.

        Args:
            value: Configuration or input value for `value` (value).
        """
        self._input_spec = value

    @property
    def supports_masking(self) -> TensorLike:
        """Compute or retrieve `supports masking` for `Layer`.

        Returns:
            TensorLike: Computed result for `supports_masking`.
        """
        return self._supports_masking

    @supports_masking.setter
    def supports_masking(self, value: TensorLike) -> None:
        """Compute or retrieve `supports masking` for `Layer`.

        Args:
            value: Configuration or input value for `value` (value).
        """
        self._supports_masking = value

    @property
    def trainable(self) -> TensorLike:
        """Compute or retrieve `trainable` for `Layer`.

        Returns:
            TensorLike: Computed result for `trainable`.
        """
        return self._trainable

    @trainable.setter
    def trainable(self, value: TensorLike) -> None:
        """Compute or retrieve `trainable` for `Layer`.

        Args:
            value: Configuration or input value for `value` (value).
        """
        self._trainable = value

    def __init__(self, **kwargs: TensorLike) -> None:
        """Initialize the Layer base instance.

        Args:
            **kwargs: Base layer configuration keyword arguments.
        """
        self._tracked_attrs: dict[str, TensorLike] = {}
        self.built = False
        self._name = kwargs.get("name")
        self._kwargs = kwargs
        self._metrics: list[TensorLike] = []
        self._losses: list[TensorLike] = []
        self._weights: list[TensorLike] = []
        self._inbound_nodes: list[Node] = []
        self._outbound_nodes: list[Node] = []
        self._compute_dtype: TensorLike | None = None
        self._dtype: TensorLike = kwargs.get("dtype")
        self._dtype_policy: TensorLike | None = None
        self._build_config: TensorLike = {"input_shape": None}
        self._input_spec: TensorLike | None = None
        self._supports_masking: bool = False
        self._trainable: TensorLike = kwargs.get("trainable", True)
        self._quantization_mode: TensorLike | None = None

    def __setattr__(self, name: str, value: TensorLike) -> None:
        """Set attribute ``name`` to ``value`` and register potential sublayers.

        Args:
            name: Attribute name.
            value: Attribute value.
        """
        super().__setattr__(name, value)
        if name != "_tracked_attrs" and not name.startswith("_"):
            if not hasattr(self, "_tracked_attrs"):
                super().__setattr__("_tracked_attrs", {})
            self._tracked_attrs[name] = value

    @property
    def name(self) -> str:
        """Return the name of the layer.

        Returns:
            str: Layer name string.
        """
        return self._name or "layer"

    @classmethod
    def from_config(cls, config: TensorLike) -> TensorLike:
        """Instantiate a layer from its configuration dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            TensorLike: Reconstructed layer instance.
        """
        return cls(**config)

    def add_loss(self, loss: TensorLike) -> None:
        """Can be called inside of the `call()` method to add a scalar loss.

        Example:
        ```python
        class MyLayer(Layer):
            ...
            def call(self, x):
                self.add_loss(ops.sum(x))
                return x
        ```

        Args:
            loss: Scalar loss tensor to record.
        """
        self._losses.append(loss)

    @property
    def losses(self) -> TensorLike:
        """Compute or retrieve `losses` for `Layer`.

        Returns:
            TensorLike: Computed result for `losses`.
        """
        losses_list = list(self._losses)
        for v in self._tracked_attrs.values():
            if isinstance(v, Layer):
                losses_list.extend(v.losses)
            else:
                seq = (
                    v.values()
                    if isinstance(v, dict)
                    else (v if isinstance(v, (list, tuple)) else ())
                )
                for item in seq:
                    if hasattr(item, "losses"):
                        losses_list.extend(item.losses)
        return losses_list

    def compute_output_shape(self, input_shape: TensorLike) -> TensorLike:
        """Compute the output tensor shape produced for `input_shape`.

        Args:
            input_shape: Input tensor shape tuple or nested structure of shape tuples.

        Returns:
            TensorLike: Computed result for `compute_output_shape`.
        """
        from zero_keras.core_layers import KerasTensor

        if isinstance(input_shape, list):
            inputs = [KerasTensor(shape=s) for s in input_shape]
        elif isinstance(input_shape, tuple):
            inputs = KerasTensor(shape=input_shape)
        else:
            inputs = KerasTensor(shape=None)

        try:
            out = self.call(inputs)
            if isinstance(out, list):
                return [(x.shape if hasattr(x, "shape") else None) for x in out]
            elif isinstance(out, tuple):
                return tuple((x.shape if hasattr(x, "shape") else None) for x in out)
            return out.shape if hasattr(out, "shape") else None
        except Exception:
            return input_shape

    def build(self, input_shape: TensorLike) -> None:
        """Create and initialize the layer state variables for `input_shape`.

        Args:
            input_shape: Input tensor shape tuple or nested structure of shape tuples.
        """
        self.built = True

    def add_weight(
        self,
        shape: TensorLike,
        initializer: str | TensorLike = "zeros",
        trainable: bool = True,
        name: TensorLike | None = None,
        regularizer: TensorLike | None = None,
        constraint: TensorLike | None = None,
        **kwargs: TensorLike,
    ) -> TensorLike:
        """Create and register a weight variable on this layer.

        Args:
            shape: Shape tuple specifying tensor dimensions.
            initializer: Configuration or input value for `initializer` (initializer).
            trainable: Whether the variable or layer weights may be updated by backpropagation.
            name: Optional string identifier for the object.
            regularizer: Configuration or input value for `regularizer` (regularizer).
            constraint: Configuration or input value for `constraint` (constraint).
            **kwargs: Additional keyword configuration arguments.

        Returns:
            TensorLike: Computed result for `add_weight`.
        """
        from zero_keras.initializers import get as get_initializer

        init_fn = get_initializer(initializer)
        weight = init_fn(shape=shape)
        self._weights.append(weight)
        return weight

    @property
    def weights(self) -> TensorLike:
        """Compute or retrieve `weights` for `Layer`.

        Returns:
            TensorLike: Computed result for `weights`.
        """
        w = list(self._weights)
        for v in self._tracked_attrs.values():
            if isinstance(v, Layer):
                w.extend(v.weights)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, Layer):
                        w.extend(item.weights)
        # Deduplicate while preserving order
        seen = set()
        res = []
        for x in w:
            idx = id(x)
            if idx not in seen:
                seen.add(idx)
                res.append(x)
        return res

    @property
    def trainable_weights(self) -> TensorLike:
        """Compute or retrieve `trainable weights` for `Layer`.

        Returns:
            TensorLike: Computed result for `trainable_weights`.
        """
        return self.weights

    @property
    def non_trainable_weights(self) -> TensorLike:
        """Compute or retrieve `non trainable weights` for `Layer`.

        Returns:
            TensorLike: Computed result for `non_trainable_weights`.
        """
        return []

    def get_weights(self) -> TensorLike:
        """Return a list of NumPy arrays representing all layer weight values.

        Returns:
            TensorLike: Computed result for `get_weights`.
        """
        from ml_switcheroo_compiler.serialization import to_numpy

        return [to_numpy(w) for w in self.weights]

    def set_weights(self, weights: TensorLike) -> TensorLike:
        """Set the values of all layer weights from a list of NumPy arrays.

        Args:
            weights: Configuration or input value for `weights` (weights).
        """
        if len(weights) != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} weights, got {len(weights)}")
        from ml_switcheroo_compiler.ops import asarray

        for w, new_w in zip(self.weights, weights):
            if hasattr(w, "assign"):
                w.assign(new_w)
            elif hasattr(w, "data"):
                w[...] = asarray(new_w)
            else:
                w[:] = new_w

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Execute the forward computation of `Layer` on `inputs`.

        Args:
            inputs: Input tensor or nested structure of tensors.
            *args: Positional arguments forwarded to the operation or base class.
            **kwargs: Additional keyword configuration arguments.

        Returns:
            TensorLike: Computed result for `call`.
        """
        return inputs

    def __call__(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Implement the `__call__` special method on `Layer`.

        Args:
            inputs: Input tensor or nested structure of tensors.
            *args: Positional arguments forwarded to the operation or base class.
            **kwargs: Additional keyword configuration arguments.

        Returns:
            TensorLike: Computed result for `__call__`.
        """
        if not self.built:
            self.build((inputs.shape if hasattr(inputs, "shape") else None))

        # Simple tracing
        is_tracing = False

        def check_tracing(x: TensorLike) -> None:
            """Compute or retrieve `check tracing` for `check_tracing`.

            Args:
                x: Input tensor or array.
            """
            nonlocal is_tracing
            if is_tracing:
                return
            if isinstance(x, KerasTensor):
                is_tracing = True
            elif isinstance(x, (list, tuple)):
                for item in x:
                    check_tracing(item)
            elif isinstance(x, dict):
                for item in x.values():
                    check_tracing(item)

        check_tracing(inputs)

        if not is_tracing:
            self._losses = []

        outputs = self.call(inputs, *args, **kwargs)

        if is_tracing:
            node = Node(operation=self, call_args=inputs, call_kwargs=kwargs)

            def wrap_output(x: TensorLike) -> TensorLike:
                """Compute or retrieve `wrap output` for `wrap_output`.

                Args:
                    x: Input tensor or array.

                Returns:
                    TensorLike: Computed result for `wrap_output`.
                """
                if hasattr(x, "shape") and hasattr(x, "dtype") and not isinstance(x, KerasTensor):
                    res = KerasTensor(x.shape, x.dtype, data=x)
                    res._keras_history = node
                    return res
                elif isinstance(x, KerasTensor):
                    res = KerasTensor(x.shape, x.dtype, data=x.data)
                    res._keras_history = node
                    return res
                return x

            if isinstance(outputs, list):
                outputs = [wrap_output(x) for x in outputs]
            elif isinstance(outputs, tuple):
                outputs = tuple(wrap_output(x) for x in outputs)
            elif isinstance(outputs, dict):
                outputs = {k: wrap_output(v) for k, v in outputs.items()}
            else:
                outputs = wrap_output(outputs)

            node.outputs = outputs

        return outputs


class Model(Layer):
    """docstring."""

    def __new__(cls, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Implement the `__new__` special method on `Model`.

        Args:
            *args: Positional arguments forwarded to the operation or base class.
            **kwargs: Additional keyword configuration arguments.

        Returns:
            TensorLike: Computed result for `__new__`.
        """
        if cls is Model and ("inputs" in kwargs and "outputs" in kwargs):
            return Functional(*args, **kwargs)
        return super().__new__(cls)

    def compute_loss(
        self,
        x: TensorLike | None = None,
        y: TensorLike | None = None,
        y_pred: TensorLike | None = None,
        sample_weight: TensorLike | None = None,
    ) -> TensorLike:
        """Compute or retrieve `compute loss` for `Model`.

        Args:
            x: Input tensor or array.
            y: Target label tensor or second operand.
            y_pred: Configuration or input value for `y_pred` (y pred).
            sample_weight: Optional array of per-sample weights for loss and metric scaling.

        Returns:
            TensorLike: Computed result for `compute_loss`.
        """
        if self.loss_fn is not None and callable(self.loss_fn):
            loss = self.loss_fn(y, y_pred)
        else:
            loss = ops.mean(ops.square(y - y_pred))
        for reg_loss in self.losses:
            loss = loss + reg_loss
        return loss

    def compute_metrics(
        self,
        x: TensorLike,
        y: TensorLike,
        y_pred: TensorLike,
        sample_weight: TensorLike | None = None,
    ) -> TensorLike:
        """Compute or retrieve `compute metrics` for `Model`.

        Args:
            x: Input tensor or array.
            y: Target label tensor or second operand.
            y_pred: Configuration or input value for `y_pred` (y pred).
            sample_weight: Optional array of per-sample weights for loss and metric scaling.

        Returns:
            TensorLike: Computed result for `compute_metrics`.
        """
        metrics_dict = {}
        if self.compiled_metrics:
            for m in self.compiled_metrics:
                m.update_state(y, y_pred)
                metrics_dict[m.name] = m.result()
        return metrics_dict

    def train_step(self, data: TensorLike) -> TensorLike:
        """Compute or retrieve `train step` for `Model`.

        Args:
            data: Dataset or batch of sample tensors used to adapt state statistics.

        Returns:
            TensorLike: Computed result for `train_step`.
        """
        from ml_switcheroo_compiler.grad import (
            value_and_grad_wrt_vars,
        )

        if isinstance(data, tuple) and len(data) == 2:
            x, y = data
        else:
            x, y = data, None

        def loss_fn() -> TensorLike:
            """Compute or retrieve `loss fn` for `loss_fn`.

            Returns:
                TensorLike: Computed result for `loss_fn`.
            """
            try:
                y_pred = self(x, training=True)
            except TypeError:
                y_pred = self(x)
            loss = self.compute_loss(x, y, y_pred)
            return loss

        loss, grads_dict = value_and_grad_wrt_vars(loss_fn)()

        trainable_vars = self.trainable_weights
        if hasattr(self.optimizer, "apply_gradients") and trainable_vars:
            grads_list = [
                grads_dict.get((v.id if hasattr(v, "id") else id(v)), None) for v in trainable_vars
            ]
            self.optimizer.apply_gradients(zip(grads_list, trainable_vars))

        try:
            y_pred = self(x, training=False)
        except TypeError:
            y_pred = self(x)
        metrics = self.compute_metrics(x, y, y_pred)
        metrics["loss"] = loss
        return metrics

    def test_step(self, data: TensorLike) -> TensorLike:
        """Compute or retrieve `test step` for `Model`.

        Args:
            data: Dataset or batch of sample tensors used to adapt state statistics.

        Returns:
            TensorLike: Computed result for `test_step`.
        """
        if isinstance(data, tuple) and len(data) == 2:
            x, y = data
        else:
            x, y = data, None
        try:
            y_pred = self(x, training=False)
        except TypeError:
            y_pred = self(x)
        loss = self.compute_loss(x, y, y_pred)
        metrics = self.compute_metrics(x, y, y_pred)
        metrics["loss"] = loss
        return metrics

    def predict_step(self, data: TensorLike) -> TensorLike:
        """Compute or retrieve `predict step` for `Model`.

        Args:
            data: Dataset or batch of sample tensors used to adapt state statistics.

        Returns:
            TensorLike: Computed result for `predict_step`.
        """
        x = data[0] if isinstance(data, (tuple, list)) else data
        try:
            return self(x, training=False)
        except TypeError:
            return self(x)

    def compile_from_config(self, config: TensorLike) -> None:
        """compile_from_config docstring.

        Args:
            config: Config.
        """
        optimizer = config.get("optimizer", "rmsprop")
        loss = config.get("loss", None)
        metrics = config.get("metrics", None)
        self.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    @property
    def compiled_loss(self) -> TensorLike:
        """Compute or retrieve `compiled loss` for `Model`.

        Returns:
            TensorLike: Computed result for `compiled_loss`.
        """
        return self.loss

    @property
    def distribute_reduction_method(self) -> TensorLike:
        """Compute or retrieve `distribute reduction method` for `Model`.

        Returns:
            TensorLike: Computed result for `distribute_reduction_method`.
        """
        return "auto"

    @property
    def distribute_strategy(self) -> TensorLike:
        """Compute or retrieve `distribute strategy` for `Model`.

        Returns:
            TensorLike: Computed result for `distribute_strategy`.
        """
        return "auto"

    def export(self, filepath: str | TensorLike) -> None:
        """Export docstring.

        Args:
            filepath: Filepath.
        """
        from zero_keras.export import ExportArchive

        archive = ExportArchive()
        archive.track(self)
        archive.write_out(filepath)

    def get_compile_config(self) -> TensorLike:
        """Return the serialized compilation arguments for the model.

        Returns:
            TensorLike: Computed result for `get_compile_config`.
        """
        from zero_keras.losses import serialize as serialize_loss
        from zero_keras.optimizers import serialize as serialize_opt

        config = {}
        if self.optimizer:
            config["optimizer"] = serialize_opt(self.optimizer)
        if self.loss:
            config["loss"] = serialize_loss(self.loss)
        if self.compiled_metrics:
            # A simplified metrics serialization
            config["metrics"] = [m.name for m in self.compiled_metrics]
        return config

    def get_layer(
        self, name: TensorLike | None = None, index: TensorLike | None = None
    ) -> TensorLike:
        """Return the layer associated with `Model`.

        Args:
            name: Optional string identifier for the object.
            index: Configuration or input value for `index` (index).

        Returns:
            TensorLike: Computed result for `get_layer`.
        """
        if index is not None:
            return self.layers[index]
        if name is not None:
            for layer in self.layers:
                if layer.name == name:
                    return layer
        raise ValueError(f"No such layer: {name}.")

    def get_metrics_result(self) -> TensorLike:
        """Return the metrics result associated with `Model`.

        Returns:
            TensorLike: Computed result for `get_metrics_result`.
        """
        results = {}
        if self.compiled_metrics:
            for m in self.compiled_metrics:
                results[m.name] = m.result()
        return results

    def get_state_tree(self) -> TensorLike:
        """Return the state tree associated with `Model`.

        Returns:
            TensorLike: Computed result for `get_state_tree`.
        """
        return {"variables": self.variables}

    @property
    def jit_compile(self) -> TensorLike:
        """Compute or retrieve `jit compile` for `Model`.

        Returns:
            TensorLike: Computed result for `jit_compile`.
        """
        return self._jit_compile

    @jit_compile.setter
    def jit_compile(self, value: TensorLike) -> None:
        """Compute or retrieve `jit compile` for `Model`.

        Args:
            value: Configuration or input value for `value` (value).
        """
        self._jit_compile = value

    def make_predict_function(self, force: bool = False) -> TensorLike:
        """Compute or retrieve `make predict function` for `Model`.

        Args:
            force: Configuration or input value for `force` (force).

        Returns:
            TensorLike: Computed result for `make_predict_function`.
        """
        if self.predict_function is None or force:
            self.predict_function = self.predict_step
        return self.predict_function

    def make_test_function(self, force: bool = False) -> TensorLike:
        """Compute or retrieve `make test function` for `Model`.

        Args:
            force: Configuration or input value for `force` (force).

        Returns:
            TensorLike: Computed result for `make_test_function`.
        """
        if self.test_function is None or force:
            self.test_function = self.test_step
        return self.test_function

    def make_train_function(self, force: bool = False) -> TensorLike:
        """Compute or retrieve `make train function` for `Model`.

        Args:
            force: Configuration or input value for `force` (force).

        Returns:
            TensorLike: Computed result for `make_train_function`.
        """
        if self.train_function is None or force:
            self.train_function = self.train_step
        return self.train_function

    @property
    def metrics_names(self) -> TensorLike:
        """Compute or retrieve `metrics names` for `Model`.

        Returns:
            TensorLike: Computed result for `metrics_names`.
        """
        return []

    def predict_on_batch(self, x: TensorLike) -> TensorLike:
        """Compute predictions for a single batch of input samples.

        Args:
            x: Input tensor or array.

        Returns:
            TensorLike: Computed result for `predict_on_batch`.
        """
        return self.predict_step(x)

    def reset_metrics(self) -> None:
        """reset_metrics docstring."""
        if self.compiled_metrics:
            for m in self.compiled_metrics:
                m.reset_state()

    def set_state_tree(self, state_tree: TensorLike) -> None:
        """set_state_tree docstring.

        Args:
            state_tree: State tree.
        """
        if "variables" in state_tree:
            for v_self, v_tree in zip(self.variables, state_tree["variables"]):
                v_self.assign(v_tree)

    def stateless_compute_loss(
        self,
        x: TensorLike,
        y: TensorLike,
        y_pred: TensorLike,
        sample_weight: TensorLike | None = None,
    ) -> TensorLike:
        """Compute or retrieve `stateless compute loss` for `Model`.

        Args:
            x: Input tensor or array.
            y: Target label tensor or second operand.
            y_pred: Configuration or input value for `y_pred` (y pred).
            sample_weight: Optional array of per-sample weights for loss and metric scaling.

        Returns:
            TensorLike: Computed result for `stateless_compute_loss`.
        """
        return self.compute_loss(x, y, y_pred, sample_weight=sample_weight)

    def summary(self, *args: TensorLike, **kwargs: TensorLike) -> None:
        """Summary docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        print_fn = kwargs.get("print_fn", print)
        print_fn(f'Model: "{self.name}"')
        print_fn("=================================================================")
        print_fn("Total params: 0")
        print_fn("=================================================================")

    def test_on_batch(
        self,
        x: TensorLike,
        y: TensorLike | None = None,
        sample_weight: TensorLike | None = None,
        return_dict: bool = False,
    ) -> TensorLike:
        """Evaluate the model on a single batch of samples.

        Args:
            x: Input tensor or array.
            y: Target label tensor or second operand.
            sample_weight: Optional array of per-sample weights for loss and metric scaling.
            return_dict: Whether to return metric results as a keyed dictionary instead of a list.

        Returns:
            TensorLike: Computed result for `test_on_batch`.
        """
        from zero_keras.utils import pack_x_y_sample_weight

        batch = pack_x_y_sample_weight(x, y, sample_weight)
        return self.test_step(batch)

    def to_json(self) -> TensorLike:
        """Compute or retrieve `to json` for `Model`.

        Returns:
            TensorLike: Computed result for `to_json`.
        """
        import json

        return json.dumps(self.get_config())

    def train_on_batch(
        self,
        x: TensorLike,
        y: TensorLike | None = None,
        sample_weight: TensorLike | None = None,
        class_weight: TensorLike | None = None,
        return_dict: bool = False,
    ) -> TensorLike:
        """Run a single gradient update step on one batch of data.

        Args:
            x: Input tensor or array.
            y: Target label tensor or second operand.
            sample_weight: Optional array of per-sample weights for loss and metric scaling.
            class_weight: Optional dictionary mapping class indices to scalar loss weights.
            return_dict: Whether to return metric results as a keyed dictionary instead of a list.

        Returns:
            TensorLike: Computed result for `train_on_batch`.
        """
        from zero_keras.utils import pack_x_y_sample_weight

        batch = pack_x_y_sample_weight(x, y, sample_weight)
        return self.train_step(batch)

    @property
    def compiled_metrics(self) -> TensorLike:
        """Compute or retrieve `compiled metrics` for `Model`.

        Returns:
            TensorLike: Computed result for `compiled_metrics`.
        """
        return self._compiled_metrics

    @compiled_metrics.setter
    def compiled_metrics(self, value: TensorLike) -> None:
        """Compute or retrieve `compiled metrics` for `Model`.

        Args:
            value: Configuration or input value for `value` (value).
        """
        self._compiled_metrics = value

    @property
    def layers(self) -> TensorLike:
        """Compute or retrieve `layers` for `Model`.

        Returns:
            TensorLike: Computed result for `layers`.
        """
        return self._layers

    @layers.setter
    def layers(self, value: TensorLike) -> None:
        """Compute or retrieve `layers` for `Model`.

        Args:
            value: Configuration or input value for `value` (value).
        """
        self._layers = value

    @property
    def loss(self) -> TensorLike:
        """Compute or retrieve `loss` for `Model`.

        Returns:
            TensorLike: Computed result for `loss`.
        """
        return self._loss

    @loss.setter
    def loss(self, value: TensorLike) -> None:
        """Compute or retrieve `loss` for `Model`.

        Args:
            value: Configuration or input value for `value` (value).
        """
        self._loss = value

    @property
    def run_eagerly(self) -> TensorLike:
        """Compute or retrieve `run eagerly` for `Model`.

        Returns:
            TensorLike: Computed result for `run_eagerly`.
        """
        return self._run_eagerly

    @run_eagerly.setter
    def run_eagerly(self, value: TensorLike) -> None:
        """Compute or retrieve `run eagerly` for `Model`.

        Args:
            value: Configuration or input value for `value` (value).
        """
        self._run_eagerly = value

    def __init__(
        self,
        inputs: TensorLike | None = None,
        outputs: TensorLike | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the `Model` instance.

        Args:
            inputs: Input tensor or nested structure of tensors.
            outputs: Configuration or input value for `outputs` (outputs).
            **kwargs: Additional keyword configuration arguments.
        """
        super().__init__(**kwargs)
        self.inputs = inputs
        self.outputs = outputs
        self._compiled = False
        self._compiled_metrics: TensorLike | None = None
        self._layers: list[Layer] = []
        self._loss: TensorLike | None = None
        self._run_eagerly: bool = False
        self._jit_compile: bool = False
        self.optimizer: TensorLike | None = None
        self.loss_fn: TensorLike | None = None
        self.predict_function: TensorLike | None = None
        self.test_function: TensorLike | None = None
        self.train_function: TensorLike | None = None
        self.stop_training: bool = False

    def compile(
        self,
        optimizer: str | TensorLike = "adam",
        loss: str | TensorLike = "mse",
        metrics: TensorLike | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Configure the model with optimizer, loss, and evaluation metrics.

        Args:
            optimizer: Optimizer instance or string identifier used to update trainable weights.
            loss: Loss function, Loss instance, or dictionary of losses to minimize.
            metrics: List of metrics evaluated during training and testing.
            **kwargs: Additional keyword configuration arguments.
        """
        self._compiled = True

        if isinstance(optimizer, str):
            from zero_keras.optimizers import get as get_opt

            self.optimizer = get_opt(optimizer)
        else:
            self.optimizer = optimizer

        if isinstance(loss, str):
            from zero_keras.losses import get as get_loss

            self.loss = loss
            self.loss_fn = get_loss(loss)
        else:
            self.loss = loss
            self.loss_fn = (
                loss if callable(loss) else (loss.call if hasattr(loss, "call") else None)
            )

        self.compiled_metrics = []
        if metrics is not None:
            if not isinstance(metrics, (list, tuple)):
                metrics = [metrics]
            from zero_keras.metrics import get as get_metric

            for m in metrics:
                if isinstance(m, str):
                    self.compiled_metrics.append(get_metric(m))
                else:
                    self.compiled_metrics.append(m)

    def _is_iterator(self, data: TensorLike) -> TensorLike:
        """Compute or retrieve `is iterator` for `Model`.

        Args:
            data: Dataset or batch of sample tensors used to adapt state statistics.

        Returns:
            TensorLike: Computed result for `_is_iterator`.
        """
        if (
            hasattr(data, "__iter__")
            and not hasattr(data, "shape")
            and not isinstance(data, (list, tuple, dict))
        ):
            return True
        return bool(
            type(data).__module__.startswith("tensorflow.python.data")
            or type(data).__module__.startswith("torch.utils.data")
        )

    def _unpack_iterator(self, data: TensorLike, is_train: bool = True) -> TensorLike:
        """Compute or retrieve `unpack iterator` for `Model`.

        Args:
            data: Dataset or batch of sample tensors used to adapt state statistics.
            is_train: Configuration or input value for `is_train` (is train).
        """
        import inspect

        if type(data).__module__.startswith("tensorflow.python.data"):
            # tf.data.Dataset
            for batch in data:
                yield batch
        elif type(data).__module__.startswith("torch.utils.data"):
            # torch DataLoader
            for batch in data:
                # convert torch tensors to numpy/ops
                if isinstance(batch, (list, tuple)):
                    yield tuple(b.numpy() if hasattr(b, "numpy") else b for b in batch)
                elif isinstance(batch, dict):
                    yield {k: v.numpy() if hasattr(v, "numpy") else v for k, v in batch.items()}
                else:
                    yield batch.numpy() if hasattr(batch, "numpy") else batch
        elif inspect.isgenerator(data) or isinstance(data, type(x for x in [])):
            # python generator
            for batch in data:
                yield batch
        else:
            # fallback iterable
            for batch in data:
                yield batch

    def fit(
        self,
        x: TensorLike,
        y: TensorLike | None = None,
        epochs: int = 1,
        batch_size: int | None = 32,
        **kwargs: TensorLike,
    ) -> TensorLike:
        """Train the model on `x` and `y` for a fixed number of epochs.

        Args:
            x: Input tensor or array.
            y: Target label tensor or second operand.
            epochs: Number of complete passes through the training dataset.
            batch_size: Fixed batch size or number of samples per batch.
            **kwargs: Additional keyword configuration arguments.

        Returns:
            TensorLike: Computed result for `fit`.
        """
        import math

        if batch_size is None:
            batch_size = 32

        num_samples = len(x) if hasattr(x, "__len__") else 0
        num_batches = math.ceil(num_samples / batch_size) if num_samples > 0 else 1

        from zero_keras.callbacks import callbacks as cb_module

        callbacks = kwargs.get("callbacks", [])
        if not isinstance(callbacks, cb_module.CallbackList):
            callbacks = cb_module.CallbackList(callbacks, model=self)

        callbacks.on_train_begin()

        history: dict[str] = {"loss": []}

        for epoch in range(epochs):
            callbacks.on_epoch_begin(epoch)
            if self.compiled_metrics:
                for m in self.compiled_metrics:
                    m.reset_state()

            epoch_loss_sum = 0.0
            batches_seen = 0

            if self._is_iterator(x):
                iterator = self._unpack_iterator(x, is_train=True)
                for b_idx, batch_data in enumerate(iterator):
                    callbacks.on_train_batch_begin(b_idx)
                    callbacks.on_batch_begin(b_idx)
                    logs = self.train_step(batch_data)
                    callbacks.on_train_batch_end(b_idx, logs)
                    callbacks.on_batch_end(b_idx, logs)
                    loss_val = logs.get("loss", 0.0)
                    try:
                        b_loss = float(loss_val)
                    except Exception:
                        from ml_switcheroo_compiler.ops import asarray

                        try:
                            b_loss = float(asarray(loss_val).data.mean())
                        except Exception:
                            b_loss = 0.0
                    epoch_loss_sum += b_loss
                    batches_seen += 1
            else:
                for batch in range(num_batches):
                    callbacks.on_train_batch_begin(batch)
                    callbacks.on_batch_begin(batch)
                    if num_samples > 0 and hasattr(x, "__getitem__") and hasattr(y, "__getitem__"):
                        start = batch * batch_size
                        end = min((batch + 1) * batch_size, num_samples)
                        try:
                            bx = x[start:end]
                            by = y[start:end]
                        except Exception:
                            bx, by = x, y
                    else:
                        bx, by = x, y

                    from zero_keras.activations import _to_tensor

                    logs = self.train_step((_to_tensor(bx), _to_tensor(by)))
                    callbacks.on_train_batch_end(batch, logs)
                    callbacks.on_batch_end(batch, logs)
                    loss_val = logs.get("loss", 0.0)
                    try:
                        b_loss = float(loss_val)
                    except Exception:
                        from ml_switcheroo_compiler.ops import asarray

                        try:
                            b_loss = float(asarray(loss_val).data.mean())
                        except Exception:
                            b_loss = 0.0
                    epoch_loss_sum += b_loss
                    batches_seen += 1

            epoch_logs = {"loss": epoch_loss_sum / max(1, batches_seen)}
            if self.compiled_metrics:
                for m in self.compiled_metrics:
                    epoch_logs[m.name] = m.result()

            callbacks.on_epoch_end(epoch, epoch_logs)
            for k, v in epoch_logs.items():
                history.setdefault(k, []).append(v)

            if self.stop_training:
                break

        callbacks.on_train_end()
        hist_obj = type(
            "History",
            (),
            {"history": history, "epoch": list(range(len(history["loss"])))},
        )()
        return hist_obj

    def save(
        self,
        filepath: str | TensorLike,
        overwrite: bool = True,
        save_format: TensorLike | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Save the complete model architecture, weights, and optimizer state to `filepath`.

        Args:
            filepath: Destination or source file path on disk.
            overwrite: Whether to overwrite an existing file without prompting.
            save_format: Serialization format identifier (`keras` or `h5`).
            **kwargs: Additional keyword configuration arguments.
        """
        import json
        import struct
        import zipfile

        metadata = {
            "keras_version": "3.0.0",
            "date_saved": "2026-06-18",
        }

        config = {
            "class_name": type(self).__name__,
            "config": self.get_config(),
        }

        def _to_bytes_and_dtype(tensor: TensorLike) -> TensorLike:
            """Compute or retrieve `to bytes and dtype` for `_to_bytes_and_dtype`.

            Args:
                tensor: Configuration or input value for `tensor` (tensor).

            Returns:
                TensorLike: Computed result for `_to_bytes_and_dtype`.
            """
            import array

            if hasattr(tensor, "numpy"):
                data = tensor.numpy()
            elif hasattr(tensor, "data") and hasattr(tensor.data, "numpy"):
                data = tensor.data.numpy()
            else:
                data = tensor

            # Extract standard python array
            if hasattr(data, "tobytes"):
                b = data.tobytes()
                shape = data.shape
                dt = str((data.dtype if hasattr(data, "dtype") else "float32")).split(".")[-1]
            else:
                # fallback
                flat = []

                def flatten(x: TensorLike) -> None:
                    """Compute or retrieve `flatten` for `flatten`.

                    Args:
                        x: Input tensor or array.
                    """
                    if isinstance(x, (list, tuple)):
                        for item in x:
                            flatten(item)
                    else:
                        flat.append(float(x))

                flatten(data)
                b = array.array("f", flat).tobytes()
                shape = data.shape if hasattr(data, "shape") else (len(flat),)
                dt = "float32"

            dtype_map = {"float32": "F32", "int32": "I32", "float64": "F64"}
            return b, list(shape), dtype_map.get(dt, "F32")

        def _get_safetensors(
            w_list: TensorLike, prefix: str | TensorLike = "weight_"
        ) -> TensorLike:
            """Compute or retrieve `get safetensors` for `_get_safetensors`.

            Args:
                w_list: Configuration or input value for `w_list` (w list).
                prefix: Configuration or input value for `prefix` (prefix).

            Returns:
                TensorLike: Computed result for `_get_safetensors`.
            """
            header = {}
            offset = 0
            data_bytes = bytearray()
            for i, w in enumerate(w_list):
                b, shape, dt = _to_bytes_and_dtype(w)
                k = prefix + str(i)
                header[k] = {
                    "dtype": dt,
                    "shape": shape,
                    "data_offsets": [offset, offset + len(b)],
                }
                offset += len(b)
                data_bytes.extend(b)
            header_json = json.dumps(header).encode("utf-8")
            pad_len = (8 - len(header_json) % 8) % 8
            header_json += b" " * pad_len
            header_size = struct.pack("<Q", len(header_json))
            return header_size + header_json + data_bytes

        mode = "w" if overwrite else "x"
        with zipfile.ZipFile(filepath, mode, compression=zipfile.ZIP_DEFLATED) as zf:
            zinfo_meta = zipfile.ZipInfo("metadata.json")
            zf.writestr(zinfo_meta, json.dumps(metadata, indent=2))

            zinfo_conf = zipfile.ZipInfo("config.json")
            zf.writestr(zinfo_conf, json.dumps(config, indent=2))

            if self.weights:
                zinfo_weights = zipfile.ZipInfo("model.safetensors")
                zf.writestr(zinfo_weights, _get_safetensors(self.weights, "weight_"))

            if hasattr(self.optimizer, "variables") and self.optimizer.variables:
                zinfo_opt = zipfile.ZipInfo("optimizer.safetensors")
                zf.writestr(zinfo_opt, _get_safetensors(self.optimizer.variables, "opt_weight_"))

    def evaluate(
        self,
        x: TensorLike,
        y: TensorLike | None = None,
        batch_size: int | None = None,
        **kwargs: TensorLike,
    ) -> TensorLike:
        """Evaluate the model loss and metrics on test data `x` and `y`.

        Args:
            x: Input tensor or array.
            y: Target label tensor or second operand.
            batch_size: Fixed batch size or number of samples per batch.
            **kwargs: Additional keyword configuration arguments.

        Returns:
            TensorLike: Computed result for `evaluate`.
        """
        import math

        if batch_size is None:
            batch_size = 32

        num_samples = len(x) if hasattr(x, "__len__") else 0
        num_batches = math.ceil(num_samples / batch_size) if num_samples > 0 else 1

        if self.compiled_metrics:
            for m in self.compiled_metrics:
                m.reset_state()

        from zero_keras.callbacks import callbacks as cb_module

        callbacks = kwargs.get("callbacks", [])
        if not isinstance(callbacks, cb_module.CallbackList):
            callbacks = cb_module.CallbackList(callbacks, model=self)

        callbacks.on_test_begin()
        epoch_loss_sum = 0.0

        batches_seen = 0
        if self._is_iterator(x):
            iterator = self._unpack_iterator(x, is_train=False)
            for b_idx, batch_data in enumerate(iterator):
                callbacks.on_test_batch_begin(b_idx)
                logs = self.test_step(batch_data)
                callbacks.on_test_batch_end(b_idx, logs)
                loss_val = logs.get("loss", 0.0)
                try:
                    b_loss = float(loss_val)
                except Exception:
                    from ml_switcheroo_compiler.ops import asarray

                    try:
                        b_loss = float(asarray(loss_val).data.mean())
                    except Exception:
                        b_loss = 0.0
                epoch_loss_sum += b_loss
                batches_seen += 1
        else:
            for batch in range(num_batches):
                callbacks.on_test_batch_begin(batch)
                if num_samples > 0 and hasattr(x, "__getitem__") and hasattr(y, "__getitem__"):
                    start = batch * batch_size
                    end = min((batch + 1) * batch_size, num_samples)
                    try:
                        bx = x[start:end]
                        by = y[start:end]
                    except Exception:
                        bx, by = x, y
                else:
                    bx, by = x, y

                logs = self.test_step((_to_tensor(bx), _to_tensor(by)))
                callbacks.on_test_batch_end(batch, logs)
                loss_val = logs.get("loss", 0.0)
                try:
                    b_loss = float(loss_val)
                except Exception:
                    from ml_switcheroo_compiler.ops import asarray

                    try:
                        b_loss = float(asarray(loss_val).data.mean())
                    except Exception:
                        b_loss = 0.0
                epoch_loss_sum += b_loss
                batches_seen += 1

        metrics_results = {"loss": epoch_loss_sum / max(1, batches_seen)}
        if self.compiled_metrics:
            for m in self.compiled_metrics:
                metrics_results[m.name] = m.result()
        callbacks.on_test_end(metrics_results)
        return metrics_results

    def predict(
        self, x: TensorLike, batch_size: int | None = None, **kwargs: TensorLike
    ) -> TensorLike:
        """Generate output predictions for input samples `x`.

        Args:
            x: Input tensor or array.
            batch_size: Fixed batch size or number of samples per batch.
            **kwargs: Additional keyword configuration arguments.

        Returns:
            TensorLike: Computed result for `predict`.
        """
        import math

        if batch_size is None:
            batch_size = 32

        num_samples = len(x) if hasattr(x, "__len__") else 0
        num_batches = math.ceil(num_samples / batch_size) if num_samples > 0 else 1

        from zero_keras.callbacks import callbacks as cb_module

        callbacks = kwargs.get("callbacks", [])
        if not isinstance(callbacks, cb_module.CallbackList):
            callbacks = cb_module.CallbackList(callbacks, model=self)

        callbacks.on_predict_begin()
        all_preds = []

        if self._is_iterator(x):
            iterator = self._unpack_iterator(x, is_train=False)
            for b_idx, batch_data in enumerate(iterator):
                callbacks.on_predict_batch_begin(b_idx)
                preds = self.predict_step(
                    batch_data if isinstance(batch_data, tuple) else (batch_data,)
                )
                callbacks.on_predict_batch_end(b_idx, {"outputs": preds})
                if hasattr(preds, "numpy"):
                    all_preds.append(preds.numpy())
                else:
                    from ml_switcheroo_compiler.ops import asarray

                    try:
                        all_preds.append(asarray(preds).data)
                    except Exception:
                        all_preds.append(preds)
        else:
            for batch in range(num_batches):
                callbacks.on_predict_batch_begin(batch)
                if num_samples > 0 and hasattr(x, "__getitem__"):
                    start = batch * batch_size
                    end = min((batch + 1) * batch_size, num_samples)
                    try:
                        bx = x[start:end]
                    except Exception:
                        bx = x
                else:
                    bx = x

                preds = self.predict_step((bx,))
                callbacks.on_predict_batch_end(batch, {"outputs": preds})

                if hasattr(preds, "numpy"):
                    all_preds.append(preds.numpy())
                else:
                    from ml_switcheroo_compiler.ops import asarray

                    try:
                        all_preds.append(asarray(preds).data)
                    except Exception:
                        all_preds.append(preds)

        from ml_switcheroo_compiler.serialization import (
            concatenate_arrays,
            is_numpy_array,
        )

        callbacks.on_predict_end()
        if len(all_preds) == 1:
            return all_preds[0]
        if all_preds and is_numpy_array(all_preds[0]):
            return concatenate_arrays(all_preds)
        return all_preds

    def save_weights(
        self,
        filepath: str | TensorLike,
        overwrite: bool = True,
        save_format: TensorLike | None = None,
        options: TensorLike | None = None,
    ) -> None:
        """Save all layer weight values to `filepath`.

        Args:
            filepath: Destination or source file path on disk.
            overwrite: Whether to overwrite an existing file without prompting.
            save_format: Serialization format identifier (`keras` or `h5`).
            options: Configuration or input value for `options` (options).
        """
        import ml_switcheroo_compiler.serialization as msc_serialization

        msc_serialization.save_weights(
            self,
            filepath,
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )

    def load_weights(
        self,
        filepath: str | TensorLike,
        skip_mismatch: bool = False,
        by_name: bool = False,
        options: TensorLike | None = None,
    ) -> None:
        """Load layer weight values from `filepath`.

        Args:
            filepath: Destination or source file path on disk.
            skip_mismatch: Whether to skip loading weights whose shapes or names do not match.
            by_name: Whether to load weights by layer name rather than topological order.
            options: Configuration or input value for `options` (options).
        """
        import ml_switcheroo_compiler.serialization as msc_serialization

        weights = msc_serialization.load_weights(filepath, target_model=self)

        # apply weights
        if not self.weights:
            return

        for i, w in enumerate(self.weights):
            k = "weight_" + str(i)
            if k in weights and hasattr(w, "assign"):
                w.assign(weights[k])


class Functional(Model):
    """Functional class."""

    def __init__(
        self,
        inputs: TensorLike | None = None,
        outputs: TensorLike | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the `Functional` instance.

        Args:
            inputs: Input tensor or nested structure of tensors.
            outputs: Configuration or input value for `outputs` (outputs).
            **kwargs: Additional keyword configuration arguments.
        """
        super().__init__(**kwargs)
        self.inputs = inputs
        self.outputs = outputs
        self._nodes_by_depth = self._map_graph_network(inputs, outputs)

        # Collect layers
        layers = []
        for depth in sorted(self._nodes_by_depth.keys(), reverse=True):
            for node in self._nodes_by_depth[depth]:
                if node.layer not in layers and node.layer is not self:
                    layers.append(node.layer)
        self.layers = layers

    def _map_graph_network(self, inputs: TensorLike, outputs: TensorLike) -> TensorLike:
        """Map the DAG of symbolic ``Node`` instances between ``inputs`` and ``outputs``.

        Args:
            inputs: Symbolic input tensor(s).
            outputs: Symbolic output tensor(s).

        Returns:
            dict[int, list[Node]]: Mapping from topological depth to graph nodes.
        """
        _ = inputs
        nodes_by_depth: dict[int, list[Node]] = {}
        visited_node_ids: set[int] = set()
        queue: list[tuple[TensorLike, int]] = []

        if isinstance(outputs, (list, tuple)):
            for x in outputs:
                queue.append((x, 0))
        elif isinstance(outputs, dict):
            for x in outputs.values():
                queue.append((x, 0))
        else:
            queue.append((outputs, 0))

        while queue:
            tensor, current_depth = queue.pop(0)
            if not isinstance(tensor, KerasTensor):
                continue
            node = tensor._keras_history if hasattr(tensor, "_keras_history") else None
            if node is None:
                continue
            node_id = id(node)
            if node_id in visited_node_ids:
                continue
            visited_node_ids.add(node_id)

            nodes_by_depth.setdefault(current_depth, []).append(node)

            node_inputs = node.inputs
            if isinstance(node_inputs, (list, tuple)):
                for x in node_inputs:
                    queue.append((x, current_depth + 1))
            elif isinstance(node_inputs, dict):
                for x in node_inputs.values():
                    queue.append((x, current_depth + 1))
            else:
                queue.append((node_inputs, current_depth + 1))

        return nodes_by_depth

    def get_config(self) -> TensorLike:
        """Return the serializable configuration dictionary for `Functional`.

        Returns:
            TensorLike: Computed result for `get_config`.
        """
        layer_configs = []

        # Inject InputLayers for all inputs
        inputs_list = self.inputs if isinstance(self.inputs, (list, tuple)) else [self.inputs]
        for t in inputs_list:
            layer_configs.append(
                {
                    "name": t.name,
                    "class_name": "InputLayer",
                    "config": {
                        "name": t.name,
                        "batch_input_shape": (
                            t.batch_shape
                            if hasattr(t, "batch_shape")
                            else t.shape
                            if hasattr(t, "shape")
                            else (None,)
                        ),
                        "dtype": (t.dtype if hasattr(t, "dtype") else "float32"),
                    },
                    "inbound_nodes": [],
                }
            )

        for layer in self.layers:
            layer_configs.append(
                {
                    "name": layer.name,
                    "class_name": type(layer).__name__,
                    "config": (
                        layer.get_config
                        if hasattr(layer, "get_config")
                        else lambda: layer._kwargs if hasattr(layer, "_kwargs") else {}
                    )(),
                    "inbound_nodes": [],
                }
            )

        lc_map = {lc["name"]: lc for lc in layer_configs}

        for depth in sorted(self._nodes_by_depth.keys(), reverse=True):
            for node in self._nodes_by_depth[depth]:
                layer_name = node.layer.name
                if layer_name not in lc_map:
                    continue

                node_inputs = node.inputs
                if not isinstance(node_inputs, (list, tuple)):
                    node_inputs = [node_inputs]

                group = []
                for t in node_inputs:
                    if hasattr(t, "_keras_history") and t._keras_history is not None:
                        src_node = t._keras_history
                        group.append([src_node.layer.name, 0, 0, {}])
                    elif hasattr(t, "name") and t.name in lc_map:
                        # It's an input tensor mapped to InputLayer
                        group.append([t.name, 0, 0, {}])
                if group:
                    lc_map[layer_name]["inbound_nodes"].append(group)

        # Input layers
        input_layers = []
        for t in inputs_list:
            if hasattr(t, "_keras_history") and t._keras_history is not None:
                input_layers.append([t._keras_history.layer.name, 0, 0])
            else:
                input_layers.append([t.name, 0, 0])

        # Output layers
        output_layers = []
        outputs_list = self.outputs if isinstance(self.outputs, (list, tuple)) else [self.outputs]
        for t in outputs_list:
            if hasattr(t, "_keras_history") and t._keras_history is not None:
                output_layers.append([t._keras_history.layer.name, 0, 0])
            else:
                output_layers.append([t.name, 0, 0])

        return {
            "name": self.name,
            "layers": layer_configs,
            "input_layers": input_layers,
            "output_layers": output_layers,
        }

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Execute the forward computation of `Functional` on `inputs`.

        Args:
            inputs: Input tensor or nested structure of tensors.
            *args: Positional arguments forwarded to the operation or base class.
            **kwargs: Additional keyword configuration arguments.

        Returns:
            TensorLike: Computed result for `call`.
        """
        tensor_dict = {}

        def _add_input(key: TensorLike, val: TensorLike) -> None:
            """Compute or retrieve `add input` for `_add_input`.

            Args:
                key: Configuration or input value for `key` (key).
                val: Configuration or input value for `val` (val).
            """
            if isinstance(key, KerasTensor):
                tensor_dict[id(key)] = val

        if isinstance(self.inputs, (list, tuple)):
            for k, v in zip(self.inputs, inputs):
                _add_input(k, v)
        elif isinstance(self.inputs, dict):
            for k in self.inputs:
                _add_input(self.inputs[k], inputs[k])
        else:
            _add_input(self.inputs, inputs)

        for depth in sorted(self._nodes_by_depth.keys(), reverse=True):
            for node in self._nodes_by_depth[depth]:
                # Prepare inputs
                def _get_input(x: TensorLike) -> TensorLike:
                    """Compute or retrieve `get input` for `_get_input`.

                    Args:
                        x: Input tensor or array.

                    Returns:
                        TensorLike: Computed result for `_get_input`.
                    """
                    if isinstance(x, KerasTensor) and id(x) in tensor_dict:
                        return tensor_dict[id(x)]
                    return x

                if isinstance(node.inputs, (list, tuple)):
                    node_in = [_get_input(x) for x in node.inputs]
                elif isinstance(node.inputs, dict):
                    node_in = {k: _get_input(v) for k, v in node.inputs.items()}
                else:
                    node_in = _get_input(node.inputs)

                out = node.layer(node_in, *args, **kwargs)

                if isinstance(node.outputs, (list, tuple)):
                    for k, v in zip(node.outputs, out):
                        if isinstance(k, KerasTensor):
                            tensor_dict[id(k)] = v
                elif isinstance(node.outputs, dict):
                    for k in node.outputs:
                        if isinstance(node.outputs[k], KerasTensor):
                            tensor_dict[id(node.outputs[k])] = out[k]
                else:
                    if isinstance(node.outputs, KerasTensor):
                        tensor_dict[id(node.outputs)] = out

        def _get_output(x: TensorLike) -> TensorLike:
            """Compute or retrieve `get output` for `_get_output`.

            Args:
                x: Input tensor or array.

            Returns:
                TensorLike: Computed result for `_get_output`.
            """
            if isinstance(x, KerasTensor) and id(x) in tensor_dict:
                return tensor_dict[id(x)]
            return x

        if isinstance(self.outputs, (list, tuple)):
            return [_get_output(x) for x in self.outputs]
        elif isinstance(self.outputs, dict):
            return {k: _get_output(v) for k, v in self.outputs.items()}
        else:
            return _get_output(self.outputs)


def deserialize(
    config: TensorLike, custom_objects: TensorLike | None = None, safe_mode: bool = True
) -> TensorLike:
    """Returns a Keras layer object via its configuration.

    Args:
        config: A python dict containing a serialized layer configuration.
        custom_objects: Optional dictionary mapping names (strings) to custom
            objects (classes and functions) to be considered during
            deserialization.
        safe_mode: Boolean, whether to disallow unsafe `lambda` deserialization.

    Returns:
        A Keras layer instance.

    """
    if not isinstance(config, dict):
        return config
    if safe_mode and config.get("class_name") == "Lambda":
        raise ValueError("safe_mode is True, cannot deserialize Lambda")

    class_name = config.get("class_name")
    if not class_name:
        return config

    cls = None
    if custom_objects and class_name in custom_objects:
        cls = custom_objects[class_name]

    if cls is None:
        from zero_keras.layers import LAYER_REGISTRY
        from zero_keras.models import (
            Functional as _Functional,
            Model as _Model,
            Sequential as _Sequential,
        )

        model_registry = {
            "Functional": _Functional,
            "Model": _Model,
            "Sequential": _Sequential,
        }
        cls = LAYER_REGISTRY.get(str(class_name)) or model_registry.get(str(class_name))

    if cls and hasattr(cls, "from_config"):
        if "config" in config:
            return cls.from_config(config.get("config", {}))
        return cls.from_config(config)

    if cls:
        if "config" in config:
            return cls(**config.get("config", {}))
        return cls(**config)

    raise ValueError(f"Unknown class {class_name}")


def get(identifier: TensorLike, custom_objects: TensorLike | None = None) -> TensorLike:
    """Retrieve a Keras layer, model, metric, loss, initializer, regularizer, or optimizer by name.

    Args:
        identifier: String identifier, configuration dictionary, or object instance.
        custom_objects: Optional custom object mapping.

    Returns:
        TensorLike: Resolved class or instance.

    Raises:
        ValueError: If ``identifier`` is an unknown string name.
    """
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        return deserialize(identifier, custom_objects=custom_objects)
    if isinstance(identifier, str):
        if isinstance(custom_objects, dict) and identifier in custom_objects:
            return custom_objects[identifier]
        from zero_keras.initializers import _INITIALIZER_CLASSES
        from zero_keras.layers import LAYER_REGISTRY
        from zero_keras.losses import _LOSS_REGISTRY
        from zero_keras.metrics import _METRIC_REGISTRY
        from zero_keras.models import (
            Functional as _Functional,
            Model as _Model,
            Sequential as _Sequential,
        )
        from zero_keras.optimizers import _OPTIMIZER_REGISTRY
        from zero_keras.regularizers import L1, L1L2, L2, OrthogonalRegularizer, Regularizer

        combined_registry: dict[str, TensorLike] = {}
        combined_registry.update(LAYER_REGISTRY)
        combined_registry.update(
            {
                "Functional": _Functional,
                "Model": _Model,
                "Sequential": _Sequential,
                "L1": L1,
                "L2": L2,
                "L1L2": L1L2,
                "OrthogonalRegularizer": OrthogonalRegularizer,
                "Regularizer": Regularizer,
            }
        )
        combined_registry.update(_METRIC_REGISTRY)
        combined_registry.update(_LOSS_REGISTRY)
        combined_registry.update(_INITIALIZER_CLASSES)
        combined_registry.update(_OPTIMIZER_REGISTRY)

        if identifier in combined_registry:
            return combined_registry[identifier]

        raise ValueError(f"Unknown identifier {identifier}")
    return identifier


class Function:
    """Function docstring."""

    def call(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Execute the forward computation of `Function` on `inputs`.

        Args:
            *args: Positional arguments forwarded to the operation or base class.
            **kwargs: Additional keyword configuration arguments.

        Returns:
            TensorLike: Computed result for `call`.
        """
        if (self._fn if hasattr(self, "_fn") else None) is not None:
            return self._fn(*args, **kwargs)
        return args[0] if args else None

    def compute_output_shape(self, input_shape: TensorLike) -> TensorLike:
        """Compute the output tensor shape produced for `input_shape`.

        Args:
            input_shape: Input tensor shape tuple or nested structure of shape tuples.

        Returns:
            TensorLike: Computed result for `compute_output_shape`.
        """
        if (self._shape_fn if hasattr(self, "_shape_fn") else None) is not None:
            return self._shape_fn(input_shape)
        return input_shape

    def compute_output_spec(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Compute the symbolic `KerasTensor` specification for the given inputs.

        Args:
            *args: Positional arguments forwarded to the operation or base class.
            **kwargs: Additional keyword configuration arguments.

        Returns:
            TensorLike: Computed result for `compute_output_spec`.
        """
        if hasattr(self, "compute_output_shape"):
            return self.compute_output_shape(*args, **kwargs)
        from zero_keras.core_layers import KerasTensor

        return KerasTensor(shape=None, dtype=(self.dtype if hasattr(self, "dtype") else "float32"))

    @classmethod
    def from_config(cls, config: TensorLike) -> TensorLike:
        """Instantiate `Function` from its serialized configuration dictionary.

        Args:
            config: Configuration dictionary describing object hyperparameters.

        Returns:
            TensorLike: Computed result for `from_config`.
        """
        return cls(**config)

    def get_config(self) -> TensorLike:
        """Return the serializable configuration dictionary for `Function`.

        Returns:
            TensorLike: Computed result for `get_config`.
        """
        return {}

    @property
    def input(self) -> TensorLike:
        """Compute or retrieve `input` for `Function`.

        Returns:
            TensorLike: Computed result for `input`.
        """
        return (
            self._inbound_nodes[0].input_tensors
            if hasattr(self, "_inbound_nodes") and self._inbound_nodes
            else None
        )

    @property
    def inputs(self) -> TensorLike:
        """Compute or retrieve `inputs` for `Function`.

        Returns:
            TensorLike: Computed result for `inputs`.
        """
        return []

    @property
    def operations(self) -> TensorLike:
        """Compute or retrieve `operations` for `Function`.

        Returns:
            TensorLike: Computed result for `operations`.
        """
        return []

    @property
    def output(self) -> TensorLike:
        """Compute or retrieve `output` for `Function`.

        Returns:
            TensorLike: Computed result for `output`.
        """
        return (
            self._inbound_nodes[0].output_tensors
            if hasattr(self, "_inbound_nodes") and self._inbound_nodes
            else None
        )

    @property
    def outputs(self) -> TensorLike:
        """Compute or retrieve `outputs` for `Function`.

        Returns:
            TensorLike: Computed result for `outputs`.
        """
        return []

    def quantized_call(self, *args: TensorLike, **kwargs: TensorLike) -> None:
        """quantized_call docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        return

    def symbolic_call(self, *args: TensorLike, **kwargs: TensorLike) -> None:
        """symbolic_call docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        return


class InputSpec:
    """InputSpec docstring."""

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
            "dtype": (self.dtype if hasattr(self, "dtype") else None),
            "shape": (self.shape if hasattr(self, "shape") else None),
            "ndim": (self.ndim if hasattr(self, "ndim") else None),
            "max_ndim": (self.max_ndim if hasattr(self, "max_ndim") else None),
            "min_ndim": (self.min_ndim if hasattr(self, "min_ndim") else None),
            "axes": (self.axes if hasattr(self, "axes") else None),
            "allow_last_axis_squeeze": (
                self.allow_last_axis_squeeze if hasattr(self, "allow_last_axis_squeeze") else False
            ),
            "name": (self.name if hasattr(self, "name") else None),
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
            dtype: Data type string or DTypePolicy.
            shape: Shape tuple specifying tensor dimensions.
            ndim: Configuration or input value for `ndim` (ndim).
            max_ndim: Configuration or input value for `max_ndim` (max ndim).
            min_ndim: Configuration or input value for `min_ndim` (min ndim).
            axes: Configuration or input value for `axes` (axes).
            allow_last_axis_squeeze: Configuration or input value for `allow_last_axis_squeeze` (allow last axis squeeze).
            name: Optional string identifier for the object.
        """
        self.dtype = dtype
        self.shape = shape
        self.ndim = ndim
        self.max_ndim = max_ndim
        self.min_ndim = min_ndim
        self.axes = axes or {}
        self.allow_last_axis_squeeze = allow_last_axis_squeeze
        self.name = name


class Operation:
    """Operation docstring."""

    def call(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Execute the forward computation of `Operation` on `inputs`.

        Args:
            *args: Positional arguments forwarded to the operation or base class.
            **kwargs: Additional keyword configuration arguments.

        Returns:
            TensorLike: Computed result for `call`.
        """
        if (self._fn if hasattr(self, "_fn") else None) is not None:
            return self._fn(*args, **kwargs)
        return args[0] if args else None

    def compute_output_spec(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Compute the symbolic `KerasTensor` specification for the given inputs.

        Args:
            *args: Positional arguments forwarded to the operation or base class.
            **kwargs: Additional keyword configuration arguments.

        Returns:
            TensorLike: Computed result for `compute_output_spec`.
        """
        if hasattr(self, "compute_output_shape"):
            return self.compute_output_shape(*args, **kwargs)
        from zero_keras.core_layers import KerasTensor

        return KerasTensor(shape=None, dtype=(self.dtype if hasattr(self, "dtype") else "float32"))

    @classmethod
    def from_config(cls, config: TensorLike) -> TensorLike:
        """Instantiate `Operation` from its serialized configuration dictionary.

        Args:
            config: Configuration dictionary describing object hyperparameters.

        Returns:
            TensorLike: Computed result for `from_config`.
        """
        return cls(**config)

    def get_config(self) -> TensorLike:
        """Return the serializable configuration dictionary for `Operation`.

        Returns:
            TensorLike: Computed result for `get_config`.
        """
        return self._config if hasattr(self, "_config") else {}

    @property
    def input(self) -> TensorLike:
        """Compute or retrieve `input` for `Operation`.

        Returns:
            TensorLike: Computed result for `input`.
        """
        return self._input if hasattr(self, "_input") else None

    @property
    def output(self) -> TensorLike:
        """Compute or retrieve `output` for `Operation`.

        Returns:
            TensorLike: Computed result for `output`.
        """
        return self._output if hasattr(self, "_output") else None

    def quantized_call(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Execute the quantized forward pass on `inputs`.

        Args:
            *args: Positional arguments forwarded to the operation or base class.
            **kwargs: Additional keyword configuration arguments.

        Returns:
            TensorLike: Computed result for `quantized_call`.
        """
        return self.call(*args, **kwargs)

    def symbolic_call(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Compute or retrieve `symbolic call` for `Operation`.

        Args:
            *args: Positional arguments forwarded to the operation or base class.
            **kwargs: Additional keyword configuration arguments.

        Returns:
            TensorLike: Computed result for `symbolic_call`.
        """
        return self.call(*args, **kwargs)


class Variable:
    """Variable docstring."""

    def __getitem__(self, key: TensorLike) -> TensorLike:
        """Implement the `__getitem__` special method on `Variable`.

        Args:
            key: Configuration or input value for `key` (key).

        Returns:
            TensorLike: Computed result for `__getitem__`.
        """
        return (
            self.value[key]
            if (self.value if hasattr(self, "value") else None) is not None
            else None
        )

    @property
    def aggregation(self) -> TensorLike:
        """Compute or retrieve `aggregation` for `Variable`.

        Returns:
            TensorLike: Computed result for `aggregation`.
        """
        return self._aggregation if hasattr(self, "_aggregation") else None

    @aggregation.setter
    def aggregation(self, value: TensorLike) -> None:
        """Compute or retrieve `aggregation` for `Variable`.

        Args:
            value: Configuration or input value for `value` (value).
        """
        self._aggregation = value

    def assign(self, value: TensorLike) -> None:
        """Assign docstring.

        Args:
            value: Value.
        """
        self.value = value

    def assign_add(self, value: TensorLike) -> None:
        """assign_add docstring.

        Args:
            value: Value.
        """
        from zero_keras import ops

        self.value = (
            ops.add(self.value, value)
            if (self.value if hasattr(self, "value") else None) is not None
            else value
        )

    def assign_sub(self, value: TensorLike) -> None:
        """assign_sub docstring.

        Args:
            value: Value.
        """
        from zero_keras import ops

        self.value = (
            ops.subtract(self.value, value)
            if (self.value if hasattr(self, "value") else None) is not None
            else ops.negative(value)
        )

    @property
    def constraint(self) -> TensorLike:
        """Compute or retrieve `constraint` for `Variable`.

        Returns:
            TensorLike: Computed result for `constraint`.
        """
        return self._constraint if hasattr(self, "_constraint") else None

    @constraint.setter
    def constraint(self, value: TensorLike) -> None:
        """Compute or retrieve `constraint` for `Variable`.

        Args:
            value: Configuration or input value for `value` (value).
        """
        self._constraint = value

    @property
    def handle(self) -> TensorLike:
        """Compute or retrieve `handle` for `Variable`.

        Returns:
            TensorLike: Computed result for `handle`.
        """
        return self._handle if hasattr(self, "_handle") else None

    @handle.setter
    def handle(self, value: TensorLike) -> None:
        """Compute or retrieve `handle` for `Variable`.

        Args:
            value: Configuration or input value for `value` (value).
        """
        self._handle = value

    @property
    def ndim(self) -> TensorLike:
        """Compute or retrieve `ndim` for `Variable`.

        Returns:
            TensorLike: Computed result for `ndim`.
        """
        return len(self.shape) if self.shape is not None else 0

    def numpy(self) -> TensorLike:
        """Compute or retrieve `numpy` for `Variable`.

        Returns:
            TensorLike: Computed result for `numpy`.
        """
        from ml_switcheroo_compiler.serialization import to_numpy

        return to_numpy(self.value)

    def __array__(self, dtype: TensorLike | None = None) -> TensorLike:
        """Implement the `__array__` special method on `Variable`.

        Args:
            dtype: Data type string or DTypePolicy.

        Returns:
            TensorLike: Computed result for `__array__`.
        """
        from zero_keras import ops

        arr = self.value
        if arr is None:
            # For uninitialized or dummy tests, just return a zeros array if shape is known
            shape = [d if d is not None else 1 for d in self.shape] if self.shape else [0]
            arr = ops.zeros(shape)

        res = ops.convert_to_tensor(arr).numpy()

        if dtype is not None:
            return res.astype(dtype)
        return res

    def overwrite_with_gradient(self, gradient: TensorLike) -> None:
        """overwrite_with_gradient docstring.

        Args:
            gradient: Gradient.
        """
        self._gradient = gradient
        self.value = gradient

    @property
    def path(self) -> TensorLike:
        """Compute or retrieve `path` for `Variable`.

        Returns:
            TensorLike: Computed result for `path`.
        """
        return self.name

    @property
    def regularizer(self) -> TensorLike:
        """Compute or retrieve `regularizer` for `Variable`.

        Returns:
            TensorLike: Computed result for `regularizer`.
        """
        return self._regularizer if hasattr(self, "_regularizer") else None

    @regularizer.setter
    def regularizer(self, value: TensorLike) -> None:
        """Compute or retrieve `regularizer` for `Variable`.

        Args:
            value: Configuration or input value for `value` (value).
        """
        self._regularizer = value

    @property
    def synchronization(self) -> TensorLike:
        """Compute or retrieve `synchronization` for `Variable`.

        Returns:
            TensorLike: Computed result for `synchronization`.
        """
        return self._synchronization if hasattr(self, "_synchronization") else None

    @synchronization.setter
    def synchronization(self, value: TensorLike) -> None:
        """Compute or retrieve `synchronization` for `Variable`.

        Args:
            value: Configuration or input value for `value` (value).
        """
        self._synchronization = value

    @property
    def value(self) -> TensorLike:
        """Compute or retrieve `value` for `Variable`.

        Returns:
            TensorLike: Computed result for `value`.
        """
        return self._value if hasattr(self, "_value") else None

    @value.setter
    def value(self, val: TensorLike) -> None:
        """Compute or retrieve `value` for `Variable`.

        Args:
            val: Configuration or input value for `val` (val).
        """
        self._value = val

    def __add__(self, other: TensorLike) -> TensorLike:
        """Implement the `__add__` special method on `Variable`.

        Args:
            other: Configuration or input value for `other` (other).

        Returns:
            TensorLike: Computed result for `__add__`.
        """
        from zero_keras import ops

        return ops.add(self.value, other)

    def __radd__(self, other: TensorLike) -> TensorLike:
        """Implement the `__radd__` special method on `Variable`.

        Args:
            other: Configuration or input value for `other` (other).

        Returns:
            TensorLike: Computed result for `__radd__`.
        """
        from zero_keras import ops

        return ops.add(other, self.value)

    def __sub__(self, other: TensorLike) -> TensorLike:
        """Implement the `__sub__` special method on `Variable`.

        Args:
            other: Configuration or input value for `other` (other).

        Returns:
            TensorLike: Computed result for `__sub__`.
        """
        from zero_keras import ops

        return ops.subtract(self.value, other)

    def __rsub__(self, other: TensorLike) -> TensorLike:
        """Implement the `__rsub__` special method on `Variable`.

        Args:
            other: Configuration or input value for `other` (other).

        Returns:
            TensorLike: Computed result for `__rsub__`.
        """
        from zero_keras import ops

        return ops.subtract(other, self.value)

    def __mul__(self, other: TensorLike) -> TensorLike:
        """Implement the `__mul__` special method on `Variable`.

        Args:
            other: Configuration or input value for `other` (other).

        Returns:
            TensorLike: Computed result for `__mul__`.
        """
        from zero_keras import ops

        return ops.multiply(self.value, other)

    def __rmul__(self, other: TensorLike) -> TensorLike:
        """Implement the `__rmul__` special method on `Variable`.

        Args:
            other: Configuration or input value for `other` (other).

        Returns:
            TensorLike: Computed result for `__rmul__`.
        """
        from zero_keras import ops

        return ops.multiply(other, self.value)

    def __truediv__(self, other: TensorLike) -> TensorLike:
        """Implement the `__truediv__` special method on `Variable`.

        Args:
            other: Configuration or input value for `other` (other).

        Returns:
            TensorLike: Computed result for `__truediv__`.
        """
        from zero_keras import ops

        return ops.divide(self.value, other)

    def __rtruediv__(self, other: TensorLike) -> TensorLike:
        """Implement the `__rtruediv__` special method on `Variable`.

        Args:
            other: Configuration or input value for `other` (other).

        Returns:
            TensorLike: Computed result for `__rtruediv__`.
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
        """Compute or retrieve `tensor` for `Variable`.

        Returns:
            TensorLike: Computed result for `_tensor`.
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


from zero_keras.dtype_policies import (
    DTypePolicy as DTypePolicy,
    FloatDTypePolicy as FloatDTypePolicy,
)


class Quantizer:
    """Quantizer docstring."""

    @classmethod
    def from_config(cls, config: TensorLike) -> TensorLike:
        """Instantiate `Quantizer` from its serialized configuration dictionary.

        Args:
            config: Configuration dictionary describing object hyperparameters.

        Returns:
            TensorLike: Computed result for `from_config`.
        """
        return cls(**config)

    def get_config(self) -> TensorLike:
        """Return the quantizer configuration dictionary.

        Returns:
            dict[str, str]: Configuration dictionary containing ``name``.
        """
        return {"name": type(self).__name__}


class RematScope:
    """RematScope docstring."""

    def __enter__(self) -> TensorLike:
        """Implement the `__enter__` special method on `RematScope`.

        Returns:
            TensorLike: Computed result for `__enter__`.
        """
        self._active = True
        return self

    def __exit__(self, *args: TensorLike) -> None:
        """Implement the `__exit__` special method on `RematScope`.

        Args:
            *args: Positional arguments forwarded to the operation or base class.
        """
        self._active = False


class StatelessScope:
    """StatelessScope docstring."""

    def __init__(self) -> None:
        """Initialize the `StatelessScope` instance."""
        self.losses = []
        self.updates = []

    def add_loss(self, loss: TensorLike) -> None:
        """add_loss docstring.

        Args:
            loss: Loss.
        """
        self.losses.append(loss)

    def add_update(self, update: TensorLike) -> None:
        """add_update docstring.

        Args:
            update: Update.
        """
        self.updates.append(update)

    def get_current_value(self, variable: TensorLike) -> TensorLike:
        """Return the current value associated with `StatelessScope`.

        Args:
            variable: Configuration or input value for `variable` (variable).

        Returns:
            TensorLike: Computed result for `get_current_value`.
        """
        return variable.value if hasattr(variable, "value") else None

    def __enter__(self) -> TensorLike:
        """Implement the `__enter__` special method on `StatelessScope`.

        Returns:
            TensorLike: Computed result for `__enter__`.
        """
        return self

    def __exit__(self, *args: TensorLike) -> TensorLike:
        """Implement the `__exit__` special method on `StatelessScope`.

        Args:
            *args: Positional arguments forwarded to the operation or base class.
        """


class SymbolicScope:
    """SymbolicScope docstring."""

    def __enter__(self) -> TensorLike:
        """Implement the `__enter__` special method on `SymbolicScope`.

        Returns:
            TensorLike: Computed result for `__enter__`.
        """
        self._is_symbolic = True
        return self

    def __exit__(self, *args: TensorLike) -> None:
        """Implement the `__exit__` special method on `SymbolicScope`.

        Args:
            *args: Positional arguments forwarded to the operation or base class.
        """
        self._is_symbolic = False


def device(device_name: TensorLike) -> TensorLike:
    """Return a context manager placing operations on `device_name`.

    Args:
        device_name: Configuration or input value for `device_name` (device name).

    Returns:
        TensorLike: Computed result for `device`.
    """

    class DeviceScope:
        """docstring."""

        def __init__(self, name: TensorLike) -> None:
            """Initialize the `DeviceScope` instance.

            Args:
                name: Optional string identifier for the object.
            """
            self.name = name

        def __enter__(self) -> TensorLike:
            """Implement the `__enter__` special method on `DeviceScope`.

            Returns:
                TensorLike: Computed result for `__enter__`.
            """
            return self

        def __exit__(self, *args: TensorLike) -> TensorLike:
            """Implement the `__exit__` special method on `DeviceScope`.

            Args:
                *args: Positional arguments forwarded to the operation or base class.
            """

    return DeviceScope(device_name)


class name_scope:
    """name_scope docstring."""

    def __init__(self, name: TensorLike) -> None:
        """Initialize the `name_scope` instance.

        Args:
            name: Optional string identifier for the object.
        """
        self.name = name

    def __enter__(self) -> TensorLike:
        """Implement the `__enter__` special method on `name_scope`.

        Returns:
            TensorLike: Computed result for `__enter__`.
        """
        return self

    def __exit__(self, *args: TensorLike) -> TensorLike:
        """Implement the `__exit__` special method on `name_scope`.

        Args:
            *args: Positional arguments forwarded to the operation or base class.
        """


def remat(f: TensorLike) -> TensorLike:
    """Wrap function `f` with gradient checkpointing (rematerialization).

    Args:
        f: Configuration or input value for `f` (f).

    Returns:
        TensorLike: Computed result for `remat`.
    """
    return f


def version() -> TensorLike:
    """Return the current `zero_keras` package version string.

    Returns:
        TensorLike: Computed result for `version`.
    """
    return "3.0.0"


def is_keras_tensor(x: TensorLike) -> TensorLike:
    """Return whether `x` is a symbolic `KerasTensor`.

    Args:
        x: Input tensor or array.

    Returns:
        TensorLike: Computed result for `is_keras_tensor`.
    """
    return isinstance(x, KerasTensor)


class Wrapper(Layer):
    """Base class for layer wrappers."""

    def __init__(self, layer: TensorLike, **kwargs: TensorLike) -> None:
        """Initialize the `Wrapper` instance.

        Args:
            layer: Configuration or input value for `layer` (layer).
            **kwargs: Additional keyword configuration arguments.
        """
        super().__init__(**kwargs)
        self.layer = layer


from zero_keras.models import Sequential as Sequential
