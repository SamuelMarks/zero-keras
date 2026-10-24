"""Module docstring."""

from __future__ import annotations

from zero_keras.types import TensorLike

from zero_keras import ops


class TensorShape(tuple):
    """TensorShape class."""

    def __new__(cls, dims: TensorLike) -> TensorLike:
        """Function docstring.

        Args:
            dims: Description.
        """
        if dims is None:
            return super().__new__(cls, ())
        if isinstance(dims, int):
            dims = (dims,)
        return super().__new__(cls, tuple(dims))

    @property
    def dims(self) -> TensorLike:
        """Function docstring."""
        return list(self)

    @property
    def rank(self) -> TensorLike:
        """Function docstring."""
        return len(self)

    def as_list(self) -> TensorLike:
        """Function docstring."""
        return list(self)

    def is_fully_defined(self) -> TensorLike:
        """Function docstring."""
        return all(d is not None for d in self)


from zero_keras.activations import _to_tensor

"""Core layers module."""


class _GraphNode:
    """Class docstring."""

    def __init__(self, layer: TensorLike, inputs: TensorLike, outputs: TensorLike) -> None:
        """Function docstring.

        Args:
            layer: Description.
            inputs: Description.
            outputs: Description.
        """
        self.layer = layer
        self.inputs = inputs
        self.outputs = outputs


class KerasTensor:
    """docstring."""

    def __init__(
        self,
        shape: TensorLike,
        dtype: str = "float32",
        name: TensorLike = None,
        data: TensorLike = None,
    ) -> None:
        """Function docstring.

        Args:
            shape: Description.
            dtype: Description.
            name: Description.
            data: Description.
        """
        self.shape = shape
        self.dtype = dtype
        self.name = name
        self._keras_history = None
        if data is not None:
            self.data = data
        else:
            try:
                import ml_switcheroo_compiler.ops as backend_ops

                self.data = backend_ops.zeros(shape) if shape is not None else None
            except Exception:
                self.data = None

    def __iter__(self) -> TensorLike:
        """__iter__ docstring."""
        raise TypeError("KerasTensor is not iterable")

    @property
    def ndim(self) -> TensorLike:
        """Ndim docstring."""
        return len(self.shape) if self.shape is not None else None

    @property
    def ragged(self) -> TensorLike:
        """Ragged docstring."""
        return False

    def reshape(self, shape: TensorLike) -> TensorLike:
        """Reshape docstring.

        Args:
            shape: Shape.
        """
        return KerasTensor(shape=shape, dtype=self.dtype, name=self.name)

    @property
    def sparse(self) -> TensorLike:
        """Sparse docstring."""
        return False

    def squeeze(self, axis: TensorLike = None) -> TensorLike:
        """Squeeze docstring.

        Args:
            axis: Axis.
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
        """Function docstring.

        Args:
            x: Description.
            args: Description.
            kwargs: Description.
        """
        return x

    def __add__(self, other: TensorLike) -> TensorLike:
        """__add__ function.

        Args:
        other: Parameter other.

        Returns:
        TensorLike: Return value.

        """
        if (self.data if hasattr(self, "data") else None) is not None:
            import ml_switcheroo_compiler.ops as msc_ops

            from zero_keras.activations import _to_tensor

            return msc_ops.add(_to_tensor(self.data), _to_tensor(other))
        return KerasTensor(self.shape, self.dtype)

    def __sub__(self, other: TensorLike) -> TensorLike:
        """__sub__ function.

        Args:
        other: Parameter other.

        Returns:
        TensorLike: Return value.

        """
        if (self.data if hasattr(self, "data") else None) is not None:
            import ml_switcheroo_compiler.ops as msc_ops

            from zero_keras.activations import _to_tensor

            return msc_ops.subtract(_to_tensor(self.data), _to_tensor(other))
        return KerasTensor(self.shape, self.dtype)

    def __mul__(self, other: TensorLike) -> TensorLike:
        """Function docstring.

        Args:
            other: Description.
        """
        if (self.data if hasattr(self, "data") else None) is not None:
            from zero_keras import ops
            from zero_keras.activations import _to_tensor

            return ops.multiply(_to_tensor(self), _to_tensor(other))
        return KerasTensor(self.shape, self.dtype)

    def __rmul__(self, other: TensorLike) -> TensorLike:
        """Function docstring.

        Args:
            other: Description.
        """
        if (self.data if hasattr(self, "data") else None) is not None:
            from zero_keras import ops
            from zero_keras.activations import _to_tensor

            return ops.multiply(_to_tensor(self), _to_tensor(other))
        return KerasTensor(self.shape, self.dtype)

    def __rtruediv__(self, other: TensorLike) -> TensorLike:
        """Function docstring.

        Args:
            other: Description.
        """
        if (self.data if hasattr(self, "data") else None) is not None:
            from zero_keras import ops
            from zero_keras.activations import _to_tensor

            return ops.true_divide(_to_tensor(self), _to_tensor(other))
        return KerasTensor(self.shape, self.dtype)

    def __neg__(self) -> TensorLike:
        """Function docstring."""
        return KerasTensor(self.shape, self.dtype)

    def __radd__(self, other: TensorLike) -> TensorLike:
        """Function docstring.

        Args:
            other: Description.
        """
        if (self.data if hasattr(self, "data") else None) is not None:
            from zero_keras import ops
            from zero_keras.activations import _to_tensor

            return ops.add(_to_tensor(self), _to_tensor(other))
        return KerasTensor(self.shape, self.dtype)

    def __rsub__(self, other: TensorLike) -> TensorLike:
        """__mul__ function.

        Args:
        other: Parameter other.

        Returns:
        TensorLike: Return value.

        """
        return KerasTensor(self.shape, self.dtype)

    def __truediv__(self, other: TensorLike) -> TensorLike:
        """__truediv__ function.

        Args:
        other: Parameter other.

        Returns:
        TensorLike: Return value.

        """
        if (self.data if hasattr(self, "data") else None) is not None:
            import ml_switcheroo_compiler.ops as msc_ops

            from zero_keras.activations import _to_tensor

            return msc_ops.true_divide(_to_tensor(self.data), _to_tensor(other))
        return KerasTensor(self.shape, self.dtype)

    def __pow__(self, other: TensorLike) -> TensorLike:
        """__pow__ function.

        Args:
        other: Parameter other.

        Returns:
        TensorLike: Return value.

        """
        return KerasTensor(self.shape, self.dtype)

    def __eq__(self, other: TensorLike) -> TensorLike:
        """__eq__ function.

        Args:
        other: Parameter other.

        Returns:
        TensorLike: Return value.

        """
        if self.data is not None:
            return self.data == other
        safe_shape = (
            tuple(1 if x is None else x for x in self.shape) if self.shape is not None else ()
        )
        import ml_switcheroo_compiler.ops as backend_ops

        return backend_ops.ones(safe_shape)

    def __bool__(self) -> TensorLike:
        """__bool__ function.

        Returns:
        TensorLike: Return value.

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
        """Numpy function.

        Returns:
        TensorLike: Return value.

        """
        return self.data

    def __array__(self, dtype: TensorLike = None, copy: TensorLike = None) -> TensorLike:
        """__array__ function.

        Args:
        dtype: Parameter dtype.
        copy: Parameter copy.

        Returns:
        TensorLike: Return value.

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
        """__getitem__ function.

        Args:
        key: Parameter key.

        Returns:
        TensorLike: Return value.

        """
        if self.data is not None:
            return self.data[key]
        return 0.0


class Node:
    """docstring."""

    def __init__(
        self,
        operation: TensorLike = None,
        call_args: TensorLike = None,
        call_kwargs: TensorLike = None,
        outputs: TensorLike = None,
    ) -> None:
        """Function docstring.

        Args:
            operation: Description.
            call_args: Description.
            call_kwargs: Description.
            outputs: Description.
        """
        self.operation = operation
        self.call_args = call_args
        self.call_kwargs = call_kwargs
        self.outputs = outputs
        self.input_tensors = []
        if not hasattr(operation, "_inbound_nodes"):
            operation._inbound_nodes = []
        operation._inbound_nodes.append(self)
        if isinstance(outputs, list):
            for t in outputs:
                t._keras_history = self
        elif hasattr(outputs, "_keras_history"):
            outputs._keras_history = self


def Input(shape: TensorLike = None, name: TensorLike = None, **kwargs: TensorLike) -> TensorLike:
    """Used to instantiate a Keras tensor.

    A Keras tensor is a symbolic tensor-like object, which we augment with
    certain attributes that allow us to build a Keras model just by knowing the
    inputs and outputs of the model.

    For instance, if `a`, `b` and `c` are Keras tensors,
    it becomes possible to do:
    `model = Model(input=[a, b], output=c)`

    Args:
        kwargs: Input parameter ``kwargs``.
        shape: A shape tuple (tuple of integers or `None` objects),
            not including the batch size.
            For instance, `shape=(32,)` indicates that the expected input
            will be batches of 32-dimensional vectors. Elements of this tuple
            can be `None`; `None` elements represent dimensions where the shape
            is not known and may vary (e.g. sequence length).
        batch_size: Optional static batch size (integer).
        dtype: The data type expected by the input, as a string
            (e.g. `"float32"`, `"int32"`...)
        sparse: A boolean specifying whether the expected input will be sparse
            tensors. Note that, if `sparse` is `False`, sparse tensors can still
            be passed into the input - they will be densified with a default
            value of 0. This feature is only supported with the TensorFlow and
            the JAX backends. Defaults to `False`.
        ragged: A boolean specifying whether the expected input will be ragged
            tensors. Note that, if `ragged` is `False`, ragged tensors can still
            be passed into the input - they will be densified with a default
            value of 0. This feature is only supported with the TensorFlow
            backend. Defaults to `False`.
        batch_shape: Optional shape tuple (tuple of integers or `None` objects),
            including the batch size.
        name: Optional name string for the layer.
            Should be unique in a model (do not reuse the same name twice).
            It will be autogenerated if it isn't provided.
        tensor: Optional existing tensor to wrap into the `Input` layer.
            If set, the layer will use this tensor rather
            than creating a new placeholder tensor.
        optional: Boolean, whether the input is optional or not.
            An optional input can accept `None` values.

    Returns:
      A Keras tensor.

    Example:
    ```python
    # This is a logistic regression in Keras
    x = Input(shape=(32,))
    y = Dense(16, activation='softmax')(x)
    model = Model(x, y)
    ```
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
        self, value: TensorLike, name: TensorLike = None, **kwargs: TensorLike
    ) -> TensorLike:
        """add_metric docstring.

        Args:
            value: Value.
            name: Name.
            **kwargs: Kwargs.
        """
        if not hasattr(self, "_metrics"):
            self._metrics = []
        if name is not None:
            self._metrics.append({name: value})
        else:
            self._metrics.append(value)

    def add_variable(
        self,
        shape: TensorLike,
        initializer: TensorLike,
        dtype: TensorLike = None,
        trainable: TensorLike = True,
        name: TensorLike = None,
    ) -> TensorLike:
        """add_variable docstring.

        Args:
            shape: Shape.
            initializer: Initializer.
            dtype: Dtype.
            trainable: Trainable.
            name: Name.
        """
        return Variable(initializer, shape=shape, dtype=dtype, trainable=trainable, name=name)

    def build_from_config(self, config: TensorLike) -> TensorLike:
        """build_from_config docstring.

        Args:
            config: Config.
        """
        self.build(config.get("input_shape"))

    @property
    def compute_dtype(self) -> TensorLike:
        """compute_dtype docstring."""
        return self._compute_dtype if hasattr(self, "_compute_dtype") else None

    def compute_mask(self, inputs: TensorLike, mask: TensorLike = None) -> TensorLike:
        """compute_mask docstring.

        Args:
            inputs: Inputs.
            mask: Mask.
        """
        return mask

    def compute_output_spec(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """compute_output_spec docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        if hasattr(self, "compute_output_shape"):
            return self.compute_output_shape(*args, **kwargs)
        from zero_keras.core_layers import KerasTensor

        return KerasTensor(shape=None, dtype=(self.dtype if hasattr(self, "dtype") else "float32"))

    def count_params(self) -> TensorLike:
        """count_params docstring."""
        return 0

    @property
    def dtype(self) -> TensorLike:
        """Dtype docstring."""
        return self._dtype if hasattr(self, "_dtype") else None

    @property
    def dtype_policy(self) -> TensorLike:
        """dtype_policy docstring."""
        return self._dtype_policy if hasattr(self, "_dtype_policy") else None

    def get_build_config(self) -> TensorLike:
        """get_build_config docstring."""
        if hasattr(self, "_build_config"):
            return self._build_config
        return {"input_shape": None}

    def get_config(self) -> TensorLike:
        """get_config docstring."""
        return {"name": self.name}

    @property
    def input(self) -> TensorLike:
        """Input docstring."""
        return (
            self._inbound_nodes[0].input_tensors
            if hasattr(self, "_inbound_nodes") and self._inbound_nodes
            else None
        )

    @property
    def input_dtype(self) -> TensorLike:
        """input_dtype docstring."""
        return self.dtype if hasattr(self, "dtype") else None

    def load_own_variables(self, store: TensorLike) -> TensorLike:
        """load_own_variables docstring.

        Args:
            store: Store.
        """
        if not hasattr(self, "variables") or not self.variables:
            return
        for i, var in enumerate(self.variables):
            if str(i) in store:
                var.assign(store[str(i)])

    @property
    def metrics(self) -> TensorLike:
        """Metrics docstring."""
        return self._metrics if hasattr(self, "_metrics") else []

    @property
    def metrics_variables(self) -> TensorLike:
        """metrics_variables docstring."""
        return []

    @property
    def non_trainable_variables(self) -> TensorLike:
        """non_trainable_variables docstring."""
        return (
            self.non_trainable_weights
            if hasattr(self, "non_trainable_weights")
            else self._non_trainable_weights
            if hasattr(self, "_non_trainable_weights")
            else []
        )

    @property
    def output(self) -> TensorLike:
        """Output docstring."""
        return (
            self._inbound_nodes[0].output_tensors
            if hasattr(self, "_inbound_nodes") and self._inbound_nodes
            else None
        )

    @property
    def path(self) -> TensorLike:
        """Path docstring."""
        return self.name if hasattr(self, "name") else None

    @property
    def quantization_mode(self) -> TensorLike:
        """quantization_mode docstring."""
        return self._quantization_mode if hasattr(self, "_quantization_mode") else None

    def quantize(self, mode: TensorLike) -> TensorLike:
        """Quantize docstring.

        Args:
            mode: Mode.
        """
        self._quantization_mode = mode
        return self

    def quantized_build(self, input_shape: TensorLike) -> TensorLike:
        """quantized_build docstring.

        Args:
            input_shape: Input shape.
        """
        self.build(input_shape)

    def quantized_call(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """quantized_call docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        return self.call(*args, **kwargs)

    def rematerialized_call(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """rematerialized_call docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        return self.call(*args, **kwargs)

    def save_own_variables(self, store: TensorLike) -> TensorLike:
        """save_own_variables docstring.

        Args:
            store: Store.
        """
        if not hasattr(self, "variables") or not self.variables:
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
        """stateless_call docstring.

        Args:
            trainable_variables: Trainable variables.
            non_trainable_variables: Non trainable variables.
            *args: Args.
            **kwargs: Kwargs.
        """
        return self.call(*args, **kwargs)

    def symbolic_call(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """symbolic_call docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        return self.call(*args, **kwargs)

    @property
    def trainable_variables(self) -> TensorLike:
        """trainable_variables docstring."""
        return (
            self.trainable_weights
            if hasattr(self, "trainable_weights")
            else self._trainable_weights
            if hasattr(self, "_trainable_weights")
            else []
        )

    @property
    def variable_dtype(self) -> TensorLike:
        """variable_dtype docstring."""
        return self.dtype if hasattr(self, "dtype") else None

    @property
    def variables(self) -> TensorLike:
        """Variables docstring."""
        return self.weights if hasattr(self, "weights") else []

    @property
    def input_spec(self) -> TensorLike:
        """input_spec docstring."""
        return self._input_spec if hasattr(self, "_input_spec") else None

    @input_spec.setter
    def input_spec(self, value: TensorLike) -> TensorLike:
        """Docstring.

        Args:
            value: Input parameter ``value``.
        """
        self._input_spec = value

    @property
    def supports_masking(self) -> TensorLike:
        """supports_masking docstring."""
        return self._supports_masking if hasattr(self, "_supports_masking") else False

    @supports_masking.setter
    def supports_masking(self, value: TensorLike) -> TensorLike:
        """Docstring.

        Args:
            value: Input parameter ``value``.
        """
        self._supports_masking = value

    @property
    def trainable(self) -> TensorLike:
        """Trainable docstring."""
        return self._trainable if hasattr(self, "_trainable") else True

    @trainable.setter
    def trainable(self, value: TensorLike) -> TensorLike:
        """Docstring.

        Args:
            value: Input parameter ``value``.
        """
        self._trainable = value

    def __init__(self, **kwargs: TensorLike) -> None:
        """Function docstring.

        Args:
            kwargs: Description.
        """
        self.built = False
        self._name = kwargs.get("name")
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        """Name function.

        Returns:
        TensorLike: Return value.

        """
        return self._name or "layer"

    @classmethod
    def from_config(cls, config: TensorLike) -> TensorLike:
        """Function docstring.

        Args:
            config: Description.
        """
        return cls(**config)

    def add_loss(self, loss: TensorLike) -> TensorLike:
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
            loss: Input parameter ``loss``.
        """
        if not hasattr(self, "_losses"):
            self._losses = []
        self._losses.append(loss)

    @property
    def losses(self) -> TensorLike:
        """List of scalar losses from `add_loss`, regularizers and sublayers."""
        losses_list = list((self._losses if hasattr(self, "_losses") else []))
        for v in self.__dict__.values():
            if isinstance(v, Layer):
                losses_list.extend(v.losses)
            elif isinstance(v, list):
                for item in v:
                    if hasattr(item, "losses"):
                        losses_list.extend(item.losses)
            elif isinstance(v, tuple):
                for item in v:
                    if hasattr(item, "losses"):
                        losses_list.extend(item.losses)
            elif isinstance(v, dict):
                for item in v.values():
                    if hasattr(item, "losses"):
                        losses_list.extend(item.losses)
        return losses_list

    def compute_output_shape(self, input_shape: TensorLike) -> TensorLike:
        """compute_output_shape docstring.

        Args:
            input_shape: Input shape.
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
        """Build function.

        Args:
        input_shape: Parameter input_shape.

        Returns:
        TensorLike: Return value.

        """
        self.built = True

    def add_weight(
        self,
        shape: TensorLike,
        initializer: TensorLike = "zeros",
        trainable: TensorLike = True,
        name: TensorLike = None,
        regularizer: TensorLike = None,
        constraint: TensorLike = None,
        **kwargs: TensorLike,
    ) -> TensorLike:
        """Add a weight variable to the layer.

        Args:
            kwargs: Input parameter ``kwargs``.
            shape: Shape tuple for the variable. Must be fully-defined
                (no `None` entries). Defaults to `()` (scalar) if unspecified.
            initializer: Initializer object to use to populate the initial
                variable value, or string name of a built-in initializer
                (e.g. `"random_normal"`). If unspecified, defaults to
                `"glorot_uniform"` for floating-point variables and to `"zeros"`
                for all other types (e.g. int, bool).
            dtype: Dtype of the variable to create, e.g. `"float32"`. If
                unspecified, defaults to the layer's variable dtype
                (which itself defaults to `"float32"` if unspecified).
            trainable: Boolean, whether the variable should be trainable via
                backprop or whether its updates are managed manually. Defaults
                to `True`.
            autocast: Boolean, whether to autocast layers variables when
                accessing them. Defaults to `True`.
            regularizer: Regularizer object to call to apply penalty on the
                weight. These penalties are summed into the loss function
                during optimization. Defaults to `None`.
            constraint: Contrainst object to call on the variable after any
                optimizer update, or string name of a built-in constraint.
                Defaults to `None`.
            aggregation: Optional string, one of `None`, `"none"`, `"mean"`,
                `"sum"` or `"only_first_replica"`. Annotates the variable with
                the type of multi-replica aggregation to be used for this
                variable when writing custom data parallel training loops.
                Defaults to `"none"`.
            overwrite_with_gradient: Boolean, whether to overwrite the variable
                with the computed gradient. This is useful for float8 training.
                Defaults to `False`.
            name: String name of the variable. Useful for debugging purposes.
        """
        from zero_keras.initializers import get as get_initializer

        init_fn = get_initializer(initializer)
        weight = init_fn(shape=shape)
        if not hasattr(self, "_weights"):
            self._weights = []
        self._weights.append(weight)
        return weight

    @property
    def weights(self) -> TensorLike:
        """List of all weight variables of the layer.

        Unlike, `layer.variables` this excludes metric state and random seeds.
        """
        w = list((self._weights if hasattr(self, "_weights") else []))
        for v in self.__dict__.values():
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
        """List of all trainable weight variables of the layer.

        These are the weights that get updated by the optimizer during training.
        """
        return self.weights

    @property
    def non_trainable_weights(self) -> TensorLike:
        """List of all non-trainable weight variables of the layer.

        These are the weights that should not be updated by the optimizer during
        training. Unlike, `layer.non_trainable_variables` this excludes metric
        state and random seeds.
        """
        return []

    def get_weights(self) -> TensorLike:
        """Return the values of `layer.weights` as a list of NumPy arrays."""
        from ml_switcheroo_compiler.serialization import to_numpy

        return [to_numpy(w) for w in self.weights]

    def set_weights(self, weights: TensorLike) -> TensorLike:
        """Sets the values of `layer.weights` from a list of NumPy arrays.

        Args:
            weights: Input parameter ``weights``.
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
        """Call function.

        Args:
        inputs: Parameter inputs.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

        """
        return inputs

    def __call__(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Call self as a function.

        Args:
            inputs: Input parameter ``inputs``.
            args: Input parameter ``args``.
            kwargs: Input parameter ``kwargs``.
        """
        if not self.built:
            self.build((inputs.shape if hasattr(inputs, "shape") else None))

        # Simple tracing
        is_tracing = False

        def check_tracing(x: TensorLike) -> TensorLike:
            """Function docstring.

            Args:
                x: Description.
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
            node = _GraphNode(self, inputs, outputs)

            def wrap_output(x: TensorLike) -> TensorLike:
                """Function docstring.

                Args:
                    x: Description.
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
        """Function docstring.

        Args:
            args: Description.
            kwargs: Description.
        """
        if cls is Model and ("inputs" in kwargs and "outputs" in kwargs):
            return Functional(*args, **kwargs)
        return super().__new__(cls)

    def compute_loss(
        self,
        x: TensorLike = None,
        y: TensorLike = None,
        y_pred: TensorLike = None,
        sample_weight: TensorLike = None,
    ) -> TensorLike:
        """compute_loss function.

        Args:
        x: Parameter x.
        y: Parameter y.
        y_pred: Parameter y_pred.
        sample_weight: Parameter sample_weight.

        Returns:
        TensorLike: Return value.

        """
        if (self.loss_fn if hasattr(self, "loss_fn") else None) is not None and callable(
            self.loss_fn
        ):
            loss = self.loss_fn(y, y_pred)
        else:
            loss = ops.mean(ops.square(y - y_pred))
        for reg_loss in self.losses if hasattr(self, "losses") else []:
            loss = loss + reg_loss
        return loss

    def compute_metrics(
        self, x: TensorLike, y: TensorLike, y_pred: TensorLike, sample_weight: TensorLike = None
    ) -> TensorLike:
        """compute_metrics function.

        Args:
        x: Parameter x.
        y: Parameter y.
        y_pred: Parameter y_pred.
        sample_weight: Parameter sample_weight.

        Returns:
        TensorLike: Return value.

        """
        metrics_dict = {}
        if self.compiled_metrics if hasattr(self, "compiled_metrics") else None:
            for m in self.compiled_metrics:
                m.update_state(y, y_pred)
                metrics_dict[m.name] = m.result()
        return metrics_dict

    def train_step(self, data: TensorLike) -> TensorLike:
        """train_step function.

        Args:
        data: Parameter data.

        Returns:
        TensorLike: Return value.

        """
        from ml_switcheroo_compiler.grad import (
            value_and_grad_wrt_vars,
        )

        if isinstance(data, tuple) and len(data) == 2:
            x, y = data
        else:
            x, y = data, None

        def loss_fn() -> TensorLike:
            """loss_fn function.

            Returns:
            TensorLike: Return value.

            """
            try:
                y_pred = self(x, training=True)
            except TypeError:
                y_pred = self(x)
            loss = self.compute_loss(x, y, y_pred)
            return loss

        loss, grads_dict = value_and_grad_wrt_vars(loss_fn)()

        trainable_vars = self.trainable_weights
        if (
            hasattr((self.optimizer if hasattr(self, "optimizer") else None), "apply_gradients")
            and trainable_vars
        ):
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
        """test_step function.

        Args:
        data: Parameter data.

        Returns:
        TensorLike: Return value.

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
        """predict_step function.

        Args:
        data: Parameter data.

        Returns:
        TensorLike: Return value.

        """
        x = data[0] if isinstance(data, (tuple, list)) else data
        try:
            return self(x, training=False)
        except TypeError:
            return self(x)

    def compile_from_config(self, config: TensorLike) -> TensorLike:
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
        """compiled_loss docstring."""
        return self.loss if hasattr(self, "loss") else None

    @property
    def distribute_reduction_method(self) -> TensorLike:
        """distribute_reduction_method docstring."""
        return "auto"

    @property
    def distribute_strategy(self) -> TensorLike:
        """distribute_strategy docstring."""
        return "auto"

    def export(self, filepath: TensorLike) -> TensorLike:
        """Export docstring.

        Args:
            filepath: Filepath.
        """
        from zero_keras.export import ExportArchive

        archive = ExportArchive()
        archive.track(self)
        archive.write_out(filepath)

    def get_compile_config(self) -> TensorLike:
        """get_compile_config docstring."""
        from zero_keras.losses import serialize as serialize_loss
        from zero_keras.optimizers import serialize as serialize_opt

        config = {}
        if self.optimizer if hasattr(self, "optimizer") else None:
            config["optimizer"] = serialize_opt(self.optimizer)
        if self.loss if hasattr(self, "loss") else None:
            config["loss"] = serialize_loss(self.loss)
        if self.compiled_metrics if hasattr(self, "compiled_metrics") else None:
            # A simplified metrics serialization
            config["metrics"] = [m.name for m in self.compiled_metrics]
        return config

    def get_layer(self, name: TensorLike = None, index: TensorLike = None) -> TensorLike:
        """get_layer docstring.

        Args:
            name: Name.
            index: Index.
        """
        if index is not None:
            return (self.layers if hasattr(self, "layers") else [])[index]
        if name is not None:
            for layer in self.layers if hasattr(self, "layers") else []:
                if layer.name == name:
                    return layer
        raise ValueError(f"No such layer: {name}.")

    def get_metrics_result(self) -> TensorLike:
        """get_metrics_result docstring."""
        results = {}
        if self.compiled_metrics if hasattr(self, "compiled_metrics") else None:
            for m in self.compiled_metrics:
                results[m.name] = m.result()
        return results

    def get_state_tree(self) -> TensorLike:
        """get_state_tree docstring."""
        return {"variables": (self.variables if hasattr(self, "variables") else [])}

    @property
    def jit_compile(self) -> TensorLike:
        """jit_compile docstring."""
        return self._jit_compile if hasattr(self, "_jit_compile") else False

    @jit_compile.setter
    def jit_compile(self, value: TensorLike) -> TensorLike:
        """Docstring.

        Args:
            value: Input parameter ``value``.
        """
        self._jit_compile = value

    def make_predict_function(self, force: TensorLike = False) -> TensorLike:
        """make_predict_function docstring.

        Args:
            force: Force.
        """
        if (self.predict_function if hasattr(self, "predict_function") else None) is None or force:
            self.predict_function = self.predict_step
        return self.predict_function

    def make_test_function(self, force: TensorLike = False) -> TensorLike:
        """make_test_function docstring.

        Args:
            force: Force.
        """
        if (self.test_function if hasattr(self, "test_function") else None) is None or force:
            self.test_function = self.test_step
        return self.test_function

    def make_train_function(self, force: TensorLike = False) -> TensorLike:
        """make_train_function docstring.

        Args:
            force: Force.
        """
        if (self.train_function if hasattr(self, "train_function") else None) is None or force:
            self.train_function = self.train_step
        return self.train_function

    @property
    def metrics_names(self) -> TensorLike:
        """metrics_names docstring."""
        return []

    def predict_on_batch(self, x: TensorLike) -> TensorLike:
        """predict_on_batch docstring.

        Args:
            x: Input.
        """
        return self.predict_step(x)

    def reset_metrics(self) -> TensorLike:
        """reset_metrics docstring."""
        if self.compiled_metrics if hasattr(self, "compiled_metrics") else None:
            for m in self.compiled_metrics:
                m.reset_state()

    def set_state_tree(self, state_tree: TensorLike) -> TensorLike:
        """set_state_tree docstring.

        Args:
            state_tree: State tree.
        """
        if "variables" in state_tree and hasattr(self, "variables"):
            for v_self, v_tree in zip(self.variables, state_tree["variables"]):
                v_self.assign(v_tree)

    def stateless_compute_loss(
        self, x: TensorLike, y: TensorLike, y_pred: TensorLike, sample_weight: TensorLike = None
    ) -> TensorLike:
        """stateless_compute_loss docstring.

        Args:
            x: Input parameter ``x``.
            y: Input parameter ``y``.
            y_pred: Input parameter ``y_pred``.
            sample_weight: Input parameter ``sample_weight``.
            *args: Args.
            **kwargs: Kwargs.
        """
        return self.compute_loss(x, y, y_pred, sample_weight=sample_weight)

    def summary(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
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
        y: TensorLike = None,
        sample_weight: TensorLike = None,
        return_dict: TensorLike = False,
    ) -> TensorLike:
        """test_on_batch docstring.

        Args:
            x: Input parameter ``x``.
            y: Input parameter ``y``.
            sample_weight: Input parameter ``sample_weight``.
            return_dict: Input parameter ``return_dict``.
            *args: Args.
            **kwargs: Kwargs.
        """
        from zero_keras.utils import pack_x_y_sample_weight

        batch = pack_x_y_sample_weight(x, y, sample_weight)
        return self.test_step(batch)

    def to_json(self) -> TensorLike:
        """to_json docstring."""
        import json

        return json.dumps(self.get_config())

    def train_on_batch(
        self,
        x: TensorLike,
        y: TensorLike = None,
        sample_weight: TensorLike = None,
        class_weight: TensorLike = None,
        return_dict: TensorLike = False,
    ) -> TensorLike:
        """train_on_batch docstring.

        Args:
            x: Input parameter ``x``.
            y: Input parameter ``y``.
            sample_weight: Input parameter ``sample_weight``.
            class_weight: Input parameter ``class_weight``.
            return_dict: Input parameter ``return_dict``.
            *args: Args.
            **kwargs: Kwargs.
        """
        from zero_keras.utils import pack_x_y_sample_weight

        batch = pack_x_y_sample_weight(x, y, sample_weight)
        return self.train_step(batch)

    @property
    def compiled_metrics(self) -> TensorLike:
        """compiled_metrics docstring."""
        return self._compiled_metrics if hasattr(self, "_compiled_metrics") else None

    @compiled_metrics.setter
    def compiled_metrics(self, value: TensorLike) -> TensorLike:
        """Docstring.

        Args:
            value: Input parameter ``value``.
        """
        self._compiled_metrics = value

    @property
    def layers(self) -> TensorLike:
        """Layers docstring."""
        return self._layers if hasattr(self, "_layers") else []

    @layers.setter
    def layers(self, value: TensorLike) -> TensorLike:
        """Docstring.

        Args:
            value: Input parameter ``value``.
        """
        self._layers = value

    @property
    def loss(self) -> TensorLike:
        """Loss docstring."""
        return self._loss if hasattr(self, "_loss") else None

    @loss.setter
    def loss(self, value: TensorLike) -> TensorLike:
        """Docstring.

        Args:
            value: Input parameter ``value``.
        """
        self._loss = value

    @property
    def run_eagerly(self) -> TensorLike:
        """run_eagerly docstring."""
        return self._run_eagerly if hasattr(self, "_run_eagerly") else False

    @run_eagerly.setter
    def run_eagerly(self, value: TensorLike) -> TensorLike:
        """Docstring.

        Args:
            value: Input parameter ``value``.
        """
        self._run_eagerly = value

    def __init__(
        self, inputs: TensorLike = None, outputs: TensorLike = None, **kwargs: TensorLike
    ) -> None:
        """Function docstring.

        Args:
            inputs: Description.
            outputs: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.inputs = inputs
        self.outputs = outputs
        self._compiled = False

    def compile(
        self,
        optimizer: TensorLike = "adam",
        loss: TensorLike = "mse",
        metrics: TensorLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Compile function.

        Args:
        optimizer: Parameter optimizer.
        loss: Parameter loss.
        metrics: Parameter metrics.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

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
        """_is_iterator function.

        Args:
        data: Parameter data.

        Returns:
        TensorLike: Return value.

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

    def _unpack_iterator(self, data: TensorLike, is_train: TensorLike = True) -> TensorLike:
        """_unpack_iterator function.

        Args:
        data: Parameter data.
        is_train: Parameter is_train.

        Returns:
        TensorLike: Return value.

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
        y: TensorLike = None,
        epochs: int = 1,
        batch_size: int = 32,
        **kwargs: TensorLike,
    ) -> TensorLike:
        """Fit function.

        Args:
        x: Parameter x.
        y: Parameter y.
        epochs: Parameter epochs.
        batch_size: Parameter batch_size.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

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
            if self.compiled_metrics if hasattr(self, "compiled_metrics") else None:
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
            if self.compiled_metrics if hasattr(self, "compiled_metrics") else None:
                for m in self.compiled_metrics:
                    epoch_logs[m.name] = m.result()

            callbacks.on_epoch_end(epoch, epoch_logs)
            for k, v in epoch_logs.items():
                history.setdefault(k, []).append(v)

            if self.stop_training if hasattr(self, "stop_training") else False:
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
        filepath: TensorLike,
        overwrite: TensorLike = True,
        save_format: TensorLike = None,
        **kwargs: TensorLike,
    ) -> TensorLike:
        """Save function.

        Args:
        filepath: Parameter filepath.
        overwrite: Parameter overwrite.
        save_format: Parameter save_format.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

        """
        import json
        import struct
        import zipfile

        metadata = {
            "keras_version": "3.0.0",
            "date_saved": "2026-06-18",
        }

        config = {
            "class_name": self.__class__.__name__,
            "config": (self.get_config if hasattr(self, "get_config") else dict)(),
        }

        def _to_bytes_and_dtype(tensor: TensorLike) -> TensorLike:
            """_to_bytes_and_dtype function.

            Args:
            tensor: Parameter tensor.

            Returns:
            TensorLike: Return value.

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

                def flatten(x: TensorLike) -> TensorLike:
                    """Flatten function.

                    Args:
                    x: Parameter x.

                    Returns:
                    TensorLike: Return value.

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

        def _get_safetensors(w_list: TensorLike, prefix: TensorLike = "weight_") -> TensorLike:
            """_get_safetensors function.

            Args:
            w_list: Parameter w_list.
            prefix: Parameter prefix.

            Returns:
            TensorLike: Return value.

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

            if (
                hasattr(self, "optimizer")
                and hasattr(self.optimizer, "variables")
                and self.optimizer.variables
            ):
                zinfo_opt = zipfile.ZipInfo("optimizer.safetensors")
                zf.writestr(zinfo_opt, _get_safetensors(self.optimizer.variables, "opt_weight_"))

    def evaluate(
        self,
        x: TensorLike,
        y: TensorLike = None,
        batch_size: int | None = None,
        **kwargs: TensorLike,
    ) -> TensorLike:
        """Evaluate function.

        Args:
        x: Parameter x.
        y: Parameter y.
        batch_size: Parameter batch_size.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

        """
        import math

        if batch_size is None:
            batch_size = 32

        num_samples = len(x) if hasattr(x, "__len__") else 0
        num_batches = math.ceil(num_samples / batch_size) if num_samples > 0 else 1

        if self.compiled_metrics if hasattr(self, "compiled_metrics") else None:
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
        if self.compiled_metrics if hasattr(self, "compiled_metrics") else None:
            for m in self.compiled_metrics:
                metrics_results[m.name] = m.result()
        callbacks.on_test_end(metrics_results)
        return metrics_results

    def predict(
        self, x: TensorLike, batch_size: int | None = None, **kwargs: TensorLike
    ) -> TensorLike:
        """Predict function.

        Args:
        x: Parameter x.
        batch_size: Parameter batch_size.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

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
        filepath: TensorLike,
        overwrite: TensorLike = True,
        save_format: TensorLike = None,
        options: TensorLike = None,
    ) -> TensorLike:
        """Saves all layer weights.

        Args:
            filepath: Input parameter ``filepath``.
            overwrite: Input parameter ``overwrite``.
            save_format: Input parameter ``save_format``.
            options: Input parameter ``options``.
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
        filepath: TensorLike,
        skip_mismatch: TensorLike = False,
        by_name: TensorLike = False,
        options: TensorLike = None,
    ) -> TensorLike:
        """Loads all layer weights.

        Args:
            filepath: Input parameter ``filepath``.
            skip_mismatch: Input parameter ``skip_mismatch``.
            by_name: Input parameter ``by_name``.
            options: Input parameter ``options``.
        """
        import ml_switcheroo_compiler.serialization as msc_serialization

        weights = msc_serialization.load_weights(filepath, target_model=self)

        # apply weights
        if not hasattr(self, "weights") or not self.weights:
            return

        for i, w in enumerate(self.weights):
            k = "weight_" + str(i)
            if k in weights and hasattr(w, "assign"):
                w.assign(weights[k])


class Functional(Model):
    """Functional class."""

    def __init__(
        self, inputs: TensorLike = None, outputs: TensorLike = None, **kwargs: TensorLike
    ) -> None:
        """Function docstring.

        Args:
            inputs: Description.
            outputs: Description.
            kwargs: Description.
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
        """Function docstring.

        Args:
            inputs: Description.
            outputs: Description.
        """
        nodes_by_depth = {}

        # Simple DFS to map depths
        def _get_depth(tensor: TensorLike, current_depth: TensorLike) -> TensorLike:
            """Function docstring.

            Args:
                tensor: Description.
                current_depth: Description.
            """
            if not isinstance(tensor, KerasTensor):
                return
            node = tensor._keras_history if hasattr(tensor, "_keras_history") else None
            if node is None:
                return

            if current_depth not in nodes_by_depth:
                nodes_by_depth[current_depth] = []
            if node not in nodes_by_depth[current_depth]:
                nodes_by_depth[current_depth].append(node)
            else:
                return

            node_inputs = node.inputs
            if isinstance(node_inputs, (list, tuple)):
                for x in node_inputs:
                    _get_depth(x, current_depth + 1)
            elif isinstance(node_inputs, dict):
                for x in node_inputs.values():
                    _get_depth(x, current_depth + 1)
            else:
                _get_depth(node_inputs, current_depth + 1)

        if isinstance(outputs, (list, tuple)):
            for x in outputs:
                _get_depth(x, 0)
        elif isinstance(outputs, dict):
            for x in outputs.values():
                _get_depth(x, 0)
        else:
            _get_depth(outputs, 0)

        return nodes_by_depth

    def get_config(self) -> TensorLike:
        """get_config docstring."""
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
                    "class_name": layer.__class__.__name__,
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
        """Function docstring.

        Args:
            inputs: Description.
            args: Description.
            kwargs: Description.
        """
        tensor_dict = {}

        def _add_input(key: TensorLike, val: TensorLike) -> TensorLike:
            """Function docstring.

            Args:
                key: Description.
                val: Description.
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
                    """Function docstring.

                    Args:
                        x: Description.
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
            """Function docstring.

            Args:
                x: Description.
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


class Sequential(Model):
    """Sequential class."""

    def __init__(self, layers: TensorLike = None, **kwargs: TensorLike) -> None:
        """Function docstring.

        Args:
            layers: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.layers = layers or []

    def add(self, layer: TensorLike) -> TensorLike:
        """Add function.

        Args:
        layer: Parameter layer.

        Returns:
        TensorLike: Return value.

        """
        self.layers.append(layer)

    def call(self, inputs: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Call function.

        Args:
        inputs: Parameter inputs.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        TensorLike: Return value.

        """
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


def deserialize(
    config: TensorLike, custom_objects: TensorLike = None, safe_mode: TensorLike = True
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
        import builtins

        import zero_keras.layers
        import zero_keras.models

        cls = zero_keras.layers.__dict__.get(class_name, None)
        if cls is None:
            cls = zero_keras.models.__dict__.get(class_name, None)
        if cls is None:
            cls = builtins.__dict__.get(class_name, None)

    if cls and hasattr(cls, "from_config"):
        if "config" in config:
            return cls.from_config(config.get("config", {}))
        return cls.from_config(config)

    if cls:
        if "config" in config:
            return cls(**config.get("config", {}))
        return cls(**config)

    raise ValueError(f"Unknown class {class_name}")


def get(identifier: TensorLike, custom_objects: TensorLike = None) -> TensorLike:
    """Get function.

    Args:
    identifier: Parameter identifier.
    custom_objects: Parameter custom_objects.

    Returns:
    TensorLike: Return value.

    """
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        return deserialize(identifier, custom_objects=custom_objects)
    if isinstance(identifier, str):
        if custom_objects and identifier in custom_objects:
            return custom_objects[identifier]
        import zero_keras.initializers
        import zero_keras.layers
        import zero_keras.losses
        import zero_keras.metrics
        import zero_keras.models
        import zero_keras.optimizers
        import zero_keras.regularizers

        modules_to_check = [
            zero_keras.layers,
            zero_keras.models,
            zero_keras.metrics,
            zero_keras.losses,
            zero_keras.initializers,
            zero_keras.regularizers,
            zero_keras.optimizers,
        ]

        for mod in modules_to_check:
            cls = mod.__dict__.get(identifier, None)
            if cls:
                return cls

        raise ValueError(f"Unknown identifier {identifier}")
    return identifier


class Function:
    """Function docstring."""

    def call(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Call docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        if (self._fn if hasattr(self, "_fn") else None) is not None:
            return self._fn(*args, **kwargs)
        return args[0] if args else None

    def compute_output_shape(self, input_shape: TensorLike) -> TensorLike:
        """compute_output_shape docstring.

        Args:
            input_shape: Input shape.
        """
        if (self._shape_fn if hasattr(self, "_shape_fn") else None) is not None:
            return self._shape_fn(input_shape)
        return input_shape

    def compute_output_spec(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """compute_output_spec docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        if hasattr(self, "compute_output_shape"):
            return self.compute_output_shape(*args, **kwargs)
        from zero_keras.core_layers import KerasTensor

        return KerasTensor(shape=None, dtype=(self.dtype if hasattr(self, "dtype") else "float32"))

    @classmethod
    def from_config(cls, config: TensorLike) -> TensorLike:
        """from_config docstring.

        Args:
            config: Config.
        """
        return cls(**config)

    def get_config(self) -> TensorLike:
        """get_config docstring."""
        return {}

    @property
    def input(self) -> TensorLike:
        """Input docstring."""
        return (
            self._inbound_nodes[0].input_tensors
            if hasattr(self, "_inbound_nodes") and self._inbound_nodes
            else None
        )

    @property
    def inputs(self) -> TensorLike:
        """Inputs docstring."""
        return []

    @property
    def operations(self) -> TensorLike:
        """Operations docstring."""
        return []

    @property
    def output(self) -> TensorLike:
        """Output docstring."""
        return (
            self._inbound_nodes[0].output_tensors
            if hasattr(self, "_inbound_nodes") and self._inbound_nodes
            else None
        )

    @property
    def outputs(self) -> TensorLike:
        """Outputs docstring."""
        return []

    def quantized_call(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """quantized_call docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        return

    def symbolic_call(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
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
        """from_config docstring.

        Args:
            config: Config.
        """
        return cls(**config)

    def get_config(self) -> TensorLike:
        """get_config docstring."""
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
        dtype: TensorLike = None,
        shape: TensorLike = None,
        ndim: TensorLike = None,
        max_ndim: TensorLike = None,
        min_ndim: TensorLike = None,
        axes: TensorLike = None,
        allow_last_axis_squeeze: TensorLike = False,
        name: TensorLike = None,
    ) -> None:
        """Docstring.

        Args:
            dtype: Input parameter ``dtype``.
            shape: Input parameter ``shape``.
            ndim: Input parameter ``ndim``.
            max_ndim: Input parameter ``max_ndim``.
            min_ndim: Input parameter ``min_ndim``.
            axes: Input parameter ``axes``.
            allow_last_axis_squeeze: Input parameter ``allow_last_axis_squeeze``.
            name: Input parameter ``name``.
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
        """Call docstring.

        Args:
            args: Input parameter ``args``.
            kwargs: Input parameter ``kwargs``.
        """
        if (self._fn if hasattr(self, "_fn") else None) is not None:
            return self._fn(*args, **kwargs)
        return args[0] if args else None

    def compute_output_spec(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """compute_output_spec docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        if hasattr(self, "compute_output_shape"):
            return self.compute_output_shape(*args, **kwargs)
        from zero_keras.core_layers import KerasTensor

        return KerasTensor(shape=None, dtype=(self.dtype if hasattr(self, "dtype") else "float32"))

    @classmethod
    def from_config(cls, config: TensorLike) -> TensorLike:
        """from_config docstring.

        Args:
            config: Config.
        """
        return cls(**config)

    def get_config(self) -> TensorLike:
        """get_config docstring."""
        return self._config if hasattr(self, "_config") else {}

    @property
    def input(self) -> TensorLike:
        """Input docstring."""
        return self._input if hasattr(self, "_input") else None

    @property
    def output(self) -> TensorLike:
        """Output docstring."""
        return self._output if hasattr(self, "_output") else None

    def quantized_call(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """quantized_call docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        return self.call(*args, **kwargs)

    def symbolic_call(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """symbolic_call docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        return self.call(*args, **kwargs)


class Variable:
    """Variable docstring."""

    def __getitem__(self, key: TensorLike) -> TensorLike:
        """__getitem__ docstring.

        Args:
            key: Key.
        """
        return (
            self.value[key]
            if (self.value if hasattr(self, "value") else None) is not None
            else None
        )

    @property
    def aggregation(self) -> TensorLike:
        """Aggregation docstring."""
        return self._aggregation if hasattr(self, "_aggregation") else None

    @aggregation.setter
    def aggregation(self, value: TensorLike) -> TensorLike:
        """Docstring.

        Args:
            value: Input parameter ``value``.
        """
        self._aggregation = value

    def assign(self, value: TensorLike) -> TensorLike:
        """Assign docstring.

        Args:
            value: Value.
        """
        self.value = value

    def assign_add(self, value: TensorLike) -> TensorLike:
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

    def assign_sub(self, value: TensorLike) -> TensorLike:
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
        """Constraint docstring."""
        return self._constraint if hasattr(self, "_constraint") else None

    @constraint.setter
    def constraint(self, value: TensorLike) -> TensorLike:
        """Docstring.

        Args:
            value: Input parameter ``value``.
        """
        self._constraint = value

    @property
    def handle(self) -> TensorLike:
        """Handle docstring."""
        return self._handle if hasattr(self, "_handle") else None

    @handle.setter
    def handle(self, value: TensorLike) -> TensorLike:
        """Docstring.

        Args:
            value: Input parameter ``value``.
        """
        self._handle = value

    @property
    def ndim(self) -> TensorLike:
        """Ndim docstring."""
        return len(self.shape) if self.shape is not None else 0

    def numpy(self) -> TensorLike:
        """Numpy docstring."""
        from ml_switcheroo_compiler.serialization import to_numpy

        return to_numpy(self.value)

    def __array__(self, dtype: TensorLike = None) -> TensorLike:
        """Allow casting to numpy array.

        Args:
            dtype: Input parameter ``dtype``.
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

    def overwrite_with_gradient(self, gradient: TensorLike) -> TensorLike:
        """overwrite_with_gradient docstring.

        Args:
            gradient: Gradient.
        """
        self._gradient = gradient
        self.value = gradient

    @property
    def path(self) -> TensorLike:
        """Path docstring."""
        return self.name

    @property
    def regularizer(self) -> TensorLike:
        """Regularizer docstring."""
        return self._regularizer if hasattr(self, "_regularizer") else None

    @regularizer.setter
    def regularizer(self, value: TensorLike) -> TensorLike:
        """Docstring.

        Args:
            value: Input parameter ``value``.
        """
        self._regularizer = value

    @property
    def synchronization(self) -> TensorLike:
        """Synchronization docstring."""
        return self._synchronization if hasattr(self, "_synchronization") else None

    @synchronization.setter
    def synchronization(self, value: TensorLike) -> TensorLike:
        """Docstring.

        Args:
            value: Input parameter ``value``.
        """
        self._synchronization = value

    @property
    def value(self) -> TensorLike:
        """Value docstring."""
        return self._value if hasattr(self, "_value") else None

    @value.setter
    def value(self, val: TensorLike) -> TensorLike:
        """Docstring.

        Args:
            val: Input parameter ``val``.
        """
        self._value = val

    def __add__(self, other: TensorLike) -> TensorLike:
        """Execute ``__add__``.

        Args:
            other: Input parameter ``other``.
        """
        from zero_keras import ops

        return ops.add(self.value, other)

    def __radd__(self, other: TensorLike) -> TensorLike:
        """Execute ``__radd__``.

        Args:
            other: Input parameter ``other``.
        """
        from zero_keras import ops

        return ops.add(other, self.value)

    def __sub__(self, other: TensorLike) -> TensorLike:
        """Execute ``__sub__``.

        Args:
            other: Input parameter ``other``.
        """
        from zero_keras import ops

        return ops.subtract(self.value, other)

    def __rsub__(self, other: TensorLike) -> TensorLike:
        """Execute ``__rsub__``.

        Args:
            other: Input parameter ``other``.
        """
        from zero_keras import ops

        return ops.subtract(other, self.value)

    def __mul__(self, other: TensorLike) -> TensorLike:
        """Execute ``__mul__``.

        Args:
            other: Input parameter ``other``.
        """
        from zero_keras import ops

        return ops.multiply(self.value, other)

    def __rmul__(self, other: TensorLike) -> TensorLike:
        """Execute ``__rmul__``.

        Args:
            other: Input parameter ``other``.
        """
        from zero_keras import ops

        return ops.multiply(other, self.value)

    def __truediv__(self, other: TensorLike) -> TensorLike:
        """Execute ``__truediv__``.

        Args:
            other: Input parameter ``other``.
        """
        from zero_keras import ops

        return ops.divide(self.value, other)

    def __rtruediv__(self, other: TensorLike) -> TensorLike:
        """Execute ``__rtruediv__``.

        Args:
            other: Input parameter ``other``.
        """
        from zero_keras import ops

        return ops.divide(other, self.value)

    def __init__(
        self,
        initializer: TensorLike,
        shape: TensorLike = None,
        dtype: TensorLike = None,
        trainable: TensorLike = True,
        name: TensorLike = None,
    ) -> None:
        """Docstring.

        Args:
            initializer: Input parameter ``initializer``.
            shape: Input parameter ``shape``.
            dtype: Input parameter ``dtype``.
            trainable: Input parameter ``trainable``.
            name: Input parameter ``name``.
        """
        self.initializer = initializer
        self.shape = shape
        self.dtype = dtype
        self.trainable = trainable
        self.name = name


from zero_keras.dtype_policies import (
    DTypePolicy as DTypePolicy,
    FloatDTypePolicy as FloatDTypePolicy,
)


class Quantizer:
    """Quantizer docstring."""

    @classmethod
    def from_config(cls, config: TensorLike) -> TensorLike:
        """from_config docstring.

        Args:
            config: Config.
        """
        return cls(**config)

    def get_config(self) -> TensorLike:
        """get_config docstring."""
        return {"name": self.__class__.__name__}


class RematScope:
    """RematScope docstring."""

    def __enter__(self) -> TensorLike:
        """Docstring."""
        self._active = True
        return self

    def __exit__(self, *args: TensorLike) -> TensorLike:
        """Docstring.

        Args:
            args: Input parameter ``args``.
        """
        self._active = False


class StatelessScope:
    """StatelessScope docstring."""

    def __init__(self) -> None:
        """Execute ``__init__``."""
        self.losses = []
        self.updates = []

    def add_loss(self, loss: TensorLike) -> TensorLike:
        """add_loss docstring.

        Args:
            loss: Loss.
        """
        self.losses.append(loss)

    def add_update(self, update: TensorLike) -> TensorLike:
        """add_update docstring.

        Args:
            update: Update.
        """
        self.updates.append(update)

    def get_current_value(self, variable: TensorLike) -> TensorLike:
        """get_current_value docstring.

        Args:
            variable: Variable.
        """
        return variable.value if hasattr(variable, "value") else None

    def __enter__(self) -> TensorLike:
        """Docstring."""
        return self

    def __exit__(self, *args: TensorLike) -> TensorLike:
        """Docstring.

        Args:
            args: Input parameter ``args``.
        """


class SymbolicScope:
    """SymbolicScope docstring."""

    def __enter__(self) -> TensorLike:
        """Docstring."""
        self._is_symbolic = True
        return self

    def __exit__(self, *args: TensorLike) -> TensorLike:
        """Docstring.

        Args:
            args: Input parameter ``args``.
        """
        self._is_symbolic = False


def device(device_name: TensorLike) -> TensorLike:
    """Device docstring.

    Args:
        device_name: Input parameter ``device_name``.
    """

    class DeviceScope:
        """docstring."""

        def __init__(self, name: TensorLike) -> None:
            """Execute ``__init__``.

            Args:
                name: Input parameter ``name``.
            """
            self.name = name

        def __enter__(self) -> TensorLike:
            """Docstring."""
            return self

        def __exit__(self, *args: TensorLike) -> TensorLike:
            """Docstring.

            Args:
                args: Input parameter ``args``.
            """

    return DeviceScope(device_name)


class name_scope:
    """name_scope docstring."""

    def __init__(self, name: TensorLike) -> None:
        """Docstring.

        Args:
            name: Input parameter ``name``.
        """
        self.name = name

    def __enter__(self) -> TensorLike:
        """Docstring."""
        return self

    def __exit__(self, *args: TensorLike) -> TensorLike:
        """Docstring.

        Args:
            args: Input parameter ``args``.
        """


def remat(f: TensorLike) -> TensorLike:
    """Remat docstring.

    Args:
        f: Input parameter ``f``.
    """
    return f


def version() -> TensorLike:
    """Version docstring."""
    return "3.0.0"


def is_keras_tensor(x: TensorLike) -> TensorLike:
    """is_keras_tensor docstring.

    Args:
        x: Input parameter ``x``.
    """
    return isinstance(x, KerasTensor)


class Wrapper(Layer):
    """Base class for layer wrappers."""

    def __init__(self, layer: TensorLike, **kwargs: TensorLike) -> None:
        """Execute ``__init__``.

        Args:
            layer: Input parameter ``layer``.
            kwargs: Input parameter ``kwargs``.
        """
        super().__init__(**kwargs)
        self.layer = layer
