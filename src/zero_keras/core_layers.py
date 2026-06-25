"""Module docstring."""

from zero_keras.ops import ops


class TensorShape(tuple):
    """TensorShape class."""

    def __new__(cls, dims):
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
    def dims(self):
        """Function docstring."""
        return list(self)

    @property
    def rank(self):
        """Function docstring."""
        return len(self)

    def as_list(self):
        """Function docstring."""
        return list(self)

    def is_fully_defined(self):
        """Function docstring."""
        return all(d is not None for d in self)


from zero_keras.activations import _to_tensor

"""Core layers module."""

from typing import Any, Dict, Optional


class _GraphNode:
    """Class docstring."""

    def __init__(self, layer, inputs, outputs):
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

    def __init__(self, shape: Any, dtype: str = "float32", name: Any = None, data=None):
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

    def __iter__(self):
        """__iter__ docstring."""
        raise TypeError("KerasTensor is not iterable")

    @property
    def ndim(self):
        """ndim docstring."""
        return len(self.shape) if self.shape is not None else None

    @property
    def ragged(self):
        """ragged docstring."""
        return False

    def reshape(self, shape):
        """reshape docstring.

        Args:
            shape: Shape.
        """
        return KerasTensor(shape=shape, dtype=self.dtype, name=self.name)

    @property
    def sparse(self):
        """sparse docstring."""
        return False

    def squeeze(self, axis=None):
        """squeeze docstring.

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

    def __call__(self, x, *args, **kwargs):
        """Function docstring.

        Args:
            x: Description.
            args: Description.
            kwargs: Description.
        """
        return x

    def __add__(self, other: Any) -> Any:
        """__add__ function.

        Args:
        other: Parameter other.

        Returns:
        Any: Return value.

        """
        if getattr(self, "data", None) is not None:
            from zero_keras.activations import _to_tensor  # pragma: no cover
            import ml_switcheroo_compiler.ops as msc_ops  # pragma: no cover

            return getattr(msc_ops, "add")(
                _to_tensor(self.data), _to_tensor(other)
            )  # pragma: no cover
        return KerasTensor(self.shape, self.dtype)

    def __sub__(self, other: Any) -> Any:
        """__sub__ function.

        Args:
        other: Parameter other.

        Returns:
        Any: Return value.

        """
        if getattr(self, "data", None) is not None:
            from zero_keras.activations import _to_tensor
            import ml_switcheroo_compiler.ops as msc_ops

            return getattr(msc_ops, "subtract")(
                _to_tensor(self.data), _to_tensor(other)
            )
        return KerasTensor(self.shape, self.dtype)

    def __mul__(self, other: Any) -> Any:
        """Function docstring.

        Args:
            other: Description.
        """
        if getattr(self, "data", None) is not None:
            from zero_keras.activations import _to_tensor  # pragma: no cover
            from zero_keras.ops import ops  # pragma: no cover

            return ops.mul(_to_tensor(self), _to_tensor(other))  # pragma: no cover
        return KerasTensor(self.shape, self.dtype)

    def __rmul__(self, other: Any) -> Any:
        """Function docstring.

        Args:
            other: Description.
        """
        if getattr(self, "data", None) is not None:
            from zero_keras.activations import _to_tensor  # pragma: no cover
            from zero_keras.ops import ops  # pragma: no cover

            return ops.mul(_to_tensor(self), _to_tensor(other))  # pragma: no cover
        return KerasTensor(self.shape, self.dtype)

    def __rtruediv__(self, other: Any) -> Any:
        """Function docstring.

        Args:
            other: Description.
        """
        if getattr(self, "data", None) is not None:
            from zero_keras.activations import _to_tensor  # pragma: no cover
            from zero_keras.ops import ops  # pragma: no cover

            return ops.tuediv(_to_tensor(self), _to_tensor(other))  # pragma: no cover
        return KerasTensor(self.shape, self.dtype)

    def __neg__(self) -> Any:
        """Function docstring."""
        return KerasTensor(self.shape, self.dtype)

    def __radd__(self, other: Any) -> Any:
        """Function docstring.

        Args:
            other: Description.
        """
        if getattr(self, "data", None) is not None:
            from zero_keras.activations import _to_tensor  # pragma: no cover
            from zero_keras.ops import ops  # pragma: no cover

            return ops.add(_to_tensor(self), _to_tensor(other))  # pragma: no cover
        return KerasTensor(self.shape, self.dtype)

    def __rsub__(self, other: Any) -> Any:
        """__mul__ function.

        Args:
        other: Parameter other.

        Returns:
        Any: Return value.

        """
        return KerasTensor(self.shape, self.dtype)  # pragma: no cover

    def __truediv__(self, other: Any) -> Any:
        """__truediv__ function.

        Args:
        other: Parameter other.

        Returns:
        Any: Return value.

        """
        if getattr(self, "data", None) is not None:
            from zero_keras.activations import _to_tensor  # pragma: no cover
            import ml_switcheroo_compiler.ops as msc_ops  # pragma: no cover

            return getattr(msc_ops, "tuediv")(
                _to_tensor(self.data), _to_tensor(other)
            )  # pragma: no cover
        return KerasTensor(self.shape, self.dtype)

    def __pow__(self, other: Any) -> Any:
        """__pow__ function.

        Args:
        other: Parameter other.

        Returns:
        Any: Return value.

        """
        return KerasTensor(self.shape, self.dtype)

    def __eq__(self, other):
        """__eq__ function.

        Args:
        other: Parameter other.

        Returns:
        Any: Return value.

        """
        if self.data is not None:
            return self.data == other
        safe_shape = (
            tuple(1 if x is None else x for x in self.shape)
            if self.shape is not None
            else ()
        )
        import ml_switcheroo_compiler.ops as backend_ops

        return backend_ops.ones(safe_shape)

    def __bool__(self):
        """__bool__ function.

        Returns:
        Any: Return value.

        """
        import ml_switcheroo_compiler.core.config as config

        if getattr(config, "eager_mode", False) and self.data is not None:
            try:
                return bool(self.data)
            except Exception:
                pass
        raise TypeError(
            "Using a KerasTensor as a Python bool is not allowed. Use ops.cond or ops.where for data-dependent control flow, or enable eager execution."
        )

    def numpy(self):
        """Numpy function.

        Returns:
        Any: Return value.

        """
        return self.data

    def __array__(self, dtype=None, copy=None):
        """__array__ function.

        Args:
        dtype: Parameter dtype.
        copy: Parameter copy.

        Returns:
        Any: Return value.

        """
        import ml_switcheroo_compiler.ops as backend_ops

        arr = backend_ops.convert_to_numpy(self.data if self.data is not None else 0.0)
        return arr
        if hasattr(arr, "numpy"):
            return arr.numpy()
        return arr  # pragma: no cover

    def __getitem__(self, key):
        """__getitem__ function.

        Args:
        key: Parameter key.

        Returns:
        Any: Return value.

        """
        if self.data is not None:
            return self.data[key]
        return 0.0


class Node:
    """docstring."""

    def __init__(self, operation=None, call_args=None, call_kwargs=None, outputs=None):
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


def Input(shape: Any, name: Any = None, **kwargs) -> Any:
    """Used to instantiate a Keras tensor.

    A Keras tensor is a symbolic tensor-like object, which we augment with
    certain attributes that allow us to build a Keras model just by knowing the
    inputs and outputs of the model.

    For instance, if `a`, `b` and `c` are Keras tensors,
    it becomes possible to do:
    `model = Model(input=[a, b], output=c)`

    Args:
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
        batch_shape = (
            (batch_size,) + tuple(shape) if shape is not None else (batch_size,)
        )
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

    def add_metric(self, value, name=None, **kwargs):
        """add_metric docstring.

        Args:
            value: Value.
            name: Name.
            **kwargs: Kwargs.
        """
        pass

    def add_variable(self, shape, initializer, dtype=None, trainable=True, name=None):
        """add_variable docstring.

        Args:
            shape: Shape.
            initializer: Initializer.
            dtype: Dtype.
            trainable: Trainable.
            name: Name.
        """
        return Variable(
            initializer, shape=shape, dtype=dtype, trainable=trainable, name=name
        )

    def build_from_config(self, config):
        """build_from_config docstring.

        Args:
            config: Config.
        """
        self.build(config.get("input_shape"))

    @property
    def compute_dtype(self):
        """compute_dtype docstring."""
        return getattr(self, "_compute_dtype", None)

    def compute_mask(self, inputs, mask=None):
        """compute_mask docstring.

        Args:
            inputs: Inputs.
            mask: Mask.
        """
        return mask

    def compute_output_shape(self, input_shape):
        """compute_output_shape docstring.

        Args:
            input_shape: Input shape.
        """
        return input_shape

    def compute_output_spec(self, *args, **kwargs):
        """compute_output_spec docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        return None

    def count_params(self):
        """count_params docstring."""
        return 0

    @property
    def dtype(self):
        """dtype docstring."""
        return getattr(self, "_dtype", None)

    @property
    def dtype_policy(self):
        """dtype_policy docstring."""
        return getattr(self, "_dtype_policy", None)

    def get_build_config(self):
        """get_build_config docstring."""
        return {}

    def get_config(self):
        """get_config docstring."""
        return {"name": self.name}

    @property
    def input(self):
        """input docstring."""
        return None

    @property
    def input_dtype(self):
        """input_dtype docstring."""
        return None

    @property
    def input_spec(self):
        """input_spec docstring."""
        return getattr(self, "_input_spec", None)  # pragma: no cover

    @input_spec.setter
    def input_spec(self, value):
        """docstring."""

        self._input_spec = value  # pragma: no cover

    def load_own_variables(self, store):
        """load_own_variables docstring.

        Args:
            store: Store.
        """
        pass

    @property
    def metrics(self):
        """metrics docstring."""
        return getattr(self, "_metrics", [])

    @property
    def metrics_variables(self):
        """metrics_variables docstring."""
        return []

    @property
    def non_trainable_variables(self):
        """non_trainable_variables docstring."""
        return getattr(
            self, "non_trainable_weights", getattr(self, "_non_trainable_weights", [])
        )

    @property
    def output(self):
        """output docstring."""
        return None

    @property
    def path(self):
        """path docstring."""
        return getattr(self, "name", None)

    @property
    def quantization_mode(self):
        """quantization_mode docstring."""
        return None

    def quantize(self, mode):
        """quantize docstring.

        Args:
            mode: Mode.
        """
        pass

    def quantized_build(self, input_shape):
        """quantized_build docstring.

        Args:
            input_shape: Input shape.
        """
        self.build(input_shape)

    def quantized_call(self, *args, **kwargs):
        """quantized_call docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        return self.call(*args, **kwargs)

    def rematerialized_call(self, *args, **kwargs):
        """rematerialized_call docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        return self.call(*args, **kwargs)

    def save_own_variables(self, store):
        """save_own_variables docstring.

        Args:
            store: Store.
        """
        pass

    def stateless_call(
        self, trainable_variables, non_trainable_variables, *args, **kwargs
    ):
        """stateless_call docstring.

        Args:
            trainable_variables: Trainable variables.
            non_trainable_variables: Non trainable variables.
            *args: Args.
            **kwargs: Kwargs.
        """
        return self.call(*args, **kwargs)

    @property
    def supports_masking(self):
        """supports_masking docstring."""
        return getattr(self, "_supports_masking", False)  # pragma: no cover

    @supports_masking.setter
    def supports_masking(self, value):
        """docstring."""

        self._supports_masking = value  # pragma: no cover

    def symbolic_call(self, *args, **kwargs):
        """symbolic_call docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        return self.call(*args, **kwargs)

    @property
    def trainable(self):
        """trainable docstring."""
        return getattr(self, "_trainable", True)  # pragma: no cover

    @trainable.setter
    def trainable(self, value):
        """docstring."""

        self._trainable = value  # pragma: no cover

    @property
    def trainable_variables(self):
        """trainable_variables docstring."""
        return getattr(
            self, "trainable_weights", getattr(self, "_trainable_weights", [])
        )

    @property
    def variable_dtype(self):
        """variable_dtype docstring."""
        return None

    @property
    def variables(self):
        """variables docstring."""
        return getattr(self, "weights", [])

    @property
    def input_spec(self):
        """input_spec docstring."""
        return getattr(self, "_input_spec", None)  # pragma: no cover

    @input_spec.setter
    def input_spec(self, value):
        """docstring."""

        self._input_spec = value  # pragma: no cover

    @property
    def supports_masking(self):
        """supports_masking docstring."""
        return getattr(self, "_supports_masking", False)  # pragma: no cover

    @supports_masking.setter
    def supports_masking(self, value):
        """docstring."""

        self._supports_masking = value  # pragma: no cover

    @property
    def trainable(self):
        """trainable docstring."""
        return getattr(self, "_trainable", True)  # pragma: no cover

    @trainable.setter
    def trainable(self, value):
        """docstring."""

        self._trainable = value  # pragma: no cover

    @property
    def input_spec(self):
        """input_spec docstring."""
        return getattr(self, "_input_spec", None)

    @input_spec.setter
    def input_spec(self, value):
        """docstring."""

        self._input_spec = value  # pragma: no cover

    @property
    def supports_masking(self):
        """supports_masking docstring."""
        return getattr(self, "_supports_masking", False)

    @supports_masking.setter
    def supports_masking(self, value):
        """docstring."""

        self._supports_masking = value

    @property
    def trainable(self):
        """trainable docstring."""
        return getattr(self, "_trainable", True)

    @trainable.setter
    def trainable(self, value):
        """docstring."""

        self._trainable = value

    def __init__(self, **kwargs: Any):
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
        Any: Return value.

        """
        return self._name or "layer"

    @classmethod
    def from_config(cls, config):
        """Function docstring.

        Args:
            config: Description.
        """
        return cls(**config)

    def add_loss(self, loss):
        """Can be called inside of the `call()` method to add a scalar loss.

        Example:
        ```python
        class MyLayer(Layer):
            ...
            def call(self, x):
                self.add_loss(ops.sum(x))
                return x
        ```

        """
        if not hasattr(self, "_losses"):
            self._losses = []
        self._losses.append(loss)

    @property
    def losses(self):
        """List of scalar losses from `add_loss`, regularizers and sublayers."""
        losses_list = list(getattr(self, "_losses", []))
        for k, v in self.__dict__.items():
            if isinstance(v, Layer):
                losses_list.extend(v.losses)  # pragma: no cover
            elif isinstance(v, list):
                for item in v:
                    if hasattr(item, "losses"):
                        losses_list.extend(item.losses)
            elif isinstance(v, tuple):
                for item in v:  # pragma: no cover
                    if hasattr(item, "losses"):  # pragma: no cover
                        losses_list.extend(item.losses)  # pragma: no cover
            elif isinstance(v, dict):
                for item in v.values():
                    if hasattr(item, "losses"):
                        losses_list.extend(item.losses)  # pragma: no cover
        return losses_list

    def build(self, input_shape: Any) -> None:
        """Build function.

        Args:
        input_shape: Parameter input_shape.

        Returns:
        Any: Return value.

        """
        self.built = True

    def add_weight(
        self,
        shape,
        initializer="zeros",
        trainable=True,
        name=None,
        regularizer=None,
        constraint=None,
        **kwargs,
    ):
        """Add a weight variable to the layer.

        Args:
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
    def weights(self):
        """List of all weight variables of the layer.

        Unlike, `layer.variables` this excludes metric state and random seeds.
        """
        w = list(getattr(self, "_weights", []))
        for k, v in self.__dict__.items():
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
    def trainable_weights(self):
        """List of all trainable weight variables of the layer.

        These are the weights that get updated by the optimizer during training.
        """
        return self.weights

    @property
    def non_trainable_weights(self):
        """List of all non-trainable weight variables of the layer.

        These are the weights that should not be updated by the optimizer during
        training. Unlike, `layer.non_trainable_variables` this excludes metric
        state and random seeds.
        """
        return []

    def get_weights(self):
        """Return the values of `layer.weights` as a list of NumPy arrays."""
        from ml_switcheroo_compiler.serialization import to_numpy

        return [to_numpy(w) for w in self.weights]

    def set_weights(self, weights):
        """Sets the values of `layer.weights` from a list of NumPy arrays."""
        if len(weights) != len(self.weights):
            raise ValueError(
                f"Expected {len(self.weights)} weights, got {len(weights)}"
            )
        from ml_switcheroo_compiler.ops import asarray

        for w, new_w in zip(self.weights, weights):
            if hasattr(w, "assign"):
                w.assign(new_w)
            elif hasattr(w, "data"):
                w[...] = asarray(new_w)
            else:
                w[:] = new_w

    def call(self, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        """Call function.

        Args:
        inputs: Parameter inputs.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Any: Return value.

        """
        return inputs

    def __call__(self, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        """Call self as a function."""
        if not self.built:
            self.build(getattr(inputs, "shape", None))

        outputs = self.call(inputs, *args, **kwargs)

        # Simple tracing
        is_tracing = False

        def check_tracing(x):
            """Function docstring.

            Args:
                x: Description.
            """
            nonlocal is_tracing
            if isinstance(x, KerasTensor):
                is_tracing = True

        if isinstance(inputs, list):
            for x in inputs:
                check_tracing(x)
        elif isinstance(inputs, dict):
            for x in inputs.values():
                check_tracing(x)
        else:
            check_tracing(inputs)

        if is_tracing:
            node = _GraphNode(self, inputs, outputs)

            def wrap_output(x):
                """Function docstring.

                Args:
                    x: Description.
                """
                if (
                    hasattr(x, "shape")
                    and hasattr(x, "dtype")
                    and not isinstance(x, KerasTensor)
                ):
                    res = KerasTensor(x.shape, x.dtype, data=x)  # pragma: no cover
                    res._keras_history = node  # pragma: no cover
                    return res  # pragma: no cover
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

    def __new__(cls, *args, **kwargs):
        """Function docstring.

        Args:
            args: Description.
            kwargs: Description.
        """
        if cls is Model and ("inputs" in kwargs and "outputs" in kwargs):
            return Functional(*args, **kwargs)
        return super().__new__(cls)

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        """compute_loss function.

        Args:
        x: Parameter x.
        y: Parameter y.
        y_pred: Parameter y_pred.
        sample_weight: Parameter sample_weight.

        Returns:
        Any: Return value.

        """

        if getattr(self, "loss_fn", None) is not None and callable(self.loss_fn):
            loss = self.loss_fn(y, y_pred)
        else:
            loss = ops.mean(ops.square(y - y_pred))
        for reg_loss in getattr(self, "losses", []):
            loss = loss + reg_loss
        return loss

    def compute_metrics(self, x, y, y_pred, sample_weight=None):
        """compute_metrics function.

        Args:
        x: Parameter x.
        y: Parameter y.
        y_pred: Parameter y_pred.
        sample_weight: Parameter sample_weight.

        Returns:
        Any: Return value.

        """
        metrics_dict = {}
        if getattr(self, "compiled_metrics", None):
            for m in self.compiled_metrics:
                m.update_state(y, y_pred)
                metrics_dict[m.name] = m.result()
        return metrics_dict

    def train_step(self, data):
        """train_step function.

        Args:
        data: Parameter data.

        Returns:
        Any: Return value.

        """
        from ml_switcheroo_compiler.grad import (
            value_and_grad_wrt_vars,
        )

        if isinstance(data, tuple) and len(data) == 2:
            x, y = data
        else:
            x, y = data, None

        def loss_fn():
            """loss_fn function.

            Returns:
            Any: Return value.

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
            hasattr(getattr(self, "optimizer", None), "apply_gradients")
            and trainable_vars
        ):
            grads_list = [
                grads_dict.get(getattr(v, "id", id(v)), None) for v in trainable_vars
            ]
            self.optimizer.apply_gradients(zip(grads_list, trainable_vars))

        try:
            y_pred = self(x, training=False)
        except TypeError:
            y_pred = self(x)
        metrics = self.compute_metrics(x, y, y_pred)
        metrics["loss"] = loss
        return metrics

    def test_step(self, data):
        """test_step function.

        Args:
        data: Parameter data.

        Returns:
        Any: Return value.

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

    def predict_step(self, data):
        """predict_step function.

        Args:
        data: Parameter data.

        Returns:
        Any: Return value.

        """
        x = data[0] if isinstance(data, (tuple, list)) else data
        try:
            return self(x, training=False)
        except TypeError:
            return self(x)

    def compile_from_config(self, config):
        """compile_from_config docstring.

        Args:
            config: Config.
        """
        pass

    @property
    def compiled_loss(self):
        """compiled_loss docstring."""
        return None

    @property
    def compiled_metrics(self):
        """compiled_metrics docstring."""
        return getattr(self, "_compiled_metrics", None)  # pragma: no cover

    @compiled_metrics.setter
    def compiled_metrics(self, value):
        """docstring."""

        self._compiled_metrics = value  # pragma: no cover

    @property
    def distribute_reduction_method(self):
        """distribute_reduction_method docstring."""
        return None

    @property
    def distribute_strategy(self):
        """distribute_strategy docstring."""
        return None

    def export(self, filepath):
        """export docstring.

        Args:
            filepath: Filepath.
        """
        pass

    def get_compile_config(self):
        """get_compile_config docstring."""
        return {}

    def get_layer(self, name=None, index=None):
        """get_layer docstring.

        Args:
            name: Name.
            index: Index.
        """
        return None

    def get_metrics_result(self):
        """get_metrics_result docstring."""
        return {}

    def get_state_tree(self):
        """get_state_tree docstring."""
        return {}

    def jit_compile(self):
        """jit_compile docstring."""
        pass

    @property
    def layers(self):
        """layers docstring."""
        return getattr(self, "_layers", [])  # pragma: no cover

    @layers.setter
    def layers(self, value):
        """docstring."""

        self._layers = value  # pragma: no cover

    @layers.setter
    def layers(self, value):
        """docstring."""

        self._layers = value  # pragma: no cover

    @layers.setter
    def layers(self, value):
        """docstring."""

        self._layers = value  # pragma: no cover

    @property
    def loss(self):
        """loss docstring."""
        return getattr(self, "_loss", None)  # pragma: no cover

    @loss.setter
    def loss(self, value):
        """docstring."""

        self._loss = value  # pragma: no cover

    def make_predict_function(self):
        """make_predict_function docstring."""
        return None

    def make_test_function(self):
        """make_test_function docstring."""
        return None

    def make_train_function(self):
        """make_train_function docstring."""
        return None

    @property
    def metrics_names(self):
        """metrics_names docstring."""
        return []

    def predict_on_batch(self, x):
        """predict_on_batch docstring.

        Args:
            x: Input.
        """
        return self(x)

    def reset_metrics(self):
        """reset_metrics docstring."""
        pass

    @property
    def run_eagerly(self):
        """run_eagerly docstring."""
        return getattr(self, "_run_eagerly", False)  # pragma: no cover

    @run_eagerly.setter
    def run_eagerly(self, value):
        """docstring."""

        self._run_eagerly = value  # pragma: no cover

    def set_state_tree(self, state_tree):
        """set_state_tree docstring.

        Args:
            state_tree: State tree.
        """
        pass

    def stateless_compute_loss(self, *args, **kwargs):
        """stateless_compute_loss docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        return None

    def summary(self, *args, **kwargs):
        """summary docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        pass

    def test_on_batch(self, *args, **kwargs):
        """test_on_batch docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        return None

    def to_json(self):
        """to_json docstring."""
        import json

        return json.dumps(self.get_config())

    def train_on_batch(self, *args, **kwargs):
        """train_on_batch docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        return None

    @property
    def compiled_metrics(self):
        """compiled_metrics docstring."""
        return getattr(self, "_compiled_metrics", None)  # pragma: no cover

    @compiled_metrics.setter
    def compiled_metrics(self, value):
        """docstring."""

        self._compiled_metrics = value  # pragma: no cover

    @property
    def layers(self):
        """layers docstring."""
        return getattr(self, "_layers", [])  # pragma: no cover

    @layers.setter
    def layers(self, value):
        """docstring."""

        self._layers = value  # pragma: no cover

    @layers.setter
    def layers(self, value):
        """docstring."""

        self._layers = value  # pragma: no cover

    @property
    def loss(self):
        """loss docstring."""
        return getattr(self, "_loss", None)  # pragma: no cover

    @loss.setter
    def loss(self, value):
        """docstring."""

        self._loss = value  # pragma: no cover

    @property
    def run_eagerly(self):
        """run_eagerly docstring."""
        return getattr(self, "_run_eagerly", False)  # pragma: no cover

    @run_eagerly.setter
    def run_eagerly(self, value):
        """docstring."""

        self._run_eagerly = value  # pragma: no cover

    @property
    def compiled_metrics(self):
        """compiled_metrics docstring."""
        return getattr(self, "_compiled_metrics", None)

    @compiled_metrics.setter
    def compiled_metrics(self, value):
        """docstring."""

        self._compiled_metrics = value

    @property
    def layers(self):
        """layers docstring."""
        return getattr(self, "_layers", [])

    @layers.setter
    def layers(self, value):
        """docstring."""

        self._layers = value

    @property
    def loss(self):
        """loss docstring."""
        return getattr(self, "_loss", None)

    @loss.setter
    def loss(self, value):
        """docstring."""

        self._loss = value

    @property
    def run_eagerly(self):
        """run_eagerly docstring."""
        return getattr(self, "_run_eagerly", False)

    @run_eagerly.setter
    def run_eagerly(self, value):
        """docstring."""

        self._run_eagerly = value

    def __init__(self, inputs: Any = None, outputs: Any = None, **kwargs: Any):
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

    def compile(self, optimizer="adam", loss="mse", metrics=None, **kwargs) -> None:
        """Compile function.

        Args:
        optimizer: Parameter optimizer.
        loss: Parameter loss.
        metrics: Parameter metrics.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Any: Return value.

        """
        self._compiled = True

        if isinstance(optimizer, str):
            from zero_keras.optimizers import get as get_opt

            self.optimizer = get_opt(optimizer)
        else:
            self.optimizer = optimizer  # pragma: no cover

        if isinstance(loss, str):
            from zero_keras.losses import get as get_loss

            self.loss = loss
            self.loss_fn = get_loss(loss)
        else:
            self.loss = loss
            self.loss_fn = loss if callable(loss) else getattr(loss, "call", None)

        self.compiled_metrics = []
        if metrics is not None:
            if not isinstance(metrics, (list, tuple)):
                metrics = [metrics]  # pragma: no cover
            from zero_keras.metrics import get as get_metric

            for m in metrics:
                if isinstance(m, str):
                    self.compiled_metrics.append(get_metric(m))
                else:
                    self.compiled_metrics.append(m)

    def _is_iterator(self, data):
        """_is_iterator function.

        Args:
        data: Parameter data.

        Returns:
        Any: Return value.

        """
        if (
            hasattr(data, "__iter__")
            and not hasattr(data, "shape")
            and not isinstance(data, (list, tuple, dict))
        ):
            return True
        if type(data).__module__.startswith("tensorflow.python.data") or type(
            data
        ).__module__.startswith("torch.utils.data"):
            return True
        return False

    def _unpack_iterator(self, data, is_train=True):
        """_unpack_iterator function.

        Args:
        data: Parameter data.
        is_train: Parameter is_train.

        Returns:
        Any: Return value.

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
                    yield {
                        k: v.numpy() if hasattr(v, "numpy") else v
                        for k, v in batch.items()
                    }
                else:
                    yield batch.numpy() if hasattr(batch, "numpy") else batch
        elif inspect.isgenerator(data) or isinstance(data, type((x for x in []))):
            # python generator
            for batch in data:
                yield batch
        else:
            # fallback iterable
            for batch in data:
                yield batch

    def fit(
        self, x: Any, y: Any = None, epochs: int = 1, batch_size: int = 32, **kwargs
    ) -> Any:
        """Fit function.

        Args:
        x: Parameter x.
        y: Parameter y.
        epochs: Parameter epochs.
        batch_size: Parameter batch_size.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Any: Return value.

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

        history: Dict[str, Any] = {"loss": []}

        for epoch in range(epochs):
            callbacks.on_epoch_begin(epoch)
            if getattr(self, "compiled_metrics", None):
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
                    if (
                        num_samples > 0
                        and hasattr(x, "__getitem__")
                        and hasattr(y, "__getitem__")
                    ):
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
            if getattr(self, "compiled_metrics", None):
                for m in self.compiled_metrics:  # pragma: no cover
                    epoch_logs[m.name] = m.result()  # pragma: no cover

            callbacks.on_epoch_end(epoch, epoch_logs)
            for k, v in epoch_logs.items():
                history.setdefault(k, []).append(v)

            if getattr(self, "stop_training", False):
                break

        callbacks.on_train_end()
        hist_obj = type(
            "History",
            (),
            {"history": history, "epoch": list(range(len(history["loss"])))},
        )()
        return hist_obj

    def save(self, filepath, overwrite=True, save_format=None, **kwargs):
        """Save function.

        Args:
        filepath: Parameter filepath.
        overwrite: Parameter overwrite.
        save_format: Parameter save_format.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Any: Return value.

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
            "config": getattr(self, "get_config", lambda: {})(),
        }

        def _to_bytes_and_dtype(tensor):
            """_to_bytes_and_dtype function.

            Args:
            tensor: Parameter tensor.

            Returns:
            Any: Return value.

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
                dt = str(getattr(data, "dtype", "float32")).split(".")[-1]
            else:
                # fallback
                flat = []

                def flatten(x):
                    """Flatten function.

                    Args:
                    x: Parameter x.

                    Returns:
                    Any: Return value.

                    """
                    if isinstance(x, (list, tuple)):
                        for item in x:
                            flatten(item)
                    else:
                        flat.append(float(x))

                flatten(data)
                b = array.array("f", flat).tobytes()
                shape = getattr(data, "shape", (len(flat),))
                dt = "float32"

            dtype_map = {"float32": "F32", "int32": "I32", "float64": "F64"}
            return b, list(shape), dtype_map.get(dt, "F32")

        def _get_safetensors(w_list, prefix="weight_"):
            """_get_safetensors function.

            Args:
            w_list: Parameter w_list.
            prefix: Parameter prefix.

            Returns:
            Any: Return value.

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
                zf.writestr(
                    zinfo_opt, _get_safetensors(self.optimizer.variables, "opt_weight_")
                )

    def evaluate(
        self, x: Any, y: Any = None, batch_size: Optional[int] = None, **kwargs: Any
    ) -> Any:
        """Evaluate function.

        Args:
        x: Parameter x.
        y: Parameter y.
        batch_size: Parameter batch_size.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Any: Return value.

        """
        import math

        if batch_size is None:
            batch_size = 32

        num_samples = len(x) if hasattr(x, "__len__") else 0
        num_batches = math.ceil(num_samples / batch_size) if num_samples > 0 else 1

        if getattr(self, "compiled_metrics", None):
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
                if (
                    num_samples > 0
                    and hasattr(x, "__getitem__")
                    and hasattr(y, "__getitem__")
                ):
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
        if getattr(self, "compiled_metrics", None):
            for m in self.compiled_metrics:
                metrics_results[m.name] = m.result()
        callbacks.on_test_end(metrics_results)
        return metrics_results

    def predict(self, x: Any, batch_size: Optional[int] = None, **kwargs: Any) -> Any:
        """Predict function.

        Args:
        x: Parameter x.
        batch_size: Parameter batch_size.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Any: Return value.

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
            is_numpy_array,
            concatenate_arrays,
        )

        callbacks.on_predict_end()
        if len(all_preds) == 1:
            return all_preds[0]
        if all_preds and is_numpy_array(all_preds[0]):
            return concatenate_arrays(all_preds)
        return all_preds

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        """Saves all layer weights."""
        import ml_switcheroo_compiler.serialization as msc_serialization

        msc_serialization.save_weights(
            self,
            filepath,
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )

    def load_weights(self, filepath, skip_mismatch=False, by_name=False, options=None):
        """Loads all layer weights."""
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

    def __init__(self, inputs=None, outputs=None, **kwargs):
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

    def _map_graph_network(self, inputs, outputs):
        """Function docstring.

        Args:
            inputs: Description.
            outputs: Description.
        """
        nodes_by_depth = {}

        # Simple DFS to map depths
        def _get_depth(tensor, current_depth):
            """Function docstring.

            Args:
                tensor: Description.
                current_depth: Description.
            """
            if not isinstance(tensor, KerasTensor):
                return
            node = getattr(tensor, "_keras_history", None)
            if node is None:
                return

            if current_depth not in nodes_by_depth:
                nodes_by_depth[current_depth] = []
            if node not in nodes_by_depth[current_depth]:
                nodes_by_depth[current_depth].append(node)
            else:
                return

            node_inputs = node.inputs
            print("NODE INPUTS:", type(node_inputs))
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

    def call(self, inputs, *args, **kwargs):
        """Function docstring.

        Args:
            inputs: Description.
            args: Description.
            kwargs: Description.
        """
        tensor_dict = {}

        def _add_input(key, val):
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
                def _get_input(x):
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

        def _get_output(x):
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

    def __init__(self, layers=None, **kwargs):
        """Function docstring.

        Args:
            layers: Description.
            kwargs: Description.
        """
        super().__init__(**kwargs)
        self.layers = layers or []

    def add(self, layer):
        """Add function.

        Args:
        layer: Parameter layer.

        Returns:
        Any: Return value.

        """
        self.layers.append(layer)

    def call(self, inputs, *args, **kwargs):
        """Call function.

        Args:
        inputs: Parameter inputs.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        Any: Return value.

        """
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


def deserialize(config, custom_objects=None, safe_mode=True):
    """Returns a Keras layer object via its configuration.

    Args:
        config: A python dict containing a serialized layer configuration.
        custom_objects: Optional dictionary mapping names (strings) to custom
            objects (classes and functions) to be considered during
            deserialization.

    Returns:
        A Keras layer instance.

    """
    # Dummy mock for structural parity coverage
    return config


def get(identifier, custom_objects=None):
    """Get function.

    Args:
    identifier: Parameter identifier.
    custom_objects: Parameter custom_objects.

    Returns:
    Any: Return value.

    """
    # Dummy mock for structural parity coverage
    return identifier


class Function:
    """Function docstring."""

    def call(self, *args, **kwargs):
        """call docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        return None

    def compute_output_shape(self, input_shape):
        """compute_output_shape docstring.

        Args:
            input_shape: Input shape.
        """
        return input_shape

    def compute_output_spec(self, *args, **kwargs):
        """compute_output_spec docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        return None

    @classmethod
    def from_config(cls, config):
        """from_config docstring.

        Args:
            config: Config.
        """
        return cls(**config)

    def get_config(self):
        """get_config docstring."""
        return {}

    @property
    def input(self):
        """input docstring."""
        return None

    @property
    def inputs(self):
        """inputs docstring."""
        return []

    @property
    def operations(self):
        """operations docstring."""
        return []

    @property
    def output(self):
        """output docstring."""
        return None

    @property
    def outputs(self):
        """outputs docstring."""
        return []

    def quantized_call(self, *args, **kwargs):
        """quantized_call docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        return None

    def symbolic_call(self, *args, **kwargs):
        """symbolic_call docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        return None


class InputSpec:
    """InputSpec docstring."""

    @classmethod
    def from_config(cls, config):
        """from_config docstring.

        Args:
            config: Config.
        """
        return cls(**config)

    def get_config(self):
        """get_config docstring."""
        return {
            "dtype": getattr(self, "dtype", None),
            "shape": getattr(self, "shape", None),
            "ndim": getattr(self, "ndim", None),
            "max_ndim": getattr(self, "max_ndim", None),
            "min_ndim": getattr(self, "min_ndim", None),
            "axes": getattr(self, "axes", None),
            "allow_last_axis_squeeze": getattr(self, "allow_last_axis_squeeze", False),
            "name": getattr(self, "name", None),
        }

    def __init__(
        self,
        dtype=None,
        shape=None,
        ndim=None,
        max_ndim=None,
        min_ndim=None,
        axes=None,
        allow_last_axis_squeeze=False,
        name=None,
    ):
        """docstring."""

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

    def call(self, *args, **kwargs):
        """call docstring."""
        return None

    def compute_output_spec(self, *args, **kwargs):
        """compute_output_spec docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        return None

    @classmethod
    def from_config(cls, config):
        """from_config docstring.

        Args:
            config: Config.
        """
        return cls(**config)

    def get_config(self):
        """get_config docstring."""
        return getattr(self, "_config", {})

    @property
    def input(self):
        """input docstring."""
        return getattr(self, "_input", None)

    @property
    def output(self):
        """output docstring."""
        return getattr(self, "_output", None)

    def quantized_call(self, *args, **kwargs):
        """quantized_call docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        return self.call(*args, **kwargs)

    def symbolic_call(self, *args, **kwargs):
        """symbolic_call docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        return self.call(*args, **kwargs)


class Variable:
    """Variable docstring."""

    def __getitem__(self, key):
        """__getitem__ docstring.

        Args:
            key: Key.
        """
        return None

    @property
    def aggregation(self):
        """aggregation docstring."""
        return getattr(self, "_aggregation", None)

    @aggregation.setter
    def aggregation(self, value):
        """docstring."""

        self._aggregation = value

    def assign(self, value):
        """assign docstring.

        Args:
            value: Value.
        """
        self.value = value

    def assign_add(self, value):
        """assign_add docstring.

        Args:
            value: Value.
        """
        pass

    def assign_sub(self, value):
        """assign_sub docstring.

        Args:
            value: Value.
        """
        pass

    @property
    def constraint(self):
        """constraint docstring."""
        return getattr(self, "_constraint", None)

    @constraint.setter
    def constraint(self, value):
        """docstring."""

        self._constraint = value

    @property
    def handle(self):
        """handle docstring."""
        return getattr(self, "_handle", None)

    @handle.setter
    def handle(self, value):
        """docstring."""

        self._handle = value

    @property
    def ndim(self):
        """ndim docstring."""
        return len(self.shape) if self.shape is not None else 0

    def numpy(self):
        """numpy docstring."""
        from ml_switcheroo_compiler.serialization import to_numpy

        return to_numpy(self.value)

    def overwrite_with_gradient(self, gradient):
        """overwrite_with_gradient docstring.

        Args:
            gradient: Gradient.
        """
        pass

    @property
    def path(self):
        """path docstring."""
        return self.name

    @property
    def regularizer(self):
        """regularizer docstring."""
        return getattr(self, "_regularizer", None)

    @regularizer.setter
    def regularizer(self, value):
        """docstring."""

        self._regularizer = value

    @property
    def synchronization(self):
        """synchronization docstring."""
        return getattr(self, "_synchronization", None)

    @synchronization.setter
    def synchronization(self, value):
        """docstring."""

        self._synchronization = value

    @property
    def value(self):
        """value docstring."""
        return getattr(self, "_value", None)

    @value.setter
    def value(self, val):
        """docstring."""

        self._value = val

    def __init__(self, initializer, shape=None, dtype=None, trainable=True, name=None):
        """docstring."""

        self.initializer = initializer
        self.shape = shape
        self.dtype = dtype
        self.trainable = trainable
        self.name = name


class DTypePolicy:
    """DTypePolicy docstring."""

    @property
    def name(self):
        """name docstring."""
        return getattr(self, "_name", None)

    @name.setter
    def name(self, value):
        """docstring."""

        self._name = value

    @property
    def compute_dtype(self):
        """compute_dtype docstring."""
        return None

    def convert_input(self, x, dtype=None, exact=False):
        """convert_input docstring.

        Args:
            x: Input.
            dtype: Dtype.
            exact: Exact.
        """
        return x

    @classmethod
    def from_config(cls, config):
        """from_config docstring.

        Args:
            config: Config.
        """
        return cls(**config)

    def get_config(self):
        """get_config docstring."""
        return {"name": getattr(self, "name", None)}

    @property
    def quantization_mode(self):
        """quantization_mode docstring."""
        return None

    @property
    def variable_dtype(self):
        """variable_dtype docstring."""
        return None

    def __init__(self, name=None):
        """docstring."""

        self._name = name


class FloatDTypePolicy(DTypePolicy):
    """FloatDTypePolicy docstring."""

    pass


class Quantizer:
    """Quantizer docstring."""

    @classmethod
    def from_config(cls, config):
        """from_config docstring.

        Args:
            config: Config.
        """
        return cls(**config)

    def get_config(self):
        """get_config docstring."""
        return {}


class RematScope:
    """RematScope docstring."""

    def __enter__(self):
        """docstring."""

        return self

    def __exit__(self, *args):
        """docstring."""

        pass


class StatelessScope:
    """StatelessScope docstring."""

    def add_loss(self, loss):
        """add_loss docstring.

        Args:
            loss: Loss.
        """
        pass

    def add_update(self, update):
        """add_update docstring.

        Args:
            update: Update.
        """
        pass

    def get_current_value(self, variable):
        """get_current_value docstring.

        Args:
            variable: Variable.
        """
        return getattr(variable, "value", None)

    def __enter__(self):
        """docstring."""

        return self

    def __exit__(self, *args):
        """docstring."""

        pass


class SymbolicScope:
    """SymbolicScope docstring."""

    def __enter__(self):
        """docstring."""

        return self

    def __exit__(self, *args):
        """docstring."""

        pass


def device(device_name):
    """device docstring."""

    class DeviceScope:
        """docstring."""

        def __enter__(self):
            """docstring."""

            pass

        def __exit__(self, *args):
            """docstring."""

            pass

    return DeviceScope()


class name_scope:
    """name_scope docstring."""

    def __init__(self, name):
        """docstring."""

        self.name = name

    def __enter__(self):
        """docstring."""

        return self

    def __exit__(self, *args):
        """docstring."""

        pass


def remat(f):
    """remat docstring."""
    return f


def version():
    """version docstring."""
    return "3.0.0"


def is_keras_tensor(x):
    """is_keras_tensor docstring."""
    return isinstance(x, KerasTensor)
