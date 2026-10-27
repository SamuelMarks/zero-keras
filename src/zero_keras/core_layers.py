"""Core Keras symbolic tensors, variables, base Layer, and Model abstractions."""

from __future__ import annotations

from zero_keras import ops as ops
from zero_keras.dtype_policies import (
    DTypePolicy as DTypePolicy,
    FloatDTypePolicy as FloatDTypePolicy,
)
from zero_keras.operations import (
    Function as Function,
    Node as Node,
    Operation as Operation,
    RematScope as RematScope,
    StatelessScope as StatelessScope,
    SymbolicScope as SymbolicScope,
)
from zero_keras.quantizers import Quantizer as Quantizer
from zero_keras.tensors import (
    Input as Input,
    InputSpec as InputSpec,
    KerasTensor as KerasTensor,
    KerasVariable as KerasVariable,
    TensorShape as TensorShape,
    Variable as Variable,
    is_keras_tensor as is_keras_tensor,
)
from zero_keras.types import (
    ComputeOutputShapeProtocol,
    DataCarrierProtocol,
    FromConfigProtocol,
    LossesCarrierProtocol,
    ShapeCarrierProtocol,
    SupportsAssign,
    SupportsShapeAndDType,
    TensorLike,
)


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
        """Create and return a ``Variable`` with the specified shape and initializer.

        Args:
            shape: Shape tuple specifying tensor dimensions.
            initializer: Initializer name, callable, or initial tensor value.
            dtype: Data type string or DTypePolicy.
            trainable: Whether the variable or layer weights may be updated by backpropagation.
            name: Optional string identifier for the object.

        Returns:
            TensorLike: Created ``Variable`` instance.
        """
        return Variable(initializer, shape=shape, dtype=dtype, trainable=trainable, name=name)

    def build_from_config(self, config: TensorLike) -> None:
        """Build the layer using the input shape stored in ``config``.

        Args:
            config: Build configuration dictionary containing ``input_shape``.
        """
        self.build(config.get("input_shape"))

    @property
    def compute_dtype(self) -> TensorLike:
        """Return the data type used for the layer's forward computations.

        Returns:
            TensorLike: Compute dtype string.
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
        if isinstance(self, ComputeOutputShapeProtocol):
            return self.compute_output_shape(*args, **kwargs)
        from zero_keras.tensors import KerasTensor

        return KerasTensor(shape=None, dtype=self.dtype or "float32")

    def count_params(self) -> TensorLike:
        """Return the total number of scalar parameters in this layer.

        Returns:
            TensorLike: Total scalar parameter count.
        """
        return 0

    @property
    def dtype(self) -> TensorLike:
        """Return the default data type of the layer's weights.

        Returns:
            TensorLike: Weight data type string.
        """
        return self._dtype

    @property
    def dtype_policy(self) -> TensorLike:
        """Return the ``DTypePolicy`` governing this layer's variables and computations.

        Returns:
            TensorLike: Active ``DTypePolicy`` instance or name.
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
        """Return the symbolic input tensor(s) connected to the first inbound node.

        Returns:
            TensorLike: Symbolic input tensor(s) or ``None``.
        """
        return self._inbound_nodes[0].input_tensors if self._inbound_nodes else None

    @property
    def input_dtype(self) -> TensorLike:
        """Return the expected input tensor data type for this layer.

        Returns:
            TensorLike: Input data type string.
        """
        return self.dtype

    def load_own_variables(self, store: TensorLike) -> None:
        """Load the layer's own weight variables from ``store``.

        Args:
            store: Mapping from stringified variable index to array values.
        """
        if not self.variables:
            return
        for i, var in enumerate(self.variables):
            if str(i) in store:
                var.assign(store[str(i)])

    @property
    def metrics(self) -> TensorLike:
        """Return the list of metrics tracked by this layer.

        Returns:
            TensorLike: List of tracked metric objects or dictionaries.
        """
        return self._metrics

    @property
    def metrics_variables(self) -> TensorLike:
        """Return the state variables belonging to this layer's metrics.

        Returns:
            TensorLike: List of metric state variables.
        """
        return []

    @property
    def non_trainable_variables(self) -> TensorLike:
        """Return the list of non-trainable variables tracked by this layer.

        Returns:
            TensorLike: List of non-trainable variables.
        """
        return self.non_trainable_weights

    @property
    def output(self) -> TensorLike:
        """Return the symbolic output tensor(s) produced by the first inbound node.

        Returns:
            TensorLike: Symbolic output tensor(s) or ``None``.
        """
        return self._inbound_nodes[0].output_tensors if self._inbound_nodes else None

    @property
    def path(self) -> TensorLike:
        """Return the hierarchical path identifier of this layer.

        Returns:
            TensorLike: Layer path string.
        """
        return self.name

    @property
    def quantization_mode(self) -> TensorLike:
        """Return the active quantization mode applied to this layer, if any.

        Returns:
            TensorLike: Quantization mode string or ``None``.
        """
        return self._quantization_mode

    def quantize(self, mode: str | TensorLike) -> TensorLike:
        """Apply post-training quantization to the layer or model weights in `mode`.

        Args:
            mode: Target quantization mode identifier (e.g., ``"int8"``, ``"float8"``).

        Returns:
            TensorLike: Quantized layer instance.
        """
        self._quantization_mode = mode
        return self

    def quantized_build(self, input_shape: TensorLike) -> None:
        """Build quantized weight variables for ``input_shape``.

        Args:
            input_shape: Input tensor shape tuple or structure of shapes.
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
        """Execute the forward pass with gradient rematerialization enabled.

        Args:
            *args: Positional arguments forwarded to ``call``.
            **kwargs: Keyword arguments forwarded to ``call``.

        Returns:
            TensorLike: Output tensor(s) from ``call``.
        """
        return self.call(*args, **kwargs)

    def save_own_variables(self, store: TensorLike) -> None:
        """Save this layer's own weight variable values into ``store``.

        Args:
            store: Mutable mapping populated with stringified variable indices and NumPy arrays.
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
            trainable_variables: Sequence of trainable variable values for the call.
            non_trainable_variables: Sequence of non-trainable variable values for the call.
            *args: Positional arguments forwarded to the operation or base class.
            **kwargs: Additional keyword configuration arguments.

        Returns:
            TensorLike: Output tensor(s) from the stateless call.
        """
        return self.call(*args, **kwargs)

    def symbolic_call(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Execute the symbolic graph construction pass for this layer.

        Args:
            *args: Positional symbolic arguments forwarded to ``call``.
            **kwargs: Keyword arguments forwarded to ``call``.

        Returns:
            TensorLike: Symbolic output tensor(s).
        """
        return self.call(*args, **kwargs)

    @property
    def trainable_variables(self) -> TensorLike:
        """Return the list of trainable weight variables tracked by this layer.

        Returns:
            TensorLike: List of trainable weight variables.
        """
        return self.trainable_weights

    @property
    def variable_dtype(self) -> TensorLike:
        """Return the data type used to allocate this layer's weight variables.

        Returns:
            TensorLike: Variable data type string.
        """
        return self.dtype

    @property
    def variables(self) -> TensorLike:
        """Return all trainable and non-trainable variables tracked by this layer.

        Returns:
            TensorLike: List of all layer variables.
        """
        return self.weights

    @property
    def input_spec(self) -> TensorLike:
        """Return the ``InputSpec`` constraints enforced on layer inputs.

        Returns:
            TensorLike: ``InputSpec`` instance, sequence of ``InputSpec``s, or ``None``.
        """
        return self._input_spec

    @input_spec.setter
    def input_spec(self, value: TensorLike) -> None:
        """Set the ``InputSpec`` constraints enforced on layer inputs.

        Args:
            value: ``InputSpec`` instance, sequence of ``InputSpec``s, or ``None``.
        """
        self._input_spec = value

    @property
    def supports_masking(self) -> TensorLike:
        """Return whether this layer supports propagating input masks.

        Returns:
            TensorLike: ``True`` if masking is supported, else ``False``.
        """
        return self._supports_masking

    @supports_masking.setter
    def supports_masking(self, value: TensorLike) -> None:
        """Set whether this layer supports propagating input masks.

        Args:
            value: Boolean flag indicating mask support.
        """
        self._supports_masking = value

    @property
    def trainable(self) -> TensorLike:
        """Return whether this layer's weights can be updated during training.

        Returns:
            TensorLike: ``True`` if trainable, else ``False``.
        """
        return self._trainable

    @trainable.setter
    def trainable(self, value: TensorLike) -> None:
        """Set whether this layer's weights can be updated during training.

        Args:
            value: Boolean flag enabling or disabling weight updates.
        """
        self._trainable = value

    def __new__(cls, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Allocate a new ``Layer`` instance with initialized tracking state.

        Args:
            *args: Positional constructor arguments.
            **kwargs: Keyword constructor arguments.

        Returns:
            TensorLike: Allocated ``Layer`` instance.
        """
        _ = args, kwargs
        inst = super().__new__(cls)
        super().__setattr__(inst, "_tracked_attrs", {})
        return inst

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
        self._symbolic_built: bool = False
        self._tracing_build: bool = False
        self._symbolic_input_shape: TensorLike | None = None

    def __setattr__(self, name: str, value: TensorLike) -> None:
        """Set attribute ``name`` to ``value`` and register potential sublayers.

        Args:
            name: Attribute name.
            value: Attribute value.
        """
        super().__setattr__(name, value)
        if name != "_tracked_attrs" and not name.startswith("_"):
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
        """Return all regularization and activity losses registered on this layer and its sublayers.

        Returns:
            TensorLike: List of scalar loss tensors.
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
                    if isinstance(item, LossesCarrierProtocol):
                        losses_list.extend(item.losses)
        return losses_list

    def compute_output_shape(self, input_shape: TensorLike) -> TensorLike:
        """Compute the output tensor shape produced for `input_shape`.

        Args:
            input_shape: Input tensor shape tuple or nested structure of shape tuples.

        Returns:
            TensorLike: Computed result for `compute_output_shape`.
        """
        from zero_keras.tensors import KerasTensor

        if isinstance(input_shape, list):
            inputs = [KerasTensor(shape=s) for s in input_shape]
        elif isinstance(input_shape, tuple):
            inputs = KerasTensor(shape=input_shape)
        else:
            inputs = KerasTensor(shape=None)

        try:
            out = self.call(inputs)
            if isinstance(out, list):
                return [(x.shape if isinstance(x, ShapeCarrierProtocol) else None) for x in out]
            elif isinstance(out, tuple):
                return tuple(
                    (x.shape if isinstance(x, ShapeCarrierProtocol) else None) for x in out
                )
            return out.shape if isinstance(out, ShapeCarrierProtocol) else None
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
            initializer: Initializer name, callable, or ``Initializer`` instance.
            trainable: Whether the variable or layer weights may be updated by backpropagation.
            name: Optional string identifier for the object.
            regularizer: Optional weight regularizer name or callable.
            constraint: Optional weight constraint name or callable.
            **kwargs: Additional keyword configuration arguments.

        Returns:
            TensorLike: Created and registered weight tensor.
        """
        if self._tracing_build:
            norm_s = (int(shape),) if isinstance(shape, int) else tuple(shape)
            weight: TensorLike = KerasTensor(shape=None, dtype=self.dtype or "float32")
            weight.shape = norm_s
            self._weights.append(weight)
            return weight

        from zero_keras.initializers import get as get_initializer

        init_fn = get_initializer(initializer)
        weight = init_fn(shape=shape)
        self._weights.append(weight)
        return weight

    def _materialize_symbolic_weights(self) -> None:
        """Materialize eager weight tensors for this layer and its sublayers if symbolically built."""
        if self._symbolic_built:
            self.built = False
            for k, val in list(self._tracked_attrs.items()):
                if isinstance(val, KerasTensor):
                    super().__setattr__(k, None)
                    self._tracked_attrs[k] = None
            self._weights.clear()
            self._symbolic_built = False
            self.build(self._symbolic_input_shape)
        for v in self._tracked_attrs.values():
            if isinstance(v, Layer):
                v._materialize_symbolic_weights()
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, Layer):
                        item._materialize_symbolic_weights()

    @property
    def weights(self) -> TensorLike:
        """Return the deduplicated list of all weights in this layer and its sublayers.

        Returns:
            TensorLike: List of weight variables in topological registration order.
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
        """Return the list of trainable weight variables in this layer and its sublayers.

        Returns:
            TensorLike: List of trainable weight variables.
        """
        return self.weights

    @property
    def non_trainable_weights(self) -> TensorLike:
        """Return the list of non-trainable weight variables in this layer and its sublayers.

        Returns:
            TensorLike: List of non-trainable weight variables.
        """
        return []

    def get_weights(self) -> TensorLike:
        """Return a list of NumPy arrays representing all layer weight values.

        Returns:
            TensorLike: Computed result for `get_weights`.
        """
        self._materialize_symbolic_weights()
        from ml_switcheroo_compiler.serialization import to_numpy

        return [to_numpy(w) for w in self.weights]

    def set_weights(self, weights: TensorLike) -> TensorLike:
        """Set the values of all layer weights from a list of NumPy arrays.

        Args:
            weights: Sequence of NumPy arrays matching the shapes of ``self.weights``.
        """
        self._materialize_symbolic_weights()
        if len(weights) != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} weights, got {len(weights)}")
        from ml_switcheroo_compiler.ops import asarray

        for w, new_w in zip(self.weights, weights):
            if isinstance(w, SupportsAssign):
                w.assign(new_w)
            elif isinstance(w, DataCarrierProtocol):
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
        # Simple tracing
        is_tracing = False

        def check_tracing(x: TensorLike) -> None:
            """Check recursively whether ``x`` contains any symbolic ``KerasTensor``.

            Args:
                x: Input tensor or nested structure of tensors.
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

        if not self.built:
            in_shape = inputs.shape if isinstance(inputs, ShapeCarrierProtocol) else None
            self._symbolic_input_shape = in_shape
            self._tracing_build = is_tracing
            try:
                self.build(in_shape)
            finally:
                self._tracing_build = False
            self._symbolic_built = is_tracing
        elif self._symbolic_built and not is_tracing:
            self._materialize_symbolic_weights()

        if not is_tracing:
            self._losses = []

        outputs = self.call(inputs, *args, **kwargs)

        if is_tracing:
            node = Node(operation=self, call_args=inputs, call_kwargs=kwargs)

            def wrap_output(x: TensorLike) -> TensorLike:
                """Wrap eager or symbolic output ``x`` in a ``KerasTensor`` linked to ``node``.

                Args:
                    x: Output tensor produced by ``self.call``.

                Returns:
                    TensorLike: Symbolic ``KerasTensor`` with ``_keras_history`` attached.
                """
                if isinstance(x, SupportsShapeAndDType) and not isinstance(x, KerasTensor):
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

    if cls and isinstance(cls, FromConfigProtocol):
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


def device(device_name: TensorLike) -> TensorLike:
    """Return a context manager placing operations on `device_name`.

    Args:
        device_name: Target device specification string (e.g., ``"cpu:0"``, ``"gpu:0"``).

    Returns:
        TensorLike: Context manager scoping operations to ``device_name``.
    """

    class DeviceScope:
        """Context manager scoping tensor placement to a target device."""

        def __init__(self, name: TensorLike) -> None:
            """Initialize the `DeviceScope` instance.

            Args:
                name: Optional string identifier for the object.
            """
            self.name = name

        def __enter__(self) -> TensorLike:
            """Enter the device placement scope.

            Returns:
                TensorLike: Active ``DeviceScope`` instance.
            """
            return self

        def __exit__(self, *args: TensorLike) -> TensorLike:
            """Exit the device placement scope.

            Args:
                *args: Exception type, value, and traceback if an error occurred.
            """

    return DeviceScope(device_name)


class name_scope:
    """Context manager prepending a hierarchical name prefix to enclosed operations."""

    def __init__(self, name: TensorLike) -> None:
        """Initialize the `name_scope` instance.

        Args:
            name: Optional string identifier for the object.
        """
        self.name = name

    def __enter__(self) -> TensorLike:
        """Enter the name scope context.

        Returns:
            TensorLike: Active ``name_scope`` instance.
        """
        return self

    def __exit__(self, *args: TensorLike) -> TensorLike:
        """Exit the name scope context.

        Args:
            *args: Exception type, value, and traceback if an error occurred.
        """


def remat(f: TensorLike) -> TensorLike:
    """Wrap function `f` with gradient checkpointing (rematerialization).

    Args:
        f: Callable function or layer computation to rematerialize during backprop.

    Returns:
        TensorLike: Wrapped callable with gradient rematerialization enabled.
    """
    return f


def version() -> TensorLike:
    """Return the current `zero_keras` package version string.

    Returns:
        TensorLike: Computed result for `version`.
    """
    return "3.0.0"


class Wrapper(Layer):
    """Base class for layer wrappers."""

    def __init__(self, layer: TensorLike, **kwargs: TensorLike) -> None:
        """Initialize the `Wrapper` instance.

        Args:
            layer: Wrapped ``Layer`` instance to delegate computation to.
            **kwargs: Additional keyword configuration arguments.
        """
        super().__init__(**kwargs)
        self.layer = layer


from zero_keras.models import (
    Functional as Functional,
    Model as Model,
    Sequential as Sequential,
)
