"""Module docstring."""


class TensorShape(tuple):
    """TensorShape class."""

    pass


from zero_keras.activations import _to_tensor

"""Core layers module."""

from typing import Any, Dict, Optional
from ml_switcheroo_compiler import ops as backend_ops


class KerasTensor:
    """docstring."""

    def __init__(self, shape: Any, dtype: str = "float32", name: Any = None, data=None):
        self.shape = shape
        self.dtype = dtype
        self.name = name
        if data is not None:
            self.data = data
        else:
            try:
                self.data = backend_ops.zeros(shape) if shape is not None else None
            except Exception:
                self.data = None

    def __add__(self, other: Any) -> Any:
        """__add__ function.

        Args:
        other: Parameter other.

        Returns:
        Any: Return value.

        """
        return KerasTensor(self.shape, self.dtype)

    def __sub__(self, other: Any) -> Any:
        """__sub__ function.

        Args:
        other: Parameter other.

        Returns:
        Any: Return value.

        """
        return KerasTensor(self.shape, self.dtype)

    def __mul__(self, other: Any) -> Any:
        """__mul__ function.

        Args:
        other: Parameter other.

        Returns:
        Any: Return value.

        """
        return KerasTensor(self.shape, self.dtype)

    def __truediv__(self, other: Any) -> Any:
        """__truediv__ function.

        Args:
        other: Parameter other.

        Returns:
        Any: Return value.

        """
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
        return backend_ops.ones(self.shape)

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
        arr = backend_ops.asarray(
            self.data if self.data is not None else 0.0, dtype=dtype
        )
        if hasattr(arr, "__array__"):
            return arr.__array__(dtype=dtype)
        if hasattr(arr, "numpy"):
            return arr.numpy()
        return arr

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
    return KerasTensor(shape, "float32", name=name)


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

    def __init__(self, **kwargs: Any):
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
        return getattr(self, "_losses", [])

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
        return self.call(inputs, *args, **kwargs)


class Model(Layer):
    """docstring."""

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
        import ml_switcheroo_compiler.ops as ops

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
        if hasattr(self, "compiled_metrics"):
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
            grads_list = [grads_dict.get(v.id, None) for v in trainable_vars]
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

    def __init__(self, inputs: Any = None, outputs: Any = None, **kwargs: Any):
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
        self.optimizer = optimizer
        self.loss = loss
        if loss == "mse":
            import ml_switcheroo_compiler.ops as ops

            def mse_loss(y_true, y_pred):
                if isinstance(y_pred, dict):
                    # just take the first one or sum them, but for mock, let's just pick one
                    y_pred = list(y_pred.values())[0]
                if y_true is None:
                    y_true = ops.zeros(y_pred.shape)
                return ops.mean(ops.square(y_true - y_pred))

            self.loss_fn = mse_loss
        else:
            self.loss_fn = loss if callable(loss) else None  # type: ignore
        self.compiled_metrics = metrics or []

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

        history: Dict[str, Any] = {"loss": []}

        for epoch in range(epochs):
            if hasattr(self, "compiled_metrics"):
                for m in self.compiled_metrics:
                    m.reset_state()

            epoch_loss_sum = 0.0
            batches_seen = 0

            if self._is_iterator(x):
                iterator = self._unpack_iterator(x, is_train=True)
                for batch_data in iterator:
                    logs = self.train_step(batch_data)
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

            history["loss"].append(epoch_loss_sum / max(1, batches_seen))

        return type("History", (), {"history": history})()

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

        if hasattr(self, "compiled_metrics"):
            for m in self.compiled_metrics:
                m.reset_state()

        epoch_loss_sum = 0.0

        batches_seen = 0
        if self._is_iterator(x):
            iterator = self._unpack_iterator(x, is_train=False)
            for batch_data in iterator:
                logs = self.test_step(batch_data)
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
        if hasattr(self, "compiled_metrics"):
            for m in self.compiled_metrics:
                metrics_results[m.name] = m.result()
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

        all_preds = []

        if self._is_iterator(x):
            iterator = self._unpack_iterator(x, is_train=False)
            for batch_data in iterator:
                preds = self.predict_step(
                    batch_data if isinstance(batch_data, tuple) else (batch_data,)
                )
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

        if len(all_preds) == 1:
            return all_preds[0]
        if all_preds and is_numpy_array(all_preds[0]):
            return concatenate_arrays(all_preds)
        return all_preds


class Functional(Model):
    """Functional class."""

    pass


class Sequential(Model):
    """Sequential class."""

    def __init__(self, layers=None, **kwargs):
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


class ops:
    """docstring."""

    @staticmethod
    def add(x: Any, y: Any) -> Any:
        """Add function.

        Args:
        x: Parameter x.
        y: Parameter y.

        Returns:
        Any: Return value.

        """
        return KerasTensor(getattr(x, "shape", None))


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
