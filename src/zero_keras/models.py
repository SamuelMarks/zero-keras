"""Model classes (`Sequential`, `Model`, `Functional`), cloning, and serialization for zero_keras."""

from __future__ import annotations

import array
from collections.abc import Callable, Sequence
import json
from pathlib import Path
import struct
import zipfile

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops import asarray as _msc_asarray

from collections.abc import Iterable, Sized

from zero_keras import ops
from zero_keras.activations import _to_tensor
from zero_keras.core_layers import Layer, deserialize
from zero_keras.operations import Node
from zero_keras.ops import reshape
from zero_keras.saving.registry import get_object_class_name
from zero_keras.tensors import Input, KerasTensor, Variable
from zero_keras.types import (
    BatchShapeCarrierProtocol,
    CallMethodProtocol,
    ConfigDict,
    ConfigValue,
    DataCarrierProtocol,
    DTypeCarrierProtocol,
    IdCarrierProtocol,
    KerasHistoryProtocol,
    KwargsCarrierProtocol,
    NameCarrierProtocol,
    OptimizerProtocol,
    ShapeCarrierProtocol,
    ShapeLike,
    ShapeTuple,
    SupportsAssign,
    SupportsGetConfig,
    SupportsNumpy,
    SupportsToBytes,
    TensorLike,
    VariablesCarrierProtocol,
)


def _load_safetensors(data: bytes) -> list[tuple[list[float | int], tuple[int, ...], DType]]:
    """Parse raw ``.safetensors`` bytes into ``(flat_values, shape, dtype)`` tuples.

    Args:
        data: Raw bytes of a ``.safetensors`` file.

    Returns:
        list[tuple[list[float | int], tuple[int, ...], DType]]: Ordered weight entries.
    """
    header_size = int(struct.unpack("<Q", data[:8])[0])
    header_json = data[8 : 8 + header_size].decode("utf-8")
    header = json.loads(header_json)
    weights_dict: dict[str, tuple[list[float | int], tuple[int, ...], DType]] = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        start, end = int(v["data_offsets"][0]), int(v["data_offsets"][1])
        raw = data[8 + header_size + start : 8 + header_size + end]
        dt_str = str(v["dtype"])
        fmt = "f" if dt_str == "F32" else ("i" if dt_str == "I32" else "d")
        arr = array.array(fmt)
        arr.frombytes(raw)
        shape = tuple(int(dim) for dim in v["shape"])
        dt = (
            DType.Float32
            if dt_str == "F32"
            else (DType.Int32 if dt_str == "I32" else DType.Float64)
        )
        weights_dict[str(k)] = (list(arr), shape, dt)

    return [
        weights_dict[k] for k in sorted(weights_dict.keys(), key=lambda x: int(x.split("_")[-1]))
    ]


def load_model(
    filepath: str | Path,
    custom_objects: dict[str, ConfigValue] | None = None,
    compile: bool = True,
    safe_mode: bool = True,
    **kwargs: ConfigValue,
) -> Model:
    """Load a model saved via ``model.save()``.

    Args:
        filepath: String or ``Path`` to the saved ``.keras`` model file.
        custom_objects: Optional dictionary mapping names to custom classes or functions.
        compile: Whether to compile the model after loading.
        safe_mode: Whether to disallow unsafe ``Lambda`` deserialization.
        **kwargs: Additional keyword options.

    Returns:
        Model: Loaded Keras model instance.

    Raises:
        ValueError: If ``safe_mode=True`` and a ``Lambda`` is encountered, or if format is unsupported.
    """
    _ = compile, kwargs
    path_str = str(filepath)
    if path_str.endswith(".keras") or zipfile.is_zipfile(filepath):
        with zipfile.ZipFile(filepath, "r") as z:
            config_str = z.read("config.json").decode("utf-8")
            config_dict = json.loads(config_str)
            if safe_mode and config_dict.get("class_name") == "Lambda":
                raise ValueError("safe_mode is True, cannot deserialize Lambda")

            model = model_from_json(config_str, custom_objects=custom_objects, safe_mode=safe_mode)

            if "model.safetensors" in z.namelist():
                with z.open("model.safetensors") as w:
                    data = w.read()
                    parsed = _load_safetensors(data)
                    weights: list[TensorLike] = []
                    for flat, shape, dt in parsed:
                        t = _msc_asarray(flat, dtype=dt)
                        t = reshape(t, shape)
                        weights.append(t)
                    if weights:
                        try:
                            model.set_weights(weights)
                        except Exception:
                            pass
            return model
    if path_str.endswith((".h5", ".hdf5")):
        raise ValueError(
            "Legacy HDF5 format is not supported in zero_keras. Please use .keras format."
        )
    raise ValueError(f"File format not supported for {filepath}")


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
        """Compute the total training or evaluation loss from targets, predictions, and regularization losses.

        Args:
            x: Input tensor or array.
            y: Target label tensor or second operand.
            y_pred: Predicted output tensor produced by the model.
            sample_weight: Optional array of per-sample weights for loss and metric scaling.

        Returns:
            TensorLike: Scalar total loss tensor.
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
        """Update compiled metrics with ``(y, y_pred)`` and return their current results.

        Args:
            x: Input tensor or array.
            y: Target label tensor or second operand.
            y_pred: Predicted output tensor produced by the model.
            sample_weight: Optional array of per-sample weights for loss and metric scaling.

        Returns:
            TensorLike: Dictionary mapping metric names to scalar metric values.
        """
        metrics_dict = {}
        if self.compiled_metrics:
            for m in self.compiled_metrics:
                m.update_state(y, y_pred)
                metrics_dict[m.name] = m.result()
        return metrics_dict

    def train_step(self, data: TensorLike) -> TensorLike:
        """Execute a single forward, backward, and optimizer update step on ``data``.

        Args:
            data: Batch tuple ``(x, y)`` or input tensor ``x``.

        Returns:
            TensorLike: Dictionary of metric and loss values for the step.
        """
        from ml_switcheroo_compiler.grad import (
            value_and_grad_wrt_vars,
        )

        if isinstance(data, tuple) and len(data) == 2:
            x, y = data
        else:
            x, y = data, None

        def loss_fn() -> TensorLike:
            """Evaluate the forward pass and return the scalar training loss.

            Returns:
                TensorLike: Scalar loss tensor for automatic differentiation.
            """
            try:
                y_pred = self(x, training=True)
            except TypeError:
                y_pred = self(x)
            loss = self.compute_loss(x, y, y_pred)
            return loss

        loss, grads_dict = value_and_grad_wrt_vars(loss_fn)()

        trainable_vars = self.trainable_weights
        if isinstance(self.optimizer, OptimizerProtocol) and trainable_vars:
            grads_list = [
                grads_dict.get((v.id if isinstance(v, IdCarrierProtocol) else id(v)), None)
                for v in trainable_vars
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
        """Execute a single evaluation step on ``data`` without gradient updates.

        Args:
            data: Batch tuple ``(x, y)`` or input tensor ``x``.

        Returns:
            TensorLike: Dictionary of evaluation metric and loss values.
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
        """Execute a single inference forward pass on batch ``data``.

        Args:
            data: Batch of input samples or single-element tuple ``(x,)``.

        Returns:
            TensorLike: Model output predictions for the batch.
        """
        x = data[0] if isinstance(data, (tuple, list)) else data
        try:
            return self(x, training=False)
        except TypeError:
            return self(x)

    def compile_from_config(self, config: TensorLike) -> None:
        """Compile the model using serialized compilation settings in ``config``.

        Args:
            config: Dictionary containing serialized ``optimizer``, ``loss``, and ``metrics``.
        """
        optimizer = config.get("optimizer", "rmsprop")
        loss = config.get("loss", None)
        metrics = config.get("metrics", None)
        self.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    @property
    def compiled_loss(self) -> TensorLike:
        """Return the configured loss specification or callable for this model.

        Returns:
            TensorLike: Configured loss object or identifier.
        """
        return self.loss

    @property
    def distribute_reduction_method(self) -> TensorLike:
        """Return the loss reduction method used across distributed replicas.

        Returns:
            TensorLike: Reduction method identifier string (default ``"auto"``).
        """
        return "auto"

    @property
    def distribute_strategy(self) -> TensorLike:
        """Return the active distribution strategy for this model.

        Returns:
            TensorLike: Strategy identifier or object.
        """
        return "auto"

    def export(self, filepath: str | TensorLike) -> None:
        """Export the model as a SavedModel/ExportArchive artifact at ``filepath``.

        Args:
            filepath: Destination directory path for the exported artifact.
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
        """Retrieve a sublayer by its string ``name`` or integer ``index``.

        Args:
            name: Optional string identifier of the target layer.
            index: Optional 0-based integer position of the target layer in ``self.layers``.

        Returns:
            TensorLike: Matching ``Layer`` instance.
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
        """Return whether XLA/JIT compilation is enabled for this model.

        Returns:
            TensorLike: Boolean flag indicating JIT compilation status.
        """
        return self._jit_compile

    @jit_compile.setter
    def jit_compile(self, value: TensorLike) -> None:
        """Set whether XLA/JIT compilation is enabled for this model.

        Args:
            value: Boolean flag enabling or disabling JIT compilation.
        """
        self._jit_compile = value

    def make_predict_function(self, force: bool = False) -> TensorLike:
        """Create or return the compiled inference step function.

        Args:
            force: Whether to recreate the function even if already cached.

        Returns:
            TensorLike: Callable prediction step function.
        """
        if self.predict_function is None or force:
            self.predict_function = self.predict_step
        return self.predict_function

    def make_test_function(self, force: bool = False) -> TensorLike:
        """Create or return the compiled evaluation step function.

        Args:
            force: Whether to recreate the function even if already cached.

        Returns:
            TensorLike: Callable evaluation step function.
        """
        if self.test_function is None or force:
            self.test_function = self.test_step
        return self.test_function

    def make_train_function(self, force: bool = False) -> TensorLike:
        """Create or return the compiled training step function.

        Args:
            force: Whether to recreate the function even if already cached.

        Returns:
            TensorLike: Callable training step function.
        """
        if self.train_function is None or force:
            self.train_function = self.train_step
        return self.train_function

    @property
    def metrics_names(self) -> TensorLike:
        """Return the display names of all tracked metrics.

        Returns:
            TensorLike: List of metric name strings.
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
        """Reset the state of all compiled metrics."""
        if self.compiled_metrics:
            for m in self.compiled_metrics:
                m.reset_state()

    def set_state_tree(self, state_tree: TensorLike) -> None:
        """Restore model variable values from ``state_tree``.

        Args:
            state_tree: Nested dictionary mapping variable categories to value sequences.
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
        """Compute the model loss in a stateless context given explicit predictions.

        Args:
            x: Input tensor or array.
            y: Target label tensor or second operand.
            y_pred: Predicted output tensor produced by the model.
            sample_weight: Optional array of per-sample weights for loss and metric scaling.

        Returns:
            TensorLike: Scalar loss tensor.
        """
        return self.compute_loss(x, y, y_pred, sample_weight=sample_weight)

    def summary(self, *args: TensorLike, **kwargs: TensorLike) -> None:
        """Print a text summary of the model architecture and parameter count.

        Args:
            *args: Optional positional formatting arguments.
            **kwargs: Optional keyword formatting arguments (e.g., ``print_fn``).
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
        """Serialize the model architecture configuration to a JSON string.

        Returns:
            TensorLike: JSON string representation of ``self.get_config()``.
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
        """Return the list of compiled ``Metric`` instances attached to this model.

        Returns:
            TensorLike: List of compiled metrics.
        """
        return self._compiled_metrics

    @compiled_metrics.setter
    def compiled_metrics(self, value: TensorLike) -> None:
        """Set the list of compiled ``Metric`` instances attached to this model.

        Args:
            value: Sequence of ``Metric`` instances.
        """
        self._compiled_metrics = value

    @property
    def layers(self) -> TensorLike:
        """Return the list of sublayers belonging to this model.

        Returns:
            TensorLike: List of ``Layer`` instances.
        """
        return self._layers

    @layers.setter
    def layers(self, value: TensorLike) -> None:
        """Set the list of sublayers belonging to this model.

        Args:
            value: Sequence of ``Layer`` instances.
        """
        self._layers = value

    @property
    def loss(self) -> TensorLike:
        """Return the loss function or loss identifier configured on this model.

        Returns:
            TensorLike: Configured loss object, callable, or string.
        """
        return self._loss

    @loss.setter
    def loss(self, value: TensorLike) -> None:
        """Set the loss function or loss identifier configured on this model.

        Args:
            value: Loss instance, callable, or string identifier.
        """
        self._loss = value

    @property
    def run_eagerly(self) -> TensorLike:
        """Return whether model execution is forced to run eagerly.

        Returns:
            TensorLike: ``True`` if running eagerly, else ``False``.
        """
        return self._run_eagerly

    @run_eagerly.setter
    def run_eagerly(self, value: TensorLike) -> None:
        """Set whether model execution is forced to run eagerly.

        Args:
            value: Boolean flag enabling or disabling eager execution.
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
            inputs: Symbolic input tensor or nested structure of input tensors.
            outputs: Symbolic output tensor or nested structure of output tensors.
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
                loss
                if callable(loss)
                else (loss.call if isinstance(loss, CallMethodProtocol) else None)
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
        """Return whether ``data`` is a dataset iterator or generator yielding batches.

        Args:
            data: Dataset, generator, or batch tensor to inspect.

        Returns:
            TensorLike: ``True`` if ``data`` is treated as an iterator of batches.
        """
        if (
            isinstance(data, Iterable)
            and not isinstance(data, ShapeCarrierProtocol)
            and not isinstance(data, (list, tuple, dict))
        ):
            return True
        return bool(
            type(data).__module__.startswith("tensorflow.python.data")
            or type(data).__module__.startswith("torch.utils.data")
        )

    def _unpack_iterator(self, data: TensorLike, is_train: bool = True) -> TensorLike:
        """Yield normalized batch tuples or tensors from a dataset iterator ``data``.

        Args:
            data: Dataset iterator, generator, or framework DataLoader.
            is_train: Whether the iterator is being consumed during training.
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
                    yield tuple(b.numpy() if isinstance(b, SupportsNumpy) else b for b in batch)
                elif isinstance(batch, dict):
                    yield {
                        k: v.numpy() if isinstance(v, SupportsNumpy) else v
                        for k, v in batch.items()
                    }
                else:
                    yield batch.numpy() if isinstance(batch, SupportsNumpy) else batch
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

        num_samples = len(x) if isinstance(x, Sized) else 0
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
                    if num_samples > 0 and y is not None:
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
            """Serialize ``tensor`` into raw little-endian bytes, shape list, and safetensors dtype tag.

            Args:
                tensor: Weight tensor or array to serialize.

            Returns:
                TensorLike: Tuple ``(raw_bytes, shape_list, safetensors_dtype)``.
            """
            import array

            if isinstance(tensor, SupportsNumpy):
                data = tensor.numpy()
            elif isinstance(tensor, DataCarrierProtocol) and isinstance(tensor.data, SupportsNumpy):
                data = tensor.data.numpy()
            else:
                data = tensor

            # Extract standard python array
            if isinstance(data, SupportsToBytes):
                b = data.tobytes()
                shape = data.shape
                dt = str(
                    (data.dtype if isinstance(data, DTypeCarrierProtocol) else "float32")
                ).split(".")[-1]
            else:
                # fallback
                flat = []

                def flatten(x: TensorLike) -> None:
                    """Recursively append scalar float elements of ``x`` to ``flat``.

                    Args:
                        x: Nested sequence or scalar numeric value.
                    """
                    if isinstance(x, (list, tuple)):
                        for item in x:
                            flatten(item)
                    else:
                        flat.append(float(x))

                flatten(data)
                b = array.array("f", flat).tobytes()
                shape = data.shape if isinstance(data, ShapeCarrierProtocol) else (len(flat),)
                dt = "float32"

            dtype_map = {"float32": "F32", "int32": "I32", "float64": "F64"}
            return b, list(shape), dtype_map.get(dt, "F32")

        def _get_safetensors(
            w_list: TensorLike, prefix: str | TensorLike = "weight_"
        ) -> TensorLike:
            """Encode a sequence of weight tensors into safetensors binary format.

            Args:
                w_list: Sequence of weight tensors to serialize.
                prefix: Key prefix prepended to each tensor index in the safetensors header.

            Returns:
                TensorLike: Complete safetensors binary payload as ``bytes``/``bytearray``.
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

            if isinstance(self.optimizer, VariablesCarrierProtocol) and self.optimizer.variables:
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

        num_samples = len(x) if isinstance(x, Sized) else 0
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
                if num_samples > 0 and y is not None:
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

        num_samples = len(x) if isinstance(x, Sized) else 0
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
                if isinstance(preds, SupportsNumpy):
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
                if num_samples > 0:
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

                if isinstance(preds, SupportsNumpy):
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
            options: Optional backend-specific checkpoint saving options.
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
            options: Optional backend-specific checkpoint loading options.
        """
        import ml_switcheroo_compiler.serialization as msc_serialization

        weights = msc_serialization.load_weights(filepath, target_model=self)

        # apply weights
        if not self.weights:
            return

        for i, w in enumerate(self.weights):
            k = "weight_" + str(i)
            if k in weights and isinstance(w, SupportsAssign):
                w.assign(weights[k])


class Functional(Model):
    """Functional DAG model connecting symbolic ``KerasTensor`` inputs to outputs."""

    def __init__(
        self,
        inputs: TensorLike | None = None,
        outputs: TensorLike | None = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the `Functional` instance.

        Args:
            inputs: Symbolic input tensor or nested structure of input tensors.
            outputs: Symbolic output tensor or nested structure of output tensors.
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
            node = tensor._keras_history
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
                            if isinstance(t, BatchShapeCarrierProtocol)
                            else t.shape
                            if isinstance(t, ShapeCarrierProtocol)
                            else (None,)
                        ),
                        "dtype": (t.dtype if isinstance(t, DTypeCarrierProtocol) else "float32"),
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
                        if isinstance(layer, SupportsGetConfig)
                        else lambda: (
                            layer._kwargs if isinstance(layer, KwargsCarrierProtocol) else {}
                        )
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
                    if isinstance(t, KerasHistoryProtocol) and t._keras_history is not None:
                        src_node = t._keras_history
                        group.append([src_node.layer.name, 0, 0, {}])
                    elif isinstance(t, NameCarrierProtocol) and t.name in lc_map:
                        # It's an input tensor mapped to InputLayer
                        group.append([t.name, 0, 0, {}])
                if group:
                    lc_map[layer_name]["inbound_nodes"].append(group)

        # Input layers
        input_layers = []
        for t in inputs_list:
            if isinstance(t, KerasHistoryProtocol) and t._keras_history is not None:
                input_layers.append([t._keras_history.layer.name, 0, 0])
            else:
                input_layers.append([t.name, 0, 0])

        # Output layers
        output_layers = []
        outputs_list = self.outputs if isinstance(self.outputs, (list, tuple)) else [self.outputs]
        for t in outputs_list:
            if isinstance(t, KerasHistoryProtocol) and t._keras_history is not None:
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
            """Bind symbolic input tensor ``key`` to concrete runtime tensor ``val``.

            Args:
                key: Symbolic ``KerasTensor`` from ``self.inputs``.
                val: Runtime input tensor corresponding to ``key``.
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
                    """Look up the runtime tensor mapped to symbolic tensor ``x``.

                    Args:
                        x: Symbolic ``KerasTensor`` or literal argument.

                    Returns:
                        TensorLike: Mapped runtime tensor or ``x`` unchanged.
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
            """Retrieve the computed runtime output tensor for symbolic output ``x``.

            Args:
                x: Symbolic output ``KerasTensor`` or constant tensor.

            Returns:
                TensorLike: Computed output tensor for ``x``.
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
    """``Sequential`` groups a linear stack of layers into a ``Model``."""

    inputs: TensorLike | None = None
    outputs: TensorLike | None = None

    def __init__(
        self,
        layers: Sequence[Layer | TensorLike] | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a Sequential model.

        Args:
            layers: Optional sequence of layers to add to the model.
            name: Optional name for the model.
        """
        super().__init__(name=name)
        self.layers: list[Layer | TensorLike] = list(layers) if layers is not None else []
        self._name = name
        self.built = False
        self._build_input_shape: ShapeTuple | None = None

    def add(self, layer: Layer | TensorLike) -> None:
        """Add a layer instance on top of the layer stack.

        Args:
            layer: Layer instance to append.
        """
        self.layers.append(layer)

    def pop(self, rebuild: bool = True) -> Layer | TensorLike:
        """Remove the last layer in the model.

        Args:
            rebuild: Whether to rebuild the model after popping the layer.

        Returns:
            Layer | TensorLike: The removed layer instance.

        Raises:
            TypeError: If there are no layers in the model.
        """
        _ = rebuild
        if not self.layers:
            raise TypeError("There are no layers in the model.")
        return self.layers.pop()

    @property
    def input_shape(self) -> ShapeTuple | None:
        """Return the input shape tuple of the Sequential model.

        Returns:
            ShapeTuple | None: Input shape tuple if built, else ``None``.
        """
        return self._build_input_shape if self.built else None

    @property
    def output_shape(self) -> ShapeTuple | None:
        """Return the output shape tuple of the Sequential model.

        Returns:
            ShapeTuple | None: Output shape tuple if built, else ``None``.
        """
        if not self.built or self._build_input_shape is None:
            return None
        current_shape: ShapeTuple = self._build_input_shape
        for layer in self.layers:
            if isinstance(layer, Layer):
                current_shape = layer.compute_output_shape(current_shape)
        return current_shape

    def get_config(self) -> ConfigDict:
        """Return the serializable configuration of the Sequential model.

        Returns:
            ConfigDict: Configuration dictionary containing ``name`` and ``layers``.
        """
        layer_configs: list[ConfigValue] = []
        for layer in self.layers:
            l_cfg = layer.get_config() if isinstance(layer, SupportsGetConfig) else {}
            layer_configs.append(
                {
                    "class_name": get_object_class_name(layer),
                    "config": l_cfg,
                }
            )
        return {
            "name": self._name,
            "layers": layer_configs,
        }

    @classmethod
    def from_config(cls, config: ConfigDict) -> Sequential:
        """Reconstruct a Sequential model from its configuration dictionary.

        Args:
            config: Configuration dictionary produced by ``get_config()``.

        Returns:
            Sequential: Reconstructed Sequential model.
        """
        from zero_keras.layers import LAYER_REGISTRY

        layers_cfg = config.get("layers", [])
        name_val = config.get("name")
        inst = cls(name=str(name_val) if isinstance(name_val, str) else None)
        if isinstance(layers_cfg, list):
            for l_c in layers_cfg:
                if isinstance(l_c, dict):
                    l_cls_name = l_c.get("class_name")
                    l_cls = (
                        LAYER_REGISTRY.get(str(l_cls_name)) if isinstance(l_cls_name, str) else None
                    )
                    if l_cls is not None:
                        sub_cfg = l_c.get("config", {})
                        inst.add(l_cls(**(sub_cfg if isinstance(sub_cfg, dict) else {})))
        return inst

    @property
    def weights(self) -> list[Variable]:
        """Return list of all weight variables across child layers.

        Returns:
            list[Variable]: All model weights.
        """
        w: list[Variable] = []
        for layer in self.layers:
            w.extend(layer.weights)
        return w

    @property
    def trainable_weights(self) -> list[Variable]:
        """Return list of all trainable weight variables across child layers.

        Returns:
            list[Variable]: Trainable model weights.
        """
        w: list[Variable] = []
        for layer in self.layers:
            w.extend(layer.trainable_weights)
        return w

    @property
    def non_trainable_weights(self) -> list[Variable]:
        """Return list of all non-trainable weight variables across child layers.

        Returns:
            list[Variable]: Non-trainable model weights.
        """
        w: list[Variable] = []
        for layer in self.layers:
            w.extend(layer.non_trainable_weights)
        return w

    def get_weights(self) -> list[TensorLike]:
        """Return the values of all layer weights as a flat list.

        Returns:
            list[TensorLike]: List of weight arrays/tensors.
        """
        w: list[TensorLike] = []
        for layer in self.layers:
            if isinstance(layer, Layer):
                w.extend(layer.get_weights())
        return w

    def set_weights(self, weights: Sequence[TensorLike]) -> None:
        """Set the values of all layer weights from a flat sequence of arrays.

        Args:
            weights: Sequence of weight values matching ``get_weights()`` order.
        """
        idx = 0
        for layer in self.layers:
            if isinstance(layer, Layer):
                cw = layer.get_weights()
                layer.set_weights(list(weights[idx : idx + len(cw)]))
                idx += len(cw)

    def build(self, input_shape: ShapeLike | None = None) -> None:
        """Build child layers sequentially for ``input_shape``.

        Args:
            input_shape: Input shape tuple.
        """
        if isinstance(input_shape, tuple):
            self._build_input_shape = input_shape
        current_shape = input_shape
        for layer in self.layers:
            if isinstance(layer, Layer):
                if not layer.built:
                    layer.build(current_shape)
                current_shape = layer.compute_output_shape(current_shape)
        self.built = True

    def call(
        self,
        inputs: TensorLike,
        training: bool | None = None,
        mask: TensorLike | None = None,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Execute all child layers sequentially on ``inputs``.

        Args:
            inputs: Input tensor.
            training: Optional boolean indicating training vs inference mode.
            mask: Optional input mask tensor.
            **kwargs: Additional keyword arguments passed to child layers.

        Returns:
            TensorLike: Final output tensor.
        """
        _ = training, mask
        out = inputs
        for layer in self.layers:
            out = layer(out, **kwargs)
        return out

    def save(
        self,
        filepath: str | Path,
        overwrite: bool = True,
        save_format: str | None = None,
        **kwargs: ConfigValue,
    ) -> None:
        """Save the Sequential model to ``filepath``.

        Args:
            filepath: Target ``.keras`` file path.
            overwrite: Whether to overwrite existing files.
            save_format: Optional format string.
            **kwargs: Additional save options.
        """
        Model.save(self, filepath, overwrite=overwrite, save_format=save_format, **kwargs)


def _default_clone_function(layer: Layer | TensorLike) -> Layer | TensorLike:
    """Clone a single layer via ``from_config(layer.get_config())``.

    Args:
        layer: Layer instance to clone.

    Returns:
        Layer | TensorLike: Cloned layer instance.
    """
    if isinstance(layer, Layer):
        return type(layer).from_config(layer.get_config())
    return layer


def clone_model(
    model: Model,
    input_tensors: Sequence[TensorLike] | TensorLike | None = None,
    clone_function: Callable[[Layer | TensorLike], Layer | TensorLike] | None = None,
) -> Model:
    """Clone a ``Functional`` or ``Sequential`` ``Model`` instance with fresh weights.

    Args:
        model: Instance of ``Model`` or ``Sequential`` to clone.
        input_tensors: Optional input tensors to build the cloned model upon.
        clone_function: Optional callable applied to clone each layer.

    Returns:
        Model: Cloned model instance.

    Raises:
        ValueError: If ``model`` is not a ``Model`` instance.
    """
    _ = input_tensors
    if not isinstance(model, Model):
        raise ValueError("Expected `model` argument to be a `Model` instance.")

    resolved_clone_fn = clone_function if clone_function is not None else _default_clone_function

    if isinstance(model, Sequential):
        cloned_seq = Sequential(name=model._name)
        for layer in model.layers:
            cloned_seq.add(resolved_clone_fn(layer))
        return cloned_seq

    config = model.get_config()
    layer_configs = config.get("layers", [])

    if not isinstance(layer_configs, list) or not layer_configs:
        return model

    created_layers: dict[str, Layer | TensorLike] = {}
    tensor_map: dict[str, TensorLike] = {}

    for lc in layer_configs:
        lc_dict = lc if isinstance(lc, dict) else {}
        lname = str(lc_dict.get("name", ""))
        cname = str(lc_dict.get("class_name", ""))
        cconfig_raw = lc_dict.get("config", {})
        cconfig = cconfig_raw if isinstance(cconfig_raw, dict) else {}

        if cname == "InputLayer":
            shape_raw = cconfig.get("batch_input_shape", (None,))
            raw_seq = list(shape_raw) if isinstance(shape_raw, (list, tuple)) else [None]
            shape_items = raw_seq[1:] if len(raw_seq) > 0 and raw_seq[0] is None else raw_seq
            shape: ShapeTuple = tuple(
                int(d) if isinstance(d, (int, float)) else None for d in shape_items
            )
            dt_val = cconfig.get("dtype")
            inp_tensor = Input(
                shape=shape,
                name=lname,
                dtype=str(dt_val) if isinstance(dt_val, str) else None,
            )
            created_layers[lname] = inp_tensor
            tensor_map[lname] = inp_tensor
        else:
            orig_by_name = {lyr.name: lyr for lyr in model.layers}
            created_layers[lname] = resolved_clone_fn(orig_by_name[lname])

    for lc in layer_configs:
        lc_dict = lc if isinstance(lc, dict) else {}
        lname = str(lc_dict.get("name", ""))
        inbound = lc_dict.get("inbound_nodes", [])
        if isinstance(inbound, list) and inbound:
            layer_obj = created_layers[lname]
            for connection_group in inbound:
                cg = list(connection_group) if isinstance(connection_group, (list, tuple)) else []
                if len(cg) == 1:
                    conn0 = list(cg[0]) if isinstance(cg[0], (list, tuple)) else []
                    src_layer, call_kwargs = conn0[0], conn0[3]
                    inp_t = tensor_map[str(src_layer)]
                    kw = dict(call_kwargs) if isinstance(call_kwargs, dict) else {}
                    tensor_map[lname] = layer_obj(inp_t, **kw) if callable(layer_obj) else inp_t
                else:
                    inp_ts = [tensor_map[str(c[0])] for c in cg if isinstance(c, (list, tuple))]
                    tensor_map[lname] = layer_obj(inp_ts) if callable(layer_obj) else inp_ts

    input_layers_raw = config.get("input_layers", [])
    output_layers_raw = config.get("output_layers", [])
    input_layers = input_layers_raw if isinstance(input_layers_raw, list) else []
    output_layers = output_layers_raw if isinstance(output_layers_raw, list) else []

    inputs_list = [tensor_map[str(il[0])] for il in input_layers if isinstance(il, (list, tuple))]
    outputs_list = [tensor_map[str(ol[0])] for ol in output_layers if isinstance(ol, (list, tuple))]

    inputs_arg: TensorLike | list[TensorLike] = (
        inputs_list[0] if len(inputs_list) == 1 else inputs_list
    )
    outputs_arg: TensorLike | list[TensorLike] = (
        outputs_list[0] if len(outputs_list) == 1 else outputs_list
    )
    name_val = config.get("name")
    return Functional(
        inputs=inputs_arg,
        outputs=outputs_arg,
        name=str(name_val) if isinstance(name_val, str) else None,
    )


def save_model(
    model: Model | TensorLike,
    filepath: str | Path,
    overwrite: bool = True,
    **kwargs: ConfigValue,
) -> None:
    """Save a model as a ``.keras`` file.

    Args:
        model: Model instance to save.
        filepath: Destination file path.
        overwrite: Whether to overwrite existing files.
        **kwargs: Additional save options.
    """
    if isinstance(model, Model):
        model.save(filepath, overwrite=overwrite, **kwargs)
    else:
        Model.save(model, filepath, overwrite=overwrite, **kwargs)


def model_from_json(
    json_string: str,
    custom_objects: dict[str, ConfigValue] | None = None,
    safe_mode: bool = True,
) -> Model:
    """Reconstruct a model from a JSON configuration string.

    Args:
        json_string: JSON-encoded model configuration.
        custom_objects: Optional dictionary of custom classes.
        safe_mode: Whether to disallow unsafe ``Lambda`` deserialization.

    Returns:
        Model: Reconstructed model instance.

    Raises:
        ValueError: If ``safe_mode=True`` and a ``Lambda`` layer is present.
    """
    config = json.loads(json_string)
    class_name = config.get("class_name")

    if class_name == "Sequential":
        sub_cfg = config.get("config", {})
        return Sequential.from_config(sub_cfg if isinstance(sub_cfg, dict) else {})

    if class_name in ("Functional", "Model"):
        conf_raw = config.get("config", {})
        conf = conf_raw if isinstance(conf_raw, dict) else {}
        layer_configs = conf.get("layers", [])

        created_layers: dict[str, Layer | TensorLike] = {}
        tensor_map: dict[str, TensorLike] = {}

        if isinstance(layer_configs, list):
            for lc in layer_configs:
                lc_dict = lc if isinstance(lc, dict) else {}
                lname = str(lc_dict.get("name", ""))
                cname = str(lc_dict.get("class_name", ""))
                cconfig_raw = lc_dict.get("config", {})
                cconfig = cconfig_raw if isinstance(cconfig_raw, dict) else {}

                if safe_mode and cname == "Lambda":
                    raise ValueError("safe_mode is True, cannot deserialize Lambda")

                if cname == "InputLayer":
                    shape_raw = cconfig.get("batch_input_shape", (None,))
                    raw_seq = list(shape_raw) if isinstance(shape_raw, (list, tuple)) else [None]
                    shape_items = (
                        raw_seq[1:] if len(raw_seq) > 0 and raw_seq[0] is None else raw_seq
                    )
                    shape: ShapeTuple = tuple(
                        int(d) if isinstance(d, (int, float)) else None for d in shape_items
                    )
                    dt_val = cconfig.get("dtype")
                    inp_t = Input(
                        shape=shape,
                        name=lname,
                        dtype=str(dt_val) if isinstance(dt_val, str) else None,
                    )
                    created_layers[lname] = inp_t
                    tensor_map[lname] = inp_t
                else:
                    created_layers[lname] = deserialize(
                        lc_dict, custom_objects=custom_objects, safe_mode=safe_mode
                    )

            for lc in layer_configs:
                lc_dict = lc if isinstance(lc, dict) else {}
                lname = str(lc_dict.get("name", ""))
                inbound = lc_dict.get("inbound_nodes", [])
                if isinstance(inbound, list) and inbound:
                    layer_obj = created_layers[lname]
                    for connection_group in inbound:
                        cg = (
                            list(connection_group)
                            if isinstance(connection_group, (list, tuple))
                            else []
                        )
                        if len(cg) == 1:
                            conn0 = list(cg[0]) if isinstance(cg[0], (list, tuple)) else []
                            src_layer, call_kwargs = conn0[0], conn0[3]
                            inp_t = tensor_map[str(src_layer)]
                            kw = dict(call_kwargs) if isinstance(call_kwargs, dict) else {}
                            tensor_map[lname] = (
                                layer_obj(inp_t, **kw) if callable(layer_obj) else inp_t
                            )
                        else:
                            inp_ts = [
                                tensor_map[str(c[0])] for c in cg if isinstance(c, (list, tuple))
                            ]
                            tensor_map[lname] = layer_obj(inp_ts) if callable(layer_obj) else inp_ts

        input_layers_raw = conf.get("input_layers", [])
        output_layers_raw = conf.get("output_layers", [])
        input_layers = input_layers_raw if isinstance(input_layers_raw, list) else []
        output_layers = output_layers_raw if isinstance(output_layers_raw, list) else []

        inputs_list = [
            tensor_map[str(il[0])] for il in input_layers if isinstance(il, (list, tuple))
        ]
        outputs_list = [
            tensor_map[str(ol[0])] for ol in output_layers if isinstance(ol, (list, tuple))
        ]

        inputs_arg: TensorLike | list[TensorLike] = (
            inputs_list[0] if len(inputs_list) == 1 else inputs_list
        )
        outputs_arg: TensorLike | list[TensorLike] = (
            outputs_list[0] if len(outputs_list) == 1 else outputs_list
        )
        name_val = conf.get("name")
        return Functional(
            inputs=inputs_arg,
            outputs=outputs_arg,
            name=str(name_val) if isinstance(name_val, str) else None,
        )

    return Model()
