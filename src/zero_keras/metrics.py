"""Evaluation metrics and confusion-matrix accumulators for zero_keras."""

from __future__ import annotations

from collections.abc import Sequence
from zero_keras.core_layers import Variable
from zero_keras.types import (
    ConfigDict,
    DTypeLike,
    ShapeTuple,
    SupportsCall,
    TensorLike,
)

# Aliases from losses (exported)
from zero_keras import ops
from zero_keras.losses import (
    binary_crossentropy,
    log_cosh,
    sparse_categorical_crossentropy,
)


def _filter_top_k_and_class_id(
    y_true: TensorLike,
    y_pred: TensorLike,
    top_k: int | None,
    class_id: int | None,
) -> tuple[TensorLike, TensorLike]:
    """Filter predictions by top-k scores and optional target class index.

    Args:
        y_true: Ground-truth target tensor.
        y_pred: Predicted score tensor.
        top_k: Optional number of top predictions to retain.
        class_id: Optional class index to slice along the last dimension.

    Returns:
        tuple[TensorLike, TensorLike]: Filtered ``(y_true, y_pred)`` pair.
    """
    from zero_keras import ops

    if top_k is not None:
        top_k_vals, _ = ops.top_k(y_pred, k=top_k)
        min_top_k = ops.min(top_k_vals, axis=-1, keepdims=True)
        y_pred = ops.where(y_pred >= min_top_k, y_pred, _to_tensor(0.0))
    if class_id is not None:
        y_true = y_true[..., class_id]
        y_pred = y_pred[..., class_id]
    return y_true, y_pred


from zero_keras.activations import _to_tensor, _wrap

"""Keras metrics."""


class Metric:
    """Encapsulates metric logic and state.

    Args:
        name: Optional name for the metric instance.
        dtype: The dtype of the metric's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    Example:
    ```python
    m = SomeMetric(...)
    for input in ...:
        m.update_state(input)
    print('Final result: ', m.result())
    ```

    Usage with `compile()` API:

    ```python
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer=keras.optimizers.RMSprop(0.01),
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=[keras.metrics.CategoricalAccuracy()])

    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))

    model.fit(data, labels, epochs=10)
    ```

    To be implemented by subclasses:

    * `__init__()`: All state variables should be created in this method by
      calling `self.add_variable()` like: `self.var = self.add_variable(...)`
    * `update_state()`: Has all updates to the state variables like:
      `self.var.assign(...)`.
    * `result()`: Computes and returns a scalar value or a dict of scalar values
      for the metric from the state variables.

    Example subclass implementation:

    ```python
    class BinaryTruePositives(Metric):

        def __init__(self, name='binary_true_positives', **kwargs):
            super().__init__(name=name, **kwargs)
            self.true_positives = self.add_variable(
                shape=(),
                initializer='zeros',
                name='true_positives'
            )

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true = ops.cast(_to_tensor(y_true), "bool")
            y_pred = ops.cast(_to_tensor(y_pred), "bool")

            values = ops.logical_and(
                ops.equal(y_true, True), ops.equal(y_pred, True))
            values = ops.cast(values, self.dtype)
            if sample_weight is not None:
                sample_weight = ops.cast(sample_weight, self.dtype)
                sample_weight = ops.broadcast_to(
                    sample_weight, ops.shape(values)
                )
                values = ops.multiply(values, sample_weight)
            self.true_positives.assign(self.true_positives + ops.sum(values))

        def result(self):
            return self.true_positives
    ```

    """

    _weights: list[Variable] = []
    weights: list[Variable] | None = None
    _dtype: DTypeLike = None
    name: str | None = None
    total: Variable | None = None
    count: Variable | None = None
    top_k: int | None = None

    def add_variable(
        self,
        shape: ShapeTuple,
        initializer: str | SupportsCall,
        dtype: DTypeLike = None,
        aggregation: str = "sum",
        name: str | None = None,
    ) -> Variable:
        """Create and track a metric state variable.

        Args:
            shape: Shape tuple of the state variable.
            initializer: Initializer name or callable.
            dtype: Data type of the variable.
            aggregation: Multi-replica aggregation strategy.
            name: Optional variable name.

        Returns:
            Variable: Newly created metric state variable.
        """
        _ = aggregation
        var = Variable(initializer, shape=shape, dtype=dtype, name=name)
        self._weights.append(var)
        return var

    def add_weight(
        self,
        shape: ShapeTuple = (),
        initializer: str | SupportsCall = "zeros",
        dtype: DTypeLike = None,
        name: str | None = None,
    ) -> Variable:
        """Create and track a metric state weight variable.

        Args:
            shape: Shape tuple of the state variable.
            initializer: Initializer name or callable.
            dtype: Data type of the variable.
            name: Optional variable name.

        Returns:
            Variable: Newly created metric state variable.
        """
        var = Variable(initializer, shape=shape, dtype=dtype, name=name)
        self._weights.append(var)
        return var

    @property
    def dtype(self) -> DTypeLike:
        """Return the data type of the metric's computations.

        Returns:
            DTypeLike: Metric computation dtype.
        """
        return self._dtype

    @dtype.setter
    def dtype(self, value: DTypeLike) -> None:
        """Set the data type of the metric's computations.

        Args:
            value: Data type string or policy.
        """
        self._dtype = value

    @classmethod
    def from_config(cls, config: ConfigDict) -> Metric:
        """Instantiate a metric from its configuration dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            Metric: Reconstructed metric instance.
        """
        return cls(**config)

    def get_config(self) -> ConfigDict:
        """Return the serializable configuration of the metric.

        Returns:
            ConfigDict: Configuration dictionary containing ``name``.
        """
        return {"name": self.name}

    def stateless_reset_state(self) -> None:
        """Reset metric state variables in stateless execution mode."""
        self.reset_state()

    def stateless_result(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Compute the metric result in stateless execution mode.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorLike: Metric result tensor.
        """
        return self.result()

    def stateless_update_state(self, *args: TensorLike, **kwargs: TensorLike) -> None:
        """Update metric state variables in stateless execution mode.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        self.update_state(*args, **kwargs)

    @property
    def variables(self) -> list[Variable]:
        """Return the list of state variables tracked by this metric.

        Returns:
            list[Variable]: Tracked metric state variables.
        """
        return list(self.weights) if self.weights is not None else list(self._weights)

    def __init__(
        self, name: str | None = None, dtype: DTypeLike = None, **kwargs: TensorLike
    ) -> None:
        """Initialize the base Metric instance.

        Args:
            name: Optional string name for the metric.
            dtype: Optional computation data type.
            **kwargs: Additional keyword arguments.
        """
        self._weights: list[Variable] = []
        self.name = name
        self.dtype = dtype
        self._kwargs = kwargs

    def update_state(self, *args: TensorLike, **kwargs: TensorLike) -> None:
        """Accumulate statistics for the metric.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        _ = args, kwargs
        return None

    def result(self) -> TensorLike:
        """Compute the current metric value.

        Returns:
            TensorLike: A scalar tensor, or a dictionary of scalar tensors.
        """
        return _wrap(_to_tensor(0.0))

    def reset_state(self) -> None:
        """Reset all of the metric state variables.

        This function is called between epochs/steps,
        when a metric is evaluated during training.
        """
        if isinstance(self.total, Variable):
            self.total.assign(_to_tensor(0.0))
        if isinstance(self.count, Variable):
            self.count.assign(_to_tensor(0.0))

    def __call__(self, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Update metric state and return the current metric result.

        Args:
            *args: Positional tensor inputs passed to ``update_state``.
            **kwargs: Keyword arguments passed to ``update_state``.

        Returns:
            TensorLike: Computed metric value after updating state.
        """
        self.update_state(*args, **kwargs)
        return self.result()


class Mean(Metric):
    """Compute the (weighted) mean of the given values.

    For example, if values is `[1, 3, 5, 7]` then the mean is 4.
    If `sample_weight` was specified as `[1, 1, 0, 0]` then the mean would be 2.

    This metric creates two variables, `total` and `count`.
    The mean value returned is simply `total` divided by `count`.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:
    >>> m = Mean()
    >>> m.update_state([1, 3, 5, 7])
    >>> m.result()
    4.0

    >>> m.reset_state()
    >>> m.update_state([1, 3, 5, 7], sample_weight=[1, 1, 0, 0])
    >>> m.result()
    2.0

    """

    def __init__(self, name: str = "mean", dtype: DTypeLike = None, **kwargs: TensorLike) -> None:
        """Initialize the Mean metric.

        Args:
            name: String name of the metric instance.
            dtype: Data type of the metric result.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = self.add_weight(shape=(), initializer="zeros", name="total")
        self.count = self.add_weight(shape=(), initializer="zeros", name="count")

    def update_state(self, values: TensorLike, sample_weight: TensorLike | None = None) -> None:
        """Accumulate weighted mean statistics.

        Args:
            values: Per-example input values.
            sample_weight: Optional weighting for each value.
        """
        values = _to_tensor(values)
        if sample_weight is not None:
            sample_weight = _to_tensor(sample_weight)
            values = ops.multiply(values, sample_weight)
            c_inc = ops.cast(ops.sum(sample_weight), dtype="float32")
        else:
            c_inc = ops.cast(ops.sum(ops.ones_like(values)), dtype="float32")
        t_inc = ops.cast(ops.sum(values), dtype="float32")
        self.count.assign(ops.add(_to_tensor(self.count), c_inc))
        self.total.assign(ops.add(_to_tensor(self.total), t_inc))

    def result(self) -> TensorLike:
        """Compute the current weighted mean value.

        Returns:
            TensorLike: Scalar mean tensor.
        """
        return _wrap(_to_tensor(self.total) / ops.maximum(_to_tensor(self.count), 1e-7))

    def reset_state(self) -> None:
        """Reset ``total`` and ``count`` state variables to zero."""
        self.total.assign(_to_tensor(0.0))
        self.count.assign(_to_tensor(0.0))


class Sum(Metric):
    """Compute the (weighted) sum of the given values.

    For example, if `values` is `[1, 3, 5, 7]` then their sum is 16.
    If `sample_weight` was specified as `[1, 1, 0, 0]` then the sum would be 4.

    This metric creates one variable, `total`.
    This is ultimately returned as the sum value.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:
    >>> m = metrics.Sum()
    >>> m.update_state([1, 3, 5, 7])
    >>> m.result()
    16.0

    >>> m = metrics.Sum()
    >>> m.update_state([1, 3, 5, 7], sample_weight=[1, 1, 0, 0])
    >>> m.result()
    4.0

    """

    def __init__(self, name: str = "sum", dtype: DTypeLike = None, **kwargs: TensorLike) -> None:
        """Initialize the Sum metric.

        Args:
            name: String name of the metric instance.
            dtype: Data type of the metric result.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = self.add_weight(shape=(), initializer="zeros", name="total")

    def update_state(self, values: TensorLike, sample_weight: TensorLike | None = None) -> None:
        """Accumulate weighted sum statistics.

        Args:
            values: Per-example input values.
            sample_weight: Optional weighting for each value.
        """
        values = _to_tensor(values)
        if sample_weight is not None:
            sample_weight = _to_tensor(sample_weight)
            values = ops.multiply(values, sample_weight)
        self.total.assign(
            ops.add(_to_tensor(self.total), ops.cast(ops.sum(values), dtype="float32"))
        )

    def result(self) -> TensorLike:
        """Compute the current sum value.

        Returns:
            TensorLike: Scalar sum tensor.
        """
        return _wrap(_to_tensor(self.total))

    def reset_state(self) -> None:
        """Reset ``total`` state variable to zero."""
        self.total.assign(_to_tensor(0.0))


class MeanMetricWrapper(Mean):
    """Wrap a stateless metric function with the `Mean` metric.

    You could use this class to quickly build a mean metric from a function. The
    function needs to have the signature `fn(y_true, y_pred)` and return a
    per-sample loss array. `MeanMetricWrapper.result()` will return
    the average metric value across all samples seen so far.

    For example:

    ```python
    def mse(y_true, y_pred):
        return (y_true - y_pred) ** 2

    mse_metric = MeanMetricWrapper(fn=mse)
    ```

    Args:
        fn: The metric function to wrap, with signature
            `fn(y_true, y_pred, **kwargs)`.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        **kwargs: Keyword arguments to pass on to `fn`.

    """

    def __init__(
        self,
        fn: SupportsCall,
        name: str | None = None,
        dtype: DTypeLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the MeanMetricWrapper.

        Args:
            fn: Stateless metric function accepting ``(y_true, y_pred)``.
            name: Optional string name of the metric instance.
            dtype: Optional data type of the metric result.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.fn = fn

    def update_state(
        self, y_true: TensorLike, y_pred: TensorLike, sample_weight: TensorLike | None = None
    ) -> None:
        """Accumulate statistics for the wrapped metric function.

        Args:
            y_true: Ground-truth target values.
            y_pred: Predicted values.
            sample_weight: Optional weighting for each example.
        """
        matches = self.fn(y_true, y_pred)
        super().update_state(matches, sample_weight=sample_weight)


class Accuracy(MeanMetricWrapper):
    """Calculates how often predictions equal labels.

    This metric creates two local variables, `total` and `count` that are used
    to compute the frequency with which `y_pred` matches `y_true`. This
    frequency is ultimately returned as `binary accuracy`: an idempotent
    operation that simply divides `total` by `count`.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Examples:
    >>> m = keras.metrics.Accuracy()
    >>> m.update_state([[1], [2], [3], [4]], [[0], [2], [3], [4]])
    >>> m.result()
    0.75

    >>> m.reset_state()
    >>> m.update_state([[1], [2], [3], [4]], [[0], [2], [3], [4]],
    ...                sample_weight=[1, 1, 0, 0])
    >>> m.result()
    0.5

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='binary_crossentropy',
                  metrics=[keras.metrics.Accuracy()])
    ```

    """

    def __init__(
        self, name: str = "accuracy", dtype: DTypeLike = None, **kwargs: TensorLike
    ) -> None:
        """Initialize the Accuracy metric.

        Args:
            name: String name of the metric instance.
            dtype: Data type of the metric result.
            **kwargs: Additional keyword arguments.
        """

        def accuracy_fn(y_true: TensorLike, y_pred: TensorLike) -> TensorLike:
            """Compute elementwise equality indicator between ``y_true`` and ``y_pred``.

            Args:
                y_true: Ground-truth labels.
                y_pred: Predicted labels.

            Returns:
                TensorLike: Float32 indicator tensor where ``1.0`` denotes exact match.
            """
            return ops.cast(ops.equal(_to_tensor(y_true), _to_tensor(y_pred)), dtype="float32")

        super().__init__(fn=accuracy_fn, name=name, dtype=dtype, **kwargs)


class BinaryAccuracy(MeanMetricWrapper):
    """Calculates how often predictions match binary labels.

    This metric creates two local variables, `total` and `count` that are used
    to compute the frequency with which `y_pred` matches `y_true`. This
    frequency is ultimately returned as `binary accuracy`: an idempotent
    operation that simply divides `total` by `count`.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        threshold: (Optional) Float representing the threshold for deciding
        whether prediction values are 1 or 0.

    Example:
    >>> m = keras.metrics.BinaryAccuracy()
    >>> m.update_state([[1], [1], [0], [0]], [[0.98], [1], [0], [0.6]])
    >>> m.result()
    0.75

    >>> m.reset_state()
    >>> m.update_state([[1], [1], [0], [0]], [[0.98], [1], [0], [0.6]],
    ...                sample_weight=[1, 0, 0, 1])
    >>> m.result()
    0.5

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='binary_crossentropy',
                  metrics=[keras.metrics.BinaryAccuracy()])
    ```

    """

    def __init__(
        self,
        name: str = "binary_accuracy",
        dtype: DTypeLike = None,
        threshold: float = 0.5,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the BinaryAccuracy metric.

        Args:
            name: String name of the metric instance.
            dtype: Data type of the metric result.
            threshold: Decision threshold for binarizing ``y_pred``.
            **kwargs: Additional keyword arguments.
        """

        def binary_accuracy_fn(y_true: TensorLike, y_pred: TensorLike) -> TensorLike:
            """Compute binary classification match indicator at ``threshold``.

            Args:
                y_true: Binary ground-truth targets.
                y_pred: Predicted probabilities or scores.

            Returns:
                TensorLike: Float32 indicator tensor where ``1.0`` denotes a match.
            """
            y_pred = ops.cast(_to_tensor(y_pred) > threshold, dtype="float32")
            y_true = ops.cast(_to_tensor(y_true), dtype="float32")
            return ops.cast(ops.equal(y_true, y_pred), dtype="float32")

        super().__init__(fn=binary_accuracy_fn, name=name, dtype=dtype, **kwargs)


class CategoricalAccuracy(MeanMetricWrapper):
    """Calculates how often predictions match one-hot labels.

    You can provide logits of classes as `y_pred`, since argmax of
    logits and probabilities are same.

    This metric creates two local variables, `total` and `count` that are used
    to compute the frequency with which `y_pred` matches `y_true`. This
    frequency is ultimately returned as `categorical accuracy`: an idempotent
    operation that simply divides `total` by `count`.

    `y_pred` and `y_true` should be passed in as vectors of probabilities,
    rather than as labels. If necessary, use `ops.one_hot` to expand `y_true` as
    a vector.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:
    >>> m = keras.metrics.CategoricalAccuracy()
    >>> m.update_state([[0, 0, 1], [0, 1, 0]], [[0.1, 0.9, 0.8],
    ...                 [0.05, 0.95, 0]])
    >>> m.result()
    0.5

    >>> m.reset_state()
    >>> m.update_state([[0, 0, 1], [0, 1, 0]], [[0.1, 0.9, 0.8],
    ...                 [0.05, 0.95, 0]],
    ...                sample_weight=[0.7, 0.3])
    >>> m.result()
    0.3

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=[keras.metrics.CategoricalAccuracy()])
    ```

    """

    def __init__(
        self,
        name: str = "categorical_accuracy",
        dtype: DTypeLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the CategoricalAccuracy metric.

        Args:
            name: String name of the metric instance.
            dtype: Data type of the metric result.
            **kwargs: Additional keyword arguments.
        """

        def categorical_accuracy_fn(y_true: TensorLike, y_pred: TensorLike) -> TensorLike:
            """Compute argmax match indicator between one-hot ``y_true`` and ``y_pred``.

            Args:
                y_true: One-hot ground-truth targets.
                y_pred: Predicted class probabilities or logits.

            Returns:
                TensorLike: Float32 indicator tensor where ``1.0`` denotes a match.
            """
            y_true = _to_tensor(y_true)
            y_pred = _to_tensor(y_pred)
            return ops.cast(
                ops.equal(ops.argmax(y_true, axis=-1), ops.argmax(y_pred, axis=-1)),
                dtype="float32",
            )

        super().__init__(fn=categorical_accuracy_fn, name=name, dtype=dtype, **kwargs)


class SparseCategoricalAccuracy(MeanMetricWrapper):
    """Calculates how often predictions match integer labels.

    ```python
    acc = np.dot(sample_weight, np.equal(y_true, np.argmax(y_pred, axis=1))
    ```

    You can provide logits of classes as `y_pred`, since argmax of
    logits and probabilities are same.

    This metric creates two local variables, `total` and `count` that are used
    to compute the frequency with which `y_pred` matches `y_true`. This
    frequency is ultimately returned as `sparse categorical accuracy`: an
    idempotent operation that simply divides `total` by `count`.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:
    >>> m = keras.metrics.SparseCategoricalAccuracy()
    >>> m.update_state([[2], [1]], [[0.1, 0.6, 0.3], [0.05, 0.95, 0]])
    >>> m.result()
    0.5

    >>> m.reset_state()
    >>> m.update_state([[2], [1]], [[0.1, 0.6, 0.3], [0.05, 0.95, 0]],
    ...                sample_weight=[0.7, 0.3])
    >>> m.result()
    0.3

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='sparse_categorical_crossentropy',
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])
    ```

    """

    def __init__(
        self,
        name: str = "sparse_categorical_accuracy",
        dtype: DTypeLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the SparseCategoricalAccuracy metric.

        Args:
            name: String name of the metric instance.
            dtype: Data type of the metric result.
            **kwargs: Additional keyword arguments.
        """

        def sparse_categorical_accuracy_fn(y_true: TensorLike, y_pred: TensorLike) -> TensorLike:
            """Compute argmax match indicator between integer ``y_true`` and ``y_pred``.

            Args:
                y_true: Integer class indices.
                y_pred: Predicted class probabilities or logits.

            Returns:
                TensorLike: Float32 indicator tensor where ``1.0`` denotes a match.
            """
            y_true = _to_tensor(y_true)
            y_pred = _to_tensor(y_pred)
            if len(y_true.shape) == len(y_pred.shape):
                if y_true.shape and y_true.shape[-1] == 1:
                    y_true = ops.squeeze(y_true, -1)
            y_pred_classes = ops.argmax(y_pred, axis=-1)
            return ops.cast(
                ops.equal(ops.cast(y_true, dtype=y_pred_classes.dtype), y_pred_classes),
                dtype="float32",
            )

        super().__init__(fn=sparse_categorical_accuracy_fn, name=name, dtype=dtype, **kwargs)


class TopKCategoricalAccuracy(MeanMetricWrapper):
    """Computes how often targets are in the top `K` predictions.

    Args:
        k: (Optional) Number of top elements to look at for computing accuracy.
            Defaults to `5`.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:
    >>> m = keras.metrics.TopKCategoricalAccuracy(k=1)
    >>> m.update_state([[0, 0, 1], [0, 1, 0]],
    ...                [[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
    >>> m.result()
    0.5

    >>> m.reset_state()
    >>> m.update_state([[0, 0, 1], [0, 1, 0]],
    ...                [[0.1, 0.9, 0.8], [0.05, 0.95, 0]],
    ...                sample_weight=[0.7, 0.3])
    >>> m.result()
    0.3

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=[keras.metrics.TopKCategoricalAccuracy()])
    ```

    """

    def __init__(
        self,
        k: int = 5,
        name: str = "top_k_categorical_accuracy",
        dtype: DTypeLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the TopKCategoricalAccuracy metric.

        Args:
            k: Number of top predictions considered.
            name: String name of the metric instance.
            dtype: Data type of the metric result.
            **kwargs: Additional keyword arguments.
        """

        def top_k_fn(y_true: TensorLike, y_pred: TensorLike) -> TensorLike:
            """Check whether the argmax class of ``y_true`` is in top-k of ``y_pred``.

            Args:
                y_true: One-hot ground-truth targets.
                y_pred: Predicted class probabilities or logits.

            Returns:
                TensorLike: Float32 indicator tensor where ``1.0`` denotes top-k hit.
            """
            y_true = _to_tensor(y_true)
            y_pred = _to_tensor(y_pred)
            y_true_rank = ops.argmax(y_true, axis=-1)
            _, top_indices = ops.top_k(y_pred, k)
            y_true_rank = ops.expand_dims(y_true_rank, -1)
            matches = ops.any(ops.equal(top_indices, y_true_rank), axis=-1)
            return ops.cast(matches, "float32")

        super().__init__(fn=top_k_fn, name=name, dtype=dtype, **kwargs)


class SparseTopKCategoricalAccuracy(MeanMetricWrapper):
    """Computes how often integer targets are in the top `K` predictions.

    By default, the arguments expected by `update_state()` are:
    - `y_true`: a tensor of shape `(batch_size)` representing indices of true
        categories.
    - `y_pred`: a tensor of shape `(batch_size, num_categories)` containing the
        scores for each sample for all possible categories.

    With `from_sorted_ids=True`, the arguments expected by `update_state` are:
    - `y_true`: a tensor of shape `(batch_size)` representing indices or IDs of
        true categories.
    - `y_pred`: a tensor of shape `(batch_size, N)` containing the indices or
        IDs of the top `N` categories sorted in order from highest score to
        lowest score. `N` must be greater or equal to `k`.

    The `from_sorted_ids=True` option can be more efficient when the set of
    categories is very large and the model has an optimized way to retrieve the
    top ones either without scoring or without maintaining the scores for all
    the possible categories.

    Args:
        k: (Optional) Number of top elements to look at for computing accuracy.
            Defaults to `5`.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        from_sorted_ids: (Optional) When `False`, the default, the tensor passed
            in `y_pred` contains the unsorted scores of all possible categories.
            When `True`, `y_pred` contains a the indices or IDs for the top
            categories.

    Example:
    >>> m = keras.metrics.SparseTopKCategoricalAccuracy(k=1)
    >>> m.update_state([2, 1], [[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
    >>> m.result()
    0.5

    >>> m.reset_state()
    >>> m.update_state([2, 1], [[0.1, 0.9, 0.8], [0.05, 0.95, 0]],
    ...                sample_weight=[0.7, 0.3])
    >>> m.result()
    0.3

    >>> m = keras.metrics.SparseTopKCategoricalAccuracy(k=1,
    ...                                                from_sorted_ids=True)
    >>> m.update_state([2, 1], [[1, 0, 3], [1, 2, 3]])
    >>> m.result()
    0.5

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='sparse_categorical_crossentropy',
                  metrics=[keras.metrics.SparseTopKCategoricalAccuracy()])
    ```

    """

    def __init__(
        self,
        k: int = 5,
        name: str = "sparse_top_k_categorical_accuracy",
        dtype: DTypeLike = None,
        from_sorted_ids: bool = False,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the SparseTopKCategoricalAccuracy metric.

        Args:
            k: Number of top predictions considered.
            name: String name of the metric instance.
            dtype: Data type of the metric result.
            from_sorted_ids: Whether ``y_pred`` is already sorted class indices.
            **kwargs: Additional keyword arguments.
        """
        self.from_sorted_ids = from_sorted_ids

        def sparse_top_k_fn(y_true: TensorLike, y_pred: TensorLike) -> TensorLike:
            """Check whether integer ``y_true`` is in top-k of ``y_pred``.

            Args:
                y_true: Integer class indices.
                y_pred: Predicted class probabilities or logits.

            Returns:
                TensorLike: Float32 indicator tensor where ``1.0`` denotes top-k hit.
            """
            y_true = _to_tensor(y_true)
            y_pred = _to_tensor(y_pred)

            y_true = ops.cast(y_true, "int32")

            _, top_indices = ops.top_k(y_pred, k)
            if len(y_true.shape) == 1:
                y_true = ops.expand_dims(y_true, -1)
            elif len(y_true.shape) == 2 and y_true.shape[-1] == 1:
                pass
            else:
                y_true = ops.expand_dims(y_true, -1)

            matches = ops.any(ops.equal(top_indices, y_true), axis=-1)
            return ops.cast(matches, "float32")

        super().__init__(fn=sparse_top_k_fn, name=name, dtype=dtype, **kwargs)


class _ConfusionMatrixMetric(Metric):
    """Base class for confusion-matrix element count metrics (TP, FP, TN, FN)."""

    def __init__(
        self,
        thresholds: float | Sequence[float] | None = None,
        metric_type: str = "FP",
        name: str | None = None,
        dtype: DTypeLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the confusion-matrix metric accumulator.

        Args:
            thresholds: Decision threshold or sequence of thresholds in ``[0, 1]``.
            metric_type: Confusion matrix entry type (``"FP"``, ``"FN"``, ``"TP"``, or ``"TN"``).
            name: Optional metric name.
            dtype: Optional metric data type.
            **kwargs: Additional keyword arguments.
        """
        if metric_type not in ("FP", "FN", "TP", "TN"):
            raise ValueError(f"Unknown metric_type {metric_type}")
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.thresholds = 0.5 if thresholds is None else thresholds
        self.metric_type = metric_type
        if isinstance(self.thresholds, (list, tuple)):
            self.accumulator: Variable | list[Variable] = [
                self.add_weight(shape=(), initializer="zeros", name=f"accumulator_{i}")
                for i in range(len(self.thresholds))
            ]
        else:
            self.accumulator = self.add_weight(shape=(), initializer="zeros", name="accumulator")
        self.reset_state()

    def update_state(
        self, y_true: TensorLike, y_pred: TensorLike, sample_weight: TensorLike | None = None
    ) -> None:
        """Accumulate confusion-matrix counts from ``y_true`` and ``y_pred``.

        Args:
            y_true: Ground-truth binary targets.
            y_pred: Predicted probabilities or scores.
            sample_weight: Optional sample weights.
        """
        y_true = _to_tensor(y_true)
        y_pred = _to_tensor(y_pred)
        y_true = ops.cast(_to_tensor(y_true), "bool")

        if isinstance(self.accumulator, list) and isinstance(self.thresholds, (list, tuple)):
            for i, t in enumerate(self.thresholds):
                y_pred_i = ops.cast(ops.greater_equal(y_pred, t), "bool")
                val = None
                if self.metric_type == "FP":
                    val = ops.logical_and(ops.logical_not(y_true), y_pred_i)
                elif self.metric_type == "FN":
                    val = ops.logical_and(y_true, ops.logical_not(y_pred_i))
                elif self.metric_type == "TP":
                    val = ops.logical_and(y_true, y_pred_i)
                elif self.metric_type == "TN":
                    val = ops.logical_and(ops.logical_not(y_true), ops.logical_not(y_pred_i))
                else:
                    raise ValueError(f"Unknown metric_type {self.metric_type}")

                val = ops.cast(val, "float32")
                if sample_weight is not None:
                    val = val * ops.cast(sample_weight, "float32")
                self.accumulator[i].assign(_to_tensor(self.accumulator[i]) + ops.sum(val))
        else:
            y_pred_bool = ops.cast(ops.greater_equal(y_pred, self.thresholds), "bool")
            val = None
            if self.metric_type == "FP":
                val = ops.logical_and(ops.logical_not(y_true), y_pred_bool)
            elif self.metric_type == "FN":
                val = ops.logical_and(y_true, ops.logical_not(y_pred_bool))
            elif self.metric_type == "TP":
                val = ops.logical_and(y_true, y_pred_bool)
            elif self.metric_type == "TN":
                val = ops.logical_and(ops.logical_not(y_true), ops.logical_not(y_pred_bool))
            else:
                raise ValueError(f"Unknown metric_type {self.metric_type}")

            val = ops.cast(val, "float32")
            if sample_weight is not None:
                val = val * ops.cast(sample_weight, "float32")

            acc_var = (
                self.accumulator[0] if isinstance(self.accumulator, list) else self.accumulator
            )
            acc_var.assign(_to_tensor(acc_var) + ops.sum(val))

    def result(self) -> TensorLike:
        """Compute the accumulated confusion-matrix count(s).

        Returns:
            TensorLike: Scalar count tensor or 1D tensor across thresholds.
        """
        if isinstance(self.accumulator, list):
            from zero_keras import ops

            return ops.stack([_to_tensor(acc) for acc in self.accumulator])
        return _to_tensor(self.accumulator)

    def reset_state(self) -> None:
        """Reset all confusion-matrix accumulator variables to zero."""
        if isinstance(self.accumulator, list):
            for acc in self.accumulator:
                acc.assign(_to_tensor(0.0))
        else:
            self.accumulator.assign(_to_tensor(0.0))


class FalsePositives(_ConfusionMatrixMetric):
    """Calculates the number of false positives.

    If `sample_weight` is given, calculates the sum of the weights of
    false positives. This metric creates one local variable, `accumulator`
    that is used to keep track of the number of false positives.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
        thresholds: (Optional) Defaults to `0.5`. A float value, or a Python
            list/tuple of float threshold values in `[0, 1]`. A threshold is
            compared with prediction values to determine the truth value of
            predictions (i.e., above the threshold is `True`, below is `False`).
            If used with a loss function that sets `from_logits=True` (i.e. no
            sigmoid applied to predictions), `thresholds` should be set to 0.
            One metric value is generated for each threshold value.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Examples:
    >>> m = keras.metrics.FalsePositives()
    >>> m.update_state([0, 1, 0, 0], [0, 0, 1, 1])
    >>> m.result()
    2.0

    >>> m.reset_state()
    >>> m.update_state([0, 1, 0, 0], [0, 0, 1, 1], sample_weight=[0, 0, 1, 0])
    >>> m.result()
    1.0

    """

    def __init__(
        self,
        thresholds: float | Sequence[float] | None = None,
        name: str | None = None,
        dtype: DTypeLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the FalsePositives metric.

        Args:
            thresholds: Decision threshold or sequence of thresholds in ``[0, 1]``.
            name: Optional string name of the metric instance.
            dtype: Optional data type of the metric result.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(thresholds=thresholds, metric_type="FP", name=name, dtype=dtype, **kwargs)


class FalseNegatives(_ConfusionMatrixMetric):
    """Calculates the number of false negatives.

    If `sample_weight` is given, calculates the sum of the weights of
    false negatives. This metric creates one local variable, `accumulator`
    that is used to keep track of the number of false negatives.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
        thresholds: (Optional) Defaults to `0.5`. A float value, or a Python
            list/tuple of float threshold values in `[0, 1]`. A threshold is
            compared with prediction values to determine the truth value of
            predictions (i.e., above the threshold is `True`, below is `False`).
            If used with a loss function that sets `from_logits=True` (i.e. no
            sigmoid applied to predictions), `thresholds` should be set to 0.
            One metric value is generated for each threshold value.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:
    >>> m = keras.metrics.FalseNegatives()
    >>> m.update_state([0, 1, 1, 1], [0, 1, 0, 0])
    >>> m.result()
    2.0

    >>> m.reset_state()
    >>> m.update_state([0, 1, 1, 1], [0, 1, 0, 0], sample_weight=[0, 0, 1, 0])
    >>> m.result()
    1.0

    """

    def __init__(
        self,
        thresholds: float | Sequence[float] | None = None,
        name: str | None = None,
        dtype: DTypeLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the FalseNegatives metric.

        Args:
            thresholds: Decision threshold or sequence of thresholds in ``[0, 1]``.
            name: Optional string name of the metric instance.
            dtype: Optional data type of the metric result.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(thresholds=thresholds, metric_type="FN", name=name, dtype=dtype, **kwargs)


class TrueNegatives(_ConfusionMatrixMetric):
    """Calculates the number of true negatives.

    If `sample_weight` is given, calculates the sum of the weights of
    true negatives. This metric creates one local variable, `accumulator`
    that is used to keep track of the number of true negatives.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
        thresholds: (Optional) Defaults to `0.5`. A float value, or a Python
            list/tuple of float threshold values in `[0, 1]`. A threshold is
            compared with prediction values to determine the truth value of
            predictions (i.e., above the threshold is `True`, below is `False`).
            If used with a loss function that sets `from_logits=True` (i.e. no
            sigmoid applied to predictions), `thresholds` should be set to 0.
            One metric value is generated for each threshold value.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:
    >>> m = keras.metrics.TrueNegatives()
    >>> m.update_state([0, 1, 0, 0], [1, 1, 0, 0])
    >>> m.result()
    2.0

    >>> m.reset_state()
    >>> m.update_state([0, 1, 0, 0], [1, 1, 0, 0], sample_weight=[0, 0, 1, 0])
    >>> m.result()
    1.0

    """

    def __init__(
        self,
        thresholds: float | Sequence[float] | None = None,
        name: str | None = None,
        dtype: DTypeLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the TrueNegatives metric.

        Args:
            thresholds: Decision threshold or sequence of thresholds in ``[0, 1]``.
            name: Optional string name of the metric instance.
            dtype: Optional data type of the metric result.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(thresholds=thresholds, metric_type="TN", name=name, dtype=dtype, **kwargs)


class TruePositives(_ConfusionMatrixMetric):
    """Calculates the number of true positives.

    If `sample_weight` is given, calculates the sum of the weights of
    true positives. This metric creates one local variable, `true_positives`
    that is used to keep track of the number of true positives.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
        thresholds: (Optional) Defaults to `0.5`. A float value, or a Python
            list/tuple of float threshold values in `[0, 1]`. A threshold is
            compared with prediction values to determine the truth value of
            predictions (i.e., above the threshold is `True`, below is `False`).
            If used with a loss function that sets `from_logits=True` (i.e. no
            sigmoid applied to predictions), `thresholds` should be set to 0.
            One metric value is generated for each threshold value.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:
    >>> m = keras.metrics.TruePositives()
    >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
    >>> m.result()
    2.0

    >>> m.reset_state()
    >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1], sample_weight=[0, 0, 1, 0])
    >>> m.result()
    1.0

    """

    def __init__(
        self,
        thresholds: float | Sequence[float] | None = None,
        name: str | None = None,
        dtype: DTypeLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the TruePositives metric.

        Args:
            thresholds: Decision threshold or sequence of thresholds in ``[0, 1]``.
            name: Optional string name of the metric instance.
            dtype: Optional data type of the metric result.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(thresholds=thresholds, metric_type="TP", name=name, dtype=dtype, **kwargs)


class Precision(Metric):
    """Computes the precision of the predictions with respect to the labels.

    The metric creates two local variables, `true_positives` and
    `false_positives` that are used to compute the precision. This value is
    ultimately returned as `precision`, an idempotent operation that simply
    divides `true_positives` by the sum of `true_positives` and
    `false_positives`.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    If `top_k` is set, we'll calculate precision as how often on average a class
    among the top-k classes with the highest predicted values of a batch entry
    is correct and can be found in the label for that entry.

    If `class_id` is specified, we calculate precision by considering only the
    entries in the batch for which `class_id` is above the threshold and/or in
    the top-k highest predictions, and computing the fraction of them for which
    `class_id` is indeed a correct label.

    Args:
        thresholds: (Optional) A float value, or a Python list/tuple of float
            threshold values in `[0, 1]`. A threshold is compared with
            prediction values to determine the truth value of predictions (i.e.,
            above the threshold is `True`, below is `False`). If used with a
            loss function that sets `from_logits=True` (i.e. no sigmoid applied
            to predictions), `thresholds` should be set to 0. One metric value
            is generated for each threshold value. If neither `thresholds` nor
            `top_k` are set, the default is to calculate precision with
            `thresholds=0.5`.
        top_k: (Optional) Unset by default. An int value specifying the top-k
            predictions to consider when calculating precision.
        class_id: (Optional) Integer class ID for which we want binary metrics.
            This must be in the half-open interval `[0, num_classes)`, where
            `num_classes` is the last dimension of predictions.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:
    >>> m = keras.metrics.Precision()
    >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
    >>> m.result()
    0.6666667

    >>> m.reset_state()
    >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1], sample_weight=[0, 0, 1, 0])
    >>> m.result()
    1.0

    >>> # With top_k=2, it will calculate precision over y_true[:2]
    >>> # and y_pred[:2]
    >>> m = keras.metrics.Precision(top_k=2)
    >>> m.update_state([0, 0, 1, 1], [1, 1, 1, 1])
    >>> m.result()
    0.0

    >>> # With top_k=4, it will calculate precision over y_true[:4]
    >>> # and y_pred[:4]
    >>> m = keras.metrics.Precision(top_k=4)
    >>> m.update_state([0, 0, 1, 1], [1, 1, 1, 1])
    >>> m.result()
    0.5

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='binary_crossentropy',
                  metrics=[keras.metrics.Precision()])
    ```

    Usage with a loss with `from_logits=True`:

    ```python
    model.compile(optimizer='adam',
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras.metrics.Precision(thresholds=0)])
    ```

    """

    def __init__(
        self,
        thresholds: float | Sequence[float] | None = None,
        top_k: int | None = None,
        class_id: int | None = None,
        name: str | None = None,
        dtype: DTypeLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the Precision metric.

        Args:
            thresholds: Decision threshold or sequence of thresholds in ``[0, 1]``.
            top_k: Optional number of top predictions to retain.
            class_id: Optional target class index.
            name: Optional string name of the metric instance.
            dtype: Optional data type of the metric result.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.thresholds = 0.5 if thresholds is None and top_k is None else thresholds
        self.top_k = top_k
        self.class_id = class_id

        self.true_positives = TruePositives(thresholds=self.thresholds)
        self.false_positives = FalsePositives(thresholds=self.thresholds)

    def update_state(
        self, y_true: TensorLike, y_pred: TensorLike, sample_weight: TensorLike | None = None
    ) -> None:
        """Accumulate true positive and false positive counts.

        Args:
            y_true: Ground-truth binary targets.
            y_pred: Predicted probabilities or scores.
            sample_weight: Optional sample weights.
        """
        y_true = _to_tensor(y_true)
        y_pred = _to_tensor(y_pred)
        y_true, y_pred = _filter_top_k_and_class_id(y_true, y_pred, self.top_k, self.class_id)
        self.true_positives.update_state(y_true, y_pred, sample_weight)
        self.false_positives.update_state(y_true, y_pred, sample_weight)

    def result(self) -> TensorLike:
        """Compute precision ``TP / (TP + FP)``.

        Returns:
            TensorLike: Scalar or per-threshold precision tensor.
        """
        tp = self.true_positives.result()
        fp = self.false_positives.result()
        from zero_keras import ops

        res = ops.where(ops.greater(tp + fp, 0.0), ops.divide(tp, tp + fp + 1e-7), _to_tensor(0.0))
        return res

    def reset_state(self) -> None:
        """Reset true positive and false positive accumulators to zero."""
        self.true_positives.reset_state()
        self.false_positives.reset_state()


class Recall(Metric):
    """Computes the recall of the predictions with respect to the labels.

    This metric creates two local variables, `true_positives` and
    `false_negatives`, that are used to compute the recall. This value is
    ultimately returned as `recall`, an idempotent operation that simply divides
    `true_positives` by the sum of `true_positives` and `false_negatives`.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    If `top_k` is set, recall will be computed as how often on average a class
    among the labels of a batch entry is in the top-k predictions.

    If `class_id` is specified, we calculate recall by considering only the
    entries in the batch for which `class_id` is in the label, and computing the
    fraction of them for which `class_id` is above the threshold and/or in the
    top-k predictions.

    Args:
        thresholds: (Optional) A float value, or a Python list/tuple of float
            threshold values in `[0, 1]`. A threshold is compared with
            prediction values to determine the truth value of predictions (i.e.,
            above the threshold is `True`, below is `False`). If used with a
            loss function that sets `from_logits=True` (i.e. no sigmoid
            applied to predictions), `thresholds` should be set to 0.
            One metric value is generated for each threshold value.
            If neither `thresholds` nor `top_k` are set,
            the default is to calculate recall with `thresholds=0.5`.
        top_k: (Optional) Unset by default. An int value specifying the top-k
            predictions to consider when calculating recall.
        class_id: (Optional) Integer class ID for which we want binary metrics.
            This must be in the half-open interval `[0, num_classes)`, where
            `num_classes` is the last dimension of predictions.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:
    >>> m = keras.metrics.Recall()
    >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
    >>> m.result()
    0.6666667

    >>> m.reset_state()
    >>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1], sample_weight=[0, 0, 1, 0])
    >>> m.result()
    1.0

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='binary_crossentropy',
                  metrics=[keras.metrics.Recall()])
    ```

    Usage with a loss with `from_logits=True`:

    ```python
    model.compile(optimizer='adam',
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras.metrics.Recall(thresholds=0)])
    ```

    """

    def __init__(
        self,
        thresholds: float | Sequence[float] | None = None,
        top_k: int | None = None,
        class_id: int | None = None,
        name: str | None = None,
        dtype: DTypeLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the Recall metric.

        Args:
            thresholds: Decision threshold or sequence of thresholds in ``[0, 1]``.
            top_k: Optional number of top predictions to retain.
            class_id: Optional target class index.
            name: Optional string name of the metric instance.
            dtype: Optional data type of the metric result.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.thresholds = 0.5 if thresholds is None and top_k is None else thresholds
        self.top_k = top_k
        self.class_id = class_id

        self.true_positives = TruePositives(thresholds=self.thresholds)
        self.false_negatives = FalseNegatives(thresholds=self.thresholds)

    def update_state(
        self, y_true: TensorLike, y_pred: TensorLike, sample_weight: TensorLike | None = None
    ) -> None:
        """Accumulate true positive and false negative counts.

        Args:
            y_true: Ground-truth binary targets.
            y_pred: Predicted probabilities or scores.
            sample_weight: Optional sample weights.
        """
        y_true = _to_tensor(y_true)
        y_pred = _to_tensor(y_pred)
        y_true, y_pred = _filter_top_k_and_class_id(y_true, y_pred, self.top_k, self.class_id)
        self.true_positives.update_state(y_true, y_pred, sample_weight)
        self.false_negatives.update_state(y_true, y_pred, sample_weight)

    def result(self) -> TensorLike:
        """Compute recall ``TP / (TP + FN)``.

        Returns:
            TensorLike: Scalar or per-threshold recall tensor.
        """
        tp = self.true_positives.result()
        fn = self.false_negatives.result()
        from zero_keras import ops

        res = ops.where(ops.greater(tp + fn, 0.0), ops.divide(tp, tp + fn + 1e-7), _to_tensor(0.0))
        return res

    def reset_state(self) -> None:
        """Reset true positive and false negative accumulators to zero."""
        self.true_positives.reset_state()
        self.false_negatives.reset_state()


class PrecisionAtRecall(Metric):
    """Computes best precision where recall is >= specified value.

    This metric creates four local variables, `true_positives`,
    `true_negatives`, `false_positives` and `false_negatives` that are used to
    compute the precision at the given recall. The threshold for the given
    recall value is computed and used to evaluate the corresponding precision.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    If `class_id` is specified, we calculate precision by considering only the
    entries in the batch for which `class_id` is above the threshold
    predictions, and computing the fraction of them for which `class_id` is
    indeed a correct label.

    Args:
        recall: A scalar value in range `[0, 1]`.
        num_thresholds: (Optional) Defaults to 200. The number of thresholds to
            use for matching the given recall.
        class_id: (Optional) Integer class ID for which we want binary metrics.
            This must be in the half-open interval `[0, num_classes)`, where
            `num_classes` is the last dimension of predictions.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:
    >>> m = keras.metrics.PrecisionAtRecall(0.5)
    >>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8])
    >>> m.result()
    0.5

    >>> m.reset_state()
    >>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8],
    ...                sample_weight=[2, 2, 2, 1, 1])
    >>> m.result()
    0.33333333

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='binary_crossentropy',
        metrics=[keras.metrics.PrecisionAtRecall(recall=0.8)])
    ```

    """

    def __init__(
        self,
        recall: float,
        num_thresholds: int = 200,
        class_id: int | None = None,
        name: str | None = None,
        dtype: DTypeLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the PrecisionAtRecall metric.

        Args:
            recall: Minimum target recall in ``[0, 1]``.
            num_thresholds: Number of discretization thresholds.
            class_id: Optional target class index.
            name: Optional string name of the metric instance.
            dtype: Optional data type of the metric result.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.recall = recall
        self.num_thresholds = num_thresholds
        self.class_id = class_id

        self.thresholds = [i / (self.num_thresholds - 1) for i in range(self.num_thresholds)]
        self.true_positives = TruePositives(thresholds=self.thresholds)
        self.false_positives = FalsePositives(thresholds=self.thresholds)
        self.true_negatives = TrueNegatives(thresholds=self.thresholds)
        self.false_negatives = FalseNegatives(thresholds=self.thresholds)

    def update_state(
        self, y_true: TensorLike, y_pred: TensorLike, sample_weight: TensorLike | None = None
    ) -> None:
        """Accumulate confusion-matrix statistics across thresholds.

        Args:
            y_true: Ground-truth binary targets.
            y_pred: Predicted probabilities or scores.
            sample_weight: Optional sample weights.
        """
        y_true = _to_tensor(y_true)
        y_pred = _to_tensor(y_pred)
        y_true, y_pred = _filter_top_k_and_class_id(y_true, y_pred, self.top_k, self.class_id)
        self.true_positives.update_state(y_true, y_pred, sample_weight)
        self.false_positives.update_state(y_true, y_pred, sample_weight)
        self.true_negatives.update_state(y_true, y_pred, sample_weight)
        self.false_negatives.update_state(y_true, y_pred, sample_weight)

    def result(self) -> TensorLike:
        """Compute maximum precision across thresholds satisfying ``recall >= self.recall``.

        Returns:
            TensorLike: Scalar precision tensor.
        """
        tp = self.true_positives.result()
        fp = self.false_positives.result()
        self.true_negatives.result()
        fn = self.false_negatives.result()
        from zero_keras import ops

        recalls = ops.where(
            ops.greater(tp + fn, 0.0), ops.divide(tp, tp + fn + 1e-7), _to_tensor(0.0)
        )
        precisions = ops.where(
            ops.greater(tp + fp, 0.0), ops.divide(tp, tp + fp + 1e-7), _to_tensor(0.0)
        )

        condition = ops.greater_equal(recalls, self.recall)
        valid_precisions = ops.where(condition, precisions, _to_tensor(0.0))
        return ops.max(valid_precisions)

    def reset_state(self) -> None:
        """Reset all confusion-matrix accumulators to zero."""
        self.true_positives.reset_state()
        self.false_positives.reset_state()
        self.true_negatives.reset_state()
        self.false_negatives.reset_state()


class RecallAtPrecision(Metric):
    """Computes best recall where precision is >= specified value.

    For a given score-label-distribution the required precision might not
    be achievable, in this case 0.0 is returned as recall.

    This metric creates four local variables, `true_positives`,
    `true_negatives`, `false_positives` and `false_negatives` that are used to
    compute the recall at the given precision. The threshold for the given
    precision value is computed and used to evaluate the corresponding recall.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    If `class_id` is specified, we calculate precision by considering only the
    entries in the batch for which `class_id` is above the threshold
    predictions, and computing the fraction of them for which `class_id` is
    indeed a correct label.

    Args:
        precision: A scalar value in range `[0, 1]`.
        num_thresholds: (Optional) Defaults to 200. The number of thresholds
            to use for matching the given precision.
        class_id: (Optional) Integer class ID for which we want binary metrics.
            This must be in the half-open interval `[0, num_classes)`, where
            `num_classes` is the last dimension of predictions.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:
    >>> m = keras.metrics.RecallAtPrecision(0.8)
    >>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])
    >>> m.result()
    0.5

    >>> m.reset_state()
    >>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9],
    ...                sample_weight=[1, 0, 0, 1])
    >>> m.result()
    1.0

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='binary_crossentropy',
        metrics=[keras.metrics.RecallAtPrecision(precision=0.8)])
    ```

    """

    def __init__(
        self,
        precision: float,
        num_thresholds: int = 200,
        class_id: int | None = None,
        name: str | None = None,
        dtype: DTypeLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the RecallAtPrecision metric.

        Args:
            precision: Minimum target precision in ``[0, 1]``.
            num_thresholds: Number of discretization thresholds.
            class_id: Optional target class index.
            name: Optional string name of the metric instance.
            dtype: Optional data type of the metric result.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.precision = precision
        self.num_thresholds = num_thresholds
        self.class_id = class_id

        self.thresholds = [i / (self.num_thresholds - 1) for i in range(self.num_thresholds)]
        self.true_positives = TruePositives(thresholds=self.thresholds)
        self.false_positives = FalsePositives(thresholds=self.thresholds)
        self.true_negatives = TrueNegatives(thresholds=self.thresholds)
        self.false_negatives = FalseNegatives(thresholds=self.thresholds)

    def update_state(
        self, y_true: TensorLike, y_pred: TensorLike, sample_weight: TensorLike | None = None
    ) -> None:
        """Accumulate confusion-matrix statistics across thresholds.

        Args:
            y_true: Ground-truth binary targets.
            y_pred: Predicted probabilities or scores.
            sample_weight: Optional sample weights.
        """
        y_true = _to_tensor(y_true)
        y_pred = _to_tensor(y_pred)
        y_true, y_pred = _filter_top_k_and_class_id(y_true, y_pred, self.top_k, self.class_id)
        self.true_positives.update_state(y_true, y_pred, sample_weight)
        self.false_positives.update_state(y_true, y_pred, sample_weight)
        self.true_negatives.update_state(y_true, y_pred, sample_weight)
        self.false_negatives.update_state(y_true, y_pred, sample_weight)

    def result(self) -> TensorLike:
        """Compute maximum recall across thresholds satisfying ``precision >= self.precision``.

        Returns:
            TensorLike: Scalar recall tensor.
        """
        tp = self.true_positives.result()
        fp = self.false_positives.result()
        self.true_negatives.result()
        fn = self.false_negatives.result()
        from zero_keras import ops

        recalls = ops.where(
            ops.greater(tp + fn, 0.0), ops.divide(tp, tp + fn + 1e-7), _to_tensor(0.0)
        )
        precisions = ops.where(
            ops.greater(tp + fp, 0.0), ops.divide(tp, tp + fp + 1e-7), _to_tensor(0.0)
        )

        condition = ops.greater_equal(precisions, self.precision)
        valid_recalls = ops.where(condition, recalls, _to_tensor(0.0))
        return ops.max(valid_recalls)

    def reset_state(self) -> None:
        """Reset all confusion-matrix accumulators to zero."""
        self.true_positives.reset_state()
        self.false_positives.reset_state()
        self.true_negatives.reset_state()
        self.false_negatives.reset_state()


class SensitivityAtSpecificity(Metric):
    """Computes best sensitivity where specificity is >= specified value.

    `Sensitivity` measures the proportion of actual positives that are correctly
    identified as such `(tp / (tp + fn))`.
    `Specificity` measures the proportion of actual negatives that are correctly
    identified as such `(tn / (tn + fp))`.

    This metric creates four local variables, `true_positives`,
    `true_negatives`, `false_positives` and `false_negatives` that are used to
    compute the sensitivity at the given specificity. The threshold for the
    given specificity value is computed and used to evaluate the corresponding
    sensitivity.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    If `class_id` is specified, we calculate precision by considering only the
    entries in the batch for which `class_id` is above the threshold
    predictions, and computing the fraction of them for which `class_id` is
    indeed a correct label.

    For additional information about specificity and sensitivity, see
    [the following](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).

    Args:
        specificity: A scalar value in range `[0, 1]`.
        num_thresholds: (Optional) Defaults to 200. The number of thresholds to
            use for matching the given specificity.
        class_id: (Optional) Integer class ID for which we want binary metrics.
            This must be in the half-open interval `[0, num_classes)`, where
            `num_classes` is the last dimension of predictions.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:
    >>> m = keras.metrics.SensitivityAtSpecificity(0.5)
    >>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8])
    >>> m.result()
    0.5

    >>> m.reset_state()
    >>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8],
    ...                sample_weight=[1, 1, 2, 2, 1])
    >>> m.result()
    0.333333

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='binary_crossentropy',
        metrics=[keras.metrics.SensitivityAtSpecificity(specificity=0.5)])
    ```

    """

    def __init__(
        self,
        specificity: float,
        num_thresholds: int = 200,
        class_id: int | None = None,
        name: str | None = None,
        dtype: DTypeLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the SensitivityAtSpecificity metric.

        Args:
            specificity: Minimum target specificity in ``[0, 1]``.
            num_thresholds: Number of discretization thresholds.
            class_id: Optional target class index.
            name: Optional string name of the metric instance.
            dtype: Optional data type of the metric result.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.specificity = specificity
        self.num_thresholds = num_thresholds
        self.class_id = class_id

        self.thresholds = [i / (self.num_thresholds - 1) for i in range(self.num_thresholds)]
        self.true_positives = TruePositives(thresholds=self.thresholds)
        self.false_positives = FalsePositives(thresholds=self.thresholds)
        self.true_negatives = TrueNegatives(thresholds=self.thresholds)
        self.false_negatives = FalseNegatives(thresholds=self.thresholds)

    def update_state(
        self, y_true: TensorLike, y_pred: TensorLike, sample_weight: TensorLike | None = None
    ) -> None:
        """Accumulate confusion-matrix statistics across thresholds.

        Args:
            y_true: Ground-truth binary targets.
            y_pred: Predicted probabilities or scores.
            sample_weight: Optional sample weights.
        """
        y_true = _to_tensor(y_true)
        y_pred = _to_tensor(y_pred)
        y_true, y_pred = _filter_top_k_and_class_id(y_true, y_pred, self.top_k, self.class_id)
        self.true_positives.update_state(y_true, y_pred, sample_weight)
        self.false_positives.update_state(y_true, y_pred, sample_weight)
        self.true_negatives.update_state(y_true, y_pred, sample_weight)
        self.false_negatives.update_state(y_true, y_pred, sample_weight)

    def result(self) -> TensorLike:
        """Compute maximum sensitivity across thresholds satisfying ``specificity >= self.specificity``.

        Returns:
            TensorLike: Scalar sensitivity tensor.
        """
        tp = self.true_positives.result()
        fp = self.false_positives.result()
        tn = self.true_negatives.result()
        fn = self.false_negatives.result()
        from zero_keras import ops

        specificities = ops.where(
            ops.greater(tn + fp, 0.0), ops.divide(tn, tn + fp + 1e-7), _to_tensor(0.0)
        )
        sensitivities = ops.where(
            ops.greater(tp + fn, 0.0), ops.divide(tp, tp + fn + 1e-7), _to_tensor(0.0)
        )

        condition = ops.greater_equal(specificities, self.specificity)
        valid_sensitivities = ops.where(condition, sensitivities, _to_tensor(0.0))
        return ops.max(valid_sensitivities)

    def reset_state(self) -> None:
        """Reset all confusion-matrix accumulators to zero."""
        self.true_positives.reset_state()
        self.false_positives.reset_state()
        self.true_negatives.reset_state()
        self.false_negatives.reset_state()


class SpecificityAtSensitivity(Metric):
    """Computes best specificity where sensitivity is >= specified value.

    `Sensitivity` measures the proportion of actual positives that are correctly
    identified as such `(tp / (tp + fn))`.
    `Specificity` measures the proportion of actual negatives that are correctly
    identified as such `(tn / (tn + fp))`.

    This metric creates four local variables, `true_positives`,
    `true_negatives`, `false_positives` and `false_negatives` that are used to
    compute the specificity at the given sensitivity. The threshold for the
    given sensitivity value is computed and used to evaluate the corresponding
    specificity.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    If `class_id` is specified, we calculate precision by considering only the
    entries in the batch for which `class_id` is above the threshold
    predictions, and computing the fraction of them for which `class_id` is
    indeed a correct label.

    For additional information about specificity and sensitivity, see
    [the following](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).

    Args:
        sensitivity: A scalar value in range `[0, 1]`.
        num_thresholds: (Optional) Defaults to 200. The number of thresholds to
            use for matching the given sensitivity.
        class_id: (Optional) Integer class ID for which we want binary metrics.
            This must be in the half-open interval `[0, num_classes)`, where
            `num_classes` is the last dimension of predictions.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:
    >>> m = keras.metrics.SpecificityAtSensitivity(0.5)
    >>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8])
    >>> m.result()
    0.66666667

    >>> m.reset_state()
    >>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8],
    ...                sample_weight=[1, 1, 2, 2, 2])
    >>> m.result()
    0.5

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='binary_crossentropy',
        metrics=[keras.metrics.SpecificityAtSensitivity()])
    ```

    """

    def __init__(
        self,
        sensitivity: float,
        num_thresholds: int = 200,
        class_id: int | None = None,
        name: str | None = None,
        dtype: DTypeLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the SpecificityAtSensitivity metric.

        Args:
            sensitivity: Minimum target sensitivity in ``[0, 1]``.
            num_thresholds: Number of discretization thresholds.
            class_id: Optional target class index.
            name: Optional string name of the metric instance.
            dtype: Optional data type of the metric result.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.sensitivity = sensitivity
        self.num_thresholds = num_thresholds
        self.class_id = class_id

        self.thresholds = [i / (self.num_thresholds - 1) for i in range(self.num_thresholds)]
        self.true_positives = TruePositives(thresholds=self.thresholds)
        self.false_positives = FalsePositives(thresholds=self.thresholds)
        self.true_negatives = TrueNegatives(thresholds=self.thresholds)
        self.false_negatives = FalseNegatives(thresholds=self.thresholds)

    def update_state(
        self, y_true: TensorLike, y_pred: TensorLike, sample_weight: TensorLike | None = None
    ) -> None:
        """Accumulate confusion-matrix statistics across thresholds.

        Args:
            y_true: Ground-truth binary targets.
            y_pred: Predicted probabilities or scores.
            sample_weight: Optional sample weights.
        """
        y_true = _to_tensor(y_true)
        y_pred = _to_tensor(y_pred)
        y_true, y_pred = _filter_top_k_and_class_id(y_true, y_pred, self.top_k, self.class_id)
        self.true_positives.update_state(y_true, y_pred, sample_weight)
        self.false_positives.update_state(y_true, y_pred, sample_weight)
        self.true_negatives.update_state(y_true, y_pred, sample_weight)
        self.false_negatives.update_state(y_true, y_pred, sample_weight)

    def result(self) -> TensorLike:
        """Compute maximum specificity across thresholds satisfying ``sensitivity >= self.sensitivity``.

        Returns:
            TensorLike: Scalar specificity tensor.
        """
        tp = self.true_positives.result()
        fp = self.false_positives.result()
        tn = self.true_negatives.result()
        fn = self.false_negatives.result()
        from zero_keras import ops

        specificities = ops.where(
            ops.greater(tn + fp, 0.0), ops.divide(tn, tn + fp + 1e-7), _to_tensor(0.0)
        )
        sensitivities = ops.where(
            ops.greater(tp + fn, 0.0), ops.divide(tp, tp + fn + 1e-7), _to_tensor(0.0)
        )

        condition = ops.greater_equal(sensitivities, self.sensitivity)
        valid_specificities = ops.where(condition, specificities, _to_tensor(0.0))
        return ops.max(valid_specificities)

    def reset_state(self) -> None:
        """Reset all confusion-matrix accumulators to zero."""
        self.true_positives.reset_state()
        self.false_positives.reset_state()
        self.true_negatives.reset_state()
        self.false_negatives.reset_state()


class AUC(Metric):
    """Approximates the AUC (Area under the curve) of the ROC or PR curves.

    The AUC (Area under the curve) of the ROC (Receiver operating
    characteristic; default) or PR (Precision Recall) curves are quality
    measures of binary classifiers. Unlike the accuracy, and like cross-entropy
    losses, ROC-AUC and PR-AUC evaluate all the operational points of a model.

    This class approximates AUCs using a Riemann sum. During the metric
    accumulation phrase, predictions are accumulated within predefined buckets
    by value. The AUC is then computed by interpolating per-bucket averages.
    These buckets define the evaluated operational points.

    This metric creates four local variables, `true_positives`,
    `true_negatives`, `false_positives` and `false_negatives` that are used to
    compute the AUC.  To discretize the AUC curve, a linearly spaced set of
    thresholds is used to compute pairs of recall and precision values. The area
    under the ROC-curve is therefore computed using the height of the recall
    values by the false positive rate, while the area under the PR-curve is the
    computed using the height of the precision values by the recall.

    This value is ultimately returned as `auc`, an idempotent operation that
    computes the area under a discretized curve of precision versus recall
    values (computed using the aforementioned variables). The `num_thresholds`
    variable controls the degree of discretization with larger numbers of
    thresholds more closely approximating the true AUC. The quality of the
    approximation may vary dramatically depending on `num_thresholds`. The
    `thresholds` parameter can be used to manually specify thresholds which
    split the predictions more evenly.

    For a best approximation of the real AUC, `predictions` should be
    distributed approximately uniformly in the range `[0, 1]` (if
    `from_logits=False`). The quality of the AUC approximation may be poor if
    this is not the case. Setting `summation_method` to 'minoring' or 'majoring'
    can help quantify the error in the approximation by providing lower or upper
    bound estimate of the AUC.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
        num_thresholds: (Optional) The number of thresholds to
            use when discretizing the roc curve. Values must be > 1.
            Defaults to `200`.
        curve: (Optional) Specifies the name of the curve to be computed,
            `'ROC'` (default) or `'PR'` for the Precision-Recall-curve.
        summation_method: (Optional) Specifies the [Riemann summation method](
              https://en.wikipedia.org/wiki/Riemann_sum) used.
              'interpolation' (default) applies mid-point summation scheme for
              `ROC`.  For PR-AUC, interpolates (true/false) positives but not
              the ratio that is precision (see Davis & Goadrich 2006 for
              details); 'minoring' applies left summation for increasing
              intervals and right summation for decreasing intervals; 'majoring'
              does the opposite.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        thresholds: (Optional) A list of floating point values to use as the
            thresholds for discretizing the curve. If set, the `num_thresholds`
            parameter is ignored. Values should be in `[0, 1]`. Endpoint
            thresholds equal to {`-epsilon`, `1+epsilon`} for a small positive
            epsilon value will be automatically included with these to correctly
            handle predictions equal to exactly 0 or 1.
        multi_label: boolean indicating whether multilabel data should be
            treated as such, wherein AUC is computed separately for each label
            and then averaged across labels, or (when `False`) if the data
            should be flattened into a single label before AUC computation. In
            the latter case, when multilabel data is passed to AUC, each
            label-prediction pair is treated as an individual data point. Should
            be set to `False` for multi-class data.
        num_labels: (Optional) The number of labels, used when `multi_label` is
            True. If `num_labels` is not specified, then state variables get
            created on the first call to `update_state`.
        label_weights: (Optional) list, array, or tensor of non-negative weights
            used to compute AUCs for multilabel data. When `multi_label` is
            True, the weights are applied to the individual label AUCs when they
            are averaged to produce the multi-label AUC. When it's False, they
            are used to weight the individual label predictions in computing the
            confusion matrix on the flattened data. Note that this is unlike
            `class_weights` in that `class_weights` weights the example
            depending on the value of its label, whereas `label_weights` depends
            only on the index of that label before flattening; therefore
            `label_weights` should not be used for multi-class data.
        from_logits: boolean indicating whether the predictions (`y_pred` in
        `update_state`) are probabilities or sigmoid logits. As a rule of thumb,
        when using a keras loss, the `from_logits` constructor argument of the
        loss should match the AUC `from_logits` constructor argument.

    Example:
    >>> m = keras.metrics.AUC(num_thresholds=3)
    >>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])
    >>> # threshold values are [0 - 1e-7, 0.5, 1 + 1e-7]
    >>> # tp = [2, 1, 0], fp = [2, 0, 0], fn = [0, 1, 2], tn = [0, 2, 2]
    >>> # tp_rate = recall = [1, 0.5, 0], fp_rate = [1, 0, 0]
    >>> # auc = ((((1 + 0.5) / 2) * (1 - 0)) + (((0.5 + 0) / 2) * (0 - 0)))
    >>> #     = 0.75
    >>> m.result()
    0.75

    >>> m.reset_state()
    >>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9],
    ...                sample_weight=[1, 0, 0, 1])
    >>> m.result()
    1.0

    Usage with `compile()` API:

    ```python
    # Reports the AUC of a model outputting a probability.
    model.compile(optimizer='sgd',
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.AUC()])

    # Reports the AUC of a model outputting a logit.
    model.compile(optimizer='sgd',
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras.metrics.AUC(from_logits=True)])
    ```

    """

    thresholds: list[float] = []

    def interpolate_pr_auc(self) -> TensorLike:
        """Interpolate the area under the precision-recall curve.

        Returns:
            TensorLike: Interpolated PR-AUC value.
        """
        return self.result()

    def __init__(
        self,
        num_thresholds: int = 200,
        curve: str = "ROC",
        summation_method: str = "interpolation",
        name: str | None = None,
        dtype: DTypeLike = None,
        thresholds: Sequence[float] | None = None,
        multi_label: bool = False,
        num_labels: int | None = None,
        label_weights: TensorLike | None = None,
        from_logits: bool = False,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the AUC metric.

        Args:
            num_thresholds: Number of discretization thresholds.
            curve: Curve type (``"ROC"`` or ``"PR"``).
            summation_method: Riemann summation scheme (``"interpolation"``, ``"minoring"``, or ``"majoring"``).
            name: Optional string name of the metric instance.
            dtype: Optional data type of the metric result.
            thresholds: Optional explicit sequence of thresholds in ``[0, 1]``.
            multi_label: Whether to compute multi-label AUC independently per label.
            num_labels: Optional number of multi-label classes.
            label_weights: Optional non-negative per-label weights.
            from_logits: Whether ``y_pred`` contains raw sigmoid logits.
            **kwargs: Additional keyword arguments.
        """
        _ = multi_label, num_labels, label_weights
        if curve not in ("ROC", "PR"):
            raise ValueError(f"Invalid curve: {curve}. Expected 'ROC' or 'PR'.")
        if summation_method not in ("interpolation", "minoring", "majoring"):
            raise ValueError(
                f"Invalid summation_method: {summation_method}. "
                "Expected 'interpolation', 'minoring' or 'majoring'."
            )
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.num_thresholds = num_thresholds
        self.curve = curve
        self.summation_method = summation_method
        self.from_logits = from_logits

        if thresholds is not None:
            self.thresholds = list(thresholds)
        else:
            self.thresholds = [i / (self.num_thresholds - 1) for i in range(self.num_thresholds)]

        self.true_positives = TruePositives(thresholds=self.thresholds)
        self.false_positives = FalsePositives(thresholds=self.thresholds)
        self.true_negatives = TrueNegatives(thresholds=self.thresholds)
        self.false_negatives = FalseNegatives(thresholds=self.thresholds)

    def update_state(
        self, y_true: TensorLike, y_pred: TensorLike, sample_weight: TensorLike | None = None
    ) -> None:
        """Accumulate confusion-matrix statistics across discretization thresholds.

        Args:
            y_true: Ground-truth binary targets.
            y_pred: Predicted probabilities or logits.
            sample_weight: Optional sample weights.
        """
        if self.from_logits:
            from zero_keras.activations import sigmoid

            y_pred = sigmoid(y_pred)

        self.true_positives.update_state(y_true, y_pred, sample_weight)
        self.false_positives.update_state(y_true, y_pred, sample_weight)
        self.true_negatives.update_state(y_true, y_pred, sample_weight)
        self.false_negatives.update_state(y_true, y_pred, sample_weight)

    def result(self) -> TensorLike:
        """Approximate the area under the ROC or PR curve via Riemann summation.

        Returns:
            TensorLike: Scalar AUC tensor.
        """
        tp = self.true_positives.result()
        fp = self.false_positives.result()
        tn = self.true_negatives.result()
        fn = self.false_negatives.result()
        from zero_keras import ops

        if self.curve == "ROC":
            tpr = ops.where(
                ops.greater(tp + fn, 0.0),
                ops.divide(tp, tp + fn + 1e-7),
                _to_tensor(0.0),
            )
            fpr = ops.where(
                ops.greater(fp + tn, 0.0),
                ops.divide(fp, fp + tn + 1e-7),
                _to_tensor(0.0),
            )
            x = fpr
            y = tpr
        elif self.curve == "PR":
            precision = ops.where(
                ops.greater(tp + fp, 0.0),
                ops.divide(tp, tp + fp + 1e-7),
                _to_tensor(1.0),
            )
            recall = ops.where(
                ops.greater(tp + fn, 0.0),
                ops.divide(tp, tp + fn + 1e-7),
                _to_tensor(0.0),
            )
            x = recall
            y = precision
        else:
            raise ValueError(f"Invalid curve: {self.curve}")

        x_diff = x[:-1] - x[1:]
        if self.summation_method in ("minoring", "minor"):
            y_val = ops.minimum(y[:-1], y[1:])
        elif self.summation_method in ("majoring", "major"):
            y_val = ops.maximum(y[:-1], y[1:])
        else:
            y_val = (y[:-1] + y[1:]) / 2.0
        return ops.sum(x_diff * y_val)

    def reset_state(self) -> None:
        """Reset all confusion-matrix accumulators to zero."""
        self.true_positives.reset_state()
        self.false_positives.reset_state()
        self.true_negatives.reset_state()
        self.false_negatives.reset_state()


class CosineSimilarity(Metric):
    """Computes the cosine similarity between the labels and predictions.

    Formula:

    ```python
    loss = sum(l2_norm(y_true) * l2_norm(y_pred))
    ```
    See: [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity).
    This metric keeps the average cosine similarity between `predictions` and
    `labels` over a stream of data.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        axis: (Optional) Defaults to `-1`. The dimension along which the cosine
            similarity is computed.

    Examples:
    >>> # l2_norm(y_true) = [[0., 1.], [1./1.414, 1./1.414]]
    >>> # l2_norm(y_pred) = [[1., 0.], [1./1.414, 1./1.414]]
    >>> # l2_norm(y_true) . l2_norm(y_pred) = [[0., 0.], [0.5, 0.5]]
    >>> # result = mean(sum(l2_norm(y_true) . l2_norm(y_pred), axis=1))
    >>> #        = ((0. + 0.) +  (0.5 + 0.5)) / 2
    >>> m = keras.metrics.CosineSimilarity(axis=1)
    >>> m.update_state([[0., 1.], [1., 1.]], [[1., 0.], [1., 1.]])
    >>> m.result()
    0.49999997

    >>> m.reset_state()
    >>> m.update_state([[0., 1.], [1., 1.]], [[1., 0.], [1., 1.]],
    ...                sample_weight=[0.3, 0.7])
    >>> m.result()
    0.6999999

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[keras.metrics.CosineSimilarity(axis=1)])
    ```

    """

    def __init__(
        self,
        name: str = "cosine_similarity",
        dtype: DTypeLike = None,
        axis: int = -1,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the CosineSimilarity metric.

        Args:
            name: String name of the metric instance.
            dtype: Data type of the metric result.
            axis: Dimension along which cosine similarity is computed.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.axis = axis
        self.total = self.add_weight(shape=(), initializer="zeros", name="total")
        self.count = self.add_weight(shape=(), initializer="zeros", name="count")

    def update_state(
        self, y_true: TensorLike, y_pred: TensorLike, sample_weight: TensorLike | None = None
    ) -> None:
        """Accumulate statistics for the metric.

        Args:
            y_true: Ground-truth target values.
            y_pred: Predicted values.
            sample_weight: Optional weighting for each example.
        """
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        from zero_keras import losses

        val = -ops.mean(losses.cosine_similarity(y_true, y_pred))
        if sample_weight is not None:
            val = ops.multiply(val, ops.mean(_to_tensor(sample_weight)))
        self.total.assign(ops.add(_to_tensor(self.total), val))
        self.count.assign(ops.add(_to_tensor(self.count), _to_tensor(1.0)))

    def result(self) -> TensorLike:
        """Compute the current metric value.

        Returns:
            TensorLike: A scalar tensor, or a dictionary of scalar tensors.
        """
        return _wrap(
            ops.sqrt(
                ops.divide(
                    _to_tensor(self.total),
                    ops.maximum(_to_tensor(self.count), _to_tensor(1e-7)),
                )
            )
        )


class MeanAbsoluteError(Metric):
    """Computes the mean absolute error between the labels and predictions.

    Formula:

    ```python
    loss = mean(abs(y_true - y_pred))
    ```

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Examples:
    >>> m = keras.metrics.MeanAbsoluteError()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result()
    0.25

    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
    ...                sample_weight=[1, 0])
    >>> m.result()
    0.5

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[keras.metrics.MeanAbsoluteError()])
    ```

    """

    def __init__(
        self, name: str = "mean_absolute_error", dtype: DTypeLike = None, **kwargs: TensorLike
    ) -> None:
        """Initialize the MeanAbsoluteError metric.

        Args:
            name: String name of the metric instance.
            dtype: Data type of the metric result.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = self.add_weight(shape=(), initializer="zeros", name="total")
        self.count = self.add_weight(shape=(), initializer="zeros", name="count")

    def update_state(
        self, y_true: TensorLike, y_pred: TensorLike, sample_weight: TensorLike | None = None
    ) -> None:
        """Accumulate statistics for the metric.

        Args:
            y_true: Ground-truth target values.
            y_pred: Predicted values.
            sample_weight: Optional weighting for each example.
        """
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        from zero_keras import losses

        val = ops.mean(losses.mean_absolute_error(y_true, y_pred))
        if sample_weight is not None:
            val = ops.multiply(val, ops.mean(_to_tensor(sample_weight)))
        self.total.assign(ops.add(_to_tensor(self.total), val))
        self.count.assign(ops.add(_to_tensor(self.count), _to_tensor(1.0)))

    def result(self) -> TensorLike:
        """Compute the current metric value.

        Returns:
            TensorLike: A scalar tensor, or a dictionary of scalar tensors.
        """
        return _wrap(
            ops.sqrt(
                ops.divide(
                    _to_tensor(self.total),
                    ops.maximum(_to_tensor(self.count), _to_tensor(1e-7)),
                )
            )
        )


class MeanAbsolutePercentageError(Metric):
    """Computes mean absolute percentage error between `y_true` and `y_pred`.

    Formula:

    ```python
    loss = 100 * mean(abs((y_true - y_pred) / y_true))
    ```

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Examples:
    >>> m = keras.metrics.MeanAbsolutePercentageError()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result()
    250000000.0

    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
    ...                sample_weight=[1, 0])
    >>> m.result()
    500000000.0

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[keras.metrics.MeanAbsolutePercentageError()])
    ```

    """

    def __init__(
        self,
        name: str = "mean_absolute_percentage_error",
        dtype: DTypeLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the MeanAbsolutePercentageError metric.

        Args:
            name: String name of the metric instance.
            dtype: Data type of the metric result.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = self.add_weight(shape=(), initializer="zeros", name="total")
        self.count = self.add_weight(shape=(), initializer="zeros", name="count")

    def update_state(
        self, y_true: TensorLike, y_pred: TensorLike, sample_weight: TensorLike | None = None
    ) -> None:
        """Accumulate statistics for the metric.

        Args:
            y_true: Ground-truth target values.
            y_pred: Predicted values.
            sample_weight: Optional weighting for each example.
        """
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        from zero_keras import losses

        val = ops.mean(losses.mean_absolute_percentage_error(y_true, y_pred))
        if sample_weight is not None:
            val = ops.multiply(val, ops.mean(_to_tensor(sample_weight)))
        self.total.assign(ops.add(_to_tensor(self.total), val))
        self.count.assign(ops.add(_to_tensor(self.count), _to_tensor(1.0)))

    def result(self) -> TensorLike:
        """Compute the current metric value.

        Returns:
            TensorLike: A scalar tensor, or a dictionary of scalar tensors.
        """
        return _wrap(
            ops.sqrt(
                ops.divide(
                    _to_tensor(self.total),
                    ops.maximum(_to_tensor(self.count), _to_tensor(1e-7)),
                )
            )
        )


class MeanSquaredError(Metric):
    """Computes the mean squared error between `y_true` and `y_pred`.

    Formula:

    ```python
    loss = mean(square(y_true - y_pred))
    ```

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:
    >>> m = keras.metrics.MeanSquaredError()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result()
    0.25

    """

    def __init__(
        self, name: str = "mean_squared_error", dtype: DTypeLike = None, **kwargs: TensorLike
    ) -> None:
        """Initialize the MeanSquaredError metric.

        Args:
            name: String name of the metric instance.
            dtype: Data type of the metric result.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = self.add_weight(shape=(), initializer="zeros", name="total")
        self.count = self.add_weight(shape=(), initializer="zeros", name="count")

    def update_state(
        self, y_true: TensorLike, y_pred: TensorLike, sample_weight: TensorLike | None = None
    ) -> None:
        """Accumulate statistics for the metric.

        Args:
            y_true: Ground-truth target values.
            y_pred: Predicted values.
            sample_weight: Optional weighting for each example.
        """
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        from zero_keras import losses

        val = ops.mean(losses.mean_squared_error(y_true, y_pred))
        if sample_weight is not None:
            val = ops.multiply(val, ops.mean(_to_tensor(sample_weight)))
        self.total.assign(ops.add(_to_tensor(self.total), val))
        self.count.assign(ops.add(_to_tensor(self.count), _to_tensor(1.0)))

    def result(self) -> TensorLike:
        """Compute the current metric value.

        Returns:
            TensorLike: A scalar tensor, or a dictionary of scalar tensors.
        """
        return _wrap(
            ops.sqrt(
                ops.divide(
                    _to_tensor(self.total),
                    ops.maximum(_to_tensor(self.count), _to_tensor(1e-7)),
                )
            )
        )


class MeanSquaredLogarithmicError(Metric):
    """Computes mean squared logarithmic error between `y_true` and `y_pred`.

    Formula:

    ```python
    loss = mean(square(log(y_true + 1) - log(y_pred + 1)))
    ```

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Examples:
    >>> m = keras.metrics.MeanSquaredLogarithmicError()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result()
    0.12011322

    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
    ...                sample_weight=[1, 0])
    >>> m.result()
    0.24022643

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[keras.metrics.MeanSquaredLogarithmicError()])
    ```

    """

    def __init__(
        self,
        name: str = "mean_squared_logarithmic_error",
        dtype: DTypeLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the MeanSquaredLogarithmicError metric.

        Args:
            name: String name of the metric instance.
            dtype: Data type of the metric result.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = self.add_weight(shape=(), initializer="zeros", name="total")
        self.count = self.add_weight(shape=(), initializer="zeros", name="count")

    def update_state(
        self, y_true: TensorLike, y_pred: TensorLike, sample_weight: TensorLike | None = None
    ) -> None:
        """Accumulate statistics for the metric.

        Args:
            y_true: Ground-truth target values.
            y_pred: Predicted values.
            sample_weight: Optional weighting for each example.
        """
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        from zero_keras import losses

        val = ops.mean(losses.mean_squared_logarithmic_error(y_true, y_pred))
        if sample_weight is not None:
            val = ops.multiply(val, ops.mean(_to_tensor(sample_weight)))
        self.total.assign(ops.add(_to_tensor(self.total), val))
        self.count.assign(ops.add(_to_tensor(self.count), _to_tensor(1.0)))

    def result(self) -> TensorLike:
        """Compute the current metric value.

        Returns:
            TensorLike: A scalar tensor, or a dictionary of scalar tensors.
        """
        return _wrap(
            ops.sqrt(
                ops.divide(
                    _to_tensor(self.total),
                    ops.maximum(_to_tensor(self.count), _to_tensor(1e-7)),
                )
            )
        )


class Hinge(Metric):
    """Computes the hinge metric between `y_true` and `y_pred`.

    `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
    provided we will convert them to -1 or 1.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Examples:
    >>> m = keras.metrics.Hinge()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
    >>> m.result()
    1.3
    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
    ...                sample_weight=[1, 0])
    >>> m.result()
    1.1

    """

    def __init__(self, name: str = "hinge", dtype: DTypeLike = None, **kwargs: TensorLike) -> None:
        """Initialize the Hinge metric.

        Args:
            name: String name of the metric instance.
            dtype: Data type of the metric result.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = self.add_weight(shape=(), initializer="zeros", name="total")
        self.count = self.add_weight(shape=(), initializer="zeros", name="count")

    def update_state(
        self, y_true: TensorLike, y_pred: TensorLike, sample_weight: TensorLike | None = None
    ) -> None:
        """Accumulate statistics for the metric.

        Args:
            y_true: Ground-truth target values.
            y_pred: Predicted values.
            sample_weight: Optional weighting for each example.
        """
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        from zero_keras import losses

        val = ops.mean(losses.hinge(y_true, y_pred))
        if sample_weight is not None:
            val = ops.multiply(val, ops.mean(_to_tensor(sample_weight)))
        self.total.assign(ops.add(_to_tensor(self.total), val))
        self.count.assign(ops.add(_to_tensor(self.count), _to_tensor(1.0)))

    def result(self) -> TensorLike:
        """Compute the current metric value.

        Returns:
            TensorLike: A scalar tensor, or a dictionary of scalar tensors.
        """
        return _wrap(
            ops.sqrt(
                ops.divide(
                    _to_tensor(self.total),
                    ops.maximum(_to_tensor(self.count), _to_tensor(1e-7)),
                )
            )
        )


class SquaredHinge(Metric):
    """Computes the hinge metric between `y_true` and `y_pred`.

    `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
    provided we will convert them to -1 or 1.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:
    >>> m = keras.metrics.SquaredHinge()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
    >>> m.result()
    1.86
    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
    ...                sample_weight=[1, 0])
    >>> m.result()
    1.46

    """

    def __init__(
        self, name: str = "squared_hinge", dtype: DTypeLike = None, **kwargs: TensorLike
    ) -> None:
        """Initialize the SquaredHinge metric.

        Args:
            name: String name of the metric instance.
            dtype: Data type of the metric result.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = self.add_weight(shape=(), initializer="zeros", name="total")
        self.count = self.add_weight(shape=(), initializer="zeros", name="count")

    def update_state(
        self, y_true: TensorLike, y_pred: TensorLike, sample_weight: TensorLike | None = None
    ) -> None:
        """Accumulate statistics for the metric.

        Args:
            y_true: Ground-truth target values.
            y_pred: Predicted values.
            sample_weight: Optional weighting for each example.
        """
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        from zero_keras import losses

        val = ops.mean(losses.squared_hinge(y_true, y_pred))
        if sample_weight is not None:
            val = ops.multiply(val, ops.mean(_to_tensor(sample_weight)))
        self.total.assign(ops.add(_to_tensor(self.total), val))
        self.count.assign(ops.add(_to_tensor(self.count), _to_tensor(1.0)))

    def result(self) -> TensorLike:
        """Compute the current metric value.

        Returns:
            TensorLike: A scalar tensor, or a dictionary of scalar tensors.
        """
        return _wrap(
            ops.sqrt(
                ops.divide(
                    _to_tensor(self.total),
                    ops.maximum(_to_tensor(self.count), _to_tensor(1e-7)),
                )
            )
        )


class CategoricalHinge(Metric):
    """Computes the categorical hinge metric between `y_true` and `y_pred`.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:
    >>> m = keras.metrics.CategoricalHinge()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
    >>> m.result().numpy()
    1.4000001
    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
    ...                sample_weight=[1, 0])
    >>> m.result()
    1.2

    """

    def __init__(
        self, name: str = "categorical_hinge", dtype: DTypeLike = None, **kwargs: TensorLike
    ) -> None:
        """Initialize the CategoricalHinge metric.

        Args:
            name: String name of the metric instance.
            dtype: Data type of the metric result.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = self.add_weight(shape=(), initializer="zeros", name="total")
        self.count = self.add_weight(shape=(), initializer="zeros", name="count")

    def update_state(
        self, y_true: TensorLike, y_pred: TensorLike, sample_weight: TensorLike | None = None
    ) -> None:
        """Accumulate statistics for the metric.

        Args:
            y_true: Ground-truth target values.
            y_pred: Predicted values.
            sample_weight: Optional weighting for each example.
        """
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        from zero_keras import losses

        val = ops.mean(losses.categorical_hinge(y_true, y_pred))
        if sample_weight is not None:
            val = ops.multiply(val, ops.mean(_to_tensor(sample_weight)))
        self.total.assign(ops.add(_to_tensor(self.total), val))
        self.count.assign(ops.add(_to_tensor(self.count), _to_tensor(1.0)))

    def result(self) -> TensorLike:
        """Compute the current metric value.

        Returns:
            TensorLike: A scalar tensor, or a dictionary of scalar tensors.
        """
        return _wrap(
            ops.sqrt(
                ops.divide(
                    _to_tensor(self.total),
                    ops.maximum(_to_tensor(self.count), _to_tensor(1e-7)),
                )
            )
        )


class RootMeanSquaredError(Metric):
    """Computes root mean squared error metric between `y_true` and `y_pred`.

    Formula:

    ```python
    loss = sqrt(mean((y_pred - y_true) ** 2))
    ```

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Examples:
    >>> m = keras.metrics.RootMeanSquaredError()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result()
    0.5

    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
    ...                sample_weight=[1, 0])
    >>> m.result()
    0.70710677

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[keras.metrics.RootMeanSquaredError()])
    ```

    """

    def __init__(
        self,
        name: str = "root_mean_squared_error",
        dtype: DTypeLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the RootMeanSquaredError metric.

        Args:
            name: String name of the metric instance.
            dtype: Data type of the metric result.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = self.add_weight(shape=(), initializer="zeros", name="total")
        self.count = self.add_weight(shape=(), initializer="zeros", name="count")

    def update_state(
        self, y_true: TensorLike, y_pred: TensorLike, sample_weight: TensorLike | None = None
    ) -> None:
        """Accumulates root mean squared error statistics.

        Args:
            y_true: The ground truth values.
            y_pred: The predicted values.
            sample_weight: Optional weighting of each example. Can
                be a `Tensor` whose rank is either 0, or the same rank as
                `y_true`, and must be broadcastable to `y_true`.
                Defaults to `1`.
        """
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        from zero_keras import losses

        val = ops.mean(losses.mean_squared_error(y_true, y_pred))
        if sample_weight is not None:
            val = ops.multiply(val, ops.mean(_to_tensor(sample_weight)))
        self.total.assign(ops.add(_to_tensor(self.total), val))
        self.count.assign(ops.add(_to_tensor(self.count), _to_tensor(1.0)))

    def result(self) -> TensorLike:
        """Compute the current metric value.

        Returns:
            TensorLike: A scalar tensor, or a dictionary of scalar tensors.
        """
        return _wrap(
            ops.sqrt(
                ops.divide(
                    _to_tensor(self.total),
                    ops.maximum(_to_tensor(self.count), _to_tensor(1e-7)),
                )
            )
        )


class SparseCategoricalCrossentropy(MeanMetricWrapper):
    """Computes the crossentropy metric between the labels and predictions.

    Use this crossentropy metric when there are two or more label classes.
    It expects labels to be provided as integers. If you want to provide labels
    that are one-hot encoded, please use the `CategoricalCrossentropy`
    metric instead.

    There should be `num_classes` floating point values per feature for `y_pred`
    and a single floating point value per feature for `y_true`.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        from_logits: (Optional) Whether output is expected
            to be a logits tensor. By default, we consider that output
            encodes a probability distribution.
        axis: (Optional) Defaults to `-1`.
            The dimension along which entropy is computed.

    Examples:
    >>> # y_true = one_hot(y_true) = [[0, 1, 0], [0, 0, 1]]
    >>> # logits = log(y_pred)
    >>> # softmax = exp(logits) / sum(exp(logits), axis=-1)
    >>> # softmax = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]
    >>> # xent = -sum(y * log(softmax), 1)
    >>> # log(softmax) = [[-2.9957, -0.0513, -16.1181],
    >>> #                [-2.3026, -0.2231, -2.3026]]
    >>> # y_true * log(softmax) = [[0, -0.0513, 0], [0, 0, -2.3026]]
    >>> # xent = [0.0513, 2.3026]
    >>> # Reduced xent = (0.0513 + 2.3026) / 2
    >>> m = keras.metrics.SparseCategoricalCrossentropy()
    >>> m.update_state([1, 2],
    ...                [[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
    >>> m.result()
    1.1769392

    >>> m.reset_state()
    >>> m.update_state([1, 2],
    ...                [[0.05, 0.95, 0], [0.1, 0.8, 0.1]],
    ...                sample_weight=np.array([0.3, 0.7]))
    >>> m.result()
    1.6271976

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[keras.metrics.SparseCategoricalCrossentropy()])
    ```

    """

    def __init__(
        self,
        name: str = "sparse_categorical_crossentropy",
        dtype: DTypeLike = None,
        from_logits: bool = False,
        ignore_class: int | None = None,
        axis: int = -1,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the SparseCategoricalCrossentropy metric.

        Args:
            name: String name of the metric instance.
            dtype: Data type of the metric result.
            from_logits: Whether ``y_pred`` contains unnormalized logits.
            ignore_class: Optional class index to ignore during metric computation.
            axis: Channel axis along which entropy is computed.
            **kwargs: Additional keyword arguments.
        """
        _ = axis, ignore_class

        def scc_fn(y_true: TensorLike, y_pred: TensorLike) -> TensorLike:
            """Compute sparse categorical crossentropy between ``y_true`` and ``y_pred``.

            Args:
                y_true: Integer class indices.
                y_pred: Predicted class probabilities or logits.

            Returns:
                TensorLike: Per-sample sparse categorical crossentropy tensor.
            """
            return sparse_categorical_crossentropy(y_true, y_pred, from_logits=from_logits)

        super().__init__(fn=scc_fn, name=name, dtype=dtype, **kwargs)


class BinaryCrossentropy(MeanMetricWrapper):
    """Computes the crossentropy metric between the labels and predictions.

    This is the crossentropy metric class to be used when there are only two
    label classes (0 and 1).

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        from_logits: (Optional) Whether output is expected
            to be a logits tensor. By default, we consider
            that output encodes a probability distribution.
        label_smoothing: (Optional) Float in `[0, 1]`.
            When > 0, label values are smoothed,
            meaning the confidence on label values are relaxed.
            e.g. `label_smoothing=0.2` means that we will use
            a value of 0.1 for label "0" and 0.9 for label "1".

    Examples:
    >>> m = keras.metrics.BinaryCrossentropy()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
    >>> m.result()
    0.81492424

    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
    ...                sample_weight=[1, 0])
    >>> m.result()
    0.9162905

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[keras.metrics.BinaryCrossentropy()])
    ```

    """

    def __init__(
        self,
        name: str = "binary_crossentropy",
        dtype: DTypeLike = None,
        from_logits: bool = False,
        label_smoothing: float = 0,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the BinaryCrossentropy metric.

        Args:
            name: String name of the metric instance.
            dtype: Data type of the metric result.
            from_logits: Whether ``y_pred`` contains unnormalized logits.
            label_smoothing: Label smoothing factor in ``[0, 1]``.
            **kwargs: Additional keyword arguments.
        """

        def bce_fn(y_true: TensorLike, y_pred: TensorLike) -> TensorLike:
            """Compute binary crossentropy between ``y_true`` and ``y_pred``.

            Args:
                y_true: Ground-truth binary targets.
                y_pred: Predicted probabilities or logits.

            Returns:
                TensorLike: Per-sample binary crossentropy tensor.
            """
            return binary_crossentropy(
                y_true, y_pred, from_logits=from_logits, label_smoothing=label_smoothing
            )

        super().__init__(fn=bce_fn, name=name, dtype=dtype, **kwargs)


class KLDivergence(Metric):
    """Compute Kullback-Leibler divergence metric between `y_true` and `y_pred`.

    Formula:

    ```python
    metric = y_true * log(y_true / y_pred)
    ```

    `y_true` and `y_pred` are expected to be probability
    distributions, with values between 0 and 1. They will get
    clipped to the `[0, 1]` range.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Examples:
    >>> m = keras.metrics.KLDivergence()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]])
    >>> m.result()
    0.45814306

    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[0.6, 0.4], [0.4, 0.6]],
    ...                sample_weight=[1, 0])
    >>> m.result()
    0.9162892

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[keras.metrics.KLDivergence()])
    ```

    """

    def __init__(
        self, name: str = "kl_divergence", dtype: DTypeLike = None, **kwargs: TensorLike
    ) -> None:
        """Initialize the KLDivergence metric.

        Args:
            name: String name of the metric instance.
            dtype: Data type of the metric result.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = self.add_weight(shape=(), initializer="zeros", name="total")
        self.count = self.add_weight(shape=(), initializer="zeros", name="count")

    def update_state(
        self, y_true: TensorLike, y_pred: TensorLike, sample_weight: TensorLike | None = None
    ) -> None:
        """Accumulate statistics for the metric.

        Args:
            y_true: Ground-truth probability distributions.
            y_pred: Predicted probability distributions.
            sample_weight: Optional weighting for each example.
        """
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        from zero_keras import losses

        val = ops.mean(losses.kl_divergence(y_true, y_pred))
        if sample_weight is not None:
            val = ops.multiply(val, ops.mean(_to_tensor(sample_weight)))
        self.total.assign(ops.add(_to_tensor(self.total), val))
        self.count.assign(ops.add(_to_tensor(self.count), _to_tensor(1.0)))

    def result(self) -> TensorLike:
        """Compute the current metric value.

        Returns:
            TensorLike: A scalar tensor, or a dictionary of scalar tensors.
        """
        return _wrap(
            ops.sqrt(
                ops.divide(
                    _to_tensor(self.total),
                    ops.maximum(_to_tensor(self.count), _to_tensor(1e-7)),
                )
            )
        )


class Poisson(Metric):
    """Computes the Poisson metric between `y_true` and `y_pred`.

    Formula:

    ```python
    metric = y_pred - y_true * log(y_pred)
    ```

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Examples:
    >>> m = keras.metrics.Poisson()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result()
    0.49999997

    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
    ...                sample_weight=[1, 0])
    >>> m.result()
    0.99999994

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[keras.metrics.Poisson()])
    ```

    """

    def __init__(
        self, name: str = "poisson", dtype: DTypeLike = None, **kwargs: TensorLike
    ) -> None:
        """Initialize the Poisson metric.

        Args:
            name: String name of the metric instance.
            dtype: Data type of the metric result.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = self.add_weight(shape=(), initializer="zeros", name="total")
        self.count = self.add_weight(shape=(), initializer="zeros", name="count")

    def update_state(
        self, y_true: TensorLike, y_pred: TensorLike, sample_weight: TensorLike | None = None
    ) -> None:
        """Accumulate statistics for the metric.

        Args:
            y_true: Ground-truth target values.
            y_pred: Predicted values.
            sample_weight: Optional weighting for each example.
        """
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        from zero_keras import losses

        val = ops.mean(losses.poisson(y_true, y_pred))
        if sample_weight is not None:
            val = ops.multiply(val, ops.mean(_to_tensor(sample_weight)))
        self.total.assign(ops.add(_to_tensor(self.total), val))
        self.count.assign(ops.add(_to_tensor(self.count), _to_tensor(1.0)))

    def result(self) -> TensorLike:
        """Compute the current metric value.

        Returns:
            TensorLike: A scalar tensor, or a dictionary of scalar tensors.
        """
        return _wrap(
            ops.sqrt(
                ops.divide(
                    _to_tensor(self.total),
                    ops.maximum(_to_tensor(self.count), _to_tensor(1e-7)),
                )
            )
        )


class LogCoshError(MeanMetricWrapper):
    """Computes the logarithm of the hyperbolic cosine of the prediction error.

    Formula:

    ```python
    error = y_pred - y_true
    logcosh = mean(log((exp(error) + exp(-error))/2), axis=-1)
    ```

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Examples:
    >>> m = keras.metrics.LogCoshError()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result()
    0.10844523

    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
    ...                sample_weight=[1, 0])
    >>> m.result()
    0.21689045

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[keras.metrics.LogCoshError()])
    ```

    """

    def __init__(
        self, name: str = "logcosh", dtype: DTypeLike = None, **kwargs: TensorLike
    ) -> None:
        """Initialize the LogCoshError metric.

        Args:
            name: String name of the metric instance.
            dtype: Data type of the metric result.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(fn=log_cosh, name=name, dtype=dtype, **kwargs)


class IoU(Metric):
    """Computes the Intersection-Over-Union metric for specific target classes.

    Formula:

    ```python
    iou = true_positives / (true_positives + false_positives + false_negatives)
    ```
    Intersection-Over-Union is a common evaluation metric for semantic image
    segmentation.

    To compute IoUs, the predictions are accumulated in a confusion matrix,
    weighted by `sample_weight` and the metric is then calculated from it.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Note, this class first computes IoUs for all individual classes, then
    returns the mean of IoUs for the classes that are specified by
    `target_class_ids`. If `target_class_ids` has only one id value, the IoU of
    that specific class is returned.

    Args:
        num_classes: The possible number of labels the prediction task can have.
        target_class_ids: A tuple or list of target class ids for which the
            metric is returned. To compute IoU for a specific class, a list
            (or tuple) of a single id value should be provided.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        ignore_class: Optional integer. The ID of a class to be ignored during
            metric computation. This is useful, for example, in segmentation
            problems featuring a "void" class (commonly -1 or 255) in
            segmentation maps. By default (`ignore_class=None`), all classes are
              considered.
        sparse_y_true: Whether labels are encoded using integers or
            dense floating point vectors. If `False`, the `argmax` function
            is used to determine each sample's most likely associated label.
        sparse_y_pred: Whether predictions are encoded using integers or
            dense floating point vectors. If `False`, the `argmax` function
            is used to determine each sample's most likely associated label.
        axis: (Optional) -1 is the dimension containing the logits.
            Defaults to `-1`.

    Examples:
    >>> # cm = [[1, 1],
    >>> #        [1, 1]]
    >>> # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
    >>> # iou = true_positives / (sum_row + sum_col - true_positives))
    >>> # iou = [0.33, 0.33]
    >>> m = keras.metrics.IoU(num_classes=2, target_class_ids=[0])
    >>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1])
    >>> m.result()
    0.33333334

    >>> m.reset_state()
    >>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1],
    ...                sample_weight=[0.3, 0.3, 0.3, 0.1])
    >>> # cm = [[0.3, 0.3],
    >>> #        [0.3, 0.1]]
    >>> # sum_row = [0.6, 0.4], sum_col = [0.6, 0.4],
    >>> # true_positives = [0.3, 0.1]
    >>> # iou = [0.33, 0.14]
    >>> m.result()
    0.33333334

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[keras.metrics.IoU(num_classes=2, target_class_ids=[0])])
    ```

    """

    def __init__(
        self,
        num_classes: int,
        target_class_ids: Sequence[int] | None,
        name: str | None = None,
        dtype: DTypeLike = None,
        ignore_class: int | None = None,
        sparse_y_true: bool = True,
        sparse_y_pred: bool = True,
        axis: int = -1,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the IoU metric.

        Args:
            num_classes: Total number of segmentation classes.
            target_class_ids: Sequence of class indices to evaluate IoU over.
            name: Optional string name of the metric instance.
            dtype: Optional data type of the metric result.
            ignore_class: Optional class index to ignore during evaluation.
            sparse_y_true: Whether ``y_true`` contains integer class labels.
            sparse_y_pred: Whether ``y_pred`` contains integer class labels.
            axis: Channel axis for dense predictions.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.num_classes = max(1, int(num_classes))
        self.target_class_ids = target_class_ids
        self.ignore_class = ignore_class
        self.sparse_y_true = sparse_y_true
        self.sparse_y_pred = sparse_y_pred
        self.axis = axis
        self.total_cm = self.add_weight(
            shape=(self.num_classes, self.num_classes),
            initializer="zeros",
            dtype="float32",
            name="total_cm",
        )

    def update_state(
        self, y_true: TensorLike, y_pred: TensorLike, sample_weight: TensorLike | None = None
    ) -> None:
        """Accumulate confusion matrix entries from ``y_true`` and ``y_pred``.

        Args:
            y_true: Ground-truth segmentation labels or one-hot probabilities.
            y_pred: Predicted segmentation labels or probabilities.
            sample_weight: Optional per-pixel or per-sample weights.
        """
        from zero_keras import ops
        from zero_keras.activations import _to_tensor

        y_true = ops.cast(_to_tensor(y_true), "int32")
        y_pred = ops.cast(_to_tensor(y_pred), "int32")

        if not self.sparse_y_true:
            y_true = ops.argmax(y_true, axis=self.axis)
        if not self.sparse_y_pred:
            y_pred = ops.argmax(y_pred, axis=self.axis)

        y_true = ops.reshape(y_true, [-1])
        y_pred = ops.reshape(y_pred, [-1])

        if self.ignore_class is not None:
            valid_mask = ops.logical_not(ops.equal(y_true, self.ignore_class))
        else:
            valid_mask = ops.ones_like(y_true)

        valid_mask = ops.cast(valid_mask, "float32")
        if sample_weight is not None:
            sample_weight = ops.reshape(ops.cast(_to_tensor(sample_weight), "float32"), [-1])
            valid_mask = valid_mask * sample_weight

        y_true_one_hot = ops.one_hot(y_true, self.num_classes)
        y_pred_one_hot = ops.one_hot(y_pred, self.num_classes)

        y_true_one_hot = y_true_one_hot * ops.expand_dims(valid_mask, -1)

        cm = ops.sum(
            ops.expand_dims(y_true_one_hot, -1) * ops.expand_dims(y_pred_one_hot, 1),
            axis=0,
        )

        self.total_cm.assign(_to_tensor(self.total_cm) + cm)

    def result(self) -> TensorLike:
        """Compute the mean Intersection-Over-Union across target classes.

        Returns:
            TensorLike: Scalar IoU tensor.
        """
        from zero_keras import ops

        cm = _to_tensor(self.total_cm)
        sum_over_row = ops.sum(cm, axis=0)
        sum_over_col = ops.sum(cm, axis=1)

        arange = ops.arange(self.num_classes)
        identity = ops.cast(
            ops.equal(ops.expand_dims(arange, 1), ops.expand_dims(arange, 0)), "float32"
        )
        true_positives = ops.sum(cm * identity, axis=1)

        denominator = sum_over_row + sum_over_col - true_positives

        iou = ops.where(
            ops.greater(denominator, 0.0), true_positives / denominator, _to_tensor(0.0)
        )

        if isinstance(self.target_class_ids, (list, tuple)):
            mask = ops.sum(
                ops.one_hot(ops.asarray(self.target_class_ids), self.num_classes),
                axis=0,
            )
            iou = iou * mask
            num_valid = ops.cast(len(self.target_class_ids), "float32")
            return ops.sum(iou) / num_valid

        return ops.mean(iou)

    def reset_state(self) -> None:
        """Reset the confusion matrix state to zeros."""
        self.total_cm.assign(ops.zeros((self.num_classes, self.num_classes), dtype="float32"))


class MeanIoU(IoU):
    """Computes the mean Intersection-Over-Union metric.

    Formula:

    ```python
    iou = true_positives / (true_positives + false_positives + false_negatives)
    ```
    Intersection-Over-Union is a common evaluation metric for semantic image
    segmentation.

    To compute IoUs, the predictions are accumulated in a confusion matrix,
    weighted by `sample_weight` and the metric is then calculated from it.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Note that this class first computes IoUs for all individual classes, then
    returns the mean of these values.

    Args:
        num_classes: The possible number of labels the prediction task can have.
            This value must be provided, since a confusion matrix of dimension =
            [num_classes, num_classes] will be allocated.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        ignore_class: Optional integer. The ID of a class to be ignored during
            metric computation. This is useful, for example, in segmentation
            problems featuring a "void" class (commonly -1 or 255) in
            segmentation maps. By default (`ignore_class=None`), all classes are
            considered.
        sparse_y_true: Whether labels are encoded using integers or
            dense floating point vectors. If `False`, the `argmax` function
            is used to determine each sample's most likely associated label.
        sparse_y_pred: Whether predictions are encoded using integers or
            dense floating point vectors. If `False`, the `argmax` function
            is used to determine each sample's most likely associated label.
        axis: (Optional) The dimension containing the logits. Defaults to `-1`.


    Example:
    >>> # cm = [[1, 1],
    >>> #        [1, 1]]
    >>> # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
    >>> # iou = true_positives / (sum_row + sum_col - true_positives))
    >>> # result = (1 / (2 + 2 - 1) + 1 / (2 + 2 - 1)) / 2 = 0.33
    >>> m = keras.metrics.MeanIoU(num_classes=2)
    >>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1])
    >>> m.result()
    0.33333334

    >>> m.reset_state()
    >>> m.update_state([0, 0, 1, 1], [0, 1, 0, 1],
    ...                sample_weight=[0.3, 0.3, 0.3, 0.1])
    >>> m.result().numpy()
    0.23809525

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[keras.metrics.MeanIoU(num_classes=2)])
    ```

    """

    def __init__(
        self,
        num_classes: int,
        name: str | None = None,
        dtype: DTypeLike = None,
        ignore_class: int | None = None,
        sparse_y_true: bool = True,
        sparse_y_pred: bool = True,
        axis: int = -1,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the MeanIoU metric.

        Args:
            num_classes: Total number of segmentation classes.
            name: Optional string name of the metric instance.
            dtype: Optional data type of the metric result.
            ignore_class: Optional class index to ignore during evaluation.
            sparse_y_true: Whether ``y_true`` contains integer class labels.
            sparse_y_pred: Whether ``y_pred`` contains integer class labels.
            axis: Channel axis for dense predictions.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            num_classes,
            target_class_ids=None,
            name=name,
            dtype=dtype,
            ignore_class=ignore_class,
            sparse_y_true=sparse_y_true,
            sparse_y_pred=sparse_y_pred,
            axis=axis,
            **kwargs,
        )


class BinaryIoU(IoU):
    """Computes the Intersection-Over-Union metric for class 0 and/or 1.

    Formula:

    ```python
    iou = true_positives / (true_positives + false_positives + false_negatives)
    ```
    Intersection-Over-Union is a common evaluation metric for semantic image
    segmentation.

    To compute IoUs, the predictions are accumulated in a confusion matrix,
    weighted by `sample_weight` and the metric is then calculated from it.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    This class can be used to compute IoUs for a binary classification task
    where the predictions are provided as logits. First a `threshold` is applied
    to the predicted values such that those that are below the `threshold` are
    converted to class 0 and those that are above the `threshold` are converted
    to class 1.

    IoUs for classes 0 and 1 are then computed, the mean of IoUs for the classes
    that are specified by `target_class_ids` is returned.

    Note: with `threshold=0`, this metric has the same behavior as `IoU`.

    Args:
        target_class_ids: A tuple or list of target class ids for which the
            metric is returned. Options are `[0]`, `[1]`, or `[0, 1]`. With
            `[0]` (or `[1]`), the IoU metric for class 0 (or class 1,
            respectively) is returned. With `[0, 1]`, the mean of IoUs for the
            two classes is returned.
        threshold: A threshold that applies to the prediction logits to convert
            them to either predicted class 0 if the logit is below `threshold`
            or predicted class 1 if the logit is above `threshold`.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:
    >>> m = keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.3)
    >>> m.update_state([0, 1, 0, 1], [0.1, 0.2, 0.4, 0.7])
    >>> m.result()
    0.33333334

    >>> m.reset_state()
    >>> m.update_state([0, 1, 0, 1], [0.1, 0.2, 0.4, 0.7],
    ...                sample_weight=[0.2, 0.3, 0.4, 0.1])
    >>> # cm = [[0.2, 0.4],
    >>> #        [0.3, 0.1]]
    >>> # sum_row = [0.6, 0.4], sum_col = [0.5, 0.5],
    >>> # true_positives = [0.2, 0.1]
    >>> # iou = [0.222, 0.125]
    >>> m.result()
    0.17361112

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[keras.metrics.BinaryIoU(
            target_class_ids=[0],
            threshold=0.5
        )]
    )
    ```

    """

    def __init__(
        self,
        target_class_ids: Sequence[int] = (0, 1),
        threshold: float = 0.5,
        name: str | None = None,
        dtype: DTypeLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the BinaryIoU metric.

        Args:
            target_class_ids: Sequence of binary class indices (``0`` and/or ``1``).
            threshold: Threshold for binarizing predicted scores.
            name: Optional string name of the metric instance.
            dtype: Optional data type of the metric result.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            num_classes=2,
            target_class_ids=target_class_ids,
            name=name,
            dtype=dtype,
            **kwargs,
        )
        self.threshold = threshold

    def update_state(
        self, y_true: TensorLike, y_pred: TensorLike, sample_weight: TensorLike | None = None
    ) -> None:
        """Threshold ``y_pred`` and accumulate binary confusion matrix counts.

        Args:
            y_true: Ground-truth binary targets.
            y_pred: Predicted scores or probabilities.
            sample_weight: Optional sample weights.
        """
        from zero_keras import ops
        from zero_keras.activations import _to_tensor

        y_true = ops.cast(_to_tensor(y_true), "float32")
        y_pred = ops.cast(_to_tensor(y_pred), "float32")
        y_pred = ops.cast(ops.greater(y_pred, self.threshold), "int32")
        super().update_state(y_true, y_pred, sample_weight)


class OneHotMeanIoU(MeanIoU):
    """Computes mean Intersection-Over-Union metric for one-hot encoded labels.

    Formula:

    ```python
    iou = true_positives / (true_positives + false_positives + false_negatives)
    ```
    Intersection-Over-Union is a common evaluation metric for semantic image
    segmentation.

    To compute IoUs, the predictions are accumulated in a confusion matrix,
    weighted by `sample_weight` and the metric is then calculated from it.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    This class can be used to compute the mean IoU for multi-class
    classification tasks where the labels are one-hot encoded (the last axis
    should have one dimension per class). Note that the predictions should also
    have the same shape. To compute the mean IoU, first the labels and
    predictions are converted back into integer format by taking the argmax over
    the class axis. Then the same computation steps as for the base `MeanIoU`
    class apply.

    Note, if there is only one channel in the labels and predictions, this class
    is the same as class `MeanIoU`. In this case, use `MeanIoU` instead.

    Also, make sure that `num_classes` is equal to the number of classes in the
    data, to avoid a "labels out of bound" error when the confusion matrix is
    computed.

    Args:
        num_classes: The possible number of labels the prediction task can have.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        ignore_class: Optional integer. The ID of a class to be ignored during
            metric computation. This is useful, for example, in segmentation
            problems featuring a "void" class (commonly -1 or 255) in
            segmentation maps. By default (`ignore_class=None`), all classes are
            considered.
        sparse_y_pred: Whether predictions are encoded using natural numbers or
            probability distribution vectors. If `False`, the `argmax`
            function will be used to determine each sample's most likely
            associated label.
        axis: (Optional) The dimension containing the logits. Defaults to `-1`.


    Example:
    >>> y_true = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> y_pred = np.array([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7], [0.5, 0.3, 0.1],
    ...                       [0.1, 0.4, 0.5]])
    >>> sample_weight = [0.1, 0.2, 0.3, 0.4]
    >>> m = keras.metrics.OneHotMeanIoU(num_classes=3)
    >>> m.update_state(
    ...     y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    >>> # cm = [[0, 0, 0.2+0.4],
    >>> #       [0.3, 0, 0],
    >>> #       [0, 0, 0.1]]
    >>> # sum_row = [0.3, 0, 0.7], sum_col = [0.6, 0.3, 0.1]
    >>> # true_positives = [0, 0, 0.1]
    >>> # single_iou = true_positives / (sum_row + sum_col - true_positives))
    >>> # mean_iou = (0 + 0 + 0.1 / (0.7 + 0.1 - 0.1)) / 3
    >>> m.result()
    0.048

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[keras.metrics.OneHotMeanIoU(num_classes=3)])
    ```

    """

    def __init__(
        self,
        num_classes: int,
        name: str | None = None,
        dtype: DTypeLike = None,
        ignore_class: int | None = None,
        sparse_y_pred: bool = False,
        axis: int = -1,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the OneHotMeanIoU metric.

        Args:
            num_classes: Total number of segmentation classes.
            name: Optional string name of the metric instance.
            dtype: Optional data type of the metric result.
            ignore_class: Optional class index to ignore during evaluation.
            sparse_y_pred: Whether ``y_pred`` contains integer class labels.
            axis: Channel axis for one-hot labels and dense predictions.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            num_classes,
            name=name,
            dtype=dtype,
            ignore_class=ignore_class,
            sparse_y_true=False,
            sparse_y_pred=sparse_y_pred,
            axis=axis,
            **kwargs,
        )


class OneHotIoU(IoU):
    """Computes the Intersection-Over-Union metric for one-hot encoded labels.

    Formula:

    ```python
    iou = true_positives / (true_positives + false_positives + false_negatives)
    ```
    Intersection-Over-Union is a common evaluation metric for semantic image
    segmentation.

    To compute IoUs, the predictions are accumulated in a confusion matrix,
    weighted by `sample_weight` and the metric is then calculated from it.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    This class can be used to compute IoU for multi-class classification tasks
    where the labels are one-hot encoded (the last axis should have one
    dimension per class). Note that the predictions should also have the same
    shape. To compute the IoU, first the labels and predictions are converted
    back into integer format by taking the argmax over the class axis. Then the
    same computation steps as for the base `IoU` class apply.

    Note, if there is only one channel in the labels and predictions, this class
    is the same as class `IoU`. In this case, use `IoU` instead.

    Also, make sure that `num_classes` is equal to the number of classes in the
    data, to avoid a "labels out of bound" error when the confusion matrix is
    computed.

    Args:
        num_classes: The possible number of labels the prediction task can have.
        target_class_ids: A tuple or list of target class ids for which the
            metric is returned. To compute IoU for a specific class, a list
            (or tuple) of a single id value should be provided.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        ignore_class: Optional integer. The ID of a class to be ignored during
            metric computation. This is useful, for example, in segmentation
            problems featuring a "void" class (commonly -1 or 255) in
            segmentation maps. By default (`ignore_class=None`), all classes are
            considered.
        sparse_y_pred: Whether predictions are encoded using integers or
            dense floating point vectors. If `False`, the `argmax` function
            is used to determine each sample's most likely associated label.
        axis: (Optional) The dimension containing the logits. Defaults to `-1`.


    Example:
    >>> y_true = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> y_pred = np.array([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7], [0.5, 0.3, 0.1],
    ...                       [0.1, 0.4, 0.5]])
    >>> sample_weight = [0.1, 0.2, 0.3, 0.4]
    >>> m = keras.metrics.OneHotIoU(num_classes=3, target_class_ids=[0, 2])
    >>> m.update_state(
    ...     y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    >>> # cm = [[0, 0, 0.2+0.4],
    >>> #       [0.3, 0, 0],
    >>> #       [0, 0, 0.1]]
    >>> # sum_row = [0.3, 0, 0.7], sum_col = [0.6, 0.3, 0.1]
    >>> # true_positives = [0, 0, 0.1]
    >>> # single_iou = true_positives / (sum_row + sum_col - true_positives))
    >>> # mean_iou = (0 / (0.3 + 0.6 - 0) + 0.1 / (0.7 + 0.1 - 0.1)) / 2
    >>> m.result()
    0.071

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[keras.metrics.OneHotIoU(
            num_classes=3,
            target_class_id=[1]
        )]
    )
    ```

    """

    def __init__(
        self,
        num_classes: int,
        target_class_ids: Sequence[int] | None,
        name: str | None = None,
        dtype: DTypeLike = None,
        ignore_class: int | None = None,
        sparse_y_pred: bool = False,
        axis: int = -1,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the OneHotIoU metric.

        Args:
            num_classes: Total number of segmentation classes.
            target_class_ids: Sequence of class indices to evaluate IoU over.
            name: Optional string name of the metric instance.
            dtype: Optional data type of the metric result.
            ignore_class: Optional class index to ignore during evaluation.
            sparse_y_pred: Whether ``y_pred`` contains integer class labels.
            axis: Channel axis for one-hot labels and dense predictions.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            num_classes,
            target_class_ids,
            name=name,
            dtype=dtype,
            ignore_class=ignore_class,
            sparse_y_true=False,
            sparse_y_pred=sparse_y_pred,
            axis=axis,
            **kwargs,
        )


class ConcordanceCorrelation(Metric):
    """Computes Lin's Concordance Correlation Coefficient between targets and predictions."""

    def __init__(
        self,
        name: str = "concordance_correlation",
        dtype: DTypeLike = None,
        axis: int = -1,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the ConcordanceCorrelation metric.

        Args:
            name: String name of the metric instance.
            dtype: Data type of the metric result.
            axis: Dimension along which correlation is computed.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.axis = axis
        self.sum_x = self.add_weight(shape=(), initializer="zeros", name="sum_x")
        self.sum_y = self.add_weight(shape=(), initializer="zeros", name="sum_y")
        self.sum_x2 = self.add_weight(shape=(), initializer="zeros", name="sum_x2")
        self.sum_y2 = self.add_weight(shape=(), initializer="zeros", name="sum_y2")
        self.sum_xy = self.add_weight(shape=(), initializer="zeros", name="sum_xy")
        self.count = self.add_weight(shape=(), initializer="zeros", name="count")

    def update_state(
        self, y_true: TensorLike, y_pred: TensorLike, sample_weight: TensorLike | None = None
    ) -> None:
        """Accumulate first and second moment statistics for concordance correlation.

        Args:
            y_true: Ground-truth target values.
            y_pred: Predicted values.
            sample_weight: Optional sample weights.
        """
        from zero_keras import ops
        from zero_keras.activations import _to_tensor

        y_true = _to_tensor(y_true)
        y_pred = _to_tensor(y_pred)
        w = _to_tensor(sample_weight) if sample_weight is not None else _to_tensor(1.0)

        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32")
        w = ops.cast(w, "float32")

        sx = ops.sum(y_true * w, axis=0)
        sy = ops.sum(y_pred * w, axis=0)
        sx2 = ops.sum((y_true ** _to_tensor(2.0)) * w, axis=0)
        sy2 = ops.sum((y_pred ** _to_tensor(2.0)) * w, axis=0)
        sxy = ops.sum(y_true * y_pred * w, axis=0)
        c = ops.sum(w, axis=0)

        self.sum_x.assign(ops.add(_to_tensor(self.sum_x), sx))
        self.sum_y.assign(ops.add(_to_tensor(self.sum_y), sy))
        self.sum_x2.assign(ops.add(_to_tensor(self.sum_x2), sx2))
        self.sum_y2.assign(ops.add(_to_tensor(self.sum_y2), sy2))
        self.sum_xy.assign(ops.add(_to_tensor(self.sum_xy), sxy))
        self.count.assign(ops.add(_to_tensor(self.count), c))

    def result(self) -> TensorLike:
        """Compute the concordance correlation coefficient.

        Returns:
            TensorLike: Scalar concordance correlation coefficient tensor.
        """
        from zero_keras import ops
        from zero_keras.activations import _to_tensor, _wrap

        n = ops.maximum(_to_tensor(self.count), _to_tensor(1e-7))
        mean_x = _to_tensor(self.sum_x) / n
        mean_y = _to_tensor(self.sum_y) / n

        var_x = (_to_tensor(self.sum_x2) / n) - (mean_x ** _to_tensor(2.0))
        var_y = (_to_tensor(self.sum_y2) / n) - (mean_y ** _to_tensor(2.0))
        cov_xy = (_to_tensor(self.sum_xy) / n) - (mean_x * mean_y)

        ccc = (ops.array(2.0, dtype="float32") * cov_xy) / ops.maximum(
            var_x + var_y + ((mean_x - mean_y) ** _to_tensor(2.0)), _to_tensor(1e-7)
        )
        return _wrap(ops.mean(ccc))

    def reset_state(self) -> None:
        """Reset all moment accumulator variables to zero."""
        from zero_keras.activations import _to_tensor

        self.sum_x.assign(_to_tensor(0.0))
        self.sum_y.assign(_to_tensor(0.0))
        self.sum_x2.assign(_to_tensor(0.0))
        self.sum_y2.assign(_to_tensor(0.0))
        self.sum_xy.assign(_to_tensor(0.0))
        self.count.assign(_to_tensor(0.0))


class PearsonCorrelation(Metric):
    """Computes Pearson's linear correlation coefficient between targets and predictions."""

    def __init__(
        self,
        name: str = "pearson_correlation",
        dtype: DTypeLike = None,
        axis: int = -1,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the PearsonCorrelation metric.

        Args:
            name: String name of the metric instance.
            dtype: Data type of the metric result.
            axis: Dimension along which correlation is computed.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.axis = axis
        self.sum_x = self.add_weight(shape=(), initializer="zeros", name="sum_x")
        self.sum_y = self.add_weight(shape=(), initializer="zeros", name="sum_y")
        self.sum_x2 = self.add_weight(shape=(), initializer="zeros", name="sum_x2")
        self.sum_y2 = self.add_weight(shape=(), initializer="zeros", name="sum_y2")
        self.sum_xy = self.add_weight(shape=(), initializer="zeros", name="sum_xy")
        self.count = self.add_weight(shape=(), initializer="zeros", name="count")

    def update_state(
        self, y_true: TensorLike, y_pred: TensorLike, sample_weight: TensorLike | None = None
    ) -> None:
        """Accumulate first and second moment statistics for Pearson correlation.

        Args:
            y_true: Ground-truth target values.
            y_pred: Predicted values.
            sample_weight: Optional sample weights.
        """
        from zero_keras import ops
        from zero_keras.activations import _to_tensor

        y_true = _to_tensor(y_true)
        y_pred = _to_tensor(y_pred)
        w = _to_tensor(sample_weight) if sample_weight is not None else _to_tensor(1.0)

        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32")
        w = ops.cast(w, "float32")

        sx = ops.sum(y_true * w, axis=0)
        sy = ops.sum(y_pred * w, axis=0)
        sx2 = ops.sum((y_true ** _to_tensor(2.0)) * w, axis=0)
        sy2 = ops.sum((y_pred ** _to_tensor(2.0)) * w, axis=0)
        sxy = ops.sum(y_true * y_pred * w, axis=0)
        c = ops.sum(w, axis=0)

        self.sum_x.assign(ops.add(_to_tensor(self.sum_x), sx))
        self.sum_y.assign(ops.add(_to_tensor(self.sum_y), sy))
        self.sum_x2.assign(ops.add(_to_tensor(self.sum_x2), sx2))
        self.sum_y2.assign(ops.add(_to_tensor(self.sum_y2), sy2))
        self.sum_xy.assign(ops.add(_to_tensor(self.sum_xy), sxy))
        self.count.assign(ops.add(_to_tensor(self.count), c))

    def result(self) -> TensorLike:
        """Compute Pearson's correlation coefficient.

        Returns:
            TensorLike: Scalar correlation coefficient tensor.
        """
        from zero_keras import ops
        from zero_keras.activations import _to_tensor, _wrap

        n = ops.maximum(_to_tensor(self.count), _to_tensor(1e-7))
        mean_x = _to_tensor(self.sum_x) / n
        mean_y = _to_tensor(self.sum_y) / n

        var_x = (_to_tensor(self.sum_x2) / n) - (mean_x ** _to_tensor(2.0))
        var_y = (_to_tensor(self.sum_y2) / n) - (mean_y ** _to_tensor(2.0))
        cov_xy = (_to_tensor(self.sum_xy) / n) - (mean_x * mean_y)

        pcc = cov_xy / ops.maximum(ops.sqrt(var_x * var_y), _to_tensor(1e-7))
        return _wrap(ops.mean(pcc))

    def reset_state(self) -> None:
        """Reset all moment accumulator variables to zero."""
        from zero_keras.activations import _to_tensor

        self.sum_x.assign(_to_tensor(0.0))
        self.sum_y.assign(_to_tensor(0.0))
        self.sum_x2.assign(_to_tensor(0.0))
        self.sum_y2.assign(_to_tensor(0.0))
        self.sum_xy.assign(_to_tensor(0.0))
        self.count.assign(_to_tensor(0.0))


class FBetaScore(Metric):
    """Computes the F-Beta score (weighted harmonic mean of precision and recall)."""

    def __init__(
        self,
        average: str | None = None,
        beta: float = 1.0,
        threshold: float | None = None,
        name: str = "fbeta_score",
        dtype: DTypeLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the FBetaScore metric.

        Args:
            average: Reduction mode across classes (``None``, ``"micro"``, ``"macro"``, or ``"weighted"``).
            beta: Weighting of recall relative to precision.
            threshold: Optional threshold for binarizing predictions.
            name: String name of the metric instance.
            dtype: Data type of the metric result.
            **kwargs: Additional keyword arguments.
        """
        if average not in (None, "micro", "macro", "weighted"):
            raise ValueError(f"Invalid average: {average}")
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.average = average
        self.beta = beta
        self.threshold = threshold
        self.tp = self.add_weight(shape=(), initializer="zeros", name="tp")
        self.fp = self.add_weight(shape=(), initializer="zeros", name="fp")
        self.fn = self.add_weight(shape=(), initializer="zeros", name="fn")

    def update_state(
        self, y_true: TensorLike, y_pred: TensorLike, sample_weight: TensorLike | None = None
    ) -> None:
        """Accumulate true positive, false positive, and false negative counts.

        Args:
            y_true: Ground-truth binary or multi-class targets.
            y_pred: Predicted probabilities or scores.
            sample_weight: Optional sample weights.
        """
        from zero_keras import ops
        from zero_keras.activations import _to_tensor

        y_true = _to_tensor(y_true)
        y_pred = _to_tensor(y_pred)
        if sample_weight is not None:
            sample_weight = _to_tensor(sample_weight)

        if self.threshold is not None:
            y_pred = ops.cast(ops.greater(y_pred, _to_tensor(self.threshold)), y_true.dtype)
        else:
            y_pred = ops.cast(
                ops.equal(y_pred, ops.max(y_pred, axis=-1, keepdims=True)), y_true.dtype
            )

        w = sample_weight if sample_weight is not None else _to_tensor(1.0)

        tp = ops.sum(y_true * y_pred * w, axis=0)
        fp = ops.sum((_to_tensor(1.0) - y_true) * y_pred * w, axis=0)
        fn = ops.sum(y_true * (_to_tensor(1.0) - y_pred) * w, axis=0)

        if self.average == "micro":
            tp = ops.sum(tp)
            fp = ops.sum(fp)
            fn = ops.sum(fn)

        self.tp.assign(ops.add(_to_tensor(self.tp), tp))
        self.fp.assign(ops.add(_to_tensor(self.fp), fp))
        self.fn.assign(ops.add(_to_tensor(self.fn), fn))

    def result(self) -> TensorLike:
        """Compute the F-Beta score.

        Returns:
            TensorLike: Scalar or per-class F-Beta score tensor.
        """
        from zero_keras import ops
        from zero_keras.activations import _to_tensor, _wrap

        beta2 = _to_tensor(self.beta**2)
        tp_t = _to_tensor(self.tp)
        fp_t = _to_tensor(self.fp)
        fn_t = _to_tensor(self.fn)
        p = tp_t / ops.maximum(tp_t + fp_t, _to_tensor(1e-7))
        r = tp_t / ops.maximum(tp_t + fn_t, _to_tensor(1e-7))
        f_beta = (
            (ops.add(_to_tensor(1.0), beta2)) * p * r / ops.maximum(beta2 * p + r, _to_tensor(1e-7))
        )

        if self.average == "macro":
            return _wrap(ops.mean(f_beta))
        return _wrap(f_beta)

    def reset_state(self) -> None:
        """Reset ``tp``, ``fp``, and ``fn`` accumulators to zero."""
        from zero_keras.activations import _to_tensor

        self.tp.assign(_to_tensor(0.0))
        self.fp.assign(_to_tensor(0.0))
        self.fn.assign(_to_tensor(0.0))


class F1Score(FBetaScore):
    """Computes the F-1 score (harmonic mean of precision and recall)."""

    def __init__(
        self,
        average: str | None = None,
        threshold: float | None = None,
        name: str = "f1_score",
        dtype: DTypeLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the F1Score metric.

        Args:
            average: Reduction mode across classes (``None``, ``"micro"``, ``"macro"``, or ``"weighted"``).
            threshold: Optional threshold for binarizing predictions.
            name: String name of the metric instance.
            dtype: Data type of the metric result.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            average=average,
            beta=1.0,
            threshold=threshold,
            name=name,
            dtype=dtype,
            **kwargs,
        )


class R2Score(Metric):
    """Computes the R^2 (coefficient of determination) regression score."""

    def __init__(
        self,
        class_aggregation: str | None = "uniform_average",
        num_regressors: int = 0,
        name: str = "r2_score",
        dtype: DTypeLike = None,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize the R2Score metric.

        Args:
            class_aggregation: Aggregation mode across output dimensions (``"uniform_average"``, ``"variance_weighted_average"``, or ``None``).
            num_regressors: Number of independent regressors for adjusted R^2.
            name: String name of the metric instance.
            dtype: Data type of the metric result.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.class_aggregation = class_aggregation
        self.num_regressors = num_regressors
        self.squared_sum = self.add_weight(shape=(), initializer="zeros", name="squared_sum")
        self.sum = self.add_weight(shape=(), initializer="zeros", name="sum")
        self.res = self.add_weight(shape=(), initializer="zeros", name="res")
        self.count = self.add_weight(shape=(), initializer="zeros", name="count")

    def update_state(
        self, y_true: TensorLike, y_pred: TensorLike, sample_weight: TensorLike | None = None
    ) -> None:
        """Accumulate sum-of-squares statistics for R^2 computation.

        Args:
            y_true: Ground-truth regression targets.
            y_pred: Predicted regression values.
            sample_weight: Optional sample weights.
        """
        from zero_keras import ops
        from zero_keras.activations import _to_tensor

        y_true = _to_tensor(y_true)
        y_pred = _to_tensor(y_pred)
        w = _to_tensor(sample_weight) if sample_weight is not None else _to_tensor(1.0)

        ss = ops.sum((y_true ** _to_tensor(2.0)) * w, axis=0)
        su = ops.sum(y_true * w, axis=0)
        re = ops.sum(((y_true - y_pred) ** _to_tensor(2.0)) * w, axis=0)
        cn = ops.sum(w, axis=0)
        self.squared_sum.assign(ops.add(_to_tensor(self.squared_sum), ss))
        self.sum.assign(ops.add(_to_tensor(self.sum), su))
        self.res.assign(ops.add(_to_tensor(self.res), re))
        self.count.assign(ops.add(_to_tensor(self.count), cn))

    def result(self) -> TensorLike:
        """Compute the R^2 score.

        Returns:
            TensorLike: Scalar or per-output R^2 score tensor.
        """
        from zero_keras import ops
        from zero_keras.activations import _to_tensor, _wrap

        cnt = ops.maximum(_to_tensor(self.count), _to_tensor(1e-7))
        sq_sum = _to_tensor(self.squared_sum)
        sm = _to_tensor(self.sum)
        rs = _to_tensor(self.res)
        total = sq_sum - (sm ** _to_tensor(2.0)) / cnt
        r2 = _to_tensor(1.0) - rs / ops.maximum(total, _to_tensor(1e-7))
        if self.class_aggregation == "uniform_average":
            return _wrap(ops.mean(r2))
        elif self.class_aggregation == "variance_weighted_average":
            weight = total / ops.maximum(ops.sum(total), _to_tensor(1e-7))
            return _wrap(ops.sum(r2 * weight))
        return _wrap(r2)

    def reset_state(self) -> None:
        """Reset all sum-of-squares accumulator variables to zero."""
        from zero_keras.activations import _to_tensor

        self.squared_sum.assign(_to_tensor(0.0))
        self.sum.assign(_to_tensor(0.0))
        self.res.assign(_to_tensor(0.0))
        self.count.assign(_to_tensor(0.0))


def serialize(metric: TensorLike) -> TensorLike:
    """Serialize a metric instance or identifier into a configuration dictionary.

    Args:
        metric: Metric instance or string identifier.

    Returns:
        TensorLike: Serialized metric dictionary, string, or None.
    """
    if metric is None:
        return None
    if isinstance(metric, str):
        return metric
    return {
        "class_name": type(metric).__name__,
        "config": metric.get_config() if isinstance(metric, Metric) else {},
    }


def deserialize(
    config: TensorLike, custom_objects: ConfigDict | TensorLike | None = None
) -> TensorLike:
    """Deserialize a metric from its configuration dictionary or string identifier.

    Args:
        config: Serialized metric configuration dictionary or string name.
        custom_objects: Optional mapping of custom metric classes.

    Returns:
        TensorLike: Reconstructed metric instance or original config.
    """
    _ = custom_objects
    if config is None:
        return None
    if isinstance(config, str):
        return get(config)
    if isinstance(config, dict):
        class_name = config.get("class_name")
        conf = config.get("config", {})
        cls = _METRIC_REGISTRY.get(str(class_name)) if isinstance(class_name, str) else None
        if cls is not None and isinstance(conf, dict):
            return cls(**conf)
    return config


def get(identifier: TensorLike) -> TensorLike:
    """Retrieve a Keras metric object via an identifier.

    Args:
        identifier: Metric instance or string name.

    Returns:
        TensorLike: Resolved metric instance, string, or None.
    """
    if identifier is None:
        return None
    if isinstance(identifier, str):
        identifier = identifier.lower()
        if identifier in ["mse", "mean_squared_error"]:
            return MeanSquaredError()
        return identifier
    return identifier


concordance_correlation = ConcordanceCorrelation
pearson_correlation = PearsonCorrelation


def binary_accuracy(y_true: TensorLike, y_pred: TensorLike, threshold: float = 0.5) -> TensorLike:
    """Compute binary accuracy between ``y_true`` and ``y_pred`` at ``threshold``.

    Args:
        y_true: Ground-truth binary targets.
        y_pred: Predicted probabilities or scores.
        threshold: Decision threshold in ``[0, 1]``.

    Returns:
        TensorLike: Per-sample binary accuracy tensor.
    """
    from zero_keras import ops

    y_true = ops.cast(y_true, y_pred.dtype)
    threshold_t = ops.cast(threshold, y_pred.dtype)
    y_pred_thresholded = ops.cast(y_pred > threshold_t, y_pred.dtype)
    return ops.mean(ops.cast(ops.equal(y_true, y_pred_thresholded), y_pred.dtype), axis=-1)


def categorical_accuracy(y_true: TensorLike, y_pred: TensorLike) -> TensorLike:
    """Compute one-hot categorical accuracy between ``y_true`` and ``y_pred``.

    Args:
        y_true: One-hot ground-truth targets.
        y_pred: Predicted class probabilities or logits.

    Returns:
        TensorLike: Per-sample categorical accuracy tensor.
    """
    from zero_keras import ops

    return ops.cast(
        ops.equal(ops.argmax(y_true, axis=-1), ops.argmax(y_pred, axis=-1)),
        y_pred.dtype,
    )


def sparse_categorical_accuracy(y_true: TensorLike, y_pred: TensorLike) -> TensorLike:
    """Compute sparse integer categorical accuracy between ``y_true`` and ``y_pred``.

    Args:
        y_true: Integer class indices.
        y_pred: Predicted class probabilities or logits.

    Returns:
        TensorLike: Per-sample sparse categorical accuracy tensor.
    """
    from zero_keras import ops

    y_pred_rank = len(y_pred.shape)
    y_true_rank = len(y_true.shape)
    y_true = ops.squeeze(y_true, -1) if y_true_rank == y_pred_rank else y_true
    return ops.cast(
        ops.equal(ops.cast(y_true, "int32"), ops.cast(ops.argmax(y_pred, axis=-1), "int32")),
        y_pred.dtype,
    )


def top_k_categorical_accuracy(y_true: TensorLike, y_pred: TensorLike, k: int = 5) -> TensorLike:
    """Compute top-k categorical accuracy for one-hot targets.

    Args:
        y_true: One-hot ground-truth targets.
        y_pred: Predicted class probabilities or logits.
        k: Number of top predictions considered.

    Returns:
        TensorLike: Per-sample top-k categorical accuracy tensor.
    """
    from zero_keras import ops

    return ops.top_k_categorical_accuracy(y_true, y_pred, k=k)


def sparse_top_k_categorical_accuracy(
    y_true: TensorLike, y_pred: TensorLike, k: int = 5
) -> TensorLike:
    """Compute top-k categorical accuracy for integer class targets.

    Args:
        y_true: Integer class indices.
        y_pred: Predicted class probabilities or logits.
        k: Number of top predictions considered.

    Returns:
        TensorLike: Per-sample sparse top-k categorical accuracy tensor.
    """
    from zero_keras import ops

    return ops.sparse_top_k_categorical_accuracy(y_true, y_pred, k=k)


class CategoricalCrossentropy(Metric):
    """Computes the crossentropy metric between the labels and predictions."""

    def __init__(
        self,
        name: str = "categorical_crossentropy",
        dtype: DTypeLike = None,
        from_logits: bool = False,
        label_smoothing: float = 0,
        axis: int = -1,
        **kwargs: TensorLike,
    ) -> None:
        """Initialize CategoricalCrossentropy metric.

        Args:
            name: String name of the metric instance.
            dtype: Data type of the metric result.
            from_logits: Whether ``y_pred`` is expected to be a logits tensor.
            label_smoothing: Float in ``[0, 1]`` for label smoothing.
            axis: The axis along which to compute crossentropy.
            **kwargs: Base metric keyword arguments.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.from_logits = bool(from_logits)
        self.label_smoothing = float(label_smoothing)
        self.axis = int(axis)
        self.total = self.add_weight(shape=(), initializer="zeros", name="total")
        self.count = self.add_weight(shape=(), initializer="zeros", name="count")

    def update_state(
        self, y_true: TensorLike, y_pred: TensorLike, sample_weight: TensorLike | None = None
    ) -> None:
        """Accumulate categorical crossentropy statistics.

        Args:
            y_true: One-hot ground truth targets.
            y_pred: Predicted probabilities or logits.
            sample_weight: Optional weighting of individual examples.
        """
        from zero_keras import losses

        yt, yp = _to_tensor(y_true), _to_tensor(y_pred)
        val = ops.mean(
            losses.categorical_crossentropy(
                yt,
                yp,
                from_logits=self.from_logits,
                label_smoothing=self.label_smoothing,
            )
        )
        if sample_weight is not None:
            val = ops.multiply(val, ops.mean(_to_tensor(sample_weight)))
        self.total.assign(ops.add(_to_tensor(self.total), val))
        self.count.assign(ops.add(_to_tensor(self.count), _to_tensor(1.0)))

    def result(self) -> TensorLike:
        """Compute the current mean categorical crossentropy value.

        Returns:
            TensorLike: Scalar metric tensor.
        """
        return _wrap(
            ops.divide(
                _to_tensor(self.total),
                ops.maximum(_to_tensor(self.count), _to_tensor(1e-7)),
            )
        )


_METRIC_REGISTRY: dict[str, type[Metric]] = {
    "AUC": AUC,
    "Accuracy": Accuracy,
    "BinaryAccuracy": BinaryAccuracy,
    "BinaryCrossentropy": BinaryCrossentropy,
    "BinaryIoU": BinaryIoU,
    "CategoricalAccuracy": CategoricalAccuracy,
    "CategoricalCrossentropy": CategoricalCrossentropy,
    "CategoricalHinge": CategoricalHinge,
    "ConcordanceCorrelation": ConcordanceCorrelation,
    "CosineSimilarity": CosineSimilarity,
    "F1Score": F1Score,
    "FBetaScore": FBetaScore,
    "FalseNegatives": FalseNegatives,
    "FalsePositives": FalsePositives,
    "Hinge": Hinge,
    "IoU": IoU,
    "KLDivergence": KLDivergence,
    "LogCoshError": LogCoshError,
    "Mean": Mean,
    "MeanAbsoluteError": MeanAbsoluteError,
    "MeanAbsolutePercentageError": MeanAbsolutePercentageError,
    "MeanIoU": MeanIoU,
    "MeanMetricWrapper": MeanMetricWrapper,
    "MeanSquaredError": MeanSquaredError,
    "MeanSquaredLogarithmicError": MeanSquaredLogarithmicError,
    "Metric": Metric,
    "OneHotIoU": OneHotIoU,
    "OneHotMeanIoU": OneHotMeanIoU,
    "PearsonCorrelation": PearsonCorrelation,
    "Poisson": Poisson,
    "Precision": Precision,
    "PrecisionAtRecall": PrecisionAtRecall,
    "R2Score": R2Score,
    "Recall": Recall,
    "RecallAtPrecision": RecallAtPrecision,
    "RootMeanSquaredError": RootMeanSquaredError,
    "SensitivityAtSpecificity": SensitivityAtSpecificity,
    "SparseCategoricalAccuracy": SparseCategoricalAccuracy,
    "SparseCategoricalCrossentropy": SparseCategoricalCrossentropy,
    "SparseTopKCategoricalAccuracy": SparseTopKCategoricalAccuracy,
    "SpecificityAtSensitivity": SpecificityAtSensitivity,
    "SquaredHinge": SquaredHinge,
    "Sum": Sum,
    "TopKCategoricalAccuracy": TopKCategoricalAccuracy,
    "TrueNegatives": TrueNegatives,
    "TruePositives": TruePositives,
}
