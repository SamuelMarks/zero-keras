"""Module docstring."""

# Aliases from losses (exported)
from zero_keras.losses import CategoricalCrossentropy as CategoricalCrossentropy
from zero_keras.losses import binary_crossentropy as binary_crossentropy
from zero_keras.losses import binary_focal_crossentropy as binary_focal_crossentropy
from zero_keras.losses import categorical_crossentropy as categorical_crossentropy
from zero_keras.losses import (
    categorical_focal_crossentropy as categorical_focal_crossentropy,
)
from zero_keras.losses import categorical_hinge as categorical_hinge
from zero_keras.losses import hinge as hinge
from zero_keras.losses import huber as huber
from zero_keras.losses import kl_divergence as kl_divergence
from zero_keras.losses import log_cosh as log_cosh
from zero_keras.losses import mean_absolute_error as mean_absolute_error
from zero_keras.losses import (
    mean_absolute_percentage_error as mean_absolute_percentage_error,
)
from zero_keras.losses import mean_squared_error as mean_squared_error
from zero_keras.losses import (
    mean_squared_logarithmic_error as mean_squared_logarithmic_error,
)
from zero_keras.losses import poisson as poisson
from zero_keras.losses import (
    sparse_categorical_crossentropy as sparse_categorical_crossentropy,
)
from zero_keras.losses import squared_hinge as squared_hinge

from zero_keras.ops import ops


def _filter_top_k_and_class_id(y_true, y_pred, top_k, class_id):
    """Function docstring.

    Args:
        y_true: Description.
        y_pred: Description.
        top_k: Description.
        class_id: Description.
    """
    from zero_keras.ops import ops

    if top_k is not None:
        top_k_vals, _ = ops.top_k(y_pred, k=top_k)
        min_top_k = ops.min(top_k_vals, axis=-1, keepdims=True)
        y_pred = ops.where(y_pred >= min_top_k, y_pred, _to_tensor(0.0))
    if class_id is not None:
        y_true = y_true[..., class_id]
        y_pred = y_pred[..., class_id]
    return y_true, y_pred


import ml_switcheroo_compiler.nn as nn
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

    def add_variable(
        self, shape, initializer, dtype=None, aggregation="sum", name=None
    ):
        """add_variable docstring.

        Args:
            shape: Shape.
            initializer: Initializer.
            dtype: Dtype.
            aggregation: Aggregation.
            name: Name.
        """
        from zero_keras.core_layers import Variable

        return Variable(initializer, shape=shape, dtype=dtype, name=name)

    def add_weight(self, shape=(), initializer="zeros", dtype=None, name=None):
        """add_weight docstring.

        Args:
            shape: Shape.
            initializer: Initializer.
            dtype: Dtype.
            name: Name.
        """
        from zero_keras.core_layers import Variable

        return Variable(initializer, shape=shape, dtype=dtype, name=name)

    @property
    def dtype(self):
        """dtype docstring."""
        return getattr(self, "_dtype", None)

    @dtype.setter
    def dtype(self, value):
        self._dtype = value

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

    def stateless_reset_state(self):
        """stateless_reset_state docstring."""
        pass

    def stateless_result(self, *args, **kwargs):
        """stateless_result docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        return self.result()

    def stateless_update_state(self, *args, **kwargs):
        """stateless_update_state docstring.

        Args:
            *args: Args.
            **kwargs: Kwargs.
        """
        return self.update_state(*args, **kwargs)

    @property
    def variables(self):
        """variables docstring."""
        return getattr(self, "weights", [])

    def __init__(self, name=None, dtype=None, **kwargs):
        """Function docstring.

        Args:
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        self.name = name
        self.dtype = dtype
        self._kwargs = kwargs

    def update_state(self, *args, **kwargs):
        """Accumulate statistics for the metric."""
        pass

    def result(self):
        """Compute the current metric value.

        Returns:
            A scalar tensor, or a dictionary of scalar tensors.

        """
        return _wrap(_to_tensor(0.0))

    def reset_state(self):
        """Reset all of the metric state variables.

        This function is called between epochs/steps,
        when a metric is evaluated during training.
        """
        if hasattr(self, "total"):
            self.total = 0.0
        if hasattr(self, "count"):
            self.count = 0.0

    def __call__(self, *args, **kwargs):
        """Call self as a function."""
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

    def __init__(self, name="mean", dtype=None, **kwargs):
        """Function docstring.

        Args:
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = 0.0
        self.count = 0.0

    def update_state(self, values, sample_weight=None):
        """Accumulate statistics for the metric."""
        values = _to_tensor(values)
        if sample_weight is not None:
            sample_weight = _to_tensor(sample_weight)
            values = ops.multiply(values, sample_weight)
            self.count += ops.cast(ops.sum(sample_weight), dtype="float32")
        else:
            self.count += ops.cast(ops.sum(ops.ones_like(values)), dtype="float32")
        self.total += ops.cast(ops.sum(values), dtype="float32")

    def result(self):
        """Compute the current metric value.

        Returns:
            A scalar tensor, or a dictionary of scalar tensors.

        """
        return _wrap(self.total / ops.maximum(_to_tensor(self.count), 1e-7))

    def reset_state(self):
        """Reset all of the metric state variables.

        This function is called between epochs/steps,
        when a metric is evaluated during training.
        """
        self.total = 0.0
        self.count = 0.0


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

    def __init__(self, name="sum", dtype=None, **kwargs):
        """Function docstring.

        Args:
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = 0.0

    def update_state(self, values, sample_weight=None):
        """Accumulate statistics for the metric."""
        values = _to_tensor(values)
        if sample_weight is not None:
            sample_weight = _to_tensor(sample_weight)
            values = ops.multiply(values, sample_weight)
        self.total += ops.cast(ops.sum(values), dtype="float32")

    def result(self):
        """Compute the current metric value.

        Returns:
            A scalar tensor, or a dictionary of scalar tensors.

        """
        return _wrap(self.total)

    def reset_state(self):
        """Reset all of the metric state variables.

        This function is called between epochs/steps,
        when a metric is evaluated during training.
        """
        self.total = 0.0


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

    def __init__(self, fn, name=None, dtype=None, **kwargs):
        """Function docstring.

        Args:
            fn: Description.
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.fn = fn

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulate statistics for the metric."""
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

    def __init__(self, name="accuracy", dtype=None, **kwargs):
        """Function docstring.

        Args:
            name: Description.
            dtype: Description.
            kwargs: Description.
        """

        def accuracy_fn(y_true, y_pred):
            """accuracy_fn function.

            Args:
            y_true: Parameter y_true.
            y_pred: Parameter y_pred.

            Returns:
            Any: Return value.

            """
            return ops.cast(
                ops.equal(_to_tensor(y_true), _to_tensor(y_pred)), dtype="float32"
            )

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

    def __init__(self, name="binary_accuracy", dtype=None, threshold=0.5, **kwargs):
        """Function docstring.

        Args:
            name: Description.
            dtype: Description.
            threshold: Description.
            kwargs: Description.
        """

        def binary_accuracy_fn(y_true, y_pred):
            """binary_accuracy_fn function.

            Args:
            y_true: Parameter y_true.
            y_pred: Parameter y_pred.

            Returns:
            Any: Return value.

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

    def __init__(self, name="categorical_accuracy", dtype=None, **kwargs):
        """Function docstring.

        Args:
            name: Description.
            dtype: Description.
            kwargs: Description.
        """

        def categorical_accuracy_fn(y_true, y_pred):
            """categorical_accuracy_fn function.

            Args:
            y_true: Parameter y_true.
            y_pred: Parameter y_pred.

            Returns:
            Any: Return value.

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

    def __init__(self, name="sparse_categorical_accuracy", dtype=None, **kwargs):
        """Function docstring.

        Args:
            name: Description.
            dtype: Description.
            kwargs: Description.
        """

        def sparse_categorical_accuracy_fn(y_true, y_pred):
            """sparse_categorical_accuracy_fn function.

            Args:
            y_true: Parameter y_true.
            y_pred: Parameter y_pred.

            Returns:
            Any: Return value.

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

        super().__init__(
            fn=sparse_categorical_accuracy_fn, name=name, dtype=dtype, **kwargs
        )


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

    def __init__(self, k=5, name="top_k_categorical_accuracy", dtype=None, **kwargs):
        """Function docstring.

        Args:
            k: Description.
            name: Description.
            dtype: Description.
            kwargs: Description.
        """

        def top_k_fn(y_true, y_pred):
            """top_k_fn function.

            Args:
            y_true: Parameter y_true.
            y_pred: Parameter y_pred.

            Returns:
            Any: Return value.

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
        self, k=5, name="sparse_top_k_categorical_accuracy", dtype=None, **kwargs
    ):
        """Function docstring.

        Args:
            k: Description.
            name: Description.
            dtype: Description.
            kwargs: Description.
        """

        def sparse_top_k_fn(y_true, y_pred):
            """sparse_top_k_fn function.

            Args:
            y_true: Parameter y_true.
            y_pred: Parameter y_pred.

            Returns:
            Any: Return value.

            """
            y_true = _to_tensor(y_true)
            y_pred = _to_tensor(y_pred)

            # Ensure y_true has the correct dtype for comparison, if needed
            y_true = ops.cast(y_true, "int32")

            _, top_indices = ops.top_k(y_pred, k)
            # Expand dims if y_true is just shape (batch,) but top_indices is (batch, k)
            # Assuming y_true is a 1D tensor of class indices
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
    """Class docstring."""

    def __init__(
        self, thresholds=None, metric_type="FP", name=None, dtype=None, **kwargs
    ):
        """Function docstring.

        Args:
            thresholds: Description.
            metric_type: Description.
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        if metric_type not in ("FP", "FN", "TP", "TN"):
            raise ValueError(f"Unknown metric_type {metric_type}")
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.thresholds = 0.5 if thresholds is None else thresholds
        self.metric_type = metric_type
        self.reset_state()

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Function docstring.

        Args:
            y_true: Description.
            y_pred: Description.
            sample_weight: Description.
        """
        y_true = _to_tensor(y_true)
        y_pred = _to_tensor(y_pred)
        y_true = ops.cast(_to_tensor(y_true), "bool")

        if isinstance(self.thresholds, (list, tuple)):
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
                    val = ops.logical_and(
                        ops.logical_not(y_true), ops.logical_not(y_pred_i)
                    )
                else:
                    raise ValueError(
                        f"Unknown metric_type {self.metric_type}"
                    )  # pragma: no cover

                val = ops.cast(val, "float32")
                if sample_weight is not None:
                    val = val * ops.cast(sample_weight, "float32")
                try:
                    self.accumulator[i].assign(self.accumulator[i] + ops.sum(val))
                except AttributeError:
                    self.accumulator[i] += ops.sum(val)
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
                val = ops.logical_and(
                    ops.logical_not(y_true), ops.logical_not(y_pred_bool)
                )
            else:
                raise ValueError(
                    f"Unknown metric_type {self.metric_type}"
                )  # pragma: no cover

            val = ops.cast(val, "float32")
            if sample_weight is not None:
                val = val * ops.cast(sample_weight, "float32")

            try:
                self.accumulator.assign(self.accumulator + ops.sum(val))
            except AttributeError:
                self.accumulator += ops.sum(val)

    def result(self):
        """Function docstring."""
        if isinstance(self.accumulator, list):
            from zero_keras.ops import ops

            return ops.stack(self.accumulator)
        return self.accumulator

    def reset_state(self):
        """Function docstring."""
        if isinstance(self.thresholds, (list, tuple)):
            self.accumulator = [0.0] * len(self.thresholds)
        else:
            self.accumulator = 0.0


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

    def __init__(self, thresholds=None, name=None, dtype=None, **kwargs):
        """Function docstring.

        Args:
            thresholds: Description.
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        super().__init__(
            thresholds=thresholds, metric_type="FP", name=name, dtype=dtype, **kwargs
        )


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

    def __init__(self, thresholds=None, name=None, dtype=None, **kwargs):
        """Function docstring.

        Args:
            thresholds: Description.
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        super().__init__(
            thresholds=thresholds, metric_type="FN", name=name, dtype=dtype, **kwargs
        )


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

    def __init__(self, thresholds=None, name=None, dtype=None, **kwargs):
        """Function docstring.

        Args:
            thresholds: Description.
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        super().__init__(
            thresholds=thresholds, metric_type="TN", name=name, dtype=dtype, **kwargs
        )


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

    def __init__(self, thresholds=None, name=None, dtype=None, **kwargs):
        """Function docstring.

        Args:
            thresholds: Description.
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        super().__init__(
            thresholds=thresholds, metric_type="TP", name=name, dtype=dtype, **kwargs
        )


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
        thresholds=None,
        top_k=None,
        class_id=None,
        name=None,
        dtype=None,
        **kwargs,
    ):
        """Function docstring.

        Args:
            thresholds: Description.
            top_k: Description.
            class_id: Description.
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.thresholds = 0.5 if thresholds is None and top_k is None else thresholds
        self.top_k = top_k
        self.class_id = class_id

        self.true_positives = TruePositives(thresholds=self.thresholds)
        self.false_positives = FalsePositives(thresholds=self.thresholds)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Function docstring.

        Args:
            y_true: Description.
            y_pred: Description.
            sample_weight: Description.
        """
        y_true = _to_tensor(y_true)
        y_pred = _to_tensor(y_pred)
        y_true, y_pred = _filter_top_k_and_class_id(
            y_true, y_pred, getattr(self, "top_k", None), self.class_id
        )
        self.true_positives.update_state(y_true, y_pred, sample_weight)
        self.false_positives.update_state(y_true, y_pred, sample_weight)

    def result(self):
        """Function docstring."""
        tp = self.true_positives.result()
        fp = self.false_positives.result()

        # tp and fp could be lists or arrays if multiple thresholds.
        # But our confusion matrix result() handles it.
        # Just safely divide.
        from zero_keras.ops import ops

        # We need to handle list of thresholds properly if it returns an array
        res = ops.where(
            ops.greater(tp + fp, 0.0), ops.divide(tp, tp + fp + 1e-7), _to_tensor(0.0)
        )
        return res

    def reset_state(self):
        """Function docstring."""
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
        thresholds=None,
        top_k=None,
        class_id=None,
        name=None,
        dtype=None,
        **kwargs,
    ):
        """Function docstring.

        Args:
            thresholds: Description.
            top_k: Description.
            class_id: Description.
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.thresholds = 0.5 if thresholds is None and top_k is None else thresholds
        self.top_k = top_k
        self.class_id = class_id

        self.true_positives = TruePositives(thresholds=self.thresholds)
        self.false_negatives = FalseNegatives(thresholds=self.thresholds)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Function docstring.

        Args:
            y_true: Description.
            y_pred: Description.
            sample_weight: Description.
        """
        y_true = _to_tensor(y_true)
        y_pred = _to_tensor(y_pred)
        y_true, y_pred = _filter_top_k_and_class_id(
            y_true, y_pred, getattr(self, "top_k", None), self.class_id
        )
        self.true_positives.update_state(y_true, y_pred, sample_weight)
        self.false_negatives.update_state(y_true, y_pred, sample_weight)

    def result(self):
        """Function docstring."""
        tp = self.true_positives.result()
        fn = self.false_negatives.result()
        from zero_keras.ops import ops

        res = ops.where(
            ops.greater(tp + fn, 0.0), ops.divide(tp, tp + fn + 1e-7), _to_tensor(0.0)
        )
        return res

    def reset_state(self):
        """Function docstring."""
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
        recall,
        num_thresholds=200,
        class_id=None,
        name=None,
        dtype=None,
        **kwargs,
    ):
        """Function docstring.

        Args:
            recall: Description.
            num_thresholds: Description.
            class_id: Description.
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.recall = recall
        self.num_thresholds = num_thresholds
        self.class_id = class_id

        self.thresholds = [
            i / (self.num_thresholds - 1) for i in range(self.num_thresholds)
        ]
        self.true_positives = TruePositives(thresholds=self.thresholds)
        self.false_positives = FalsePositives(thresholds=self.thresholds)
        self.true_negatives = TrueNegatives(thresholds=self.thresholds)
        self.false_negatives = FalseNegatives(thresholds=self.thresholds)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Function docstring.

        Args:
            y_true: Description.
            y_pred: Description.
            sample_weight: Description.
        """
        y_true = _to_tensor(y_true)
        y_pred = _to_tensor(y_pred)
        y_true, y_pred = _filter_top_k_and_class_id(
            y_true, y_pred, getattr(self, "top_k", None), self.class_id
        )
        self.true_positives.update_state(y_true, y_pred, sample_weight)
        self.false_positives.update_state(y_true, y_pred, sample_weight)
        self.true_negatives.update_state(y_true, y_pred, sample_weight)
        self.false_negatives.update_state(y_true, y_pred, sample_weight)

    def result(self):
        """Function docstring."""
        tp = self.true_positives.result()
        fp = self.false_positives.result()
        self.true_negatives.result()
        fn = self.false_negatives.result()
        from zero_keras.ops import ops

        recalls = ops.where(
            ops.greater(tp + fn, 0.0), ops.divide(tp, tp + fn + 1e-7), _to_tensor(0.0)
        )
        precisions = ops.where(
            ops.greater(tp + fp, 0.0), ops.divide(tp, tp + fp + 1e-7), _to_tensor(0.0)
        )

        # We need the maximum precision where recall >= self.recall
        # If no such threshold exists, return 0.0
        condition = ops.greater_equal(recalls, self.recall)
        valid_precisions = ops.where(condition, precisions, _to_tensor(0.0))
        return ops.max(valid_precisions)

    def reset_state(self):
        """Function docstring."""
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
        precision,
        num_thresholds=200,
        class_id=None,
        name=None,
        dtype=None,
        **kwargs,
    ):
        """Function docstring.

        Args:
            precision: Description.
            num_thresholds: Description.
            class_id: Description.
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.precision = precision
        self.num_thresholds = num_thresholds
        self.class_id = class_id

        self.thresholds = [
            i / (self.num_thresholds - 1) for i in range(self.num_thresholds)
        ]
        self.true_positives = TruePositives(thresholds=self.thresholds)
        self.false_positives = FalsePositives(thresholds=self.thresholds)
        self.true_negatives = TrueNegatives(thresholds=self.thresholds)
        self.false_negatives = FalseNegatives(thresholds=self.thresholds)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Function docstring.

        Args:
            y_true: Description.
            y_pred: Description.
            sample_weight: Description.
        """
        y_true = _to_tensor(y_true)
        y_pred = _to_tensor(y_pred)
        y_true, y_pred = _filter_top_k_and_class_id(
            y_true, y_pred, getattr(self, "top_k", None), self.class_id
        )
        self.true_positives.update_state(y_true, y_pred, sample_weight)
        self.false_positives.update_state(y_true, y_pred, sample_weight)
        self.true_negatives.update_state(y_true, y_pred, sample_weight)
        self.false_negatives.update_state(y_true, y_pred, sample_weight)

    def result(self):
        """Function docstring."""
        tp = self.true_positives.result()
        fp = self.false_positives.result()
        self.true_negatives.result()
        fn = self.false_negatives.result()
        from zero_keras.ops import ops

        recalls = ops.where(
            ops.greater(tp + fn, 0.0), ops.divide(tp, tp + fn + 1e-7), _to_tensor(0.0)
        )
        precisions = ops.where(
            ops.greater(tp + fp, 0.0), ops.divide(tp, tp + fp + 1e-7), _to_tensor(0.0)
        )

        condition = ops.greater_equal(precisions, self.precision)
        valid_recalls = ops.where(condition, recalls, _to_tensor(0.0))
        return ops.max(valid_recalls)

    def reset_state(self):
        """Function docstring."""
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
        specificity,
        num_thresholds=200,
        class_id=None,
        name=None,
        dtype=None,
        **kwargs,
    ):
        """Function docstring.

        Args:
            specificity: Description.
            num_thresholds: Description.
            class_id: Description.
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.specificity = specificity
        self.num_thresholds = num_thresholds
        self.class_id = class_id

        self.thresholds = [
            i / (self.num_thresholds - 1) for i in range(self.num_thresholds)
        ]
        self.true_positives = TruePositives(thresholds=self.thresholds)
        self.false_positives = FalsePositives(thresholds=self.thresholds)
        self.true_negatives = TrueNegatives(thresholds=self.thresholds)
        self.false_negatives = FalseNegatives(thresholds=self.thresholds)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Function docstring.

        Args:
            y_true: Description.
            y_pred: Description.
            sample_weight: Description.
        """
        y_true = _to_tensor(y_true)
        y_pred = _to_tensor(y_pred)
        y_true, y_pred = _filter_top_k_and_class_id(
            y_true, y_pred, getattr(self, "top_k", None), self.class_id
        )
        self.true_positives.update_state(y_true, y_pred, sample_weight)
        self.false_positives.update_state(y_true, y_pred, sample_weight)
        self.true_negatives.update_state(y_true, y_pred, sample_weight)
        self.false_negatives.update_state(y_true, y_pred, sample_weight)

    def result(self):
        """Function docstring."""
        tp = self.true_positives.result()
        fp = self.false_positives.result()
        tn = self.true_negatives.result()
        fn = self.false_negatives.result()
        from zero_keras.ops import ops

        specificities = ops.where(
            ops.greater(tn + fp, 0.0), ops.divide(tn, tn + fp + 1e-7), _to_tensor(0.0)
        )
        sensitivities = ops.where(
            ops.greater(tp + fn, 0.0), ops.divide(tp, tp + fn + 1e-7), _to_tensor(0.0)
        )

        condition = ops.greater_equal(specificities, self.specificity)
        valid_sensitivities = ops.where(condition, sensitivities, _to_tensor(0.0))
        return ops.max(valid_sensitivities)

    def reset_state(self):
        """Function docstring."""
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
        sensitivity,
        num_thresholds=200,
        class_id=None,
        name=None,
        dtype=None,
        **kwargs,
    ):
        """Function docstring.

        Args:
            sensitivity: Description.
            num_thresholds: Description.
            class_id: Description.
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.sensitivity = sensitivity
        self.num_thresholds = num_thresholds
        self.class_id = class_id

        self.thresholds = [
            i / (self.num_thresholds - 1) for i in range(self.num_thresholds)
        ]
        self.true_positives = TruePositives(thresholds=self.thresholds)
        self.false_positives = FalsePositives(thresholds=self.thresholds)
        self.true_negatives = TrueNegatives(thresholds=self.thresholds)
        self.false_negatives = FalseNegatives(thresholds=self.thresholds)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Function docstring.

        Args:
            y_true: Description.
            y_pred: Description.
            sample_weight: Description.
        """
        y_true = _to_tensor(y_true)
        y_pred = _to_tensor(y_pred)
        y_true, y_pred = _filter_top_k_and_class_id(
            y_true, y_pred, getattr(self, "top_k", None), self.class_id
        )
        self.true_positives.update_state(y_true, y_pred, sample_weight)
        self.false_positives.update_state(y_true, y_pred, sample_weight)
        self.true_negatives.update_state(y_true, y_pred, sample_weight)
        self.false_negatives.update_state(y_true, y_pred, sample_weight)

    def result(self):
        """Function docstring."""
        tp = self.true_positives.result()
        fp = self.false_positives.result()
        tn = self.true_negatives.result()
        fn = self.false_negatives.result()
        from zero_keras.ops import ops

        specificities = ops.where(
            ops.greater(tn + fp, 0.0), ops.divide(tn, tn + fp + 1e-7), _to_tensor(0.0)
        )
        sensitivities = ops.where(
            ops.greater(tp + fn, 0.0), ops.divide(tp, tp + fn + 1e-7), _to_tensor(0.0)
        )

        condition = ops.greater_equal(sensitivities, self.sensitivity)
        valid_specificities = ops.where(condition, specificities, _to_tensor(0.0))
        return ops.max(valid_specificities)

    def reset_state(self):
        """Function docstring."""
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

    def __init__(
        self,
        num_thresholds=200,
        curve="ROC",
        summation_method="interpolation",
        name=None,
        dtype=None,
        thresholds=None,
        multi_label=False,
        num_labels=None,
        label_weights=None,
        from_logits=False,
        **kwargs,
    ):
        """Function docstring.

        Args:
            num_thresholds: Description.
            curve: Description.
            summation_method: Description.
            name: Description.
            dtype: Description.
            thresholds: Description.
            multi_label: Description.
            num_labels: Description.
            label_weights: Description.
            from_logits: Description.
            kwargs: Description.
        """
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
            self.thresholds = thresholds
        else:
            self.thresholds = [
                i / (self.num_thresholds - 1) for i in range(self.num_thresholds)
            ]

        self.true_positives = TruePositives(thresholds=self.thresholds)
        self.false_positives = FalsePositives(thresholds=self.thresholds)
        self.true_negatives = TrueNegatives(thresholds=self.thresholds)
        self.false_negatives = FalseNegatives(thresholds=self.thresholds)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Function docstring.

        Args:
            y_true: Description.
            y_pred: Description.
            sample_weight: Description.
        """
        if self.from_logits:
            from zero_keras.activations import sigmoid

            y_pred = sigmoid(y_pred)

        self.true_positives.update_state(y_true, y_pred, sample_weight)
        self.false_positives.update_state(y_true, y_pred, sample_weight)
        self.true_negatives.update_state(y_true, y_pred, sample_weight)
        self.false_negatives.update_state(y_true, y_pred, sample_weight)

    def result(self):
        """Function docstring."""
        tp = self.true_positives.result()
        fp = self.false_positives.result()
        tn = self.true_negatives.result()
        fn = self.false_negatives.result()
        from zero_keras.ops import ops

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
            raise ValueError(f"Invalid curve: {self.curve}")  # pragma: no cover

        # Riemann sum interpolation
        # area = sum( (x[i] - x[i+1]) * (y[i] + y[i+1]) / 2 )
        # Note: x is descending as threshold increases.
        # So x[i-1] - x[i] is positive.
        x_diff = x[:-1] - x[1:]
        if self.summation_method in ("minoring", "minor"):
            y_val = ops.minimum(y[:-1], y[1:])
        elif self.summation_method in ("majoring", "major"):
            y_val = ops.maximum(y[:-1], y[1:])
        else:
            y_val = (y[:-1] + y[1:]) / 2.0
        return ops.sum(x_diff * y_val)

    def reset_state(self):
        """Function docstring."""
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

    def __init__(self, name="cosinesimilarity", dtype=None, **kwargs):
        """Function docstring.

        Args:
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = 0.0
        self.count = 0.0

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulate statistics for the metric."""
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        from zero_keras import losses

        val = -ops.mean(losses.cosine_similarity(y_true, y_pred))
        if sample_weight is not None:
            val = ops.multiply(val, ops.mean(_to_tensor(sample_weight)))
        self.total = ops.add(_to_tensor(self.total), val)
        self.count = ops.add(_to_tensor(self.count), _to_tensor(1.0))

    def result(self):
        """Compute the current metric value.

        Returns:
            A scalar tensor, or a dictionary of scalar tensors.

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

    def __init__(self, name="meanabsoluteerror", dtype=None, **kwargs):
        """Function docstring.

        Args:
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = 0.0
        self.count = 0.0

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulate statistics for the metric."""
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        from zero_keras import losses

        val = ops.mean(losses.mean_absolute_error(y_true, y_pred))
        if sample_weight is not None:
            val = ops.multiply(val, ops.mean(_to_tensor(sample_weight)))
        self.total = ops.add(_to_tensor(self.total), val)
        self.count = ops.add(_to_tensor(self.count), _to_tensor(1.0))

    def result(self):
        """Compute the current metric value.

        Returns:
            A scalar tensor, or a dictionary of scalar tensors.

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

    def __init__(self, name="meanabsolutepercentageerror", dtype=None, **kwargs):
        """Function docstring.

        Args:
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = 0.0
        self.count = 0.0

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulate statistics for the metric."""
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        from zero_keras import losses

        val = ops.mean(losses.mean_absolute_percentage_error(y_true, y_pred))
        if sample_weight is not None:
            val = ops.multiply(val, ops.mean(_to_tensor(sample_weight)))
        self.total = ops.add(_to_tensor(self.total), val)
        self.count = ops.add(_to_tensor(self.count), _to_tensor(1.0))

    def result(self):
        """Compute the current metric value.

        Returns:
            A scalar tensor, or a dictionary of scalar tensors.

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

    def __init__(self, name="meansquarederror", dtype=None, **kwargs):
        """Function docstring.

        Args:
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = 0.0
        self.count = 0.0

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulate statistics for the metric."""
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        from zero_keras import losses

        val = ops.mean(losses.mean_squared_error(y_true, y_pred))
        if sample_weight is not None:
            val = ops.multiply(val, ops.mean(_to_tensor(sample_weight)))
        self.total = ops.add(_to_tensor(self.total), val)
        self.count = ops.add(_to_tensor(self.count), _to_tensor(1.0))

    def result(self):
        """Compute the current metric value.

        Returns:
            A scalar tensor, or a dictionary of scalar tensors.

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

    def __init__(self, name="meansquaredlogarithmicerror", dtype=None, **kwargs):
        """Function docstring.

        Args:
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = 0.0
        self.count = 0.0

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulate statistics for the metric."""
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        from zero_keras import losses

        val = ops.mean(losses.mean_squared_logarithmic_error(y_true, y_pred))
        if sample_weight is not None:
            val = ops.multiply(val, ops.mean(_to_tensor(sample_weight)))
        self.total = ops.add(_to_tensor(self.total), val)
        self.count = ops.add(_to_tensor(self.count), _to_tensor(1.0))

    def result(self):
        """Compute the current metric value.

        Returns:
            A scalar tensor, or a dictionary of scalar tensors.

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

    def __init__(self, name="hinge", dtype=None, **kwargs):
        """Function docstring.

        Args:
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = 0.0
        self.count = 0.0

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulate statistics for the metric."""
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        from zero_keras import losses

        val = ops.mean(losses.hinge(y_true, y_pred))
        if sample_weight is not None:
            val = ops.multiply(val, ops.mean(_to_tensor(sample_weight)))
        self.total = ops.add(_to_tensor(self.total), val)
        self.count = ops.add(_to_tensor(self.count), _to_tensor(1.0))

    def result(self):
        """Compute the current metric value.

        Returns:
            A scalar tensor, or a dictionary of scalar tensors.

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

    def __init__(self, name="squaredhinge", dtype=None, **kwargs):
        """Function docstring.

        Args:
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = 0.0
        self.count = 0.0

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulate statistics for the metric."""
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        from zero_keras import losses

        val = ops.mean(losses.squared_hinge(y_true, y_pred))
        if sample_weight is not None:
            val = ops.multiply(val, ops.mean(_to_tensor(sample_weight)))
        self.total = ops.add(_to_tensor(self.total), val)
        self.count = ops.add(_to_tensor(self.count), _to_tensor(1.0))

    def result(self):
        """Compute the current metric value.

        Returns:
            A scalar tensor, or a dictionary of scalar tensors.

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

    def __init__(self, name="categoricalhinge", dtype=None, **kwargs):
        """Function docstring.

        Args:
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = 0.0
        self.count = 0.0

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulate statistics for the metric."""
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        from zero_keras import losses

        val = ops.mean(losses.categorical_hinge(y_true, y_pred))
        if sample_weight is not None:
            val = ops.multiply(val, ops.mean(_to_tensor(sample_weight)))
        self.total = ops.add(_to_tensor(self.total), val)
        self.count = ops.add(_to_tensor(self.count), _to_tensor(1.0))

    def result(self):
        """Compute the current metric value.

        Returns:
            A scalar tensor, or a dictionary of scalar tensors.

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

    def __init__(self, name="rootmeansquarederror", dtype=None, **kwargs):
        """Function docstring.

        Args:
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = 0.0
        self.count = 0.0

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates root mean squared error statistics.

        Args:
            y_true: The ground truth values.
            y_pred: The predicted values.
            sample_weight: Optional weighting of each example. Can
                be a `Tensor` whose rank is either 0, or the same rank as
                `y_true`, and must be broadcastable to `y_true`.
                Defaults to `1`.

        Returns:
            Update op.

        """
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        from zero_keras import losses

        val = ops.mean(losses.mean_squared_error(y_true, y_pred))
        if sample_weight is not None:
            val = ops.multiply(val, ops.mean(_to_tensor(sample_weight)))
        self.total = ops.add(_to_tensor(self.total), val)
        self.count = ops.add(_to_tensor(self.count), _to_tensor(1.0))

    def result(self):
        """Compute the current metric value.

        Returns:
            A scalar tensor, or a dictionary of scalar tensors.

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
        name="sparse_categorical_crossentropy",
        dtype=None,
        from_logits=False,
        axis=-1,
        **kwargs,
    ):
        """Function docstring.

        Args:
            name: Description.
            dtype: Description.
            from_logits: Description.
            axis: Description.
            kwargs: Description.
        """

        def scc_fn(y_true, y_pred):
            """Function docstring.

            Args:
                y_true: Description.
                y_pred: Description.
            """

            return sparse_categorical_crossentropy(
                y_true, y_pred, from_logits=from_logits
            )

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
        name="binary_crossentropy",
        dtype=None,
        from_logits=False,
        label_smoothing=0.0,
        **kwargs,
    ):
        """Function docstring.

        Args:
            name: Description.
            dtype: Description.
            from_logits: Description.
            label_smoothing: Description.
            kwargs: Description.
        """

        def bce_fn(y_true, y_pred):
            """Function docstring.

            Args:
                y_true: Description.
                y_pred: Description.
            """

            return binary_crossentropy(
                y_true, y_pred, from_logits=from_logits, label_smoothing=label_smoothing
            )

        super().__init__(fn=bce_fn, name=name, dtype=dtype, **kwargs)


class KLDivergence(Metric):
    """Computes Kullback-Leibler divergence metric between `y_true` and
    `y_pred`.

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

    def __init__(self, name="kldivergence", dtype=None, **kwargs):
        """Function docstring.

        Args:
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = 0.0
        self.count = 0.0

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulate statistics for the metric."""
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        from zero_keras import losses

        val = ops.mean(losses.kl_divergence(y_true, y_pred))
        if sample_weight is not None:
            val = ops.multiply(val, ops.mean(_to_tensor(sample_weight)))
        self.total = ops.add(_to_tensor(self.total), val)
        self.count = ops.add(_to_tensor(self.count), _to_tensor(1.0))

    def result(self):
        """Compute the current metric value.

        Returns:
            A scalar tensor, or a dictionary of scalar tensors.

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

    def __init__(self, name="poisson", dtype=None, **kwargs):
        """Function docstring.

        Args:
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = 0.0
        self.count = 0.0

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulate statistics for the metric."""
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        from zero_keras import losses

        val = ops.mean(losses.poisson(y_true, y_pred))
        if sample_weight is not None:
            val = ops.multiply(val, ops.mean(_to_tensor(sample_weight)))
        self.total = ops.add(_to_tensor(self.total), val)
        self.count = ops.add(_to_tensor(self.count), _to_tensor(1.0))

    def result(self):
        """Compute the current metric value.

        Returns:
            A scalar tensor, or a dictionary of scalar tensors.

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

    def __init__(self, name="logcosh", dtype=None, **kwargs):
        """Function docstring.

        Args:
            name: Description.
            dtype: Description.
            kwargs: Description.
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
        num_classes,
        target_class_ids,
        name=None,
        dtype=None,
        ignore_class=None,
        sparse_y_true=True,
        sparse_y_pred=True,
        axis=-1,
        **kwargs,
    ):
        """Function docstring.

        Args:
            num_classes: Description.
            target_class_ids: Description.
            name: Description.
            dtype: Description.
            ignore_class: Description.
            sparse_y_true: Description.
            sparse_y_pred: Description.
            axis: Description.
            kwargs: Description.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.num_classes = num_classes
        self.target_class_ids = target_class_ids
        self.ignore_class = ignore_class
        self.sparse_y_true = sparse_y_true
        self.sparse_y_pred = sparse_y_pred
        self.axis = axis
        self.total_cm = 0.0

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Function docstring.

        Args:
            y_true: Description.
            y_pred: Description.
            sample_weight: Description.
        """
        from zero_keras.ops import ops
        from zero_keras.activations import _to_tensor

        y_true = ops.cast(_to_tensor(y_true), "int32")
        print("y_true init shape", y_true.shape)
        print("sparse_y_true", self.sparse_y_true)
        y_pred = ops.cast(_to_tensor(y_pred), "int32")

        if not self.sparse_y_true:
            y_true = ops.argmax(y_true, axis=self.axis)
        if not self.sparse_y_pred:
            y_pred = ops.argmax(y_pred, axis=self.axis)

        y_true = ops.reshape(y_true, [-1])
        y_pred = ops.reshape(y_pred, [-1])

        if self.ignore_class is not None:
            valid_mask = ops.logical_not(ops.equal(y_true, self.ignore_class))
            # This is hard because boolean masking is not trivial without boolean indexing.
            # But we can just zero out the one-hots where valid_mask is False.
        else:
            valid_mask = ops.ones_like(y_true)

        valid_mask = ops.cast(valid_mask, "float32")
        if sample_weight is not None:
            sample_weight = ops.reshape(
                ops.cast(_to_tensor(sample_weight), "float32"), [-1]
            )
            valid_mask = valid_mask * sample_weight

        print("y_true before one_hot", y_true.shape)
        y_true_one_hot = nn.one_hot(y_true, self.num_classes)
        y_pred_one_hot = nn.one_hot(y_pred, self.num_classes)

        # apply weights and mask
        y_true_one_hot = y_true_one_hot * ops.expand_dims(valid_mask, -1)

        # Transpose and matmul
        print(
            "y_true_one_hot",
            y_true_one_hot.shape,
            "y_pred_one_hot",
            y_pred_one_hot.shape,
        )
        cm = ops.sum(
            ops.expand_dims(y_true_one_hot, -1) * ops.expand_dims(y_pred_one_hot, 1),
            axis=0,
        )

        if hasattr(self.total_cm, "assign"):
            self.total_cm.assign(self.total_cm + cm)
        else:
            self.total_cm += cm

    def result(self):
        """Function docstring."""
        from zero_keras.ops import ops

        cm = self.total_cm
        sum_over_row = ops.sum(cm, axis=0)
        sum_over_col = ops.sum(cm, axis=1)

        # True positives are on the diagonal
        # Keras uses trace, but we don't have trace. We can just use one-hot multiply or extract diagonal.
        # Wait, how to extract diagonal without gather?
        # elementwise multiply with identity matrix!
        arange = ops.arange(self.num_classes)
        identity = ops.cast(
            ops.equal(ops.expand_dims(arange, 1), ops.expand_dims(arange, 0)), "float32"
        )
        true_positives = ops.sum(cm * identity, axis=1)

        denominator = sum_over_row + sum_over_col - true_positives

        iou = ops.where(
            ops.greater(denominator, 0.0), true_positives / denominator, _to_tensor(0.0)
        )

        # Filter by target_class_ids if provided
        if isinstance(self.target_class_ids, (list, tuple)):
            # gather the specific indices
            # we can use one-hot mask
            mask = ops.sum(
                nn.one_hot(ops.asarray(self.target_class_ids), self.num_classes), axis=0
            )
            iou = iou * mask
            num_valid = ops.cast(len(self.target_class_ids), "float32")
            return ops.sum(iou) / num_valid

        # If target_class_ids is not provided, mean over all classes
        return ops.mean(iou)

    def reset_state(self):
        """Function docstring."""
        self.total_cm = 0.0


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
        num_classes,
        name=None,
        dtype=None,
        ignore_class=None,
        sparse_y_true=True,
        sparse_y_pred=True,
        axis=-1,
        **kwargs,
    ):
        """Function docstring.

        Args:
            num_classes: Description.
            name: Description.
            dtype: Description.
            ignore_class: Description.
            sparse_y_true: Description.
            sparse_y_pred: Description.
            axis: Description.
            kwargs: Description.
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
        target_class_ids=(0, 1),
        threshold=0.5,
        name=None,
        dtype=None,
        **kwargs,
    ):
        """Function docstring.

        Args:
            target_class_ids: Description.
            threshold: Description.
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        super().__init__(
            num_classes=2,
            target_class_ids=target_class_ids,
            name=name,
            dtype=dtype,
            **kwargs,
        )
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Function docstring.

        Args:
            y_true: Description.
            y_pred: Description.
            sample_weight: Description.
        """
        from zero_keras.ops import ops
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
        num_classes,
        name=None,
        dtype=None,
        ignore_class=None,
        sparse_y_pred=False,
        axis=-1,
        **kwargs,
    ):
        """Function docstring.

        Args:
            num_classes: Description.
            name: Description.
            dtype: Description.
            ignore_class: Description.
            sparse_y_pred: Description.
            axis: Description.
            kwargs: Description.
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
        num_classes,
        target_class_ids,
        name=None,
        dtype=None,
        ignore_class=None,
        sparse_y_pred=False,
        axis=-1,
        **kwargs,
    ):
        """Function docstring.

        Args:
            num_classes: Description.
            target_class_ids: Description.
            name: Description.
            dtype: Description.
            ignore_class: Description.
            sparse_y_pred: Description.
            axis: Description.
            kwargs: Description.
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
    """Class docstring."""

    def __init__(self, name="concordancecorrelation", dtype=None, **kwargs):
        """Function docstring.

        Args:
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.reset_state()

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Function docstring.

        Args:
            y_true: Description.
            y_pred: Description.
            sample_weight: Description.
        """
        from zero_keras.ops import ops
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

        try:
            self.sum_x.assign(ops.add(self.sum_x, sx))
            self.sum_y.assign(ops.add(self.sum_y, sy))
            self.sum_x2.assign(ops.add(self.sum_x2, sx2))
            self.sum_y2.assign(ops.add(self.sum_y2, sy2))
            self.sum_xy.assign(ops.add(self.sum_xy, sxy))
            self.count.assign(ops.add(self.count, c))
        except AttributeError:
            self.sum_x += sx
            self.sum_y += sy
            self.sum_x2 += sx2
            self.sum_y2 += sy2
            self.sum_xy += sxy
            self.count += c

    def result(self):
        """Function docstring."""
        from zero_keras.ops import ops
        from zero_keras.activations import _wrap, _to_tensor

        n = ops.maximum(self.count, _to_tensor(1e-7))
        mean_x = self.sum_x / n
        mean_y = self.sum_y / n

        var_x = (self.sum_x2 / n) - (mean_x ** _to_tensor(2.0))
        var_y = (self.sum_y2 / n) - (mean_y ** _to_tensor(2.0))
        cov_xy = (self.sum_xy / n) - (mean_x * mean_y)

        ccc = (ops.array(2.0, dtype="float32") * cov_xy) / ops.maximum(
            var_x + var_y + ((mean_x - mean_y) ** _to_tensor(2.0)), _to_tensor(1e-7)
        )
        return _wrap(ops.mean(ccc))

    def reset_state(self):
        """Function docstring."""
        from zero_keras.activations import _to_tensor

        self.sum_x = _to_tensor(0.0)
        self.sum_y = _to_tensor(0.0)
        self.sum_x2 = _to_tensor(0.0)
        self.sum_y2 = _to_tensor(0.0)
        self.sum_xy = _to_tensor(0.0)
        self.count = _to_tensor(0.0)


class PearsonCorrelation(Metric):
    """Class docstring."""

    def __init__(self, name="pearsoncorrelation", dtype=None, **kwargs):
        """Function docstring.

        Args:
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.reset_state()

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Function docstring.

        Args:
            y_true: Description.
            y_pred: Description.
            sample_weight: Description.
        """
        from zero_keras.ops import ops
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

        try:
            self.sum_x.assign(ops.add(self.sum_x, sx))
            self.sum_y.assign(ops.add(self.sum_y, sy))
            self.sum_x2.assign(ops.add(self.sum_x2, sx2))
            self.sum_y2.assign(ops.add(self.sum_y2, sy2))
            self.sum_xy.assign(ops.add(self.sum_xy, sxy))
            self.count.assign(ops.add(self.count, c))
        except AttributeError:
            self.sum_x += sx
            self.sum_y += sy
            self.sum_x2 += sx2
            self.sum_y2 += sy2
            self.sum_xy += sxy
            self.count += c

    def result(self):
        """Function docstring."""
        from zero_keras.ops import ops
        from zero_keras.activations import _wrap, _to_tensor

        n = ops.maximum(self.count, _to_tensor(1e-7))
        mean_x = self.sum_x / n
        mean_y = self.sum_y / n

        var_x = (self.sum_x2 / n) - (mean_x ** _to_tensor(2.0))
        var_y = (self.sum_y2 / n) - (mean_y ** _to_tensor(2.0))
        cov_xy = (self.sum_xy / n) - (mean_x * mean_y)

        pcc = cov_xy / ops.maximum(ops.sqrt(var_x * var_y), _to_tensor(1e-7))
        return _wrap(ops.mean(pcc))

    def reset_state(self):
        """Function docstring."""
        from zero_keras.activations import _to_tensor

        self.sum_x = _to_tensor(0.0)
        self.sum_y = _to_tensor(0.0)
        self.sum_x2 = _to_tensor(0.0)
        self.sum_y2 = _to_tensor(0.0)
        self.sum_xy = _to_tensor(0.0)
        self.count = _to_tensor(0.0)


class FBetaScore(Metric):
    """Class docstring."""

    def __init__(
        self,
        average=None,
        beta=1.0,
        threshold=None,
        name="fbeta_score",
        dtype=None,
        **kwargs,
    ):
        """Function docstring.

        Args:
            average: Description.
            beta: Description.
            threshold: Description.
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        if average not in (None, "micro", "macro", "weighted"):
            raise ValueError(f"Invalid average: {average}")  # pragma: no cover
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.average = average
        self.beta = beta
        self.threshold = threshold
        self.reset_state()

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Function docstring.

        Args:
            y_true: Description.
            y_pred: Description.
            sample_weight: Description.
        """
        from zero_keras.ops import ops
        from zero_keras.activations import _to_tensor

        y_true = _to_tensor(y_true)
        y_pred = _to_tensor(y_pred)
        if sample_weight is not None:
            sample_weight = _to_tensor(sample_weight)

        if self.threshold is not None:
            y_pred = ops.cast(
                ops.greater(y_pred, _to_tensor(self.threshold)), y_true.dtype
            )
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

        try:
            self.tp.assign(ops.add(self.tp, tp))
            self.fp.assign(ops.add(self.fp, fp))
            self.fn.assign(ops.add(self.fn, fn))
        except AttributeError:
            self.tp += tp
            self.fp += fp
            self.fn += fn

    def result(self):
        """Function docstring."""
        from zero_keras.ops import ops
        from zero_keras.activations import _wrap, _to_tensor

        beta2 = _to_tensor(self.beta**2)
        p = self.tp / ops.maximum(self.tp + self.fp, _to_tensor(1e-7))
        r = self.tp / ops.maximum(self.tp + self.fn, _to_tensor(1e-7))
        f_beta = (
            (ops.add(_to_tensor(1.0), beta2))
            * p
            * r
            / ops.maximum(beta2 * p + r, _to_tensor(1e-7))
        )

        if self.average == "macro":
            return _wrap(ops.mean(f_beta))
        return _wrap(f_beta)

    def reset_state(self):
        """Function docstring."""
        from zero_keras.activations import _to_tensor

        self.tp = _to_tensor(0.0)
        self.fp = _to_tensor(0.0)
        self.fn = _to_tensor(0.0)


class F1Score(FBetaScore):
    """Class docstring."""

    def __init__(
        self, average=None, threshold=None, name="f1_score", dtype=None, **kwargs
    ):
        """Function docstring.

        Args:
            average: Description.
            threshold: Description.
            name: Description.
            dtype: Description.
            kwargs: Description.
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
    """Class docstring."""

    def __init__(
        self,
        class_aggregation="uniform_average",
        num_regressors=0,
        name="r2_score",
        dtype=None,
        **kwargs,
    ):
        """Function docstring.

        Args:
            class_aggregation: Description.
            num_regressors: Description.
            name: Description.
            dtype: Description.
            kwargs: Description.
        """
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.class_aggregation = class_aggregation
        self.num_regressors = num_regressors
        self.reset_state()

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Function docstring.

        Args:
            y_true: Description.
            y_pred: Description.
            sample_weight: Description.
        """
        from zero_keras.ops import ops
        from zero_keras.activations import _to_tensor

        y_true = _to_tensor(y_true)
        y_pred = _to_tensor(y_pred)
        w = _to_tensor(sample_weight) if sample_weight is not None else _to_tensor(1.0)

        ss = ops.sum((y_true ** _to_tensor(2.0)) * w, axis=0)
        su = ops.sum(y_true * w, axis=0)
        re = ops.sum(((y_true - y_pred) ** _to_tensor(2.0)) * w, axis=0)
        cn = ops.sum(w, axis=0)
        try:
            self.squared_sum.assign(ops.add(self.squared_sum, ss))
            self.sum.assign(ops.add(self.sum, su))
            self.res.assign(ops.add(self.res, re))
            self.count.assign(ops.add(self.count, cn))
        except AttributeError:
            self.squared_sum += ss
            self.sum += su
            self.res += re
            self.count += cn

    def result(self):
        """Function docstring."""
        from zero_keras.ops import ops
        from zero_keras.activations import _wrap, _to_tensor

        cnt = ops.maximum(self.count, _to_tensor(1e-7))
        self.sum / cnt
        total = self.squared_sum - (self.sum ** _to_tensor(2.0)) / cnt
        r2 = _to_tensor(1.0) - self.res / ops.maximum(total, _to_tensor(1e-7))
        if self.class_aggregation == "uniform_average":
            return _wrap(ops.mean(r2))
        elif self.class_aggregation == "variance_weighted_average":
            weight = total / ops.maximum(ops.sum(total), _to_tensor(1e-7))
            return _wrap(ops.sum(r2 * weight))
        return _wrap(r2)

    def reset_state(self):
        """Function docstring."""
        from zero_keras.activations import _to_tensor

        self.squared_sum = _to_tensor(0.0)
        self.sum = _to_tensor(0.0)
        self.res = _to_tensor(0.0)
        self.count = _to_tensor(0.0)


def serialize(metric):
    """Serialize a metric."""
    if metric is None:
        return None
    if isinstance(metric, str):
        return metric
    return {
        "class_name": metric.__class__.__name__,
        "config": metric.get_config() if hasattr(metric, "get_config") else {},
    }


def deserialize(config, custom_objects=None):
    """Deserialize a metric."""
    if config is None:
        return None
    if isinstance(config, str):
        return get(config)
    if isinstance(config, dict):
        class_name = config.get("class_name")
        conf = config.get("config", {})
        cls = globals().get(class_name)
        if cls:
            return cls(**conf)
    return config


def get(identifier):
    """Retrieve a Keras metric object via an identifier."""
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


def binary_accuracy(y_true, y_pred, threshold=0.5):
    """Function docstring.

    Args:
        y_true: Description.
        y_pred: Description.
        threshold: Description.
    """
    from zero_keras import ops

    y_true = ops.cast(y_true, y_pred.dtype)
    threshold = ops.cast(threshold, y_pred.dtype)
    y_pred_thresholded = ops.cast(y_pred > threshold, y_pred.dtype)
    return ops.mean(
        ops.cast(ops.equal(y_true, y_pred_thresholded), y_pred.dtype), axis=-1
    )


def categorical_accuracy(y_true, y_pred):
    """Function docstring.

    Args:
        y_true: Description.
        y_pred: Description.
    """
    from zero_keras import ops

    return ops.cast(
        ops.equal(ops.argmax(y_true, axis=-1), ops.argmax(y_pred, axis=-1)),
        y_pred.dtype,
    )


def sparse_categorical_accuracy(y_true, y_pred):
    """Function docstring.

    Args:
        y_true: Description.
        y_pred: Description.
    """
    from zero_keras import ops

    y_pred_rank = len(y_pred.shape)
    y_true_rank = len(y_true.shape)
    y_true = ops.squeeze(y_true, -1) if y_true_rank == y_pred_rank else y_true
    return ops.cast(
        ops.equal(
            ops.cast(y_true, "int32"), ops.cast(ops.argmax(y_pred, axis=-1), "int32")
        ),
        y_pred.dtype,
    )


def top_k_categorical_accuracy(y_true, y_pred, k=5):
    """Function docstring.

    Args:
        y_true: Description.
        y_pred: Description.
        k: Description.
    """
    from zero_keras import ops

    return ops.top_k_categorical_accuracy(y_true, y_pred, k=k)


def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
    """Function docstring.

    Args:
        y_true: Description.
        y_pred: Description.
        k: Description.
    """
    from zero_keras import ops

    return ops.sparse_top_k_categorical_accuracy(y_true, y_pred, k=k)
