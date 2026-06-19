"""Module docstring."""

import ml_switcheroo_compiler.ops as ops
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
            y_true = ops.cast(y_true, "bool")
            y_pred = ops.cast(y_pred, "bool")

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

    def __init__(self, name=None, dtype=None, **kwargs):
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
            # Find the value of the k-th top element
            # Actually, `numpy.argsort` is what we need. We can just use python/numpy for eager mode if we need to.
            # But ops.top_k is not implemented.

            if hasattr(y_pred, "data") and not hasattr(y_pred.data, "id"):
                # Eager mode
                np = __import__("numpy")
                np_pred = np.asarray(y_pred.data)
                np = __import__("numpy")
                top_indices = np.argsort(np_pred, axis=-1)[..., -k:]
                y_true_np = np.asarray(y_true_rank.data)[..., np.newaxis]
                matches = np.any(top_indices == y_true_np, axis=-1).astype(np.float32)

                return ops.asarray(matches)
            else:
                raise NotImplementedError(
                    "ops.top_k not implemented for symbolic tensors"
                )

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

            if hasattr(y_pred, "data") and not hasattr(y_pred.data, "id"):
                # Eager mode
                np = __import__("numpy")
                np_pred = np.asarray(y_pred.data)
                np = __import__("numpy")
                top_indices = np.argsort(np_pred, axis=-1)[..., -k:]
                y_true_np = np.asarray(y_true.data)[..., np.newaxis]
                matches = np.any(top_indices == y_true_np, axis=-1).astype(np.float32)
                from zero_keras.core_layers import KerasTensor

                return KerasTensor(matches.shape, "float32", data=matches)
            else:
                raise NotImplementedError(
                    "ops.top_k not implemented for symbolic tensors"
                )

        super().__init__(fn=sparse_top_k_fn, name=name, dtype=dtype, **kwargs)


class FalsePositives(Metric):
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
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass


class FalseNegatives(Metric):
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
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass


class TrueNegatives(Metric):
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
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass


class TruePositives(Metric):
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
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass


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
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass


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
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass


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
        self, recall, num_thresholds=200, class_id=None, name=None, dtype=None, **kwargs
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass


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
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass


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
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass


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
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass


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
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass


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
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        pass


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
        label_smoothing=0,
        **kwargs,
    ):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        pass


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
        from zero_keras.losses import log_cosh

        super().__init__(fn=log_cosh, name=name, dtype=dtype, **kwargs)


class MeanIoU(Metric):
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
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass


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
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass


class BinaryIoU(Metric):
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
        self, target_class_ids=(0, 1), threshold=0.5, name=None, dtype=None, **kwargs
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass


class OneHotMeanIoU(Metric):
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
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass


class OneHotIoU(Metric):
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
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass


class ConcordanceCorrelation(Metric):
    """Calculates the Concordance Correlation Coefficient (CCC).

    CCC evaluates the agreement between true values (`y_true`) and predicted
    values (`y_pred`) by considering both precision and accuracy. The
    coefficient ranges from -1 to 1, where a value of 1 indicates perfect
    agreement.

    This metric is useful in regression tasks where it is important to assess
    how well the predictions match the true values, taking into account both
    their correlation and proximity to the 45-degree line of perfect
    concordance.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        axis: (Optional) integer or tuple of integers of the axis/axes along
            which to compute the metric. Defaults to `-1`.

    Example:
    >>> ccc = keras.metrics.ConcordanceCorrelation(axis=-1)
    >>> y_true = [[0, 1, 0.5], [1, 1, 0.2]]
    >>> y_pred = [[0.1, 0.9, 0.5], [1, 0.9, 0.2]]
    >>> ccc.update_state(y_true, y_pred)
    >>> ccc.result()
    0.9816320385426076

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=[keras.metrics.ConcordanceCorrelation()])
    ```

    """

    def __init__(self, name="concordancecorrelation", dtype=None, **kwargs):
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


class PearsonCorrelation(Metric):
    """Calculates the Pearson Correlation Coefficient (PCC).

    PCC measures the linear relationship between the true values (`y_true`) and
    the predicted values (`y_pred`). The coefficient ranges from -1 to 1, where
    a value of 1 implies a perfect positive linear correlation, 0 indicates no
    linear correlation, and -1 indicates a perfect negative linear correlation.

    This metric is widely used in regression tasks where the strength of the
    linear relationship between predictions and true labels is an
    important evaluation criterion.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        axis: (Optional) integer or tuple of integers of the axis/axes along
            which to compute the metric. Defaults to `-1`.

    Example:
    >>> pcc = keras.metrics.PearsonCorrelation(axis=-1)
    >>> y_true = [[0, 1, 0.5], [1, 1, 0.2]]
    >>> y_pred = [[0.1, 0.9, 0.5], [1, 0.9, 0.2]]
    >>> pcc.update_state(y_true, y_pred)
    >>> pcc.result()
    0.9966996338993913

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=[keras.metrics.PearsonCorrelation()])
    ```

    """

    def __init__(self, name="pearsoncorrelation", dtype=None, **kwargs):
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


class F1Score(Metric):
    """Computes F-1 Score.

    Formula:

    ```python
    f1_score = 2 * (precision * recall) / (precision + recall)
    ```
    This is the harmonic mean of precision and recall.
    Its output range is `[0, 1]`. It works for both multi-class
    and multi-label classification.

    Args:
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `"micro"`, `"macro"`
            and `"weighted"`. Defaults to `None`.
            If `None`, no averaging is performed and `result()` will return
            the score for each class.
            If `"micro"`, compute metrics globally by counting the total
            true positives, false negatives and false positives.
            If `"macro"`, compute metrics for each label,
            and return their unweighted mean.
            This does not take label imbalance into account.
            If `"weighted"`, compute metrics for each label,
            and return their average weighted by support
            (the number of true instances for each label).
            This alters `"macro"` to account for label imbalance.
            It can result in an score that is not between precision and recall.
        threshold: Elements of `y_pred` greater than `threshold` are
            converted to be 1, and the rest 0. If `threshold` is
            `None`, the argmax of `y_pred` is converted to 1, and the rest to 0.
        name: Optional. String name of the metric instance.
        dtype: Optional. Data type of the metric result.

    Returns:
        F-1 Score: float.

    Example:
    >>> metric = keras.metrics.F1Score(threshold=0.5)
    >>> y_true = np.array([[1, 1, 1],
    ...                    [1, 0, 0],
    ...                    [1, 1, 0]], np.int32)
    >>> y_pred = np.array([[0.2, 0.6, 0.7],
    ...                    [0.2, 0.6, 0.6],
    ...                    [0.6, 0.8, 0.0]], np.float32)
    >>> metric.update_state(y_true, y_pred)
    >>> result = metric.result()
    array([0.5      , 0.8      , 0.6666667], dtype=float32)

    """

    def __init__(
        self, average=None, threshold=None, name="f1_score", dtype=None, **kwargs
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass


class FBetaScore(Metric):
    """Computes F-Beta score.

    Formula:

    ```python
    b2 = beta ** 2
    f_beta_score = (1 + b2) * (precision * recall) / (precision * b2 + recall)
    ```
    This is the weighted harmonic mean of precision and recall.
    Its output range is `[0, 1]`. It works for both multi-class
    and multi-label classification.

    Args:
        average: Type of averaging to be performed across per-class results
            in the multi-class case.
            Acceptable values are `None`, `"micro"`, `"macro"` and
            `"weighted"`. Defaults to `None`.
            If `None`, no averaging is performed and `result()` will return
            the score for each class.
            If `"micro"`, compute metrics globally by counting the total
            true positives, false negatives and false positives.
            If `"macro"`, compute metrics for each label,
            and return their unweighted mean.
            This does not take label imbalance into account.
            If `"weighted"`, compute metrics for each label,
            and return their average weighted by support
            (the number of true instances for each label).
            This alters `"macro"` to account for label imbalance.
            It can result in an score that is not between precision and recall.
        beta: Determines the weight of given to recall
            in the harmonic mean between precision and recall (see pseudocode
            equation above). Defaults to `1`.
        threshold: Elements of `y_pred` greater than `threshold` are
            converted to be 1, and the rest 0. If `threshold` is
            `None`, the argmax of `y_pred` is converted to 1, and the rest to 0.
        name: Optional. String name of the metric instance.
        dtype: Optional. Data type of the metric result.

    Returns:
        F-Beta Score: float.

    Example:
    >>> metric = keras.metrics.FBetaScore(beta=2.0, threshold=0.5)
    >>> y_true = np.array([[1, 1, 1],
    ...                    [1, 0, 0],
    ...                    [1, 1, 0]], np.int32)
    >>> y_pred = np.array([[0.2, 0.6, 0.7],
    ...                    [0.2, 0.6, 0.6],
    ...                    [0.6, 0.8, 0.0]], np.float32)
    >>> metric.update_state(y_true, y_pred)
    >>> result = metric.result()
    >>> result
    [0.3846154 , 0.90909094, 0.8333334 ]

    """

    def __init__(
        self,
        average=None,
        beta=1,
        threshold=None,
        name="fbeta_score",
        dtype=None,
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass


class R2Score(Metric):
    """Computes R2 score.

    Formula:

    ```python
    sum_squares_residuals = sum((y_true - y_pred) ** 2)
    sum_squares = sum((y_true - mean(y_true)) ** 2)
    R2 = 1 - sum_squares_residuals / sum_squares
    ```

    This is also called the
    [coefficient of determination](
    https://en.wikipedia.org/wiki/Coefficient_of_determination).

    It indicates how close the fitted regression line
    is to ground-truth data.

    - The highest score possible is 1.0. It indicates that the predictors
        perfectly accounts for variation in the target.
    - A score of 0.0 indicates that the predictors do not
        account for variation in the target.
    - It can also be negative if the model is worse than random.

    This metric can also compute the "Adjusted R2" score.

    Args:
        class_aggregation: Specifies how to aggregate scores corresponding to
            different output classes (or target dimensions),
            i.e. different dimensions on the last axis of the predictions.
            Equivalent to `multioutput` argument in Scikit-Learn.
            Should be one of
            `None` (no aggregation), `"uniform_average"`,
            `"variance_weighted_average"`.
        num_regressors: Number of independent regressors used
            ("Adjusted R2" score). 0 is the standard R2 score.
            Defaults to `0`.
        name: Optional. string name of the metric instance.
        dtype: Optional. data type of the metric result.

    Example:
    >>> y_true = np.array([[1], [4], [3]], dtype=np.float32)
    >>> y_pred = np.array([[2], [4], [4]], dtype=np.float32)
    >>> metric = keras.metrics.R2Score()
    >>> metric.update_state(y_true, y_pred)
    >>> result = metric.result()
    >>> result
    0.57142854

    """

    def __init__(
        self,
        class_aggregation="uniform_average",
        num_regressors=0,
        name="r2_score",
        dtype=None,
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass
