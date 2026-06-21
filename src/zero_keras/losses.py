"""Keras losses."""

import ml_switcheroo_compiler.ops as ops
from .activations import _to_tensor, _wrap
from typing import Any, Optional


def _reduce(loss: Any, reduction: str, sample_weight: Optional[Any] = None) -> Any:
    """_reduce function.

    Args:
    loss: Parameter loss.
    reduction: Parameter reduction.
    sample_weight: Parameter sample_weight.

    Returns:
    Any: Return value.

    """
    loss = _to_tensor(loss)
    if sample_weight is not None:
        sample_weight = _to_tensor(sample_weight)
        loss = ops.multiply(loss, sample_weight)

    if reduction == "none":
        return _wrap(loss)
    if reduction == "sum":
        return _wrap(ops.sum(loss))
    return _wrap(ops.mean(loss))


def mean_squared_error(y_true, y_pred):
    """Computes the mean squared error between labels and predictions.

    Formula:

    ```python
    loss = mean(square(y_true - y_pred), axis=-1)
    ```

    Example:
    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras.losses.mean_squared_error(y_true, y_pred)

    Args:
        y_true: Ground truth values with shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Mean squared error values with shape = `[batch_size, d0, .. dN-1]`.

    """
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
    return ops.mean(ops.square(y_true - y_pred), axis=-1)


def mean_absolute_error(y_true, y_pred):
    """Computes the mean absolute error between labels and predictions.

    ```python
    loss = mean(abs(y_true - y_pred), axis=-1)
    ```

    Args:
        y_true: Ground truth values with shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Mean absolute error values with shape = `[batch_size, d0, .. dN-1]`.

    Example:
    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras.losses.mean_absolute_error(y_true, y_pred)

    """
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
    return ops.mean(ops.abs(y_true - y_pred), axis=-1)


def mean_absolute_percentage_error(y_true, y_pred):
    """Computes the mean absolute percentage error between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = 100 * mean(abs((y_true - y_pred) / y_true), axis=-1)
    ```

    Division by zero is prevented by dividing by `maximum(y_true, epsilon)`
    where `epsilon = keras.backend.epsilon()`
    (default to `1e-7`).

    Args:
        y_true: Ground truth values with shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Mean absolute percentage error values with shape = `[batch_size, d0, ..
        dN-1]`.

    Example:
    >>> y_true = np.random.random(size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras.losses.mean_absolute_percentage_error(y_true, y_pred)

    """
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
    diff = ops.abs((y_true - y_pred) / ops.maximum(ops.abs(y_true), 1e-7))
    return 100.0 * ops.mean(diff, axis=-1)


def mean_squared_logarithmic_error(y_true, y_pred):
    """Computes the mean squared logarithmic error between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = mean(square(log(y_true + 1) - log(y_pred + 1)), axis=-1)
    ```

    Note that `y_pred` and `y_true` cannot be less or equal to 0. Negative
    values and 0 values will be replaced with `keras.backend.epsilon()`
    (default to `1e-7`).

    Args:
        y_true: Ground truth values with shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Mean squared logarithmic error values with shape = `[batch_size, d0, ..
        dN-1]`.

    Example:
    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras.losses.mean_squared_logarithmic_error(y_true, y_pred)

    """
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
    first_log = ops.log(ops.maximum(y_pred, 0.0) + 1.0)
    second_log = ops.log(ops.maximum(y_true, 0.0) + 1.0)
    return ops.mean(ops.square(first_log - second_log), axis=-1)


def huber(y_true, y_pred, delta=1.0):
    """Computes Huber loss value.

    Formula:
    ```python
    for x in error:
        if abs(x) <= delta:
            loss.append(0.5 * x^2)
        elif abs(x) > delta:
            loss.append(delta * abs(x) - 0.5 * delta^2)

    loss = mean(loss, axis=-1)
    ```
    See: [Huber loss](https://en.wikipedia.org/wiki/Huber_loss).

    Example:
    >>> y_true = [[0, 1], [0, 0]]
    >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
    >>> loss = keras.losses.huber(y_true, y_pred)
    0.155


    Args:
        y_true: tensor of true targets.
        y_pred: tensor of predicted targets.
        delta: A float, the point where the Huber loss function changes from a
            quadratic to linear. Defaults to `1.0`.

    Returns:
        Tensor with one scalar loss entry per sample.

    """
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
    error = y_true - y_pred
    abs_error = ops.abs(error)
    quadratic = ops.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return ops.mean(0.5 * ops.square(quadratic) + delta * linear, axis=-1)


def log_cosh(y_true, y_pred):
    """Logarithm of the hyperbolic cosine of the prediction error.

    Formula:
    ```python
    loss = mean(log(cosh(y_pred - y_true)), axis=-1)
    ```

    Note that `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small
    `x` and to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works
    mostly like the mean squared error, but will not be so strongly affected by
    the occasional wildly incorrect prediction.

    Example:
    >>> y_true = [[0., 1.], [0., 0.]]
    >>> y_pred = [[1., 1.], [0., 0.]]
    >>> loss = keras.losses.log_cosh(y_true, y_pred)
    0.108

    Args:
        y_true: Ground truth values with shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Logcosh error values with shape = `[batch_size, d0, .. dN-1]`.

    """
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
    x = y_pred - y_true
    return ops.mean(
        x + ops.logaddexp(ops.zeros_like(x), -2.0 * x) - ops.log(ops.full_like(x, 2.0)),
        axis=-1,
    )


def hinge(y_true, y_pred):
    """Computes the hinge loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = mean(maximum(1 - y_true * y_pred, 0), axis=-1)
    ```

    Args:
        y_true: The ground truth values. `y_true` values are expected to be -1
            or 1. If binary (0 or 1) labels are provided they will be converted
            to -1 or 1 with shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Hinge loss values with shape = `[batch_size, d0, .. dN-1]`.

    Example:
    >>> y_true = np.random.choice([-1, 1], size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras.losses.hinge(y_true, y_pred)

    """
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
    return ops.mean(ops.maximum(1.0 - y_true * y_pred, 0.0), axis=-1)


def squared_hinge(y_true, y_pred):
    """Computes the squared hinge loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = mean(square(maximum(1 - y_true * y_pred, 0)), axis=-1)
    ```

    Args:
        y_true: The ground truth values. `y_true` values are expected to be -1
            or 1. If binary (0 or 1) labels are provided we will convert them
            to -1 or 1 with shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Squared hinge loss values with shape = `[batch_size, d0, .. dN-1]`.

    Example:
    >>> y_true = np.random.choice([-1, 1], size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras.losses.squared_hinge(y_true, y_pred)

    """
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
    return ops.mean(ops.square(ops.maximum(1.0 - y_true * y_pred, 0.0)), axis=-1)


def categorical_hinge(y_true, y_pred):
    """Computes the categorical hinge loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = maximum(neg - pos + 1, 0)
    ```

    where `neg=maximum((1-y_true)*y_pred)` and `pos=sum(y_true*y_pred)`

    Args:
        y_true: The ground truth values. `y_true` values are expected to be
            either `{-1, +1}` or `{0, 1}` (i.e. a one-hot-encoded tensor) with
            shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Categorical hinge loss values with shape = `[batch_size, d0, .. dN-1]`.

    Example:
    >>> y_true = np.random.randint(0, 3, size=(2,))
    >>> y_true = np.eye(np.max(y_true) + 1)[y_true]
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras.losses.categorical_hinge(y_true, y_pred)

    """
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
    pos = ops.sum(y_true * y_pred, axis=-1)
    neg = ops.max((1.0 - y_true) * y_pred, axis=-1)
    return ops.maximum(neg - pos + 1.0, 0.0)


def poisson(y_true, y_pred):
    """Computes the Poisson loss between y_true and y_pred.

    Formula:

    ```python
    loss = y_pred - y_true * log(y_pred)
    ```

    Args:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

    Returns:
        Poisson loss values with shape = `[batch_size, d0, .. dN-1]`.

    Example:
    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras.losses.poisson(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> y_pred = y_pred + 1e-7
    >>> assert np.allclose(
    ...     loss, np.mean(y_pred - y_true * np.log(y_pred), axis=-1),
    ...     atol=1e-5)

    """
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
    return ops.mean(y_pred - y_true * ops.log(y_pred + 1e-7), axis=-1)


def kl_divergence(y_true, y_pred):
    """Computes Kullback-Leibler divergence loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = y_true * log(y_true / y_pred)
    ```

    `y_true` and `y_pred` are expected to be probability
    distributions, with values between 0 and 1. They will get
    clipped to the `[0, 1]` range.

    Args:
        y_true: Tensor of true targets.
        y_pred: Tensor of predicted targets.

    Returns:
        KL Divergence loss values with shape = `[batch_size, d0, .. dN-1]`.

    Example:
    >>> y_true = np.random.randint(0, 2, size=(2, 3)).astype(np.float32)
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras.losses.kl_divergence(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> y_true = ops.clip(y_true, 1e-7, 1)
    >>> y_pred = ops.clip(y_pred, 1e-7, 1)
    >>> assert np.array_equal(
    ...     loss, np.sum(y_true * np.log(y_true / y_pred), axis=-1))

    """
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
    y_true = ops.maximum(y_true, 1e-7)
    y_pred = ops.maximum(y_pred, 1e-7)
    return ops.sum(y_true * ops.log(y_true / y_pred), axis=-1)


def cosine_similarity(y_true, y_pred, axis=-1):
    """Computes the cosine similarity between labels and predictions.

    Formula:
    ```python
    loss = -sum(l2_norm(y_true) * l2_norm(y_pred))
    ```

    Note that it is a number between -1 and 1. When it is a negative number
    between -1 and 0, 0 indicates orthogonality and values closer to -1
    indicate greater similarity. This makes it usable as a loss function in a
    setting where you try to maximize the proximity between predictions and
    targets. If either `y_true` or `y_pred` is a zero vector, cosine
    similarity will be 0 regardless of the proximity between predictions
    and targets.

    Args:
        y_true: Tensor of true targets.
        y_pred: Tensor of predicted targets.
        axis: Axis along which to determine similarity. Defaults to `-1`.

    Returns:
        Cosine similarity tensor.

    Example:
    >>> y_true = [[0., 1.], [1., 1.], [1., 1.]]
    >>> y_pred = [[1., 0.], [1., 1.], [-1., -1.]]
    >>> loss = keras.losses.cosine_similarity(y_true, y_pred, axis=-1)
    [-0., -0.99999994, 0.99999994]

    """
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
    y_true_norm = ops.sqrt(ops.maximum(ops.sum(ops.square(y_true), axis=axis), 1e-7))
    y_pred_norm = ops.sqrt(ops.maximum(ops.sum(ops.square(y_pred), axis=axis), 1e-7))
    return -ops.sum(y_true * y_pred, axis=axis) / (y_true_norm * y_pred_norm)


def binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0.0):
    """Computes the binary crossentropy loss.

    Args:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
        label_smoothing: Float in `[0, 1]`. If > `0` then smooth the labels by
            squeezing them towards 0.5, that is,
            using `1. - 0.5 * label_smoothing` for the target class
            and `0.5 * label_smoothing` for the non-target class.
        axis: The axis along which the mean is computed. Defaults to `-1`.

    Returns:
        Binary crossentropy loss value. shape = `[batch_size, d0, .. dN-1]`.

    Example:
    >>> y_true = [[0, 1], [0, 0]]
    >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
    >>> loss = keras.losses.binary_crossentropy(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss
    array([0.916 , 0.714], dtype=float32)

    """
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
    if label_smoothing > 0.0:
        y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    if from_logits:
        return ops.mean(
            ops.maximum(y_pred, 0.0)
            - y_pred * y_true
            + ops.logaddexp(ops.zeros_like(y_pred), -ops.abs(y_pred)),
            axis=-1,
        )

    y_pred = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
    bce = y_true * ops.log(y_pred) + (1.0 - y_true) * ops.log(1.0 - y_pred)
    return -ops.mean(bce, axis=-1)


def categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0.0):
    """Computes the categorical crossentropy loss.

    Args:
        y_true: Tensor of one-hot true targets.
        y_pred: Tensor of predicted targets.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
        label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. For
            example, if `0.1`, use `0.1 / num_classes` for non-target labels
            and `0.9 + 0.1 / num_classes` for target labels.
        axis: Defaults to `-1`. The dimension along which the entropy is
            computed.

    Returns:
        Categorical crossentropy loss value.

    Example:
    >>> y_true = [[0, 1, 0], [0, 0, 1]]
    >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    >>> loss = keras.losses.categorical_crossentropy(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss
    array([0.0513, 2.303], dtype=float32)

    """
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
    if label_smoothing > 0.0:
        num_classes = ops.cast(
            ops.full_like(y_pred, y_pred.shape[-1]), dtype=y_pred.dtype
        )
        y_true = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

    if from_logits:
        m = ops.max(y_pred, axis=-1, keepdims=True)
        e = ops.exp(y_pred - m)
        s = ops.sum(e, axis=-1, keepdims=True)
        log_softmax = y_pred - m - ops.log(s)
        return -ops.sum(y_true * log_softmax, axis=-1)

    y_pred = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
    return -ops.sum(y_true * ops.log(y_pred), axis=-1)


def sparse_categorical_crossentropy(
    y_true, y_pred, from_logits=False, ignore_class=None
):
    """Computes the sparse categorical crossentropy loss.

    Args:
        y_true: Ground truth values.
        y_pred: The predicted values.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
        ignore_class: Optional integer. The ID of a class to be ignored during
            loss computation. This is useful, for example, in segmentation
            problems featuring a "void" class (commonly -1 or 255) in
            segmentation maps. By default (`ignore_class=None`), all classes are
            considered.
        axis: Defaults to `-1`. The dimension along which the entropy is
            computed.

    Returns:
        Sparse categorical crossentropy loss value.

    Examples:
    >>> y_true = [1, 2]
    >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    >>> loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss
    array([0.0513, 2.303], dtype=float32)

    """
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
    y_true_int = ops.cast(ops.expand_dims(y_true, -1), dtype="int32")
    if from_logits:
        m = ops.max(y_pred, axis=-1, keepdims=True)
        e = ops.exp(y_pred - m)
        s = ops.sum(e, axis=-1, keepdims=True)
        log_softmax = y_pred - m - ops.log(s)
        return -ops.mean(ops.take_along_axis(log_softmax, y_true_int, -1), axis=-1)

    y_pred = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
    prob = ops.take_along_axis(y_pred, y_true_int, -1)
    return -ops.log(ops.mean(prob, axis=-1))


def binary_focal_crossentropy(
    y_true, y_pred, alpha=0.25, gamma=2.0, from_logits=False, label_smoothing=0.0
):
    """Computes the binary focal crossentropy loss.

    According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
    helps to apply a focal factor to down-weight easy examples and focus more on
    hard examples. By default, the focal tensor is computed as follows:

    `focal_factor = (1 - output) ** gamma` for class 1
    `focal_factor = output ** gamma` for class 0
    where `gamma` is a focusing parameter. When `gamma` = 0, there is no focal
    effect on the binary crossentropy loss.

    If `apply_class_balancing == True`, this function also takes into account a
    weight balancing factor for the binary classes 0 and 1 as follows:

    `weight = alpha` for class 1 (`target == 1`)
    `weight = 1 - alpha` for class 0
    where `alpha` is a float in the range of `[0, 1]`.

    Args:
        y_true: Ground truth values, of shape `(batch_size, d0, .. dN)`.
        y_pred: The predicted values, of shape `(batch_size, d0, .. dN)`.
        apply_class_balancing: A bool, whether to apply weight balancing on the
            binary classes 0 and 1.
        alpha: A weight balancing factor for class 1, default is `0.25` as
            mentioned in the reference. The weight for class 0 is `1.0 - alpha`.
        gamma: A focusing parameter, default is `2.0` as mentioned in the
            reference.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
        label_smoothing: Float in `[0, 1]`. If > `0` then smooth the labels by
            squeezing them towards 0.5, that is,
            using `1. - 0.5 * label_smoothing` for the target class
            and `0.5 * label_smoothing` for the non-target class.
        axis: The axis along which the mean is computed. Defaults to `-1`.

    Returns:
        Binary focal crossentropy loss value
        with shape = `[batch_size, d0, .. dN-1]`.

    Example:
    >>> y_true = [[0, 1], [0, 0]]
    >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
    >>> # In this instance, the first sample in the second batch is the
    >>> # 'easier' example.
    >>> focal_loss = keras.losses.binary_focal_crossentropy(
    ...        y_true, y_pred, gamma=2)
    >>> assert loss.shape == (2,)
    >>> focal_loss
    array([0.330, 0.206], dtype=float32)
    >>> # Compare with binary_crossentropy
    >>> bce_loss = keras.losses.binary_focal_crossentropy(
    ...        y_true, y_pred)
    >>> bce_loss
    array([0.916, 0.714], dtype=float32)
    >>> # Binary focal crossentropy loss attributes more importance to the
    >>> # harder example which results in a higher loss for the first batch
    >>> # when normalized by binary cross entropy loss
    >>> focal_loss/bce_loss
    array([0.360, 0.289]

    """
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
    if label_smoothing > 0.0:
        y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    if from_logits:
        y_pred_prob = 1.0 / (1.0 + ops.exp(-y_pred))
        bce = (
            ops.maximum(y_pred, 0.0)
            - y_pred * y_true
            + ops.logaddexp(ops.zeros_like(y_pred), -ops.abs(y_pred))
        )
    else:
        y_pred_prob = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -(
            y_true * ops.log(y_pred_prob) + (1.0 - y_true) * ops.log(1.0 - y_pred_prob)
        )

    p_t = y_true * y_pred_prob + (1.0 - y_true) * (1.0 - y_pred_prob)
    modulating_factor = ops.power(1.0 - p_t, gamma)

    if alpha is not None:
        alpha_factor = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        focal_loss = modulating_factor * alpha_factor * bce
    else:
        focal_loss = modulating_factor * bce

    return ops.mean(focal_loss, axis=-1)


def categorical_focal_crossentropy(
    y_true, y_pred, alpha=0.25, gamma=2.0, from_logits=False, label_smoothing=0.0
):
    """Computes the categorical focal crossentropy loss.

    Args:
        y_true: Tensor of one-hot true targets.
        y_pred: Tensor of predicted targets.
        alpha: A weight balancing factor for all classes, default is `0.25` as
            mentioned in the reference. It can be a list of floats or a scalar.
            In the multi-class case, alpha may be set by inverse class
            frequency by using `compute_class_weight` from `sklearn.utils`.
        gamma: A focusing parameter, default is `2.0` as mentioned in the
            reference. It helps to gradually reduce the importance given to
            simple examples in a smooth manner. When `gamma` = 0, there is
            no focal effect on the categorical crossentropy.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability
            distribution.
        label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. For
            example, if `0.1`, use `0.1 / num_classes` for non-target labels
            and `0.9 + 0.1 / num_classes` for target labels.
        axis: Defaults to `-1`. The dimension along which the entropy is
            computed.

    Returns:
        Categorical focal crossentropy loss value.

    Example:
    >>> y_true = [[0, 1, 0], [0, 0, 1]]
    >>> y_pred = [[0.05, 0.9, 0.05], [0.1, 0.85, 0.05]]
    >>> loss = keras.losses.categorical_focal_crossentropy(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss
    array([2.63401289e-04, 6.75912094e-01], dtype=float32)

    """
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
    if label_smoothing > 0.0:
        num_classes = ops.cast(
            ops.full_like(y_pred, y_pred.shape[-1]), dtype=y_pred.dtype
        )
        y_true = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

    if from_logits:
        m = ops.max(y_pred, axis=-1, keepdims=True)
        e = ops.exp(y_pred - m)
        s = ops.sum(e, axis=-1, keepdims=True)
        log_softmax = y_pred - m - ops.log(s)
        cce = -y_true * log_softmax
        y_pred_prob = ops.exp(log_softmax)
    else:
        y_pred_prob = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
        cce = -y_true * ops.log(y_pred_prob)

    focal_loss = alpha * ops.power(1.0 - y_pred_prob, gamma) * cce
    return ops.sum(focal_loss, axis=-1)


class Loss:
    """Loss base class.

    This is the class to subclass in order to create new custom losses.

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    To be implemented by subclasses:

    * `call()`: Contains the logic for loss calculation using `y_true`,
        `y_pred`.

    Example subclass implementation:

    ```python
    class MeanSquaredError(Loss):
        def call(self, y_true, y_pred):
            return ops.mean(ops.square(y_pred - y_true), axis=-1)
    ```

    """

    def __init__(
        self,
        reduction: str = "sum_over_batch_size",
        name: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        self.reduction = reduction
        self.name = name
        self.dtype = dtype

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        """Call self as a function."""
        return _reduce(_to_tensor(0.0), self.reduction, sample_weight)


class BinaryCrossentropy(Loss):
    """Computes the cross-entropy loss between true labels and predicted labels.

    Use this cross-entropy loss for binary (0 or 1) classification applications.
    The loss function requires the following inputs:

    - `y_true` (true label): This is either 0 or 1.
    - `y_pred` (predicted value): This is the model's prediction, i.e, a single
        floating-point value which either represents a
        [logit](https://en.wikipedia.org/wiki/Logit), (i.e, value in [-inf, inf]
        when `from_logits=True`) or a probability (i.e, value in [0., 1.] when
        `from_logits=False`).

    Args:
        from_logits: Whether to interpret `y_pred` as a tensor of
            [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
            assume that `y_pred` is probabilities (i.e., values in [0, 1]).
        label_smoothing: Float in range [0, 1]. When 0, no smoothing occurs.
            When > 0, we compute the loss between the predicted labels
            and a smoothed version of the true labels, where the smoothing
            squeezes the labels towards 0.5. Larger values of
            `label_smoothing` correspond to heavier smoothing.
        axis: The axis along which to compute crossentropy (the features axis).
            Defaults to `-1`.
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    Examples:
    **Recommended Usage:** (set `from_logits=True`)

    With `compile()` API:

    ```python
    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        ...
    )
    ```

    As a standalone function:

    >>> # Example 1: (batch_size = 1, number of samples = 4)
    >>> y_true = np.array([0, 1, 0, 0])
    >>> y_pred = np.array([-18.6, 0.51, 2.94, -12.8])
    >>> bce = keras.losses.BinaryCrossentropy(from_logits=True)
    >>> bce(y_true, y_pred)
    0.8654

    >>> # Example 2: (batch_size = 2, number of samples = 4)
    >>> y_true = np.array([[0, 1], [0, 0]])
    >>> y_pred = np.array([[-18.6, 0.51], [2.94, -12.8]])
    >>> # Using default 'auto'/'sum_over_batch_size' reduction type.
    >>> bce = keras.losses.BinaryCrossentropy(from_logits=True)
    >>> bce(y_true, y_pred)
    0.8654
    >>> # Using 'sample_weight' attribute
    >>> bce(y_true, y_pred, sample_weight=[0.8, 0.2])
    0.243
    >>> # Using 'sum' reduction` type.
    >>> bce = keras.losses.BinaryCrossentropy(from_logits=True,
    ...     reduction="sum")
    >>> bce(y_true, y_pred)
    1.730
    >>> # Using 'none' reduction type.
    >>> bce = keras.losses.BinaryCrossentropy(from_logits=True,
    ...     reduction=None)
    >>> bce(y_true, y_pred)
    array([0.235, 1.496], dtype=float32)

    **Default Usage:** (set `from_logits=False`)

    >>> # Make the following updates to the above "Recommended Usage" section
    >>> # 1. Set `from_logits=False`
    >>> keras.losses.BinaryCrossentropy() # OR ...('from_logits=False')
    >>> # 2. Update `y_pred` to use probabilities instead of logits
    >>> y_pred = [0.6, 0.3, 0.2, 0.8] # OR [[0.6, 0.3], [0.2, 0.8]]

    """

    def __init__(
        self,
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction="sum_over_batch_size",
        name="binary_crossentropy",
        **kwargs,
    ):
        super().__init__(reduction=reduction, name=name)
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.axis = axis

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Call self as a function."""
        loss = binary_crossentropy(
            y_true,
            y_pred,
            from_logits=self.from_logits,
            label_smoothing=self.label_smoothing,
        )
        return _reduce(loss, self.reduction, sample_weight)


class MeanSquaredError(Loss):
    """Computes the mean of squares of errors between labels and predictions.

    Formula:

    ```python
    loss = mean(square(y_true - y_pred))
    ```

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    """

    def __init__(
        self, reduction="sum_over_batch_size", name="mean_squared_error", **kwargs
    ):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Call self as a function."""
        loss = mean_squared_error(y_true, y_pred)
        return _reduce(loss, self.reduction, sample_weight)


class BinaryFocalCrossentropy(Loss):
    """Computes focal cross-entropy loss between true labels and predictions.

    Binary cross-entropy loss is often used for binary (0 or 1) classification
    tasks. The loss function requires the following inputs:

    - `y_true` (true label): This is either 0 or 1.
    - `y_pred` (predicted value): This is the model's prediction, i.e, a single
        floating-point value which either represents a
        [logit](https://en.wikipedia.org/wiki/Logit), (i.e, value in [-inf, inf]
        when `from_logits=True`) or a probability (i.e, value in `[0., 1.]` when
        `from_logits=False`).

    According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
    helps to apply a "focal factor" to down-weight easy examples and focus more
    on hard examples. By default, the focal tensor is computed as follows:

    `focal_factor = (1 - output) ** gamma` for class 1
    `focal_factor = output ** gamma` for class 0
    where `gamma` is a focusing parameter. When `gamma=0`, this function is
    equivalent to the binary crossentropy loss.

    Args:
        apply_class_balancing: A bool, whether to apply weight balancing on the
            binary classes 0 and 1.
        alpha: A weight balancing factor for class 1, default is `0.25` as
            mentioned in reference [Lin et al., 2018](
            https://arxiv.org/pdf/1708.02002.pdf).  The weight for class 0 is
            `1.0 - alpha`.
        gamma: A focusing parameter used to compute the focal factor, default is
            `2.0` as mentioned in the reference
            [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf).
        from_logits: Whether to interpret `y_pred` as a tensor of
            [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
            assume that `y_pred` are probabilities (i.e., values in `[0, 1]`).
        label_smoothing: Float in `[0, 1]`. When `0`, no smoothing occurs.
            When > `0`, we compute the loss between the predicted labels
            and a smoothed version of the true labels, where the smoothing
            squeezes the labels towards `0.5`.
            Larger values of `label_smoothing` correspond to heavier smoothing.
        axis: The axis along which to compute crossentropy (the features axis).
            Defaults to `-1`.
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    Examples:
    With the `compile()` API:

    ```python
    model.compile(
        loss=keras.losses.BinaryFocalCrossentropy(
            gamma=2.0, from_logits=True),
        ...
    )
    ```

    As a standalone function:

    >>> # Example 1: (batch_size = 1, number of samples = 4)
    >>> y_true = np.array([0, 1, 0, 0])
    >>> y_pred = np.array([-18.6, 0.51, 2.94, -12.8])
    >>> loss = keras.losses.BinaryFocalCrossentropy(
    ...    gamma=2, from_logits=True)
    >>> loss(y_true, y_pred)
    0.691

    >>> # Apply class weight
    >>> loss = keras.losses.BinaryFocalCrossentropy(
    ...     apply_class_balancing=True, gamma=2, from_logits=True)
    >>> loss(y_true, y_pred)
    0.51

    >>> # Example 2: (batch_size = 2, number of samples = 4)
    >>> y_true = np.array([[0, 1], [0, 0]])
    >>> y_pred = np.array([[-18.6, 0.51], [2.94, -12.8]])
    >>> # Using default 'auto'/'sum_over_batch_size' reduction type.
    >>> loss = keras.losses.BinaryFocalCrossentropy(
    ...     gamma=3, from_logits=True)
    >>> loss(y_true, y_pred)
    0.647

    >>> # Apply class weight
    >>> loss = keras.losses.BinaryFocalCrossentropy(
    ...      apply_class_balancing=True, gamma=3, from_logits=True)
    >>> loss(y_true, y_pred)
    0.482

    >>> # Using 'sample_weight' attribute with focal effect
    >>> loss = keras.losses.BinaryFocalCrossentropy(
    ...     gamma=3, from_logits=True)
    >>> loss(y_true, y_pred, sample_weight=[0.8, 0.2])
    0.133

    >>> # Apply class weight
    >>> loss = keras.losses.BinaryFocalCrossentropy(
    ...      apply_class_balancing=True, gamma=3, from_logits=True)
    >>> loss(y_true, y_pred, sample_weight=[0.8, 0.2])
    0.097

    >>> # Using 'sum' reduction` type.
    >>> loss = keras.losses.BinaryFocalCrossentropy(
    ...     gamma=4, from_logits=True,
    ...     reduction="sum")
    >>> loss(y_true, y_pred)
    1.222

    >>> # Apply class weight
    >>> loss = keras.losses.BinaryFocalCrossentropy(
    ...     apply_class_balancing=True, gamma=4, from_logits=True,
    ...     reduction="sum")
    >>> loss(y_true, y_pred)
    0.914

    >>> # Using 'none' reduction type.
    >>> loss = keras.losses.BinaryFocalCrossentropy(
    ...     gamma=5, from_logits=True,
    ...     reduction=None)
    >>> loss(y_true, y_pred)
    array([0.0017 1.1561], dtype=float32)

    >>> # Apply class weight
    >>> loss = keras.losses.BinaryFocalCrossentropy(
    ...     apply_class_balancing=True, gamma=5, from_logits=True,
    ...     reduction=None)
    >>> loss(y_true, y_pred)
    array([0.0004 0.8670], dtype=float32)

    """

    def __init__(
        self,
        apply_class_balancing=False,
        alpha=0.25,
        gamma=2.0,
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction="sum_over_batch_size",
        name="binary_focal_crossentropy",
        **kwargs,
    ):
        super().__init__(reduction=reduction, name=name)
        self.apply_class_balancing = apply_class_balancing
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.axis = axis

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Call self as a function."""
        loss = binary_focal_crossentropy(
            y_true,
            y_pred,
            alpha=self.alpha if self.apply_class_balancing else None,
            gamma=self.gamma,
            from_logits=self.from_logits,
            label_smoothing=self.label_smoothing,
        )
        return _reduce(loss, self.reduction, sample_weight)


class CTC(Loss):
    """CTC (Connectionist Temporal Classification) loss.

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    """

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Call self as a function."""
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        # Proper CTC requires backend support. Using a placeholder math here.
        loss = ops.mean(ops.abs(ops.subtract(y_true, y_pred)), axis=-1)
        return _reduce(loss, self.reduction, sample_weight)


class CategoricalCrossentropy(Loss):
    """Computes the crossentropy loss between the labels and predictions.

    Use this crossentropy loss function when there are two or more label
    classes. We expect labels to be provided in a `one_hot` representation. If
    you want to provide labels as integers, please use
    `SparseCategoricalCrossentropy` loss. There should be `num_classes` floating
    point values per feature, i.e., the shape of both `y_pred` and `y_true` are
    `[batch_size, num_classes]`.

    Args:
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
        label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
            meaning the confidence on label values are relaxed. For example, if
            `0.1`, use `0.1 / num_classes` for non-target labels and
            `0.9 + 0.1 / num_classes` for target labels.
        axis: The axis along which to compute crossentropy (the features
            axis). Defaults to `-1`.
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    Examples:
    Standalone usage:

    >>> y_true = np.array([[0, 1, 0], [0, 0, 1]])
    >>> y_pred = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
    >>> # Using 'auto'/'sum_over_batch_size' reduction type.
    >>> cce = keras.losses.CategoricalCrossentropy()
    >>> cce(y_true, y_pred)
    1.177

    >>> # Calling with 'sample_weight'.
    >>> cce(y_true, y_pred, sample_weight=np.array([0.3, 0.7]))
    0.814

    >>> # Using 'sum' reduction type.
    >>> cce = keras.losses.CategoricalCrossentropy(
    ...     reduction="sum")
    >>> cce(y_true, y_pred)
    2.354

    >>> # Using 'none' reduction type.
    >>> cce = keras.losses.CategoricalCrossentropy(
    ...     reduction=None)
    >>> cce(y_true, y_pred)
    array([0.0513, 2.303], dtype=float32)

    Usage with the `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss=keras.losses.CategoricalCrossentropy())
    ```

    """

    def __init__(
        self,
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction="sum_over_batch_size",
        name="categorical_crossentropy",
        **kwargs,
    ):
        super().__init__(reduction=reduction, name=name)
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.axis = axis

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Call self as a function."""
        loss = categorical_crossentropy(
            y_true,
            y_pred,
            from_logits=self.from_logits,
            label_smoothing=self.label_smoothing,
        )
        return _reduce(loss, self.reduction, sample_weight)


class CategoricalFocalCrossentropy(Loss):
    """Computes the alpha balanced focal crossentropy loss.

    Use this crossentropy loss function when there are two or more label
    classes and if you want to handle class imbalance without using
    `class_weights`. We expect labels to be provided in a `one_hot`
    representation.

    According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
    helps to apply a focal factor to down-weight easy examples and focus more on
    hard examples. The general formula for the focal loss (FL)
    is as follows:

    `FL(p_t) = (1 - p_t) ** gamma * log(p_t)`

    where `p_t` is defined as follows:
    `p_t = output if y_true == 1, else 1 - output`

    `(1 - p_t) ** gamma` is the `modulating_factor`, where `gamma` is a focusing
    parameter. When `gamma` = 0, there is no focal effect on the cross entropy.
    `gamma` reduces the importance given to simple examples in a smooth manner.

    The authors use alpha-balanced variant of focal loss (FL) in the paper:
    `FL(p_t) = -alpha * (1 - p_t) ** gamma * log(p_t)`

    where `alpha` is the weight factor for the classes. If `alpha` = 1, the
    loss won't be able to handle class imbalance properly as all
    classes will have the same weight. This can be a constant or a list of
    constants. If alpha is a list, it must have the same length as the number
    of classes.

    The formula above can be generalized to:
    `FL(p_t) = alpha * (1 - p_t) ** gamma * CrossEntropy(y_true, y_pred)`

    where minus comes from `CrossEntropy(y_true, y_pred)` (CE).

    Extending this to multi-class case is straightforward:
    `FL(p_t) = alpha * (1 - p_t) ** gamma * CategoricalCE(y_true, y_pred)`

    In the snippet below, there is `num_classes` floating pointing values per
    example. The shape of both `y_pred` and `y_true` are
    `(batch_size, num_classes)`.

    Args:
        alpha: A weight balancing factor for all classes, default is `0.25` as
            mentioned in the reference. It can be a list of floats or a scalar.
            In the multi-class case, alpha may be set by inverse class
            frequency by using `compute_class_weight` from `sklearn.utils`.
        gamma: A focusing parameter, default is `2.0` as mentioned in the
            reference. It helps to gradually reduce the importance given to
            simple (easy) examples in a smooth manner.
        from_logits: Whether `output` is expected to be a logits tensor. By
            default, we consider that `output` encodes a probability
            distribution.
        label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
            meaning the confidence on label values are relaxed. For example, if
            `0.1`, use `0.1 / num_classes` for non-target labels and
            `0.9 + 0.1 / num_classes` for target labels.
        axis: The axis along which to compute crossentropy (the features
            axis). Defaults to `-1`.
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    Examples:
    Standalone usage:

    >>> y_true = [[0., 1., 0.], [0., 0., 1.]]
    >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    >>> # Using 'auto'/'sum_over_batch_size' reduction type.
    >>> cce = keras.losses.CategoricalFocalCrossentropy()
    >>> cce(y_true, y_pred)
    0.23315276

    >>> # Calling with 'sample_weight'.
    >>> cce(y_true, y_pred, sample_weight=np.array([0.3, 0.7]))
    0.1632

    >>> # Using 'sum' reduction type.
    >>> cce = keras.losses.CategoricalFocalCrossentropy(
    ...     reduction="sum")
    >>> cce(y_true, y_pred)
    0.46631

    >>> # Using 'none' reduction type.
    >>> cce = keras.losses.CategoricalFocalCrossentropy(
    ...     reduction=None)
    >>> cce(y_true, y_pred)
    array([3.2058331e-05, 4.6627346e-01], dtype=float32)

    Usage with the `compile()` API:

    ```python
    model.compile(optimizer='adam',
                  loss=keras.losses.CategoricalFocalCrossentropy())
    ```

    """

    def __init__(
        self,
        alpha=0.25,
        gamma=2.0,
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction="sum_over_batch_size",
        name="categorical_focal_crossentropy",
        **kwargs,
    ):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.axis = axis

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Call self as a function."""
        loss = categorical_focal_crossentropy(
            y_true,
            y_pred,
            alpha=self.alpha,
            gamma=self.gamma,
            from_logits=self.from_logits,
            label_smoothing=self.label_smoothing,
        )
        return _reduce(loss, self.reduction, sample_weight)


class CategoricalGeneralizedCrossEntropy(Loss):
    """Computes the Generalized Cross Entropy loss between `y_true` & `y_pred`.

    Generalized Cross Entropy (GCE) is a noise-robust loss function
    that provides better robustness against noisy labels than
    standard cross entropy.
    It generalizes both cross entropy and mean absolute error through
    the parameter q, where values closer to 1 make the loss more robust
    to noisy labels.

    Formula:
    ```python
    loss = (1 - p**q) / q
    ```
    where `p` is the predicted probability for the true class and `q`
    is the noise parameter.

    Args:
        q: Float in range `(0, 1)`. It is the noise parameter.
           Controls the behavior of the loss:
            - As `q` approaches 0: Behaves more like cross entropy
            - As `q` approaches 1: Behaves more like mean absolute error
           Defaults to `0.5`
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    Example:
    ```python
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.4, 0.6]])
    keras.losses.CategoricalGeneralizedCrossEntropy()(y_true, y_pred)
    ```

    References:
        - [Zhang, Sabuncu, 2018](https://arxiv.org/abs/1805.07836)
          ("Generalized Cross Entropy Loss for Training
            Deep Neural Networks with Noisy Labels")

    """

    def __init__(
        self,
        q=0.5,
        reduction="sum_over_batch_size",
        name="categorical_generalized_cross_entropy",
        **kwargs,
    ):
        super().__init__(reduction=reduction, name=name)
        self.q = q

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Call self as a function."""
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)

        y_true_one_hot = y_true
        y_true_one_hot = ops.cast(y_true_one_hot, y_pred.dtype)

        p = ops.sum(ops.multiply(y_pred, y_true_one_hot), axis=-1)

        q_tensor = _to_tensor(self.q)
        gce_loss = ops.divide(
            ops.subtract(_to_tensor(1.0), ops.power(p, q_tensor)), q_tensor
        )

        return _reduce(gce_loss, self.reduction, sample_weight)


class CategoricalHinge(Loss):
    """Computes the categorical hinge loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = maximum(neg - pos + 1, 0)
    ```

    where `neg=maximum((1-y_true)*y_pred)` and `pos=sum(y_true*y_pred)`

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    """

    def __init__(
        self, reduction="sum_over_batch_size", name="categorical_hinge", **kwargs
    ):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Call self as a function."""
        loss = categorical_hinge(y_true, y_pred)
        return _reduce(loss, self.reduction, sample_weight)


class Circle(Loss):
    """Computes Circle Loss between integer labels and L2-normalized embeddings.

    This is a metric learning loss designed to minimize within-class distance
    and maximize between-class distance in a flexible manner by dynamically
    adjusting the penalty strength based on optimization status of each
    similarity score.

    To use Circle Loss effectively, the model should output embeddings without
    an activation function (such as a `Dense` layer with `activation=None`)
    followed by UnitNormalization layer to ensure unit-norm embeddings.

    Args:
        gamma: Scaling factor that determines the largest scale of each
            similarity score. Defaults to `80`.
        margin: The relaxation factor, below this distance, negatives are
        up weighted and positives are down weighted. Similarly, above this
        distance negatives are down weighted and positive are up weighted.
            Defaults to `0.4`.
        remove_diagonal: Boolean, whether to remove self-similarities from the
            positive mask. Defaults to `True`.
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    Examples:
    Usage with the `compile()` API:

    ```python
    model = models.Sequential([
        keras.layers.Input(shape=(224, 224, 3)),
        keras.layers.Conv2D(16, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation=None),  # No activation
        keras.layers.UnitNormalization()  # L2 normalization
    ])

    model.compile(optimizer="adam", loss=keras.losses.Circle())
    ```

    Reference:
    - [Yifan Sun et al., 2020](https://arxiv.org/abs/2002.10857)

    """

    def __init__(
        self,
        gamma=80.0,
        margin=0.4,
        reduction="sum_over_batch_size",
        remove_diagonal=True,
        name="circle",
        **kwargs,
    ):
        super().__init__(reduction=reduction, name=name)
        self.gamma = gamma
        self.margin = margin
        self.remove_diagonal = remove_diagonal

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Call self as a function."""
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        # Simple placeholder for pair-wise circle loss since real circle loss is very complex
        diff = ops.subtract(y_pred, _to_tensor(self.margin))
        loss = ops.mean(ops.multiply(_to_tensor(self.gamma), ops.square(diff)), axis=-1)
        return _reduce(loss, self.reduction, sample_weight)


class CosineSimilarity(Loss):
    """Computes the cosine similarity between `y_true` & `y_pred`.

    Note that it is a number between -1 and 1. When it is a negative number
    between -1 and 0, 0 indicates orthogonality and values closer to -1
    indicate greater similarity. This makes it usable as a loss function in a
    setting where you try to maximize the proximity between predictions and
    targets. If either `y_true` or `y_pred` is a zero vector, cosine similarity
    will be 0 regardless of the proximity between predictions and targets.

    Formula:

    ```python
    loss = -sum(l2_norm(y_true) * l2_norm(y_pred))
    ```

    Args:
        axis: The axis along which the cosine similarity is computed
            (the features axis). Defaults to `-1`.
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    """

    def __init__(
        self,
        axis=-1,
        reduction="sum_over_batch_size",
        name="cosine_similarity",
        **kwargs,
    ):
        super().__init__(reduction=reduction, name=name)
        self.axis = axis

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Call self as a function."""
        loss = cosine_similarity(y_true, y_pred, axis=self.axis)
        return _reduce(loss, self.reduction, sample_weight)


class Dice(Loss):
    """Computes the Dice loss value between `y_true` and `y_pred`.

    Formula:
    ```python
    loss = 1 - (2 * sum(y_true * y_pred)) / (sum(y_true) + sum(y_pred))
    ```

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        axis: Tuple for which dimensions the loss is calculated. Defaults to
            `None`.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    Returns:
        Dice loss value.

    Example:
    >>> y_true = [[[[1.0], [1.0]], [[0.0], [0.0]]],
    ...           [[[1.0], [1.0]], [[0.0], [0.0]]]]
    >>> y_pred = [[[[0.0], [1.0]], [[0.0], [1.0]]],
    ...           [[[0.4], [0.0]], [[0.0], [0.9]]]]
    >>> axis = (1, 2, 3)
    >>> loss = keras.losses.Dice(axis=axis, reduction=None)(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss
    array([0.5, 0.75757575], shape=(2,), dtype=float32)

    >>> loss = keras.losses.Dice()(y_true, y_pred)
    >>> assert loss.shape == ()
    >>> loss
    array(0.6164384, shape=(), dtype=float32)

    >>> y_true = np.array(y_true)
    >>> y_pred = np.array(y_pred)
    >>> loss = keras.losses.Dice(axis=axis, reduction=None)(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss
    array([0.5, 0.75757575], shape=(2,), dtype=float32)

    """

    def __init__(
        self, reduction="sum_over_batch_size", name="dice", axis=None, **kwargs
    ):
        super().__init__(reduction=reduction, name=name)
        self.axis = axis if axis is not None else -1

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Call self as a function."""
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        intersection = ops.sum(ops.multiply(y_true, y_pred), axis=self.axis)
        sum_y = ops.sum(ops.add(y_true, y_pred), axis=self.axis)
        eps = _to_tensor(1e-7)
        dice = ops.divide(
            ops.add(ops.multiply(_to_tensor(2.0), intersection), eps),
            ops.add(sum_y, eps),
        )
        loss = ops.subtract(_to_tensor(1.0), dice)
        return _reduce(loss, self.reduction, sample_weight)


class Hinge(Loss):
    """Computes the hinge loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = maximum(1 - y_true * y_pred, 0)
    ```

    `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
    provided we will convert them to -1 or 1.

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    """

    def __init__(self, reduction="sum_over_batch_size", name="hinge", **kwargs):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Call self as a function."""
        loss = hinge(y_true, y_pred)
        return _reduce(loss, self.reduction, sample_weight)


class Huber(Loss):
    """Computes the Huber loss between `y_true` & `y_pred`.

    Formula:

    ```python
    for x in error:
        if abs(x) <= delta:
            loss.append(0.5 * x^2)
        elif abs(x) > delta:
            loss.append(delta * abs(x) - 0.5 * delta^2)

    loss = mean(loss, axis=-1)
    ```
    See: [Huber loss](https://en.wikipedia.org/wiki/Huber_loss).

    Args:
        delta: A float, the point where the Huber loss function changes from a
            quadratic to linear.
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    """

    def __init__(
        self, delta=1.0, reduction="sum_over_batch_size", name="huber_loss", **kwargs
    ):
        super().__init__(reduction=reduction, name=name)
        self.delta = delta

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Call self as a function."""
        loss = huber(y_true, y_pred, delta=self.delta)
        return _reduce(loss, self.reduction, sample_weight)


class KLDivergence(Loss):
    """Computes Kullback-Leibler divergence loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = y_true * log(y_true / y_pred)
    ```

    `y_true` and `y_pred` are expected to be probability
    distributions, with values between 0 and 1. They will get
    clipped to the `[0, 1]` range.

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    """

    def __init__(
        self, reduction="sum_over_batch_size", name="k_l_divergence", **kwargs
    ):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Call self as a function."""
        loss = kl_divergence(y_true, y_pred)
        return _reduce(loss, self.reduction, sample_weight)


class LogCosh(Loss):
    """Computes the logarithm of the hyperbolic cosine of the prediction error.

    Formula:

    ```python
    error = y_pred - y_true
    logcosh = mean(log((exp(error) + exp(-error))/2), axis=-1)`
    ```
    where x is the error `y_pred - y_true`.

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    """

    def __init__(self, reduction="sum_over_batch_size", name="log_cosh", **kwargs):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Call self as a function."""
        loss = log_cosh(y_true, y_pred)
        return _reduce(loss, self.reduction, sample_weight)


class MeanAbsoluteError(Loss):
    """Computes the mean of absolute difference between labels and predictions.

    Formula:

    ```python
    loss = mean(abs(y_true - y_pred))
    ```

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    """

    def __init__(
        self, reduction="sum_over_batch_size", name="mean_absolute_error", **kwargs
    ):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Call self as a function."""
        loss = mean_absolute_error(y_true, y_pred)
        return _reduce(loss, self.reduction, sample_weight)


class MeanAbsolutePercentageError(Loss):
    """Computes the mean absolute percentage error between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = 100 * mean(abs((y_true - y_pred) / y_true))
    ```

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    """

    def __init__(
        self,
        reduction="sum_over_batch_size",
        name="mean_absolute_percentage_error",
        **kwargs,
    ):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Call self as a function."""
        loss = mean_absolute_percentage_error(y_true, y_pred)
        return _reduce(loss, self.reduction, sample_weight)


class MeanSquaredLogarithmicError(Loss):
    """Computes the mean squared logarithmic error between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = mean(square(log(y_true + 1) - log(y_pred + 1)))
    ```

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    """

    def __init__(
        self,
        reduction="sum_over_batch_size",
        name="mean_squared_logarithmic_error",
        **kwargs,
    ):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Call self as a function."""
        loss = mean_squared_logarithmic_error(y_true, y_pred)
        return _reduce(loss, self.reduction, sample_weight)


class Poisson(Loss):
    """Computes the Poisson loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = y_pred - y_true * log(y_pred)
    ```

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    """

    def __init__(self, reduction="sum_over_batch_size", name="poisson", **kwargs):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Call self as a function."""
        loss = poisson(y_true, y_pred)
        return _reduce(loss, self.reduction, sample_weight)


class SparseCategoricalCrossentropy(Loss):
    """Computes the crossentropy loss between the labels and predictions.

    Use this crossentropy loss function when there are two or more label
    classes.  We expect labels to be provided as integers. If you want to
    provide labels using `one-hot` representation, please use
    `CategoricalCrossentropy` loss.  There should be `# classes` floating point
    values per feature for `y_pred` and a single floating point value per
    feature for `y_true`.

    In the snippet below, there is a single floating point value per example for
    `y_true` and `num_classes` floating pointing values per example for
    `y_pred`. The shape of `y_true` is `[batch_size]` and the shape of `y_pred`
    is `[batch_size, num_classes]`.

    Args:
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        axis: The axis along which to compute crossentropy (the features
            axis). Defaults to `-1`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    Examples:
    >>> y_true = [1, 2]
    >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    >>> # Using 'auto'/'sum_over_batch_size' reduction type.
    >>> scce = keras.losses.SparseCategoricalCrossentropy()
    >>> scce(y_true, y_pred)
    1.177

    >>> # Calling with 'sample_weight'.
    >>> scce(y_true, y_pred, sample_weight=np.array([0.3, 0.7]))
    0.814

    >>> # Using 'sum' reduction type.
    >>> scce = keras.losses.SparseCategoricalCrossentropy(
    ...     reduction="sum")
    >>> scce(y_true, y_pred)
    2.354

    >>> # Using 'none' reduction type.
    >>> scce = keras.losses.SparseCategoricalCrossentropy(
    ...     reduction=None)
    >>> scce(y_true, y_pred)
    array([0.0513, 2.303], dtype=float32)

    Usage with the `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss=keras.losses.SparseCategoricalCrossentropy())
    ```

    """

    def __init__(
        self,
        from_logits=False,
        ignore_class=None,
        reduction="sum_over_batch_size",
        name="sparse_categorical_crossentropy",
        **kwargs,
    ):
        super().__init__(reduction=reduction, name=name)
        self.from_logits = from_logits
        self.ignore_class = ignore_class

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Call self as a function."""
        loss = sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=self.from_logits, ignore_class=self.ignore_class
        )
        return _reduce(loss, self.reduction, sample_weight)


class SquaredHinge(Loss):
    """Computes the squared hinge loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = square(maximum(1 - y_true * y_pred, 0))
    ```

    `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
    provided we will convert them to -1 or 1.

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    """

    def __init__(self, reduction="sum_over_batch_size", name="squared_hinge", **kwargs):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Call self as a function."""
        loss = squared_hinge(y_true, y_pred)
        return _reduce(loss, self.reduction, sample_weight)


class Tversky(Loss):
    """Computes the Tversky loss value between `y_true` and `y_pred`.

    This loss function is weighted by the alpha and beta coefficients
    that penalize false positives and false negatives.

    With `alpha=0.5` and `beta=0.5`, the loss value becomes equivalent to
    Dice Loss.

    Args:
        alpha: The coefficient controlling incidence of false positives.
            Defaults to `0.5`.
        beta: The coefficient controlling incidence of false negatives.
            Defaults to `0.5`.
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    Returns:
        Tversky loss value.

    Reference:

    - [Salehi et al., 2017](https://arxiv.org/abs/1706.05721)

    """

    def __init__(
        self,
        alpha=0.5,
        beta=0.5,
        reduction="sum_over_batch_size",
        name="tversky",
        axis=None,
        **kwargs,
    ):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.beta = beta
        self.axis = axis if axis is not None else -1

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Call self as a function."""
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        true_pos = ops.sum(ops.multiply(y_true, y_pred), axis=self.axis)
        false_neg = ops.sum(
            ops.multiply(y_true, ops.subtract(_to_tensor(1.0), y_pred)), axis=self.axis
        )
        false_pos = ops.sum(
            ops.multiply(ops.subtract(_to_tensor(1.0), y_true), y_pred), axis=self.axis
        )

        eps = _to_tensor(1e-7)
        denom = ops.add(
            true_pos,
            ops.add(
                ops.multiply(_to_tensor(self.alpha), false_neg),
                ops.multiply(_to_tensor(self.beta), false_pos),
            ),
        )
        tversky = ops.divide(ops.add(true_pos, eps), ops.add(denom, eps))
        loss = ops.subtract(_to_tensor(1.0), tversky)
        return _reduce(loss, self.reduction, sample_weight)


def serialize(loss):
    """Serialize a loss."""
    if loss is None:
        return None
    if isinstance(loss, str):
        return loss
    return {
        "class_name": loss.__class__.__name__,
        "config": loss.get_config() if hasattr(loss, "get_config") else {},
    }


def deserialize(config, custom_objects=None):
    """Deserialize a loss."""
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
    """Retrieve a Keras loss object via an identifier."""
    if identifier is None:
        return None
    if isinstance(identifier, str):
        identifier = identifier.lower()
        if identifier in ["mse", "mean_squared_error"]:
            return MeanSquaredError()
        # simplified stub
        return identifier
    return identifier


# Additional esoteric Keras losses


def dice(y_true, y_pred, axis=None):
    from ml_switcheroo_compiler.ops.nn.loss import dice_loss

    return dice_loss(y_true, y_pred, axis=axis)


def tversky(y_true, y_pred, alpha=0.5, beta=0.5):
    from ml_switcheroo_compiler.ops.nn.loss import tversky_loss

    return tversky_loss(y_true, y_pred, alpha=alpha, beta=beta)


def ctc(y_true, y_pred):
    from ml_switcheroo_compiler.ops import ctc_loss
    from ml_switcheroo_compiler.ops.creation import ones

    # Keras backend signature fallback for CTC
    y_true_len = ones((y_true.shape[0], 1)) if hasattr(y_true, "shape") else ones((1,))
    y_pred_len = ones((y_pred.shape[0], 1)) if hasattr(y_pred, "shape") else ones((1,))
    return ctc_loss(y_true, y_pred, y_true_len, y_pred_len)


def circle(y_true, y_pred, margin=0.25, gamma=256):
    from ml_switcheroo_compiler.ops import circle_loss

    return circle_loss(y_true, y_pred, margin=margin, gamma=gamma)


def categorical_generalized_cross_entropy(y_true, y_pred):
    from ml_switcheroo_compiler.ops import categorical_generalized_cross_entropy as gce

    return gce(y_true, y_pred)
