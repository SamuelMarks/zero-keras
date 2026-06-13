"""Keras losses."""

import ml_switcheroo_compiler.ops as ops
from .activations import _to_tensor, _wrap
from typing import Any, Optional


def _reduce(loss: Any, reduction: str, sample_weight: Optional[Any] = None) -> Any:
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
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
    return ops.mean(ops.square(y_true - y_pred), axis=-1)


def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
    return ops.mean(ops.abs(y_true - y_pred), axis=-1)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
    diff = ops.abs((y_true - y_pred) / ops.maximum(ops.abs(y_true), 1e-7))
    return 100.0 * ops.mean(diff, axis=-1)


def mean_squared_logarithmic_error(y_true, y_pred):
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
    first_log = ops.log(ops.maximum(y_pred, 0.0) + 1.0)
    second_log = ops.log(ops.maximum(y_true, 0.0) + 1.0)
    return ops.mean(ops.square(first_log - second_log), axis=-1)


def huber(y_true, y_pred, delta=1.0):
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
    error = y_true - y_pred
    abs_error = ops.abs(error)
    quadratic = ops.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return ops.mean(0.5 * ops.square(quadratic) + delta * linear, axis=-1)


def log_cosh(y_true, y_pred):
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
    x = y_pred - y_true
    return ops.mean(
        x + ops.logaddexp(ops.zeros_like(x), -2.0 * x) - ops.log(ops.full_like(x, 2.0)),
        axis=-1,
    )


def hinge(y_true, y_pred):
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
    return ops.mean(ops.maximum(1.0 - y_true * y_pred, 0.0), axis=-1)


def squared_hinge(y_true, y_pred):
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
    return ops.mean(ops.square(ops.maximum(1.0 - y_true * y_pred, 0.0)), axis=-1)


def categorical_hinge(y_true, y_pred):
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
    pos = ops.sum(y_true * y_pred, axis=-1)
    neg = ops.max((1.0 - y_true) * y_pred, axis=-1)
    return ops.maximum(neg - pos + 1.0, 0.0)


def poisson(y_true, y_pred):
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
    return ops.mean(y_pred - y_true * ops.log(y_pred + 1e-7), axis=-1)


def kl_divergence(y_true, y_pred):
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
    y_true = ops.maximum(y_true, 1e-7)
    y_pred = ops.maximum(y_pred, 1e-7)
    return ops.sum(y_true * ops.log(y_true / y_pred), axis=-1)


def cosine_similarity(y_true, y_pred, axis=-1):
    y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
    y_true_norm = ops.sqrt(ops.maximum(ops.sum(ops.square(y_true), axis=axis), 1e-7))
    y_pred_norm = ops.sqrt(ops.maximum(ops.sum(ops.square(y_pred), axis=axis), 1e-7))
    return -ops.sum(y_true * y_pred, axis=axis) / (y_true_norm * y_pred_norm)


def binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0.0):
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
    """Base class for all Keras losses."""

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
        return _reduce(_to_tensor(0.0), self.reduction, sample_weight)


class BinaryCrossentropy(Loss):
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
        loss = binary_crossentropy(
            y_true,
            y_pred,
            from_logits=self.from_logits,
            label_smoothing=self.label_smoothing,
        )
        return _reduce(loss, self.reduction, sample_weight)


class MeanSquaredError(Loss):
    def __init__(
        self, reduction="sum_over_batch_size", name="mean_squared_error", **kwargs
    ):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = mean_squared_error(y_true, y_pred)
        return _reduce(loss, self.reduction, sample_weight)


class BinaryFocalCrossentropy(Loss):
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
    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = y_pred
        return _reduce(loss, self.reduction, sample_weight)


class CategoricalCrossentropy(Loss):
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
        loss = categorical_crossentropy(
            y_true,
            y_pred,
            from_logits=self.from_logits,
            label_smoothing=self.label_smoothing,
        )
        return _reduce(loss, self.reduction, sample_weight)


class CategoricalFocalCrossentropy(Loss):
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
        loss = y_pred
        return _reduce(loss, self.reduction, sample_weight)


class CategoricalHinge(Loss):
    def __init__(
        self, reduction="sum_over_batch_size", name="categorical_hinge", **kwargs
    ):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = categorical_hinge(y_true, y_pred)
        return _reduce(loss, self.reduction, sample_weight)


class Circle(Loss):
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
        loss = y_pred
        return _reduce(loss, self.reduction, sample_weight)


class CosineSimilarity(Loss):
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
        loss = cosine_similarity(y_true, y_pred, axis=self.axis)
        return _reduce(loss, self.reduction, sample_weight)


class Dice(Loss):
    def __init__(
        self, reduction="sum_over_batch_size", name="dice", axis=None, **kwargs
    ):
        super().__init__(reduction=reduction, name=name)
        self.axis = axis

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = y_pred
        return _reduce(loss, self.reduction, sample_weight)


class Hinge(Loss):
    def __init__(self, reduction="sum_over_batch_size", name="hinge", **kwargs):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = hinge(y_true, y_pred)
        return _reduce(loss, self.reduction, sample_weight)


class Huber(Loss):
    def __init__(
        self, delta=1.0, reduction="sum_over_batch_size", name="huber_loss", **kwargs
    ):
        super().__init__(reduction=reduction, name=name)
        self.delta = delta

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = huber(y_true, y_pred, delta=self.delta)
        return _reduce(loss, self.reduction, sample_weight)


class KLDivergence(Loss):
    def __init__(
        self, reduction="sum_over_batch_size", name="k_l_divergence", **kwargs
    ):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = kl_divergence(y_true, y_pred)
        return _reduce(loss, self.reduction, sample_weight)


class LogCosh(Loss):
    def __init__(self, reduction="sum_over_batch_size", name="log_cosh", **kwargs):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = log_cosh(y_true, y_pred)
        return _reduce(loss, self.reduction, sample_weight)


class MeanAbsoluteError(Loss):
    def __init__(
        self, reduction="sum_over_batch_size", name="mean_absolute_error", **kwargs
    ):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = mean_absolute_error(y_true, y_pred)
        return _reduce(loss, self.reduction, sample_weight)


class MeanAbsolutePercentageError(Loss):
    def __init__(
        self,
        reduction="sum_over_batch_size",
        name="mean_absolute_percentage_error",
        **kwargs,
    ):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = mean_absolute_percentage_error(y_true, y_pred)
        return _reduce(loss, self.reduction, sample_weight)


class MeanSquaredLogarithmicError(Loss):
    def __init__(
        self,
        reduction="sum_over_batch_size",
        name="mean_squared_logarithmic_error",
        **kwargs,
    ):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = mean_squared_logarithmic_error(y_true, y_pred)
        return _reduce(loss, self.reduction, sample_weight)


class Poisson(Loss):
    def __init__(self, reduction="sum_over_batch_size", name="poisson", **kwargs):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = poisson(y_true, y_pred)
        return _reduce(loss, self.reduction, sample_weight)


class SparseCategoricalCrossentropy(Loss):
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
        loss = sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=self.from_logits, ignore_class=self.ignore_class
        )
        return _reduce(loss, self.reduction, sample_weight)


class SquaredHinge(Loss):
    def __init__(self, reduction="sum_over_batch_size", name="squared_hinge", **kwargs):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = squared_hinge(y_true, y_pred)
        return _reduce(loss, self.reduction, sample_weight)


class Tversky(Loss):
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
        self.axis = axis

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = y_pred
        return _reduce(loss, self.reduction, sample_weight)
