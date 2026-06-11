"""Keras losses."""

import ml_switcheroo.ops as ops
import ml_switcheroo.nn as nn
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
        loss = nn.binary_crossentropy(
            y_true, y_pred, self.from_logits, self.label_smoothing
        )
        loss = ops.mean(loss, axis=-1)
        return _reduce(loss, self.reduction, sample_weight)


class MeanSquaredError(Loss):
    def __init__(
        self, reduction="sum_over_batch_size", name="mean_squared_error", **kwargs
    ):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = nn.mean_squared_error(y_true, y_pred)
        loss = ops.mean(loss, axis=-1)
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
        loss = nn.binary_focal_crossentropy(
            y_true,
            y_pred,
            self.alpha,
            self.gamma,
            self.from_logits,
            self.label_smoothing,
            self.apply_class_balancing,
        )
        loss = ops.mean(loss, axis=-1)
        return _reduce(loss, self.reduction, sample_weight)


class CTC(Loss):
    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = nn.ctc_loss(y_true, y_pred)
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
        cce = nn.categorical_crossentropy(
            y_true, y_pred, self.from_logits, self.label_smoothing, self.axis
        )
        loss = ops.sum(cce, axis=self.axis)
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
        focal_loss = nn.categorical_focal_crossentropy(
            y_true,
            y_pred,
            self.alpha,
            self.gamma,
            self.from_logits,
            self.label_smoothing,
            self.axis,
        )
        loss = ops.sum(focal_loss, axis=self.axis)
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
        loss = nn.categorical_generalized_cross_entropy(y_true, y_pred, self.q)
        return _reduce(loss, self.reduction, sample_weight)


class CategoricalHinge(Loss):
    def __init__(
        self, reduction="sum_over_batch_size", name="categorical_hinge", **kwargs
    ):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = nn.categorical_hinge(y_true, y_pred)
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
        loss = nn.circle_loss(
            y_true, y_pred, self.gamma, self.margin, self.remove_diagonal
        )
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
        loss = nn.cosine_similarity(y_true, y_pred, self.axis)
        return _reduce(loss, self.reduction, sample_weight)


class Dice(Loss):
    def __init__(
        self, reduction="sum_over_batch_size", name="dice", axis=None, **kwargs
    ):
        super().__init__(reduction=reduction, name=name)
        self.axis = axis

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = nn.dice_loss(y_true, y_pred, self.axis)
        return _reduce(loss, self.reduction, sample_weight)


class Hinge(Loss):
    def __init__(self, reduction="sum_over_batch_size", name="hinge", **kwargs):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = nn.hinge(y_true, y_pred)
        loss = ops.mean(loss, axis=-1)
        return _reduce(loss, self.reduction, sample_weight)


class Huber(Loss):
    def __init__(
        self, delta=1.0, reduction="sum_over_batch_size", name="huber_loss", **kwargs
    ):
        super().__init__(reduction=reduction, name=name)
        self.delta = delta

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = nn.huber(y_true, y_pred, self.delta)
        loss = ops.mean(loss, axis=-1)
        return _reduce(loss, self.reduction, sample_weight)


class KLDivergence(Loss):
    def __init__(
        self, reduction="sum_over_batch_size", name="k_l_divergence", **kwargs
    ):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = nn.kl_divergence(y_true, y_pred)
        loss = ops.sum(loss, axis=-1)
        return _reduce(loss, self.reduction, sample_weight)


class LogCosh(Loss):
    def __init__(self, reduction="sum_over_batch_size", name="log_cosh", **kwargs):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = nn.log_cosh(y_true, y_pred)
        loss = ops.mean(loss, axis=-1)
        return _reduce(loss, self.reduction, sample_weight)


class MeanAbsoluteError(Loss):
    def __init__(
        self, reduction="sum_over_batch_size", name="mean_absolute_error", **kwargs
    ):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = nn.mean_absolute_error(y_true, y_pred)
        loss = ops.mean(loss, axis=-1)
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
        loss = nn.mean_absolute_percentage_error(y_true, y_pred)
        loss = ops.mean(loss, axis=-1)
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
        loss = nn.mean_squared_logarithmic_error(y_true, y_pred)
        loss = ops.mean(loss, axis=-1)
        return _reduce(loss, self.reduction, sample_weight)


class Poisson(Loss):
    def __init__(self, reduction="sum_over_batch_size", name="poisson", **kwargs):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = nn.poisson(y_true, y_pred)
        loss = ops.mean(loss, axis=-1)
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
        loss = nn.sparse_categorical_crossentropy(
            y_true, y_pred, self.from_logits, self.ignore_class
        )
        return _reduce(loss, self.reduction, sample_weight)


class SquaredHinge(Loss):
    def __init__(self, reduction="sum_over_batch_size", name="squared_hinge", **kwargs):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = nn.squared_hinge(y_true, y_pred)
        loss = ops.mean(loss, axis=-1)
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
        loss = nn.tversky(y_true, y_pred, self.alpha, self.beta, self.axis)
        return _reduce(loss, self.reduction, sample_weight)
