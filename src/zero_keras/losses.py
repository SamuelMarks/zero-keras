"""Keras losses."""

import ml_switcheroo.ops as ops
from ml_switcheroo.core.dtype import DType
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
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        if self.label_smoothing > 0:
            y_true = ops.add(
                ops.multiply(y_true, _to_tensor(1.0 - self.label_smoothing)),
                _to_tensor(0.5 * self.label_smoothing),
            )
        if self.from_logits:
            y_pred = nn.sigmoid(y_pred)
        y_pred = ops.maximum(
            _to_tensor(1e-7), ops.minimum(_to_tensor(1.0 - 1e-7), y_pred)
        )
        bce = ops.subtract(
            ops.multiply(_to_tensor(-1.0), ops.multiply(y_true, ops.log(y_pred))),
            ops.multiply(
                ops.subtract(_to_tensor(1.0), y_true),
                ops.log(ops.subtract(_to_tensor(1.0), y_pred)),
            ),
        )
        loss = ops.mean(bce, axis=-1)
        return _reduce(loss, self.reduction, sample_weight)


class MeanSquaredError(Loss):
    def __init__(
        self, reduction="sum_over_batch_size", name="mean_squared_error", **kwargs
    ):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        diff = ops.subtract(y_true, y_pred)
        loss = ops.mean(ops.square(diff), axis=-1)
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
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        if self.label_smoothing > 0:
            y_true = ops.add(
                ops.multiply(y_true, _to_tensor(1.0 - self.label_smoothing)),
                _to_tensor(0.5 * self.label_smoothing),
            )
        if self.from_logits:
            y_pred = nn.sigmoid(y_pred)
        y_pred = ops.maximum(
            _to_tensor(1e-7), ops.minimum(_to_tensor(1.0 - 1e-7), y_pred)
        )

        # p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        p_t = ops.add(
            ops.multiply(y_true, y_pred),
            ops.multiply(
                ops.subtract(_to_tensor(1.0), y_true),
                ops.subtract(_to_tensor(1.0), y_pred),
            ),
        )
        alpha_t = (
            ops.add(
                ops.multiply(y_true, _to_tensor(self.alpha)),
                ops.multiply(
                    ops.subtract(_to_tensor(1.0), y_true), _to_tensor(1.0 - self.alpha)
                ),
            )
            if self.apply_class_balancing
            else _to_tensor(1.0)
        )

        bce = ops.subtract(
            ops.multiply(_to_tensor(-1.0), ops.multiply(y_true, ops.log(y_pred))),
            ops.multiply(
                ops.subtract(_to_tensor(1.0), y_true),
                ops.log(ops.subtract(_to_tensor(1.0), y_pred)),
            ),
        )

        focal_loss = ops.multiply(
            ops.multiply(
                alpha_t,
                ops.power(ops.subtract(_to_tensor(1.0), p_t), _to_tensor(self.gamma)),
            ),
            bce,
        )
        loss = ops.mean(focal_loss, axis=-1)
        return _reduce(loss, self.reduction, sample_weight)


class CTC(Loss):
    def __call__(self, y_true, y_pred, sample_weight=None):
        return super().__call__(y_true, y_pred, sample_weight)


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
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        if self.label_smoothing > 0:
            num_classes = y_pred.shape[-1]
            y_true = ops.add(
                ops.multiply(y_true, _to_tensor(1.0 - self.label_smoothing)),
                _to_tensor(self.label_smoothing / num_classes),
            )
        if self.from_logits:
            y_pred = nn.softmax(y_pred, dim=self.axis)
        y_pred = ops.maximum(
            _to_tensor(1e-7), ops.minimum(_to_tensor(1.0 - 1e-7), y_pred)
        )
        y_pred = ops.divide(y_pred, ops.sum(y_pred, axis=self.axis, keepdims=True))
        cce = ops.multiply(
            _to_tensor(-1.0),
            ops.sum(ops.multiply(y_true, ops.log(y_pred)), axis=self.axis),
        )
        return _reduce(cce, self.reduction, sample_weight)


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
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        if self.label_smoothing > 0:
            num_classes = y_pred.shape[-1]
            y_true = ops.add(
                ops.multiply(y_true, _to_tensor(1.0 - self.label_smoothing)),
                _to_tensor(self.label_smoothing / num_classes),
            )
        if self.from_logits:
            y_pred = nn.softmax(y_pred, dim=self.axis)
        y_pred = ops.maximum(
            _to_tensor(1e-7), ops.minimum(_to_tensor(1.0 - 1e-7), y_pred)
        )
        y_pred = ops.divide(y_pred, ops.sum(y_pred, axis=self.axis, keepdims=True))

        focal_loss = ops.multiply(
            ops.multiply(
                _to_tensor(-self.alpha),
                ops.power(
                    ops.subtract(_to_tensor(1.0), y_pred), _to_tensor(self.gamma)
                ),
            ),
            ops.multiply(y_true, ops.log(y_pred)),
        )
        loss = ops.sum(focal_loss, axis=self.axis)
        return _reduce(loss, self.reduction, sample_weight)


class CategoricalGeneralizedCrossEntropy(Loss):
    def __call__(self, y_true, y_pred, sample_weight=None):
        return super().__call__(y_true, y_pred, sample_weight)


class CategoricalHinge(Loss):
    def __init__(
        self, reduction="sum_over_batch_size", name="categorical_hinge", **kwargs
    ):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        pos = ops.sum(ops.multiply(y_true, y_pred), axis=-1)
        neg = ops.max(
            ops.multiply(ops.subtract(_to_tensor(1.0), y_true), y_pred), axis=-1
        )
        loss = ops.maximum(
            ops.add(ops.subtract(neg, pos), _to_tensor(1.0)), _to_tensor(0.0)
        )
        return _reduce(loss, self.reduction, sample_weight)


class Circle(Loss):
    def __call__(self, y_true, y_pred, sample_weight=None):
        return super().__call__(y_true, y_pred, sample_weight)


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
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        y_true_norm = ops.maximum(
            ops.norm(y_true, axis=self.axis, keepdims=True), _to_tensor(1e-7)
        )
        y_pred_norm = ops.maximum(
            ops.norm(y_pred, axis=self.axis, keepdims=True), _to_tensor(1e-7)
        )
        y_true_normalized = ops.divide(y_true, y_true_norm)
        y_pred_normalized = ops.divide(y_pred, y_pred_norm)
        loss = ops.multiply(
            _to_tensor(-1.0),
            ops.sum(ops.multiply(y_true_normalized, y_pred_normalized), axis=self.axis),
        )
        return _reduce(loss, self.reduction, sample_weight)


class Dice(Loss):
    def __call__(self, y_true, y_pred, sample_weight=None):
        return super().__call__(y_true, y_pred, sample_weight)


class Hinge(Loss):
    def __init__(self, reduction="sum_over_batch_size", name="hinge", **kwargs):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        # hinge = max(1 - y_true * y_pred, 0)
        prod = ops.multiply(y_true, y_pred)
        loss = ops.maximum(ops.subtract(_to_tensor(1.0), prod), _to_tensor(0.0))
        loss = ops.mean(loss, axis=-1)
        return _reduce(loss, self.reduction, sample_weight)


class Huber(Loss):
    def __init__(
        self, delta=1.0, reduction="sum_over_batch_size", name="huber_loss", **kwargs
    ):
        super().__init__(reduction=reduction, name=name)
        self.delta = delta

    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        error = ops.subtract(y_pred, y_true)
        abs_error = ops.abs(error)
        delta = _to_tensor(self.delta)
        quadratic = ops.minimum(abs_error, delta)
        linear = ops.subtract(abs_error, quadratic)
        loss = ops.add(
            ops.multiply(_to_tensor(0.5), ops.square(quadratic)),
            ops.multiply(delta, linear),
        )
        loss = ops.mean(loss, axis=-1)
        return _reduce(loss, self.reduction, sample_weight)


class KLDivergence(Loss):
    def __init__(
        self, reduction="sum_over_batch_size", name="k_l_divergence", **kwargs
    ):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        y_true = ops.maximum(y_true, _to_tensor(1e-7))
        y_pred = ops.maximum(y_pred, _to_tensor(1e-7))
        loss = ops.sum(
            ops.multiply(y_true, ops.log(ops.divide(y_true, y_pred))), axis=-1
        )
        return _reduce(loss, self.reduction, sample_weight)


class LogCosh(Loss):
    def __init__(self, reduction="sum_over_batch_size", name="log_cosh", **kwargs):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        diff = ops.subtract(y_pred, y_true)
        # log(cosh(x)) approx abs(x) - log(2)
        loss = ops.subtract(
            ops.log(ops.cosh(diff)), _to_tensor(0.0)
        )  # simplified, we can just use cosh since we have it in unary. wait, do we have logcosh? no.
        loss = ops.mean(
            ops.subtract(
                ops.add(diff, nn.softplus(ops.multiply(_to_tensor(-2.0), diff))),
                ops.log(_to_tensor(2.0)),
            ),
            axis=-1,
        )
        # Actually softplus might not be in ops. Let's just use log(cosh(x)).
        loss = ops.mean(ops.log(ops.cosh(diff)), axis=-1)
        return _reduce(loss, self.reduction, sample_weight)


class MeanAbsoluteError(Loss):
    def __init__(
        self, reduction="sum_over_batch_size", name="mean_absolute_error", **kwargs
    ):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        diff = ops.subtract(y_true, y_pred)
        loss = ops.mean(ops.abs(diff), axis=-1)
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
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        diff = ops.subtract(y_true, y_pred)
        den = ops.maximum(ops.abs(y_true), _to_tensor(1e-7))
        loss = ops.multiply(
            _to_tensor(100.0), ops.mean(ops.divide(ops.abs(diff), den), axis=-1)
        )
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
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        first_log = ops.log(
            ops.maximum(ops.add(y_pred, _to_tensor(1.0)), _to_tensor(1e-7))
        )
        second_log = ops.log(
            ops.maximum(ops.add(y_true, _to_tensor(1.0)), _to_tensor(1e-7))
        )
        diff = ops.subtract(first_log, second_log)
        loss = ops.mean(ops.square(diff), axis=-1)
        return _reduce(loss, self.reduction, sample_weight)


class Poisson(Loss):
    def __init__(self, reduction="sum_over_batch_size", name="poisson", **kwargs):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        loss = ops.mean(
            ops.subtract(
                y_pred,
                ops.multiply(y_true, ops.log(ops.maximum(y_pred, _to_tensor(1e-7)))),
            ),
            axis=-1,
        )
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
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        if self.from_logits:
            y_pred = nn.softmax(y_pred, dim=-1)
        y_pred = ops.maximum(
            _to_tensor(1e-7), ops.minimum(_to_tensor(1.0 - 1e-7), y_pred)
        )
        y_pred = ops.divide(y_pred, ops.sum(y_pred, axis=-1, keepdims=True))

        # We need take_along_axis. But for simplicity let's just use a one-hot-like gather or array ops if possible.
        # Actually in ml_switcheroo, we can just use Keras to pass if it's too complex... Wait! Rule: DO NOT leak framework specific concepts.
        # How to do sparse CE? We can use take_along_axis in eager mode if we could, but we have ops.take_along_axis!

        y_true_int = ops.cast(y_true, DType.Int64)
        if len(y_true.shape) < len(y_pred.shape):
            y_true_int = ops.unsqueeze(y_true_int, dim=-1)

        probs = ops.take_along_axis(y_pred, y_true_int, axis=-1)
        probs = ops.squeeze(probs, dim=-1)
        loss = ops.multiply(_to_tensor(-1.0), ops.log(probs))
        return _reduce(loss, self.reduction, sample_weight)


class SquaredHinge(Loss):
    def __init__(self, reduction="sum_over_batch_size", name="squared_hinge", **kwargs):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _to_tensor(y_true), _to_tensor(y_pred)
        prod = ops.multiply(y_true, y_pred)
        loss = ops.square(
            ops.maximum(ops.subtract(_to_tensor(1.0), prod), _to_tensor(0.0))
        )
        loss = ops.mean(loss, axis=-1)
        return _reduce(loss, self.reduction, sample_weight)


class Tversky(Loss):
    def __call__(self, y_true, y_pred, sample_weight=None):
        return super().__call__(y_true, y_pred, sample_weight)
