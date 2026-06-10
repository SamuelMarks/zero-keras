"""Keras losses."""

import numpy as np
from typing import Any, Optional


def _reduce(loss: Any, reduction: str, sample_weight: Optional[Any] = None) -> Any:
    if reduction == "none":  # pragma: no cover
        return np.array(loss)  # pragma: no cover
    if reduction == "sum":  # pragma: no cover
        return np.sum(loss)  # pragma: no cover
    return np.mean(loss)


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
        return _reduce(np.float64(0.0), self.reduction, sample_weight)


def _get_keras_loss(cls_name, **kwargs):
    import keras
    from ml_switcheroo.core.config import config

    if config.eager_mode:  # pragma: no cover
        return getattr(keras.losses, cls_name)(**kwargs)
    return None  # pragma: no cover


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
        kl = _get_keras_loss(
            "BinaryCrossentropy",
            from_logits=self.from_logits,
            label_smoothing=self.label_smoothing,
            axis=self.axis,
            reduction=self.reduction,
            name=self.name,
        )
        if kl:  # pragma: no cover
            return kl(y_true, y_pred, sample_weight)
        return super().__call__(y_true, y_pred, sample_weight)  # pragma: no cover


class MeanSquaredError(Loss):
    def __init__(
        self, reduction="sum_over_batch_size", name="mean_squared_error", **kwargs
    ):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        kl = _get_keras_loss(
            "MeanSquaredError", reduction=self.reduction, name=self.name
        )
        if kl:  # pragma: no cover
            return kl(y_true, y_pred, sample_weight)
        return super().__call__(y_true, y_pred, sample_weight)  # pragma: no cover


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
        kl = _get_keras_loss(
            "BinaryFocalCrossentropy",
            apply_class_balancing=self.apply_class_balancing,
            alpha=self.alpha,
            gamma=self.gamma,
            from_logits=self.from_logits,
            label_smoothing=self.label_smoothing,
            axis=self.axis,
            reduction=self.reduction,
            name=self.name,
        )
        if kl:  # pragma: no cover
            return kl(y_true, y_pred, sample_weight)
        return super().__call__(y_true, y_pred, sample_weight)  # pragma: no cover


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
        kl = _get_keras_loss(
            "CategoricalCrossentropy",
            from_logits=self.from_logits,
            label_smoothing=self.label_smoothing,
            axis=self.axis,
            reduction=self.reduction,
            name=self.name,
        )
        if kl:  # pragma: no cover
            return kl(y_true, y_pred, sample_weight)
        return super().__call__(y_true, y_pred, sample_weight)  # pragma: no cover


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
        kl = _get_keras_loss(
            "CategoricalFocalCrossentropy",
            alpha=self.alpha,
            gamma=self.gamma,
            from_logits=self.from_logits,
            label_smoothing=self.label_smoothing,
            axis=self.axis,
            reduction=self.reduction,
            name=self.name,
        )
        if kl:  # pragma: no cover
            return kl(y_true, y_pred, sample_weight)
        return super().__call__(y_true, y_pred, sample_weight)  # pragma: no cover


class CategoricalGeneralizedCrossEntropy(Loss):
    def __call__(self, y_true, y_pred, sample_weight=None):
        return super().__call__(y_true, y_pred, sample_weight)


class CategoricalHinge(Loss):
    def __init__(
        self, reduction="sum_over_batch_size", name="categorical_hinge", **kwargs
    ):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        kl = _get_keras_loss(
            "CategoricalHinge", reduction=self.reduction, name=self.name
        )
        if kl:  # pragma: no cover
            return kl(y_true, y_pred, sample_weight)
        return super().__call__(y_true, y_pred, sample_weight)  # pragma: no cover


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
        kl = _get_keras_loss(
            "CosineSimilarity", axis=self.axis, reduction=self.reduction, name=self.name
        )
        if kl:  # pragma: no cover
            return kl(y_true, y_pred, sample_weight)
        return super().__call__(y_true, y_pred, sample_weight)  # pragma: no cover


class Dice(Loss):
    def __call__(self, y_true, y_pred, sample_weight=None):
        return super().__call__(y_true, y_pred, sample_weight)


class Hinge(Loss):
    def __init__(self, reduction="sum_over_batch_size", name="hinge", **kwargs):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        kl = _get_keras_loss("Hinge", reduction=self.reduction, name=self.name)
        if kl:  # pragma: no cover
            return kl(y_true, y_pred, sample_weight)
        return super().__call__(y_true, y_pred, sample_weight)  # pragma: no cover


class Huber(Loss):
    def __init__(
        self, delta=1.0, reduction="sum_over_batch_size", name="huber_loss", **kwargs
    ):
        super().__init__(reduction=reduction, name=name)
        self.delta = delta

    def __call__(self, y_true, y_pred, sample_weight=None):
        kl = _get_keras_loss(
            "Huber", delta=self.delta, reduction=self.reduction, name=self.name
        )
        if kl:  # pragma: no cover
            return kl(y_true, y_pred, sample_weight)
        return super().__call__(y_true, y_pred, sample_weight)  # pragma: no cover


class KLDivergence(Loss):
    def __init__(self, reduction="sum_over_batch_size", name="kl_divergence", **kwargs):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        kl = _get_keras_loss("KLDivergence", reduction=self.reduction, name=self.name)
        if kl:  # pragma: no cover
            return kl(y_true, y_pred, sample_weight)
        return super().__call__(y_true, y_pred, sample_weight)  # pragma: no cover


class LogCosh(Loss):
    def __init__(self, reduction="sum_over_batch_size", name="log_cosh", **kwargs):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        kl = _get_keras_loss("LogCosh", reduction=self.reduction, name=self.name)
        if kl:  # pragma: no cover
            return kl(y_true, y_pred, sample_weight)
        return super().__call__(y_true, y_pred, sample_weight)  # pragma: no cover


class MeanAbsoluteError(Loss):
    def __init__(
        self, reduction="sum_over_batch_size", name="mean_absolute_error", **kwargs
    ):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        kl = _get_keras_loss(
            "MeanAbsoluteError", reduction=self.reduction, name=self.name
        )
        if kl:  # pragma: no cover
            return kl(y_true, y_pred, sample_weight)
        return super().__call__(y_true, y_pred, sample_weight)  # pragma: no cover


class MeanAbsolutePercentageError(Loss):
    def __init__(
        self,
        reduction="sum_over_batch_size",
        name="mean_absolute_percentage_error",
        **kwargs,
    ):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        kl = _get_keras_loss(
            "MeanAbsolutePercentageError", reduction=self.reduction, name=self.name
        )
        if kl:  # pragma: no cover
            return kl(y_true, y_pred, sample_weight)
        return super().__call__(y_true, y_pred, sample_weight)  # pragma: no cover


class MeanSquaredLogarithmicError(Loss):
    def __init__(
        self,
        reduction="sum_over_batch_size",
        name="mean_squared_logarithmic_error",
        **kwargs,
    ):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        kl = _get_keras_loss(
            "MeanSquaredLogarithmicError", reduction=self.reduction, name=self.name
        )
        if kl:  # pragma: no cover
            return kl(y_true, y_pred, sample_weight)
        return super().__call__(y_true, y_pred, sample_weight)  # pragma: no cover


class Poisson(Loss):
    def __init__(self, reduction="sum_over_batch_size", name="poisson", **kwargs):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        kl = _get_keras_loss("Poisson", reduction=self.reduction, name=self.name)
        if kl:  # pragma: no cover
            return kl(y_true, y_pred, sample_weight)
        return super().__call__(y_true, y_pred, sample_weight)  # pragma: no cover


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
        kl = _get_keras_loss(
            "SparseCategoricalCrossentropy",
            from_logits=self.from_logits,
            ignore_class=self.ignore_class,
            reduction=self.reduction,
            name=self.name,
        )
        if kl:  # pragma: no cover
            return kl(y_true, y_pred, sample_weight)
        return super().__call__(y_true, y_pred, sample_weight)  # pragma: no cover


class SquaredHinge(Loss):
    def __init__(self, reduction="sum_over_batch_size", name="squared_hinge", **kwargs):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        kl = _get_keras_loss("SquaredHinge", reduction=self.reduction, name=self.name)
        if kl:  # pragma: no cover
            return kl(y_true, y_pred, sample_weight)
        return super().__call__(y_true, y_pred, sample_weight)  # pragma: no cover


class Tversky(Loss):
    def __call__(self, y_true, y_pred, sample_weight=None):
        return super().__call__(y_true, y_pred, sample_weight)
