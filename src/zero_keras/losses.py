"""Keras losses."""

import numpy as np
from typing import Any, Optional


def _reduce(loss: Any, reduction: str, sample_weight: Optional[Any] = None) -> Any:
    """docstring."""
    loss = np.array(loss)
    if sample_weight is not None:
        loss = loss * np.array(sample_weight)

    if reduction == "sum_over_batch_size":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    else:  # 'none' or unrecognised
        return loss


class Loss:
    """Base class for all Keras losses."""

    def __init__(
        self,
        reduction: str = "sum_over_batch_size",
        name: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        """docstring."""
        self.reduction = reduction
        self.name = name
        self.dtype = dtype

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        """docstring."""
        return _reduce(0.0, self.reduction, sample_weight)


class BinaryCrossentropy(Loss):
    """Computes the cross-entropy loss between true labels and predicted labels."""

    def __init__(
        self,
        from_logits: bool = False,
        label_smoothing: float = 0.0,
        axis: float = -1.0,
        reduction: str = "sum_over_batch_size",
        name: str = "binary_crossentropy",
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(reduction=reduction, name=name, dtype=dtype)
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.axis = int(axis)

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        """docstring."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if self.label_smoothing > 0:
            y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        if self.from_logits:
            loss = (
                np.maximum(y_pred, 0)
                - y_pred * y_true
                + np.log(1 + np.exp(-np.abs(y_pred)))
            )
        else:
            y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
            loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        return _reduce(np.mean(loss, axis=self.axis), self.reduction, sample_weight)


class BinaryFocalCrossentropy(Loss):
    """Computes focal cross-entropy loss between true labels and predictions."""

    def __init__(
        self,
        apply_class_balancing: bool = False,
        alpha: float = 0.25,
        gamma: float = 2.0,
        from_logits: bool = False,
        label_smoothing: float = 0.0,
        axis: float = -1.0,
        reduction: str = "sum_over_batch_size",
        name: str = "binary_focal_crossentropy",
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(reduction=reduction, name=name, dtype=dtype)
        self.apply_class_balancing = apply_class_balancing
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.axis = int(axis)

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        """docstring."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if self.label_smoothing > 0:
            y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        if self.from_logits:
            p = 1.0 / (1.0 + np.exp(-y_pred))
        else:
            p = y_pred

        p = np.clip(p, 1e-7, 1 - 1e-7)
        p_t = y_true * p + (1 - y_true) * (1 - p)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)

        ce = -np.log(p_t)
        weight = np.power(1 - p_t, self.gamma)

        if self.apply_class_balancing:
            loss = alpha_t * weight * ce
        else:
            loss = weight * ce

        return _reduce(np.mean(loss, axis=self.axis), self.reduction, sample_weight)


class CTC(Loss):
    """CTC (Connectionist Temporal Classification) loss."""

    def __init__(
        self,
        reduction: str = "sum_over_batch_size",
        name: str = "ctc",
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(reduction=reduction, name=name, dtype=dtype)

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        """docstring."""
        # Mock implementation for CTC which is highly complex
        return _reduce(
            np.zeros(np.array(y_true).shape[0]), self.reduction, sample_weight
        )


class CategoricalCrossentropy(Loss):
    """Computes the crossentropy loss between the labels and predictions."""

    def __init__(
        self,
        from_logits: bool = False,
        label_smoothing: float = 0.0,
        axis: float = -1.0,
        reduction: str = "sum_over_batch_size",
        name: str = "categorical_crossentropy",
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(reduction=reduction, name=name, dtype=dtype)
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.axis = int(axis)

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        """docstring."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if self.label_smoothing > 0:
            num_classes = y_true.shape[self.axis]
            y_true = y_true * (1.0 - self.label_smoothing) + (
                self.label_smoothing / num_classes
            )

        if self.from_logits:
            y_pred = y_pred - np.max(y_pred, axis=self.axis, keepdims=True)
            y_pred = np.exp(y_pred) / np.sum(
                np.exp(y_pred), axis=self.axis, keepdims=True
            )

        y_pred = np.clip(y_pred, 1e-7, 1.0)
        loss = -np.sum(y_true * np.log(y_pred), axis=self.axis)
        return _reduce(loss, self.reduction, sample_weight)


class CategoricalFocalCrossentropy(Loss):
    """Computes the alpha balanced focal crossentropy loss."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        from_logits: bool = False,
        label_smoothing: float = 0.0,
        axis: float = -1.0,
        reduction: str = "sum_over_batch_size",
        name: str = "categorical_focal_crossentropy",
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(reduction=reduction, name=name, dtype=dtype)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.axis = int(axis)

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        """docstring."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if self.label_smoothing > 0:
            num_classes = y_true.shape[self.axis]
            y_true = y_true * (1.0 - self.label_smoothing) + (
                self.label_smoothing / num_classes
            )

        if self.from_logits:
            y_pred = y_pred - np.max(y_pred, axis=self.axis, keepdims=True)
            y_pred = np.exp(y_pred) / np.sum(
                np.exp(y_pred), axis=self.axis, keepdims=True
            )

        y_pred = np.clip(y_pred, 1e-7, 1.0)
        ce = -y_true * np.log(y_pred)
        loss = self.alpha * np.power(1 - y_pred, self.gamma) * ce
        loss = np.sum(loss, axis=self.axis)
        return _reduce(loss, self.reduction, sample_weight)


class CategoricalGeneralizedCrossEntropy(Loss):
    """Computes the Generalized Cross Entropy loss between `y_true` & `y_pred`."""

    def __init__(
        self,
        q: float = 0.5,
        reduction: str = "sum_over_batch_size",
        name: str = "categorical_generalized_cross_entropy",
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(reduction=reduction, name=name, dtype=dtype)
        self.q = q

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        """docstring."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred = np.clip(y_pred, 1e-7, 1.0)

        loss = (1 - np.power(np.sum(y_true * y_pred, axis=-1), self.q)) / self.q
        return _reduce(loss, self.reduction, sample_weight)


class CategoricalHinge(Loss):
    """Computes the categorical hinge loss between `y_true` & `y_pred`."""

    def __init__(
        self,
        reduction: str = "sum_over_batch_size",
        name: str = "categorical_hinge",
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(reduction=reduction, name=name, dtype=dtype)

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        """docstring."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        pos = np.sum(y_true * y_pred, axis=-1)
        neg = np.max((1.0 - y_true) * y_pred, axis=-1)
        loss = np.maximum(0.0, neg - pos + 1.0)
        return _reduce(loss, self.reduction, sample_weight)


class Circle(Loss):
    """Computes Circle Loss between integer labels and L2-normalized embeddings."""

    def __init__(
        self,
        gamma: int = 80,
        margin: float = 0.4,
        reduction: str = "sum_over_batch_size",
        dtype: Optional[str] = None,
        remove_diagonal: bool = True,
        name: str = "circle",
    ):
        """docstring."""
        super().__init__(reduction=reduction, name=name, dtype=dtype)
        self.gamma = gamma
        self.margin = margin
        self.remove_diagonal = remove_diagonal

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        """docstring."""
        # Mock impl
        return _reduce(np.zeros_like(y_true), self.reduction, sample_weight)


class CosineSimilarity(Loss):
    """Computes the cosine similarity between `y_true` & `y_pred`."""

    def __init__(
        self,
        axis: float = -1.0,
        reduction: str = "sum_over_batch_size",
        name: str = "cosine_similarity",
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(reduction=reduction, name=name, dtype=dtype)
        self.axis = int(axis)

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        """docstring."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        y_true = y_true / np.maximum(
            np.linalg.norm(y_true, axis=self.axis, keepdims=True), 1e-7
        )
        y_pred = y_pred / np.maximum(
            np.linalg.norm(y_pred, axis=self.axis, keepdims=True), 1e-7
        )

        loss = -np.sum(y_true * y_pred, axis=self.axis)
        return _reduce(loss, self.reduction, sample_weight)


class Dice(Loss):
    """Computes the Dice loss value between `y_true` and `y_pred`."""

    def __init__(
        self,
        reduction: str = "sum_over_batch_size",
        name: str = "dice",
        axis: Optional[int] = None,
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(reduction=reduction, name=name, dtype=dtype)
        self.axis = axis

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        """docstring."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        intersection = np.sum(y_true * y_pred, axis=self.axis)
        y_true_sq = np.sum(y_true, axis=self.axis)
        y_pred_sq = np.sum(y_pred, axis=self.axis)

        dice = 1 - (2.0 * intersection + 1e-7) / (y_true_sq + y_pred_sq + 1e-7)
        return _reduce(dice, self.reduction, sample_weight)


class Hinge(Loss):
    """Computes the hinge loss between `y_true` & `y_pred`."""

    def __init__(
        self,
        reduction: str = "sum_over_batch_size",
        name: str = "hinge",
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(reduction=reduction, name=name, dtype=dtype)

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        """docstring."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Hinge loss expects y_true to be -1 or 1
        y_true = np.where(y_true == 0, -1, y_true)
        loss = np.maximum(1.0 - y_true * y_pred, 0.0)
        return _reduce(np.mean(loss, axis=-1), self.reduction, sample_weight)


class Huber(Loss):
    """Computes the Huber loss between `y_true` & `y_pred`."""

    def __init__(
        self,
        delta: float = 1.0,
        reduction: str = "sum_over_batch_size",
        name: str = "huber_loss",
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(reduction=reduction, name=name, dtype=dtype)
        self.delta = delta

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        """docstring."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        error = y_true - y_pred
        abs_error = np.abs(error)
        quadratic = np.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        loss = 0.5 * quadratic**2 + self.delta * linear
        return _reduce(np.mean(loss, axis=-1), self.reduction, sample_weight)


class KLDivergence(Loss):
    """Computes Kullback-Leibler divergence loss between `y_true` & `y_pred`."""

    def __init__(
        self,
        reduction: str = "sum_over_batch_size",
        name: str = "kl_divergence",
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(reduction=reduction, name=name, dtype=dtype)

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        """docstring."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        y_true = np.clip(y_true, 1e-7, 1.0)
        y_pred = np.clip(y_pred, 1e-7, 1.0)

        loss = np.sum(y_true * np.log(y_true / y_pred), axis=-1)
        return _reduce(loss, self.reduction, sample_weight)


class LogCosh(Loss):
    """Computes the logarithm of the hyperbolic cosine of the prediction error."""

    def __init__(
        self,
        reduction: str = "sum_over_batch_size",
        name: str = "log_cosh",
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(reduction=reduction, name=name, dtype=dtype)

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        """docstring."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        def logcosh(x):
            """docstring."""
            return x + np.log(np.exp(-2 * x) + 1) - np.log(2.0)

        loss = logcosh(y_pred - y_true)
        return _reduce(np.mean(loss, axis=-1), self.reduction, sample_weight)


class MeanAbsoluteError(Loss):
    """Computes the mean of absolute difference between labels and predictions."""

    def __init__(
        self,
        reduction: str = "sum_over_batch_size",
        name: str = "mean_absolute_error",
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(reduction=reduction, name=name, dtype=dtype)

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        """docstring."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        loss = np.abs(y_true - y_pred)
        return _reduce(np.mean(loss, axis=-1), self.reduction, sample_weight)


class MeanAbsolutePercentageError(Loss):
    """Computes the mean absolute percentage error between `y_true` & `y_pred`."""

    def __init__(
        self,
        reduction: str = "sum_over_batch_size",
        name: str = "mean_absolute_percentage_error",
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(reduction=reduction, name=name, dtype=dtype)

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        """docstring."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        loss = 100.0 * np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-7))
        return _reduce(np.mean(loss, axis=-1), self.reduction, sample_weight)


class MeanSquaredError(Loss):
    """Computes the mean of squares of errors between labels and predictions."""

    def __init__(
        self,
        reduction: str = "sum_over_batch_size",
        name: str = "mean_squared_error",
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(reduction=reduction, name=name, dtype=dtype)

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        """docstring."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        loss = np.square(y_true - y_pred)
        return _reduce(np.mean(loss, axis=-1), self.reduction, sample_weight)


class MeanSquaredLogarithmicError(Loss):
    """Computes the mean squared logarithmic error between `y_true` & `y_pred`."""

    def __init__(
        self,
        reduction: str = "sum_over_batch_size",
        name: str = "mean_squared_logarithmic_error",
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(reduction=reduction, name=name, dtype=dtype)

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        """docstring."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        y_true = np.clip(y_true, 0.0, None)
        y_pred = np.clip(y_pred, 0.0, None)

        loss = np.square(np.log(y_true + 1.0) - np.log(y_pred + 1.0))
        return _reduce(np.mean(loss, axis=-1), self.reduction, sample_weight)


class Poisson(Loss):
    """Computes the Poisson loss between `y_true` & `y_pred`."""

    def __init__(
        self,
        reduction: str = "sum_over_batch_size",
        name: str = "poisson",
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(reduction=reduction, name=name, dtype=dtype)

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        """docstring."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        loss = y_pred - y_true * np.log(y_pred + 1e-7)
        return _reduce(np.mean(loss, axis=-1), self.reduction, sample_weight)


class SparseCategoricalCrossentropy(Loss):
    """Computes the crossentropy loss between the labels and predictions."""

    def __init__(
        self,
        from_logits: bool = False,
        reduction: str = "sum_over_batch_size",
        axis: float = -1.0,
        name: str = "sparse_categorical_crossentropy",
        dtype: Optional[str] = None,
        ignore_class: Optional[int] = None,
    ):
        """docstring."""
        super().__init__(reduction=reduction, name=name, dtype=dtype)
        self.from_logits = from_logits
        self.axis = int(axis)
        self.ignore_class = ignore_class

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        """docstring."""
        y_true = np.array(y_true, dtype=int)
        y_pred = np.array(y_pred)

        if self.from_logits:
            y_pred = y_pred - np.max(y_pred, axis=self.axis, keepdims=True)
            y_pred = np.exp(y_pred) / np.sum(
                np.exp(y_pred), axis=self.axis, keepdims=True
            )

        y_pred = np.clip(y_pred, 1e-7, 1.0)

        # advanced masking if ignore_class
        if self.ignore_class is not None:
            valid_mask = y_true != self.ignore_class
        else:
            valid_mask = np.ones_like(y_true, dtype=bool)

        # extract probs
        probs = np.take_along_axis(
            y_pred, np.expand_dims(y_true, axis=self.axis), axis=self.axis
        ).squeeze(axis=self.axis)
        loss = -np.log(probs)

        if self.ignore_class is not None:
            loss = np.where(valid_mask, loss, 0.0)

        return _reduce(loss, self.reduction, sample_weight)


class SquaredHinge(Loss):
    """Computes the squared hinge loss between `y_true` & `y_pred`."""

    def __init__(
        self,
        reduction: str = "sum_over_batch_size",
        name: str = "squared_hinge",
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(reduction=reduction, name=name, dtype=dtype)

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        """docstring."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        y_true = np.where(y_true == 0, -1, y_true)
        loss = np.square(np.maximum(1.0 - y_true * y_pred, 0.0))
        return _reduce(np.mean(loss, axis=-1), self.reduction, sample_weight)


class Tversky(Loss):
    """Computes the Tversky loss value between `y_true` and `y_pred`."""

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        reduction: str = "sum_over_batch_size",
        name: str = "tversky",
        dtype: Optional[str] = None,
        axis: Optional[int] = None,
    ):
        """docstring."""
        super().__init__(reduction=reduction, name=name, dtype=dtype)
        self.alpha = alpha
        self.beta = beta
        self.axis = axis

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        """docstring."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        tp = np.sum(y_true * y_pred, axis=self.axis)
        fp = np.sum((1 - y_true) * y_pred, axis=self.axis)
        fn = np.sum(y_true * (1 - y_pred), axis=self.axis)

        tversky = tp / (tp + self.alpha * fp + self.beta * fn + 1e-7)
        loss = 1 - tversky
        return _reduce(loss, self.reduction, sample_weight)
