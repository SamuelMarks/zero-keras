"""Keras losses."""

import numpy as np
from typing import Any, Optional


def _reduce(loss: Any, reduction: str, sample_weight: Optional[Any] = None) -> Any:
    if reduction == "none":
        return np.array(loss)
    if reduction == "sum":
        return np.sum(loss)
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

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        return _reduce(np.float64(0.0), self.reduction, sample_weight)


class BinaryCrossentropy(Loss):
    def __init__(
        self,
        from_logits: bool = False,
        label_smoothing: float = 0.0,
        axis: float = -1.0,
        reduction: str = "sum_over_batch_size",
        name: str = "binary_crossentropy",
        dtype: Optional[str] = None,
    ):
        super().__init__(reduction=reduction, name=name, dtype=dtype)

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        return _reduce(np.float64(0.0), self.reduction, sample_weight)


class MeanSquaredError(Loss):
    def __init__(
        self,
        reduction: str = "sum_over_batch_size",
        name: str = "mean_squared_error",
        dtype: Optional[str] = None,
    ):
        super().__init__(reduction=reduction, name=name, dtype=dtype)

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        return _reduce(np.float64(0.0), self.reduction, sample_weight)


class BinaryFocalCrossentropy(Loss):
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
        super().__init__(reduction=reduction, name=name, dtype=dtype)

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        return _reduce(np.float64(0.0), self.reduction, sample_weight)


class CTC(Loss):
    pass


class CategoricalCrossentropy(Loss):
    def __init__(
        self,
        from_logits: bool = False,
        label_smoothing: float = 0.0,
        axis: float = -1.0,
        reduction: str = "sum_over_batch_size",
        name: str = "categorical_crossentropy",
        dtype: Optional[str] = None,
    ):
        super().__init__(reduction=reduction, name=name, dtype=dtype)

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        return _reduce(np.float64(0.0), self.reduction, sample_weight)


class CategoricalFocalCrossentropy(Loss):
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
        super().__init__(reduction=reduction, name=name, dtype=dtype)

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        return _reduce(np.float64(0.0), self.reduction, sample_weight)


class CategoricalGeneralizedCrossEntropy(Loss):
    pass


class CategoricalHinge(Loss):
    pass


class Circle(Loss):
    pass


class CosineSimilarity(Loss):
    pass


class Dice(Loss):
    pass


class Hinge(Loss):
    pass


class Huber(Loss):
    pass


class KLDivergence(Loss):
    pass


class LogCosh(Loss):
    pass


class MeanAbsoluteError(Loss):
    pass


class MeanAbsolutePercentageError(Loss):
    pass


class MeanSquaredLogarithmicError(Loss):
    pass


class Poisson(Loss):
    pass


class SparseCategoricalCrossentropy(Loss):
    def __init__(
        self,
        from_logits: bool = False,
        reduction: str = "sum_over_batch_size",
        axis: float = -1.0,
        name: str = "sparse_categorical_crossentropy",
        dtype: Optional[str] = None,
        ignore_class: Optional[int] = None,
    ):
        super().__init__(reduction=reduction, name=name, dtype=dtype)

    def __call__(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        return _reduce(np.float64(0.0), self.reduction, sample_weight)


class SquaredHinge(Loss):
    pass


class Tversky(Loss):
    pass
