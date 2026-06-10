"""Keras metrics."""

import numpy as np
from typing import Any, Optional


class Metric:
    """Base class for all Keras metrics."""

    def __init__(
        self, name: Optional[str] = None, dtype: Optional[str] = None, **kwargs: Any
    ):
        self.name = name

    def update_state(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> None:
        pass

    def result(self) -> float:
        return 0.0


class Mean(Metric):
    def __init__(self, name: str = "mean", dtype: Optional[str] = None):
        super().__init__(name=name, dtype=dtype)
        self.total = 0.0
        self.count = 0.0

    def update_state(
        self, values: Any, sample_weight: Optional[Any] = None, **kwargs
    ) -> None:
        v = np.array(values)
        if sample_weight is not None:
            sw = np.array(sample_weight)
            self.total += np.sum(v * sw)
            self.count += np.sum(sw)
        else:
            self.total += np.sum(v)
            self.count += np.prod(v.shape)

    def result(self) -> float:
        if self.count == 0:
            return 0.0
        return float(self.total / self.count)


class Sum(Metric):
    def __init__(self, name: str = "sum", dtype: Optional[str] = None):
        super().__init__(name=name, dtype=dtype)
        self.total = 0.0

    def update_state(
        self, values: Any, sample_weight: Optional[Any] = None, **kwargs
    ) -> None:
        v = np.array(values)
        if sample_weight is not None:
            sw = np.array(sample_weight)
            self.total += np.sum(v * sw)
        else:
            self.total += np.sum(v)

    def result(self) -> float:
        return float(self.total)


class MeanMetricWrapper(Mean):
    def __init__(
        self,
        fn: Any,
        name: Optional[str] = None,
        dtype: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name, dtype=dtype)
        self.fn = fn
        self.kwargs = kwargs

    def update_state(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> None:
        values = self.fn(y_true, y_pred, **self.kwargs)
        super().update_state(values, sample_weight=sample_weight)


class Accuracy(Metric):
    def __init__(self, name: str = "accuracy", dtype: Optional[str] = None):
        super().__init__(name=name, dtype=dtype)
        self.correct = 0.0
        self.total = 0.0

    def update_state(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> None:
        y_t = np.array(y_true)
        y_p = np.array(y_pred)
        matches = y_t == y_p
        if sample_weight is not None:
            sw = np.array(sample_weight)
            self.correct += np.sum(matches * sw)
            self.total += np.sum(sw)
        else:
            self.correct += np.sum(matches)
            self.total += np.prod(y_t.shape)

    def result(self) -> float:
        if self.total == 0:
            return 0.0
        return float(self.correct / self.total)


class BinaryAccuracy(Accuracy):
    def __init__(
        self,
        name: str = "binary_accuracy",
        dtype: Optional[str] = None,
        threshold: float = 0.5,
    ):
        super().__init__(name=name, dtype=dtype)
        self.threshold = threshold

    def update_state(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> None:
        y_p = (np.array(y_pred) > self.threshold).astype(int)
        super().update_state(y_true, y_p, sample_weight)


class CategoricalAccuracy(Metric):
    def __init__(self, name: str = "categorical_accuracy", dtype: Optional[str] = None):
        super().__init__(name=name, dtype=dtype)
        self.correct = 0.0
        self.total = 0.0

    def update_state(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> None:
        y_t = np.argmax(np.array(y_true), axis=-1)
        y_p = np.argmax(np.array(y_pred), axis=-1)
        matches = y_t == y_p
        if sample_weight is not None:
            sw = np.array(sample_weight)
            self.correct += np.sum(matches * sw)
            self.total += np.sum(sw)
        else:
            self.correct += np.sum(matches)
            self.total += np.prod(y_t.shape)

    def result(self) -> float:
        if self.total == 0:
            return 0.0
        return float(self.correct / self.total)


class SparseCategoricalAccuracy(Metric):
    def __init__(
        self, name: str = "sparse_categorical_accuracy", dtype: Optional[str] = None
    ):
        super().__init__(name=name, dtype=dtype)
        self.correct = 0.0
        self.total = 0.0

    def update_state(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> None:
        y_t = np.array(y_true)
        if len(y_t.shape) > 1 and y_t.shape[-1] == 1:
            y_t = np.squeeze(y_t, axis=-1)
        y_p = np.argmax(np.array(y_pred), axis=-1)
        matches = y_t == y_p
        if sample_weight is not None:
            sw = np.array(sample_weight)
            self.correct += np.sum(matches * sw)
            self.total += np.sum(sw)
        else:
            self.correct += np.sum(matches)
            self.total += np.prod(y_t.shape)

    def result(self) -> float:
        if self.total == 0:
            return 0.0
        return float(self.correct / self.total)


class AUC(Metric):
    pass


class BinaryCrossentropy(Metric):
    pass


class BinaryIoU(Metric):
    pass


class CategoricalCrossentropy(Metric):
    pass


class CategoricalHinge(Metric):
    pass


class ConcordanceCorrelation(Metric):
    pass


class CosineSimilarity(Metric):
    pass


class F1Score(Metric):
    pass


class FBetaScore(Metric):
    pass


class FalseNegatives(Metric):
    pass


class FalsePositives(Metric):
    pass


class Hinge(Metric):
    pass


class IoU(Metric):
    pass


class KLDivergence(Metric):
    pass


class LogCoshError(Metric):
    pass


class MeanAbsoluteError(Metric):
    pass


class MeanAbsolutePercentageError(Metric):
    pass


class MeanIoU(Metric):
    pass


class MeanSquaredError(Metric):
    pass


class MeanSquaredLogarithmicError(Metric):
    pass


class OneHotIoU(Metric):
    pass


class OneHotMeanIoU(Metric):
    pass


class PearsonCorrelation(Metric):
    pass


class Poisson(Metric):
    pass


class Precision(Metric):
    pass


class PrecisionAtRecall(Metric):
    pass


class R2Score(Metric):
    pass


class Recall(Metric):
    pass


class RecallAtPrecision(Metric):
    pass


class RootMeanSquaredError(Metric):
    pass


class SensitivityAtSpecificity(Metric):
    pass


class SparseCategoricalCrossentropy(Metric):
    pass


class SparseTopKCategoricalAccuracy(Metric):
    pass


class SpecificityAtSensitivity(Metric):
    pass


class SquaredHinge(Metric):
    pass


class TopKCategoricalAccuracy(Metric):
    pass


class TrueNegatives(Metric):
    pass


class TruePositives(Metric):
    pass
