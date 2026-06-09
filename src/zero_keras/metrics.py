"""Keras metrics."""

import numpy as np
from typing import Any, Optional, Tuple


class Metric:
    """Base class for all Keras metrics."""

    def __init__(
        self, name: Optional[str] = None, dtype: Optional[str] = None, **kwargs: Any
    ):
        """docstring."""
        self.name = name
        self.dtype = dtype
        self._val = 0.0

    def update_state(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> None:
        """docstring."""
        self._val = float(np.mean(y_pred))  # Mock state update

    def result(self) -> float:
        """docstring."""
        return self._val


class AUC(Metric):
    """Approximates the AUC (Area under the curve) of the ROC or PR curves."""

    def __init__(
        self,
        num_thresholds: Optional[int] = 200,
        curve: str = "ROC",
        summation_method: str = "interpolation",
        name: Optional[str] = None,
        dtype: Optional[str] = None,
        thresholds: Optional[Any] = None,
        multi_label: bool = False,
        num_labels: Optional[int] = None,
        label_weights: Optional[Any] = None,
        from_logits: bool = False,
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.num_thresholds = num_thresholds
        self.curve = curve
        self.summation_method = summation_method
        self.thresholds = thresholds
        self.multi_label = multi_label
        self.num_labels = num_labels
        self.label_weights = label_weights
        self.from_logits = from_logits


class Accuracy(Metric):
    """Calculates how often predictions equal labels."""

    def __init__(self, name: str = "accuracy", dtype: Optional[str] = None):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.correct = 0.0
        self.total = 0.0

    def update_state(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> None:
        """docstring."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        matches = y_true == y_pred
        if sample_weight is not None:
            matches = matches * np.array(sample_weight)
            self.total += np.sum(sample_weight)
        else:
            self.total += matches.size
        self.correct += np.sum(matches)

    def result(self) -> float:
        """docstring."""
        return float(self.correct / max(self.total, 1e-7))


class BinaryAccuracy(Accuracy):
    """Calculates how often predictions match binary labels."""

    def __init__(
        self,
        name: str = "binary_accuracy",
        dtype: Optional[str] = None,
        threshold: float = 0.5,
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.threshold = threshold

    def update_state(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> None:
        """docstring."""
        y_pred = (np.array(y_pred) > self.threshold).astype(int)
        super().update_state(y_true, y_pred, sample_weight)


class BinaryCrossentropy(Metric):
    """Computes the crossentropy metric between the labels and predictions."""

    def __init__(
        self,
        name: str = "binary_crossentropy",
        dtype: Optional[str] = None,
        from_logits: bool = False,
        label_smoothing: float = 0.0,
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing


class BinaryIoU(Metric):
    """Computes the Intersection-Over-Union metric for class 0 and/or 1."""

    def __init__(
        self,
        target_class_ids: Tuple[int, ...] = (0, 1),
        threshold: float = 0.5,
        name: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.target_class_ids = target_class_ids
        self.threshold = threshold


class CategoricalAccuracy(Accuracy):
    """Calculates how often predictions match one-hot labels."""

    def __init__(self, name: str = "categorical_accuracy", dtype: Optional[str] = None):
        """docstring."""
        super().__init__(name=name, dtype=dtype)

    def update_state(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> None:
        """docstring."""
        y_true = np.argmax(np.array(y_true), axis=-1)
        y_pred = np.argmax(np.array(y_pred), axis=-1)
        super().update_state(y_true, y_pred, sample_weight)


class CategoricalCrossentropy(Metric):
    """Computes the crossentropy metric between the labels and predictions."""

    def __init__(
        self,
        name: str = "categorical_crossentropy",
        dtype: Optional[str] = None,
        from_logits: bool = False,
        label_smoothing: float = 0.0,
        axis: float = -1.0,
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.axis = int(axis)


class CategoricalHinge(Metric):
    """Computes the categorical hinge metric between `y_true` and `y_pred`."""

    def __init__(self, name: str = "categorical_hinge", dtype: Optional[str] = None):
        """docstring."""
        super().__init__(name=name, dtype=dtype)


class ConcordanceCorrelation(Metric):
    """Calculates the Concordance Correlation Coefficient (CCC)."""

    def __init__(
        self,
        name: str = "concordance_correlation",
        dtype: Optional[str] = None,
        axis: float = -1.0,
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.axis = int(axis)


class CosineSimilarity(Metric):
    """Computes the cosine similarity between the labels and predictions."""

    def __init__(
        self,
        name: str = "cosine_similarity",
        dtype: Optional[str] = None,
        axis: float = -1.0,
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.axis = int(axis)


class F1Score(Metric):
    """Computes F-1 Score."""

    def __init__(
        self,
        average: Optional[str] = None,
        threshold: Optional[float] = None,
        name: str = "f1_score",
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.average = average
        self.threshold = threshold


class FBetaScore(Metric):
    """Computes F-Beta score."""

    def __init__(
        self,
        average: Optional[str] = None,
        beta: float = 1.0,
        threshold: Optional[float] = None,
        name: str = "fbeta_score",
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.average = average
        self.beta = beta
        self.threshold = threshold


class FalseNegatives(Metric):
    """Calculates the number of false negatives."""

    def __init__(
        self,
        thresholds: Optional[float] = 0.5,
        name: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.thresholds = thresholds


class FalsePositives(Metric):
    """Calculates the number of false positives."""

    def __init__(
        self,
        thresholds: Optional[float] = 0.5,
        name: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.thresholds = thresholds


class Hinge(Metric):
    """Computes the hinge metric between `y_true` and `y_pred`."""

    def __init__(self, name: str = "hinge", dtype: Optional[str] = None):
        """docstring."""
        super().__init__(name=name, dtype=dtype)


class IoU(Metric):
    """Computes the Intersection-Over-Union metric for specific target classes."""

    def __init__(
        self,
        num_classes: int,
        target_class_ids: Any,
        name: Optional[str] = None,
        dtype: Optional[str] = None,
        ignore_class: Optional[int] = None,
        sparse_y_true: bool = True,
        sparse_y_pred: bool = True,
        axis: float = -1.0,
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.target_class_ids = target_class_ids
        self.ignore_class = ignore_class
        self.sparse_y_true = sparse_y_true
        self.sparse_y_pred = sparse_y_pred
        self.axis = int(axis)


class KLDivergence(Metric):
    """Computes Kullback-Leibler divergence metric between `y_true` and `y_pred`."""

    def __init__(self, name: str = "kl_divergence", dtype: Optional[str] = None):
        """docstring."""
        super().__init__(name=name, dtype=dtype)


class LogCoshError(Metric):
    """Computes the logarithm of the hyperbolic cosine of the prediction error."""

    def __init__(self, name: str = "logcosh", dtype: Optional[str] = None):
        """docstring."""
        super().__init__(name=name, dtype=dtype)


class Mean(Metric):
    """Compute the (weighted) mean of the given values."""

    def __init__(self, name: str = "mean", dtype: Optional[str] = None):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.total = 0.0
        self.count = 0.0

    def update_state(self, values: Any, sample_weight: Optional[Any] = None) -> None:
        """docstring."""
        values = np.array(values)
        if sample_weight is not None:
            values = values * np.array(sample_weight)
            self.count += np.sum(sample_weight)
        else:
            self.count += values.size
        self.total += np.sum(values)

    def result(self) -> float:
        """docstring."""
        return float(self.total / max(self.count, 1e-7))


class MeanAbsoluteError(Metric):
    """Computes the mean absolute error between the labels and predictions."""

    def __init__(self, name: str = "mean_absolute_error", dtype: Optional[str] = None):
        """docstring."""
        super().__init__(name=name, dtype=dtype)


class MeanAbsolutePercentageError(Metric):
    """Computes mean absolute percentage error between `y_true` and `y_pred`."""

    def __init__(
        self, name: str = "mean_absolute_percentage_error", dtype: Optional[str] = None
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)


class MeanIoU(Metric):
    """Computes the mean Intersection-Over-Union metric."""

    def __init__(
        self,
        num_classes: int,
        name: Optional[str] = None,
        dtype: Optional[str] = None,
        ignore_class: Optional[int] = None,
        sparse_y_true: bool = True,
        sparse_y_pred: bool = True,
        axis: float = -1.0,
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.ignore_class = ignore_class
        self.sparse_y_true = sparse_y_true
        self.sparse_y_pred = sparse_y_pred
        self.axis = int(axis)


class MeanMetricWrapper(Mean):
    """Wrap a stateless metric function with the `Mean` metric."""

    def __init__(
        self,
        fn: Any,
        name: Optional[str] = None,
        dtype: Optional[str] = None,
        **kwargs: Any,
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.fn = fn
        self.kwargs = kwargs

    def update_state(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> None:
        """docstring."""
        val = self.fn(y_true, y_pred, **self.kwargs)
        super().update_state(val, sample_weight)


class MeanSquaredError(Metric):
    """Computes the mean squared error between `y_true` and `y_pred`."""

    def __init__(self, name: str = "mean_squared_error", dtype: Optional[str] = None):
        """docstring."""
        super().__init__(name=name, dtype=dtype)


class MeanSquaredLogarithmicError(Metric):
    """Computes mean squared logarithmic error between `y_true` and `y_pred`."""

    def __init__(
        self, name: str = "mean_squared_logarithmic_error", dtype: Optional[str] = None
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)


class OneHotIoU(Metric):
    """Computes the Intersection-Over-Union metric for one-hot encoded labels."""

    def __init__(
        self,
        num_classes: int,
        target_class_ids: Any,
        name: Optional[str] = None,
        dtype: Optional[str] = None,
        ignore_class: Optional[int] = None,
        sparse_y_pred: bool = False,
        axis: float = -1.0,
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.target_class_ids = target_class_ids
        self.ignore_class = ignore_class
        self.sparse_y_pred = sparse_y_pred
        self.axis = int(axis)


class OneHotMeanIoU(Metric):
    """Computes mean Intersection-Over-Union metric for one-hot encoded labels."""

    def __init__(
        self,
        num_classes: int,
        name: Optional[str] = None,
        dtype: Optional[str] = None,
        ignore_class: Optional[int] = None,
        sparse_y_pred: bool = False,
        axis: float = -1.0,
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.ignore_class = ignore_class
        self.sparse_y_pred = sparse_y_pred
        self.axis = int(axis)


class PearsonCorrelation(Metric):
    """Calculates the Pearson Correlation Coefficient (PCC)."""

    def __init__(
        self,
        name: str = "pearson_correlation",
        dtype: Optional[str] = None,
        axis: float = -1.0,
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.axis = int(axis)


class Poisson(Metric):
    """Computes the Poisson metric between `y_true` and `y_pred`."""

    def __init__(self, name: str = "poisson", dtype: Optional[str] = None):
        """docstring."""
        super().__init__(name=name, dtype=dtype)


class Precision(Metric):
    """Computes the precision of the predictions with respect to the labels."""

    def __init__(
        self,
        thresholds: Optional[float] = None,
        top_k: Optional[int] = None,
        class_id: Optional[int] = None,
        name: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id


class PrecisionAtRecall(Metric):
    """Computes best precision where recall is >= specified value."""

    def __init__(
        self,
        recall: float,
        num_thresholds: Optional[int] = 200,
        class_id: Optional[int] = None,
        name: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.recall = recall
        self.num_thresholds = num_thresholds
        self.class_id = class_id


class R2Score(Metric):
    """Computes R2 score."""

    def __init__(
        self,
        class_aggregation: str = "uniform_average",
        num_regressors: int = 0,
        name: str = "r2_score",
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.class_aggregation = class_aggregation
        self.num_regressors = num_regressors


class Recall(Metric):
    """Computes the recall of the predictions with respect to the labels."""

    def __init__(
        self,
        thresholds: Optional[float] = None,
        top_k: Optional[int] = None,
        class_id: Optional[int] = None,
        name: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id


class RecallAtPrecision(Metric):
    """Computes best recall where precision is >= specified value."""

    def __init__(
        self,
        precision: float,
        num_thresholds: Optional[int] = 200,
        class_id: Optional[int] = None,
        name: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.precision = precision
        self.num_thresholds = num_thresholds
        self.class_id = class_id


class RootMeanSquaredError(Metric):
    """Computes root mean squared error metric between `y_true` and `y_pred`."""

    def __init__(
        self, name: str = "root_mean_squared_error", dtype: Optional[str] = None
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)


class SensitivityAtSpecificity(Metric):
    """Computes best sensitivity where specificity is >= specified value."""

    def __init__(
        self,
        specificity: float,
        num_thresholds: Optional[int] = 200,
        class_id: Optional[int] = None,
        name: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.specificity = specificity
        self.num_thresholds = num_thresholds
        self.class_id = class_id


class SparseCategoricalAccuracy(Accuracy):
    """Calculates how often predictions match integer labels."""

    def __init__(
        self, name: str = "sparse_categorical_accuracy", dtype: Optional[str] = None
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)

    def update_state(
        self, y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None
    ) -> None:
        """docstring."""
        y_pred = np.argmax(np.array(y_pred), axis=-1)
        super().update_state(y_true, y_pred, sample_weight)


class SparseCategoricalCrossentropy(Metric):
    """Computes the crossentropy metric between the labels and predictions."""

    def __init__(
        self,
        name: str = "sparse_categorical_crossentropy",
        dtype: Optional[str] = None,
        from_logits: bool = False,
        axis: float = -1.0,
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.from_logits = from_logits
        self.axis = int(axis)


class SparseTopKCategoricalAccuracy(Metric):
    """Computes how often integer targets are in the top `K` predictions."""

    def __init__(
        self,
        k: Optional[int] = 5,
        name: str = "sparse_top_k_categorical_accuracy",
        dtype: Optional[str] = None,
        from_sorted_ids: bool = False,
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.k = k
        self.from_sorted_ids = from_sorted_ids


class SpecificityAtSensitivity(Metric):
    """Computes best specificity where sensitivity is >= specified value."""

    def __init__(
        self,
        sensitivity: float,
        num_thresholds: Optional[int] = 200,
        class_id: Optional[int] = None,
        name: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.sensitivity = sensitivity
        self.num_thresholds = num_thresholds
        self.class_id = class_id


class SquaredHinge(Metric):
    """Computes the hinge metric between `y_true` and `y_pred`."""

    def __init__(self, name: str = "squared_hinge", dtype: Optional[str] = None):
        """docstring."""
        super().__init__(name=name, dtype=dtype)


class Sum(Metric):
    """Compute the (weighted) sum of the given values."""

    def __init__(self, name: str = "sum", dtype: Optional[str] = None):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.total = 0.0

    def update_state(self, values: Any, sample_weight: Optional[Any] = None) -> None:
        """docstring."""
        values = np.array(values)
        if sample_weight is not None:
            values = values * np.array(sample_weight)
        self.total += np.sum(values)

    def result(self) -> float:
        """docstring."""
        return float(self.total)


class TopKCategoricalAccuracy(Metric):
    """Computes how often targets are in the top `K` predictions."""

    def __init__(
        self,
        k: Optional[int] = 5,
        name: str = "top_k_categorical_accuracy",
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.k = k


class TrueNegatives(Metric):
    """Calculates the number of true negatives."""

    def __init__(
        self,
        thresholds: Optional[float] = 0.5,
        name: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.thresholds = thresholds


class TruePositives(Metric):
    """Calculates the number of true positives."""

    def __init__(
        self,
        thresholds: Optional[float] = 0.5,
        name: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        """docstring."""
        super().__init__(name=name, dtype=dtype)
        self.thresholds = thresholds
