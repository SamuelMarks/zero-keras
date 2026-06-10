"""Keras metrics."""

from typing import Any, Optional


def _get_keras_metric(cls_name, **kwargs):
    import keras
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        return getattr(keras.metrics, cls_name)(**kwargs)
    return None


class Metric:
    """Base class for all Keras metrics."""

    def __init__(
        self, name: Optional[str] = None, dtype: Optional[str] = None, **kwargs: Any
    ):
        self.name = name
        self.dtype = dtype
        self._keras_metric = None
        self._keras_class = self.__class__.__name__
        self._kwargs = kwargs

    def update_state(self, *args: Any, **kwargs: Any) -> Any:
        if self._keras_metric is None:
            km = _get_keras_metric(
                self._keras_class, name=self.name, dtype=self.dtype, **self._kwargs
            )
            if km:
                self._keras_metric = km
        if self._keras_metric:
            try:
                return self._keras_metric.update_state(*args, **kwargs)
            except NotImplementedError:
                pass

    def result(self) -> Any:
        if self._keras_metric:
            try:
                return self._keras_metric.result()
            except NotImplementedError:
                pass
        return 0.0

    def reset_state(self) -> None:
        if self._keras_metric:
            self._keras_metric.reset_state()

    def __call__(self, *args, **kwargs):
        self.update_state(*args, **kwargs)
        return self.result()


class Mean(Metric):
    def __init__(self, name="mean", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)


class Sum(Metric):
    def __init__(self, name="sum", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)


class MeanMetricWrapper(Mean):
    def __init__(self, fn, name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, fn=fn, **kwargs)
        self.fn = fn


class Accuracy(Metric):
    def __init__(self, name="accuracy", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)


class BinaryAccuracy(Metric):
    def __init__(self, name="binary_accuracy", dtype=None, threshold=0.5, **kwargs):
        super().__init__(name=name, dtype=dtype, threshold=threshold, **kwargs)


class CategoricalAccuracy(Metric):
    def __init__(self, name="categorical_accuracy", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)


class SparseCategoricalAccuracy(Metric):
    def __init__(self, name="sparse_categorical_accuracy", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)


class AUC(Metric):
    def __init__(
        self,
        num_thresholds=200,
        curve="ROC",
        summation_method="interpolation",
        name="auc",
        dtype=None,
        thresholds=None,
        multi_label=False,
        num_labels=None,
        label_weights=None,
        from_logits=False,
        **kwargs,
    ):
        super().__init__(
            name=name,
            dtype=dtype,
            num_thresholds=num_thresholds,
            curve=curve,
            summation_method=summation_method,
            thresholds=thresholds,
            multi_label=multi_label,
            num_labels=num_labels,
            label_weights=label_weights,
            from_logits=from_logits,
            **kwargs,
        )


class BinaryCrossentropy(Metric):
    def __init__(
        self,
        name="binary_crossentropy",
        dtype=None,
        from_logits=False,
        label_smoothing=0.0,
        **kwargs,
    ):
        super().__init__(
            name=name,
            dtype=dtype,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            **kwargs,
        )


class BinaryIoU(Metric):
    def __init__(
        self,
        target_class_ids=(0, 1),
        threshold=0.5,
        name="binary_io_u",
        dtype=None,
        **kwargs,
    ):
        kwargs.pop("num_classes", None)
        kwargs.pop("precision", None)
        kwargs.pop("recall", None)
        kwargs.pop("specificity", None)
        kwargs.pop("sensitivity", None)
        super().__init__(
            name=name,
            dtype=dtype,
            target_class_ids=target_class_ids,
            threshold=threshold,
            **kwargs,
        )


class CategoricalCrossentropy(Metric):
    def __init__(
        self,
        name="categorical_crossentropy",
        dtype=None,
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        **kwargs,
    ):
        super().__init__(
            name=name,
            dtype=dtype,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
            **kwargs,
        )


class CategoricalHinge(Metric):
    def __init__(self, name="categorical_hinge", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)


class ConcordanceCorrelation(Metric):
    def __init__(self, name="concordance_correlation", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)


class CosineSimilarity(Metric):
    def __init__(self, name="cosine_similarity", dtype=None, axis=-1, **kwargs):
        super().__init__(name=name, dtype=dtype, axis=axis, **kwargs)


class F1Score(Metric):
    def __init__(
        self, average=None, threshold=None, name="f1_score", dtype=None, **kwargs
    ):
        super().__init__(
            name=name, dtype=dtype, average=average, threshold=threshold, **kwargs
        )


class FBetaScore(Metric):
    def __init__(
        self,
        beta=1.0,
        average=None,
        threshold=None,
        name="fbeta_score",
        dtype=None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            dtype=dtype,
            beta=beta,
            average=average,
            threshold=threshold,
            **kwargs,
        )


class FalseNegatives(Metric):
    def __init__(self, thresholds=None, name="false_negatives", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, thresholds=thresholds, **kwargs)


class FalsePositives(Metric):
    def __init__(self, thresholds=None, name="false_positives", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, thresholds=thresholds, **kwargs)


class Hinge(Metric):
    def __init__(self, name="hinge", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)


class IoU(Metric):
    def __init__(
        self,
        num_classes,
        target_class_ids=None,
        name="io_u",
        dtype=None,
        ignore_class=None,
        sparse_y_true=True,
        sparse_y_pred=False,
        axis=-1,
        **kwargs,
    ):
        super().__init__(
            name=name,
            dtype=dtype,
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            ignore_class=ignore_class,
            sparse_y_true=sparse_y_true,
            sparse_y_pred=sparse_y_pred,
            axis=axis,
            **kwargs,
        )


class KLDivergence(Metric):
    def __init__(self, name="kl_divergence", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)


class LogCoshError(Metric):
    def __init__(self, name="log_cosh_error", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)


class MeanAbsoluteError(Metric):
    def __init__(self, name="mean_absolute_error", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)


class MeanAbsolutePercentageError(Metric):
    def __init__(self, name="mean_absolute_percentage_error", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)


class MeanIoU(Metric):
    def __init__(
        self,
        num_classes,
        name="mean_io_u",
        dtype=None,
        ignore_class=None,
        sparse_y_true=True,
        sparse_y_pred=False,
        axis=-1,
        **kwargs,
    ):
        super().__init__(
            name=name,
            dtype=dtype,
            num_classes=num_classes,
            ignore_class=ignore_class,
            sparse_y_true=sparse_y_true,
            sparse_y_pred=sparse_y_pred,
            axis=axis,
            **kwargs,
        )


class MeanSquaredError(Metric):
    def __init__(self, name="mean_squared_error", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)


class MeanSquaredLogarithmicError(Metric):
    def __init__(self, name="mean_squared_logarithmic_error", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)


class OneHotIoU(Metric):
    def __init__(
        self,
        num_classes,
        target_class_ids=None,
        name="one_hot_io_u",
        dtype=None,
        ignore_class=None,
        sparse_y_pred=False,
        axis=-1,
        **kwargs,
    ):
        super().__init__(
            name=name,
            dtype=dtype,
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            ignore_class=ignore_class,
            sparse_y_pred=sparse_y_pred,
            axis=axis,
            **kwargs,
        )


class OneHotMeanIoU(Metric):
    def __init__(
        self,
        num_classes,
        name="one_hot_mean_io_u",
        dtype=None,
        ignore_class=None,
        sparse_y_pred=False,
        axis=-1,
        **kwargs,
    ):
        super().__init__(
            name=name,
            dtype=dtype,
            num_classes=num_classes,
            ignore_class=ignore_class,
            sparse_y_pred=sparse_y_pred,
            axis=axis,
            **kwargs,
        )


class PearsonCorrelation(Metric):
    def __init__(self, name="pearson_correlation", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)


class Poisson(Metric):
    def __init__(self, name="poisson", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)


class Precision(Metric):
    def __init__(
        self,
        thresholds=None,
        top_k=None,
        class_id=None,
        name="precision",
        dtype=None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            dtype=dtype,
            thresholds=thresholds,
            top_k=top_k,
            class_id=class_id,
            **kwargs,
        )


class PrecisionAtRecall(Metric):
    def __init__(
        self,
        recall,
        num_thresholds=200,
        class_id=None,
        name="precision_at_recall",
        dtype=None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            dtype=dtype,
            recall=recall,
            num_thresholds=num_thresholds,
            class_id=class_id,
            **kwargs,
        )


class R2Score(Metric):
    def __init__(
        self,
        class_aggregation="uniform_average",
        num_classes=None,
        name="r2_score",
        dtype=None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            dtype=dtype,
            class_aggregation=class_aggregation,
            num_classes=num_classes,
            **kwargs,
        )


class Recall(Metric):
    def __init__(
        self,
        thresholds=None,
        top_k=None,
        class_id=None,
        name="recall",
        dtype=None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            dtype=dtype,
            thresholds=thresholds,
            top_k=top_k,
            class_id=class_id,
            **kwargs,
        )


class RecallAtPrecision(Metric):
    def __init__(
        self,
        precision,
        num_thresholds=200,
        class_id=None,
        name="recall_at_precision",
        dtype=None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            dtype=dtype,
            precision=precision,
            num_thresholds=num_thresholds,
            class_id=class_id,
            **kwargs,
        )


class RootMeanSquaredError(Metric):
    def __init__(self, name="root_mean_squared_error", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)


class SensitivityAtSpecificity(Metric):
    def __init__(
        self,
        specificity,
        num_thresholds=200,
        class_id=None,
        name="sensitivity_at_specificity",
        dtype=None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            dtype=dtype,
            specificity=specificity,
            num_thresholds=num_thresholds,
            class_id=class_id,
            **kwargs,
        )


class SparseCategoricalCrossentropy(Metric):
    def __init__(
        self,
        name="sparse_categorical_crossentropy",
        dtype=None,
        from_logits=False,
        ignore_class=None,
        axis=-1,
        **kwargs,
    ):
        super().__init__(
            name=name, dtype=dtype, from_logits=from_logits, axis=axis, **kwargs
        )


class SparseTopKCategoricalAccuracy(Metric):
    def __init__(
        self, k=5, name="sparse_top_k_categorical_accuracy", dtype=None, **kwargs
    ):
        super().__init__(name=name, dtype=dtype, k=k, **kwargs)


class SpecificityAtSensitivity(Metric):
    def __init__(
        self,
        sensitivity,
        num_thresholds=200,
        class_id=None,
        name="specificity_at_sensitivity",
        dtype=None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            dtype=dtype,
            sensitivity=sensitivity,
            num_thresholds=num_thresholds,
            class_id=class_id,
            **kwargs,
        )


class SquaredHinge(Metric):
    def __init__(self, name="squared_hinge", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)


class TopKCategoricalAccuracy(Metric):
    def __init__(self, k=5, name="top_k_categorical_accuracy", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, k=k, **kwargs)


class TrueNegatives(Metric):
    def __init__(self, thresholds=None, name="true_negatives", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, thresholds=thresholds, **kwargs)


class TruePositives(Metric):
    def __init__(self, thresholds=None, name="true_positives", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, thresholds=thresholds, **kwargs)
