import ml_switcheroo.nn.metrics as metrics_impl

"""Keras metrics."""

from typing import Any, Optional


class Metric:
    def __init__(self, name: str = "", dtype: Optional[str] = None, **kwargs: Any):
        self.name = name
        self.dtype = dtype
        self._kwargs = kwargs
        self._impl = metrics_impl.Metric(name=name, dtype=dtype, **kwargs)

    def update_state(self, *args, **kwargs):
        self._impl.update_state(*args, **kwargs)

    def result(self):
        return self._impl.result()

    def reset_state(self):
        self._impl.reset_state()

    def __call__(self, *args, **kwargs):
        self.update_state(*args, **kwargs)
        return self.result()


class Mean(Metric):
    def __init__(self, name="mean", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.Mean(name=name, dtype=dtype, **kwargs)


class Sum(Metric):
    def __init__(self, name="sum", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.Sum(name=name, dtype=dtype, **kwargs)


class MeanMetricWrapper(Mean):
    def __init__(self, fn, name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.MeanMetricWrapper(
            fn=fn, name=name, dtype=dtype, **kwargs
        )


class Accuracy(MeanMetricWrapper):
    def __init__(self, name="accuracy", dtype=None, **kwargs):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.Accuracy(name=name, dtype=dtype, **kwargs)


class BinaryAccuracy(MeanMetricWrapper):
    def __init__(self, name="binary_accuracy", dtype=None, threshold=0.5, **kwargs):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.BinaryAccuracy(
            name=name, dtype=dtype, threshold=threshold, **kwargs
        )


class CategoricalAccuracy(MeanMetricWrapper):
    def __init__(self, name="categorical_accuracy", dtype=None, **kwargs):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.CategoricalAccuracy(name=name, dtype=dtype, **kwargs)


class SparseCategoricalAccuracy(MeanMetricWrapper):
    def __init__(self, name="sparse_categorical_accuracy", dtype=None, **kwargs):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.SparseCategoricalAccuracy(
            name=name, dtype=dtype, **kwargs
        )


class TopKCategoricalAccuracy(MeanMetricWrapper):
    def __init__(self, k=5, name="top_k_categorical_accuracy", dtype=None, **kwargs):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.TopKCategoricalAccuracy(
            k=k, name=name, dtype=dtype, **kwargs
        )


class SparseTopKCategoricalAccuracy(MeanMetricWrapper):
    def __init__(
        self,
        k=5,
        name="sparse_top_k_categorical_accuracy",
        dtype=None,
        from_sorted_ids=False,
        **kwargs,
    ):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.SparseTopKCategoricalAccuracy(
            k=k, name=name, dtype=dtype, from_sorted_ids=from_sorted_ids, **kwargs
        )


class FalsePositives(Metric):
    def __init__(self, thresholds=None, name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.FalsePositives(
            thresholds=thresholds, name=name, dtype=dtype, **kwargs
        )


class FalseNegatives(Metric):
    def __init__(self, thresholds=None, name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.FalseNegatives(
            thresholds=thresholds, name=name, dtype=dtype, **kwargs
        )


class TrueNegatives(Metric):
    def __init__(self, thresholds=None, name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.TrueNegatives(
            thresholds=thresholds, name=name, dtype=dtype, **kwargs
        )


class TruePositives(Metric):
    def __init__(self, thresholds=None, name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.TruePositives(
            thresholds=thresholds, name=name, dtype=dtype, **kwargs
        )


class Precision(Metric):
    def __init__(
        self,
        thresholds=None,
        top_k=None,
        class_id=None,
        name=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.Precision(
            thresholds=thresholds,
            top_k=top_k,
            class_id=class_id,
            name=name,
            dtype=dtype,
            **kwargs,
        )


class Recall(Metric):
    def __init__(
        self,
        thresholds=None,
        top_k=None,
        class_id=None,
        name=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.Recall(
            thresholds=thresholds,
            top_k=top_k,
            class_id=class_id,
            name=name,
            dtype=dtype,
            **kwargs,
        )


class PrecisionAtRecall(Metric):
    def __init__(
        self, recall, num_thresholds=200, class_id=None, name=None, dtype=None, **kwargs
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.PrecisionAtRecall(
            recall=recall,
            num_thresholds=num_thresholds,
            class_id=class_id,
            name=name,
            dtype=dtype,
            **kwargs,
        )


class RecallAtPrecision(Metric):
    def __init__(
        self,
        precision,
        num_thresholds=200,
        class_id=None,
        name=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.RecallAtPrecision(
            precision=precision,
            num_thresholds=num_thresholds,
            class_id=class_id,
            name=name,
            dtype=dtype,
            **kwargs,
        )


class SensitivityAtSpecificity(Metric):
    def __init__(
        self,
        specificity,
        num_thresholds=200,
        class_id=None,
        name=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.SensitivityAtSpecificity(
            specificity=specificity,
            num_thresholds=num_thresholds,
            class_id=class_id,
            name=name,
            dtype=dtype,
            **kwargs,
        )


class SpecificityAtSensitivity(Metric):
    def __init__(
        self,
        sensitivity,
        num_thresholds=200,
        class_id=None,
        name=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.SpecificityAtSensitivity(
            sensitivity=sensitivity,
            num_thresholds=num_thresholds,
            class_id=class_id,
            name=name,
            dtype=dtype,
            **kwargs,
        )


class AUC(Metric):
    def __init__(
        self,
        num_thresholds=200,
        curve="ROC",
        summation_method="interpolation",
        name=None,
        dtype=None,
        thresholds=None,
        multi_label=False,
        num_labels=None,
        label_weights=None,
        from_logits=False,
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.AUC(
            num_thresholds=num_thresholds,
            curve=curve,
            summation_method=summation_method,
            name=name,
            dtype=dtype,
            thresholds=thresholds,
            multi_label=multi_label,
            num_labels=num_labels,
            label_weights=label_weights,
            from_logits=from_logits,
            **kwargs,
        )


class CosineSimilarity(MeanMetricWrapper):
    def __init__(self, name="cosine_similarity", dtype=None, axis=-1, **kwargs):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.CosineSimilarity(
            name=name, dtype=dtype, axis=axis, **kwargs
        )


class MeanAbsoluteError(MeanMetricWrapper):
    def __init__(self, name="mean_absolute_error", dtype=None, **kwargs):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.MeanAbsoluteError(name=name, dtype=dtype, **kwargs)


class MeanAbsolutePercentageError(MeanMetricWrapper):
    def __init__(self, name="mean_absolute_percentage_error", dtype=None, **kwargs):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.MeanAbsolutePercentageError(
            name=name, dtype=dtype, **kwargs
        )


class MeanSquaredError(MeanMetricWrapper):
    def __init__(self, name="mean_squared_error", dtype=None, **kwargs):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.MeanSquaredError(name=name, dtype=dtype, **kwargs)


class MeanSquaredLogarithmicError(MeanMetricWrapper):
    def __init__(self, name="mean_squared_logarithmic_error", dtype=None, **kwargs):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.MeanSquaredLogarithmicError(
            name=name, dtype=dtype, **kwargs
        )


class Hinge(MeanMetricWrapper):
    def __init__(self, name="hinge", dtype=None, **kwargs):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.Hinge(name=name, dtype=dtype, **kwargs)


class SquaredHinge(MeanMetricWrapper):
    def __init__(self, name="squared_hinge", dtype=None, **kwargs):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.SquaredHinge(name=name, dtype=dtype, **kwargs)


class CategoricalHinge(MeanMetricWrapper):
    def __init__(self, name="categorical_hinge", dtype=None, **kwargs):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.CategoricalHinge(name=name, dtype=dtype, **kwargs)


class RootMeanSquaredError(Mean):
    def __init__(self, name="root_mean_squared_error", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.RootMeanSquaredError(name=name, dtype=dtype, **kwargs)


class CategoricalCrossentropy(MeanMetricWrapper):
    def __init__(
        self,
        name="categorical_crossentropy",
        dtype=None,
        from_logits=False,
        label_smoothing=0,
        axis=-1,
        **kwargs,
    ):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.CategoricalCrossentropy(
            name=name,
            dtype=dtype,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
            **kwargs,
        )


class SparseCategoricalCrossentropy(MeanMetricWrapper):
    def __init__(
        self,
        name="sparse_categorical_crossentropy",
        dtype=None,
        from_logits=False,
        axis=-1,
        **kwargs,
    ):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.SparseCategoricalCrossentropy(
            name=name, dtype=dtype, from_logits=from_logits, axis=axis, **kwargs
        )


class BinaryCrossentropy(MeanMetricWrapper):
    def __init__(
        self,
        name="binary_crossentropy",
        dtype=None,
        from_logits=False,
        label_smoothing=0,
        **kwargs,
    ):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.BinaryCrossentropy(
            name=name,
            dtype=dtype,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            **kwargs,
        )


class KLDivergence(MeanMetricWrapper):
    def __init__(self, name="kl_divergence", dtype=None, **kwargs):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.KLDivergence(name=name, dtype=dtype, **kwargs)


class Poisson(MeanMetricWrapper):
    def __init__(self, name="poisson", dtype=None, **kwargs):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.Poisson(name=name, dtype=dtype, **kwargs)


class LogCoshError(MeanMetricWrapper):
    def __init__(self, name="logcosh", dtype=None, **kwargs):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.LogCoshError(name=name, dtype=dtype, **kwargs)


class MeanIoU(Metric):
    def __init__(
        self,
        num_classes,
        name=None,
        dtype=None,
        ignore_class=None,
        sparse_y_true=True,
        sparse_y_pred=True,
        axis=-1,
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.MeanIoU(
            num_classes=num_classes,
            name=name,
            dtype=dtype,
            ignore_class=ignore_class,
            sparse_y_true=sparse_y_true,
            sparse_y_pred=sparse_y_pred,
            axis=axis,
            **kwargs,
        )


class IoU(Metric):
    def __init__(
        self,
        num_classes,
        target_class_ids,
        name=None,
        dtype=None,
        ignore_class=None,
        sparse_y_true=True,
        sparse_y_pred=True,
        axis=-1,
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.IoU(
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            name=name,
            dtype=dtype,
            ignore_class=ignore_class,
            sparse_y_true=sparse_y_true,
            sparse_y_pred=sparse_y_pred,
            axis=axis,
            **kwargs,
        )


class BinaryIoU(Metric):
    def __init__(
        self, target_class_ids=(0, 1), threshold=0.5, name=None, dtype=None, **kwargs
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.BinaryIoU(
            target_class_ids=target_class_ids,
            threshold=threshold,
            name=name,
            dtype=dtype,
            **kwargs,
        )


class OneHotMeanIoU(Metric):
    def __init__(
        self,
        num_classes,
        name=None,
        dtype=None,
        ignore_class=None,
        sparse_y_pred=False,
        axis=-1,
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.OneHotMeanIoU(
            num_classes=num_classes,
            name=name,
            dtype=dtype,
            ignore_class=ignore_class,
            sparse_y_pred=sparse_y_pred,
            axis=axis,
            **kwargs,
        )


class OneHotIoU(Metric):
    def __init__(
        self,
        num_classes,
        target_class_ids,
        name=None,
        dtype=None,
        ignore_class=None,
        sparse_y_pred=False,
        axis=-1,
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.OneHotIoU(
            num_classes=num_classes,
            target_class_ids=target_class_ids,
            name=name,
            dtype=dtype,
            ignore_class=ignore_class,
            sparse_y_pred=sparse_y_pred,
            axis=axis,
            **kwargs,
        )


class ConcordanceCorrelation(Metric):
    def __init__(self, name="concordance_correlation", dtype=None, axis=-1, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.ConcordanceCorrelation(
            name=name, dtype=dtype, axis=axis, **kwargs
        )


class PearsonCorrelation(Metric):
    def __init__(self, name="pearson_correlation", dtype=None, axis=-1, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.PearsonCorrelation(
            name=name, dtype=dtype, axis=axis, **kwargs
        )


class F1Score(Metric):
    def __init__(
        self, average=None, threshold=None, name="f1_score", dtype=None, **kwargs
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.F1Score(
            average=average, threshold=threshold, name=name, dtype=dtype, **kwargs
        )


class FBetaScore(Metric):
    def __init__(
        self,
        average=None,
        beta=1,
        threshold=None,
        name="fbeta_score",
        dtype=None,
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.FBetaScore(
            average=average,
            beta=beta,
            threshold=threshold,
            name=name,
            dtype=dtype,
            **kwargs,
        )


class R2Score(Metric):
    def __init__(
        self,
        class_aggregation="uniform_average",
        num_regressors=0,
        name="r2_score",
        dtype=None,
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._impl = metrics_impl.R2Score(
            class_aggregation=class_aggregation,
            num_regressors=num_regressors,
            name=name,
            dtype=dtype,
            **kwargs,
        )
