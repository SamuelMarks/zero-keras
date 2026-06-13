import ml_switcheroo_compiler.ops as ops

"""Keras metrics."""

from typing import Any, Optional


class Metric:
    def __init__(self, name: str = "", dtype: Optional[str] = None, **kwargs: Any):
        self.name = name
        self.dtype = dtype
        self._kwargs = kwargs
        pass

    def update_state(self, *args, **kwargs):
        pass

    def result(self):
        pass

    def reset_state(self):
        pass

    def __call__(self, *args, **kwargs):
        self.update_state(*args, **kwargs)
        return self.result()


from zero_keras.activations import _to_tensor, _wrap


class Mean(Metric):
    def __init__(self, name="mean", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = 0.0
        self.count = 0.0

    def update_state(self, values, sample_weight=None):
        values = _to_tensor(values)
        if sample_weight is not None:
            sample_weight = _to_tensor(sample_weight)
            values = ops.multiply(values, sample_weight)
            self.count += ops.cast(ops.sum(sample_weight), dtype="float32")
        else:
            self.count += ops.cast(ops.sum(ops.ones_like(values)), dtype="float32")
        self.total += ops.cast(ops.sum(values), dtype="float32")

    def result(self):
        return _wrap(self.total / ops.maximum(_to_tensor(self.count), 1e-7))

    def reset_state(self):
        self.total = 0.0
        self.count = 0.0


class Sum(Metric):
    def __init__(self, name="sum", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.total = 0.0

    def update_state(self, values, sample_weight=None):
        values = _to_tensor(values)
        if sample_weight is not None:
            sample_weight = _to_tensor(sample_weight)
            values = ops.multiply(values, sample_weight)
        self.total += ops.cast(ops.sum(values), dtype="float32")

    def result(self):
        return _wrap(self.total)

    def reset_state(self):
        self.total = 0.0


class MeanMetricWrapper(Mean):
    def __init__(self, fn, name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.fn = fn

    def update_state(self, y_true, y_pred, sample_weight=None):
        matches = self.fn(y_true, y_pred)
        super().update_state(matches, sample_weight=sample_weight)


class Accuracy(MeanMetricWrapper):
    def __init__(self, name="accuracy", dtype=None, **kwargs):
        def accuracy_fn(y_true, y_pred):
            return ops.cast(
                ops.equal(_to_tensor(y_true), _to_tensor(y_pred)), dtype="float32"
            )

        super().__init__(fn=accuracy_fn, name=name, dtype=dtype, **kwargs)


class BinaryAccuracy(MeanMetricWrapper):
    def __init__(self, name="binary_accuracy", dtype=None, threshold=0.5, **kwargs):
        def binary_accuracy_fn(y_true, y_pred):
            y_pred = ops.cast(_to_tensor(y_pred) > threshold, dtype="float32")
            y_true = ops.cast(_to_tensor(y_true), dtype="float32")
            return ops.cast(ops.equal(y_true, y_pred), dtype="float32")

        super().__init__(fn=binary_accuracy_fn, name=name, dtype=dtype, **kwargs)


class CategoricalAccuracy(MeanMetricWrapper):
    def __init__(self, name="categorical_accuracy", dtype=None, **kwargs):
        def categorical_accuracy_fn(y_true, y_pred):
            y_true = _to_tensor(y_true)
            y_pred = _to_tensor(y_pred)
            return ops.cast(
                ops.equal(ops.argmax(y_true, axis=-1), ops.argmax(y_pred, axis=-1)),
                dtype="float32",
            )

        super().__init__(fn=categorical_accuracy_fn, name=name, dtype=dtype, **kwargs)


class SparseCategoricalAccuracy(MeanMetricWrapper):
    def __init__(self, name="sparse_categorical_accuracy", dtype=None, **kwargs):
        def sparse_categorical_accuracy_fn(y_true, y_pred):
            y_true = _to_tensor(y_true)
            y_pred = _to_tensor(y_pred)
            if len(y_true.shape) == len(y_pred.shape):
                y_true = ops.squeeze(y_true, -1)
            y_pred_classes = ops.argmax(y_pred, axis=-1)
            return ops.cast(
                ops.equal(ops.cast(y_true, dtype=y_pred_classes.dtype), y_pred_classes),
                dtype="float32",
            )

        super().__init__(
            fn=sparse_categorical_accuracy_fn, name=name, dtype=dtype, **kwargs
        )


class TopKCategoricalAccuracy(MeanMetricWrapper):
    def __init__(self, k=5, name="top_k_categorical_accuracy", dtype=None, **kwargs):
        def top_k_fn(y_true, y_pred):
            y_true = _to_tensor(y_true)
            y_pred = _to_tensor(y_pred)
            y_true_rank = ops.argmax(y_true, axis=-1)
            # Find the value of the k-th top element
            # Actually, `numpy.argsort` is what we need. We can just use python/numpy for eager mode if we need to.
            # But ops.top_k is not implemented.

            if hasattr(y_pred, "data"):
                # Eager mode
                np = __import__("numpy")
                np_pred = np.asarray(y_pred.data)
                np = __import__("numpy")
                top_indices = np.argsort(np_pred, axis=-1)[..., -k:]
                y_true_np = np.asarray(y_true_rank.data)[..., np.newaxis]
                matches = np.any(top_indices == y_true_np, axis=-1).astype(np.float32)

                return ops.asarray(matches)

        super().__init__(fn=top_k_fn, name=name, dtype=dtype, **kwargs)


class SparseTopKCategoricalAccuracy(MeanMetricWrapper):
    def __init__(
        self, k=5, name="sparse_top_k_categorical_accuracy", dtype=None, **kwargs
    ):
        def sparse_top_k_fn(y_true, y_pred):
            y_true = _to_tensor(y_true)
            y_pred = _to_tensor(y_pred)

            if hasattr(y_pred, "data"):
                # Eager mode
                np = __import__("numpy")
                np_pred = np.asarray(y_pred.data)
                np = __import__("numpy")
                top_indices = np.argsort(np_pred, axis=-1)[..., -k:]
                y_true_np = np.asarray(y_true.data)[..., np.newaxis]
                matches = np.any(top_indices == y_true_np, axis=-1).astype(np.float32)
                from zero_keras.core_layers import KerasTensor

                return KerasTensor(matches.shape, "float32", data=matches)

        super().__init__(fn=sparse_top_k_fn, name=name, dtype=dtype, **kwargs)


class FalsePositives(Metric):
    def __init__(self, thresholds=None, name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass


class FalseNegatives(Metric):
    def __init__(self, thresholds=None, name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass


class TrueNegatives(Metric):
    def __init__(self, thresholds=None, name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass


class TruePositives(Metric):
    def __init__(self, thresholds=None, name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass


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
        pass


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
        pass


class PrecisionAtRecall(Metric):
    def __init__(
        self, recall, num_thresholds=200, class_id=None, name=None, dtype=None, **kwargs
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass


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
        pass


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
        pass


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
        pass


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
        pass


class CosineSimilarity(MeanMetricWrapper):
    def __init__(self, name="cosine_similarity", dtype=None, axis=-1, **kwargs):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        pass


class MeanAbsoluteError(MeanMetricWrapper):
    def __init__(self, name="mean_absolute_error", dtype=None, **kwargs):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        pass


class MeanAbsolutePercentageError(MeanMetricWrapper):
    def __init__(self, name="mean_absolute_percentage_error", dtype=None, **kwargs):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        pass


class MeanSquaredError(MeanMetricWrapper):
    def __init__(self, name="mean_squared_error", dtype=None, **kwargs):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        pass


class MeanSquaredLogarithmicError(MeanMetricWrapper):
    def __init__(self, name="mean_squared_logarithmic_error", dtype=None, **kwargs):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        pass


class Hinge(MeanMetricWrapper):
    def __init__(self, name="hinge", dtype=None, **kwargs):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        pass


class SquaredHinge(MeanMetricWrapper):
    def __init__(self, name="squared_hinge", dtype=None, **kwargs):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        pass


class CategoricalHinge(MeanMetricWrapper):
    def __init__(self, name="categorical_hinge", dtype=None, **kwargs):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        pass


class RootMeanSquaredError(Mean):
    def __init__(self, name="root_mean_squared_error", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass


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
        pass


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
        pass


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
        pass


class KLDivergence(MeanMetricWrapper):
    def __init__(self, name="kl_divergence", dtype=None, **kwargs):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        pass


class Poisson(MeanMetricWrapper):
    def __init__(self, name="poisson", dtype=None, **kwargs):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        pass


class LogCoshError(MeanMetricWrapper):
    def __init__(self, name="logcosh", dtype=None, **kwargs):
        super().__init__(fn=None, name=name, dtype=dtype, **kwargs)
        pass


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
        pass


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
        pass


class BinaryIoU(Metric):
    def __init__(
        self, target_class_ids=(0, 1), threshold=0.5, name=None, dtype=None, **kwargs
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass


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
        pass


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
        pass


class ConcordanceCorrelation(Metric):
    def __init__(self, name="concordance_correlation", dtype=None, axis=-1, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass


class PearsonCorrelation(Metric):
    def __init__(self, name="pearson_correlation", dtype=None, axis=-1, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass


class F1Score(Metric):
    def __init__(
        self, average=None, threshold=None, name="f1_score", dtype=None, **kwargs
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        pass


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
        pass


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
        pass
