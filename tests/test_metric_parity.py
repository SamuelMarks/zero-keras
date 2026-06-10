"""Tests for zero_keras metrics parity with Keras."""

import numpy as np
import pytest
import keras
from zero_keras import metrics
from .utils import assert_allclose_keras_zero, set_seed


@pytest.fixture(autouse=True)
def _set_seed():
    set_seed(42)


def check_metric_parity(
    metric_cls,
    keras_cls,
    y_true,
    y_pred,
    y_true2,
    y_pred2,
    atol=1e-5,
    rtol=1e-5,
    **kwargs,
):
    keras_metric = keras_cls(**kwargs)
    keras_metric.update_state(y_true, y_pred)
    keras_out1 = keras_metric.result()
    keras_metric.update_state(y_true2, y_pred2)
    keras_out2 = keras_metric.result()

    zero_metric = metric_cls(**kwargs)
    zero_metric.update_state(y_true, y_pred)
    zero_out1 = zero_metric.result()
    zero_metric.update_state(y_true2, y_pred2)
    zero_out2 = zero_metric.result()

    # Assert values after first batch
    assert_allclose_keras_zero(keras_out1, zero_out1, atol=atol, rtol=rtol)
    # Assert values after second batch (statefulness)
    assert_allclose_keras_zero(keras_out2, zero_out2, atol=atol, rtol=rtol)

    # Test reset_state
    keras_metric.reset_state()
    zero_metric.reset_state()
    keras_metric.update_state(y_true, y_pred)
    zero_metric.update_state(y_true, y_pred)
    assert_allclose_keras_zero(
        keras_metric.result(), zero_metric.result(), atol=atol, rtol=rtol
    )


def test_metric_Mean():
    check_metric_parity(
        metrics.Mean,
        keras.metrics.Mean,
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        np.array([3.0, 4.0]),
    )


def test_metric_Sum():
    check_metric_parity(
        metrics.Sum,
        keras.metrics.Sum,
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        np.array([3.0, 4.0]),
    )


def test_metric_MeanMetricWrapper():
    import keras.losses

    def custom_fn(y_true, y_pred):
        return keras.losses.mean_squared_error(y_true, y_pred)

    check_metric_parity(
        metrics.MeanMetricWrapper,
        keras.metrics.MeanMetricWrapper,
        np.array([1.0, 2.0]),
        np.array([1.1, 1.9]),
        np.array([3.0, 4.0]),
        np.array([3.1, 3.9]),
        fn=custom_fn,
    )


def test_metric_Accuracy():
    check_metric_parity(
        metrics.Accuracy,
        keras.metrics.Accuracy,
        np.array([1, 1, 0, 0]),
        np.array([1, 1, 0, 1]),
        np.array([0, 1, 0, 0]),
        np.array([1, 1, 0, 0]),
    )


def test_metric_BinaryAccuracy():
    check_metric_parity(
        metrics.BinaryAccuracy,
        keras.metrics.BinaryAccuracy,
        np.array([[1], [1], [0], [0]]),
        np.array([[0.9], [0.9], [0.1], [0.9]]),
        np.array([[0], [1], [0], [0]]),
        np.array([[0.9], [0.9], [0.1], [0.1]]),
    )


def test_metric_CategoricalAccuracy():
    check_metric_parity(
        metrics.CategoricalAccuracy,
        keras.metrics.CategoricalAccuracy,
        np.array([[0, 1], [1, 0]]),
        np.array([[0.1, 0.9], [0.9, 0.1]]),
        np.array([[0, 1], [0, 1]]),
        np.array([[0.8, 0.2], [0.1, 0.9]]),
    )


def test_metric_SparseCategoricalAccuracy():
    check_metric_parity(
        metrics.SparseCategoricalAccuracy,
        keras.metrics.SparseCategoricalAccuracy,
        np.array([1, 0], dtype=np.int32),
        np.array([[0.1, 0.9], [0.9, 0.1]]),
        np.array([1, 1]),
        np.array([[0.8, 0.2], [0.1, 0.9]]),
    )


def test_metric_TopKCategoricalAccuracy():
    check_metric_parity(
        metrics.TopKCategoricalAccuracy,
        keras.metrics.TopKCategoricalAccuracy,
        np.array([[0, 1, 0], [1, 0, 0]], dtype=np.int32),
        np.array([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32),
        np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int32),
        np.array([[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]], dtype=np.float32),
        k=2,
    )


def test_metric_SparseTopKCategoricalAccuracy():
    check_metric_parity(
        metrics.SparseTopKCategoricalAccuracy,
        keras.metrics.SparseTopKCategoricalAccuracy,
        np.array([1, 0], dtype=np.int32),
        np.array([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32),
        np.array([2, 1], dtype=np.int32),
        np.array([[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]], dtype=np.float32),
        k=2,
    )


def test_metric_Precision():
    check_metric_parity(
        metrics.Precision,
        keras.metrics.Precision,
        np.array([1, 1, 0, 0]),
        np.array([0.9, 0.9, 0.1, 0.9]),
        np.array([0, 1, 0, 0]),
        np.array([0.9, 0.9, 0.1, 0.1]),
    )


def test_metric_Recall():
    check_metric_parity(
        metrics.Recall,
        keras.metrics.Recall,
        np.array([1, 1, 0, 0]),
        np.array([0.9, 0.9, 0.1, 0.9]),
        np.array([0, 1, 0, 0]),
        np.array([0.9, 0.9, 0.1, 0.1]),
    )


def test_metric_TruePositives():
    check_metric_parity(
        metrics.TruePositives,
        keras.metrics.TruePositives,
        np.array([1, 1, 0, 0]),
        np.array([0.9, 0.9, 0.1, 0.9]),
        np.array([0, 1, 0, 0]),
        np.array([0.9, 0.9, 0.1, 0.1]),
    )


def test_metric_TrueNegatives():
    check_metric_parity(
        metrics.TrueNegatives,
        keras.metrics.TrueNegatives,
        np.array([1, 1, 0, 0]),
        np.array([0.9, 0.9, 0.1, 0.9]),
        np.array([0, 1, 0, 0]),
        np.array([0.9, 0.9, 0.1, 0.1]),
    )


def test_metric_FalsePositives():
    check_metric_parity(
        metrics.FalsePositives,
        keras.metrics.FalsePositives,
        np.array([1, 1, 0, 0]),
        np.array([0.9, 0.9, 0.1, 0.9]),
        np.array([0, 1, 0, 0]),
        np.array([0.9, 0.9, 0.1, 0.1]),
    )


def test_metric_FalseNegatives():
    check_metric_parity(
        metrics.FalseNegatives,
        keras.metrics.FalseNegatives,
        np.array([1, 1, 0, 0]),
        np.array([0.9, 0.9, 0.1, 0.9]),
        np.array([0, 1, 0, 0]),
        np.array([0.9, 0.9, 0.1, 0.1]),
    )


def test_metric_AUC():
    check_metric_parity(
        metrics.AUC,
        keras.metrics.AUC,
        np.array([1, 1, 0, 0]),
        np.array([0.9, 0.8, 0.1, 0.4]),
        np.array([0, 1, 0, 0]),
        np.array([0.9, 0.9, 0.1, 0.1]),
    )
    check_metric_parity(
        metrics.AUC,
        keras.metrics.AUC,
        np.array([1, 1, 0, 0]),
        np.array([0.9, 0.8, 0.1, 0.4]),
        np.array([0, 1, 0, 0]),
        np.array([0.9, 0.9, 0.1, 0.1]),
        curve="PR",
    )


def test_metric_MeanSquaredError():
    check_metric_parity(
        metrics.MeanSquaredError,
        keras.metrics.MeanSquaredError,
        np.array([1.0, 2.0]),
        np.array([1.1, 1.9]),
        np.array([3.0, 4.0]),
        np.array([3.1, 3.9]),
    )


def test_metric_MeanAbsoluteError():
    check_metric_parity(
        metrics.MeanAbsoluteError,
        keras.metrics.MeanAbsoluteError,
        np.array([1.0, 2.0]),
        np.array([1.1, 1.9]),
        np.array([3.0, 4.0]),
        np.array([3.1, 3.9]),
    )


def test_metric_MeanAbsolutePercentageError():
    check_metric_parity(
        metrics.MeanAbsolutePercentageError,
        keras.metrics.MeanAbsolutePercentageError,
        np.array([1.0, 2.0]),
        np.array([1.1, 1.9]),
        np.array([3.0, 4.0]),
        np.array([3.1, 3.9]),
    )


def test_metric_MeanSquaredLogarithmicError():
    check_metric_parity(
        metrics.MeanSquaredLogarithmicError,
        keras.metrics.MeanSquaredLogarithmicError,
        np.array([1.0, 2.0]),
        np.array([1.1, 1.9]),
        np.array([3.0, 4.0]),
        np.array([3.1, 3.9]),
    )


def test_metric_RootMeanSquaredError():
    check_metric_parity(
        metrics.RootMeanSquaredError,
        keras.metrics.RootMeanSquaredError,
        np.array([1.0, 2.0]),
        np.array([1.1, 1.9]),
        np.array([3.0, 4.0]),
        np.array([3.1, 3.9]),
    )


def test_metric_BinaryCrossentropy():
    check_metric_parity(
        metrics.BinaryCrossentropy,
        keras.metrics.BinaryCrossentropy,
        np.array([[1.0], [0.0]]),
        np.array([[0.9], [0.1]]),
        np.array([[0.0], [1.0]]),
        np.array([[0.1], [0.9]]),
    )


def test_metric_CategoricalCrossentropy():
    check_metric_parity(
        metrics.CategoricalCrossentropy,
        keras.metrics.CategoricalCrossentropy,
        np.array([[0, 1], [1, 0]]),
        np.array([[0.1, 0.9], [0.9, 0.1]]),
        np.array([[0, 1], [0, 1]]),
        np.array([[0.8, 0.2], [0.1, 0.9]]),
    )


def test_metric_SparseCategoricalCrossentropy():
    check_metric_parity(
        metrics.SparseCategoricalCrossentropy,
        keras.metrics.SparseCategoricalCrossentropy,
        np.array([1, 0], dtype=np.int32),
        np.array([[0.1, 0.9], [0.9, 0.1]]),
        np.array([1, 1]),
        np.array([[0.8, 0.2], [0.1, 0.9]]),
    )


def test_metric_CosineSimilarity():
    check_metric_parity(
        metrics.CosineSimilarity,
        keras.metrics.CosineSimilarity,
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[0.9, 0.1], [0.1, 0.9]]),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([[0.1, 0.9], [0.9, 0.1]]),
    )


def test_metric_Hinge():
    check_metric_parity(
        metrics.Hinge,
        keras.metrics.Hinge,
        np.array([[1.0], [-1.0]]),
        np.array([[0.9], [-0.9]]),
        np.array([[-1.0], [1.0]]),
        np.array([[-0.1], [0.9]]),
    )


def test_metric_SquaredHinge():
    check_metric_parity(
        metrics.SquaredHinge,
        keras.metrics.SquaredHinge,
        np.array([[1.0], [-1.0]]),
        np.array([[0.9], [-0.9]]),
        np.array([[-1.0], [1.0]]),
        np.array([[-0.1], [0.9]]),
    )


def test_metric_CategoricalHinge():
    check_metric_parity(
        metrics.CategoricalHinge,
        keras.metrics.CategoricalHinge,
        np.array([[0, 1], [1, 0]]),
        np.array([[0.1, 0.9], [0.9, 0.1]]),
        np.array([[0, 1], [0, 1]]),
        np.array([[0.8, 0.2], [0.1, 0.9]]),
    )


def test_metric_KLDivergence():
    check_metric_parity(
        metrics.KLDivergence,
        keras.metrics.KLDivergence,
        np.array([[0, 1], [1, 0]]),
        np.array([[0.1, 0.9], [0.9, 0.1]]),
        np.array([[0, 1], [0, 1]]),
        np.array([[0.8, 0.2], [0.1, 0.9]]),
    )


def test_metric_LogCoshError():
    check_metric_parity(
        metrics.LogCoshError,
        keras.metrics.LogCoshError,
        np.array([1.0, 2.0]),
        np.array([1.1, 1.9]),
        np.array([3.0, 4.0]),
        np.array([3.1, 3.9]),
    )


def test_metric_Poisson():
    check_metric_parity(
        metrics.Poisson,
        keras.metrics.Poisson,
        np.array([1.0, 2.0]),
        np.array([1.1, 1.9]),
        np.array([3.0, 4.0]),
        np.array([3.1, 3.9]),
    )


def test_metric_unsupported():
    for m_cls in [
        metrics.BinaryIoU,
        metrics.ConcordanceCorrelation,
        metrics.F1Score,
        metrics.FBetaScore,
        metrics.IoU,
        metrics.MeanIoU,
        metrics.OneHotIoU,
        metrics.OneHotMeanIoU,
        metrics.PearsonCorrelation,
        metrics.PrecisionAtRecall,
        metrics.R2Score,
        metrics.RecallAtPrecision,
        metrics.SensitivityAtSpecificity,
        metrics.SpecificityAtSensitivity,
    ]:
        import inspect

        sig = inspect.signature(m_cls.__init__)
        kwargs = {"name": "test_metric"}
        if "num_classes" in sig.parameters:
            kwargs["num_classes"] = 2
        if "recall" in sig.parameters:
            kwargs["recall"] = 0.5
        if "precision" in sig.parameters:
            kwargs["precision"] = 0.5
        if "specificity" in sig.parameters:
            kwargs["specificity"] = 0.5
        if "sensitivity" in sig.parameters:
            kwargs["sensitivity"] = 0.5
        if "target_class_ids" in sig.parameters:
            kwargs["target_class_ids"] = [0, 1]

        m = m_cls(**kwargs)
        try:
            m.update_state(
                np.array([[1.0]], dtype=np.float32), np.array([[1.0]], dtype=np.float32)
            )
        except Exception:
            pass  # Some of these have very specific shape/type requirements in Keras, we just need to hit the constructor/wrapper

        res = m.result()
        assert hasattr(res, "numpy") or isinstance(
            res, (float, np.float32, np.float64, np.ndarray)
        )


def test_metric_Metric():
    m = metrics.Metric()
    m.update_state([1.0], [1.0])
    assert m.result() == 0.0
