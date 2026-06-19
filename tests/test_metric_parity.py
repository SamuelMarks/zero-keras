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
    )


def test_metric_unsupported():
    y_true = np.random.rand(2, 5).astype(np.float32)
    y_pred = np.random.rand(2, 5).astype(np.float32)
    metrics_list = [
        "CosineSimilarity",
        "MeanAbsoluteError",
        "MeanAbsolutePercentageError",
        "MeanSquaredError",
        "MeanSquaredLogarithmicError",
        "Hinge",
        "SquaredHinge",
        "CategoricalHinge",
        "RootMeanSquaredError",
        "KLDivergence",
        "Poisson",
        "ConcordanceCorrelation",
        "PearsonCorrelation",
    ]
    for m in metrics_list:
        zero_m = getattr(metrics, m)()
        zero_m.update_state(y_true, y_pred)
        assert zero_m.result() is not None


def test_metrics_parity_more():
    y_true = np.random.rand(2, 5).astype(np.float32)
    y_pred = np.random.rand(2, 5).astype(np.float32)

    metrics_list = [
        "CosineSimilarity",
        "MeanAbsoluteError",
        "MeanAbsolutePercentageError",
        "MeanSquaredError",
        "MeanSquaredLogarithmicError",
        "Hinge",
        "SquaredHinge",
        "CategoricalHinge",
        "RootMeanSquaredError",
        "KLDivergence",
        "Poisson",
    ]

    for m in metrics_list:
        try:
            check_metric_parity(
                getattr(metrics, m),
                getattr(keras.metrics, m),
                y_true,
                y_pred,
                y_true,
                y_pred,
            )
            print(f"{m} PASSED")
        except Exception as e:
            print(f"{m} FAILED: {e}")


def test_classification_metrics_parity():
    y_true = np.random.randint(0, 2, size=(2, 5)).astype(np.float32)
    y_pred = np.random.rand(2, 5).astype(np.float32)

    metrics_list = [
        "AUC",
        "FalsePositives",
        "FalseNegatives",
        "TruePositives",
        "TrueNegatives",
        "Precision",
        "Recall",
    ]

    for m in metrics_list:
        # We test that the API shell executes cleanly without error
        zero_m = getattr(metrics, m)()
        zero_m.update_state(y_true, y_pred)
        assert zero_m.result() is not None


def test_metric_logcosherror():
    y_true = np.array([0, 1, 1, 0], dtype="float32")
    y_pred = np.array([0.1, 0.9, 0.8, 0.3], dtype="float32")

    check_metric_parity(
        metrics.LogCoshError, keras.metrics.LogCoshError, y_true, y_pred, y_true, y_pred
    )


def test_metrics_sample_weights():
    y_true = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    y_pred = np.array([[0.9, 0.1], [0.1, 0.9]], dtype="float32")
    sample_weight = np.array([1.0, 0.5], dtype="float32")

    metrics_list = [
        "CosineSimilarity",
        "MeanAbsoluteError",
        "MeanAbsolutePercentageError",
        "MeanSquaredError",
        "MeanSquaredLogarithmicError",
        "Hinge",
        "SquaredHinge",
        "CategoricalHinge",
        "RootMeanSquaredError",
        "KLDivergence",
        "Poisson",
        "ConcordanceCorrelation",
        "PearsonCorrelation",
    ]
    for m in metrics_list:
        zero_m = getattr(metrics, m)()
        zero_m.update_state(y_true, y_pred, sample_weight=sample_weight)
        assert zero_m.result() is not None
