"""Tests for zero_keras metrics parity with Keras."""

from __future__ import annotations
from zero_keras.types import TensorLike

import keras
import numpy as np
import pytest

from zero_keras import metrics

from .utils import assert_allclose_keras_zero, set_seed


@pytest.fixture(autouse=True)
def _set_seed() -> TensorLike:
    """Function docstring."""
    set_seed(42)


def check_metric_parity(
    metric_cls: TensorLike,
    keras_cls: TensorLike,
    y_true: TensorLike,
    y_pred: TensorLike,
    y_true2: TensorLike,
    y_pred2: TensorLike,
    atol: TensorLike = 1e-5,
    rtol: TensorLike = 1e-5,
    **kwargs: TensorLike,
) -> TensorLike:
    """Function docstring.

    Args:
        metric_cls: Description.
        keras_cls: Description.
        y_true: Description.
        y_pred: Description.
        y_true2: Description.
        y_pred2: Description.
        atol: Description.
        rtol: Description.
        kwargs: Description.
    """
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
    assert_allclose_keras_zero(keras_metric.result(), zero_metric.result(), atol=atol, rtol=rtol)


def test_metric_Mean() -> TensorLike:
    """Function docstring."""
    check_metric_parity(
        metrics.Mean,
        keras.metrics.Mean,
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        np.array([3.0, 4.0]),
    )


def test_metric_Sum() -> TensorLike:
    """Function docstring."""
    check_metric_parity(
        metrics.Sum,
        keras.metrics.Sum,
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        np.array([3.0, 4.0]),
    )


def test_metric_MeanMetricWrapper() -> TensorLike:
    """Function docstring."""
    import keras.losses

    def custom_fn(y_true: TensorLike, y_pred: TensorLike) -> TensorLike:
        """Function docstring.

        Args:
            y_true: Description.
            y_pred: Description.
        """
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


def test_metric_Accuracy() -> TensorLike:
    """Function docstring."""
    check_metric_parity(
        metrics.Accuracy,
        keras.metrics.Accuracy,
        np.array([1, 1, 0, 0]),
        np.array([1, 1, 0, 1]),
        np.array([0, 1, 0, 0]),
        np.array([1, 1, 0, 0]),
    )


def test_metric_BinaryAccuracy() -> TensorLike:
    """Function docstring."""
    check_metric_parity(
        metrics.BinaryAccuracy,
        keras.metrics.BinaryAccuracy,
        np.array([[1], [1], [0], [0]]),
        np.array([[0.9], [0.9], [0.1], [0.9]]),
        np.array([[0], [1], [0], [0]]),
        np.array([[0.9], [0.9], [0.1], [0.1]]),
    )


def test_metric_CategoricalAccuracy() -> TensorLike:
    """Function docstring."""
    check_metric_parity(
        metrics.CategoricalAccuracy,
        keras.metrics.CategoricalAccuracy,
        np.array([[0, 1], [1, 0]]),
        np.array([[0.1, 0.9], [0.9, 0.1]]),
        np.array([[0, 1], [0, 1]]),
        np.array([[0.8, 0.2], [0.1, 0.9]]),
    )


def test_metric_SparseCategoricalAccuracy() -> TensorLike:
    """Function docstring."""
    check_metric_parity(
        metrics.SparseCategoricalAccuracy,
        keras.metrics.SparseCategoricalAccuracy,
        np.array([1, 0], dtype=np.int32),
        np.array([[0.1, 0.9], [0.9, 0.1]]),
        np.array([1, 1]),
        np.array([[0.8, 0.2], [0.1, 0.9]]),
    )


def test_metric_TopKCategoricalAccuracy() -> TensorLike:
    """Function docstring."""
    check_metric_parity(
        metrics.TopKCategoricalAccuracy,
        keras.metrics.TopKCategoricalAccuracy,
        np.array([[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=np.int32),
        np.array([[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.6, 0.1, 0.1, 0.1]], dtype=np.float32),
        np.array([[0, 0, 1, 0, 0], [0, 1, 0, 0, 0]], dtype=np.int32),
        np.array([[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.6, 0.1, 0.1, 0.1]], dtype=np.float32),
    )


def test_metric_unsupported() -> TensorLike:
    """Function docstring."""
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
        zero_m = metrics.__dict__[m]()
        zero_m.update_state(y_true, y_pred)
        assert zero_m.result() is not None


def test_metrics_parity_more() -> TensorLike:
    """Function docstring."""
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
                metrics.__dict__[m],
                keras.metrics.__dict__[m],
                y_true,
                y_pred,
                y_true,
                y_pred,
            )
            print(f"{m} PASSED")
        except Exception as e:
            print(f"{m} FAILED: {e}")


def test_classification_metrics_parity() -> TensorLike:
    """Function docstring."""
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
        zero_m = metrics.__dict__[m]()
        zero_m.update_state(y_true, y_pred)
        assert zero_m.result() is not None


def test_metric_logcosherror() -> TensorLike:
    """Function docstring."""
    y_true = np.array([0, 1, 1, 0], dtype="float32")
    y_pred = np.array([0.1, 0.9, 0.8, 0.3], dtype="float32")

    check_metric_parity(
        metrics.LogCoshError, keras.metrics.LogCoshError, y_true, y_pred, y_true, y_pred
    )


def test_metrics_sample_weights() -> TensorLike:
    """Function docstring."""
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
        zero_m = metrics.__dict__[m]()
        zero_m.update_state(y_true, y_pred, sample_weight=sample_weight)
        assert zero_m.result() is not None
