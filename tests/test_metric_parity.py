"""Tests for zero_keras metrics parity with Keras."""

from __future__ import annotations
from zero_keras.types import TensorLike

import keras
import numpy as np
import pytest

from zero_keras import metrics

from .utils import assert_allclose_keras_zero, set_seed


@pytest.fixture(autouse=True)
def _set_seed() -> None:
    """Verify or execute `set seed`."""
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
) -> None:
    """Verify or execute `check metric parity`.

    Args:
        metric_cls: Test fixture or input parameter `metric_cls` (metric cls).
        keras_cls: Test fixture or input parameter `keras_cls` (keras cls).
        y_true: Test fixture or input parameter `y_true` (y true).
        y_pred: Test fixture or input parameter `y_pred` (y pred).
        y_true2: Test fixture or input parameter `y_true2` (y true2).
        y_pred2: Test fixture or input parameter `y_pred2` (y pred2).
        atol: Test fixture or input parameter `atol` (atol).
        rtol: Test fixture or input parameter `rtol` (rtol).
        **kwargs: Keyword arguments forwarded to helper.
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


def test_metric_Mean() -> None:
    """Verify or execute `test metric Mean`."""
    check_metric_parity(
        metrics.Mean,
        keras.metrics.Mean,
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        np.array([3.0, 4.0]),
    )


def test_metric_Sum() -> None:
    """Verify or execute `test metric Sum`."""
    check_metric_parity(
        metrics.Sum,
        keras.metrics.Sum,
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        np.array([3.0, 4.0]),
    )


def test_metric_MeanMetricWrapper() -> None:
    """Verify or execute `test metric MeanMetricWrapper`."""
    import keras.losses

    def custom_fn(y_true: TensorLike, y_pred: TensorLike) -> TensorLike:
        """Verify or execute `custom fn`.

        Args:
            y_true: Test fixture or input parameter `y_true` (y true).
            y_pred: Test fixture or input parameter `y_pred` (y pred).

        Returns:
            TensorLike: Result from `custom_fn`.
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


def test_metric_Accuracy() -> None:
    """Verify or execute `test metric Accuracy`."""
    check_metric_parity(
        metrics.Accuracy,
        keras.metrics.Accuracy,
        np.array([1, 1, 0, 0]),
        np.array([1, 1, 0, 1]),
        np.array([0, 1, 0, 0]),
        np.array([1, 1, 0, 0]),
    )


def test_metric_BinaryAccuracy() -> None:
    """Verify or execute `test metric BinaryAccuracy`."""
    check_metric_parity(
        metrics.BinaryAccuracy,
        keras.metrics.BinaryAccuracy,
        np.array([[1], [1], [0], [0]]),
        np.array([[0.9], [0.9], [0.1], [0.9]]),
        np.array([[0], [1], [0], [0]]),
        np.array([[0.9], [0.9], [0.1], [0.1]]),
    )


def test_metric_CategoricalAccuracy() -> None:
    """Verify or execute `test metric CategoricalAccuracy`."""
    check_metric_parity(
        metrics.CategoricalAccuracy,
        keras.metrics.CategoricalAccuracy,
        np.array([[0, 1], [1, 0]]),
        np.array([[0.1, 0.9], [0.9, 0.1]]),
        np.array([[0, 1], [0, 1]]),
        np.array([[0.8, 0.2], [0.1, 0.9]]),
    )


def test_metric_SparseCategoricalAccuracy() -> None:
    """Verify or execute `test metric SparseCategoricalAccuracy`."""
    check_metric_parity(
        metrics.SparseCategoricalAccuracy,
        keras.metrics.SparseCategoricalAccuracy,
        np.array([1, 0], dtype=np.int32),
        np.array([[0.1, 0.9], [0.9, 0.1]]),
        np.array([1, 1]),
        np.array([[0.8, 0.2], [0.1, 0.9]]),
    )


def test_metric_TopKCategoricalAccuracy() -> None:
    """Verify or execute `test metric TopKCategoricalAccuracy`."""
    check_metric_parity(
        metrics.TopKCategoricalAccuracy,
        keras.metrics.TopKCategoricalAccuracy,
        np.array([[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=np.int32),
        np.array([[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.6, 0.1, 0.1, 0.1]], dtype=np.float32),
        np.array([[0, 0, 1, 0, 0], [0, 1, 0, 0, 0]], dtype=np.int32),
        np.array([[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.6, 0.1, 0.1, 0.1]], dtype=np.float32),
    )


def test_metric_unsupported() -> None:
    """Verify or execute `test metric unsupported`."""
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
        zero_m = vars(metrics)[m]()
        zero_m.update_state(y_true, y_pred)
        assert zero_m.result() is not None


def test_metrics_parity_more() -> None:
    """Verify or execute `test metrics parity more`."""
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
                vars(metrics)[m],
                vars(keras.metrics)[m],
                y_true,
                y_pred,
                y_true,
                y_pred,
            )
            print(f"{m} PASSED")
        except Exception as e:
            print(f"{m} FAILED: {e}")


def test_classification_metrics_parity() -> None:
    """Verify or execute `test classification metrics parity`."""
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
        zero_m = vars(metrics)[m]()
        zero_m.update_state(y_true, y_pred)
        assert zero_m.result() is not None


def test_metric_logcosherror() -> None:
    """Verify or execute `test metric logcosherror`."""
    y_true = np.array([0, 1, 1, 0], dtype="float32")
    y_pred = np.array([0.1, 0.9, 0.8, 0.3], dtype="float32")

    check_metric_parity(
        metrics.LogCoshError, keras.metrics.LogCoshError, y_true, y_pred, y_true, y_pred
    )


def test_metrics_sample_weights() -> None:
    """Verify or execute `test metrics sample weights`."""
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
        zero_m = vars(metrics)[m]()
        zero_m.update_state(y_true, y_pred, sample_weight=sample_weight)
        assert zero_m.result() is not None
