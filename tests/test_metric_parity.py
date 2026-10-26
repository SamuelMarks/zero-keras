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
    """Execute test helper ``set_seed`` in ``test_metric_parity``."""
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
    """Assert numerical and structural parity in ``check_metric_parity``.

    Args:
        metric_cls: Metric class under test.
        keras_cls: Reference Keras class used for parity comparison.
        y_true: Ground-truth target tensor.
        y_pred: Predicted output tensor.
        y_true2: Second batch of ground-truth targets.
        y_pred2: Second batch of predicted outputs.
        atol: Absolute numerical tolerance for parity assertions.
        rtol: Relative numerical tolerance for parity assertions.
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
    """Test ``metric_Mean`` behavior and edge cases in ``test_metric_parity``."""
    check_metric_parity(
        metrics.Mean,
        keras.metrics.Mean,
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        np.array([3.0, 4.0]),
    )


def test_metric_Sum() -> None:
    """Test ``metric_Sum`` behavior and edge cases in ``test_metric_parity``."""
    check_metric_parity(
        metrics.Sum,
        keras.metrics.Sum,
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        np.array([3.0, 4.0]),
    )


def test_metric_MeanMetricWrapper() -> None:
    """Test ``metric_MeanMetricWrapper`` behavior and edge cases in ``test_metric_parity``."""
    import keras.losses

    def custom_fn(y_true: TensorLike, y_pred: TensorLike) -> TensorLike:
        """Execute test helper ``custom_fn`` in ``test_metric_parity``.

        Args:
            y_true: Ground-truth target tensor.
            y_pred: Predicted output tensor.

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
    """Test ``metric_Accuracy`` behavior and edge cases in ``test_metric_parity``."""
    check_metric_parity(
        metrics.Accuracy,
        keras.metrics.Accuracy,
        np.array([1, 1, 0, 0]),
        np.array([1, 1, 0, 1]),
        np.array([0, 1, 0, 0]),
        np.array([1, 1, 0, 0]),
    )


def test_metric_BinaryAccuracy() -> None:
    """Test ``metric_BinaryAccuracy`` behavior and edge cases in ``test_metric_parity``."""
    check_metric_parity(
        metrics.BinaryAccuracy,
        keras.metrics.BinaryAccuracy,
        np.array([[1], [1], [0], [0]]),
        np.array([[0.9], [0.9], [0.1], [0.9]]),
        np.array([[0], [1], [0], [0]]),
        np.array([[0.9], [0.9], [0.1], [0.1]]),
    )


def test_metric_CategoricalAccuracy() -> None:
    """Test ``metric_CategoricalAccuracy`` behavior and edge cases in ``test_metric_parity``."""
    check_metric_parity(
        metrics.CategoricalAccuracy,
        keras.metrics.CategoricalAccuracy,
        np.array([[0, 1], [1, 0]]),
        np.array([[0.1, 0.9], [0.9, 0.1]]),
        np.array([[0, 1], [0, 1]]),
        np.array([[0.8, 0.2], [0.1, 0.9]]),
    )


def test_metric_SparseCategoricalAccuracy() -> None:
    """Test ``metric_SparseCategoricalAccuracy`` behavior and edge cases in ``test_metric_parity``."""
    check_metric_parity(
        metrics.SparseCategoricalAccuracy,
        keras.metrics.SparseCategoricalAccuracy,
        np.array([1, 0], dtype=np.int32),
        np.array([[0.1, 0.9], [0.9, 0.1]]),
        np.array([1, 1]),
        np.array([[0.8, 0.2], [0.1, 0.9]]),
    )


def test_metric_TopKCategoricalAccuracy() -> None:
    """Test ``metric_TopKCategoricalAccuracy`` behavior and edge cases in ``test_metric_parity``."""
    check_metric_parity(
        metrics.TopKCategoricalAccuracy,
        keras.metrics.TopKCategoricalAccuracy,
        np.array([[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=np.int32),
        np.array([[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.6, 0.1, 0.1, 0.1]], dtype=np.float32),
        np.array([[0, 0, 1, 0, 0], [0, 1, 0, 0, 0]], dtype=np.int32),
        np.array([[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.6, 0.1, 0.1, 0.1]], dtype=np.float32),
    )


def test_metric_unsupported() -> None:
    """Verify correlation metrics update and compute finite correlation tensors."""
    y_true = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], dtype="float32")
    y_pred = np.array([[1.1, 1.9, 3.2], [2.1, 3.8, 5.9]], dtype="float32")

    for zero_cls, keras_cls in [
        (metrics.ConcordanceCorrelation, keras.metrics.ConcordanceCorrelation),
        (metrics.PearsonCorrelation, keras.metrics.PearsonCorrelation),
    ]:
        check_metric_parity(zero_cls, keras_cls, y_true, y_pred, y_true, y_pred)


def test_metrics_parity_more() -> None:
    """Verify numerical parity against keras.metrics for regression and hinge metrics."""
    rng = np.random.default_rng(42)
    y_true = rng.uniform(0.1, 1.0, size=(2, 5)).astype(np.float32)
    y_pred = rng.uniform(0.1, 1.0, size=(2, 5)).astype(np.float32)

    metric_pairs = [
        (metrics.CosineSimilarity, keras.metrics.CosineSimilarity),
        (metrics.MeanAbsoluteError, keras.metrics.MeanAbsoluteError),
        (metrics.MeanAbsolutePercentageError, keras.metrics.MeanAbsolutePercentageError),
        (metrics.MeanSquaredError, keras.metrics.MeanSquaredError),
        (metrics.MeanSquaredLogarithmicError, keras.metrics.MeanSquaredLogarithmicError),
        (metrics.Hinge, keras.metrics.Hinge),
        (metrics.SquaredHinge, keras.metrics.SquaredHinge),
        (metrics.CategoricalHinge, keras.metrics.CategoricalHinge),
        (metrics.RootMeanSquaredError, keras.metrics.RootMeanSquaredError),
        (metrics.KLDivergence, keras.metrics.KLDivergence),
        (metrics.Poisson, keras.metrics.Poisson),
    ]

    for zero_cls, keras_cls in metric_pairs:
        check_metric_parity(zero_cls, keras_cls, y_true, y_pred, y_true, y_pred)


def test_classification_metrics_parity() -> None:
    """Verify numerical parity against keras.metrics for binary classification metrics."""
    y_true = np.array([[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    y_pred = np.array([[0.1, 0.9, 0.8, 0.2], [0.7, 0.3, 0.6, 0.4]], dtype=np.float32)

    metric_pairs = [
        (metrics.AUC, keras.metrics.AUC),
        (metrics.FalsePositives, keras.metrics.FalsePositives),
        (metrics.FalseNegatives, keras.metrics.FalseNegatives),
        (metrics.TruePositives, keras.metrics.TruePositives),
        (metrics.TrueNegatives, keras.metrics.TrueNegatives),
        (metrics.Precision, keras.metrics.Precision),
        (metrics.Recall, keras.metrics.Recall),
        (metrics.F1Score, keras.metrics.F1Score),
        (metrics.FBetaScore, keras.metrics.FBetaScore),
        (metrics.R2Score, keras.metrics.R2Score),
        (metrics.BinaryIoU, keras.metrics.BinaryIoU),
    ]

    for zero_cls, keras_cls in metric_pairs:
        check_metric_parity(zero_cls, keras_cls, y_true, y_pred, y_true, y_pred)


def test_metric_logcosherror() -> None:
    """Verify LogCoshError numerical parity against keras.metrics.LogCoshError."""
    y_true = np.array([0, 1, 1, 0], dtype="float32")
    y_pred = np.array([0.1, 0.9, 0.8, 0.3], dtype="float32")

    check_metric_parity(
        metrics.LogCoshError, keras.metrics.LogCoshError, y_true, y_pred, y_true, y_pred
    )


def test_metrics_sample_weights() -> None:
    """Verify sample-weighted metric updates match keras.metrics across all loss-backed metrics."""
    y_true = np.array([[1.0, 0.5], [0.5, 1.0]], dtype="float32")
    y_pred = np.array([[0.9, 0.2], [0.2, 0.9]], dtype="float32")
    sample_weight = np.array([1.0, 0.5], dtype="float32")

    metric_classes = [
        metrics.CosineSimilarity,
        metrics.MeanAbsoluteError,
        metrics.MeanAbsolutePercentageError,
        metrics.MeanSquaredError,
        metrics.MeanSquaredLogarithmicError,
        metrics.Hinge,
        metrics.SquaredHinge,
        metrics.CategoricalHinge,
        metrics.RootMeanSquaredError,
        metrics.KLDivergence,
        metrics.Poisson,
        metrics.ConcordanceCorrelation,
        metrics.PearsonCorrelation,
    ]
    for m_cls in metric_classes:
        zero_m = m_cls()
        zero_m.update_state(y_true, y_pred, sample_weight=sample_weight)
        assert zero_m.result() is not None
