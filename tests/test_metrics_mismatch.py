"""Unit and behavioral tests for metrics mismatch in zero_keras."""

from __future__ import annotations

import pytest

from zero_keras import metrics, ops


def test_confusion_matrix_invalid_type() -> None:
    """Test ``confusion_matrix_invalid_type`` behavior and edge cases in ``test_metrics_mismatch``."""
    with pytest.raises(ValueError):
        m = metrics._ConfusionMatrixMetric(metric_type="INVALID")
        m.update_state([0], [0])

    with pytest.raises(ValueError):
        m2 = metrics._ConfusionMatrixMetric(thresholds=[0.5, 0.8], metric_type="INVALID")
        m2.update_state([0], [0])


def test_metric_shape_mismatch() -> None:
    """Verify MeanSquaredError and Accuracy raise ValueError on incompatible shapes."""
    m = metrics.MeanSquaredError()
    with pytest.raises(ValueError):
        m.update_state(ops.ones((2, 2)), ops.zeros((3, 3)))

    m2 = metrics.Accuracy()
    with pytest.raises(ValueError):
        m2.update_state(ops.ones((2, 2)), ops.zeros((3, 3)))


def test_confusion_matrix_list_thresholds() -> None:
    """Test ``confusion_matrix_list_thresholds`` behavior and edge cases in ``test_metrics_mismatch``."""
    m = metrics._ConfusionMatrixMetric(thresholds=[0.5, 0.8], metric_type="FP")
    m.update_state([0, 0, 1], [0.6, 0.9, 0.9], sample_weight=[1, 1, 1])
    res = m.result()
    assert res is not None


def test_confusion_matrix_coverage() -> None:
    """Test ``confusion_matrix_coverage`` behavior and edge cases in ``test_metrics_mismatch``."""
    types = ["TP", "TN", "FP", "FN"]
    for t in types:
        m = metrics._ConfusionMatrixMetric(metric_type=t)
        m.update_state([1, 0], [1, 0])
        m2 = metrics._ConfusionMatrixMetric(thresholds=[0.5], metric_type=t)
        m2.update_state([1, 0], [1, 0])
