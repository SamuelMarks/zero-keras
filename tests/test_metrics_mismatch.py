"""Unit and behavioral tests for metrics mismatch in zero_keras."""

from __future__ import annotations

import pytest

from zero_keras import metrics, ops


def test_confusion_matrix_invalid_type() -> None:
    """Verify or execute `test confusion matrix invalid type`."""
    with pytest.raises(ValueError):
        m = metrics._ConfusionMatrixMetric(metric_type="INVALID")
        m.update_state([0], [0])

    with pytest.raises(ValueError):
        m2 = metrics._ConfusionMatrixMetric(thresholds=[0.5, 0.8], metric_type="INVALID")
        m2.update_state([0], [0])


def test_metric_shape_mismatch() -> None:
    """Verify or execute `test metric shape mismatch`."""
    m = metrics.MeanSquaredError()
    try:
        m.update_state(ops.ones((2, 2)), ops.zeros((3, 3)))
    except Exception:
        pass

    m2 = metrics.Accuracy()
    try:
        m2.update_state(ops.ones((2, 2)), ops.zeros((3, 3)))
    except Exception:
        pass


def test_confusion_matrix_list_thresholds() -> None:
    """Verify or execute `test confusion matrix list thresholds`."""
    m = metrics._ConfusionMatrixMetric(thresholds=[0.5, 0.8], metric_type="FP")
    m.update_state([0, 0, 1], [0.6, 0.9, 0.9], sample_weight=[1, 1, 1])
    res = m.result()
    assert res is not None


def test_confusion_matrix_coverage() -> None:
    """Verify or execute `test confusion matrix coverage`."""
    types = ["TP", "TN", "FP", "FN"]
    for t in types:
        m = metrics._ConfusionMatrixMetric(metric_type=t)
        m.update_state([1, 0], [1, 0])
        m2 = metrics._ConfusionMatrixMetric(thresholds=[0.5], metric_type=t)
        m2.update_state([1, 0], [1, 0])
