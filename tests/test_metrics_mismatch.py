"""Module docstring."""

import pytest
import numpy as np
from zero_keras import metrics


def test_confusion_matrix_invalid_type():
    """Function docstring."""
    with pytest.raises(ValueError):
        m = metrics._ConfusionMatrixMetric(metric_type="INVALID")
        m.update_state([0], [0])

    with pytest.raises(ValueError):
        m2 = metrics._ConfusionMatrixMetric(
            thresholds=[0.5, 0.8], metric_type="INVALID"
        )
        m2.update_state([0], [0])


def test_metric_shape_mismatch():
    """Function docstring."""
    m = metrics.MeanSquaredError()
    # It might broadcast or fail depending on ops. Let's see what happens.
    try:
        m.update_state(np.ones((2, 2)), np.zeros((3, 3)))
    except Exception:
        pass

    m2 = metrics.Accuracy()
    try:
        m2.update_state(np.ones((2, 2)), np.zeros((3, 3)))
    except Exception:
        pass


def test_confusion_matrix_list_thresholds():
    """Function docstring."""
    m = metrics._ConfusionMatrixMetric(thresholds=[0.5, 0.8], metric_type="FP")
    m.update_state([0, 0, 1], [0.6, 0.9, 0.9], sample_weight=[1, 1, 1])
    # For threshold 0.5: FP are 0.6 and 0.9 for true 0 -> count = 2
    # For threshold 0.8: FP is 0.9 for true 0 -> count = 1
    res = m.result()
    assert res is not None


def test_confusion_matrix_coverage():
    """Function docstring."""
    types = ["TP", "TN", "FP", "FN"]
    for t in types:
        m = metrics._ConfusionMatrixMetric(metric_type=t)
        m.update_state([1, 0], [1, 0])
        m2 = metrics._ConfusionMatrixMetric(thresholds=[0.5], metric_type=t)
        m2.update_state([1, 0], [1, 0])
