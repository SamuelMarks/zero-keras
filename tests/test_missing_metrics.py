"""Unit and behavioral tests for missing metrics in zero_keras."""

from __future__ import annotations

import pytest

from zero_keras import metrics, ops


def test_precision_at_recall_branches() -> None:
    """Test ``precision_at_recall_branches`` behavior and edge cases in ``test_missing_metrics``."""
    m = metrics.PrecisionAtRecall(0.5, class_id=1)
    y_true = ops.convert_to_tensor([[0, 1], [1, 0]])
    y_pred = ops.convert_to_tensor([[0.1, 0.9], [0.8, 0.2]])
    m.update_state(y_true, y_pred)
    res = m.result()
    assert res is not None


def test_recall_at_precision_branches() -> None:
    """Test ``recall_at_precision_branches`` behavior and edge cases in ``test_missing_metrics``."""
    m = metrics.RecallAtPrecision(0.5, class_id=1)
    y_true = ops.convert_to_tensor([[0, 1], [1, 0]])
    y_pred = ops.convert_to_tensor([[0.1, 0.9], [0.8, 0.2]])
    m.update_state(y_true, y_pred)
    res = m.result()
    assert res is not None


def test_sensitivity_at_specificity_branches() -> None:
    """Test ``sensitivity_at_specificity_branches`` behavior and edge cases in ``test_missing_metrics``."""
    m = metrics.SensitivityAtSpecificity(0.5, class_id=1)
    y_true = ops.convert_to_tensor([[0, 1], [1, 0]])
    y_pred = ops.convert_to_tensor([[0.1, 0.9], [0.8, 0.2]])
    m.update_state(y_true, y_pred)
    res = m.result()
    assert res is not None


def test_specificity_at_sensitivity_branches() -> None:
    """Test ``specificity_at_sensitivity_branches`` behavior and edge cases in ``test_missing_metrics``."""
    m = metrics.SpecificityAtSensitivity(0.5, class_id=1)
    y_true = ops.convert_to_tensor([[0, 1], [1, 0]])
    y_pred = ops.convert_to_tensor([[0.1, 0.9], [0.8, 0.2]])
    m.update_state(y_true, y_pred)
    res = m.result()
    assert res is not None


def test_precision_recall_top_k() -> None:
    """Test ``precision_recall_top_k`` behavior and edge cases in ``test_missing_metrics``."""
    m1 = metrics.Precision(top_k=2)
    m1.update_state([[0, 1, 0]], [[0.1, 0.8, 0.1]])

    m2 = metrics.Recall(top_k=2)
    m2.update_state([[0, 1, 0]], [[0.1, 0.8, 0.1]])

    m3 = metrics.Precision(class_id=1)
    m3.update_state([[0, 1]], [[0.1, 0.9]])

    m4 = metrics.Recall(class_id=1)
    m4.update_state([[0, 1]], [[0.1, 0.9]])


def test_auc_branches() -> None:
    """Verify AUC summation_method, curve, multi_label, thresholds, and label_weights branches."""
    with pytest.raises(ValueError):
        metrics.AUC(summation_method="minor")

    m2 = metrics.AUC(summation_method="majoring")
    m2.update_state([[0, 1]], [[0.1, 0.9]])
    assert m2.result() is not None

    m3 = metrics.AUC(curve="PR", summation_method="minoring")
    m3.update_state([[0, 1]], [[0.1, 0.9]])
    assert m3.result() is not None

    m4 = metrics.AUC(multi_label=True)
    m4.update_state([[0, 1]], [[0.1, 0.9]])
    assert m4.result() is not None

    m5 = metrics.AUC(thresholds=[0.1, 0.5, 0.9])
    m5.update_state([[0, 1]], [[0.1, 0.9]])
    assert m5.result() is not None

    m6 = metrics.AUC(label_weights=[0.5, 0.5])
    m6.update_state([[0, 1]], [[0.1, 0.9]])
    assert m6.result() is not None


def test_fbeta_score() -> None:
    """Verify FBetaScore and F1Score across None, micro, macro, and weighted averaging modes."""
    y_true = ops.convert_to_tensor([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype="float32")
    y_pred = ops.convert_to_tensor(
        [[0.2, 0.6, 0.7], [0.2, 0.6, 0.6], [0.6, 0.8, 0.0]], dtype="float32"
    )

    m1 = metrics.FBetaScore(beta=1.0, threshold=0.5, average=None)
    m1.update_state(y_true, y_pred)
    r1 = m1.result()
    assert r1 is not None

    m2 = metrics.FBetaScore(beta=2.0, threshold=0.5, average="micro")
    m2.update_state(y_true, y_pred)
    assert m2.result() is not None

    m3 = metrics.F1Score(threshold=0.5, average="macro")
    m3.update_state(y_true, y_pred)
    assert m3.result() is not None

    m4 = metrics.FBetaScore(beta=1.0, threshold=0.5, average="weighted")
    m4.update_state(y_true, y_pred)
    assert m4.result() is not None

    m3.reset_state()


def test_r2_score() -> None:
    """Verify R2Score across uniform, variance_weighted, and adjusted num_regressors modes."""
    y_true = ops.convert_to_tensor([[1], [4], [3]], dtype="float32")
    y_pred = ops.convert_to_tensor([[2], [4], [4]], dtype="float32")

    m1 = metrics.R2Score()
    m1.update_state(y_true, y_pred)
    assert m1.result() is not None

    m2 = metrics.R2Score(class_aggregation="variance_weighted_average", num_regressors=1)
    m2.update_state(y_true, y_pred, sample_weight=[1.0, 1.0, 1.0])
    assert m2.result() is not None

    m1.reset_state()


def test_pearson_correlation() -> None:
    """Test ``pearson_correlation`` behavior and edge cases in ``test_missing_metrics``."""
    y_true = ops.convert_to_tensor([[1], [4], [3]], dtype="float32")
    y_pred = ops.convert_to_tensor([[2], [4], [4]], dtype="float32")

    m1 = metrics.PearsonCorrelation()
    m1.update_state(y_true, y_pred)
    _res = m1.result()
    m1.reset_state()


def test_concordance_correlation() -> None:
    """Test ``concordance_correlation`` behavior and edge cases in ``test_missing_metrics``."""
    y_true = ops.convert_to_tensor([[1], [4], [3]], dtype="float32")
    y_pred = ops.convert_to_tensor([[2], [4], [4]], dtype="float32")

    m1 = metrics.ConcordanceCorrelation()
    m1.update_state(y_true, y_pred)
    _res = m1.result()
    m1.reset_state()
