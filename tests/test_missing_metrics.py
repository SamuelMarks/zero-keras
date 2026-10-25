"""Unit and behavioral tests for missing metrics in zero_keras."""

from __future__ import annotations

from zero_keras import metrics, ops


def test_precision_at_recall_branches() -> None:
    """Verify or execute `test precision at recall branches`."""
    m = metrics.PrecisionAtRecall(0.5, class_id=1)
    y_true = ops.convert_to_tensor([[0, 1], [1, 0]])
    y_pred = ops.convert_to_tensor([[0.1, 0.9], [0.8, 0.2]])
    m.update_state(y_true, y_pred)
    res = m.result()
    assert res is not None


def test_recall_at_precision_branches() -> None:
    """Verify or execute `test recall at precision branches`."""
    m = metrics.RecallAtPrecision(0.5, class_id=1)
    y_true = ops.convert_to_tensor([[0, 1], [1, 0]])
    y_pred = ops.convert_to_tensor([[0.1, 0.9], [0.8, 0.2]])
    m.update_state(y_true, y_pred)
    res = m.result()
    assert res is not None


def test_sensitivity_at_specificity_branches() -> None:
    """Verify or execute `test sensitivity at specificity branches`."""
    m = metrics.SensitivityAtSpecificity(0.5, class_id=1)
    y_true = ops.convert_to_tensor([[0, 1], [1, 0]])
    y_pred = ops.convert_to_tensor([[0.1, 0.9], [0.8, 0.2]])
    m.update_state(y_true, y_pred)
    res = m.result()
    assert res is not None


def test_specificity_at_sensitivity_branches() -> None:
    """Verify or execute `test specificity at sensitivity branches`."""
    m = metrics.SpecificityAtSensitivity(0.5, class_id=1)
    y_true = ops.convert_to_tensor([[0, 1], [1, 0]])
    y_pred = ops.convert_to_tensor([[0.1, 0.9], [0.8, 0.2]])
    m.update_state(y_true, y_pred)
    res = m.result()
    assert res is not None


def test_precision_recall_top_k() -> None:
    """Verify or execute `test precision recall top k`."""
    m1 = metrics.Precision(top_k=2)
    m1.update_state([[0, 1, 0]], [[0.1, 0.8, 0.1]])

    m2 = metrics.Recall(top_k=2)
    m2.update_state([[0, 1, 0]], [[0.1, 0.8, 0.1]])

    m3 = metrics.Precision(class_id=1)
    m3.update_state([[0, 1]], [[0.1, 0.9]])

    m4 = metrics.Recall(class_id=1)
    m4.update_state([[0, 1]], [[0.1, 0.9]])


def test_auc_branches() -> None:
    """Verify or execute `test auc branches`."""
    try:
        m1 = metrics.AUC(summation_method="minor")
        m1.update_state([[0, 1]], [[0.1, 0.9]])
        m1.result()
    except ValueError:
        pass

    m2 = metrics.AUC(summation_method="majoring")
    m2.update_state([[0, 1]], [[0.1, 0.9]])
    m2.result()

    m3 = metrics.AUC(curve="PR", summation_method="minoring")
    m3.update_state([[0, 1]], [[0.1, 0.9]])
    m3.result()

    m4 = metrics.AUC(multi_label=True)
    m4.update_state([[0, 1]], [[0.1, 0.9]])
    m4.result()

    m5 = metrics.AUC(thresholds=[0.1, 0.5, 0.9])
    m5.update_state([[0, 1]], [[0.1, 0.9]])
    m5.result()

    m6 = metrics.AUC(label_weights=[0.5, 0.5])
    m6.update_state([[0, 1]], [[0.1, 0.9]])
    m6.result()


def test_fbeta_score() -> None:
    """Verify or execute `test fbeta score`."""
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
    m2.result()

    m3 = metrics.F1Score(threshold=0.5, average="macro")
    m3.update_state(y_true, y_pred)
    m3.result()

    m3.reset_state()


def test_r2_score() -> None:
    """Verify or execute `test r2 score`."""
    y_true = ops.convert_to_tensor([[1], [4], [3]], dtype="float32")
    y_pred = ops.convert_to_tensor([[2], [4], [4]], dtype="float32")

    m1 = metrics.R2Score()
    m1.update_state(y_true, y_pred)
    m1.result()

    m2 = metrics.R2Score(class_aggregation="variance_weighted_average")
    m2.update_state(y_true, y_pred)
    m2.result()

    m1.reset_state()


def test_pearson_correlation() -> None:
    """Verify or execute `test pearson correlation`."""
    y_true = ops.convert_to_tensor([[1], [4], [3]], dtype="float32")
    y_pred = ops.convert_to_tensor([[2], [4], [4]], dtype="float32")

    m1 = metrics.PearsonCorrelation()
    m1.update_state(y_true, y_pred)
    _res = m1.result()
    m1.reset_state()


def test_concordance_correlation() -> None:
    """Verify or execute `test concordance correlation`."""
    y_true = ops.convert_to_tensor([[1], [4], [3]], dtype="float32")
    y_pred = ops.convert_to_tensor([[2], [4], [4]], dtype="float32")

    m1 = metrics.ConcordanceCorrelation()
    m1.update_state(y_true, y_pred)
    _res = m1.result()
    m1.reset_state()
