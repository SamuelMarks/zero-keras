import pytest

"""Tests for zero_keras metrics."""

import numpy as np
from zero_keras import metrics


@pytest.mark.skip(reason="pending")
def test_metrics():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 1, 1, 0])
    sample_weight = np.array([1.0, 1.0, 0.5, 0.5])

    # Base class
    m_base = metrics.Metric()
    m_base.update_state(y_true, y_pred)
    res = m_base.result()

    # Accuracy
    m = metrics.Accuracy()
    m.update_state(y_true, y_pred)
    res = m.result()
    if hasattr(res, "numpy"):
        res = res.numpy()
    assert np.allclose(res, 0.5)

    m2 = metrics.Accuracy()
    m2.update_state(y_true, y_pred, sample_weight=sample_weight)
    res = m2.result()
    if hasattr(res, "numpy"):
        res = res.numpy()
    assert np.allclose(res, 1.5 / 3.0)

    # BinaryAccuracy
    m_bin = metrics.BinaryAccuracy()
    m_bin.update_state(y_true, y_pred)
    res = m_bin.result()
    if hasattr(res, "numpy"):
        res = res.numpy()
    assert np.allclose(res, 0.5)

    # CategoricalAccuracy
    y_true_cat = np.array([[1, 0], [0, 1]])
    y_pred_cat = np.array([[0.9, 0.1], [0.8, 0.2]])
    m_cat = metrics.CategoricalAccuracy()
    m_cat.update_state(y_true_cat, y_pred_cat)
    res = m_cat.result()
    if hasattr(res, "numpy"):
        res = res.numpy()
    assert np.allclose(res, 0.5)

    # SparseCategoricalAccuracy
    y_true_sparse = np.array([0, 1])
    m_sparse_cat = metrics.SparseCategoricalAccuracy()
    m_sparse_cat.update_state(y_true_sparse, y_pred_cat)
    res = m_sparse_cat.result()
    if hasattr(res, "numpy"):
        res = res.numpy()
    assert np.allclose(res, 0.5)

    # Mean
    m_mean = metrics.Mean()
    m_mean.update_state(y_true)
    res = m_mean.result()
    if hasattr(res, "numpy"):
        res = res.numpy()
    assert np.allclose(res, 3.0 / 4.0)

    m_mean2 = metrics.Mean()
    m_mean2.update_state(y_true, sample_weight=sample_weight)
    res = m_mean2.result()
    if hasattr(res, "numpy"):
        res = res.numpy()
    assert np.allclose(res, 2.0 / 3.0)

    # Sum
    m_sum = metrics.Sum()
    m_sum.update_state(y_true)
    assert m_sum.result() == 3.0
    m_sum.update_state(y_true, sample_weight=sample_weight)
    assert m_sum.result() == 5.0

    # Init all others just for coverage

    metrics.BinaryCrossentropy()
    metrics.BinaryIoU()
    metrics.CategoricalCrossentropy()
    metrics.CategoricalHinge()
    metrics.ConcordanceCorrelation()
    metrics.CosineSimilarity()
    metrics.F1Score()
    metrics.FBetaScore()
    metrics.FalseNegatives()
    metrics.FalsePositives()
    metrics.Hinge()
    metrics.IoU(num_classes=2, target_class_ids=[0])
    metrics.KLDivergence()
    metrics.LogCoshError()
    metrics.MeanAbsoluteError()
    metrics.MeanAbsolutePercentageError()
    metrics.MeanIoU(num_classes=2)
    metrics.MeanSquaredError()
    metrics.MeanSquaredLogarithmicError()
    metrics.OneHotIoU(num_classes=2, target_class_ids=[0])
    metrics.OneHotMeanIoU(num_classes=2)
    metrics.PearsonCorrelation()
    metrics.Poisson()
    metrics.Precision()
    metrics.PrecisionAtRecall(recall=0.5)
    metrics.R2Score()
    metrics.Recall()
    metrics.RecallAtPrecision(precision=0.5)
    metrics.RootMeanSquaredError()
    metrics.SensitivityAtSpecificity(specificity=0.5)
    metrics.SparseCategoricalCrossentropy()
    metrics.SparseTopKCategoricalAccuracy()
    metrics.SpecificityAtSensitivity(sensitivity=0.5)
    metrics.SquaredHinge()
    metrics.TopKCategoricalAccuracy()
    metrics.TrueNegatives()
    metrics.TruePositives()


@pytest.mark.skip(reason="pending")
def test_metrics_coverage():
    from zero_keras.metrics import _get_keras_metric, Metric
    from ml_switcheroo.core.config import config

    # test 12
    # mock eager_mode to False
    old_eager = config.eager_mode
    config.eager_mode = False
    assert _get_keras_metric("Mean") is None
    config.eager_mode = old_eager

    # test 53-54
    m = Metric(name="dummy")
    # since we don't implement anything, result() will return 0.0
    res = m("foo", bar="baz")
    assert res == 0.0


@pytest.mark.skip(reason="pending")
def test_metrics_reset_state():
    from zero_keras.metrics import Metric

    m = Metric(name="dummy")
    m.reset_state()  # _keras_metric is None, so this should just pass silently

    class MockKerasMetric:
        def __init__(self):
            self.reset_called = False

        def reset_state(self):
            self.reset_called = True

    m._keras_metric = MockKerasMetric()
    m.reset_state()
    assert m._keras_metric.reset_called
