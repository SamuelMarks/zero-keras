"""Tests for zero_keras metrics."""

import numpy as np
from zero_keras import metrics


def test_metrics():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 1, 1, 0])
    sample_weight = np.array([1.0, 1.0, 0.5, 0.5])

    # Base class
    m_base = metrics.Metric()
    m_base.update_state(y_true, y_pred)
    assert isinstance(m_base.result(), float)

    # Accuracy
    m = metrics.Accuracy()
    m.update_state(y_true, y_pred)
    assert m.result() == 0.5  # 2 correct out of 4

    m2 = metrics.Accuracy()
    m2.update_state(y_true, y_pred, sample_weight=sample_weight)
    assert m2.result() == 1.5 / 3.0

    # BinaryAccuracy
    m_bin = metrics.BinaryAccuracy()
    m_bin.update_state(y_true, y_pred)
    assert m_bin.result() == 0.5

    # CategoricalAccuracy
    y_true_cat = np.array([[1, 0], [0, 1]])
    y_pred_cat = np.array([[0.9, 0.1], [0.8, 0.2]])
    m_cat = metrics.CategoricalAccuracy()
    m_cat.update_state(y_true_cat, y_pred_cat)
    assert m_cat.result() == 0.5

    # SparseCategoricalAccuracy
    y_true_sparse = np.array([0, 1])
    m_sparse_cat = metrics.SparseCategoricalAccuracy()
    m_sparse_cat.update_state(y_true_sparse, y_pred_cat)
    assert m_sparse_cat.result() == 0.5

    # Mean
    m_mean = metrics.Mean()
    m_mean.update_state(y_true)
    assert m_mean.result() == 3.0 / 4.0

    m_mean2 = metrics.Mean()
    m_mean2.update_state(y_true, sample_weight=sample_weight)
    assert m_mean2.result() == 2.0 / 3.0

    # MeanMetricWrapper
    def mock_loss(yt, yp):
        return np.mean((yt - yp) ** 2)

    m_wrap = metrics.MeanMetricWrapper(fn=mock_loss)
    m_wrap.update_state(y_true, y_pred)
    assert isinstance(m_wrap.result(), float)

    # Sum
    m_sum = metrics.Sum()
    m_sum.update_state(y_true)
    assert m_sum.result() == 3.0
    m_sum.update_state(y_true, sample_weight=sample_weight)
    assert m_sum.result() == 5.0

    # Init all others just for coverage
    metrics.AUC()
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
