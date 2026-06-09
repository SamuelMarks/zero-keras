"""Tests for zero_keras losses."""

import numpy as np
from zero_keras import losses


def test_losses():
    y_true_binary = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_pred_binary = np.array([[0.9, 0.1], [0.1, 0.9]])

    y_true_cat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y_pred_cat = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])

    y_true_sparse = np.array([0, 1])

    assert isinstance(
        losses.BinaryCrossentropy()(y_true_binary, y_pred_binary), np.float64
    )
    assert isinstance(
        losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)(
            y_true_binary, y_pred_binary
        ),
        np.float64,
    )
    assert isinstance(
        losses.BinaryFocalCrossentropy()(y_true_binary, y_pred_binary), np.float64
    )
    assert isinstance(
        losses.BinaryFocalCrossentropy(
            apply_class_balancing=True, from_logits=True, label_smoothing=0.1
        )(y_true_binary, y_pred_binary),
        np.float64,
    )
    assert isinstance(losses.CTC()(y_true_binary, y_pred_binary), np.float64)
    assert isinstance(
        losses.CategoricalCrossentropy()(y_true_cat, y_pred_cat), np.float64
    )
    assert isinstance(
        losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)(
            y_true_cat, y_pred_cat
        ),
        np.float64,
    )
    assert isinstance(
        losses.CategoricalFocalCrossentropy()(y_true_cat, y_pred_cat), np.float64
    )
    assert isinstance(
        losses.CategoricalFocalCrossentropy(from_logits=True, label_smoothing=0.1)(
            y_true_cat, y_pred_cat
        ),
        np.float64,
    )
    assert isinstance(
        losses.CategoricalGeneralizedCrossEntropy()(y_true_cat, y_pred_cat), np.float64
    )
    assert isinstance(losses.CategoricalHinge()(y_true_cat, y_pred_cat), np.float64)
    assert isinstance(losses.Circle()(y_true_cat, y_pred_cat), np.float64)
    assert isinstance(losses.CosineSimilarity()(y_true_cat, y_pred_cat), np.float64)
    assert isinstance(losses.Dice()(y_true_cat, y_pred_cat), np.float64)
    assert isinstance(losses.Hinge()(y_true_cat, y_pred_cat), np.float64)
    assert isinstance(losses.Huber()(y_true_cat, y_pred_cat), np.float64)
    assert isinstance(losses.KLDivergence()(y_true_cat, y_pred_cat), np.float64)
    assert isinstance(losses.LogCosh()(y_true_cat, y_pred_cat), np.float64)
    assert isinstance(losses.MeanAbsoluteError()(y_true_cat, y_pred_cat), np.float64)
    assert isinstance(
        losses.MeanAbsolutePercentageError()(y_true_cat, y_pred_cat), np.float64
    )
    assert isinstance(losses.MeanSquaredError()(y_true_cat, y_pred_cat), np.float64)
    assert isinstance(
        losses.MeanSquaredLogarithmicError()(y_true_cat, y_pred_cat), np.float64
    )
    assert isinstance(losses.Poisson()(y_true_cat, y_pred_cat), np.float64)
    assert isinstance(
        losses.SparseCategoricalCrossentropy()(y_true_sparse, y_pred_cat), np.float64
    )
    assert isinstance(
        losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=1)(
            y_true_sparse, y_pred_cat
        ),
        np.float64,
    )
    assert isinstance(losses.SquaredHinge()(y_true_cat, y_pred_cat), np.float64)
    assert isinstance(losses.Tversky()(y_true_cat, y_pred_cat), np.float64)


def test_reductions():
    y_true = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_pred = np.array([[0.9, 0.1], [0.1, 0.9]])

    assert isinstance(
        losses.MeanSquaredError(reduction="sum")(y_true, y_pred), np.float64
    )
    assert isinstance(
        losses.MeanSquaredError(reduction="none")(y_true, y_pred), np.ndarray
    )
    assert isinstance(
        losses.MeanSquaredError(reduction="sum_over_batch_size")(
            y_true, y_pred, sample_weight=np.array([1.0, 0.5])
        ),
        np.float64,
    )


def test_base_loss():
    assert isinstance(losses.Loss()(1, 1), np.float64)
    assert isinstance(losses.Loss(reduction="none")(1, 1), np.ndarray)
