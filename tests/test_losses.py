"""Tests for zero_keras losses."""

import numpy as np
from zero_keras import losses


def test_losses():
    y_true_binary = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_pred_binary = np.array([[0.9, 0.1], [0.1, 0.9]])

    y_true_cat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y_pred_cat = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])

    y_true_sparse = np.array([0, 1])

    res = losses.BinaryCrossentropy()(y_true_binary, y_pred_binary)
    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray)
    )

    res = losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)(
        y_true_binary, y_pred_binary
    )
    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray)
    )

    res = losses.BinaryFocalCrossentropy()(y_true_binary, y_pred_binary)
    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray)
    )

    res = losses.BinaryFocalCrossentropy(
        apply_class_balancing=True, from_logits=True, label_smoothing=0.1
    )(y_true_binary, y_pred_binary)
    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray)
    )

    res = losses.CategoricalCrossentropy()(y_true_cat, y_pred_cat)
    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray)
    )

    res = losses.CategoricalFocalCrossentropy()(y_true_cat, y_pred_cat)
    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray)
    )

    res = losses.CategoricalHinge()(y_true_cat, y_pred_cat)
    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray)
    )

    res = losses.CosineSimilarity()(y_true_binary, y_pred_binary)
    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray)
    )

    res = losses.Hinge()(y_true_binary, y_pred_binary)
    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray)
    )

    res = losses.Huber()(y_true_binary, y_pred_binary)
    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray)
    )

    res = losses.KLDivergence()(y_true_cat, y_pred_cat)
    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray)
    )

    res = losses.LogCosh()(y_true_binary, y_pred_binary)
    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray)
    )

    res = losses.MeanAbsoluteError()(y_true_binary, y_pred_binary)
    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray)
    )

    res = losses.MeanAbsolutePercentageError()(y_true_binary, y_pred_binary)
    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray)
    )

    res = losses.MeanSquaredError()(y_true_binary, y_pred_binary)
    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray)
    )

    res = losses.MeanSquaredLogarithmicError()(y_true_binary, y_pred_binary)
    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray)
    )

    res = losses.Poisson()(y_true_binary, y_pred_binary)
    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray)
    )

    res = losses.SparseCategoricalCrossentropy()(y_true_sparse, y_pred_cat)
    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray)
    )

    res = losses.SquaredHinge()(y_true_binary, y_pred_binary)
    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray)
    )


def test_reductions():
    y_true = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_pred = np.array([[0.9, 0.1], [0.1, 0.9]])

    res = losses.MeanSquaredError(reduction="sum")(y_true, y_pred)
    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray)
    )

    res = losses.MeanSquaredError(reduction="none")(y_true, y_pred)
    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray)
    )

    res = losses.MeanSquaredError(reduction="sum_over_batch_size")(
        y_true, y_pred, sample_weight=np.array([1.0, 0.5])
    )
    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray)
    )


def test_base_loss():
    loss = losses.Loss()
    res = loss(np.array([1.0]), np.array([1.0]))
    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray)
    )
