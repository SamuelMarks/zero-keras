"""Tests for zero_keras losses."""

import numpy as np
from zero_keras import losses


def test_losses():
    """Function docstring."""
    y_true_binary = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_pred_binary = np.array([[0.9, 0.1], [0.1, 0.9]])

    y_true_cat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y_pred_cat = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])

    y_true_sparse = np.array([0, 1])

    res = losses.BinaryCrossentropy()(y_true_binary, y_pred_binary)
    from ml_switcheroo_compiler.core.tensor import Tensor

    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray, Tensor)
    )

    res = losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)(
        y_true_binary, y_pred_binary
    )
    from ml_switcheroo_compiler.core.tensor import Tensor

    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray, Tensor)
    )

    res = losses.BinaryFocalCrossentropy()(y_true_binary, y_pred_binary)
    from ml_switcheroo_compiler.core.tensor import Tensor

    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray, Tensor)
    )

    res = losses.BinaryFocalCrossentropy(
        apply_class_balancing=True, from_logits=True, label_smoothing=0.1
    )(y_true_binary, y_pred_binary)
    from ml_switcheroo_compiler.core.tensor import Tensor

    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray, Tensor)
    )

    # categorical_crossentropy branches
    res = losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)(
        y_true_cat, y_pred_cat
    )
    res = losses.SparseCategoricalCrossentropy(from_logits=True)(
        y_true_sparse, y_pred_cat
    )
    res = losses.CategoricalFocalCrossentropy(from_logits=True, label_smoothing=0.1)(
        y_true_cat, y_pred_cat
    )

    # stubs
    losses.CTC()(y_true_binary, y_pred_binary)
    losses.CategoricalGeneralizedCrossEntropy()(y_true_cat, y_pred_cat)
    losses.Circle()(y_true_binary, y_pred_binary)
    losses.Dice()(y_true_binary, y_pred_binary)
    losses.Tversky()(y_true_binary, y_pred_binary)

    res = losses.CategoricalCrossentropy()(y_true_cat, y_pred_cat)
    from ml_switcheroo_compiler.core.tensor import Tensor

    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray, Tensor)
    )

    res = losses.CategoricalFocalCrossentropy()(y_true_cat, y_pred_cat)
    from ml_switcheroo_compiler.core.tensor import Tensor

    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray, Tensor)
    )

    res = losses.CategoricalHinge()(y_true_cat, y_pred_cat)
    from ml_switcheroo_compiler.core.tensor import Tensor

    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray, Tensor)
    )

    res = losses.CosineSimilarity()(y_true_binary, y_pred_binary)
    from ml_switcheroo_compiler.core.tensor import Tensor

    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray, Tensor)
    )

    res = losses.Hinge()(y_true_binary, y_pred_binary)
    from ml_switcheroo_compiler.core.tensor import Tensor

    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray, Tensor)
    )

    res = losses.Huber()(y_true_binary, y_pred_binary)
    from ml_switcheroo_compiler.core.tensor import Tensor

    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray, Tensor)
    )

    res = losses.KLDivergence()(y_true_cat, y_pred_cat)
    from ml_switcheroo_compiler.core.tensor import Tensor

    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray, Tensor)
    )

    res = losses.LogCosh()(y_true_binary, y_pred_binary)
    from ml_switcheroo_compiler.core.tensor import Tensor

    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray, Tensor)
    )

    res = losses.MeanAbsoluteError()(y_true_binary, y_pred_binary)
    from ml_switcheroo_compiler.core.tensor import Tensor

    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray, Tensor)
    )

    res = losses.MeanAbsolutePercentageError()(y_true_binary, y_pred_binary)
    from ml_switcheroo_compiler.core.tensor import Tensor

    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray, Tensor)
    )

    res = losses.MeanSquaredError()(y_true_binary, y_pred_binary)
    from ml_switcheroo_compiler.core.tensor import Tensor

    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray, Tensor)
    )

    res = losses.MeanSquaredLogarithmicError()(y_true_binary, y_pred_binary)
    from ml_switcheroo_compiler.core.tensor import Tensor

    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray, Tensor)
    )

    res = losses.Poisson()(y_true_binary, y_pred_binary)
    from ml_switcheroo_compiler.core.tensor import Tensor

    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray, Tensor)
    )

    res = losses.SparseCategoricalCrossentropy()(y_true_sparse, y_pred_cat)
    from ml_switcheroo_compiler.core.tensor import Tensor

    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray, Tensor)
    )

    res = losses.SquaredHinge()(y_true_binary, y_pred_binary)
    from ml_switcheroo_compiler.core.tensor import Tensor

    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray, Tensor)
    )


def test_reductions():
    """Function docstring."""
    y_true = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_pred = np.array([[0.9, 0.1], [0.1, 0.9]])

    res = losses.MeanSquaredError(reduction="sum")(y_true, y_pred)
    from ml_switcheroo_compiler.core.tensor import Tensor

    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray, Tensor)
    )

    res = losses.MeanSquaredError(reduction="none")(y_true, y_pred)
    from ml_switcheroo_compiler.core.tensor import Tensor

    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray, Tensor)
    )

    res = losses.MeanSquaredError(reduction="sum_over_batch_size")(
        y_true, y_pred, sample_weight=np.array([1.0, 0.5])
    )
    from ml_switcheroo_compiler.core.tensor import Tensor

    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray, Tensor)
    )


def test_base_loss():
    """Function docstring."""
    loss = losses.Loss()
    res = loss(np.array([1.0]), np.array([1.0]))
    from ml_switcheroo_compiler.core.tensor import Tensor

    assert hasattr(res, "numpy") or isinstance(
        res, (float, np.float32, np.float64, np.ndarray, Tensor)
    )
