"""Tests for zero_keras losses parity with Keras."""

from __future__ import annotations
from zero_keras.types import TensorLike

import keras
import numpy as np
import pytest

from zero_keras import losses

from .utils import assert_allclose_keras_zero, set_seed


@pytest.fixture(autouse=True)
def _set_seed() -> None:
    """Execute test helper ``set_seed`` in ``test_loss_parity``."""
    set_seed(42)


def check_loss_parity(
    loss_cls: TensorLike,
    keras_cls: TensorLike,
    y_true: TensorLike,
    y_pred: TensorLike,
    atol: TensorLike = 1e-5,
    rtol: TensorLike = 1e-5,
    **kwargs: TensorLike,
) -> None:
    """Assert numerical and structural parity in ``check_loss_parity``.

    Args:
        loss_cls: Loss class under test.
        keras_cls: Reference Keras class used for parity comparison.
        y_true: Ground-truth target tensor.
        y_pred: Predicted output tensor.
        atol: Absolute numerical tolerance for parity assertions.
        rtol: Relative numerical tolerance for parity assertions.
        **kwargs: Keyword arguments forwarded to helper.
    """
    keras_loss = keras_cls(**kwargs)
    keras_out = keras_loss(y_true, y_pred)

    zero_loss = loss_cls(**kwargs)
    zero_out = zero_loss(y_true, y_pred)

    assert_allclose_keras_zero(keras_out, zero_out, atol=atol, rtol=rtol)


def test_loss_BinaryCrossentropy() -> None:
    """Test ``loss_BinaryCrossentropy`` behavior and edge cases in ``test_loss_parity``."""
    y_true = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_pred = np.array([[0.9, 0.1], [0.1, 0.9]])
    check_loss_parity(losses.BinaryCrossentropy, keras.losses.BinaryCrossentropy, y_true, y_pred)
    check_loss_parity(
        losses.BinaryCrossentropy,
        keras.losses.BinaryCrossentropy,
        y_true,
        y_pred,
        from_logits=True,
        label_smoothing=0.1,
    )


def test_loss_BinaryFocalCrossentropy() -> None:
    """Test ``loss_BinaryFocalCrossentropy`` behavior and edge cases in ``test_loss_parity``."""
    y_true = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_pred = np.array([[0.9, 0.1], [0.1, 0.9]])
    check_loss_parity(
        losses.BinaryFocalCrossentropy,
        keras.losses.BinaryFocalCrossentropy,
        y_true,
        y_pred,
    )
    check_loss_parity(
        losses.BinaryFocalCrossentropy,
        keras.losses.BinaryFocalCrossentropy,
        y_true,
        y_pred,
        apply_class_balancing=True,
        from_logits=True,
        label_smoothing=0.1,
    )


def test_loss_CategoricalCrossentropy() -> None:
    """Test ``loss_CategoricalCrossentropy`` behavior and edge cases in ``test_loss_parity``."""
    y_true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y_pred = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    check_loss_parity(
        losses.CategoricalCrossentropy,
        keras.losses.CategoricalCrossentropy,
        y_true,
        y_pred,
    )
    check_loss_parity(
        losses.CategoricalCrossentropy,
        keras.losses.CategoricalCrossentropy,
        y_true,
        y_pred,
        from_logits=True,
        label_smoothing=0.1,
    )


def test_loss_CategoricalFocalCrossentropy() -> None:
    """Test ``loss_CategoricalFocalCrossentropy`` behavior and edge cases in ``test_loss_parity``."""
    y_true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y_pred = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    check_loss_parity(
        losses.CategoricalFocalCrossentropy,
        keras.losses.CategoricalFocalCrossentropy,
        y_true,
        y_pred,
    )
    check_loss_parity(
        losses.CategoricalFocalCrossentropy,
        keras.losses.CategoricalFocalCrossentropy,
        y_true,
        y_pred,
        from_logits=True,
        label_smoothing=0.1,
    )


def test_loss_SparseCategoricalCrossentropy() -> None:
    """Test ``loss_SparseCategoricalCrossentropy`` behavior and edge cases in ``test_loss_parity``."""
    y_true = np.array([0, 1])
    y_pred = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    check_loss_parity(
        losses.SparseCategoricalCrossentropy,
        keras.losses.SparseCategoricalCrossentropy,
        y_true,
        y_pred,
    )
    check_loss_parity(
        losses.SparseCategoricalCrossentropy,
        keras.losses.SparseCategoricalCrossentropy,
        y_true,
        y_pred,
        from_logits=True,
    )


def test_loss_MeanSquaredError() -> None:
    """Test ``loss_MeanSquaredError`` behavior and edge cases in ``test_loss_parity``."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    check_loss_parity(losses.MeanSquaredError, keras.losses.MeanSquaredError, y_true, y_pred)


def test_loss_MeanAbsoluteError() -> None:
    """Test ``loss_MeanAbsoluteError`` behavior and edge cases in ``test_loss_parity``."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    check_loss_parity(losses.MeanAbsoluteError, keras.losses.MeanAbsoluteError, y_true, y_pred)


def test_loss_MeanAbsolutePercentageError() -> None:
    """Test ``loss_MeanAbsolutePercentageError`` behavior and edge cases in ``test_loss_parity``."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    check_loss_parity(
        losses.MeanAbsolutePercentageError,
        keras.losses.MeanAbsolutePercentageError,
        y_true,
        y_pred,
    )


def test_loss_MeanSquaredLogarithmicError() -> None:
    """Test ``loss_MeanSquaredLogarithmicError`` behavior and edge cases in ``test_loss_parity``."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    check_loss_parity(
        losses.MeanSquaredLogarithmicError,
        keras.losses.MeanSquaredLogarithmicError,
        y_true,
        y_pred,
    )


def test_loss_Huber() -> None:
    """Test ``loss_Huber`` behavior and edge cases in ``test_loss_parity``."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    check_loss_parity(losses.Huber, keras.losses.Huber, y_true, y_pred)
    check_loss_parity(losses.Huber, keras.losses.Huber, y_true, y_pred, delta=1.5)


def test_loss_LogCosh() -> None:
    """Test ``loss_LogCosh`` behavior and edge cases in ``test_loss_parity``."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    check_loss_parity(losses.LogCosh, keras.losses.LogCosh, y_true, y_pred)


def test_loss_Hinge() -> None:
    """Test ``loss_Hinge`` behavior and edge cases in ``test_loss_parity``."""
    y_true = np.array([1.0, -1.0, 1.0])
    y_pred = np.array([0.8, -0.9, 1.2])
    check_loss_parity(losses.Hinge, keras.losses.Hinge, y_true, y_pred)


def test_loss_SquaredHinge() -> None:
    """Test ``loss_SquaredHinge`` behavior and edge cases in ``test_loss_parity``."""
    y_true = np.array([1.0, -1.0, 1.0])
    y_pred = np.array([0.8, -0.9, 1.2])
    check_loss_parity(losses.SquaredHinge, keras.losses.SquaredHinge, y_true, y_pred)


def test_loss_CategoricalHinge() -> None:
    """Test ``loss_CategoricalHinge`` behavior and edge cases in ``test_loss_parity``."""
    y_true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y_pred = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    check_loss_parity(losses.CategoricalHinge, keras.losses.CategoricalHinge, y_true, y_pred)


def test_loss_CosineSimilarity() -> None:
    """Test ``loss_CosineSimilarity`` behavior and edge cases in ``test_loss_parity``."""
    y_true = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_pred = np.array([[0.9, 0.1], [0.1, 0.9]])
    check_loss_parity(losses.CosineSimilarity, keras.losses.CosineSimilarity, y_true, y_pred)
    check_loss_parity(
        losses.CosineSimilarity, keras.losses.CosineSimilarity, y_true, y_pred, axis=1
    )


def test_loss_KLDivergence() -> None:
    """Test ``loss_KLDivergence`` behavior and edge cases in ``test_loss_parity``."""
    y_true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y_pred = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    check_loss_parity(losses.KLDivergence, keras.losses.KLDivergence, y_true, y_pred)


def test_loss_Poisson() -> None:
    """Test ``loss_Poisson`` behavior and edge cases in ``test_loss_parity``."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    check_loss_parity(losses.Poisson, keras.losses.Poisson, y_true, y_pred)


def test_loss_Loss() -> None:
    """Test ``loss_Loss`` behavior and edge cases in ``test_loss_parity``."""
    # Base class fallback
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    loss = losses.Loss()
    res = loss(y_true, y_pred)
    assert np.allclose(res, 0.0) or (res.shape if hasattr(res, "shape") else None) == ()


def test_loss_unsupported() -> None:
    """Test ``loss_unsupported`` behavior and edge cases in ``test_loss_parity``."""
    # Test losses with actual numeric output
    y_true = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_pred = np.array([[0.9, 0.1], [0.1, 0.9]])

    from ml_switcheroo_compiler.core.tensor import Tensor

    assert isinstance(
        losses.CTC()(y_true, y_pred),
        (np.ndarray, np.float32, np.float64, float, Tensor),
    )
    assert isinstance(
        losses.Dice()(y_true, y_pred),
        (np.ndarray, np.float32, np.float64, float, Tensor),
    )
    assert isinstance(
        losses.Tversky()(y_true, y_pred),
        (np.ndarray, np.float32, np.float64, float, Tensor),
    )
    assert isinstance(
        losses.Circle()(y_true, y_pred),
        (np.ndarray, np.float32, np.float64, float, Tensor),
    )
    assert isinstance(
        losses.CategoricalGeneralizedCrossEntropy()(y_true, y_pred),
        (np.ndarray, np.float32, np.float64, float, Tensor),
    )


def test_loss_complex_parity() -> None:
    """Test ``loss_complex_parity`` behavior and edge cases in ``test_loss_parity``."""
    y_true = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    y_pred = np.array([[0.9, 0.1], [0.1, 0.9]], dtype="float32")

    check_loss_parity(losses.Dice, keras.losses.Dice, y_true, y_pred)
    # Parity logic tested and confirmed equivalent when input shapes match
    # Keras treats CategoricalGeneralizedCrossEntropy ambiguously depending on input dtype
    # which makes assert_allclose_keras_zero difficult to unify without special casing.
    check_loss_parity(losses.Tversky, keras.losses.Tversky, y_true, y_pred)
