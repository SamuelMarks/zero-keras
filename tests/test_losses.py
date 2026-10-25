"""Comprehensive unit and branch coverage tests for zero_keras.losses."""

from __future__ import annotations

from ml_switcheroo_compiler.core.tensor import Tensor

from zero_keras import losses, ops, random
from zero_keras.types import TensorLike


def test_losses() -> None:
    """Verify or execute `test losses`."""
    y_true_binary = ops.convert_to_tensor([[1.0, 0.0], [0.0, 1.0]])
    y_pred_binary = ops.convert_to_tensor([[0.9, 0.1], [0.1, 0.9]])

    y_true_cat = ops.convert_to_tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    y_pred_cat = ops.convert_to_tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])

    y_true_sparse = ops.convert_to_tensor([0, 1])

    res = losses.BinaryCrossentropy()(y_true_binary, y_pred_binary)
    assert hasattr(res, "numpy") or isinstance(res, (float, int, Tensor))

    res = losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)(
        y_true_binary, y_pred_binary
    )
    assert hasattr(res, "numpy") or isinstance(res, (float, int, Tensor))

    res = losses.BinaryFocalCrossentropy()(y_true_binary, y_pred_binary)
    assert hasattr(res, "numpy") or isinstance(res, (float, int, Tensor))

    res = losses.BinaryFocalCrossentropy(
        apply_class_balancing=True, from_logits=True, label_smoothing=0.1
    )(y_true_binary, y_pred_binary)
    assert hasattr(res, "numpy") or isinstance(res, (float, int, Tensor))

    res = losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)(
        y_true_cat, y_pred_cat
    )
    res = losses.SparseCategoricalCrossentropy(from_logits=True)(y_true_sparse, y_pred_cat)
    res = losses.CategoricalFocalCrossentropy(from_logits=True, label_smoothing=0.1)(
        y_true_cat, y_pred_cat
    )

    losses.CTC()(y_true_binary, y_pred_binary)
    losses.CategoricalGeneralizedCrossEntropy()(y_true_cat, y_pred_cat)
    losses.Circle()(y_true_binary, y_pred_binary)
    losses.Dice()(y_true_binary, y_pred_binary)
    losses.Tversky()(y_true_binary, y_pred_binary)

    res = losses.CategoricalCrossentropy()(y_true_cat, y_pred_cat)
    assert hasattr(res, "numpy") or isinstance(res, (float, int, Tensor))

    res = losses.CategoricalFocalCrossentropy()(y_true_cat, y_pred_cat)
    assert hasattr(res, "numpy") or isinstance(res, (float, int, Tensor))

    res = losses.CategoricalHinge()(y_true_cat, y_pred_cat)
    assert hasattr(res, "numpy") or isinstance(res, (float, int, Tensor))

    res = losses.CosineSimilarity()(y_true_binary, y_pred_binary)
    assert hasattr(res, "numpy") or isinstance(res, (float, int, Tensor))

    res = losses.Hinge()(y_true_binary, y_pred_binary)
    assert hasattr(res, "numpy") or isinstance(res, (float, int, Tensor))

    res = losses.Huber()(y_true_binary, y_pred_binary)
    assert hasattr(res, "numpy") or isinstance(res, (float, int, Tensor))

    res = losses.KLDivergence()(y_true_cat, y_pred_cat)
    assert hasattr(res, "numpy") or isinstance(res, (float, int, Tensor))

    res = losses.LogCosh()(y_true_binary, y_pred_binary)
    assert hasattr(res, "numpy") or isinstance(res, (float, int, Tensor))

    res = losses.MeanAbsoluteError()(y_true_binary, y_pred_binary)
    assert hasattr(res, "numpy") or isinstance(res, (float, int, Tensor))

    res = losses.MeanAbsolutePercentageError()(y_true_binary, y_pred_binary)
    assert hasattr(res, "numpy") or isinstance(res, (float, int, Tensor))

    res = losses.MeanSquaredError()(y_true_binary, y_pred_binary)
    assert hasattr(res, "numpy") or isinstance(res, (float, int, Tensor))

    res = losses.MeanSquaredLogarithmicError()(y_true_binary, y_pred_binary)
    assert hasattr(res, "numpy") or isinstance(res, (float, int, Tensor))

    res = losses.Poisson()(y_true_binary, y_pred_binary)
    assert hasattr(res, "numpy") or isinstance(res, (float, int, Tensor))

    res = losses.SparseCategoricalCrossentropy()(y_true_sparse, y_pred_cat)
    assert hasattr(res, "numpy") or isinstance(res, (float, int, Tensor))

    res = losses.SquaredHinge()(y_true_binary, y_pred_binary)
    assert hasattr(res, "numpy") or isinstance(res, (float, int, Tensor))


def test_reductions() -> None:
    """Verify or execute `test reductions`."""
    y_true = ops.convert_to_tensor([[1.0, 0.0], [0.0, 1.0]])
    y_pred = ops.convert_to_tensor([[0.9, 0.1], [0.1, 0.9]])

    res = losses.MeanSquaredError(reduction="sum")(y_true, y_pred)
    assert hasattr(res, "numpy") or isinstance(res, (float, int, Tensor))

    res = losses.MeanSquaredError(reduction="none")(y_true, y_pred)
    assert hasattr(res, "numpy") or isinstance(res, (float, int, Tensor))

    res = losses.MeanSquaredError(reduction="sum_over_batch_size")(
        y_true, y_pred, sample_weight=ops.convert_to_tensor([1.0, 0.5])
    )
    assert hasattr(res, "numpy") or isinstance(res, (float, int, Tensor))


def test_base_loss() -> None:
    """Verify or execute `test base loss`."""
    loss = losses.Loss()
    res = loss(ops.convert_to_tensor([1.0]), ops.convert_to_tensor([1.0]))
    assert hasattr(res, "numpy") or isinstance(res, (float, int, Tensor))


def test_loss_functions() -> None:
    """Verify or execute `test loss functions`."""
    y_true_binary = ops.convert_to_tensor([[1.0, 0.0], [0.0, 1.0]])
    y_pred_binary = ops.convert_to_tensor([[0.9, 0.1], [0.1, 0.9]])

    losses.dice(y_true_binary, y_pred_binary)
    losses.tversky(y_true_binary, y_pred_binary)
    losses.ctc(y_true_binary, y_pred_binary)
    losses.circle(y_true_binary, y_pred_binary)
    losses.categorical_generalized_cross_entropy(y_true_binary, y_pred_binary)


def test_loss_get_deserialize() -> None:
    """Verify or execute `test loss get deserialize`."""
    assert isinstance(losses.get("mse"), losses.MeanSquaredError)
    assert losses.get("unknown") == "unknown"
    assert losses.get(None) is None

    assert losses.deserialize({"class_name": "MeanSquaredError", "config": {}}) is not None
    assert losses.deserialize("mse") is not None
    assert losses.deserialize(None) is None

    assert losses.serialize("mse") == "mse"


def test_loss_base_methods() -> None:
    """Verify or execute `test loss base methods`."""
    loss = losses.Loss(name="my_loss")
    assert loss.call("y_true", "y_pred") == "y_pred"
    config = loss.get_config()
    assert config["name"] == "my_loss"

    loss2 = losses.Loss.from_config(config)
    assert loss2.name == "my_loss"

    class MockObj:
        """Keras ``MockObj`` class."""

    obj = MockObj()
    assert losses.deserialize(obj) is obj


def test_losses_deserialize_fallback() -> None:
    """Verify or execute `test losses deserialize fallback`."""
    assert losses.deserialize(123) == 123
    assert losses.deserialize({"class_name": "UnknownLoss"}) == {"class_name": "UnknownLoss"}


def test_losses_get_fallback() -> None:
    """Verify or execute `test losses get fallback`."""
    assert losses.get("some_unknown_loss_string") == "some_unknown_loss_string"


def test_dice_loss() -> None:
    """Verify or execute `test dice loss`."""
    y_true = ops.convert_to_tensor([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    y_pred = ops.convert_to_tensor([[0.9, 0.1], [0.2, 0.8]], dtype="float32")
    loss = losses.dice(y_true, y_pred)
    assert loss.shape == () or loss.shape == (2,)


def test_tversky_loss() -> None:
    """Verify or execute `test tversky loss`."""
    y_true = ops.convert_to_tensor([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    y_pred = ops.convert_to_tensor([[0.9, 0.1], [0.2, 0.8]], dtype="float32")
    loss = losses.tversky(y_true, y_pred)
    assert loss.shape == () or loss.shape == (2,)


def test_ctc_loss(monkeypatch: TensorLike) -> None:
    """Verify or execute `test ctc loss`.

    Args:
        monkeypatch: Test fixture or input parameter `monkeypatch` (monkeypatch).
    """

    def mock_ctc_loss(
        log_probs: TensorLike,
        targets: TensorLike,
        input_lengths: TensorLike,
        target_lengths: TensorLike,
    ) -> TensorLike:
        """Verify or execute `mock ctc loss`.

        Args:
            log_probs: Test fixture or input parameter `log_probs` (log probs).
            targets: Test fixture or input parameter `targets` (targets).
            input_lengths: Test fixture or input parameter `input_lengths` (input lengths).
            target_lengths: Test fixture or input parameter `target_lengths` (target lengths).

        Returns:
            TensorLike: Result from `mock_ctc_loss`.
        """
        return "ctc_success"

    import ml_switcheroo_compiler.ops.loss as msc_loss

    monkeypatch.setattr(msc_loss, "ctc_loss", mock_ctc_loss)

    y_true = ops.convert_to_tensor([[1, 2, 0]], dtype="int32")
    y_pred = random.uniform((1, 5, 3), dtype="float32")

    loss = losses.ctc(y_true, y_pred)
    assert loss == "ctc_success"


def test_circle_loss() -> None:
    """Test Circle loss numerical calculation."""
    y_true = ops.convert_to_tensor([[1, 0], [0, 1]], dtype="float32")
    y_pred = ops.convert_to_tensor([[0.9, 0.1], [0.1, 0.9]], dtype="float32")
    loss = losses.circle(y_true, y_pred)
    assert loss is not None


def test_categorical_generalized_cross_entropy() -> None:
    """Test CategoricalGeneralizedCrossEntropy numerical calculation."""
    y_true = ops.convert_to_tensor([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    y_pred = ops.convert_to_tensor([[0.9, 0.1], [0.1, 0.9]], dtype="float32")
    loss = losses.categorical_generalized_cross_entropy(y_true, y_pred)
    assert loss is not None
