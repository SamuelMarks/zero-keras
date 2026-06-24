"""Module docstring."""

import numpy as np
from zero_keras import losses


def test_losses_deserialize_fallback():
    """Function docstring."""
    assert losses.deserialize(123) == 123
    assert losses.deserialize({"class_name": "UnknownLoss"}) == {
        "class_name": "UnknownLoss"
    }


def test_losses_get_fallback():
    """Function docstring."""
    assert losses.get("some_unknown_loss_string") == "some_unknown_loss_string"


def test_dice_loss():
    """Function docstring."""
    y_true = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    y_pred = np.array([[0.9, 0.1], [0.2, 0.8]], dtype="float32")
    loss = losses.dice(y_true, y_pred)
    assert loss.shape == () or loss.shape == (2,)


def test_tversky_loss():
    """Function docstring."""
    y_true = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    y_pred = np.array([[0.9, 0.1], [0.2, 0.8]], dtype="float32")
    loss = losses.tversky(y_true, y_pred)
    assert loss.shape == () or loss.shape == (2,)


def test_ctc_loss(monkeypatch):
    """Function docstring.

    Args:
        monkeypatch: Description.
    """
    import ml_switcheroo_compiler.ops as compiler_ops

    def mock_ctc_loss(log_probs, targets, input_lengths, target_lengths):
        """Function docstring.

        Args:
            log_probs: Description.
            targets: Description.
            input_lengths: Description.
            target_lengths: Description.
        """
        return "ctc_success"

    monkeypatch.setattr(compiler_ops, "ctc_loss", mock_ctc_loss)

    y_true = np.array([[1, 2, 0]], dtype="int32")
    y_pred = np.random.uniform(size=(1, 5, 3)).astype("float32")

    loss = losses.ctc(y_true, y_pred)
    assert loss == "ctc_success"


def test_circle_loss():
    """Function docstring."""
    y_true = np.array([[1, 0], [0, 1]], dtype="float32")
    y_pred = np.array([[0.9, 0.1], [0.1, 0.9]], dtype="float32")
    loss = losses.circle(y_true, y_pred)


def test_categorical_generalized_cross_entropy():
    """Function docstring."""
    # categorical generalized cross entropy uses basic math ops that DO support eager mode
    y_true = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    y_pred = np.array([[0.9, 0.1], [0.1, 0.9]], dtype="float32")
    loss = losses.categorical_generalized_cross_entropy(y_true, y_pred)
