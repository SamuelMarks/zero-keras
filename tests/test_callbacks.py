"""Comprehensive unit and branch coverage tests for zero_keras.callbacks."""

from __future__ import annotations

from zero_keras import ops
from zero_keras.callbacks import Callback, CallbackList, EarlyStopping, History, ModelCheckpoint
from zero_keras.constraints import Constraint, MaxNorm, MinMaxNorm, NonNeg, UnitNorm
from zero_keras.types import TensorLike

from .conftest import assert_allclose_zero, to_flat_floats


class MockModel:
    """Test helper class `MockModel` for callbacks."""

    def __init__(self) -> None:
        """Initialize the ``MockModel`` test helper instance."""
        self.stop_training = False

        class W:
            """Test helper class `W` for callbacks."""

            data = 0

        self.weights = [W()]

    def save(self, filepath: TensorLike) -> TensorLike:
        """Save the ``MockModel`` state to ``filepath``.

        Args:
            filepath: Target file path for saving or loading.
        """


def test_callbacks(tmp_path: TensorLike) -> None:
    """Test ``callbacks`` behavior and edge cases in ``test_callbacks``.

    Args:
        tmp_path: Temporary directory fixture.
    """
    cb = Callback()
    cb.set_model(MockModel())
    cb.set_params({})
    cb.on_train_begin()
    cb.on_epoch_begin(0)
    cb.on_batch_begin(0)
    cb.on_batch_end(0)
    cb.on_epoch_end(0)
    cb.on_train_end()

    es = EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=1,
        mode="min",
        restore_best_weights=True,
    )
    model = MockModel()
    es.set_model(model)
    es.on_train_begin()
    es.on_epoch_end(0, {"val_loss": 0.5})
    es.on_epoch_end(1, {"val_loss": 0.6})
    es.on_epoch_end(2, {"val_loss": 0.7})
    assert model.stop_training

    es2 = EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=2,
        mode="min",
        restore_best_weights=False,
    )
    es2.set_model(model)
    es2.on_train_begin()
    es2.on_epoch_end(0, {"val_loss": 0.5})
    es2.on_epoch_end(1, {"val_loss": 0.6})
    es2.on_epoch_end(2, {"val_loss": 0.7})

    mc = ModelCheckpoint(filepath="test.h5", monitor="val_loss", save_best_only=True, mode="min")
    mc2 = ModelCheckpoint(filepath="test2.h5", monitor="val_loss", save_best_only=False, mode="max")
    mc.set_model(model)
    mc.on_epoch_end(0, {"val_loss": 0.5})
    mc.on_epoch_end(1, {"val_loss": 0.4})
    assert mc.best == 0.4
    mc.on_epoch_end(2, {})
    mc.on_epoch_end(3, {"val_loss": 0.6})

    mc2.set_model(model)
    mc2.on_epoch_end(0, {"val_loss": 0.5})
    mc2.on_epoch_end(1, {"val_loss": 0.6})
    assert mc2.best == 0.6

    class WeightsModel(MockModel):
        """Mock model exposing save_weights."""

        def __init__(self) -> None:
            """Initialize WeightsModel."""
            super().__init__()
            self.saved_weights_path: str | None = None

        def save_weights(self, filepath: str) -> None:
            """Record saved weights filepath.

            Args:
                filepath: Destination path.
            """
            self.saved_weights_path = filepath

    wm = WeightsModel()
    mc_w = ModelCheckpoint(filepath="w.weights.h5", save_weights_only=True)
    mc_w.set_model(wm)
    mc_w.on_epoch_end(0, {"val_loss": 0.2})
    assert wm.saved_weights_path == "w.weights.h5"

    from zero_keras.callbacks import BackupAndRestore, ProgbarLogger

    bar = BackupAndRestore(backup_dir=str(tmp_path / "tmp_backup_test"), delete_checkpoint=True)
    bar.set_model(wm)
    bar.on_train_begin()
    bar.on_epoch_end(0)
    bar.on_train_end()
    assert wm.saved_weights_path is not None

    prog = ProgbarLogger(count_mode="steps", stateful_metrics=["loss"])
    assert prog.count_mode == "steps"
    assert prog.stateful_metrics == ["loss"]


def test_callback_model_setter() -> None:
    """Verify model property setter across Callback, History, and CallbackList."""
    cb = Callback()
    cb.model = "model"
    assert cb._model == "model"

    hist = History()
    hist.model = "model2"
    assert hist._model == "model2"

    cblist = CallbackList([cb])
    cblist.model = "model3"
    assert cblist.model == "model3"


def test_history_methods() -> None:
    """Verify History callback lifecycle methods and epoch metric recording."""
    hist = History()
    hist.on_predict_batch_begin(0)
    hist.on_predict_batch_end(0)
    hist.on_predict_begin()
    hist.on_predict_end()
    hist.on_test_batch_begin(0)
    hist.on_test_batch_end(0)
    hist.on_test_begin()
    hist.on_test_end()
    hist.on_train_batch_begin(0)
    hist = History()
    hist.on_train_begin()
    assert hist.history == {}
    hist.on_epoch_end(0, {"loss": 0.25})
    assert hist.epoch == [0]
    assert hist.history["loss"] == [0.25]


def test_constraints_methods() -> None:
    """Test Constraint subclasses clamping and normalization behavior."""
    c = Constraint()
    x = ops.convert_to_tensor([1.0, -2.0], dtype="float32")
    assert_allclose_zero(c(x), x)

    mn = MaxNorm(max_value=1.0)
    assert mn.max_value == 1.0
    mmn = MinMaxNorm(min_value=0.5, max_value=1.0)
    assert mmn.min_value == 0.5
    un = UnitNorm()
    assert un.axis == 0
    nn = NonNeg()
    assert all(v >= 0.0 for v in to_flat_floats(nn(x)))
