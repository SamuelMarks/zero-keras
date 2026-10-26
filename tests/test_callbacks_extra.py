"""Unit and behavioral tests for callbacks extra in zero_keras."""

from __future__ import annotations
from zero_keras.types import TensorLike

import csv
import math
import os
import tempfile

from zero_keras import ops
from zero_keras.callbacks import callbacks


def test_history() -> None:
    """Test ``history`` behavior and edge cases in ``test_callbacks_extra``."""
    hist = callbacks.History()
    hist.on_train_begin()
    hist.on_epoch_end(0, {"loss": 0.5})
    hist.on_epoch_end(1, {"loss": 0.4})
    assert hist.epoch == [0, 1]
    assert hist.history == {"loss": [0.5, 0.4]}


def test_lambda_callback() -> None:
    """Test ``lambda_callback`` behavior and edge cases in ``test_callbacks_extra``."""
    call_counts = {
        "epoch_begin": 0,
        "epoch_end": 0,
        "batch_begin": 0,
        "batch_end": 0,
        "train_begin": 0,
        "train_end": 0,
    }

    def on_epoch_begin(epoch: TensorLike, logs: TensorLike) -> None:
        """Record callback hook invocation for ``on_epoch_begin``.

        Args:
            epoch: Zero-based training epoch index.
            logs: Dictionary of scalar metric and loss results.
        """
        call_counts["epoch_begin"] += 1

    def on_epoch_end(epoch: TensorLike, logs: TensorLike) -> None:
        """Record callback hook invocation for ``on_epoch_end``.

        Args:
            epoch: Zero-based training epoch index.
            logs: Dictionary of scalar metric and loss results.
        """
        call_counts["epoch_end"] += 1

    def on_batch_begin(batch: TensorLike, logs: TensorLike) -> None:
        """Record callback hook invocation for ``on_batch_begin``.

        Args:
            batch: Zero-based batch index within the current epoch.
            logs: Dictionary of scalar metric and loss results.
        """
        call_counts["batch_begin"] += 1

    def on_batch_end(batch: TensorLike, logs: TensorLike) -> None:
        """Record callback hook invocation for ``on_batch_end``.

        Args:
            batch: Zero-based batch index within the current epoch.
            logs: Dictionary of scalar metric and loss results.
        """
        call_counts["batch_end"] += 1

    def on_train_begin(logs: TensorLike) -> None:
        """Record callback hook invocation for ``on_train_begin``.

        Args:
            logs: Dictionary of scalar metric and loss results.
        """
        call_counts["train_begin"] += 1

    def on_train_end(logs: TensorLike) -> None:
        """Record callback hook invocation for ``on_train_end``.

        Args:
            logs: Dictionary of scalar metric and loss results.
        """
        call_counts["train_end"] += 1

    cb = callbacks.LambdaCallback(
        on_epoch_begin=on_epoch_begin,
        on_epoch_end=on_epoch_end,
        on_batch_begin=on_batch_begin,
        on_batch_end=on_batch_end,
        on_train_begin=on_train_begin,
        on_train_end=on_train_end,
    )
    cb.on_epoch_begin(0)
    cb.on_epoch_end(0)
    cb.on_batch_begin(0)
    cb.on_batch_end(0)
    cb.on_train_begin()
    cb.on_train_end()

    assert all(v == 1 for v in call_counts.values())


def test_learning_rate_scheduler() -> None:
    """Test ``learning_rate_scheduler`` behavior and edge cases in ``test_callbacks_extra``."""

    class DummyOptimizer:
        """Test helper class `DummyOptimizer` for callbacks extra."""

        def __init__(self) -> None:
            """Initialize the ``DummyOptimizer`` test helper instance."""
            self.learning_rate = 0.1

    class DummyModel:
        """Test helper class `DummyModel` for callbacks extra."""

        def __init__(self) -> None:
            """Initialize the ``DummyModel`` test helper instance."""
            self.optimizer = DummyOptimizer()

    def schedule(epoch: TensorLike, lr: TensorLike) -> TensorLike:
        """Execute test helper ``schedule`` in ``test_callbacks_extra``.

        Args:
            epoch: Zero-based training epoch index.
            lr: Current scalar learning rate value.

        Returns:
            TensorLike: Result from `schedule`.
        """
        return lr * 0.5

    cb = callbacks.LearningRateScheduler(schedule)
    cb.model = DummyModel()
    cb.on_epoch_begin(0)
    assert cb.model.optimizer.learning_rate == 0.05

    # Test without optimizer
    class DummyNoOptModel:
        """Dummy model without an optimizer attribute."""

    cb2 = callbacks.LearningRateScheduler(schedule)
    cb2.model = DummyNoOptModel()
    cb2.on_epoch_begin(0)


def test_terminate_on_nan() -> None:
    """Test ``terminate_on_nan`` behavior and edge cases in ``test_callbacks_extra``."""

    class DummyModel:
        """Test helper class `DummyModel` for callbacks extra."""

        def __init__(self) -> None:
            """Initialize the ``DummyModel`` test helper instance."""
            self.stop_training = False

    cb = callbacks.TerminateOnNaN()
    cb.model = DummyModel()

    cb.on_batch_end(0, {"loss": 1.0})
    assert not cb.model.stop_training

    cb.on_batch_end(1, {"loss": math.nan})
    assert cb.model.stop_training


def test_csv_logger() -> None:
    """Test ``csv_logger`` behavior and edge cases in ``test_callbacks_extra``."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
        filepath = f.name

    cb = callbacks.CSVLogger(filepath)
    cb.on_train_begin()
    cb.on_epoch_end(0, {"loss": 0.5, "val_loss": ops.convert_to_tensor(0.4), "arr": [1, 2]})
    cb.on_epoch_end(1, {"loss": 0.4})
    cb.on_train_end()

    with open(filepath, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) == 3

    # Append
    cb = callbacks.CSVLogger(filepath, append=True)
    cb.on_train_begin()
    cb.on_epoch_end(2, {"loss": 0.3})
    cb.on_train_end()

    with open(filepath, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) == 4

    os.remove(filepath)


def test_reduce_lr_on_plateau() -> None:
    """Test ``reduce_lr_on_plateau`` behavior and edge cases in ``test_callbacks_extra``."""

    class DummyOptimizer:
        """Test helper class `DummyOptimizer` for callbacks extra."""

        def __init__(self) -> None:
            """Initialize the ``DummyOptimizer`` test helper instance."""
            self.learning_rate = 0.1

    class DummyModel:
        """Test helper class `DummyModel` for callbacks extra."""

        def __init__(self) -> None:
            """Initialize the ``DummyModel`` test helper instance."""
            self.optimizer = DummyOptimizer()

    cb = callbacks.ReduceLROnPlateau(patience=2, factor=0.5, cooldown=1)
    cb.model = DummyModel()
    cb.on_train_begin()

    # Epoch 1: val_loss=1.0 (best)
    cb.on_epoch_end(0, {"val_loss": 1.0})
    assert cb.model.optimizer.learning_rate == 0.1

    # Epoch 2: val_loss=1.0 (wait=1)
    cb.on_epoch_end(1, {"val_loss": 1.0})
    assert cb.model.optimizer.learning_rate == 0.1

    # Epoch 3: val_loss=1.0 (wait=2 -> reduce LR)
    cb.on_epoch_end(2, {"val_loss": 1.0})
    assert cb.model.optimizer.learning_rate == 0.05
    assert cb.in_cooldown()

    # Epoch 4: in cooldown
    cb.on_epoch_end(3, {"val_loss": 1.0})
    assert not cb.in_cooldown()


def test_backup_and_restore() -> None:
    """Test ``backup_and_restore`` behavior and edge cases in ``test_callbacks_extra``."""
    with tempfile.TemporaryDirectory() as d:
        cb = callbacks.BackupAndRestore(d)
        cb.on_train_begin()
        cb.on_epoch_end(0)
        cb.on_train_end()
        assert not os.path.exists(d)


def test_remote_monitor() -> None:
    """Test ``remote_monitor`` behavior and edge cases in ``test_callbacks_extra``."""
    cb = callbacks.RemoteMonitor()
    assert cb.root == "http://localhost:9000"


def test_swap_ema_weights() -> None:
    """Test ``swap_ema_weights`` behavior and edge cases in ``test_callbacks_extra``."""
    cb = callbacks.SwapEMAWeights()
    assert not cb.swap_on_epoch


def test_tensorboard() -> None:
    """Test ``tensorboard`` behavior and edge cases in ``test_callbacks_extra``."""
    cb = callbacks.TensorBoard()
    assert cb.log_dir == "logs"


def test_learning_rate_scheduler_missing_lr() -> None:
    """Test ``learning_rate_scheduler_missing_lr`` behavior and edge cases in ``test_callbacks_extra``."""

    class DummyOptimizerMissingLR:
        """Test helper class `DummyOptimizerMissingLR` for callbacks extra."""

    class DummyModelMissingLR:
        """Test helper class `DummyModelMissingLR` for callbacks extra."""

        def __init__(self) -> None:
            """Initialize the ``DummyModelMissingLR`` test helper instance."""
            self.optimizer = DummyOptimizerMissingLR()

    cb = callbacks.LearningRateScheduler(lambda e, lr: lr)
    cb.model = DummyModelMissingLR()
    cb.on_epoch_begin(0)  # Should hit the second return


def test_learning_rate_scheduler_type_error() -> None:
    """Test ``learning_rate_scheduler_type_error`` behavior and edge cases in ``test_callbacks_extra``."""

    class TypeErrorFloat:
        """Test helper class `TypeErrorFloat` for callbacks extra."""

        def __float__(self) -> TensorLike:
            """Convert ``TypeErrorFloat`` to a Python ``float``."""
            raise TypeError()

        def __mul__(self, other: TensorLike) -> TensorLike:
            """Return the elementwise product for ``TypeErrorFloat``.

            Args:
                other: Right-hand operand for binary arithmetic.

            Returns:
                TensorLike: Result from `__mul__`.
            """
            return 0.05

    class DummyOptimizerTypeError:
        """Test helper class `DummyOptimizerTypeError` for callbacks extra."""

        def __init__(self) -> None:
            """Initialize the ``DummyOptimizerTypeError`` test helper instance."""
            self.learning_rate = TypeErrorFloat()

    class DummyModelTypeError:
        """Test helper class `DummyModelTypeError` for callbacks extra."""

        def __init__(self) -> None:
            """Initialize the ``DummyModelTypeError`` test helper instance."""
            self.optimizer = DummyOptimizerTypeError()

    cb = callbacks.LearningRateScheduler(lambda e, lr: lr * 0.5)
    cb.model = DummyModelTypeError()
    cb.on_epoch_begin(0)
    assert cb.model.optimizer.learning_rate == 0.05


def test_csv_logger_append_new_file() -> None:
    """Test ``csv_logger_append_new_file`` behavior and edge cases in ``test_callbacks_extra``."""
    with tempfile.NamedTemporaryFile(delete=True) as f:
        filepath = f.name
    # File doesn't exist, append=True
    cb = callbacks.CSVLogger(filepath, append=True)
    cb.on_train_begin()
    cb.on_epoch_end(0, {"loss": "str_loss"})
    cb.on_train_end()
    assert os.path.exists(filepath)
    os.remove(filepath)


def test_reduce_lr_on_plateau_max_mode() -> None:
    """Test ``reduce_lr_on_plateau_max_mode`` behavior and edge cases in ``test_callbacks_extra``."""

    class DummyOptimizer:
        """Test helper class `DummyOptimizer` for callbacks extra."""

        def __init__(self) -> None:
            """Initialize the ``DummyOptimizer`` test helper instance."""
            self.learning_rate = 0.1

    class DummyModel:
        """Test helper class `DummyModel` for callbacks extra."""

        def __init__(self) -> None:
            """Initialize the ``DummyModel`` test helper instance."""
            self.optimizer = DummyOptimizer()

    cb = callbacks.ReduceLROnPlateau(mode="max", patience=1, factor=0.5)
    cb.model = DummyModel()
    cb.on_train_begin()
    # current is None
    cb.on_epoch_end(0, {})
    # Epoch 1: val_loss=1.0 (best)
    cb.on_epoch_end(0, {"val_loss": 1.0})
    # Epoch 2: val_loss=1.0 (not better than 1.0 + min_delta)
    cb.on_epoch_end(1, {"val_loss": 1.0})
    assert cb.model.optimizer.learning_rate == 0.05


def test_backup_and_restore_create_dir() -> None:
    """Test ``backup_and_restore_create_dir`` behavior and edge cases in ``test_callbacks_extra``."""
    with tempfile.TemporaryDirectory() as d:
        target_dir = os.path.join(d, "new_sub_dir")
        cb = callbacks.BackupAndRestore(target_dir)
        cb.on_train_begin()
        assert os.path.exists(target_dir)
        cb.on_train_end()


def test_callback_methods_extra() -> None:
    """Test ``callback_methods_extra`` behavior and edge cases in ``test_callbacks_extra``."""
    from zero_keras.callbacks.callbacks import (
        Callback,
        CallbackList,
        EarlyStopping,
    )

    c = Callback()
    c.model = None
    assert c.model is None
    c.on_predict_batch_begin(0)
    c.on_predict_batch_end(0)
    c.on_predict_begin()
    c.on_predict_end()
    c.on_test_batch_begin(0)
    c.on_test_batch_end(0)
    c.on_test_begin()
    c.on_test_end()
    c.on_train_batch_begin(0)
    c.on_train_batch_end(0)

    cl = CallbackList()
    cl.set_model("model")
    assert cl.model == "model"
    cl.set_params({})
    assert hasattr(cl, "params")

    es = EarlyStopping(monitor="val_loss")
    assert es.get_monitor_value({"val_loss": 0.5}) == 0.5


def test_lambda_callback_more() -> None:
    """Test ``lambda_callback_more`` behavior and edge cases in ``test_callbacks_extra``."""
    from zero_keras.callbacks import LambdaCallback

    cb = LambdaCallback(
        on_epoch_begin=lambda epoch, logs: None,
        on_epoch_end=lambda epoch, logs: None,
        on_batch_begin=lambda batch, logs: None,
        on_batch_end=lambda batch, logs: None,
        on_train_begin=lambda logs: None,
        on_train_end=lambda logs: None,
    )
    cb.on_epoch_begin(0, {})
    cb.on_epoch_end(0, {})
    cb.on_batch_begin(0, {})
    cb.on_batch_end(0, {})
    cb.on_train_begin({})
    cb.on_train_end({})

    cb_none = LambdaCallback()
    cb_none.on_epoch_begin(0, {})
    cb_none.on_epoch_end(0, {})
    cb_none.on_batch_begin(0, {})
    cb_none.on_batch_end(0, {})
    cb_none.on_train_begin({})
    cb_none.on_train_end({})


def test_callback_list_coverage() -> None:
    """Test ``callback_list_coverage`` behavior and edge cases in ``test_callbacks_extra``."""
    from zero_keras.callbacks import Callback, CallbackList

    class MyCB(Callback):
        """Keras ``MyCB`` class."""

        def on_train_batch_begin(self, batch: TensorLike, logs: TensorLike | None = None) -> None:
            """Handle the ``on_train_batch_begin`` lifecycle event on ``MyCB``.

            Args:
                batch: Zero-based batch index within the current epoch.
                logs: Dictionary of scalar metric and loss results.
            """
            pass

        def on_train_batch_end(self, batch: TensorLike, logs: TensorLike | None = None) -> None:
            """Handle the ``on_train_batch_end`` lifecycle event on ``MyCB``.

            Args:
                batch: Zero-based batch index within the current epoch.
                logs: Dictionary of scalar metric and loss results.
            """
            pass

        def on_test_batch_begin(self, batch: TensorLike, logs: TensorLike | None = None) -> None:
            """Handle the ``on_test_batch_begin`` lifecycle event on ``MyCB``.

            Args:
                batch: Zero-based batch index within the current epoch.
                logs: Dictionary of scalar metric and loss results.
            """
            pass

        def on_test_batch_end(self, batch: TensorLike, logs: TensorLike | None = None) -> None:
            """Handle the ``on_test_batch_end`` lifecycle event on ``MyCB``.

            Args:
                batch: Zero-based batch index within the current epoch.
                logs: Dictionary of scalar metric and loss results.
            """
            pass

        def on_predict_batch_begin(self, batch: TensorLike, logs: TensorLike | None = None) -> None:
            """Handle the ``on_predict_batch_begin`` lifecycle event on ``MyCB``.

            Args:
                batch: Zero-based batch index within the current epoch.
                logs: Dictionary of scalar metric and loss results.
            """
            pass

        def on_predict_batch_end(self, batch: TensorLike, logs: TensorLike | None = None) -> None:
            """Handle the ``on_predict_batch_end`` lifecycle event on ``MyCB``.

            Args:
                batch: Zero-based batch index within the current epoch.
                logs: Dictionary of scalar metric and loss results.
            """
            pass

        def on_epoch_end(self, epoch: TensorLike, logs: TensorLike | None = None) -> None:
            """Handle the ``on_epoch_end`` lifecycle event on ``MyCB``.

            Args:
                epoch: Zero-based training epoch index.
                logs: Dictionary of scalar metric and loss results.
            """
            pass

        def on_train_end(self, logs: TensorLike | None = None) -> None:
            """Handle the ``on_train_end`` lifecycle event on ``MyCB``.

            Args:
                logs: Dictionary of scalar metric and loss results.
            """
            pass

        def on_test_begin(self, logs: TensorLike | None = None) -> None:
            """Handle the ``on_test_begin`` lifecycle event on ``MyCB``.

            Args:
                logs: Dictionary of scalar metric and loss results.
            """
            pass

        def on_test_end(self, logs: TensorLike | None = None) -> None:
            """Handle the ``on_test_end`` lifecycle event on ``MyCB``.

            Args:
                logs: Dictionary of scalar metric and loss results.
            """
            pass

        def on_predict_begin(self, logs: TensorLike | None = None) -> None:
            """Handle the ``on_predict_begin`` lifecycle event on ``MyCB``.

            Args:
                logs: Dictionary of scalar metric and loss results.
            """
            pass

        def on_predict_end(self, logs: TensorLike | None = None) -> None:
            """Handle the ``on_predict_end`` lifecycle event on ``MyCB``.

            Args:
                logs: Dictionary of scalar metric and loss results.
            """
            pass

        def on_batch_begin(self, batch: TensorLike, logs: TensorLike | None = None) -> None:
            """Handle the ``on_batch_begin`` lifecycle event on ``MyCB``.

            Args:
                batch: Zero-based batch index within the current epoch.
                logs: Dictionary of scalar metric and loss results.
            """
            pass

        def on_batch_end(self, batch: TensorLike, logs: TensorLike | None = None) -> None:
            """Handle the ``on_batch_end`` lifecycle event on ``MyCB``.

            Args:
                batch: Zero-based batch index within the current epoch.
                logs: Dictionary of scalar metric and loss results.
            """
            pass

    cbl = CallbackList([MyCB()], model=None)
    cbl.on_train_begin()
    cbl.on_epoch_begin(0)
    cbl.on_train_batch_begin(0)
    cbl.on_train_batch_end(0)
    cbl.on_test_batch_begin(0)
    cbl.on_test_batch_end(0)
    cbl.on_predict_batch_begin(0)
    cbl.on_predict_batch_end(0)
    cbl.on_epoch_end(0)
    cbl.on_train_end()
    cbl.on_test_begin()
    cbl.on_test_end()
    cbl.on_predict_begin()
    cbl.on_predict_end()

    # also cover the missing fallback batch calls
    cbl.on_batch_begin(0)
    cbl.on_batch_end(0)


def test_callback_list_coverage_more() -> None:
    """Test ``callback_list_coverage_more`` behavior and edge cases in ``test_callbacks_extra``."""
    from zero_keras.callbacks import Callback, CallbackList

    class MyEmptyCB(Callback):
        """Keras ``MyEmptyCB`` class."""

        pass

    cbl = CallbackList([MyEmptyCB()], model=None)
    cbl.on_train_batch_begin(0)
    cbl.on_train_batch_end(0)
    cbl.on_test_batch_begin(0)
    cbl.on_test_batch_end(0)
    cbl.on_predict_batch_begin(0)
    cbl.on_predict_batch_end(0)
    cbl.on_test_begin()
    cbl.on_test_end()
    cbl.on_predict_begin()
    cbl.on_predict_end()
