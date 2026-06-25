"""Module docstring."""

import os
import csv
import tempfile
import numpy as np
from zero_keras.callbacks import callbacks


def test_history():
    """Function docstring."""
    hist = callbacks.History()
    hist.on_train_begin()
    hist.on_epoch_end(0, {"loss": 0.5})
    hist.on_epoch_end(1, {"loss": 0.4})
    assert hist.epoch == [0, 1]
    assert hist.history == {"loss": [0.5, 0.4]}


def test_lambda_callback():
    """Function docstring."""
    call_counts = {
        "epoch_begin": 0,
        "epoch_end": 0,
        "batch_begin": 0,
        "batch_end": 0,
        "train_begin": 0,
        "train_end": 0,
    }

    def on_epoch_begin(epoch, logs):
        """Function docstring.

        Args:
            epoch: Description.
            logs: Description.
        """
        call_counts["epoch_begin"] += 1

    def on_epoch_end(epoch, logs):
        """Function docstring.

        Args:
            epoch: Description.
            logs: Description.
        """
        call_counts["epoch_end"] += 1

    def on_batch_begin(batch, logs):
        """Function docstring.

        Args:
            batch: Description.
            logs: Description.
        """
        call_counts["batch_begin"] += 1

    def on_batch_end(batch, logs):
        """Function docstring.

        Args:
            batch: Description.
            logs: Description.
        """
        call_counts["batch_end"] += 1

    def on_train_begin(logs):
        """Function docstring.

        Args:
            logs: Description.
        """
        call_counts["train_begin"] += 1

    def on_train_end(logs):
        """Function docstring.

        Args:
            logs: Description.
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


def test_learning_rate_scheduler():
    """Function docstring."""

    class DummyOptimizer:
        """Class docstring."""

        def __init__(self):
            """Function docstring."""
            self.learning_rate = 0.1

    class DummyModel:
        """Class docstring."""

        def __init__(self):
            """Function docstring."""
            self.optimizer = DummyOptimizer()

    def schedule(epoch, lr):
        """Function docstring.

        Args:
            epoch: Description.
            lr: Description.
        """
        return lr * 0.5

    cb = callbacks.LearningRateScheduler(schedule)
    cb.model = DummyModel()
    cb.on_epoch_begin(0)
    assert cb.model.optimizer.learning_rate == 0.05

    # Test without optimizer
    cb2 = callbacks.LearningRateScheduler(schedule)
    cb2.model = object()
    cb2.on_epoch_begin(0)


def test_terminate_on_nan():
    """Function docstring."""

    class DummyModel:
        """Class docstring."""

        def __init__(self):
            """Function docstring."""
            self.stop_training = False

    cb = callbacks.TerminateOnNaN()
    cb.model = DummyModel()

    cb.on_batch_end(0, {"loss": 1.0})
    assert not cb.model.stop_training

    cb.on_batch_end(1, {"loss": np.nan})
    assert cb.model.stop_training


def test_csv_logger():
    """Function docstring."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
        filepath = f.name

    cb = callbacks.CSVLogger(filepath)
    cb.on_train_begin()
    cb.on_epoch_end(0, {"loss": 0.5, "val_loss": np.array(0.4), "arr": [1, 2]})
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


def test_reduce_lr_on_plateau():
    """Function docstring."""

    class DummyOptimizer:
        """Class docstring."""

        def __init__(self):
            """Function docstring."""
            self.learning_rate = 0.1

    class DummyModel:
        """Class docstring."""

        def __init__(self):
            """Function docstring."""
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


def test_backup_and_restore():
    """Function docstring."""
    with tempfile.TemporaryDirectory() as d:
        cb = callbacks.BackupAndRestore(d)
        cb.on_train_begin()
        cb.on_epoch_end(0)
        cb.on_train_end()
        assert not os.path.exists(d)


def test_remote_monitor():
    """Function docstring."""
    cb = callbacks.RemoteMonitor()
    assert cb.root == "http://localhost:9000"


def test_swap_ema_weights():
    """Function docstring."""
    cb = callbacks.SwapEMAWeights()
    assert not cb.swap_on_epoch


def test_tensorboard():
    """Function docstring."""
    cb = callbacks.TensorBoard()
    assert cb.log_dir == "logs"


def test_learning_rate_scheduler_missing_lr():
    """Function docstring."""

    class DummyOptimizerMissingLR:
        """Class docstring."""

        pass

    class DummyModelMissingLR:
        """Class docstring."""

        def __init__(self):
            """Function docstring."""
            self.optimizer = DummyOptimizerMissingLR()

    cb = callbacks.LearningRateScheduler(lambda e, lr: lr)
    cb.model = DummyModelMissingLR()
    cb.on_epoch_begin(0)  # Should hit the second return


def test_learning_rate_scheduler_type_error():
    """Function docstring."""

    class TypeErrorFloat:
        """Class docstring."""

        def __float__(self):
            """Function docstring."""
            raise TypeError()

        def __mul__(self, other):
            """Function docstring.

            Args:
                other: Description.
            """
            return 0.05

    class DummyOptimizerTypeError:
        """Class docstring."""

        def __init__(self):
            """Function docstring."""
            self.learning_rate = TypeErrorFloat()

    class DummyModelTypeError:
        """Class docstring."""

        def __init__(self):
            """Function docstring."""
            self.optimizer = DummyOptimizerTypeError()

    cb = callbacks.LearningRateScheduler(lambda e, lr: lr * 0.5)
    cb.model = DummyModelTypeError()
    cb.on_epoch_begin(0)
    assert cb.model.optimizer.learning_rate == 0.05


def test_csv_logger_append_new_file():
    """Function docstring."""
    with tempfile.NamedTemporaryFile(delete=True) as f:
        filepath = f.name
    # File doesn't exist, append=True
    cb = callbacks.CSVLogger(filepath, append=True)
    cb.on_train_begin()
    cb.on_epoch_end(0, {"loss": "str_loss"})
    cb.on_train_end()
    assert os.path.exists(filepath)
    os.remove(filepath)


def test_reduce_lr_on_plateau_max_mode():
    """Function docstring."""

    class DummyOptimizer:
        """Class docstring."""

        def __init__(self):
            """Function docstring."""
            self.learning_rate = 0.1

    class DummyModel:
        """Class docstring."""

        def __init__(self):
            """Function docstring."""
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


def test_backup_and_restore_create_dir():
    """Function docstring."""
    with tempfile.TemporaryDirectory() as d:
        target_dir = os.path.join(d, "new_sub_dir")
        cb = callbacks.BackupAndRestore(target_dir)
        cb.on_train_begin()
        assert os.path.exists(target_dir)
        cb.on_train_end()


def test_callback_methods_extra():
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
