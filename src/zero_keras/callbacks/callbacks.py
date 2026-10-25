"""Training, evaluation, and prediction lifecycle callbacks for zero_keras."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import csv
import math
import os
import shutil
from typing import Protocol, runtime_checkable

from zero_keras.types import (
    ConfigValue,
    EpochLogCallbackProtocol,
    ScheduleFnProtocol,
    TensorLike,
)


@runtime_checkable
class _HasNdimProtocol(Protocol):
    """Protocol for array-like objects exposing an ``ndim`` attribute."""

    ndim: int


@runtime_checkable
class _HasOptimizerProtocol(Protocol):
    """Protocol for model objects exposing an ``optimizer`` attribute."""

    optimizer: TensorLike


@runtime_checkable
class _HasLearningRateProtocol(Protocol):
    """Protocol for optimizer objects exposing a ``learning_rate`` attribute."""

    learning_rate: TensorLike


@runtime_checkable
class _HasSaveWeightsProtocol(Protocol):
    """Protocol for model objects exposing ``save_weights``."""

    def save_weights(self, filepath: str) -> None:
        """Save model weights to ``filepath``.

        Args:
            filepath: Target file path.
        """


@runtime_checkable
class _HasLoadWeightsProtocol(Protocol):
    """Protocol for model objects exposing ``load_weights``."""

    def load_weights(self, filepath: str) -> None:
        """Load model weights from ``filepath``.

        Args:
            filepath: Source file path.
        """


class Callback:
    """Base class used to build new callbacks.

    Callbacks can be passed to keras methods such as ``fit()``, ``evaluate()``, and
    ``predict()`` in order to hook into the various stages of the model training,
    evaluation, and inference lifecycle.
    """

    def __init__(self) -> None:
        """Initialize the base Callback instance."""
        self._model: TensorLike | None = None
        self.params: dict[str, ConfigValue] | TensorLike = {}

    @property
    def model(self) -> TensorLike:
        """Return the reference to the Keras model being trained or evaluated.

        Returns:
            TensorLike: Attached model instance or ``None``.
        """
        return self._model

    @model.setter
    def model(self, value: TensorLike) -> None:
        """Attach a Keras model instance to this callback.

        Args:
            value: Model instance to attach.
        """
        self._model = value

    def on_predict_batch_begin(self, batch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Called at the beginning of a batch in ``predict`` methods.

        Args:
            batch: Integer index of batch within the current prediction run.
            logs: Optional metric/log dictionary for the current batch.
        """
        _ = batch, logs

    def on_predict_batch_end(self, batch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Called at the end of a batch in ``predict`` methods.

        Args:
            batch: Integer index of batch within the current prediction run.
            logs: Optional metric/log dictionary for the current batch.
        """
        _ = batch, logs

    def on_predict_begin(self, logs: dict[str, TensorLike] | None = None) -> None:
        """Called at the beginning of prediction.

        Args:
            logs: Optional log dictionary at prediction start.
        """
        _ = logs

    def on_predict_end(self, logs: dict[str, TensorLike] | None = None) -> None:
        """Called at the end of prediction.

        Args:
            logs: Optional log dictionary at prediction completion.
        """
        _ = logs

    def on_test_batch_begin(self, batch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Called at the beginning of a batch in ``evaluate`` methods.

        Args:
            batch: Integer index of batch within the current evaluation run.
            logs: Optional metric/log dictionary for the current batch.
        """
        _ = batch, logs

    def on_test_batch_end(self, batch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Called at the end of a batch in ``evaluate`` methods.

        Args:
            batch: Integer index of batch within the current evaluation run.
            logs: Optional metric/log dictionary for the current batch.
        """
        _ = batch, logs

    def on_test_begin(self, logs: dict[str, TensorLike] | None = None) -> None:
        """Called at the beginning of evaluation or validation.

        Args:
            logs: Optional log dictionary at evaluation start.
        """
        _ = logs

    def on_test_end(self, logs: dict[str, TensorLike] | None = None) -> None:
        """Called at the end of evaluation or validation.

        Args:
            logs: Optional log dictionary at evaluation completion.
        """
        _ = logs

    def on_train_batch_begin(self, batch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Called at the beginning of a training batch in ``fit`` methods.

        Args:
            batch: Integer index of batch within the current epoch.
            logs: Optional metric/log dictionary for the current batch.
        """
        _ = batch, logs

    def on_train_batch_end(self, batch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Called at the end of a training batch in ``fit`` methods.

        Args:
            batch: Integer index of batch within the current epoch.
            logs: Optional metric/log dictionary for the current batch.
        """
        _ = batch, logs

    def set_model(self, model: TensorLike) -> None:
        """Set the model reference tracked by this callback.

        Args:
            model: Model instance being trained or evaluated.
        """
        self.model = model

    def set_params(self, params: dict[str, ConfigValue] | TensorLike) -> None:
        """Set training configuration parameters on this callback.

        Args:
            params: Dictionary of training parameters (e.g., ``epochs``, ``steps``).
        """
        self.params = params

    def on_epoch_begin(self, epoch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Called at the start of an epoch during training.

        Args:
            epoch: Integer index of the starting epoch.
            logs: Optional dictionary of logs.
        """
        _ = epoch, logs

    def on_epoch_end(self, epoch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Called at the end of an epoch during training.

        Args:
            epoch: Integer index of the completed epoch.
            logs: Dictionary of metric results for this training/validation epoch.
        """
        _ = epoch, logs

    def on_batch_begin(self, batch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Backwards-compatible alias for ``on_train_batch_begin``.

        Args:
            batch: Integer index of the starting batch.
            logs: Optional dictionary of batch logs.
        """
        _ = batch, logs

    def on_batch_end(self, batch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Backwards-compatible alias for ``on_train_batch_end``.

        Args:
            batch: Integer index of the completed batch.
            logs: Optional dictionary of batch logs.
        """
        _ = batch, logs

    def on_train_begin(self, logs: dict[str, TensorLike] | None = None) -> None:
        """Called at the beginning of model training.

        Args:
            logs: Optional dictionary of initial training logs.
        """
        _ = logs

    def on_train_end(self, logs: dict[str, TensorLike] | None = None) -> None:
        """Called at the end of model training.

        Args:
            logs: Optional dictionary of final training logs.
        """
        _ = logs


class EarlyStopping(Callback):
    """Stop training when a monitored metric has stopped improving."""

    def get_monitor_value(self, logs: dict[str, TensorLike]) -> TensorLike:
        """Retrieve the monitored metric value from ``logs``.

        Args:
            logs: Epoch metric dictionary.

        Returns:
            TensorLike: Monitored metric value or ``None``.
        """
        return logs.get(self.monitor)

    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 0.0,
        patience: int = 0,
        verbose: int = 0,
        mode: str = "auto",
        baseline: float | None = None,
        restore_best_weights: bool = False,
        start_from_epoch: int = 0,
    ) -> None:
        """Initialize the EarlyStopping callback.

        Args:
            monitor: Quantity name to monitor in ``logs``.
            min_delta: Minimum change in monitored quantity to qualify as improvement.
            patience: Number of epochs with no improvement before stopping training.
            verbose: Verbosity mode (0 or 1).
            mode: One of ``{"auto", "min", "max"}``.
            baseline: Optional baseline value that the monitored quantity must beat.
            restore_best_weights: Whether to restore model weights from the best epoch.
            start_from_epoch: Number of warm-up epochs before monitoring begins.
        """
        super().__init__()
        self.monitor = str(monitor)
        self.min_delta = float(min_delta)
        self.patience = int(patience)
        self.verbose = int(verbose)
        self.mode = str(mode)
        self.baseline = baseline
        self.restore_best_weights = bool(restore_best_weights)
        self.start_from_epoch = int(start_from_epoch)
        self.wait = 0
        self.stopped_epoch = 0
        self.best = (
            float("inf")
            if self.mode == "min" or (self.mode == "auto" and "acc" not in self.monitor)
            else -float("inf")
        )
        self.best_weights: list[TensorLike] | None = None

    def on_train_begin(self, logs: dict[str, TensorLike] | None = None) -> None:
        """Reset wait counters at the beginning of training.

        Args:
            logs: Optional dictionary of initial training logs.
        """
        _ = logs
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Check monitored metric at epoch end and stop training if patience is exceeded.

        Args:
            epoch: Integer index of the completed epoch.
            logs: Dictionary of epoch metric results.
        """
        resolved_logs = logs or {}
        current = resolved_logs.get(self.monitor)
        if current is None:
            return

        if self.mode == "min" or (self.mode == "auto" and "acc" not in self.monitor):
            improved = float(current) < self.best - self.min_delta
        else:
            improved = float(current) > self.best + self.min_delta

        if improved:
            self.best = float(current)
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = [w.data for w in self.model.weights]
        else:
            self.wait += 1
            if self.wait >= self.patience and epoch >= self.start_from_epoch:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    for w, val in zip(self.model.weights, self.best_weights):
                        w.data = val


class ModelCheckpoint(Callback):
    """Callback to save the Keras model or model weights at some frequency."""

    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        verbose: int = 0,
        save_best_only: bool = False,
        save_weights_only: bool = False,
        mode: str = "auto",
        save_freq: str | int = "epoch",
        initial_value_threshold: float | None = None,
    ) -> None:
        """Initialize the ModelCheckpoint callback.

        Args:
            filepath: String path to save the model or weight file.
            monitor: Metric name to monitor.
            verbose: Verbosity mode (0 or 1).
            save_best_only: If True, only save when the monitored metric improves.
            save_weights_only: If True, call ``model.save_weights`` instead of ``model.save``.
            mode: One of ``{"auto", "min", "max"}``.
            save_freq: ``"epoch"`` or integer number of batches between saves.
            initial_value_threshold: Initial best value threshold for ``monitor``.
        """
        super().__init__()
        self.filepath = str(filepath)
        self.monitor = str(monitor)
        self.verbose = int(verbose)
        self.save_best_only = bool(save_best_only)
        self.save_weights_only = bool(save_weights_only)
        self.mode = str(mode)
        self.save_freq = save_freq
        self.best = (
            float(initial_value_threshold)
            if initial_value_threshold is not None
            else (
                float("inf")
                if self.mode == "min" or (self.mode == "auto" and "acc" not in self.monitor)
                else -float("inf")
            )
        )

    def on_epoch_end(self, epoch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Evaluate checkpoint criteria at the end of an epoch and save model or weights.

        Args:
            epoch: Integer index of the completed epoch.
            logs: Dictionary of epoch metric results.
        """
        resolved_logs = logs or {}
        filepath = self.filepath.format(epoch=epoch + 1, **resolved_logs)

        should_save = False
        improved = False
        current = resolved_logs.get(self.monitor)
        if current is not None:
            if self.mode == "min" or (self.mode == "auto" and "acc" not in self.monitor):
                improved = float(current) < self.best
            else:
                improved = float(current) > self.best

            if improved:
                self.best = float(current)

        if not self.save_best_only or (current is not None and improved):
            should_save = True

        if should_save and self.model is not None:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)
            if self.save_weights_only and isinstance(self.model, _HasSaveWeightsProtocol):
                self.model.save_weights(filepath)
            else:
                self.model.save(filepath)


class History(Callback):
    """Callback that records events into a ``History`` object."""

    def __init__(self) -> None:
        """Initialize the History callback with empty ``history`` and ``epoch`` tracking."""
        super().__init__()
        self.history: dict[str, list[TensorLike]] = {}
        self.epoch: list[int] = []

    def on_train_begin(self, logs: dict[str, TensorLike] | None = None) -> None:
        """Reset recorded ``history`` and ``epoch`` lists at the start of training.

        Args:
            logs: Optional initial training log dictionary.
        """
        _ = logs
        self.history = {}
        self.epoch = []

    def on_epoch_end(self, epoch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Append epoch index and metric values from ``logs`` into ``self.history``.

        Args:
            epoch: Completed epoch index.
            logs: Dictionary of metric values for the epoch.
        """
        resolved_logs = logs or {}
        self.epoch.append(epoch)
        for k, v in resolved_logs.items():
            self.history.setdefault(k, []).append(v)


class LambdaCallback(Callback):
    """Callback for creating simple, custom callbacks on-the-fly."""

    def __init__(
        self,
        on_epoch_begin: EpochLogCallbackProtocol | None = None,
        on_epoch_end: EpochLogCallbackProtocol | None = None,
        on_batch_begin: EpochLogCallbackProtocol | None = None,
        on_batch_end: EpochLogCallbackProtocol | None = None,
        on_train_begin: EpochLogCallbackProtocol | None = None,
        on_train_end: EpochLogCallbackProtocol | None = None,
        **kwargs: ConfigValue,
    ) -> None:
        """Initialize LambdaCallback with optional hook callables.

        Args:
            on_epoch_begin: Callable invoked at the start of every epoch.
            on_epoch_end: Callable invoked at the end of every epoch.
            on_batch_begin: Callable invoked at the start of every batch.
            on_batch_end: Callable invoked at the end of every batch.
            on_train_begin: Callable invoked at the start of training.
            on_train_end: Callable invoked at the end of training.
            **kwargs: Additional keyword configuration stored in ``extra_callbacks``.
        """
        super().__init__()
        self.extra_callbacks: dict[str, ConfigValue] = dict(kwargs)
        self._on_epoch_begin = on_epoch_begin
        self._on_epoch_end = on_epoch_end
        self._on_batch_begin = on_batch_begin
        self._on_batch_end = on_batch_end
        self._on_train_begin = on_train_begin
        self._on_train_end = on_train_end

    def on_epoch_begin(self, epoch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Invoke the ``on_epoch_begin`` callable if provided.

        Args:
            epoch: Starting epoch index.
            logs: Optional epoch logs.
        """
        if self._on_epoch_begin is not None:
            self._on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Invoke the ``on_epoch_end`` callable if provided.

        Args:
            epoch: Completed epoch index.
            logs: Optional epoch logs.
        """
        if self._on_epoch_end is not None:
            self._on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Invoke the ``on_batch_begin`` callable if provided.

        Args:
            batch: Starting batch index.
            logs: Optional batch logs.
        """
        if self._on_batch_begin is not None:
            self._on_batch_begin(batch, logs)

    def on_batch_end(self, batch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Invoke the ``on_batch_end`` callable if provided.

        Args:
            batch: Completed batch index.
            logs: Optional batch logs.
        """
        if self._on_batch_end is not None:
            self._on_batch_end(batch, logs)

    def on_train_begin(self, logs: dict[str, TensorLike] | None = None) -> None:
        """Invoke the ``on_train_begin`` callable if provided.

        Args:
            logs: Optional training start logs.
        """
        if self._on_train_begin is not None:
            self._on_train_begin(logs)

    def on_train_end(self, logs: dict[str, TensorLike] | None = None) -> None:
        """Invoke the ``on_train_end`` callable if provided.

        Args:
            logs: Optional training completion logs.
        """
        if self._on_train_end is not None:
            self._on_train_end(logs)


class LearningRateScheduler(Callback):
    """Learning rate scheduler callback."""

    def __init__(self, schedule: ScheduleFnProtocol, verbose: int = 0) -> None:
        """Initialize LearningRateScheduler.

        Args:
            schedule: Function taking ``(epoch, current_lr)`` and returning a new learning rate.
            verbose: Verbosity mode (0 or 1).
        """
        super().__init__()
        self.schedule = schedule
        self.verbose = int(verbose)

    def on_epoch_begin(self, epoch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Update the model optimizer's ``learning_rate`` at the start of ``epoch``.

        Args:
            epoch: Starting epoch index.
            logs: Optional epoch logs.
        """
        _ = logs
        if not isinstance(self.model, _HasOptimizerProtocol) or not isinstance(
            self.model.optimizer, _HasLearningRateProtocol
        ):
            return
        try:
            lr: TensorLike = float(self.model.optimizer.learning_rate)
        except TypeError:
            lr = self.model.optimizer.learning_rate
        lr = self.schedule(epoch, lr)
        self.model.optimizer.learning_rate = lr


class TerminateOnNaN(Callback):
    """Callback that terminates training when a ``NaN`` or ``Inf`` loss is encountered."""

    def on_batch_end(self, batch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Inspect ``loss`` in ``logs`` and set ``model.stop_training = True`` if non-finite.

        Args:
            batch: Completed batch index.
            logs: Dictionary containing batch ``loss``.
        """
        _ = batch
        resolved_logs = logs or {}
        loss = resolved_logs.get("loss")
        if loss is not None and (math.isnan(float(loss)) or math.isinf(float(loss))):
            self.model.stop_training = True


class CSVLogger(Callback):
    """Callback that streams epoch results to a CSV file."""

    def __init__(
        self,
        filename: str,
        separator: str = ",",
        append: bool = False,
    ) -> None:
        """Initialize the CSVLogger callback.

        Args:
            filename: Path of the destination CSV file.
            separator: Delimiter string used between CSV fields.
            append: If True, append to an existing file rather than overwriting.
        """
        super().__init__()
        self.filename = str(filename)
        self.separator = str(separator)
        self.append = bool(append)
        self.csv_file: TensorLike | None = None
        self.writer: TensorLike | None = None
        self.keys: list[str] | None = None
        self.append_header = True

    def on_train_begin(self, logs: dict[str, TensorLike] | None = None) -> None:
        """Open the CSV file handle at the start of training.

        Args:
            logs: Optional initial training logs.
        """
        _ = logs
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, "r", encoding="utf-8") as f:
                    self.append_header = not bool(len(f.readline()))
            else:
                self.append_header = True
        else:
            self.append_header = True
        self.csv_file = open(
            self.filename,
            "a" if self.append else "w",
            newline="",
            encoding="utf-8",
        )

    def on_epoch_end(self, epoch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Write a row of epoch metrics to the open CSV file.

        Args:
            epoch: Completed epoch index.
            logs: Dictionary of epoch metric results.
        """
        resolved_logs = logs or {}

        def handle_value(k: TensorLike) -> TensorLike:
            """Format a metric value ``k`` for CSV serialization.

            Args:
                k: Metric value from ``logs``.

            Returns:
                TensorLike: Formatted scalar or string value.
            """
            is_zero_dim_ndarray = isinstance(k, _HasNdimProtocol) and k.ndim == 0
            if isinstance(k, str):
                return k
            if isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return f'"[{", ".join(map(str, k))}]"'
            return k

        if self.keys is None:
            self.keys = sorted(resolved_logs.keys())

            class CustomDialect(csv.excel):
                """Custom CSV dialect configured with ``self.separator``."""

                delimiter = self.separator

            self.writer = csv.DictWriter(
                self.csv_file,
                fieldnames=["epoch"] + self.keys,
                dialect=CustomDialect,
            )
            if self.append_header:
                self.writer.writeheader()

        row_dict: dict[str, TensorLike] = {"epoch": epoch}
        row_dict.update(
            (key, handle_value(resolved_logs[key])) for key in self.keys if key in resolved_logs
        )
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs: dict[str, TensorLike] | None = None) -> None:
        """Close the CSV file handle at the end of training.

        Args:
            logs: Optional final training logs.
        """
        _ = logs
        if self.csv_file is not None:
            self.csv_file.close()

    def __del__(self) -> None:
        """Ensure the CSV file handle is closed upon garbage collection."""
        if self.csv_file is not None and not self.csv_file.closed:
            self.csv_file.close()


class ReduceLROnPlateau(Callback):
    """Reduce learning rate when a metric has stopped improving."""

    def get_monitor_value(self, logs: dict[str, TensorLike]) -> TensorLike:
        """Retrieve the monitored metric value from ``logs``.

        Args:
            logs: Epoch metric dictionary.

        Returns:
            TensorLike: Monitored metric value or ``None``.
        """
        return logs.get(self.monitor)

    def __init__(
        self,
        monitor: str = "val_loss",
        factor: float = 0.1,
        patience: int = 10,
        verbose: int = 0,
        mode: str = "auto",
        min_delta: float = 1e-4,
        cooldown: int = 0,
        min_lr: float = 0.0,
    ) -> None:
        """Initialize the ReduceLROnPlateau callback.

        Args:
            monitor: Metric quantity name to monitor.
            factor: Multiplicative factor by which the learning rate will be reduced.
            patience: Number of epochs with no improvement before reducing learning rate.
            verbose: Verbosity mode (0 or 1).
            mode: One of ``{"auto", "min", "max"}``.
            min_delta: Threshold for measuring new optimum.
            cooldown: Number of epochs to wait before resuming normal operation after LR reduction.
            min_lr: Lower bound on the learning rate.
        """
        super().__init__()
        self.monitor = str(monitor)
        self.factor = float(factor)
        self.min_lr = float(min_lr)
        self.patience = int(patience)
        self.verbose = int(verbose)
        self.cooldown = int(cooldown)
        self.cooldown_counter = 0
        self.wait = 0
        self.best = 0.0
        self.mode = str(mode)
        self.min_delta = float(min_delta)

        if self.mode == "min" or (self.mode == "auto" and "acc" not in self.monitor):
            self.monitor_op: Callable[[float, float], bool] = lambda a, b: a < (b - self.min_delta)
            self.best = float("inf")
        else:
            self.monitor_op = lambda a, b: a > (b + self.min_delta)
            self.best = -float("inf")

    def on_train_begin(self, logs: dict[str, TensorLike] | None = None) -> None:
        """Reset wait and cooldown counters at the start of training.

        Args:
            logs: Optional initial training logs.
        """
        _ = logs
        self.wait = 0
        self.cooldown_counter = 0
        self.best = (
            float("inf")
            if self.mode == "min" or (self.mode == "auto" and "acc" not in self.monitor)
            else -float("inf")
        )

    def on_epoch_end(self, epoch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Evaluate plateau criteria at epoch end and reduce learning rate if triggered.

        Args:
            epoch: Completed epoch index.
            logs: Dictionary of epoch metric results.
        """
        _ = epoch
        resolved_logs = logs or {}
        current = resolved_logs.get(self.monitor)
        if current is None:
            return

        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.wait = 0

        if self.monitor_op(float(current), self.best):
            self.best = float(current)
            self.wait = 0
        elif not self.in_cooldown():
            self.wait += 1
            if self.wait >= self.patience:
                if isinstance(self.model, _HasOptimizerProtocol) and isinstance(
                    self.model.optimizer, _HasLearningRateProtocol
                ):
                    old_lr = float(self.model.optimizer.learning_rate)
                    if old_lr > self.min_lr:
                        new_lr = max(old_lr * self.factor, self.min_lr)
                        self.model.optimizer.learning_rate = new_lr
                        self.cooldown_counter = self.cooldown
                        self.wait = 0

    def in_cooldown(self) -> bool:
        """Return True if the callback is currently in its cooldown window.

        Returns:
            bool: True when ``cooldown_counter > 0``.
        """
        return self.cooldown_counter > 0


class BackupAndRestore(Callback):
    """Callback to back up and restore the training state across interruptions."""

    def __init__(
        self,
        backup_dir: str,
        save_freq: str | int = "epoch",
        delete_checkpoint: bool = True,
    ) -> None:
        """Initialize the BackupAndRestore callback.

        Args:
            backup_dir: Directory path where backup checkpoints are stored.
            save_freq: ``"epoch"`` or integer number of batches between backups.
            delete_checkpoint: Whether to delete the backup directory when training finishes.
        """
        super().__init__()
        self.backup_dir = str(backup_dir)
        self.save_freq = save_freq
        self.delete_checkpoint = bool(delete_checkpoint)

    def on_train_begin(self, logs: dict[str, TensorLike] | None = None) -> None:
        """Ensure ``backup_dir`` exists at the beginning of training.

        Args:
            logs: Optional initial training logs.
        """
        _ = logs
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)

    def on_epoch_end(self, epoch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Save backup checkpoint weights at the end of an epoch.

        Args:
            epoch: Completed epoch index.
            logs: Optional epoch metric logs.
        """
        _ = epoch, logs
        if isinstance(self.model, _HasSaveWeightsProtocol):
            ckpt_path = os.path.join(self.backup_dir, "checkpoint.weights.h5")
            self.model.save_weights(ckpt_path)

    def on_train_end(self, logs: dict[str, TensorLike] | None = None) -> None:
        """Remove ``backup_dir`` at the end of training when ``delete_checkpoint=True``.

        Args:
            logs: Optional final training logs.
        """
        _ = logs
        if self.delete_checkpoint:
            shutil.rmtree(self.backup_dir, ignore_errors=True)


class RemoteMonitor(Callback):
    """Callback used to stream events to a server."""

    def __init__(
        self,
        root: str = "http://localhost:9000",
        path: str = "/publish/epoch/end/",
        field: str = "data",
        headers: dict[str, str] | None = None,
        send_as_json: bool = False,
    ) -> None:
        """Initialize the RemoteMonitor callback.

        Args:
            root: Root URL of the target server.
            path: Endpoint path relative to ``root``.
            field: Form field name under which payload data is sent.
            headers: Optional dictionary of custom HTTP headers.
            send_as_json: Whether the request should be sent as ``application/json``.
        """
        super().__init__()
        self.root = str(root)
        self.path = str(path)
        self.field = str(field)
        self.headers = headers
        self.send_as_json = bool(send_as_json)


class SwapEMAWeights(Callback):
    """Swaps model weights with their exponential moving average during evaluation."""

    def __init__(self, swap_on_epoch: bool = False) -> None:
        """Initialize the SwapEMAWeights callback.

        Args:
            swap_on_epoch: Whether to swap weights at the end of each epoch for validation.
        """
        super().__init__()
        self.swap_on_epoch = bool(swap_on_epoch)


class TensorBoard(Callback):
    """Enable visualizations for TensorBoard."""

    def __init__(
        self,
        log_dir: str = "logs",
        histogram_freq: int = 0,
        write_graph: bool = True,
        write_images: bool = False,
        write_steps_per_second: bool = False,
        update_freq: str | int = "epoch",
        profile_batch: int = 0,
        embeddings_freq: int = 0,
        embeddings_metadata: dict[str, str] | str | None = None,
    ) -> None:
        """Initialize the TensorBoard callback.

        Args:
            log_dir: Path of the directory where to save the log files to be parsed by TensorBoard.
            histogram_freq: Frequency (in epochs) at which to compute weight histograms.
            write_graph: Whether to visualize the graph in TensorBoard.
            write_images: Whether to write model weights to visualize as image in TensorBoard.
            write_steps_per_second: Whether to log training steps per second.
            update_freq: ``"batch"``, ``"epoch"``, or integer number of batches.
            profile_batch: Profile the batch(es) to sample compute characteristics.
            embeddings_freq: Frequency (in epochs) at which embedding layers will be visualized.
            embeddings_metadata: Optional metadata mapping for embedding layer names.
        """
        super().__init__()
        self.log_dir = str(log_dir)
        self.histogram_freq = int(histogram_freq)
        self.write_graph = bool(write_graph)
        self.write_images = bool(write_images)
        self.write_steps_per_second = bool(write_steps_per_second)
        self.update_freq = update_freq
        self.profile_batch = int(profile_batch)
        self.embeddings_freq = int(embeddings_freq)
        self.embeddings_metadata = embeddings_metadata


class CallbackList:
    """Container abstracting a list of callbacks and dispatching lifecycle events."""

    def __init__(
        self,
        callbacks: Sequence[Callback] | None = None,
        model: TensorLike | None = None,
    ) -> None:
        """Initialize CallbackList with a sequence of ``Callback`` instances.

        Args:
            callbacks: Optional sequence of callbacks.
            model: Optional model instance to attach to all callbacks.
        """
        self._model: TensorLike | None = model
        self.params: dict[str, ConfigValue] | TensorLike = {}
        self.callbacks: list[Callback] = list(callbacks) if callbacks is not None else []
        for cb in self.callbacks:
            cb.model = model

    @property
    def model(self) -> TensorLike:
        """Return the model attached to this callback list.

        Returns:
            TensorLike: Attached model instance or ``None``.
        """
        return self._model

    @model.setter
    def model(self, value: TensorLike) -> None:
        """Set the model reference on this list.

        Args:
            value: Model instance to attach.
        """
        self._model = value

    def set_model(self, model: TensorLike) -> None:
        """Set the model reference on this list and all child callbacks.

        Args:
            model: Model instance to attach.
        """
        self.model = model

    def set_params(self, params: dict[str, ConfigValue] | TensorLike) -> None:
        """Set training parameters on this callback list.

        Args:
            params: Dictionary of training parameters.
        """
        self.params = params

    def on_train_begin(self, logs: dict[str, TensorLike] | None = None) -> None:
        """Dispatch ``on_train_begin`` to all callbacks.

        Args:
            logs: Optional initial training logs.
        """
        for cb in self.callbacks:
            cb.on_train_begin(logs)

    def on_epoch_begin(self, epoch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Dispatch ``on_epoch_begin`` to all callbacks.

        Args:
            epoch: Starting epoch index.
            logs: Optional epoch logs.
        """
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, logs)

    def on_train_batch_begin(self, batch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Dispatch ``on_train_batch_begin`` to all callbacks.

        Args:
            batch: Starting training batch index.
            logs: Optional batch logs.
        """
        for cb in self.callbacks:
            cb.on_train_batch_begin(batch, logs)

    def on_batch_begin(self, batch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Dispatch ``on_batch_begin`` to all callbacks.

        Args:
            batch: Starting batch index.
            logs: Optional batch logs.
        """
        for cb in self.callbacks:
            cb.on_batch_begin(batch, logs)

    def on_train_batch_end(self, batch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Dispatch ``on_train_batch_end`` to all callbacks.

        Args:
            batch: Completed training batch index.
            logs: Optional batch logs.
        """
        for cb in self.callbacks:
            cb.on_train_batch_end(batch, logs)

    def on_batch_end(self, batch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Dispatch ``on_batch_end`` to all callbacks.

        Args:
            batch: Completed batch index.
            logs: Optional batch logs.
        """
        for cb in self.callbacks:
            cb.on_batch_end(batch, logs)

    def on_epoch_end(self, epoch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Dispatch ``on_epoch_end`` to all callbacks.

        Args:
            epoch: Completed epoch index.
            logs: Optional epoch logs.
        """
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs)

    def on_train_end(self, logs: dict[str, TensorLike] | None = None) -> None:
        """Dispatch ``on_train_end`` to all callbacks.

        Args:
            logs: Optional final training logs.
        """
        for cb in self.callbacks:
            cb.on_train_end(logs)

    def on_test_begin(self, logs: dict[str, TensorLike] | None = None) -> None:
        """Dispatch ``on_test_begin`` to all callbacks.

        Args:
            logs: Optional evaluation start logs.
        """
        for cb in self.callbacks:
            cb.on_test_begin(logs)

    def on_test_batch_begin(self, batch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Dispatch ``on_test_batch_begin`` to all callbacks.

        Args:
            batch: Starting evaluation batch index.
            logs: Optional batch logs.
        """
        for cb in self.callbacks:
            cb.on_test_batch_begin(batch, logs)

    def on_test_batch_end(self, batch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Dispatch ``on_test_batch_end`` to all callbacks.

        Args:
            batch: Completed evaluation batch index.
            logs: Optional batch logs.
        """
        for cb in self.callbacks:
            cb.on_test_batch_end(batch, logs)

    def on_test_end(self, logs: dict[str, TensorLike] | None = None) -> None:
        """Dispatch ``on_test_end`` to all callbacks.

        Args:
            logs: Optional evaluation completion logs.
        """
        for cb in self.callbacks:
            cb.on_test_end(logs)

    def on_predict_begin(self, logs: dict[str, TensorLike] | None = None) -> None:
        """Dispatch ``on_predict_begin`` to all callbacks.

        Args:
            logs: Optional prediction start logs.
        """
        for cb in self.callbacks:
            cb.on_predict_begin(logs)

    def on_predict_batch_begin(self, batch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Dispatch ``on_predict_batch_begin`` to all callbacks.

        Args:
            batch: Starting prediction batch index.
            logs: Optional batch logs.
        """
        for cb in self.callbacks:
            cb.on_predict_batch_begin(batch, logs)

    def on_predict_batch_end(self, batch: int, logs: dict[str, TensorLike] | None = None) -> None:
        """Dispatch ``on_predict_batch_end`` to all callbacks.

        Args:
            batch: Completed prediction batch index.
            logs: Optional batch logs.
        """
        for cb in self.callbacks:
            cb.on_predict_batch_end(batch, logs)

    def on_predict_end(self, logs: dict[str, TensorLike] | None = None) -> None:
        """Dispatch ``on_predict_end`` to all callbacks.

        Args:
            logs: Optional prediction completion logs.
        """
        for cb in self.callbacks:
            cb.on_predict_end(logs)


class ProgbarLogger(Callback):
    """Callback that prints metrics to stdout during training and evaluation."""

    def __init__(
        self, count_mode: str = "samples", stateful_metrics: Sequence[str] | None = None
    ) -> None:
        """Initialize the ProgbarLogger callback.

        Args:
            count_mode: One of ``"samples"`` or ``"steps"``.
            stateful_metrics: Optional iterable of metric names that should not be averaged.
        """
        super().__init__()
        self.count_mode = str(count_mode)
        self.stateful_metrics: list[str] = (
            list(stateful_metrics) if stateful_metrics is not None else []
        )
