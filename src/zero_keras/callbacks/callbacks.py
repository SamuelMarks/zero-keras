"""Module docstring."""


class Callback:
    """Base class used to build new callbacks.

    Callbacks can be passed to keras methods such as `fit()`, `evaluate()`, and
    `predict()` in order to hook into the various stages of the model training,
    evaluation, and inference lifecycle.

    To create a custom callback, subclass `keras.callbacks.Callback` and
    override the method associated with the stage of interest.

    Example:
    >>> training_finished = False
    >>> class MyCallback(Callback):
    ...   def on_train_end(self, logs=None):
    ...     global training_finished
    ...     training_finished = True
    >>> model = Sequential([
    ...     layers.Dense(1, input_shape=(1,))])
    >>> model.compile(loss='mean_squared_error')
    >>> model.fit(np.array([[1.0]]), np.array([[1.0]]),
    ...           callbacks=[MyCallback()])
    >>> assert training_finished == True

    If you want to use `Callback` objects in a custom training loop:

    1. You should pack all your callbacks into a single `callbacks.CallbackList`
       so they can all be called together.
    2. You will need to manually call all the `on_*` methods at the appropriate
       locations in your loop. Like this:

    Example:
    ```python
    callbacks =  keras.callbacks.CallbackList([...])
    callbacks.append(...)
    callbacks.on_train_begin(...)
    for epoch in range(EPOCHS):
        callbacks.on_epoch_begin(epoch)
        for i, data in dataset.enumerate():
        callbacks.on_train_batch_begin(i)
        batch_logs = model.train_step(data)
        callbacks.on_train_batch_end(i, batch_logs)
        epoch_logs = ...
        callbacks.on_epoch_end(epoch, epoch_logs)
    final_logs=...
    callbacks.on_train_end(final_logs)
    ```

    Attributes:
        params: Dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: Instance of `Model`.
            Reference of the model being trained.

    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch (see method-specific docstrings).

    """

    def __init__(self):
        """Function docstring."""
        self.model = None

    def set_model(self, model):
        """set_model function.

        Args:
        model: Parameter model.

        Returns:
        Any: Return value.

        """
        self.model = model

    def set_params(self, params):
        """set_params function.

        Args:
        params: Parameter params.

        Returns:
        Any: Return value.

        """
        self.params = params

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.

        """
        pass

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict, metric results for this training epoch, and for the
              validation epoch if validation is performed. Validation result
              keys are prefixed with `val_`. For training epoch, the values of
              the `Model`'s metrics are returned. Example:
              `{'loss': 0.2, 'accuracy': 0.7}`.

        """
        pass

    def on_batch_begin(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_begin`."""
        pass

    def on_batch_end(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_end`."""
        pass

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.

        """
        pass

    def on_train_end(self, logs=None):
        """Called at the end of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently the output of the last call to
              `on_epoch_end()` is passed to this argument for this method but
              that may change in the future.

        """
        pass


class EarlyStopping(Callback):
    """Stop training when a monitored metric has stopped improving.

    Assuming the goal of a training is to minimize the loss. With this, the
    metric to be monitored would be `'loss'`, and mode would be `'min'`. A
    `model.fit()` training loop will check at end of every epoch whether
    the loss is no longer decreasing, considering the `min_delta` and
    `patience` if applicable. Once it's found no longer decreasing,
    `model.stop_training` is marked True and the training terminates.

    The quantity to be monitored needs to be available in `logs` dict.
    To make it so, pass the loss or metrics at `model.compile()`.

    Args:
        monitor: Quantity to be monitored. Defaults to `"val_loss"`.
        min_delta: Minimum change in the monitored quantity to qualify as an
            improvement, i.e. an absolute change of less than min_delta, will
            count as no improvement. Defaults to `0`.
        patience: Number of epochs with no improvement after which training will
            be stopped. Defaults to `0`.
        verbose: Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1 displays
            messages when the callback takes an action. Defaults to `0`.
        mode: One of `{"auto", "min", "max"}`. In `min` mode, training will stop
            when the quantity monitored has stopped decreasing; in `"max"` mode
            it will stop when the quantity monitored has stopped increasing; in
            `"auto"` mode, the direction is automatically inferred from the name
            of the monitored quantity. Defaults to `"auto"`.
        baseline: Baseline value for the monitored quantity. If not `None`,
            training will stop if the model doesn't show improvement over the
            baseline. Defaults to `None`.
        restore_best_weights: Whether to restore model weights from the epoch
            with the best value of the monitored quantity. If `False`, the model
            weights obtained at the last step of training are used. An epoch
            will be restored regardless of the performance relative to the
            `baseline`. If no epoch improves on `baseline`, training will run
            for `patience` epochs and restore weights from the best epoch in
            that set. Defaults to `False`.
        start_from_epoch: Number of epochs to wait before starting to monitor
            improvement. This allows for a warm-up period in which no
            improvement is expected and thus training will not be stopped.
            Defaults to `0`.

    Example:
    >>> callback = keras.callbacks.EarlyStopping(monitor='loss',
    ...                                               patience=3)
    >>> # This callback will stop the training when there is no improvement in
    >>> # the loss for three consecutive epochs.
    >>> model = keras.models.Sequential([keras.layers.Dense(10)])
    >>> model.compile(keras.optimizers.SGD(), loss='mse')
    >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
    ...                     epochs=10, batch_size=1, callbacks=[callback],
    ...                     verbose=0)
    >>> len(history.history['loss'])  # Only 4 epochs are run.
    4

    """

    def __init__(
        self,
        monitor="val_loss",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
        start_from_epoch=0,
    ):
        """Function docstring.

        Args:
            monitor: Description.
            min_delta: Description.
            patience: Description.
            verbose: Description.
            mode: Description.
            baseline: Description.
            restore_best_weights: Description.
            start_from_epoch: Description.
        """
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.start_from_epoch = start_from_epoch
        self.wait = 0
        self.stopped_epoch = 0
        self.best = (
            float("inf")
            if mode == "min" or (mode == "auto" and "acc" not in monitor)
            else -float("inf")
        )
        self.best_weights = None

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.

        """
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict, metric results for this training epoch, and for the
              validation epoch if validation is performed. Validation result
              keys are prefixed with `val_`. For training epoch, the values of
              the `Model`'s metrics are returned. Example:
              `{'loss': 0.2, 'accuracy': 0.7}`.

        """
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return  # pragma: no cover

        if self.mode == "min" or (self.mode == "auto" and "acc" not in self.monitor):
            improved = current < self.best - self.min_delta
        else:
            improved = current > self.best + self.min_delta  # pragma: no cover

        if improved:
            self.best = current
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
    """Callback to save the Keras model or model weights at some frequency.

    `ModelCheckpoint` callback is used in conjunction with training using
    `model.fit()` to save a model or weights (in a checkpoint file) at some
    interval, so the model or weights can be loaded later to continue the
    training from the state saved.

    A few options this callback provides include:

    - Whether to only keep the model that has achieved the "best performance" so
      far, or whether to save the model at the end of every epoch regardless of
      performance.
    - Definition of "best"; which quantity to monitor and whether it should be
      maximized or minimized.
    - The frequency it should save at. Currently, the callback supports saving
      at the end of every epoch, or after a fixed number of training batches.
    - Whether only weights are saved, or the whole model is saved.

    Example:
    ```python
    model.compile(loss=..., optimizer=...,
                  metrics=['accuracy'])

    EPOCHS = 10
    checkpoint_filepath = '/tmp/ckpt/checkpoint.model.keras'
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # Model is saved at the end of every epoch, if it's the best seen so far.
    model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])

    # The model (that are considered the best) can be loaded as -
    keras.models.load_model(checkpoint_filepath)

    # Alternatively, one could checkpoint just the model weights as -
    checkpoint_filepath = '/tmp/ckpt/checkpoint.weights.h5'
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # Model weights are saved at the end of every epoch, if it's the best seen
    # so far.
    model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])

    # The model weights (that are considered the best) can be loaded as -
    model.load_weights(checkpoint_filepath)
    ```

    Args:
        filepath: string or `PathLike`, path to save the model file.
            `filepath` can contain named formatting options,
            which will be filled the value of `epoch` and keys in `logs`
            (passed in `on_epoch_end`).
            The `filepath` name needs to end with `".weights.h5"` when
            `save_weights_only=True` or should end with `".keras"` or `".h5"`
            when checkpoint saving the whole model (default).
            For example:
            if `filepath` is `"{epoch:02d}-{val_loss:.2f}.keras"` or
            "{epoch:02d}-{val_loss:.2f}.weights.h5"`, then the model
            checkpoints will be saved with the epoch number and the validation
            loss in the filename. The directory of the filepath
            should not be reused by any other callbacks to avoid conflicts.
        monitor: The metric name to monitor. Typically the metrics are set by
            the `Model.compile` method. Note:
            * Prefix the name with `"val_"` to monitor validation metrics.
            * Use `"loss"` or `"val_loss"` to monitor the model's total loss.
            * If you specify metrics as strings, like `"accuracy"`, pass the
                same string (with or without the `"val_"` prefix).
            * If you pass `metrics.Metric` objects, `monitor` should be set to
                `metric.name`
            * If you're not sure about the metric names you can check the
                contents of the `history.history` dictionary returned by
                `history = model.fit()`
            * Multi-output models set additional prefixes on the metric names.
        verbose: Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1
            displays messages when the callback takes an action.
        save_best_only: if `save_best_only=True`, it only saves when the model
            is considered the "best" and the latest best model according to the
            quantity monitored will not be overwritten. If `filepath` doesn't
            contain formatting options like `{epoch}` then `filepath` will be
            overwritten by each new better model.
        mode: one of {`"auto"`, `"min"`, `"max"`}. If `save_best_only=True`, the
            decision to overwrite the current save file is made based on either
            the maximization or the minimization of the monitored quantity.
            For `val_acc`, this should be `"max"`, for `val_loss` this should be
            `"min"`, etc. In `"auto"` mode, the mode is set to `"max"` if the
            quantities monitored are `"acc"` or start with `"fmeasure"` and are
            set to `"min"` for the rest of the quantities.
        save_weights_only: if `True`, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model is
            saved (`model.save(filepath)`).
        save_freq: `"epoch"` or integer. When using `"epoch"`, the callback
            saves the model after each epoch. When using integer, the callback
            saves the model at end of this many batches. If the `Model` is
            compiled with `steps_per_execution=N`, then the saving criteria will
            be checked every Nth batch. Note that if the saving isn't aligned to
            epochs, the monitored metric may potentially be less reliable (it
            could reflect as little as 1 batch, since the metrics get reset
            every epoch). Defaults to `"epoch"`.
        initial_value_threshold: Floating point initial "best" value of the
            metric to be monitored. Only applies if `save_best_value=True`. Only
            overwrites the model weights already saved if the performance of
            current model is better than this value.

    """

    def __init__(
        self,
        filepath,
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        initial_value_threshold=None,
    ):
        """Function docstring.

        Args:
            filepath: Description.
            monitor: Description.
            verbose: Description.
            save_best_only: Description.
            save_weights_only: Description.
            mode: Description.
            save_freq: Description.
            initial_value_threshold: Description.
        """
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.save_freq = save_freq
        self.best = (
            initial_value_threshold
            if initial_value_threshold is not None
            else (
                float("inf")
                if mode == "min" or (mode == "auto" and "acc" not in monitor)
                else -float("inf")
            )
        )

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict, metric results for this training epoch, and for the
              validation epoch if validation is performed. Validation result
              keys are prefixed with `val_`. For training epoch, the values of
              the `Model`'s metrics are returned. Example:
              `{'loss': 0.2, 'accuracy': 0.7}`.

        """
        logs = logs or {}
        filepath = self.filepath.format(epoch=epoch + 1, **logs)

        save_model = False
        current = logs.get(self.monitor)
        if current is not None:
            if self.mode == "min" or (
                self.mode == "auto" and "acc" not in self.monitor
            ):
                improved = current < self.best
            else:
                improved = current > self.best

            if improved:
                self.best = current

        if not self.save_best_only:
            save_model = True
        elif current is not None and improved:
            save_model = True

        if save_model and self.model is not None:
            import os

            os.makedirs(
                os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True
            )
            self.model.save(filepath)


class History(Callback):
    """Callback that records events into a `History` object."""

    def __init__(self):
        """Function docstring."""
        super().__init__()
        self.history = {}
        self.epoch = []

    def on_train_begin(self, logs=None):
        """Called at the beginning of training."""
        self.history = {}
        self.epoch = []

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch."""
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


class LambdaCallback(Callback):
    """Callback for creating simple, custom callbacks on-the-fly."""

    def __init__(
        self,
        on_epoch_begin=None,
        on_epoch_end=None,
        on_batch_begin=None,
        on_batch_end=None,
        on_train_begin=None,
        on_train_end=None,
        **kwargs,
    ):
        """Function docstring.

        Args:
            on_epoch_begin: Description.
            on_epoch_end: Description.
            on_batch_begin: Description.
            on_batch_end: Description.
            on_train_begin: Description.
            on_train_end: Description.
            kwargs: Description.
        """
        super().__init__()
        self.__dict__.update(kwargs)
        self._on_epoch_begin = on_epoch_begin
        self._on_epoch_end = on_epoch_end
        self._on_batch_begin = on_batch_begin
        self._on_batch_end = on_batch_end
        self._on_train_begin = on_train_begin
        self._on_train_end = on_train_end

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch."""
        if self._on_epoch_begin is not None:
            self._on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch."""
        if self._on_epoch_end is not None:
            self._on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        """Called at the start of a batch."""
        if self._on_batch_begin is not None:
            self._on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        """Called at the end of a batch."""
        if self._on_batch_end is not None:
            self._on_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        """Called at the beginning of training."""
        if self._on_train_begin is not None:
            self._on_train_begin(logs)

    def on_train_end(self, logs=None):
        """Called at the end of training."""
        if self._on_train_end is not None:
            self._on_train_end(logs)


class LearningRateScheduler(Callback):
    """Learning rate scheduler."""

    def __init__(self, schedule, verbose=0):
        """Function docstring.

        Args:
            schedule: Description.
            verbose: Description.
        """
        super().__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch."""
        if not hasattr(self.model, "optimizer"):
            return
        if not hasattr(self.model.optimizer, "learning_rate"):
            return
        try:
            lr = float(self.model.optimizer.learning_rate)
        except TypeError:
            lr = self.model.optimizer.learning_rate
        lr = self.schedule(epoch, lr)
        self.model.optimizer.learning_rate = lr


class TerminateOnNaN(Callback):
    """Callback that terminates training when a NaN loss is encountered."""

    def on_batch_end(self, batch, logs=None):
        """Called at the end of a batch."""
        import math

        logs = logs or {}
        loss = logs.get("loss")
        if loss is not None:
            if math.isnan(loss) or math.isinf(loss):
                self.model.stop_training = True


class CSVLogger(Callback):
    """Callback that streams epoch results to a CSV file."""

    def __init__(self, filename, separator=",", append=False):
        """Function docstring.

        Args:
            filename: Description.
            separator: Description.
            append: Description.
        """
        super().__init__()
        self.filename = filename
        self.separator = separator
        self.append = append
        self.csv_file = None
        self.writer = None
        self.keys = None
        self.append_header = True

    def on_train_begin(self, logs=None):
        """Called at the beginning of training."""
        import os

        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, "r") as f:
                    self.append_header = not bool(len(f.readline()))
            else:
                self.append_header = True
        else:
            self.append_header = True
        self.csv_file = open(self.filename, "a" if self.append else "w", newline="")

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch."""
        import csv

        logs = logs or {}

        def handle_value(k):
            """Function docstring.

            Args:
                k: Description.
            """
            is_zero_dim_ndarray = hasattr(k, "ndim") and k.ndim == 0
            if isinstance(k, str):
                return k
            elif hasattr(k, "__iter__") and not is_zero_dim_ndarray:
                return f'"[{", ".join(map(str, k))}]"'
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())
            if self.writer is None:

                class CustomDialect(csv.excel):
                    """Class docstring."""

                    delimiter = self.separator

                self.writer = csv.DictWriter(
                    self.csv_file,
                    fieldnames=["epoch"] + self.keys,
                    dialect=CustomDialect,
                )
                if self.append_header:
                    self.writer.writeheader()

        row_dict = {"epoch": epoch}
        row_dict.update(
            (key, handle_value(logs[key])) for key in self.keys if key in logs
        )
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        """Called at the end of training."""
        if self.csv_file is not None:
            self.csv_file.close()


class ReduceLROnPlateau(Callback):
    """Reduce learning rate when a metric has stopped improving."""

    def __init__(
        self,
        monitor="val_loss",
        factor=0.1,
        patience=10,
        verbose=0,
        mode="auto",
        min_delta=1e-4,
        cooldown=0,
        min_lr=0,
    ):
        """Function docstring.

        Args:
            monitor: Description.
            factor: Description.
            patience: Description.
            verbose: Description.
            mode: Description.
            min_delta: Description.
            cooldown: Description.
            min_lr: Description.
        """
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.min_lr = min_lr
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.min_delta = min_delta

        if mode == "min" or (mode == "auto" and "acc" not in monitor):
            self.monitor_op = lambda a, b: a < (b - self.min_delta)
            self.best = float("inf")
        else:
            self.monitor_op = lambda a, b: a > (b + self.min_delta)
            self.best = -float("inf")

    def on_train_begin(self, logs=None):
        """Called at the beginning of training."""

        self.wait = 0
        self.cooldown_counter = 0
        self.best = (
            float("inf")
            if self.mode == "min" or (self.mode == "auto" and "acc" not in self.monitor)
            else -float("inf")
        )

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch."""
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.wait = 0

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        elif not self.in_cooldown():
            self.wait += 1
            if self.wait >= self.patience:
                if hasattr(self.model, "optimizer") and hasattr(
                    self.model.optimizer, "learning_rate"
                ):
                    old_lr = float(self.model.optimizer.learning_rate)
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        self.model.optimizer.learning_rate = new_lr
                        self.cooldown_counter = self.cooldown
                        self.wait = 0

    def in_cooldown(self):
        """in_cooldown."""
        return self.cooldown_counter > 0


class BackupAndRestore(Callback):
    """Callback to back up and restore the training state."""

    def __init__(self, backup_dir, save_freq="epoch", delete_checkpoint=True):
        """Function docstring.

        Args:
            backup_dir: Description.
            save_freq: Description.
            delete_checkpoint: Description.
        """
        super().__init__()
        self.backup_dir = backup_dir
        self.save_freq = save_freq
        self.delete_checkpoint = delete_checkpoint

    def on_train_begin(self, logs=None):
        """Called at the beginning of training."""
        import os

        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch."""
        pass

    def on_train_end(self, logs=None):
        """Called at the end of training."""
        if self.delete_checkpoint:
            import shutil

            shutil.rmtree(self.backup_dir, ignore_errors=True)


class RemoteMonitor(Callback):
    """Callback used to stream events to a server."""

    def __init__(
        self,
        root="http://localhost:9000",
        path="/publish/epoch/end/",
        field="data",
        headers=None,
        send_as_json=False,
    ):
        """Function docstring.

        Args:
            root: Description.
            path: Description.
            field: Description.
            headers: Description.
            send_as_json: Description.
        """
        super().__init__()
        self.root = root
        self.path = path
        self.field = field
        self.headers = headers
        self.send_as_json = send_as_json


class SwapEMAWeights(Callback):
    """Swaps model weights with their exponential moving average."""

    def __init__(self, swap_on_epoch=False):
        """Function docstring.

        Args:
            swap_on_epoch: Description.
        """
        super().__init__()
        self.swap_on_epoch = swap_on_epoch


class TensorBoard(Callback):
    """Enable visualizations for TensorBoard."""

    def __init__(
        self,
        log_dir="logs",
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        write_steps_per_second=False,
        update_freq="epoch",
        profile_batch=0,
        embeddings_freq=0,
        embeddings_metadata=None,
    ):
        """Function docstring.

        Args:
            log_dir: Description.
            histogram_freq: Description.
            write_graph: Description.
            write_images: Description.
            write_steps_per_second: Description.
            update_freq: Description.
            profile_batch: Description.
            embeddings_freq: Description.
            embeddings_metadata: Description.
        """
        super().__init__()
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.write_graph = write_graph
        self.write_images = write_images
        self.write_steps_per_second = write_steps_per_second
        self.update_freq = update_freq
        self.profile_batch = profile_batch
        self.embeddings_freq = embeddings_freq
        self.embeddings_metadata = embeddings_metadata


class CallbackList:
    """Class docstring."""

    def __init__(self, callbacks=None, model=None):
        """Function docstring.

        Args:
            callbacks: Description.
            model: Description.
        """
        self.callbacks = callbacks or []
        self.model = model
        for cb in self.callbacks:
            cb.model = model

    def on_train_begin(self, logs=None):
        """Function docstring.

        Args:
            logs: Description.
        """
        for cb in self.callbacks:
            cb.on_train_begin(logs)

    def on_epoch_begin(self, epoch, logs=None):
        """Function docstring.

        Args:
            epoch: Description.
            logs: Description.
        """
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, logs)

    def on_train_batch_begin(self, batch, logs=None):
        """Function docstring.

        Args:
            batch: Description.
            logs: Description.
        """
        for cb in self.callbacks:
            if hasattr(cb, "on_train_batch_begin"):
                cb.on_train_batch_begin(batch, logs)

    def on_batch_begin(self, batch, logs=None):
        """Function docstring.

        Args:
            batch: Description.
            logs: Description.
        """
        for cb in self.callbacks:
            cb.on_batch_begin(batch, logs)

    def on_train_batch_end(self, batch, logs=None):
        """Function docstring.

        Args:
            batch: Description.
            logs: Description.
        """
        for cb in self.callbacks:
            if hasattr(cb, "on_train_batch_end"):
                cb.on_train_batch_end(batch, logs)

    def on_batch_end(self, batch, logs=None):
        """Function docstring.

        Args:
            batch: Description.
            logs: Description.
        """
        for cb in self.callbacks:
            cb.on_batch_end(batch, logs)

    def on_epoch_end(self, epoch, logs=None):
        """Function docstring.

        Args:
            epoch: Description.
            logs: Description.
        """
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        """Function docstring.

        Args:
            logs: Description.
        """
        for cb in self.callbacks:
            cb.on_train_end(logs)

    def on_test_begin(self, logs=None):
        """Function docstring.

        Args:
            logs: Description.
        """
        for cb in self.callbacks:
            if hasattr(cb, "on_test_begin"):
                cb.on_test_begin(logs)

    def on_test_batch_begin(self, batch, logs=None):
        """Function docstring.

        Args:
            batch: Description.
            logs: Description.
        """
        for cb in self.callbacks:
            if hasattr(cb, "on_test_batch_begin"):
                cb.on_test_batch_begin(batch, logs)

    def on_test_batch_end(self, batch, logs=None):
        """Function docstring.

        Args:
            batch: Description.
            logs: Description.
        """
        for cb in self.callbacks:
            if hasattr(cb, "on_test_batch_end"):
                cb.on_test_batch_end(batch, logs)

    def on_test_end(self, logs=None):
        """Function docstring.

        Args:
            logs: Description.
        """
        for cb in self.callbacks:
            if hasattr(cb, "on_test_end"):
                cb.on_test_end(logs)

    def on_predict_begin(self, logs=None):
        """Function docstring.

        Args:
            logs: Description.
        """
        for cb in self.callbacks:
            if hasattr(cb, "on_predict_begin"):
                cb.on_predict_begin(logs)

    def on_predict_batch_begin(self, batch, logs=None):
        """Function docstring.

        Args:
            batch: Description.
            logs: Description.
        """
        for cb in self.callbacks:
            if hasattr(cb, "on_predict_batch_begin"):
                cb.on_predict_batch_begin(batch, logs)

    def on_predict_batch_end(self, batch, logs=None):
        """Function docstring.

        Args:
            batch: Description.
            logs: Description.
        """
        for cb in self.callbacks:
            if hasattr(cb, "on_predict_batch_end"):
                cb.on_predict_batch_end(batch, logs)

    def on_predict_end(self, logs=None):
        """Function docstring.

        Args:
            logs: Description.
        """
        for cb in self.callbacks:
            if hasattr(cb, "on_predict_end"):
                cb.on_predict_end(logs)


class ProgbarLogger:
    """Class docstring."""

    pass
