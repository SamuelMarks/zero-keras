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
