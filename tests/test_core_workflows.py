"""Tests for core model training, evaluation, prediction, and callback workflows."""

from __future__ import annotations

import os
import tempfile

from zero_keras import random
from zero_keras.callbacks.callbacks import Callback
from zero_keras.layers import Dense
from zero_keras.models import Sequential
from zero_keras.types import TensorLike

from .conftest import assert_allclose_zero


class CountingCallback(Callback):
    """Callback that counts invocations of every lifecycle hook."""

    def __init__(self) -> None:
        """Initialize the CountingCallback with zeroed counters."""
        super().__init__()
        self.counts = {
            "on_train_begin": 0,
            "on_epoch_begin": 0,
            "on_train_batch_begin": 0,
            "on_batch_begin": 0,
            "on_train_batch_end": 0,
            "on_batch_end": 0,
            "on_epoch_end": 0,
            "on_train_end": 0,
            "on_test_begin": 0,
            "on_test_batch_begin": 0,
            "on_test_batch_end": 0,
            "on_test_end": 0,
            "on_predict_begin": 0,
            "on_predict_batch_begin": 0,
            "on_predict_batch_end": 0,
            "on_predict_end": 0,
        }

    def on_train_begin(self, logs: TensorLike | None = None) -> None:
        """Increment ``on_train_begin`` count.

        Args:
            logs: Optional training logs dictionary.
        """
        _ = logs
        self.counts["on_train_begin"] += 1

    def on_epoch_begin(self, epoch: int, logs: TensorLike | None = None) -> None:
        """Increment ``on_epoch_begin`` count.

        Args:
            epoch: Current epoch index.
            logs: Optional epoch logs dictionary.
        """
        _ = epoch, logs
        self.counts["on_epoch_begin"] += 1

    def on_train_batch_begin(self, batch: int, logs: TensorLike | None = None) -> None:
        """Increment ``on_train_batch_begin`` count.

        Args:
            batch: Current batch index.
            logs: Optional batch logs dictionary.
        """
        _ = batch, logs
        self.counts["on_train_batch_begin"] += 1

    def on_batch_begin(self, batch: int, logs: TensorLike | None = None) -> None:
        """Increment ``on_batch_begin`` count.

        Args:
            batch: Current batch index.
            logs: Optional batch logs dictionary.
        """
        _ = batch, logs
        self.counts["on_batch_begin"] += 1

    def on_train_batch_end(self, batch: int, logs: TensorLike | None = None) -> None:
        """Increment ``on_train_batch_end`` count.

        Args:
            batch: Current batch index.
            logs: Optional batch logs dictionary.
        """
        _ = batch, logs
        self.counts["on_train_batch_end"] += 1

    def on_batch_end(self, batch: int, logs: TensorLike | None = None) -> None:
        """Increment ``on_batch_end`` count.

        Args:
            batch: Current batch index.
            logs: Optional batch logs dictionary.
        """
        _ = batch, logs
        self.counts["on_batch_end"] += 1

    def on_epoch_end(self, epoch: int, logs: TensorLike | None = None) -> None:
        """Increment ``on_epoch_end`` count.

        Args:
            epoch: Current epoch index.
            logs: Optional epoch logs dictionary.
        """
        _ = epoch, logs
        self.counts["on_epoch_end"] += 1

    def on_train_end(self, logs: TensorLike | None = None) -> None:
        """Increment ``on_train_end`` count.

        Args:
            logs: Optional training logs dictionary.
        """
        _ = logs
        self.counts["on_train_end"] += 1

    def on_test_begin(self, logs: TensorLike | None = None) -> None:
        """Increment ``on_test_begin`` count.

        Args:
            logs: Optional evaluation logs dictionary.
        """
        _ = logs
        self.counts["on_test_begin"] += 1

    def on_test_batch_begin(self, batch: int, logs: TensorLike | None = None) -> None:
        """Increment ``on_test_batch_begin`` count.

        Args:
            batch: Current evaluation batch index.
            logs: Optional batch logs dictionary.
        """
        _ = batch, logs
        self.counts["on_test_batch_begin"] += 1

    def on_test_batch_end(self, batch: int, logs: TensorLike | None = None) -> None:
        """Increment ``on_test_batch_end`` count.

        Args:
            batch: Current evaluation batch index.
            logs: Optional batch logs dictionary.
        """
        _ = batch, logs
        self.counts["on_test_batch_end"] += 1

    def on_test_end(self, logs: TensorLike | None = None) -> None:
        """Increment ``on_test_end`` count.

        Args:
            logs: Optional evaluation logs dictionary.
        """
        _ = logs
        self.counts["on_test_end"] += 1

    def on_predict_begin(self, logs: TensorLike | None = None) -> None:
        """Increment ``on_predict_begin`` count.

        Args:
            logs: Optional prediction logs dictionary.
        """
        _ = logs
        self.counts["on_predict_begin"] += 1

    def on_predict_batch_begin(self, batch: int, logs: TensorLike | None = None) -> None:
        """Increment ``on_predict_batch_begin`` count.

        Args:
            batch: Current prediction batch index.
            logs: Optional batch logs dictionary.
        """
        _ = batch, logs
        self.counts["on_predict_batch_begin"] += 1

    def on_predict_batch_end(self, batch: int, logs: TensorLike | None = None) -> None:
        """Increment ``on_predict_batch_end`` count.

        Args:
            batch: Current prediction batch index.
            logs: Optional batch logs dictionary.
        """
        _ = batch, logs
        self.counts["on_predict_batch_end"] += 1

    def on_predict_end(self, logs: TensorLike | None = None) -> None:
        """Increment ``on_predict_end`` count.

        Args:
            logs: Optional prediction logs dictionary.
        """
        _ = logs
        self.counts["on_predict_end"] += 1


def test_compile_resolution() -> None:
    """Verify or execute `test compile resolution`."""
    model = Sequential([Dense(2)])
    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
    assert type(model.optimizer).__name__ == "Adam"
    assert model.loss_fn.name == "mean_squared_error"
    assert type(model.compiled_metrics[0]).__name__ == "accuracy" or isinstance(
        model.compiled_metrics[0], str
    )

    # Test callables
    def my_loss(y: TensorLike, p: TensorLike) -> TensorLike:
        """Verify or execute `my loss`.

        Args:
            y: Test fixture or input parameter `y` (y).
            p: Test fixture or input parameter `p` (p).

        Returns:
            TensorLike: Result from `my_loss`.
        """
        return y - p

    model.compile(optimizer="adam", loss=my_loss, metrics=[my_loss])
    assert model.loss_fn == my_loss
    assert model.compiled_metrics[0] == my_loss


def test_fit_evaluate_predict_callbacks() -> None:
    """Verify or execute `test fit evaluate predict callbacks`."""
    model = Sequential([Dense(1)])

    def dummy_loss(y: TensorLike, p: TensorLike) -> TensorLike:
        """Verify or execute `dummy loss`.

        Args:
            y: Test fixture or input parameter `y` (y).
            p: Test fixture or input parameter `p` (p).

        Returns:
            TensorLike: Result from `dummy_loss`.
        """
        return p

    model.compile(optimizer="sgd", loss=dummy_loss)
    x = random.uniform((10, 2))
    y = random.uniform((10, 1))

    cb = CountingCallback()

    # Fit
    hist = model.fit(x, y, batch_size=5, epochs=2, callbacks=[cb])
    assert cb.counts["on_train_begin"] == 1
    assert cb.counts["on_epoch_begin"] == 2
    assert cb.counts["on_train_batch_begin"] == 4
    assert cb.counts["on_batch_begin"] == 4
    assert cb.counts["on_train_batch_end"] == 4
    assert cb.counts["on_batch_end"] == 4
    assert cb.counts["on_epoch_end"] == 2
    assert cb.counts["on_train_end"] == 1
    assert "loss" in hist.history

    # Evaluate
    model.evaluate(x, y, batch_size=5, callbacks=[cb])
    assert cb.counts["on_test_begin"] == 1
    assert cb.counts["on_test_batch_begin"] == 2
    assert cb.counts["on_test_batch_end"] == 2
    assert cb.counts["on_test_end"] == 1

    # Predict
    model.predict(x, batch_size=5, callbacks=[cb])
    assert cb.counts["on_predict_begin"] == 1
    assert cb.counts["on_predict_batch_begin"] == 2
    assert cb.counts["on_predict_batch_end"] == 2
    assert cb.counts["on_predict_end"] == 1


def test_fit_evaluate_predict_iterator_callbacks() -> None:
    """Verify or execute `test fit evaluate predict iterator callbacks`."""
    model = Sequential([Dense(1)])

    def dummy_loss(y: TensorLike, p: TensorLike) -> TensorLike:
        """Verify or execute `dummy loss`.

        Args:
            y: Test fixture or input parameter `y` (y).
            p: Test fixture or input parameter `p` (p).

        Returns:
            TensorLike: Result from `dummy_loss`.
        """
        return p

    model.compile(optimizer="sgd", loss=dummy_loss)

    # Create simple generator
    def data_gen() -> TensorLike:
        """Verify or execute `data gen`."""
        for _ in range(2):
            yield random.uniform((5, 2)), random.uniform((5, 1))

    # Fake TF/Torch dataset by patching __module__ checks via a mock wrapper
    class MockIterator:
        """Test helper class `MockIterator` for core workflows."""

        def __init__(self) -> None:
            """Verify or execute `init` on `MockIterator`."""
            self.gen = data_gen()

        def __iter__(self) -> TensorLike:
            """Verify or execute `iter` on `MockIterator`.

            Returns:
                TensorLike: Result from `__iter__`.
            """
            return self

        def __next__(self) -> TensorLike:
            """Verify or execute `next` on `MockIterator`.

            Returns:
                TensorLike: Result from `__next__`.
            """
            return next(self.gen)

    # Manually bypass _is_iterator check if needed, or rely on python iterator detection
    x = MockIterator()

    cb = CountingCallback()
    model.fit(x, epochs=1, callbacks=[cb])
    assert cb.counts["on_train_begin"] == 1
    assert cb.counts["on_train_batch_begin"] == 2

    x_eval = MockIterator()
    model.evaluate(x_eval, callbacks=[cb])
    assert cb.counts["on_test_begin"] == 1
    assert cb.counts["on_test_batch_begin"] == 2

    x_pred = MockIterator()
    model.predict(x_pred, callbacks=[cb])
    assert cb.counts["on_predict_begin"] == 1
    assert cb.counts["on_predict_batch_begin"] == 2


def test_save_and_load_weights() -> None:
    """Verify or execute `test save and load weights`."""
    model = Sequential([Dense(1)])
    model.build((None, 2))

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.keras")
        model.save_weights(path)
        assert os.path.exists(path)

        w = model.weights[0]

        model.load_weights(path)
        w_new = model.weights[0]
        assert_allclose_zero(w, w_new)


def test_stop_training() -> None:
    """Verify or execute `test stop training`."""
    model = Sequential([Dense(1)])

    def dummy_loss(y: TensorLike, p: TensorLike) -> TensorLike:
        """Verify or execute `dummy loss`.

        Args:
            y: Test fixture or input parameter `y` (y).
            p: Test fixture or input parameter `p` (p).

        Returns:
            TensorLike: Result from `dummy_loss`.
        """
        return p

    model.compile(optimizer="sgd", loss=dummy_loss)
    x = random.uniform((10, 2))
    y = random.uniform((10, 1))

    class StopCb(Callback):
        """Test helper class `StopCb` for core workflows."""

        def on_epoch_end(self, epoch: TensorLike, logs: TensorLike | None = None) -> None:
            """Verify or execute `on epoch end` on `StopCb`.

            Args:
                epoch: Test fixture or input parameter `epoch` (epoch).
                logs: Test fixture or input parameter `logs` (logs).
            """
            self.model.stop_training = True

    hist = model.fit(x, y, epochs=10, callbacks=[StopCb()])
    assert len(hist.epoch) == 1
