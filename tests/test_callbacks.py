"""Module docstring."""

from zero_keras.callbacks import Callback, EarlyStopping, ModelCheckpoint


class MockModel:
    """Class docstring."""

    def __init__(self):
        """Function docstring."""
        self.stop_training = False

        class W:
            """Class docstring."""

            data = 0

        self.weights = [W()]

    def save(self, filepath):
        """Function docstring.

        Args:
            filepath: Description.
        """
        pass


def test_callbacks():
    """Function docstring."""
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
    es2.on_epoch_end(0, {"val_loss": 0.5})  # improves (wait=0) -> hits 282->exit
    es2.on_epoch_end(1, {"val_loss": 0.6})  # no improve (wait=1) -> hits 286->exit
    es2.on_epoch_end(
        2, {"val_loss": 0.7}
    )  # no improve (wait=2) -> stops, hits 289->exit

    mc = ModelCheckpoint(
        filepath="test.h5", monitor="val_loss", save_best_only=True, mode="min"
    )
    mc2 = ModelCheckpoint(
        filepath="test2.h5", monitor="val_loss", save_best_only=False, mode="max"
    )
    mc.set_model(model)
    mc.on_epoch_end(0, {"val_loss": 0.5})
    mc.on_epoch_end(1, {"val_loss": 0.4})
    assert mc.best == 0.4
    mc.on_epoch_end(2, {})  # current is None
    mc.on_epoch_end(3, {"val_loss": 0.6})  # current not improved, save_model=False

    mc2.set_model(model)
    mc2.on_epoch_end(0, {"val_loss": 0.5})
    mc2.on_epoch_end(1, {"val_loss": 0.6})
    assert mc2.best == 0.6
