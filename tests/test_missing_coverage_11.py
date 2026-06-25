from zero_keras.callbacks import Callback, History, CallbackList
from zero_keras.constraints import MaxNorm, MinMaxNorm, UnitNorm, Constraint
import numpy as np


def test_callback_model_setter():
    cb = Callback()
    cb.model = "model"
    assert cb._model == "model"

    hist = History()
    hist.model = "model2"
    assert hist._model == "model2"

    cblist = CallbackList([cb])
    cblist.model = "model3"


def test_history_methods():
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
    hist.on_train_batch_end(0)
    hist.on_train_begin()
    hist.on_epoch_end(0)


def test_constraints_methods():
    c = Constraint()
    c(np.array([1]))

    MaxNorm()
    MinMaxNorm()
    UnitNorm()
