"""callbacks API."""

from zero_keras.callbacks.callbacks import BackupAndRestore as BackupAndRestore
from zero_keras.callbacks.callbacks import CSVLogger as CSVLogger
from zero_keras.callbacks.callbacks import Callback as Callback
from zero_keras.callbacks.callbacks import CallbackList as CallbackList
from zero_keras.callbacks.callbacks import EarlyStopping as EarlyStopping
from zero_keras.callbacks.callbacks import History as History
from zero_keras.callbacks.callbacks import LambdaCallback as LambdaCallback
from zero_keras.callbacks.callbacks import (
    LearningRateScheduler as LearningRateScheduler,
)
from zero_keras.callbacks.callbacks import ModelCheckpoint as ModelCheckpoint
from zero_keras.callbacks.callbacks import ProgbarLogger as ProgbarLogger
from zero_keras.callbacks.callbacks import ReduceLROnPlateau as ReduceLROnPlateau
from zero_keras.callbacks.callbacks import RemoteMonitor as RemoteMonitor
from zero_keras.callbacks.callbacks import SwapEMAWeights as SwapEMAWeights
from zero_keras.callbacks.callbacks import TensorBoard as TensorBoard
from zero_keras.callbacks.callbacks import TerminateOnNaN as TerminateOnNaN

__all__ = [
    "BackupAndRestore",
    "CSVLogger",
    "Callback",
    "CallbackList",
    "EarlyStopping",
    "History",
    "LambdaCallback",
    "LearningRateScheduler",
    "ModelCheckpoint",
    "ProgbarLogger",
    "ReduceLROnPlateau",
    "RemoteMonitor",
    "SwapEMAWeights",
    "TensorBoard",
    "TerminateOnNaN",
]
