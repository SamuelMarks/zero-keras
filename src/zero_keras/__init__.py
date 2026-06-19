"""zero_keras API."""

import ml_switcheroo_compiler as msc

msc.core.config.eager_mode = True

from zero_keras.core_layers import KerasTensor, Input, Layer, Model, ops
from zero_keras import activations
from zero_keras import initializers
from zero_keras import losses
from zero_keras import metrics
from zero_keras import optimizers
from zero_keras import layers
from zero_keras import models
from zero_keras import regularizers

__all__ = [
    "activations",
    "initializers",
    "losses",
    "metrics",
    "optimizers",
    "layers",
    "models",
    "regularizers",
    "KerasTensor",
    "Input",
    "Layer",
    "Model",
    "ops",
]
