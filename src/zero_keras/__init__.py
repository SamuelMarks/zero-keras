"""Module docstring."""

import sys

sys.setrecursionlimit(10000)
"""zero_keras API."""

import ml_switcheroo_compiler as msc

msc.core.config.eager_mode = True

from zero_keras.core_layers import KerasTensor, Input, Layer, Model

from zero_keras.core_layers import (
    Function,
    InputSpec,
    Operation,
    Variable,
    DTypePolicy,
    FloatDTypePolicy,
    Quantizer,
    RematScope,
    StatelessScope,
    SymbolicScope,
    device,
    name_scope,
    remat,
    version,
)
from zero_keras.models import Sequential
from zero_keras.initializers import Initializer
from zero_keras.losses import Loss
from zero_keras.metrics import Metric
from zero_keras.optimizers import Optimizer
from zero_keras.regularizers import Regularizer

from zero_keras.ops import ops
from zero_keras import activations
from zero_keras import initializers
from zero_keras import losses
from zero_keras import metrics
from zero_keras import optimizers
from zero_keras import layers
from zero_keras import models
from zero_keras import regularizers
from zero_keras import utils
from zero_keras import datasets
from zero_keras import applications
from zero_keras import callbacks
from zero_keras import distribution
from zero_keras import saving
from zero_keras import export
from zero_keras import dtype_policies
from zero_keras import constraints
from zero_keras import preprocessing
from zero_keras import mixed_precision
from zero_keras import quantizers
from zero_keras import random
from zero_keras import tree
from zero_keras import visualization
from zero_keras import wrappers
from zero_keras import legacy
from zero_keras import backend
from zero_keras import config

__version__ = "3.8.0"

__all__ = [
    "__version__",
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
    "Function",
    "InputSpec",
    "Operation",
    "Variable",
    "DTypePolicy",
    "FloatDTypePolicy",
    "Quantizer",
    "RematScope",
    "Sequential",
    "StatelessScope",
    "SymbolicScope",
    "device",
    "name_scope",
    "remat",
    "version",
    "Initializer",
    "Loss",
    "Metric",
    "Optimizer",
    "Regularizer",
    "datasets",
    "applications",
    "utils",
    "saving",
    "export",
    "visualization",
    "wrappers",
    "backend",
    "tree",
    "random",
    "quantizers",
    "preprocessing",
    "mixed_precision",
    "legacy",
    "dtype_policies",
    "constraints",
    "config",
    "distribution",
    "callbacks",
]
