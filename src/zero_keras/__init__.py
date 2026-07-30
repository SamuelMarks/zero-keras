"""Module docstring."""

import sys

sys.setrecursionlimit(10000)
"""zero_keras API."""

from ml_switcheroo_compiler.core.config import config as msc_config

msc_config.eager_mode = True


from zero_keras import (
    activations,
    applications,
    backend,
    callbacks,
    config,
    constraints,
    datasets,
    distribution,
    dtype_policies,
    export,
    initializers,
    layers,
    legacy,
    losses,
    metrics,
    mixed_precision,
    models,
    ops,
    optimizers,
    preprocessing,
    quantizers,
    random,
    regularizers,
    saving,
    tree,
    utils,
    visualization,
    wrappers,
)
from zero_keras.core_layers import (
    DTypePolicy,
    FloatDTypePolicy,
    Function,
    Input,
    InputSpec,
    KerasTensor,
    Layer,
    Model,
    Operation,
    Quantizer,
    RematScope,
    StatelessScope,
    SymbolicScope,
    Variable,
    device,
    name_scope,
    remat,
    version,
)
from zero_keras.initializers import Initializer
from zero_keras.losses import Loss
from zero_keras.metrics import Metric
from zero_keras.models import Sequential
from zero_keras.optimizers import Optimizer
from zero_keras.regularizers import Regularizer

__version__ = "3.8.0"

__all__ = [
    "DTypePolicy",
    "FloatDTypePolicy",
    "Function",
    "Initializer",
    "Input",
    "InputSpec",
    "KerasTensor",
    "Layer",
    "Loss",
    "Metric",
    "Model",
    "Operation",
    "Optimizer",
    "Quantizer",
    "Regularizer",
    "RematScope",
    "Sequential",
    "StatelessScope",
    "SymbolicScope",
    "Variable",
    "__version__",
    "activations",
    "applications",
    "backend",
    "callbacks",
    "config",
    "constraints",
    "datasets",
    "device",
    "distribution",
    "dtype_policies",
    "export",
    "initializers",
    "layers",
    "legacy",
    "losses",
    "metrics",
    "mixed_precision",
    "models",
    "name_scope",
    "ops",
    "optimizers",
    "preprocessing",
    "quantizers",
    "random",
    "regularizers",
    "remat",
    "saving",
    "tree",
    "utils",
    "version",
    "visualization",
    "wrappers",
]
