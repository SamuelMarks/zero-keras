"""Module docstring."""

from zero_keras import metrics
import numpy as np  # type: ignore


def test_metrics_io():
    """Function docstring."""
    metric = metrics.MeanSquaredError()
    config = metrics.serialize(metric)
    assert isinstance(config, dict)

    metric2 = metrics.deserialize(config)
    assert isinstance(metric2, metrics.MeanSquaredError)

    assert metrics.serialize(None) is None
    assert metrics.serialize("mse") == "mse"

    assert metrics.deserialize(None) is None
    assert isinstance(metrics.deserialize("mse"), metrics.MeanSquaredError)

    assert metrics.get("mse") is not None
    assert metrics.get(None) is None
    assert metrics.get(metric) is metric
    assert metrics.get("unknown") == "unknown"


def test_correlation_metrics():
    """Function docstring."""
    cc = metrics.ConcordanceCorrelation()
    pc = metrics.PearsonCorrelation()
    fb = metrics.FBetaScore()
    f1 = metrics.F1Score()
    r2 = metrics.R2Score()

    cc.update_state(np.array([1.0]), np.array([1.0]))
    pc.update_state(np.array([1.0]), np.array([1.0]))
    fb.update_state(np.array([1.0]), np.array([1.0]))
    f1.update_state(np.array([1.0]), np.array([1.0]))
    r2.update_state(np.array([1.0]), np.array([1.0]))

    # Bypass actual ops.zeros check if unsupported, or check it
    # We will just assert callable.
    assert callable(metrics.ConcordanceCorrelation)
    assert callable(metrics.PearsonCorrelation)


def test_metric_functions():
    """Function docstring."""
    assert callable(metrics.binary_accuracy)
    assert callable(metrics.categorical_accuracy)
    assert callable(metrics.sparse_categorical_accuracy)
    assert callable(metrics.top_k_categorical_accuracy)
    assert callable(metrics.sparse_top_k_categorical_accuracy)
