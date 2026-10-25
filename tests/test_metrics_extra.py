"""Extra unit tests for Metric base class methods and variable tracking."""

from __future__ import annotations


def test_metric_methods_extra() -> None:
    """Test Metric variable tracking, stateless methods, and configuration serialization."""
    from zero_keras.metrics import Metric

    m = Metric(name="test_metric")
    v1 = m.add_variable((1,), "zeros")
    v2 = m.add_weight()
    assert m.dtype is None
    assert m.get_config() == {"name": "test_metric"}
    m2 = Metric.from_config({"name": "test_metric2"})
    assert m2.name == "test_metric2"
    m.stateless_reset_state()
    m.result = lambda: 1
    m.update_state = lambda *args, **kwargs: None
    assert m.stateless_result() == 1
    assert m.stateless_update_state() is None
    assert m.variables == [v1, v2]
    m.weights = [v1]
    assert m.variables == [v1]

    raw_m = Metric.__new__(Metric)
    raw_m.add_variable((1,), "zeros")
    raw_m.add_weight()
