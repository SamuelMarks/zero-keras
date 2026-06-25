def test_metric_methods_extra():
    from zero_keras.metrics import Metric

    m = Metric(name="test_metric")
    m.add_variable((1,), "zeros")
    m.add_weight()
    assert m.dtype is None
    assert m.get_config() == {"name": "test_metric"}
    m2 = Metric.from_config({"name": "test_metric2"})
    assert m2.name == "test_metric2"
    m.stateless_reset_state()
    # Mock result and update_state
    m.result = lambda: 1
    m.update_state = lambda *args, **kwargs: 2
    assert m.stateless_result() == 1
    assert m.stateless_update_state() == 2
    assert m.variables == []
