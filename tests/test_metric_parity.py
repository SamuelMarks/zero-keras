"""Tests for zero_keras metrics parity with Keras."""

import numpy as np
import pytest
import keras
from zero_keras import metrics
from .utils import assert_allclose_keras_zero, set_seed


@pytest.fixture(autouse=True)
def _set_seed():
    set_seed(42)


def check_metric_parity(
    metric_cls,
    keras_cls,
    y_true,
    y_pred,
    y_true2,
    y_pred2,
    atol=1e-5,
    rtol=1e-5,
    **kwargs,
):
    keras_metric = keras_cls(**kwargs)
    keras_metric.update_state(y_true, y_pred)
    keras_out1 = keras_metric.result()
    keras_metric.update_state(y_true2, y_pred2)
    keras_out2 = keras_metric.result()

    zero_metric = metric_cls(**kwargs)
    zero_metric.update_state(y_true, y_pred)
    zero_out1 = zero_metric.result()
    zero_metric.update_state(y_true2, y_pred2)
    zero_out2 = zero_metric.result()

    # Assert values after first batch
    assert_allclose_keras_zero(keras_out1, zero_out1, atol=atol, rtol=rtol)
    # Assert values after second batch (statefulness)
    assert_allclose_keras_zero(keras_out2, zero_out2, atol=atol, rtol=rtol)

    # Test reset_state
    keras_metric.reset_state()
    zero_metric.reset_state()
    keras_metric.update_state(y_true, y_pred)
    zero_metric.update_state(y_true, y_pred)
    assert_allclose_keras_zero(
        keras_metric.result(), zero_metric.result(), atol=atol, rtol=rtol
    )


def test_metric_Mean():
    check_metric_parity(
        metrics.Mean,
        keras.metrics.Mean,
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        np.array([3.0, 4.0]),
    )


def test_metric_Sum():
    check_metric_parity(
        metrics.Sum,
        keras.metrics.Sum,
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        np.array([3.0, 4.0]),
    )


def test_metric_MeanMetricWrapper():
    import keras.losses

    def custom_fn(y_true, y_pred):
        return keras.losses.mean_squared_error(y_true, y_pred)

    check_metric_parity(
        metrics.MeanMetricWrapper,
        keras.metrics.MeanMetricWrapper,
        np.array([1.0, 2.0]),
        np.array([1.1, 1.9]),
        np.array([3.0, 4.0]),
        np.array([3.1, 3.9]),
        fn=custom_fn,
    )


def test_metric_Accuracy():
    check_metric_parity(
        metrics.Accuracy,
        keras.metrics.Accuracy,
        np.array([1, 1, 0, 0]),
        np.array([1, 1, 0, 1]),
        np.array([0, 1, 0, 0]),
        np.array([1, 1, 0, 0]),
    )


def test_metric_BinaryAccuracy():
    check_metric_parity(
        metrics.BinaryAccuracy,
        keras.metrics.BinaryAccuracy,
        np.array([[1], [1], [0], [0]]),
        np.array([[0.9], [0.9], [0.1], [0.9]]),
        np.array([[0], [1], [0], [0]]),
        np.array([[0.9], [0.9], [0.1], [0.1]]),
    )


def test_metric_CategoricalAccuracy():
    check_metric_parity(
        metrics.CategoricalAccuracy,
        keras.metrics.CategoricalAccuracy,
        np.array([[0, 1], [1, 0]]),
        np.array([[0.1, 0.9], [0.9, 0.1]]),
        np.array([[0, 1], [0, 1]]),
        np.array([[0.8, 0.2], [0.1, 0.9]]),
    )


def test_metric_SparseCategoricalAccuracy():
    check_metric_parity(
        metrics.SparseCategoricalAccuracy,
        keras.metrics.SparseCategoricalAccuracy,
        np.array([1, 0], dtype=np.int32),
        np.array([[0.1, 0.9], [0.9, 0.1]]),
        np.array([1, 1]),
        np.array([[0.8, 0.2], [0.1, 0.9]]),
    )


def test_metric_TopKCategoricalAccuracy():
    check_metric_parity(
        metrics.TopKCategoricalAccuracy,
        keras.metrics.TopKCategoricalAccuracy,
        np.array([[0, 1, 0], [1, 0, 0]], dtype=np.int32),
        np.array([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32),
        np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int32),
        np.array([[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]], dtype=np.float32),
    )
