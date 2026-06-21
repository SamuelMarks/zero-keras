from zero_keras.initializers import deserialize as deserialize_initializer
from zero_keras.layers import Conv1DTranspose, Conv2DTranspose, Conv3DTranspose
from zero_keras.layers import Input as LayersInput
from zero_keras.layers import deserialize as deserialize_layer
from zero_keras.metrics import _ConfusionMatrixMetric
from zero_keras.models import Sequential, clone_model, model_from_json
from zero_keras.optimizers import LossScaleOptimizer
from zero_keras.regularizers import deserialize as deserialize_regularizer
import numpy as np
from ml_switcheroo_compiler.ops import asarray


def test_initializers_deserialize_invalid():
    res = deserialize_initializer({"class_name": "InvalidClassThatDoesNotExist"})
    assert res == {"class_name": "InvalidClassThatDoesNotExist"}


def test_layers_conv_transpose_channels_first_no_bias():
    # 2793->2796, 3221->3224, 3651->3654
    c1 = Conv1DTranspose(2, 2, data_format="channels_first", use_bias=False)
    c1.build((None, 3, 10))
    c1(asarray(np.ones((1, 3, 10))))

    c2 = Conv2DTranspose(2, 2, data_format="channels_first", use_bias=False)
    c2.build((None, 3, 10, 10))
    c2(asarray(np.ones((1, 3, 10, 10))))

    c3 = Conv3DTranspose(2, 2, data_format="channels_first", use_bias=False)
    c3.build((None, 3, 10, 10, 10))
    c3(asarray(np.ones((1, 3, 10, 10, 10))))


def test_layers_inputlayer_batch_shape_not_none():
    layer = LayersInput(shape=(10,), batch_shape=(32, 10))
    assert layer.batch_shape == (32, 10)


def test_layers_deserialize_invalid():
    res = deserialize_layer({"class_name": "InvalidLayerClassThatDoesNotExist"})
    assert res == {"class_name": "InvalidLayerClassThatDoesNotExist"}


def test_metrics_confusion_matrix_invalid_type():
    m1 = _ConfusionMatrixMetric(metric_type="INVALID")
    try:
        m1.update_state(np.array([[1]]), np.array([[1]]))
    except UnboundLocalError:
        pass

    m2 = _ConfusionMatrixMetric(metric_type="INVALID", thresholds=[0.5])
    try:
        m2.update_state(np.array([[1]]), np.array([[1]]))
    except UnboundLocalError:
        pass


def test_models_clone_model_custom_function():
    model = Sequential()
    clone_model(model, clone_function=lambda x: x)


def test_models_from_json_edges():
    json_str = """{
        "class_name": "Functional",
        "config": {
            "layers": [
                {
                    "class_name": "InputLayer",
                    "name": "input_1",
                    "config": {
                        "batch_input_shape": [32, 10]
                    }
                },
                {
                    "class_name": "InvalidClassXYZ",
                    "name": "invalid_1",
                    "config": {}
                }
            ]
        }
    }"""
    try:
        model_from_json(json_str)
    except Exception:
        pass


class DummyOptWithoutMethods:
    pass


def test_optimizers_lossscale_no_inner_methods():
    opt = LossScaleOptimizer(DummyOptWithoutMethods())
    opt.build([])
    opt.apply_gradients([])


def test_regularizers_deserialize_invalid():
    res = deserialize_regularizer({"class_name": "InvalidRegularizerClassXYZ"})
    assert res == {"class_name": "InvalidRegularizerClassXYZ"}
