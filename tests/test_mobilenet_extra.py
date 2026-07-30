import pytest

from zero_keras.applications.mobilenet import MobileNet
from zero_keras.applications.mobilenet_v2 import MobileNetV2
from zero_keras.applications.mobilenet_v3 import MobileNetV3Large


def test_mobilenet_extra():
    model = MobileNet(
        input_shape=(224, 224, 3),
        alpha=1.0,
        depth_multiplier=1,
        dropout=0.001,
        include_top=False,
        weights=None,
        input_tensor=None,
        pooling=None,
        name=None,
    )
    assert model.name.startswith("mobilenet")

    model_v2 = MobileNetV2(
        input_shape=(224, 224, 3),
        alpha=1.0,
        include_top=False,
        weights=None,
        input_tensor=None,
        pooling=None,
        name=None,
    )
    assert model_v2.name.startswith("mobilenetv2")

    model_v3 = MobileNetV3Large(
        input_shape=None,
        alpha=1.0,
        minimalistic=False,
        include_top=True,
        weights=None,
        input_tensor=None,
        pooling=None,
        dropout_rate=0.2,
        include_preprocessing=True,
    )
    assert model_v3.name.startswith("MobileNetV3")


if __name__ == "__main__":
    pytest.main(["-v", "test_mobilenet_extra.py"])
