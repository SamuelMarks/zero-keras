import pytest

from zero_keras.applications.nasnet import NASNetLarge, NASNetMobile


def test_nasnet_extra():
    from zero_keras import backend

    backend.set_image_data_format("channels_last")
    # NASNetMobile with include_top=False, custom input_shape
    model_mobile = NASNetMobile(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None,
        input_tensor=None,
        pooling="avg",
        classes=1000,
    )
    assert model_mobile.name.startswith("nasnet")

    # NASNetLarge with custom pooling="max"
    model_large = NASNetLarge(
        input_shape=(331, 331, 3),
        include_top=False,
        weights=None,
        input_tensor=None,
        pooling="max",
        classes=1000,
    )
    assert model_large.name.startswith("nasnet")


if __name__ == "__main__":
    pytest.main(["-v", "test_nasnet_extra.py"])
