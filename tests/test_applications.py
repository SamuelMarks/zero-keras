"""Unit and behavioral tests for applications in zero_keras."""

from __future__ import annotations

from pathlib import Path

import pytest

from zero_keras import applications, random


def test_applications() -> None:
    """Test ``applications`` behavior and edge cases in ``test_applications``."""
    funcs = [
        applications.convnext.ConvNeXtTiny,
        applications.convnext.ConvNeXtSmall,
        applications.convnext.ConvNeXtBase,
        applications.convnext.ConvNeXtLarge,
        applications.convnext.ConvNeXtXLarge,
        applications.densenet.DenseNet121,
        applications.densenet.DenseNet169,
        applications.densenet.DenseNet201,
        applications.efficientnet.EfficientNetB0,
        applications.efficientnet.EfficientNetB1,
        applications.efficientnet.EfficientNetB2,
        applications.efficientnet.EfficientNetB3,
        applications.efficientnet.EfficientNetB4,
        applications.efficientnet.EfficientNetB5,
        applications.efficientnet.EfficientNetB6,
        applications.efficientnet.EfficientNetB7,
        applications.efficientnet_v2.EfficientNetV2B0,
        applications.efficientnet_v2.EfficientNetV2B1,
        applications.efficientnet_v2.EfficientNetV2B2,
        applications.efficientnet_v2.EfficientNetV2B3,
        applications.efficientnet_v2.EfficientNetV2S,
        applications.efficientnet_v2.EfficientNetV2M,
        applications.efficientnet_v2.EfficientNetV2L,
        applications.inception_resnet_v2.InceptionResNetV2,
        applications.inception_v3.InceptionV3,
        applications.mobilenet.MobileNet,
        applications.mobilenet_v2.MobileNetV2,
        applications.mobilenet_v3.MobileNetV3Small,
        applications.mobilenet_v3.MobileNetV3Large,
        applications.nasnet.NASNetMobile,
        applications.nasnet.NASNetLarge,
        applications.resnet.ResNet50,
        applications.resnet.ResNet101,
        applications.resnet.ResNet152,
        applications.resnet_v2.ResNet50V2,
        applications.resnet_v2.ResNet101V2,
        applications.resnet_v2.ResNet152V2,
        applications.vgg16.VGG16,
        applications.vgg19.VGG19,
        applications.xception.Xception,
    ]
    for fn in funcs:
        fn(weights=None)


def test_resnet50_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify resnet50.ResNet50, preprocess_input, and decode_predictions.

    Args:
        tmp_path: Pytest temporary directory path fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    import json

    from zero_keras import utils
    from zero_keras.applications import resnet50

    idx_file = tmp_path / "imagenet_class_index.json"
    idx_file.write_text(
        json.dumps({str(i): [f"n{i:08d}", f"class_{i}"] for i in range(1000)}),
        encoding="utf-8",
    )
    monkeypatch.setattr(utils, "get_file", lambda *a, **kw: str(idx_file))

    assert resnet50.ResNet50(weights=None) is not None
    decoded = resnet50.decode_predictions(random.uniform((1, 1000)), top=3)
    assert len(decoded) == 1 and len(decoded[0]) == 3
    pre = resnet50.preprocess_input(random.uniform((1, 224, 224, 3)))
    assert pre.shape == (1, 224, 224, 3)
