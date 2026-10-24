"""Exhaustive unit and branch coverage tests for ``zero_keras.applications``."""

from __future__ import annotations

from pathlib import Path
from unittest import mock
import warnings

import pytest

from zero_keras import backend, layers, models, ops
from zero_keras.applications import (
    convnext,
    densenet,
    efficientnet,
    efficientnet_v2,
    imagenet_utils,
    inception_resnet_v2,
    inception_v3,
    mobilenet,
    mobilenet_v2,
    mobilenet_v3,
    nasnet,
    resnet,
    resnet_v2,
    vgg16,
    vgg19,
    xception,
)


def test_imagenet_utils_exhaustive() -> None:
    """Cover 100% of lines and branches in ``imagenet_utils.py``."""
    with pytest.raises(ValueError):
        imagenet_utils.preprocess_input(ops.zeros((1, 4, 4, 3)), mode="bad_mode")
    with pytest.raises(ValueError):
        imagenet_utils.preprocess_input(ops.zeros((1, 4, 4, 3)), data_format="bad_fmt")

    for mode in ("caffe", "tf", "torch"):
        for df, shp3, shp4 in [
            ("channels_last", (4, 4, 3), (1, 4, 4, 3)),
            ("channels_first", (3, 4, 4), (1, 3, 4, 4)),
        ]:
            assert (
                imagenet_utils.preprocess_input(ops.ones(shp3), data_format=df, mode=mode)
                is not None
            )
            assert (
                imagenet_utils.preprocess_input(ops.ones(shp4), data_format=df, mode=mode)
                is not None
            )
    assert (
        imagenet_utils.preprocess_input(ops.ones((1, 4, 4, 3)), data_format=None, mode="tf")
        is not None
    )

    with pytest.raises(ValueError):
        imagenet_utils.decode_predictions(ops.zeros((1, 10)))

    imagenet_utils.CLASS_INDEX = None
    with (
        mock.patch("zero_keras.utils.generic_utils.get_file", return_value="dummy_idx.json"),
        mock.patch(
            "builtins.open", mock.mock_open(read_data='{"0": ["n0", "c0"], "1": ["n1", "c1"]}')
        ),
    ):
        imagenet_utils.CLASS_INDEX = None
        # Fill 1000 entries via json.load mock
        with mock.patch(
            "json.load", return_value={str(i): [f"n{i}", f"cls_{i}"] for i in range(1000)}
        ):
            preds = ops.zeros((1, 1000))
            decoded = imagenet_utils.decode_predictions(preds, top=2)
            assert len(decoded) == 1 and len(decoded[0]) == 2

    # obtain_input_shape branches
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert imagenet_utils.obtain_input_shape(
            (5, 64, 64), 224, 32, "channels_first", False, weights=None
        ) == (5, 64, 64)
        assert imagenet_utils.obtain_input_shape(
            (3, 64, 64), 224, 32, "channels_first", False, weights=None
        ) == (3, 64, 64)
        assert imagenet_utils.obtain_input_shape(
            (64, 64, 5), 224, 32, "channels_last", False, weights=None
        ) == (64, 64, 5)

    with pytest.raises(ValueError):
        imagenet_utils.obtain_input_shape(
            (128, 128, 3), 224, 32, "channels_last", True, weights="imagenet"
        )
    assert imagenet_utils.obtain_input_shape(
        (224, 224, 3), 224, 32, "channels_last", True, weights="imagenet"
    ) == (224, 224, 3)

    # Invalid input_shape lengths, channels, min_size for channels_first and channels_last
    for df in ("channels_first", "channels_last"):
        with pytest.raises(ValueError):
            imagenet_utils.obtain_input_shape((64, 64), 224, 32, df, False, weights=None)
        bad_ch = (1, 64, 64) if df == "channels_first" else (64, 64, 1)
        with pytest.raises(ValueError):
            imagenet_utils.obtain_input_shape(bad_ch, 224, 32, df, False, weights="imagenet")
        small_shp = (3, 16, 64) if df == "channels_first" else (16, 64, 3)
        with pytest.raises(ValueError):
            imagenet_utils.obtain_input_shape(small_shp, 224, 32, df, False, weights=None)

    assert imagenet_utils.obtain_input_shape(
        None, 224, 32, "channels_first", False, weights=None
    ) == (3, None, None)
    assert imagenet_utils.obtain_input_shape(
        None, 224, 32, "channels_last", False, weights=None
    ) == (None, None, 3)

    with pytest.raises(ValueError):
        imagenet_utils.obtain_input_shape(
            (None, 64, 3), 224, 32, "channels_last", True, weights=None
        )

    # correct_pad with None spatial size
    kt_none = layers.Input(shape=(None, None, 3))
    assert imagenet_utils.correct_pad(kt_none, 3) == ((0, 1), (0, 1))

    # validate_activation
    imagenet_utils.validate_activation("relu", weights=None)
    imagenet_utils.validate_activation("softmax", weights="imagenet")
    with pytest.raises(ValueError):
        imagenet_utils.validate_activation("relu", weights="imagenet")


def test_standard_applications_branches(tmp_path: Path) -> None:
    """Cover common branches (`weights`, `include_top`, `pooling`, `input_tensor`, `preprocess_input`, `decode_predictions`) across standard application modules.

    Args:
        tmp_path: Temporary directory fixture.
    """
    imagenet_utils.CLASS_INDEX = {str(i): [f"n{i}", f"cls_{i}"] for i in range(1000)}
    fake_weights = str(tmp_path / "dummy.weights.h5")
    Path(fake_weights).write_text("dummy", encoding="utf-8")

    kt_in = layers.Input(shape=(75, 75, 3))
    raw_in = ops.zeros((1, 75, 75, 3))
    preds = ops.zeros((1, 1000))

    app_specs = [
        (vgg16, vgg16.VGG16, (64, 64, 3)),
        (vgg19, vgg19.VGG19, (64, 64, 3)),
        (xception, xception.Xception, (75, 75, 3)),
        (inception_v3, inception_v3.InceptionV3, (75, 75, 3)),
        (inception_resnet_v2, inception_resnet_v2.InceptionResNetV2, (75, 75, 3)),
        (densenet, densenet.DenseNet121, (64, 64, 3)),
        (resnet, resnet.ResNet50, (64, 64, 3)),
        (efficientnet, efficientnet.EfficientNetB0, (64, 64, 3)),
    ]

    with (
        mock.patch("zero_keras.utils.generic_utils.get_file", return_value=fake_weights),
        mock.patch.object(models.Functional, "load_weights"),
    ):
        for mod, builder, shp in app_specs:
            # 1. Invalid weights
            with pytest.raises(ValueError):
                builder(weights="invalid_weights_path")
            # 2. imagenet + include_top=True + classes != 1000
            with pytest.raises(ValueError):
                builder(weights="imagenet", include_top=True, classes=10)
            # 3. weights="imagenet", include_top=True
            builder(weights="imagenet", include_top=True)
            # 4. weights="imagenet", include_top=False, pooling="avg" + input_tensor (KerasTensor)
            kt_shp = layers.Input(shape=shp)
            builder(
                weights="imagenet",
                include_top=False,
                pooling="avg",
                input_tensor=kt_shp,
                input_shape=shp,
            )
            # 5. weights=fake_weights, include_top=False, pooling="max" + non-KerasTensor input_tensor
            builder(
                weights=fake_weights,
                include_top=False,
                pooling="max",
                input_tensor=ops.zeros((1, *shp)),
                input_shape=shp,
            )
            # 6. preprocess_input & decode_predictions
            assert mod.preprocess_input(raw_in) is not None
            assert mod.decode_predictions(preds, top=1) is not None

        # Also cover DenseNet169 and DenseNet201 imagenet weight branches and invalid blocks/channels_first
        densenet.DenseNet169(weights="imagenet", include_top=True)
        densenet.DenseNet169(weights="imagenet", include_top=False)
        densenet.DenseNet201(weights="imagenet", include_top=True)
        densenet.DenseNet201(weights="imagenet", include_top=False)
        backend.set_image_data_format("channels_first")
        with pytest.raises(ValueError):
            densenet.DenseNet([1, 1, 1, 1], weights=None)
        backend.set_image_data_format("channels_last")
        with pytest.raises(ValueError):
            densenet.DenseNet([1, 1, 1, 1], weights="imagenet", include_top=True)
        with pytest.raises(ValueError):
            densenet.DenseNet([1, 1, 1, 1], weights="imagenet", include_top=False)

        # inception_resnet_v2 CustomScaleLayer.get_config & invalid block_type
        csl = inception_resnet_v2.CustomScaleLayer(scale=0.5)
        assert csl.get_config()["scale"] == 0.5
        with pytest.raises(ValueError):
            inception_resnet_v2.inception_resnet_block(layers.Input((8, 8, 4)), 0.1, "bad_type", 0)

        # inception_v3 conv2d_bn with name and channels_first
        assert (
            inception_v3.conv2d_bn(layers.Input((8, 8, 3)), 4, 1, 1, name="named_c2d") is not None
        )
        backend.set_image_data_format("channels_first")
        inception_v3.InceptionV3(weights=None, include_top=False, input_shape=(3, 75, 75))
        resnet.ResNet50(weights=None, include_top=False, input_shape=(3, 64, 64))
        with pytest.raises(ValueError):
            mobilenet_v2.MobileNetV2(
                weights=None,
                include_top=False,
                input_shape=(3, 64, 64),
                input_tensor=layers.Input(batch_shape=(None, 3, 32, 32)),
            )
        mobilenet_v2.MobileNetV2(
            weights=None,
            include_top=False,
            input_tensor=layers.Input(batch_shape=(None, 3, 96, 96)),
        )
        mobilenet_v2.MobileNetV2(
            weights=None,
            include_top=False,
            input_shape=(3, 96, 96),
            input_tensor=layers.Input(batch_shape=(None, 96, 96, 3)),
        )
        mobilenet_v2.MobileNetV2(
            weights=None,
            include_top=False,
            input_shape=(3, 96, 96),
        )
        with pytest.raises(ValueError):
            mobilenet_v3.MobileNetV3Small(
                weights=None,
                include_top=False,
                input_shape=(3, 64, 64),
                input_tensor=layers.Input(batch_shape=(None, 3, 32, 32)),
            )
        mobilenet_v3.MobileNetV3Small(
            weights=None,
            include_top=False,
            input_tensor=layers.Input(batch_shape=(None, 3, 64, 64)),
        )
        mobilenet_v3.MobileNetV3Small(
            weights=None,
            include_top=False,
            input_shape=(3, 64, 64),
            input_tensor=layers.Input(batch_shape=(None, 3, 64, 64)),
        )
        mobilenet_v3.MobileNetV3Small(
            weights=None, include_top=False, input_shape=None, input_tensor=None
        )
        assert (
            resnet.residual_block_v2(
                layers.Input(batch_shape=(None, 4, 8, 8)), 1, conv_shortcut=True, name="r2_cf"
            )
            is not None
        )
        mobilenet.MobileNet(
            weights=None,
            include_top=False,
            input_shape=(3, 128, 128),
        )
        backend.set_image_data_format("channels_last")
        mobilenet_v3.MobileNetV3Small(
            weights=None, include_top=False, input_shape=None, input_tensor=None
        )
        assert mobilenet_v3.preprocess_input(raw_in) is not None
        resnet.ResNet(
            stack_fn=lambda x: x,
            preact=False,
            use_bias=True,
            weights=None,
            include_top=False,
            input_shape=(64, 64, 3),
        )
        assert (
            resnet.residual_block_v2(layers.Input((8, 8, 4)), 1, conv_shortcut=True, name="r2")
            is not None
        )

        # resnet_v2 preprocess_input & decode_predictions
        assert resnet_v2.preprocess_input(raw_in) is not None
        assert resnet_v2.decode_predictions(preds, top=1) is not None

        # Also test pooling=None when include_top=False for vgg16, vgg19, xception, inception_v3, inception_resnet_v2, resnet, efficientnet
        for _mod, builder, shp in app_specs:
            builder(weights=None, include_top=False, pooling=None, input_shape=shp)


def test_convnext_efficientnetv2_mobilenet_nasnet_exhaustive(tmp_path: Path) -> None:
    """Cover 100% of lines and branches in ConvNeXt, EfficientNetV2, MobileNet, MobileNetV2, MobileNetV3, and NASNet.

    Args:
        tmp_path: Temporary directory fixture.
    """
    imagenet_utils.CLASS_INDEX = {str(i): [f"n{i}", f"cls_{i}"] for i in range(1000)}
    fake_weights = str(tmp_path / "dummy.weights.h5")
    Path(fake_weights).write_text("dummy", encoding="utf-8")
    raw_in = ops.zeros((1, 64, 64, 3))
    preds = ops.zeros((1, 1000))

    with (
        mock.patch("zero_keras.utils.generic_utils.get_file", return_value=fake_weights),
        mock.patch.object(models.Functional, "load_weights"),
    ):
        # 1. ConvNeXt
        sd = convnext.StochasticDepth(drop_path_rate=0.2)
        assert sd.call(ops.ones((2, 4, 4, 3)), training=True) is not None
        assert sd.call(ops.ones((2, 4, 4, 3)), training=False) is not None
        assert sd.get_config()["drop_path_rate"] == 0.2

        ls = convnext.LayerScale(init_values=1e-4, projection_dim=3)
        ls.build((2, 4, 4, 3))
        assert ls.call(ops.ones((2, 4, 4, 3))) is not None
        assert ls.get_config()["projection_dim"] == 3

        assert (
            convnext.ConvNeXtBlock(4, drop_path_rate=0.1, name=None)(layers.Input((8, 8, 4)))
            is not None
        )
        assert convnext.PreStem(name=None)(layers.Input((8, 8, 3))) is not None
        assert convnext.Head(num_classes=5, name=None)(layers.Input((8, 8, 4))) is not None

        backend.set_image_data_format("channels_first")
        with pytest.raises(ValueError):
            convnext.ConvNeXtTiny(weights=None)
        backend.set_image_data_format("channels_last")

        with pytest.raises(ValueError):
            convnext.ConvNeXtTiny(weights="bad_w")
        with pytest.raises(ValueError):
            convnext.ConvNeXtTiny(weights="imagenet", include_top=True, classes=5)

        convnext.ConvNeXtTiny(weights="imagenet", include_top=True)
        convnext.ConvNeXtTiny(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_tensor=layers.Input((64, 64, 3)),
            input_shape=(64, 64, 3),
        )
        convnext.ConvNeXtTiny(
            weights=fake_weights,
            include_top=False,
            pooling="max",
            input_tensor=ops.zeros((1, 64, 64, 3)),
            input_shape=(64, 64, 3),
        )
        with pytest.raises(ValueError):
            convnext.ConvNeXt(
                depths=[1, 1, 1, 1],
                projection_dims=[8, 8, 8, 8],
                weights="imagenet",
                weights_name="convnext_tiny",
                name="convnext_tiny",
            )
        with pytest.raises(ValueError):
            convnext.ConvNeXt(
                depths=[3, 3, 9, 3],
                projection_dims=[96, 192, 384, 768],
                weights="imagenet",
                weights_name="convnext_tiny",
                name="wrong_name",
            )
        assert convnext.preprocess_input(raw_in) is not None
        assert convnext.decode_predictions(preds, top=1) is not None

        # 2. EfficientNet & EfficientNetV2
        with pytest.raises(ValueError):
            efficientnet.EfficientNet(
                1.0,
                1.0,
                64,
                blocks_args=[
                    {
                        "repeats": 0,
                        "kernel_size": 3,
                        "filters_in": 8,
                        "filters_out": 8,
                        "expand_ratio": 1,
                        "id_skip": True,
                        "strides": 1,
                        "se_ratio": 0.0,
                    }
                ],
                weights=None,
            )
        assert (
            efficientnet_v2.MBConvBlock(8, 8, expand_ratio=2, se_ratio=0.25, name=None)(
                layers.Input((8, 8, 8))
            )
            is not None
        )
        assert (
            efficientnet_v2.FusedMBConvBlock(8, 8, expand_ratio=2, se_ratio=0.25, name=None)(
                layers.Input((8, 8, 8))
            )
            is not None
        )

        with pytest.raises(ValueError):
            efficientnet_v2.EfficientNetV2B0(weights="bad_w")
        with pytest.raises(ValueError):
            efficientnet_v2.EfficientNetV2B0(weights="imagenet", include_top=True, classes=5)
        with pytest.raises(ValueError):
            efficientnet_v2.EfficientNetV2(
                1.0,
                1.0,
                64,
                blocks_args=[
                    {
                        "num_repeat": 0,
                        "kernel_size": 3,
                        "input_filters": 8,
                        "output_filters": 8,
                        "expand_ratio": 1,
                        "strides": 1,
                        "se_ratio": 0.0,
                        "conv_type": 0,
                    }
                ],
                weights=None,
            )

        efficientnet_v2.EfficientNetV2B0(weights="imagenet", include_top=True)
        efficientnet_v2.EfficientNetV2B0(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_tensor=layers.Input((64, 64, 3)),
            input_shape=(64, 64, 3),
        )
        efficientnet_v2.EfficientNetV2B0(
            weights=fake_weights,
            include_top=False,
            pooling="max",
            input_tensor=ops.zeros((1, 64, 64, 3)),
            input_shape=(64, 64, 3),
        )
        efficientnet_v2.EfficientNetV2B0(
            weights=None, include_top=False, pooling=None, input_shape=(64, 64, 3)
        )
        assert efficientnet_v2.preprocess_input(raw_in) is not None
        assert efficientnet_v2.decode_predictions(preds, top=1) is not None

        # 3. MobileNet
        with pytest.raises(ValueError):
            mobilenet.MobileNet(weights="bad_w")
        with pytest.raises(ValueError):
            mobilenet.MobileNet(weights="imagenet", include_top=True, classes=5)
        with pytest.raises(ValueError):
            mobilenet.MobileNet(weights="imagenet", depth_multiplier=2)
        with pytest.raises(ValueError):
            mobilenet.MobileNet(weights="imagenet", alpha=0.1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mobilenet.MobileNet(weights="imagenet", include_top=False, input_shape=(100, 100, 3))
        for a in (1.0, 0.75, 0.50, 0.25):
            mobilenet.MobileNet(weights="imagenet", alpha=a, include_top=True)
        mobilenet.MobileNet(
            weights=fake_weights,
            include_top=False,
            pooling="avg",
            input_tensor=layers.Input((128, 128, 3)),
            input_shape=(128, 128, 3),
        )
        mobilenet.MobileNet(
            weights=None,
            include_top=False,
            pooling="max",
            input_tensor=ops.zeros((1, 128, 128, 3)),
            input_shape=(128, 128, 3),
        )
        mobilenet.MobileNet(
            weights=None,
            include_top=False,
            pooling=None,
            name=None,
            input_shape=(100, 100, 3),
        )
        assert mobilenet.preprocess_input(raw_in) is not None
        assert mobilenet.decode_predictions(preds, top=1) is not None

        # 4. MobileNetV2
        assert mobilenet_v2._make_divisible(7, 8, min_value=None) == 8
        assert mobilenet_v2._make_divisible(9, 8, min_value=8) == 16
        mobilenet_v2.MobileNetV2(
            weights="imagenet",
            include_top=False,
            pooling=None,
            name=None,
            input_tensor=ops.zeros((1, 224, 224, 3)),
        )
        mobilenet_v2.MobileNetV2(
            weights=None,
            include_top=False,
            input_shape=(96, 96, 3),
            input_tensor=layers.Input((96, 96, 3)),
        )
        mobilenet_v3.MobileNetV3Small(
            weights=None,
            include_top=False,
            pooling=None,
            dropout_rate=0.0,
            include_preprocessing=False,
            input_shape=(64, 64, 3),
            input_tensor=layers.Input((64, 64, 3)),
        )
        mobilenet_v3.MobileNetV3Small(
            weights=None,
            include_top=True,
            dropout_rate=0.0,
            input_tensor=ops.zeros((1, 64, 64, 3)),
        )
        assert mobilenet_v3._depth(9, 8) == 16
        assert mobilenet_v3._depth(9, 8, min_value=8) == 16
        assert (
            mobilenet_v3._inverted_res_block(
                layers.Input((8, 8, 16)),
                expansion=1,
                filters=16,
                kernel_size=3,
                stride=1,
                se_ratio=None,
                activation=mobilenet_v3.relu,
                block_id=1,
            )
            is not None
        )
        assert (
            nasnet._adjust_block(layers.Input((8, 8, 4)), layers.Input((8, 8, 4)), 4, "b0")
            is not None
        )
        with pytest.raises(ValueError):
            mobilenet_v2.MobileNetV2(weights="bad_w")
        with pytest.raises(ValueError):
            mobilenet_v2.MobileNetV2(weights="imagenet", include_top=True, classes=5)
        with pytest.raises(ValueError):
            mobilenet_v2.MobileNetV2(weights="imagenet", alpha=0.1)
        with pytest.raises(ValueError):
            mobilenet_v2.MobileNetV2(
                weights=None,
                include_top=False,
                input_shape=(64, 64, 3),
                input_tensor=ops.zeros((1, 64, 64, 3)),
            )
        with pytest.raises(ValueError):
            mobilenet_v2.MobileNetV2(
                weights=None,
                include_top=False,
                input_shape=(64, 64, 3),
                input_tensor=layers.Input((32, 32, 3)),
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mobilenet_v2.MobileNetV2(
                weights="imagenet", include_top=False, input_shape=(100, 100, 3)
            )
        mobilenet_v2.MobileNetV2(weights="imagenet", include_top=True, alpha=1.4)
        mobilenet_v2.MobileNetV2(
            weights=fake_weights,
            include_top=False,
            pooling="avg",
            input_tensor=layers.Input((96, 96, 3)),
        )
        mobilenet_v2.MobileNetV2(
            weights=None,
            include_top=False,
            pooling="max",
            input_tensor=layers.Input((100, 100, 3)),
        )
        assert mobilenet_v2.preprocess_input(raw_in) is not None
        assert mobilenet_v2.decode_predictions(preds, top=1) is not None

        # 5. MobileNetV3
        with pytest.raises(ValueError):
            mobilenet_v3.MobileNetV3Small(weights="bad_w")
        with pytest.raises(ValueError):
            mobilenet_v3.MobileNetV3Small(weights="imagenet", include_top=True, classes=5)
        with pytest.raises(ValueError):
            mobilenet_v3.MobileNetV3Small(weights="imagenet", minimalistic=True, alpha=0.75)
        with pytest.raises(ValueError):
            mobilenet_v3.MobileNetV3Small(weights=None, include_top=False, input_shape=(16, 16, 3))
        with pytest.raises(ValueError):
            mobilenet_v3.MobileNetV3Small(
                weights=None,
                include_top=False,
                input_shape=(64, 64, 3),
                input_tensor=ops.zeros((1, 64, 64, 3)),
            )
        with pytest.raises(ValueError):
            mobilenet_v3.MobileNetV3Small(
                weights=None,
                include_top=False,
                input_shape=(64, 64, 3),
                input_tensor=layers.Input((32, 32, 3)),
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mobilenet_v3.MobileNetV3Small(
                weights="imagenet", include_top=True, minimalistic=True, input_shape=(224, 224, 3)
            )
            mobilenet_v3.MobileNetV3Small(
                weights="imagenet", include_top=False, pooling="avg", input_shape=(64, 64, 3)
            )
        mobilenet_v3.MobileNetV3Large(
            weights=fake_weights,
            include_top=False,
            pooling="max",
            alpha=1.2,
            input_tensor=layers.Input((64, 64, 3)),
        )
        assert mobilenet_v3.preprocess_input(raw_in) is not None
        assert mobilenet_v3.decode_predictions(preds, top=1) is not None

        # 6. NASNet
        backend.set_image_data_format("channels_first")
        with pytest.raises(ValueError):
            nasnet.NASNetMobile(weights=None)
        backend.set_image_data_format("channels_last")

        with pytest.raises(ValueError):
            nasnet.NASNetMobile(weights="bad_w")
        with pytest.raises(ValueError):
            nasnet.NASNetMobile(weights="imagenet", include_top=True, classes=5)
        with pytest.raises(ValueError):
            nasnet.NASNetMobile(weights="imagenet", include_top=False, input_shape=(None, 224, 3))
        with pytest.raises(ValueError):
            nasnet.NASNet(
                input_shape=(64, 64, 3),
                penultimate_filters=100,
                weights=None,
                include_top=False,
            )
        with pytest.raises(ValueError):
            nasnet.NASNet(
                input_shape=(128, 128, 3),
                default_size=128,
                weights="imagenet",
                include_top=False,
            )

        nasnet.NASNetMobile(weights="imagenet", include_top=True)
        nasnet.NASNetMobile(
            weights="imagenet", include_top=False, input_tensor=layers.Input((224, 224, 3))
        )
        nasnet.NASNetLarge(weights="imagenet", include_top=True)
        nasnet.NASNetLarge(
            weights="imagenet",
            include_top=False,
            input_shape=(331, 331, 3),
            input_tensor=ops.zeros((1, 331, 331, 3)),
        )
        nasnet.NASNet(
            input_shape=(64, 64, 3),
            default_size=None,
            num_blocks=1,
            penultimate_filters=96,
            weights=fake_weights,
            include_top=False,
            pooling=None,
        )
        assert nasnet.preprocess_input(raw_in) is not None
        assert nasnet.decode_predictions(preds, top=1) is not None
