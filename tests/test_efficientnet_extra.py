"""Module `test_efficientnet_extra` for zero_keras."""

from __future__ import annotations
from zero_keras.types import TensorLike
import pytest

from zero_keras.applications.efficientnet import EfficientNet
from zero_keras.applications.efficientnet_v2 import EfficientNetV2


def test_efficientnet_extra_coverage() -> TensorLike:
    # EfficientNet custom args
    """Execute ``test_efficientnet_extra_coverage``."""
    model = EfficientNet(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=224,
        dropout_rate=0.0,
        drop_connect_rate=0.0,
        depth_divisor=8,
        activation="swish",
        blocks_args=[
            {
                "kernel_size": 3,
                "repeats": 1,
                "filters_in": 32,
                "filters_out": 32,  # same so id_skip triggers
                "expand_ratio": 1,
                "id_skip": True,
                "strides": 1,
                "se_ratio": 0.0,
            }
        ],
        name="efficientnet",
        include_top=True,
        weights=None,
        input_shape=(224, 224, 3),
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )

    # EfficientNetV2 custom args
    model_v2 = EfficientNetV2(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=224,
        dropout_rate=0.0,
        drop_connect_rate=0.0,
        depth_divisor=8,
        min_depth=8,
        activation="swish",
        blocks_args=[
            {
                "kernel_size": 3,
                "num_repeat": 1,
                "input_filters": 32,
                "output_filters": 32,
                "expand_ratio": 1,
                "strides": 1,
                "se_ratio": 0.0,
                "conv_type": 1,
            },
            {
                "kernel_size": 3,
                "num_repeat": 1,
                "input_filters": 32,
                "output_filters": 32,
                "expand_ratio": 1,
                "strides": 1,
                "se_ratio": 0.0,
                "conv_type": 0,
            },
        ],
        name="efficientnetv2",
        include_preprocessing=False,
        include_top=True,
        weights=None,
        input_shape=(224, 224, 3),
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )


if __name__ == "__main__":
    pytest.main(["-v", "test_efficientnet_extra.py"])
