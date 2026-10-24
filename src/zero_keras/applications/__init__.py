"""Pre-trained vision application models for zero_keras."""

from __future__ import annotations

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
    resnet50,
    resnet_v2,
    vgg16,
    vgg19,
    xception,
)
from zero_keras.applications.convnext import (
    ConvNeXtBase as ConvNeXtBase,
    ConvNeXtLarge as ConvNeXtLarge,
    ConvNeXtSmall as ConvNeXtSmall,
    ConvNeXtTiny as ConvNeXtTiny,
    ConvNeXtXLarge as ConvNeXtXLarge,
)
from zero_keras.applications.densenet import (
    DenseNet121 as DenseNet121,
    DenseNet169 as DenseNet169,
    DenseNet201 as DenseNet201,
)
from zero_keras.applications.efficientnet import (
    EfficientNetB0 as EfficientNetB0,
    EfficientNetB1 as EfficientNetB1,
    EfficientNetB2 as EfficientNetB2,
    EfficientNetB3 as EfficientNetB3,
    EfficientNetB4 as EfficientNetB4,
    EfficientNetB5 as EfficientNetB5,
    EfficientNetB6 as EfficientNetB6,
    EfficientNetB7 as EfficientNetB7,
)
from zero_keras.applications.efficientnet_v2 import (
    EfficientNetV2B0 as EfficientNetV2B0,
    EfficientNetV2B1 as EfficientNetV2B1,
    EfficientNetV2B2 as EfficientNetV2B2,
    EfficientNetV2B3 as EfficientNetV2B3,
    EfficientNetV2L as EfficientNetV2L,
    EfficientNetV2M as EfficientNetV2M,
    EfficientNetV2S as EfficientNetV2S,
)
from zero_keras.applications.imagenet_utils import (
    decode_predictions as decode_predictions,
    preprocess_input as preprocess_input,
)
from zero_keras.applications.inception_resnet_v2 import (
    InceptionResNetV2 as InceptionResNetV2,
)
from zero_keras.applications.inception_v3 import InceptionV3 as InceptionV3
from zero_keras.applications.mobilenet import MobileNet as MobileNet
from zero_keras.applications.mobilenet_v2 import MobileNetV2 as MobileNetV2
from zero_keras.applications.mobilenet_v3 import (
    MobileNetV3Large as MobileNetV3Large,
    MobileNetV3Small as MobileNetV3Small,
)
from zero_keras.applications.nasnet import (
    NASNetLarge as NASNetLarge,
    NASNetMobile as NASNetMobile,
)
from zero_keras.applications.resnet import (
    ResNet50 as ResNet50,
    ResNet101 as ResNet101,
    ResNet152 as ResNet152,
)
from zero_keras.applications.resnet_v2 import (
    ResNet50V2 as ResNet50V2,
    ResNet101V2 as ResNet101V2,
    ResNet152V2 as ResNet152V2,
)
from zero_keras.applications.vgg16 import VGG16 as VGG16
from zero_keras.applications.vgg19 import VGG19 as VGG19
from zero_keras.applications.xception import Xception as Xception

__all__ = [
    "VGG16",
    "VGG19",
    "ConvNeXtBase",
    "ConvNeXtLarge",
    "ConvNeXtSmall",
    "ConvNeXtTiny",
    "ConvNeXtXLarge",
    "DenseNet121",
    "DenseNet169",
    "DenseNet201",
    "EfficientNetB0",
    "EfficientNetB1",
    "EfficientNetB2",
    "EfficientNetB3",
    "EfficientNetB4",
    "EfficientNetB5",
    "EfficientNetB6",
    "EfficientNetB7",
    "EfficientNetV2B0",
    "EfficientNetV2B1",
    "EfficientNetV2B2",
    "EfficientNetV2B3",
    "EfficientNetV2L",
    "EfficientNetV2M",
    "EfficientNetV2S",
    "InceptionResNetV2",
    "InceptionV3",
    "MobileNet",
    "MobileNetV2",
    "MobileNetV3Large",
    "MobileNetV3Small",
    "NASNetLarge",
    "NASNetMobile",
    "ResNet50",
    "ResNet50V2",
    "ResNet101",
    "ResNet101V2",
    "ResNet152",
    "ResNet152V2",
    "Xception",
    "convnext",
    "decode_predictions",
    "densenet",
    "efficientnet",
    "efficientnet_v2",
    "imagenet_utils",
    "inception_resnet_v2",
    "inception_v3",
    "mobilenet",
    "mobilenet_v2",
    "mobilenet_v3",
    "nasnet",
    "preprocess_input",
    "resnet",
    "resnet50",
    "resnet_v2",
    "vgg16",
    "vgg19",
    "xception",
]
