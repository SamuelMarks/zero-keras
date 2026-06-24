"""applications API."""

from zero_keras.applications.convnext import ConvNeXtBase as ConvNeXtBase
from zero_keras.applications.convnext import ConvNeXtLarge as ConvNeXtLarge
from zero_keras.applications.convnext import ConvNeXtSmall as ConvNeXtSmall
from zero_keras.applications.convnext import ConvNeXtTiny as ConvNeXtTiny
from zero_keras.applications.convnext import ConvNeXtXLarge as ConvNeXtXLarge
from zero_keras.applications.densenet import DenseNet121 as DenseNet121
from zero_keras.applications.densenet import DenseNet169 as DenseNet169
from zero_keras.applications.densenet import DenseNet201 as DenseNet201
from zero_keras.applications.efficientnet import EfficientNetB0 as EfficientNetB0
from zero_keras.applications.efficientnet import EfficientNetB1 as EfficientNetB1
from zero_keras.applications.efficientnet import EfficientNetB2 as EfficientNetB2
from zero_keras.applications.efficientnet import EfficientNetB3 as EfficientNetB3
from zero_keras.applications.efficientnet import EfficientNetB4 as EfficientNetB4
from zero_keras.applications.efficientnet import EfficientNetB5 as EfficientNetB5
from zero_keras.applications.efficientnet import EfficientNetB6 as EfficientNetB6
from zero_keras.applications.efficientnet import EfficientNetB7 as EfficientNetB7
from zero_keras.applications.efficientnet_v2 import EfficientNetV2B0 as EfficientNetV2B0
from zero_keras.applications.efficientnet_v2 import EfficientNetV2B1 as EfficientNetV2B1
from zero_keras.applications.efficientnet_v2 import EfficientNetV2B2 as EfficientNetV2B2
from zero_keras.applications.efficientnet_v2 import EfficientNetV2B3 as EfficientNetV2B3
from zero_keras.applications.efficientnet_v2 import EfficientNetV2L as EfficientNetV2L
from zero_keras.applications.efficientnet_v2 import EfficientNetV2M as EfficientNetV2M
from zero_keras.applications.efficientnet_v2 import EfficientNetV2S as EfficientNetV2S
from zero_keras.applications.inception_resnet_v2 import (
    InceptionResNetV2 as InceptionResNetV2,
)
from zero_keras.applications.inception_v3 import InceptionV3 as InceptionV3
from zero_keras.applications.mobilenet import MobileNet as MobileNet
from zero_keras.applications.mobilenet_v2 import MobileNetV2 as MobileNetV2
from zero_keras.applications.mobilenet_v3 import MobileNetV3Large as MobileNetV3Large
from zero_keras.applications.mobilenet_v3 import MobileNetV3Small as MobileNetV3Small
from zero_keras.applications.nasnet import NASNetLarge as NASNetLarge
from zero_keras.applications.nasnet import NASNetMobile as NASNetMobile
from zero_keras.applications.resnet import ResNet101 as ResNet101
from zero_keras.applications.resnet_v2 import ResNet101V2 as ResNet101V2
from zero_keras.applications.resnet import ResNet152 as ResNet152
from zero_keras.applications.resnet_v2 import ResNet152V2 as ResNet152V2
from zero_keras.applications.resnet import ResNet50 as ResNet50
from zero_keras.applications.resnet_v2 import ResNet50V2 as ResNet50V2
from zero_keras.applications.vgg16 import VGG16 as VGG16
from zero_keras.applications.vgg19 import VGG19 as VGG19
from zero_keras.applications.xception import Xception as Xception
from zero_keras.applications.convnext import *  # noqa: F403
from zero_keras.applications.densenet import *  # noqa: F403
from zero_keras.applications.efficientnet import *  # noqa: F403
from zero_keras.applications.efficientnet_v2 import *  # noqa: F403
from zero_keras.applications.imagenet_utils import *  # noqa: F403
from zero_keras.applications.inception_resnet_v2 import *  # noqa: F403
from zero_keras.applications.inception_v3 import *  # noqa: F403
from zero_keras.applications.mobilenet import *  # noqa: F403
from zero_keras.applications.mobilenet_v2 import *  # noqa: F403
from zero_keras.applications.mobilenet_v3 import *  # noqa: F403
from zero_keras.applications.nasnet import *  # noqa: F403
from zero_keras.applications.resnet import *  # noqa: F403
from zero_keras.applications.resnet50 import *  # noqa: F403
from zero_keras.applications.resnet_v2 import *  # noqa: F403
from zero_keras.applications.vgg16 import *  # noqa: F403
from zero_keras.applications.vgg19 import *  # noqa: F403
from zero_keras.applications.xception import *  # noqa: F403

__all__ = [
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
    "ResNet101",
    "ResNet101V2",
    "ResNet152",
    "ResNet152V2",
    "ResNet50",
    "ResNet50V2",
    "VGG16",
    "VGG19",
    "Xception",
]
