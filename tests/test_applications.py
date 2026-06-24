"""Module docstring."""

from zero_keras import applications


def test_applications():
    """Function docstring."""
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
