"""ResNetV2 architectures (ResNet50V2, ResNet101V2, ResNet152V2) for Keras applications."""

from __future__ import annotations
from zero_keras.types import ShapeTuple, TensorLike

from zero_keras.applications import imagenet_utils, resnet


def ResNet50V2(
    include_top: bool = True,
    weights: str | None = "imagenet",
    input_tensor: TensorLike | None = None,
    input_shape: ShapeTuple | TensorLike | None = None,
    pooling: str | None = None,
    classes: int = 1000,
    classifier_activation: str | TensorLike | None = "softmax",
    name: str = "resnet50v2",
) -> TensorLike:
    """Instantiates the ResNet50V2 architecture.

    Args:
        include_top: Whether to include the fully-connected classification layer at the top.
        weights: Pretrained weight specification (``"imagenet"``, ``None``, or file path).
        input_tensor: Optional Keras tensor to use as image input.
        input_shape: Optional input shape tuple when ``include_top=False``.
        pooling: Optional feature pooling mode (``None``, ``"avg"``, or ``"max"``).
        classes: Number of classification output classes.
        classifier_activation: Activation function applied on the top classification layer.
        name: Name of the model instance.

    Returns:
        TensorLike: Instantiated ResNet50V2 Functional model.
    """

    def stack_fn(x: TensorLike) -> TensorLike:
        """Apply the 4-stage ResNet50V2 residual block stack.

        Args:
            x: Input stem feature tensor.

        Returns:
            TensorLike: Output feature tensor after stage 5.
        """
        x = resnet.stack_residual_blocks_v2(x, 64, 3, name="conv2")
        x = resnet.stack_residual_blocks_v2(x, 128, 4, name="conv3")
        x = resnet.stack_residual_blocks_v2(x, 256, 6, name="conv4")
        return resnet.stack_residual_blocks_v2(x, 512, 3, stride1=1, name="conv5")

    return resnet.ResNet(
        stack_fn,
        True,
        True,
        name=name,
        weights_name="resnet50v2",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
    )


def ResNet101V2(
    include_top: bool = True,
    weights: str | None = "imagenet",
    input_tensor: TensorLike | None = None,
    input_shape: ShapeTuple | TensorLike | None = None,
    pooling: str | None = None,
    classes: int = 1000,
    classifier_activation: str | TensorLike | None = "softmax",
    name: str = "resnet101v2",
) -> TensorLike:
    """Instantiates the ResNet101V2 architecture.

    Args:
        include_top: Whether to include the fully-connected classification layer at the top.
        weights: Pretrained weight specification (``"imagenet"``, ``None``, or file path).
        input_tensor: Optional Keras tensor to use as image input.
        input_shape: Optional input shape tuple when ``include_top=False``.
        pooling: Optional feature pooling mode (``None``, ``"avg"``, or ``"max"``).
        classes: Number of classification output classes.
        classifier_activation: Activation function applied on the top classification layer.
        name: Name of the model instance.

    Returns:
        TensorLike: Instantiated ResNet101V2 Functional model.
    """

    def stack_fn(x: TensorLike) -> TensorLike:
        """Apply the 4-stage ResNet101V2 residual block stack.

        Args:
            x: Input stem feature tensor.

        Returns:
            TensorLike: Output feature tensor after stage 5.
        """
        x = resnet.stack_residual_blocks_v2(x, 64, 3, name="conv2")
        x = resnet.stack_residual_blocks_v2(x, 128, 4, name="conv3")
        x = resnet.stack_residual_blocks_v2(x, 256, 23, name="conv4")
        return resnet.stack_residual_blocks_v2(x, 512, 3, stride1=1, name="conv5")

    return resnet.ResNet(
        stack_fn,
        True,
        True,
        name=name,
        weights_name="resnet101v2",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
    )


def ResNet152V2(
    include_top: bool = True,
    weights: str | None = "imagenet",
    input_tensor: TensorLike | None = None,
    input_shape: ShapeTuple | TensorLike | None = None,
    pooling: str | None = None,
    classes: int = 1000,
    classifier_activation: str | TensorLike | None = "softmax",
    name: str = "resnet152v2",
) -> TensorLike:
    """Instantiates the ResNet152V2 architecture.

    Args:
        include_top: Whether to include the fully-connected classification layer at the top.
        weights: Pretrained weight specification (``"imagenet"``, ``None``, or file path).
        input_tensor: Optional Keras tensor to use as image input.
        input_shape: Optional input shape tuple when ``include_top=False``.
        pooling: Optional feature pooling mode (``None``, ``"avg"``, or ``"max"``).
        classes: Number of classification output classes.
        classifier_activation: Activation function applied on the top classification layer.
        name: Name of the model instance.

    Returns:
        TensorLike: Instantiated ResNet152V2 Functional model.
    """

    def stack_fn(x: TensorLike) -> TensorLike:
        """Apply the 4-stage ResNet152V2 residual block stack.

        Args:
            x: Input stem feature tensor.

        Returns:
            TensorLike: Output feature tensor after stage 5.
        """
        x = resnet.stack_residual_blocks_v2(x, 64, 3, name="conv2")
        x = resnet.stack_residual_blocks_v2(x, 128, 8, name="conv3")
        x = resnet.stack_residual_blocks_v2(x, 256, 36, name="conv4")
        return resnet.stack_residual_blocks_v2(x, 512, 3, stride1=1, name="conv5")

    return resnet.ResNet(
        stack_fn,
        True,
        True,
        name=name,
        weights_name="resnet152v2",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
    )


def preprocess_input(x: TensorLike, data_format: str | None = None) -> TensorLike:
    """Preprocess an image batch for ResNetV2 by scaling pixels between -1 and 1.

    Args:
        x: Input image tensor or array.
        data_format: Optional image data format (``"channels_first"`` or ``"channels_last"``).

    Returns:
        TensorLike: Preprocessed image tensor.
    """
    return imagenet_utils.preprocess_input(x, data_format=data_format, mode="tf")


def decode_predictions(preds: TensorLike, top: TensorLike = 5) -> TensorLike:
    """Decode top-k ImageNet predictions for ResNetV2.

    Args:
        preds: 2D probability tensor of shape ``(samples, 1000)``.
        top: Number of top predictions to return per sample.

    Returns:
        TensorLike: List of lists of ``(class_id, class_name, score)`` tuples.
    """
    return imagenet_utils.decode_predictions(preds, top=top)
