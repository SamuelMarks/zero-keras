"""Module docstring."""

from unittest.mock import patch
from zero_keras import layers
from ml_switcheroo_compiler.ops import asarray


def test_dense_and_conv_missing_branches():
    """Function docstring."""
    with (
        patch("zero_keras.layers.ops") as mock_ops,
        patch(
            "ml_switcheroo_compiler.ops.linalg.conv.conv_general_dilated"
        ) as mock_cgd,
    ):
        mock_ops.matmul.return_value = asarray([[1.0]])
        mock_ops.add.return_value = asarray([[1.0]])
        mock_ops.conv1d.return_value = asarray([[1.0]])
        mock_ops.conv2d.return_value = asarray([[1.0]])
        mock_ops.conv3d.return_value = asarray([[1.0]])
        mock_ops.conv1d_transpose.return_value = asarray([[1.0]])
        mock_ops.conv2d_transpose.return_value = asarray([[1.0]])
        mock_ops.conv3d_transpose.return_value = asarray([[1.0]])
        mock_cgd.return_value = asarray([[1.0]])

        for layer_cls in [
            layers.Dense,
            layers.Conv1D,
            layers.Conv2D,
            layers.Conv3D,
            layers.Conv1DTranspose,
            layers.Conv2DTranspose,
            layers.Conv3DTranspose,
            layers.DepthwiseConv1D,
            layers.DepthwiseConv2D,
            layers.SeparableConv1D,
            layers.SeparableConv2D,
        ]:
            kwargs = {"use_bias": False, "activation": None}
            if layer_cls == layers.Dense:
                kwargs["units"] = 4
            elif "Conv" in layer_cls.__name__:
                if "Depthwise" not in layer_cls.__name__:
                    kwargs["filters"] = 4
                kwargs["kernel_size"] = 2

            layer = layer_cls(**kwargs)
            layer.kernel = asarray([1.0])

            if layer_cls.__name__ in (
                "DepthwiseConv1D",
                "DepthwiseConv2D",
                "SeparableConv1D",
                "SeparableConv2D",
            ):
                layer.depthwise_kernel = asarray([1.0])
                layer.pointwise_kernel = asarray([1.0])

            input_shape = (2, 4, 4, 4, 4) if layer_cls != layers.Dense else (2, 4)
            if "1D" in layer_cls.__name__:
                input_shape = (2, 4, 4)
            elif "2D" in layer_cls.__name__:
                input_shape = (2, 4, 4, 4)
            layer.build(input_shape)

            layer2 = layer_cls(**kwargs)
            # manually set built to True and weights so we don't fail in eager ops
            layer2.built = True
            layer2.kernel = asarray([1.0])
            if layer_cls.__name__ in (
                "DepthwiseConv1D",
                "DepthwiseConv2D",
                "SeparableConv1D",
                "SeparableConv2D",
            ):
                layer2.depthwise_kernel = asarray([1.0])
                layer2.pointwise_kernel = asarray([1.0])

            if layer_cls == layers.Dense:
                layer2(asarray([[1.0] * 4] * 2))
            elif "1D" in layer_cls.__name__:
                layer2(asarray([[[1.0] * 4] * 4] * 2))
            elif "2D" in layer_cls.__name__:
                layer2(asarray([[[[1.0] * 4] * 4] * 4] * 2))
            elif "3D" in layer_cls.__name__:
                layer2(asarray([[[[[1.0] * 4] * 4] * 4] * 4] * 2))


def test_pre_set_weights_with_bias():
    """Function docstring."""
    for layer_cls in [
        layers.Dense,
        layers.Conv1D,
        layers.Conv2D,
        layers.Conv3D,
        layers.Conv1DTranspose,
        layers.Conv2DTranspose,
        layers.Conv3DTranspose,
        layers.DepthwiseConv1D,
        layers.DepthwiseConv2D,
        layers.SeparableConv1D,
        layers.SeparableConv2D,
    ]:
        kwargs = {"use_bias": True}
        if layer_cls == layers.Dense:
            kwargs["units"] = 4
        elif "Conv" in layer_cls.__name__:
            if "Depthwise" not in layer_cls.__name__:
                kwargs["filters"] = 4
            kwargs["kernel_size"] = 2

        layer = layer_cls(**kwargs)
        layer.kernel = asarray([1.0])
        layer.bias = asarray([1.0])

        if layer_cls.__name__ in (
            "DepthwiseConv1D",
            "DepthwiseConv2D",
            "SeparableConv1D",
            "SeparableConv2D",
        ):
            layer.depthwise_kernel = asarray([1.0])
            layer.pointwise_kernel = asarray([1.0])

        input_shape = (2, 4, 4, 4, 4) if layer_cls != layers.Dense else (2, 4)
        if "1D" in layer_cls.__name__:
            input_shape = (2, 4, 4)
        elif "2D" in layer_cls.__name__:
            input_shape = (2, 4, 4, 4)
        layer.build(input_shape)
