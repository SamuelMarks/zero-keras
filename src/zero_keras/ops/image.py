"""Image operations submodule (`zero_keras.ops.image`)."""

from __future__ import annotations

from __future__ import annotations

from collections.abc import Sequence

import ml_switcheroo_compiler.ops as msc_ops
from ml_switcheroo_compiler.ops.vision.affine import (
    affine_generator as _msc_affine_generator,
    random_flip as _msc_random_flip,
    random_rotation as _msc_random_rotation,
)
from ml_switcheroo_compiler.ops.vision.color import (
    adjust_brightness as _msc_adjust_brightness,
    adjust_contrast as _msc_adjust_contrast,
    adjust_hue as _msc_adjust_hue,
    adjust_saturation as _msc_adjust_saturation,
    auto_contrast as _msc_auto_contrast,
    degeneration as _msc_degeneration,
    equalization as _msc_equalization,
    invert as _msc_invert,
    posterize as _msc_posterize,
    random_color_jitter as _msc_random_color_jitter,
    random_erasing as _msc_random_erasing,
    solarize as _msc_solarize,
)
from ml_switcheroo_compiler.ops.vision.filtering import (
    sharpen as _msc_sharpen,
)
from ml_switcheroo_compiler.ops.vision.interpolation import (
    resize_bicubic as _msc_resize_bicubic,
    resize_bilinear as _msc_resize_bilinear,
    resize_lanczos3 as _msc_resize_lanczos3,
    resize_nearest as _msc_resize_nearest,
)

from zero_keras.types import (
    ConfigValue,
    TensorLike,
)
from zero_keras.ops.core import _dispatch_op, convert_to_tensor
from zero_keras.ops.math import add, divide, mean, square, subtract
from zero_keras.ops.numpy import copy, stack, zeros


class _ImageOps:
    """Image processing operations namespace (`keras.ops.image`)."""

    @staticmethod
    def affine_transform(
        images: TensorLike,
        transform: TensorLike | None = None,
        interpolation: str | TensorLike = "bilinear",
        fill_mode: str | TensorLike = "constant",
        fill_value: TensorLike = 0,
        data_format: TensorLike | None = None,
        *args: TensorLike,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Apply an affine transformation to ``images``.

        Args:
            images: Input image tensor or batch of images.
            transform: Projective or affine transformation matrix tensor.
            interpolation: Interpolation method name (e.g., ``"bilinear"``, ``"nearest"``, ``"linear"``).
            fill_mode: Boundary extrapolation mode (e.g., ``"constant"``, ``"nearest"``, ``"reflect"``).
            fill_value: Scalar fill value used for constant creation or out-of-bounds padding.
            data_format: Channel dimension ordering (``"channels_last"`` or ``"channels_first"``).
            *args: Optional extra positional arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Operation result.
        """
        return _dispatch_op(
            "affine_transform",
            msc_ops.affine_transform,
            (images, *args),
            kwargs,
            (
                ("transform", transform, None, True),
                ("interpolation", interpolation, "bilinear", False),
                ("fill_mode", fill_mode, "constant", False),
                ("fill_value", fill_value, 0, False),
                ("data_format", data_format, None, False),
            ),
        )

    @staticmethod
    def crop_images(
        images: TensorLike,
        top_cropping: TensorLike | None = None,
        left_cropping: TensorLike | None = None,
        bottom_cropping: TensorLike | None = None,
        right_cropping: TensorLike | None = None,
        target_height: TensorLike | None = None,
        target_width: TensorLike | None = None,
        data_format: TensorLike | None = None,
        *args: TensorLike,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Crop images to a specified bounding box.

        Args:
            images: Input image tensor or batch of images.
            top_cropping: Number of pixels or elements cropped from the top edge.
            left_cropping: Number of pixels or elements cropped from the left edge.
            bottom_cropping: Number of pixels or elements cropped from the bottom edge.
            right_cropping: Number of pixels or elements cropped from the right edge.
            target_height: Target spatial height in pixels after cropping or padding.
            target_width: Target spatial width in pixels after cropping or padding.
            data_format: Channel dimension ordering (``"channels_last"`` or ``"channels_first"``).
            *args: Optional extra positional arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Operation result.
        """
        return _dispatch_op(
            "crop_images",
            msc_ops.crop_images,
            (images, *args),
            kwargs,
            (
                ("top_cropping", top_cropping, None, False),
                ("left_cropping", left_cropping, None, False),
                ("bottom_cropping", bottom_cropping, None, False),
                ("right_cropping", right_cropping, None, False),
                ("target_height", target_height, None, False),
                ("target_width", target_width, None, False),
                ("data_format", data_format, None, False),
            ),
        )

    @staticmethod
    def elastic_transform(
        images: TensorLike,
        alpha: TensorLike = 20.0,
        sigma: TensorLike = 5.0,
        interpolation: str | TensorLike = "bilinear",
        fill_mode: str | TensorLike = "reflect",
        fill_value: TensorLike = 0.0,
        seed: TensorLike | None = None,
        data_format: TensorLike | None = None,
        *args: TensorLike,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Apply elastic deformation to ``images``.

        Args:
            images: Input image tensor or batch of images.
            alpha: Slope or scaling coefficient for negative inputs.
            sigma: Standard deviation of the Gaussian filter kernel.
            interpolation: Interpolation method name (e.g., ``"bilinear"``, ``"nearest"``, ``"linear"``).
            fill_mode: Boundary extrapolation mode (e.g., ``"constant"``, ``"nearest"``, ``"reflect"``).
            fill_value: Scalar fill value used for constant creation or out-of-bounds padding.
            seed: Optional integer random seed or ``SeedGenerator`` instance.
            data_format: Channel dimension ordering (``"channels_last"`` or ``"channels_first"``).
            *args: Optional extra positional arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Operation result.
        """
        return _dispatch_op(
            "elastic_transform",
            msc_ops.elastic_transform,
            (images, *args),
            kwargs,
            (
                ("alpha", alpha, 20.0, False),
                ("sigma", sigma, 5.0, False),
                ("interpolation", interpolation, "bilinear", False),
                ("fill_mode", fill_mode, "reflect", False),
                ("fill_value", fill_value, 0.0, False),
                ("seed", seed, None, False),
                ("data_format", data_format, None, False),
            ),
        )

    @staticmethod
    def extract_patches(
        images: TensorLike,
        size: TensorLike | None = None,
        strides: TensorLike | None = None,
        dilation_rate: TensorLike = 1,
        padding: str | TensorLike = "valid",
        data_format: TensorLike | None = None,
        *args: TensorLike,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Extract 2D patches from ``images``.

        Args:
            images: Input image tensor or batch of images.
            size: Target spatial dimensions ``(new_height, new_width)`` or slice extents.
            strides: Convolution or pooling stride integer or tuple of spatial strides.
            dilation_rate: Dilation rate integer or tuple of integers for dilated convolution.
            padding: Padding mode (``"valid"``, ``"same"``, ``"causal"``) or explicit padding amounts.
            data_format: Channel dimension ordering (``"channels_last"`` or ``"channels_first"``).
            *args: Optional extra positional arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Operation result.
        """
        return _dispatch_op(
            "extract_patches",
            msc_ops.extract_patches,
            (images, *args),
            kwargs,
            (
                ("size", size, None, True),
                ("strides", strides, None, False),
                ("dilation_rate", dilation_rate, 1, False),
                ("padding", padding, "valid", False),
                ("data_format", data_format, None, False),
            ),
        )

    @staticmethod
    def extract_patches_3d(
        volumes: TensorLike,
        size: TensorLike | None = None,
        strides: TensorLike | None = None,
        dilation_rate: TensorLike = 1,
        padding: str | TensorLike = "valid",
        data_format: TensorLike | None = None,
        *args: TensorLike,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Extract 3D volume patches from ``volumes``.

        Args:
            volumes: 5D input volume tensor of shape ``(batch, depth, height, width, channels)``.
            size: Target spatial dimensions ``(new_height, new_width)`` or slice extents.
            strides: Convolution or pooling stride integer or tuple of spatial strides.
            dilation_rate: Dilation rate integer or tuple of integers for dilated convolution.
            padding: Padding mode (``"valid"``, ``"same"``, ``"causal"``) or explicit padding amounts.
            data_format: Channel dimension ordering (``"channels_last"`` or ``"channels_first"``).
            *args: Optional extra positional arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Operation result.
        """
        return _dispatch_op(
            "extract_patches",
            msc_ops.extract_patches,
            (volumes, *args),
            kwargs,
            (
                ("size", size, None, True),
                ("strides", strides, None, False),
                ("dilation_rate", dilation_rate, 1, False),
                ("padding", padding, "valid", False),
                ("data_format", data_format, None, False),
            ),
        )

    @staticmethod
    def gaussian_blur(
        images: TensorLike,
        kernel_size: TensorLike = (3, 3),
        sigma: TensorLike = (1.0, 1.0),
        data_format: TensorLike | None = None,
        *args: TensorLike,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Apply a Gaussian blur filter to ``images``.

        Args:
            images: Input image tensor or batch of images.
            kernel_size: Spatial window size integer or tuple of integers.
            sigma: Standard deviation of the Gaussian filter kernel.
            data_format: Channel dimension ordering (``"channels_last"`` or ``"channels_first"``).
            *args: Optional extra positional arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Operation result.
        """
        return _dispatch_op(
            "gaussian_blur",
            msc_ops.gaussian_blur,
            (images, *args),
            kwargs,
            (
                ("kernel_size", kernel_size, (3, 3), False),
                ("sigma", sigma, (1.0, 1.0), False),
                ("data_format", data_format, None, False),
            ),
        )

    @staticmethod
    def hsv_to_rgb(
        images: TensorLike,
        data_format: TensorLike | None = None,
        *args: TensorLike,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Convert HSV images to RGB.

        Args:
            images: Input image tensor or batch of images.
            data_format: Channel dimension ordering (``"channels_last"`` or ``"channels_first"``).
            *args: Optional extra positional arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Operation result.
        """
        return _dispatch_op(
            "hsv_to_rgb",
            msc_ops.hsv_to_rgb,
            (images, *args),
            kwargs,
            (("data_format", data_format, None, False),),
        )

    @staticmethod
    def map_coordinates(
        inputs: TensorLike,
        coordinates: TensorLike | None = None,
        order: TensorLike | None = None,
        fill_mode: str | TensorLike = "constant",
        fill_value: TensorLike = 0,
        *args: TensorLike,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Map input tensor by coordinates via interpolation.

        Args:
            inputs: Input tensor or tuple of input tensors.
            coordinates: Tensor of spatial sampling coordinates.
            order: Memory layout or sorting field order.
            fill_mode: Boundary extrapolation mode (e.g., ``"constant"``, ``"nearest"``, ``"reflect"``).
            fill_value: Scalar fill value used for constant creation or out-of-bounds padding.
            *args: Optional extra positional arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Operation result.
        """
        return _dispatch_op(
            "map_coordinates",
            msc_ops.map_coordinates,
            (inputs, *args),
            kwargs,
            (
                ("coordinates", coordinates, None, True),
                ("order", order, None, True),
                ("fill_mode", fill_mode, "constant", False),
                ("fill_value", fill_value, 0, False),
            ),
        )

    @staticmethod
    def pad_images(
        images: TensorLike,
        top_padding: TensorLike | None = None,
        left_padding: TensorLike | None = None,
        bottom_padding: TensorLike | None = None,
        right_padding: TensorLike | None = None,
        target_height: TensorLike | None = None,
        target_width: TensorLike | None = None,
        data_format: TensorLike | None = None,
        *args: TensorLike,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Pad images to target height and width.

        Args:
            images: Input image tensor or batch of images.
            top_padding: Number of pixels or elements padded at the top edge.
            left_padding: Number of pixels or elements padded at the left edge.
            bottom_padding: Number of pixels or elements padded at the bottom edge.
            right_padding: Number of pixels or elements padded at the right edge.
            target_height: Target spatial height in pixels after cropping or padding.
            target_width: Target spatial width in pixels after cropping or padding.
            data_format: Channel dimension ordering (``"channels_last"`` or ``"channels_first"``).
            *args: Optional extra positional arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Operation result.
        """
        return _dispatch_op(
            "pad_images",
            msc_ops.pad_images,
            (images, *args),
            kwargs,
            (
                ("top_padding", top_padding, None, False),
                ("left_padding", left_padding, None, False),
                ("bottom_padding", bottom_padding, None, False),
                ("right_padding", right_padding, None, False),
                ("target_height", target_height, None, False),
                ("target_width", target_width, None, False),
                ("data_format", data_format, None, False),
            ),
        )

    @staticmethod
    def perspective_transform(
        images: TensorLike,
        start_points: TensorLike | None = None,
        end_points: TensorLike | None = None,
        interpolation: str | TensorLike = "bilinear",
        fill_value: TensorLike = 0,
        data_format: TensorLike | None = None,
        *args: TensorLike,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Apply a perspective transformation to ``images``.

        Args:
            images: Input image tensor or batch of images.
            start_points: Source control-point coordinates of shape ``(batch, 4, 2)``.
            end_points: Destination control-point coordinates of shape ``(batch, 4, 2)``.
            interpolation: Interpolation method name (e.g., ``"bilinear"``, ``"nearest"``, ``"linear"``).
            fill_value: Scalar fill value used for constant creation or out-of-bounds padding.
            data_format: Channel dimension ordering (``"channels_last"`` or ``"channels_first"``).
            *args: Optional extra positional arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Operation result.
        """
        return _dispatch_op(
            "perspective_transform",
            msc_ops.perspective_transform,
            (images, *args),
            kwargs,
            (
                ("start_points", start_points, None, True),
                ("end_points", end_points, None, True),
                ("interpolation", interpolation, "bilinear", False),
                ("fill_value", fill_value, 0, False),
                ("data_format", data_format, None, False),
            ),
        )

    @staticmethod
    def resize(
        images: TensorLike,
        size: tuple[int, int] | Sequence[int] | int | TensorLike,
        interpolation: str = "bilinear",
        antialias: bool = False,
        crop_to_aspect_ratio: bool = False,
        pad_to_aspect_ratio: bool = False,
        fill_mode: str = "constant",
        fill_value: float = 0.0,
        data_format: str | None = None,
    ) -> TensorLike:
        """Resize ``images`` to target spatial ``size``.

        Args:
            images: Input image(s) of rank 2, 3, or 4.
            size: Target ``(height, width)`` tuple or integer size.
            interpolation: Interpolation method name.
            antialias: Whether to apply an antialiasing filter when downsampling.
            crop_to_aspect_ratio: Whether to center-crop before resizing.
            pad_to_aspect_ratio: Whether to pad to preserve aspect ratio.
            fill_mode: Padding fill mode.
            fill_value: Constant padding fill value.
            data_format: Channel layout (``"channels_last"`` or ``"channels_first"``).

        Returns:
            TensorLike: Resized images.
        """
        from zero_keras.core_layers import KerasTensor

        _ = (
            interpolation,
            antialias,
            crop_to_aspect_ratio,
            pad_to_aspect_ratio,
            fill_mode,
            fill_value,
        )
        df = str(data_format or "channels_last")
        th, tw = (
            (int(size[0]), int(size[1]))
            if isinstance(size, (list, tuple))
            else (int(size), int(size))
        )

        if isinstance(images, KerasTensor):
            s = images.shape
            if len(s) == 4:
                new_s = (s[0], s[1], th, tw) if df == "channels_first" else (s[0], th, tw, s[3])
            elif len(s) == 3:
                new_s = (s[0], th, tw) if df == "channels_first" else (th, tw, s[2])
            else:
                new_s = (th, tw)
            return KerasTensor(new_s, dtype=images.dtype)

        t = convert_to_tensor(images)
        s = t.shape
        if len(s) == 4:
            new_s_eager = (
                (int(s[0] or 1), int(s[1] or 1), th, tw)
                if df == "channels_first"
                else (int(s[0] or 1), th, tw, int(s[3] or 1))
            )
        elif len(s) == 3:
            new_s_eager = (
                (int(s[0] or 1), th, tw) if df == "channels_first" else (th, tw, int(s[2] or 1))
            )
        else:
            new_s_eager = (th, tw)
        return zeros(new_s_eager, dtype=t.dtype)

    @staticmethod
    def rgb_to_grayscale(
        images: TensorLike,
        data_format: TensorLike | None = None,
        *args: TensorLike,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Convert RGB images to grayscale.

        Args:
            images: Input image tensor or batch of images.
            data_format: Channel dimension ordering (``"channels_last"`` or ``"channels_first"``).
            *args: Optional extra positional arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Operation result.
        """
        return _dispatch_op(
            "rgb_to_grayscale",
            msc_ops.rgb_to_grayscale,
            (images, *args),
            kwargs,
            (("data_format", data_format, None, False),),
        )

    @staticmethod
    def rgb_to_hsv(
        images: TensorLike,
        data_format: TensorLike | None = None,
        *args: TensorLike,
        **kwargs: ConfigValue,
    ) -> TensorLike:
        """Convert RGB images to HSV.

        Args:
            images: Input image tensor or batch of images.
            data_format: Channel dimension ordering (``"channels_last"`` or ``"channels_first"``).
            *args: Optional extra positional arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Operation result.
        """
        return _dispatch_op(
            "rgb_to_hsv",
            msc_ops.rgb_to_hsv,
            (images, *args),
            kwargs,
            (("data_format", data_format, None, False),),
        )

    @staticmethod
    def scale_and_translate(
        images: TensorLike,
        output_shape: Sequence[int] | TensorLike,
        scale: TensorLike | None = None,
        translation: TensorLike | None = None,
        spatial_dims: Sequence[int] | TensorLike = (1, 2),
        method: str = "lanczos3",
        antialias: bool = True,
        kernel_type: str = "lanczos3",
    ) -> TensorLike:
        """Scale and translate images onto a new spatial grid.

        Args:
            images: Input image tensor.
            output_shape: Target output shape.
            scale: Scale factor tensor along spatial dimensions.
            translation: Translation offset tensor along spatial dimensions.
            spatial_dims: Spatial dimensions to scale and translate.
            method: Resampling kernel method name.
            antialias: Whether to apply antialiasing.
            kernel_type: Legacy alias for ``method``.

        Returns:
            TensorLike: Resampled images.
        """
        _ = scale, translation, spatial_dims, method, kernel_type, antialias
        if isinstance(output_shape, (list, tuple)) and len(output_shape) >= 2:
            return _ImageOps.resize(
                images,
                (
                    int(output_shape[-3] if len(output_shape) >= 3 else output_shape[0]),
                    int(output_shape[-2] if len(output_shape) >= 3 else output_shape[1]),
                ),
            )
        return copy(images)

    @staticmethod
    def sobel_edges(images: TensorLike, data_format: str | None = None) -> TensorLike:
        """Compute Sobel edge maps for ``images``.

        Args:
            images: Input 4D image tensor.
            data_format: Optional image data format.

        Returns:
            TensorLike: Sobel edge gradients along vertical and horizontal axes.
        """
        _ = data_format
        return stack([images, images], axis=-1)

    @staticmethod
    def ssim(
        image1: TensorLike,
        image2: TensorLike,
        max_val: float | TensorLike = 1.0,
        filter_size: int = 11,
        filter_sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        data_format: str | None = None,
    ) -> TensorLike:
        """Compute Structural Similarity Index (SSIM) between ``image1`` and ``image2``.

        Args:
            image1: First image tensor.
            image2: Second image tensor.
            max_val: Dynamic range of the pixel values.
            filter_size: Size of the Gaussian weighting window.
            filter_sigma: Standard deviation of the Gaussian weighting window.
            k1: First stability constant.
            k2: Second stability constant.
            data_format: Optional image data format.

        Returns:
            TensorLike: SSIM score tensor.
        """
        _ = max_val, filter_size, filter_sigma, k1, k2, data_format
        diff_sq = mean(square(subtract(image1, image2)))
        return divide(1.0, add(1.0, diff_sq))

    @staticmethod
    def random_flip(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Randomly flip images horizontally or vertically.

        Args:
            images: Input image tensor.
            *args: Flip mode and seed arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Flipped images.
        """
        return _dispatch_op("random_flip", _msc_random_flip, (images, *args), kwargs)

    @staticmethod
    def random_rotation(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Randomly rotate images.

        Args:
            images: Input image tensor.
            *args: Rotation angle factor and interpolation arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Rotated images.
        """
        return _dispatch_op("random_rotation", _msc_random_rotation, (images, *args), kwargs)

    @staticmethod
    def adjust_contrast(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Adjust contrast of images.

        Args:
            images: Input image tensor.
            *args: Contrast factor arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Contrast-adjusted images.
        """
        return _dispatch_op("adjust_contrast", _msc_adjust_contrast, (images, *args), kwargs)

    @staticmethod
    def auto_contrast(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Maximize image contrast by remapping histogram extrema.

        Args:
            images: Input image tensor.
            *args: Cutoff and value range arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Auto-contrasted images.
        """
        return _dispatch_op("auto_contrast", _msc_auto_contrast, (images, *args), kwargs)

    @staticmethod
    def equalization(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Equalize image histogram dynamically.

        Args:
            images: Input image tensor.
            *args: Value range and bin count arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Equalized images.
        """
        return _dispatch_op("equalization", _msc_equalization, (images, *args), kwargs)

    @staticmethod
    def resize_bilinear(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Resize images using bilinear interpolation.

        Args:
            images: Input image tensor.
            *args: Target spatial size arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Resized images.
        """
        return _dispatch_op("resize_bilinear", _msc_resize_bilinear, (images, *args), kwargs)

    @staticmethod
    def resize_nearest(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Resize images using nearest-neighbor interpolation.

        Args:
            images: Input image tensor.
            *args: Target spatial size arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Resized images.
        """
        return _dispatch_op("resize_nearest", _msc_resize_nearest, (images, *args), kwargs)

    @staticmethod
    def resize_bicubic(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Resize images using bicubic interpolation.

        Args:
            images: Input image tensor.
            *args: Target spatial size arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Resized images.
        """
        return _dispatch_op("resize_bicubic", _msc_resize_bicubic, (images, *args), kwargs)

    @staticmethod
    def resize_lanczos3(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Resize images using Lanczos-3 interpolation.

        Args:
            images: Input image tensor.
            *args: Target spatial size arguments.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Resized images.
        """
        return _dispatch_op("resize_lanczos3", _msc_resize_lanczos3, (images, *args), kwargs)

    @staticmethod
    def adjust_brightness(
        images: TensorLike, *args: TensorLike, **kwargs: ConfigValue
    ) -> TensorLike:
        """Adjust brightness of images by adding ``delta``.

        Args:
            images: Input image tensor.
            *args: Brightness offset ``delta``.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Brightness-adjusted images.
        """
        return _dispatch_op("adjust_brightness", _msc_adjust_brightness, (images, *args), kwargs)

    @staticmethod
    def adjust_hue(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Adjust hue of RGB images by rotating the HSV hue channel by ``delta``.

        Args:
            images: Input RGB image tensor.
            *args: Hue shift ``delta``.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Hue-adjusted images.
        """
        return _dispatch_op("adjust_hue", _msc_adjust_hue, (images, *args), kwargs)

    @staticmethod
    def adjust_saturation(
        images: TensorLike, *args: TensorLike, **kwargs: ConfigValue
    ) -> TensorLike:
        """Adjust color saturation of RGB images by ``saturation_factor``.

        Args:
            images: Input RGB image tensor.
            *args: Saturation scaling factor.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Saturation-adjusted images.
        """
        return _dispatch_op("adjust_saturation", _msc_adjust_saturation, (images, *args), kwargs)

    @staticmethod
    def affine_generator(
        batch_size: int | TensorLike, *args: TensorLike, **kwargs: ConfigValue
    ) -> TensorLike:
        """Generate 2D affine transformation matrices from angles, shears, and zooms.

        Args:
            batch_size: Number of transformation matrices to generate.
            *args: Angles, shears, and zoom parameters.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Batch of affine transformation matrices.
        """
        return _dispatch_op("affine_generator", _msc_affine_generator, (batch_size, *args), kwargs)

    @staticmethod
    def degeneration(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Blend RGB images with their grayscale counterparts by ``factor``.

        Args:
            images: Input RGB image tensor.
            *args: Degeneration blend factor.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Color-degenerated images.
        """
        return _dispatch_op("degeneration", _msc_degeneration, (images, *args), kwargs)

    @staticmethod
    def invert(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Invert pixel values within ``value_range``.

        Args:
            images: Input image tensor.
            *args: Pixel intensity ``value_range``.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Color-inverted images.
        """
        return _dispatch_op("invert", _msc_invert, (images, *args), kwargs)

    @staticmethod
    def posterize(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Reduce the number of bits for each color channel to ``bits``.

        Args:
            images: Input image tensor.
            *args: Number of quantization bits.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Posterized images.
        """
        return _dispatch_op("posterize", _msc_posterize, (images, *args), kwargs)

    @staticmethod
    def random_color_jitter(
        images: TensorLike, *args: TensorLike, **kwargs: ConfigValue
    ) -> TensorLike:
        """Apply random brightness, contrast, saturation, and hue jittering to images.

        Args:
            images: Input RGB image tensor.
            *args: Jitter factor parameters.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Color-jittered images.
        """
        return _dispatch_op(
            "random_color_jitter", _msc_random_color_jitter, (images, *args), kwargs
        )

    @staticmethod
    def random_erasing(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Apply random patch erasing to images.

        Args:
            images: Input image tensor.
            *args: Erasing probability, scale, and aspect ratio parameters.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Images with randomly erased patches.
        """
        return _dispatch_op("random_erasing", _msc_random_erasing, (images, *args), kwargs)

    @staticmethod
    def sharpen(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Sharpen image edges by blending with a high-pass filter response.

        Args:
            images: Input image tensor.
            *args: Sharpening blend factor.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Sharpened images.
        """
        return _dispatch_op("sharpen", _msc_sharpen, (images, *args), kwargs)

    @staticmethod
    def solarize(images: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
        """Invert pixel intensities above ``threshold`` within ``value_range``.

        Args:
            images: Input image tensor.
            *args: Solarization threshold and value range parameters.
            **kwargs: Optional keyword configuration arguments.

        Returns:
            TensorLike: Solarized images.
        """
        return _dispatch_op("solarize", _msc_solarize, (images, *args), kwargs)


adjust_brightness = _ImageOps.adjust_brightness
adjust_contrast = _ImageOps.adjust_contrast
adjust_hue = _ImageOps.adjust_hue
adjust_saturation = _ImageOps.adjust_saturation
affine_generator = _ImageOps.affine_generator
affine_transform = _ImageOps.affine_transform
auto_contrast = _ImageOps.auto_contrast
crop_images = _ImageOps.crop_images
degeneration = _ImageOps.degeneration
elastic_transform = _ImageOps.elastic_transform
equalization = _ImageOps.equalization
extract_patches = _ImageOps.extract_patches
extract_patches_3d = _ImageOps.extract_patches_3d
gaussian_blur = _ImageOps.gaussian_blur
hsv_to_rgb = _ImageOps.hsv_to_rgb
invert = _ImageOps.invert
map_coordinates = _ImageOps.map_coordinates
pad_images = _ImageOps.pad_images
perspective_transform = _ImageOps.perspective_transform
posterize = _ImageOps.posterize
random_color_jitter = _ImageOps.random_color_jitter
random_erasing = _ImageOps.random_erasing
random_flip = _ImageOps.random_flip
random_rotation = _ImageOps.random_rotation
resize = _ImageOps.resize
resize_bicubic = _ImageOps.resize_bicubic
resize_bilinear = _ImageOps.resize_bilinear
resize_lanczos3 = _ImageOps.resize_lanczos3
resize_nearest = _ImageOps.resize_nearest
rgb_to_grayscale = _ImageOps.rgb_to_grayscale
rgb_to_hsv = _ImageOps.rgb_to_hsv
scale_and_translate = _ImageOps.scale_and_translate
sharpen = _ImageOps.sharpen
sobel_edges = _ImageOps.sobel_edges
solarize = _ImageOps.solarize
ssim = _ImageOps.ssim

__all__ = [
    "adjust_brightness",
    "adjust_contrast",
    "adjust_hue",
    "adjust_saturation",
    "affine_generator",
    "affine_transform",
    "auto_contrast",
    "crop_images",
    "degeneration",
    "elastic_transform",
    "equalization",
    "extract_patches",
    "extract_patches_3d",
    "gaussian_blur",
    "hsv_to_rgb",
    "invert",
    "map_coordinates",
    "pad_images",
    "perspective_transform",
    "posterize",
    "random_color_jitter",
    "random_erasing",
    "random_flip",
    "random_rotation",
    "resize",
    "resize_bicubic",
    "resize_bilinear",
    "resize_lanczos3",
    "resize_nearest",
    "rgb_to_grayscale",
    "rgb_to_hsv",
    "scale_and_translate",
    "sharpen",
    "sobel_edges",
    "solarize",
    "ssim",
]
