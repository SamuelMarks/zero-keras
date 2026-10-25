"""Image operations submodule (`zero_keras.ops.image`)."""

from __future__ import annotations

from zero_keras.ops import _ImageOps

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
