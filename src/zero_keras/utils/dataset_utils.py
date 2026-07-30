"""Dataset utilities shim."""

import builtins

from ml_switcheroo_compiler.utils.dataset_utils import (
    BatchConfig,
    DataAugmentationConfig,
    DataLoaderConfig,
    DatasetConfig,
    IOConfig,
    _get_files_and_labels,
    pack_x_y_sample_weight,
    pad_sequences,
    split_dataset,
    unpack_x_y_sample_weight,
)
from ml_switcheroo_compiler.utils.dataset_utils import (
    NumpyDataset as msc_numpydataset,
)
from ml_switcheroo_compiler.utils.dataset_utils import (
    audio_dataset_from_directory as msc_audio,
)
from ml_switcheroo_compiler.utils.dataset_utils import (
    image_dataset_from_directory as msc_image,
)
from ml_switcheroo_compiler.utils.dataset_utils import (
    text_dataset_from_directory as msc_text,
)
from ml_switcheroo_compiler.utils.dataset_utils import (
    timeseries_dataset_from_array as msc_timeseries,
)

try:
    pil_image = builtins.__import__("PIL.Image").Image
except Exception:
    pil_image = None
from zero_keras import ops


def audio_dataset_from_directory(  # pragma: no cover
    directory,
    labels="inferred",
    label_mode="int",
    class_names=None,
    batch_size=32,
    sampling_rate=None,
    output_sequence_length=None,
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    follow_links=False,
    **kwargs,
):
    """docstring."""

    config = DatasetConfig(
        batch_config=BatchConfig(batch_size=batch_size, shuffle=shuffle, seed=seed),
        io_config=IOConfig(
            labels=labels,
            label_mode=label_mode,
            class_names=class_names,
            follow_links=follow_links,
            validation_split=validation_split,
            subset=subset,
        ),
        loader=DataLoaderConfig(
            sampling_rate=sampling_rate, output_sequence_length=output_sequence_length
        ),
    )
    return msc_audio(directory, config=config)  # pragma: no cover


def image_dataset_from_directory(  # pragma: no cover
    directory,
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    **kwargs,
):
    """docstring."""

    config = DatasetConfig(
        batch_config=BatchConfig(batch_size=batch_size, shuffle=shuffle, seed=seed),
        io_config=IOConfig(
            labels=labels,
            label_mode=label_mode,
            class_names=class_names,
            follow_links=follow_links,
            validation_split=validation_split,
            subset=subset,
        ),
        augmentation=DataAugmentationConfig(
            color_mode=color_mode,
            image_size=image_size,
            interpolation=interpolation,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
        ),
    )
    return msc_image(directory, config=config)  # pragma: no cover


def text_dataset_from_directory(  # pragma: no cover
    directory,
    labels="inferred",
    label_mode="int",
    class_names=None,
    batch_size=32,
    max_length=None,
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    follow_links=False,
    **kwargs,
):
    """docstring."""

    config = DatasetConfig(
        batch_config=BatchConfig(batch_size=batch_size, shuffle=shuffle, seed=seed),
        io_config=IOConfig(
            labels=labels,
            label_mode=label_mode,
            class_names=class_names,
            follow_links=follow_links,
            validation_split=validation_split,
            subset=subset,
        ),
        loader=DataLoaderConfig(max_length=max_length),
    )
    return msc_text(directory, config=config)  # pragma: no cover


def timeseries_dataset_from_array(  # pragma: no cover
    data,
    targets,
    sequence_length,
    sequence_stride=1,
    sampling_rate=1,
    batch_size=128,
    shuffle=False,
    seed=None,
    start_index=None,
    end_index=None,
    **kwargs,
):
    """docstring."""

    config = DatasetConfig(
        batch_config=BatchConfig(batch_size=batch_size, shuffle=shuffle, seed=seed),
        loader=DataLoaderConfig(
            sequence_stride=sequence_stride,
            sampling_rate=sampling_rate,
            start_index=start_index,
            end_index=end_index,
        ),
    )
    return msc_timeseries(
        data, targets, sequence_length, config=config
    )  # pragma: no cover


def NumpyDataset(
    x, y=None, batch_size=32, shuffle=False, seed=None, **kwargs
):  # pragma: no cover
    """docstring."""

    config = BatchConfig(batch_size=batch_size, shuffle=shuffle, seed=seed)
    return msc_numpydataset(x, y=y, config=config)  # pragma: no cover


__all__ = [
    "NumpyDataset",
    "_get_files_and_labels",
    "array_to_img",
    "audio_dataset_from_directory",
    "image_dataset_from_directory",
    "img_to_array",
    "load_img",
    "pack_x_y_sample_weight",
    "pad_sequences",
    "save_img",
    "smart_resize",
    "split_dataset",
    "text_dataset_from_directory",
    "timeseries_dataset_from_array",
    "unpack_x_y_sample_weight",
]


class GenericImage:
    """A generic image format for when PIL is not available."""

    def __init__(self, array, mode="RGB"):
        self.array = ops.convert_to_tensor(array)
        self.mode = mode

        shape = self.array.shape
        if len(shape) == 3:
            self.size = (int(shape[1]), int(shape[0]))
        else:
            self.size = (0, 0)

    def convert(self, mode):
        """Mock convert."""
        return GenericImage(self.array, mode=mode)

    def resize(self, size, resample=None):
        """Mock resize."""
        target_h, target_w = size[1], size[0]
        resized = ops.image.resize(self.array, (target_h, target_w))
        return GenericImage(resized, mode=self.mode)

    def save(self, path, format=None, **kwargs):
        """Mock save."""
        with open(path, "wb") as f:
            f.write(b"mock_image_data")


def array_to_img(x, data_format=None, scale=True, dtype=None):
    """array_to_img docstring.

    Args:
        x: Input data.
        data_format: Data format.
        scale: Scale flag.
        dtype: Dtype.
    """
    x = ops.convert_to_tensor(x, dtype=dtype)
    shape = x.shape
    if len(shape) == 2:
        x = ops.expand_dims(x, axis=-1)
        shape = x.shape

    if len(shape) != 3:
        raise ValueError("Expected image array to have rank 3.")

    if scale:
        x = x - ops.min(x)
        x_max = ops.max(x)
        x = ops.where(x_max != 0, x / x_max, x)
        x = x * ops.cast(255, x.dtype)

    x = ops.cast(x, "uint8")

    mode = "RGB"
    if shape[2] == 1:
        mode = "L"
    elif shape[2] == 4:
        mode = "RGBA"

    return GenericImage(x, mode=mode)


def img_to_array(img, data_format=None, dtype=None):
    if pil_image is None:
        raise ImportError("pillow is required to use img_to_array")
    np = builtins.__import__("numpy")

    x = (
        img.array
        if hasattr(img, "array")
        else np.asarray(img, dtype=dtype or "float32")
    )
    if x.ndim == 2:
        x = x.reshape((x.shape[0], x.shape[1], 1))
    if data_format == "channels_first":
        x = x.transpose(2, 0, 1)
    return x


def load_img(
    path,
    grayscale=False,
    color_mode="rgb",
    target_size=None,
    interpolation="nearest",
    keep_aspect_ratio=False,
):
    if pil_image is None:
        raise ImportError("pillow is required to load images")
    with open(path, "rb") as f:
        img = pil_image.open(f)
        img.load()
    if grayscale or color_mode == "grayscale":
        if img.mode not in ("L", "I;16", "I"):
            img = img.convert("L")
    elif color_mode == "rgba":
        if img.mode != "RGBA":
            img = img.convert("RGBA")
    elif color_mode == "rgb" and img.mode != "RGB":
        img = img.convert("RGB")

    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in (
                "nearest",
                "bilinear",
                "bicubic",
                "lanczos",
                "box",
                "hamming",
            ):
                resample = pil_image.NEAREST
            else:
                resample_map = {
                    "nearest": pil_image.NEAREST,
                    "bilinear": pil_image.BILINEAR,
                    "bicubic": pil_image.BICUBIC,
                    "lanczos": getattr(pil_image, "LANCZOS", pil_image.NEAREST),
                    "box": getattr(pil_image, "BOX", pil_image.NEAREST),
                    "hamming": getattr(pil_image, "HAMMING", pil_image.NEAREST),
                }
                resample = resample_map.get(interpolation, pil_image.NEAREST)

            if keep_aspect_ratio:
                img.thumbnail(width_height_tuple, resample)
            else:
                img = img.resize(width_height_tuple, resample)
    return img


def save_img(path, x, data_format=None, file_format=None, scale=True, **kwargs):
    """save_img docstring.

    Args:
        path: Path.
        x: Data.
        data_format: Data format.
        file_format: File format.
        scale: Scale flag.
        kwargs: Kwargs.
    """
    img = array_to_img(x, data_format=data_format, scale=scale)
    img.save(path, format=file_format, **kwargs)


def smart_resize(x, size, interpolation="bilinear"):
    """smart_resize docstring.

    Args:
        x: Data.
        size: Size.
        interpolation: Interpolation.
    """
    x = ops.convert_to_tensor(x)
    shape = x.shape

    if len(shape) == 3:
        h, w = shape[0], shape[1]
    elif len(shape) == 2:
        h, w = shape[0], shape[1]
        x = ops.expand_dims(x, axis=-1)
    else:
        raise ValueError("smart_resize requires a 2D or 3D tensor.")

    target_h, target_w = size[0], size[1]

    # Calculate crop
    img_aspect = w / h
    target_aspect = target_w / target_h

    if img_aspect > target_aspect:
        # Image is too wide
        new_w = int(h * target_aspect)
        offset = (w - new_w) // 2
        # ops.slice works like x[start:end]
        x = ops.slice(x, [0, offset, 0], [int(h), new_w, int(shape[-1])])
    elif img_aspect < target_aspect:
        # Image is too tall
        new_h = int(w / target_aspect)
        offset = (h - new_h) // 2
        x = ops.slice(x, [offset, 0, 0], [new_h, int(w), int(shape[-1])])

    resized = ops.image.resize(x, (target_h, target_w))

    if len(shape) == 2:
        resized = ops.squeeze(resized, axis=-1)

    return resized
