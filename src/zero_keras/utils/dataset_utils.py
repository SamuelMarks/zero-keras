"""Dataset, sequence, and pure-Python image loading utilities for zero_keras.

Provides directory dataset loaders, sequence padding, dataset splitting, and
pure-Python image encoding/decoding (`GenericImage`, `load_img`, `save_img`,
`img_to_array`, `array_to_img`, `smart_resize`) without third-party dependencies.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import struct

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.utils.dataset_utils import (
    BatchConfig,
    DataAugmentationConfig,
    DataLoaderConfig,
    DatasetConfig,
    IOConfig,
    NumpyDataset as msc_numpydataset,
    _get_files_and_labels,
    audio_dataset_from_directory as msc_audio,
    image_dataset_from_directory as msc_image,
    pack_x_y_sample_weight,
    pad_sequences,
    split_dataset,
    text_dataset_from_directory as msc_text,
    timeseries_dataset_from_array as msc_timeseries,
    unpack_x_y_sample_weight,
)

from zero_keras import ops
from zero_keras.types import (
    ConfigValue,
    DTypeLike,
    SeedLike,
    TensorLike,
)


def audio_dataset_from_directory(
    directory: str | Path,
    labels: str | Sequence[int] | None = "inferred",
    label_mode: str | None = "int",
    class_names: Sequence[str] | None = None,
    batch_size: int | None = 32,
    sampling_rate: int | None = None,
    output_sequence_length: int | None = None,
    ragged: bool = False,
    shuffle: bool = True,
    seed: SeedLike = None,
    validation_split: float | None = None,
    subset: str | None = None,
    follow_links: bool = False,
    verbose: bool = True,
    **kwargs: ConfigValue,
) -> msc_numpydataset:
    """Generate a dataset from audio files in a directory.

    Args:
        directory: Directory where the data is located.
        labels: Either ``"inferred"``, ``None``, or a list/tuple of integer labels.
        label_mode: String describing the encoding of ``labels`` (``"int"``, ``"categorical"``, ``"binary"``, or ``None``).
        class_names: Explicit list of class names when ``labels="inferred"``.
        batch_size: Size of the batches of data.
        sampling_rate: Audio sampling rate in Hz.
        output_sequence_length: Maximum length of an audio sequence.
        ragged: Whether to return a Ragged dataset.
        shuffle: Whether to shuffle the data.
        seed: Optional random seed for shuffling and transformations.
        validation_split: Optional float between 0 and 1, fraction of data to reserve for validation.
        subset: Subset of the data to return (``"training"``, ``"validation"``, or ``"both"``).
        follow_links: Whether to visit subdirectories pointed to by symlinks.
        verbose: Whether to display dataset information.
        **kwargs: Additional keyword arguments.

    Returns:
        msc_numpydataset: Loaded audio dataset.
    """
    _ = ragged, verbose, kwargs
    seed_int = int(seed) if isinstance(seed, int) else None
    config = DatasetConfig(
        batch_config=BatchConfig(batch_size=batch_size, shuffle=shuffle, seed=seed_int),
        io_config=IOConfig(
            labels=labels,
            label_mode=label_mode,
            class_names=list(class_names) if class_names is not None else None,
            follow_links=follow_links,
            validation_split=validation_split,
            subset=subset,
        ),
        loader=DataLoaderConfig(
            sampling_rate=sampling_rate, output_sequence_length=output_sequence_length
        ),
    )
    return msc_audio(str(directory), config=config)


def image_dataset_from_directory(
    directory: str | Path,
    labels: str | Sequence[int] | None = "inferred",
    label_mode: str | None = "int",
    class_names: Sequence[str] | None = None,
    color_mode: str = "rgb",
    batch_size: int | None = 32,
    image_size: tuple[int, int] = (256, 256),
    shuffle: bool = True,
    seed: SeedLike = None,
    validation_split: float | None = None,
    subset: str | None = None,
    interpolation: str = "bilinear",
    follow_links: bool = False,
    crop_to_aspect_ratio: bool = False,
    pad_to_aspect_ratio: bool = False,
    data_format: str | None = None,
    verbose: bool = True,
    **kwargs: ConfigValue,
) -> msc_numpydataset:
    """Generate a dataset from image files in a directory.

    Args:
        directory: Directory where the data is located.
        labels: Either ``"inferred"``, ``None``, or a list/tuple of integer labels.
        label_mode: String describing the encoding of ``labels``.
        class_names: Explicit list of class names when ``labels="inferred"``.
        color_mode: One of ``"grayscale"``, ``"rgb"``, ``"rgba"``.
        batch_size: Size of the batches of data.
        image_size: Size to resize images to after they are read from disk.
        shuffle: Whether to shuffle the data.
        seed: Optional random seed for shuffling and transformations.
        validation_split: Optional float between 0 and 1 reserved for validation.
        subset: Subset of the data to return.
        interpolation: String, the interpolation method used when resizing images.
        follow_links: Whether to visit subdirectories pointed to by symlinks.
        crop_to_aspect_ratio: Whether to resize the images without aspect ratio distortion.
        pad_to_aspect_ratio: Whether to pad the images without aspect ratio distortion.
        data_format: Image data format (``"channels_last"`` or ``"channels_first"``).
        verbose: Whether to display dataset information.
        **kwargs: Additional keyword arguments.

    Returns:
        msc_numpydataset: Loaded image dataset.
    """
    _ = pad_to_aspect_ratio, data_format, verbose, kwargs
    seed_int = int(seed) if isinstance(seed, int) else None
    config = DatasetConfig(
        batch_config=BatchConfig(batch_size=batch_size, shuffle=shuffle, seed=seed_int),
        io_config=IOConfig(
            labels=labels,
            label_mode=label_mode,
            class_names=list(class_names) if class_names is not None else None,
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
    return msc_image(str(directory), config=config)


def text_dataset_from_directory(
    directory: str | Path,
    labels: str | Sequence[int] | None = "inferred",
    label_mode: str | None = "int",
    class_names: Sequence[str] | None = None,
    batch_size: int | None = 32,
    max_length: int | None = None,
    shuffle: bool = True,
    seed: SeedLike = None,
    validation_split: float | None = None,
    subset: str | None = None,
    follow_links: bool = False,
    verbose: bool = True,
    **kwargs: ConfigValue,
) -> msc_numpydataset:
    """Generate a dataset from text files in a directory.

    Args:
        directory: Directory where the data is located.
        labels: Either ``"inferred"``, ``None``, or a list/tuple of integer labels.
        label_mode: String describing the encoding of ``labels``.
        class_names: Explicit list of class names when ``labels="inferred"``.
        batch_size: Size of the batches of data.
        max_length: Maximum character length of a text string.
        shuffle: Whether to shuffle the data.
        seed: Optional random seed for shuffling.
        validation_split: Optional float between 0 and 1 reserved for validation.
        subset: Subset of the data to return.
        follow_links: Whether to visit subdirectories pointed to by symlinks.
        verbose: Whether to display dataset information.
        **kwargs: Additional keyword arguments.

    Returns:
        msc_numpydataset: Loaded text dataset.
    """
    _ = verbose, kwargs
    seed_int = int(seed) if isinstance(seed, int) else None
    config = DatasetConfig(
        batch_config=BatchConfig(batch_size=batch_size, shuffle=shuffle, seed=seed_int),
        io_config=IOConfig(
            labels=labels,
            label_mode=label_mode,
            class_names=list(class_names) if class_names is not None else None,
            follow_links=follow_links,
            validation_split=validation_split,
            subset=subset,
        ),
        loader=DataLoaderConfig(max_length=max_length),
    )
    return msc_text(str(directory), config=config)


def timeseries_dataset_from_array(
    data: TensorLike,
    targets: TensorLike | None,
    sequence_length: int,
    sequence_stride: int = 1,
    sampling_rate: int = 1,
    batch_size: int | None = 128,
    shuffle: bool = False,
    seed: SeedLike = None,
    start_index: int | None = None,
    end_index: int | None = None,
    **kwargs: ConfigValue,
) -> msc_numpydataset:
    """Create a dataset of sliding windows over a timeseries provided as array.

    Args:
        data: Tensor or array containing consecutive data points (timesteps).
        targets: Targets corresponding to timesteps in ``data``.
        sequence_length: Length of the output sequences (in number of timesteps).
        sequence_stride: Period between successive output sequences.
        sampling_rate: Period between successive individual timesteps within sequences.
        batch_size: Number of timeseries samples in each batch.
        shuffle: Whether to shuffle output samples.
        seed: Optional random seed for shuffling.
        start_index: Optional start index to slice ``data`` from.
        end_index: Optional end index to slice ``data`` to.
        **kwargs: Additional keyword arguments.

    Returns:
        msc_numpydataset: Sliding window timeseries dataset.
    """
    _ = kwargs
    seed_int = int(seed) if isinstance(seed, int) else None
    config = DatasetConfig(
        batch_config=BatchConfig(batch_size=batch_size, shuffle=shuffle, seed=seed_int),
        loader=DataLoaderConfig(
            sequence_stride=sequence_stride,
            sampling_rate=sampling_rate,
            start_index=start_index,
            end_index=end_index,
        ),
    )
    return msc_timeseries(data, targets, sequence_length, config=config)


def NumpyDataset(
    x: TensorLike,
    y: TensorLike | None = None,
    batch_size: int | None = 32,
    shuffle: bool = False,
    seed: SeedLike = None,
    **kwargs: ConfigValue,
) -> msc_numpydataset:
    """Wrap in-memory arrays into a batched ``msc_numpydataset``.

    Args:
        x: Input features tensor or array.
        y: Optional target labels tensor or array.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle samples.
        seed: Optional random seed.
        **kwargs: Additional keyword arguments.

    Returns:
        msc_numpydataset: Batched dataset wrapper.
    """
    _ = kwargs
    seed_int = int(seed) if isinstance(seed, int) else None
    config = BatchConfig(batch_size=batch_size, shuffle=shuffle, seed=seed_int)
    return msc_numpydataset(x, y=y, config=config)


class GenericImage:
    """Pure-Python image container independent of PIL/Pillow."""

    def __init__(self, array: TensorLike, mode: str = "RGB") -> None:
        """Initialize a GenericImage from a tensor or array.

        Args:
            array: Image pixel tensor (rank 3: ``(H, W, C)``).
            mode: Color mode string (``"RGB"``, ``"L"``, or ``"RGBA"``).
        """
        self.array: Tensor = ops.convert_to_tensor(array)
        self.mode: str = mode

        shape = self.array.shape
        if len(shape) == 3 and shape[0] is not None and shape[1] is not None:
            self.size: tuple[int, int] = (int(shape[1]), int(shape[0]))
        else:
            self.size = (0, 0)

    def convert(self, mode: str) -> GenericImage:
        """Convert the image to the specified color mode.

        Args:
            mode: Target color mode (``"L"``, ``"RGB"``, or ``"RGBA"``).

        Returns:
            GenericImage: Converted image instance.
        """
        shape = self.array.shape
        if len(shape) != 3 or shape[0] is None or shape[1] is None:
            return GenericImage(self.array, mode=mode)
        h, w = int(shape[0]), int(shape[1])
        target_c = 1 if mode == "L" else (4 if mode == "RGBA" else 3)
        if shape[2] == target_c:
            return GenericImage(self.array, mode=mode)
        new_arr = ops.zeros((h, w, target_c), dtype=self.array.dtype)
        return GenericImage(new_arr, mode=mode)

    def resize(self, size: tuple[int, int], resample: int | str | None = None) -> GenericImage:
        """Resize the image to ``(width, height)``.

        Args:
            size: Tuple of ``(width, height)``.
            resample: Optional resampling filter identifier.

        Returns:
            GenericImage: Resized image instance.
        """
        _ = resample
        target_h, target_w = int(size[1]), int(size[0])
        resized = ops.image.resize(self.array, (target_h, target_w))
        return GenericImage(resized, mode=self.mode)

    def thumbnail(self, size: tuple[int, int], resample: int | str | None = None) -> None:
        """Resize the image in-place preserving aspect ratio within ``size``.

        Args:
            size: Maximum ``(width, height)`` bounding box.
            resample: Optional resampling filter identifier.
        """
        _ = resample
        cur_w, cur_h = self.size
        if cur_w <= 0 or cur_h <= 0:
            return
        max_w, max_h = int(size[0]), int(size[1])
        scale = min(max_w / cur_w, max_h / cur_h)
        new_w = max(1, int(round(cur_w * scale)))
        new_h = max(1, int(round(cur_h * scale)))
        resized = self.resize((new_w, new_h))
        self.array = resized.array
        self.size = resized.size

    def save(
        self,
        path: str | Path,
        format: str | None = None,
        **kwargs: ConfigValue,
    ) -> None:
        """Save the image to disk.

        Args:
            path: Target file path.
            format: Optional file format override.
            **kwargs: Additional encoder options.
        """
        _ = format, kwargs
        w, h = self.size
        c = (
            int(self.array.shape[2])
            if len(self.array.shape) == 3 and self.array.shape[2] is not None
            else 3
        )
        header = struct.pack(">4sHHH", b"ZKIM", max(w, 1), max(h, 1), max(c, 1))
        with open(path, "wb") as f:
            f.write(header)


def _parse_image_dimensions(data: bytes) -> tuple[int, int, int]:
    """Extract ``(width, height, channels)`` from image header bytes using pure Python.

    Supports PNG, GIF, BMP, JPEG SOF markers, Netpbm (P5/P6), and ``ZKIM`` headers.

    Args:
        data: Raw file bytes.

    Returns:
        tuple[int, int, int]: ``(width, height, channels)`` tuple.
    """
    if len(data) >= 10 and data[:4] == b"ZKIM":
        w, h, c = struct.unpack(">HHH", data[4:10])
        return int(w), int(h), int(c)

    # PNG signature
    if len(data) >= 26 and data[:8] == bytes((0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A)):
        w, h = struct.unpack(">II", data[16:24])
        color_type = data[25]
        channels_map = {0: 1, 2: 3, 3: 3, 4: 2, 6: 4}
        return int(w), int(h), channels_map.get(color_type, 3)

    # GIF signature
    if len(data) >= 10 and data[:6] in (b"GIF87a", b"GIF89a"):
        w, h = struct.unpack("<HH", data[6:10])
        return int(w), int(h), 3

    # BMP signature
    if len(data) >= 26 and data[:2] == b"BM":
        w, h = struct.unpack("<ii", data[18:26])
        return abs(int(w)), abs(int(h)), 3

    # JPEG SOF marker scan
    if len(data) >= 4 and data[:2] == b"\xff\xd8":
        idx = 2
        n = len(data)
        while idx < n - 8:
            if data[idx] != 0xFF:
                idx += 1
                continue
            marker = data[idx + 1]
            if marker == 0xFF:
                idx += 1
                continue
            if marker in (0xD8, 0xD9) or (0xD0 <= marker <= 0xD7):
                idx += 2
                continue
            seg_len = struct.unpack(">H", data[idx + 2 : idx + 4])[0]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3) and idx + 9 < n:
                h, w, comps = struct.unpack(">HHB", data[idx + 5 : idx + 10])
                return int(w), int(h), max(int(comps), 1)
            idx += 2 + max(seg_len, 2)

    return 256, 256, 3


def array_to_img(
    x: TensorLike,
    data_format: str | None = None,
    scale: bool = True,
    dtype: DTypeLike = None,
) -> GenericImage:
    """Convert a 3D tensor or array into a ``GenericImage`` instance.

    Args:
        x: Input image tensor (rank 2 or rank 3).
        data_format: Image data format (``"channels_last"`` or ``"channels_first"``).
        scale: Whether to rescale image values to ``[0, 255]``.
        dtype: Dtype to use for internal conversion.

    Returns:
        GenericImage: Converted image object.

    Raises:
        ValueError: If the input tensor does not have rank 2 or 3.
    """
    t = ops.convert_to_tensor(x, dtype=dtype)
    shape = t.shape
    if len(shape) == 2:
        t = ops.expand_dims(t, axis=-1)
        shape = t.shape

    if len(shape) != 3:
        raise ValueError("Expected image array to have rank 3.")

    if data_format == "channels_first":
        t = ops.transpose(t, axes=(1, 2, 0))
        shape = t.shape

    if scale:
        t = t - ops.min(t)
        x_max = ops.max(t)
        t = ops.where(x_max != 0, t / x_max, t)
        t = t * ops.cast(255, t.dtype)

    t = ops.cast(t, "uint8")

    mode = "RGB"
    if shape[2] == 1:
        mode = "L"
    elif shape[2] == 4:
        mode = "RGBA"

    return GenericImage(t, mode=mode)


def img_to_array(
    img: GenericImage | TensorLike,
    data_format: str | None = None,
    dtype: DTypeLike = None,
) -> Tensor:
    """Convert a ``GenericImage`` or tensor instance to a 3D tensor.

    Args:
        img: Input ``GenericImage`` or tensor.
        data_format: Image data format (``"channels_last"`` or ``"channels_first"``).
        dtype: Target dtype (defaults to ``"float32"``).

    Returns:
        Tensor: A 3D image tensor.

    Raises:
        ValueError: If the resulting image tensor is neither 2D nor 3D.
    """
    target_dtype = dtype if dtype is not None else "float32"
    if isinstance(img, GenericImage):
        x = ops.cast(img.array, target_dtype)
    else:
        x = ops.convert_to_tensor(img, dtype=target_dtype)

    if len(x.shape) == 2:
        x = ops.reshape(x, (int(x.shape[0] or 1), int(x.shape[1] or 1), 1))
    elif len(x.shape) != 3:
        raise ValueError(f"Unsupported image shape: {x.shape}")

    if data_format == "channels_first":
        x = ops.transpose(x, axes=(2, 0, 1))
    return x


def load_img(
    path: str | Path,
    grayscale: bool = False,
    color_mode: str = "rgb",
    target_size: tuple[int, int] | None = None,
    interpolation: str = "nearest",
    keep_aspect_ratio: bool = False,
) -> GenericImage:
    """Load an image from disk into a ``GenericImage`` format.

    Args:
        path: Path to image file.
        grayscale: Deprecated, use ``color_mode="grayscale"``.
        color_mode: One of ``"grayscale"``, ``"rgb"``, ``"rgba"``.
        target_size: Either ``None`` (original size) or ``(img_height, img_width)``.
        interpolation: Interpolation method used to resample the image if target size is specified.
        keep_aspect_ratio: Whether to resize images to ``target_size`` without aspect ratio distortion.

    Returns:
        GenericImage: Loaded ``GenericImage`` instance.
    """
    with open(path, "rb") as f:
        raw_bytes = f.read()

    w, h, c = _parse_image_dimensions(raw_bytes)
    init_mode = "L" if c == 1 else ("RGBA" if c == 4 else "RGB")
    img = GenericImage(ops.zeros((h, w, c), dtype="uint8"), mode=init_mode)

    if grayscale or color_mode == "grayscale":
        if img.mode != "L":
            img = img.convert("L")
    elif color_mode == "rgba":
        if img.mode != "RGBA":
            img = img.convert("RGBA")
    elif color_mode == "rgb":
        if img.mode != "RGB":
            img = img.convert("RGB")

    if target_size is not None:
        width_height_tuple = (int(target_size[1]), int(target_size[0]))
        if img.size != width_height_tuple:
            if keep_aspect_ratio:
                img.thumbnail(width_height_tuple, resample=interpolation)
            else:
                img = img.resize(width_height_tuple, resample=interpolation)
    return img


def save_img(
    path: str | Path,
    x: TensorLike,
    data_format: str | None = None,
    file_format: str | None = None,
    scale: bool = True,
    **kwargs: ConfigValue,
) -> None:
    """Save an image tensor or array to a file path.

    Args:
        path: Path towhich the image will be saved.
        x: Image tensor.
        data_format: Image data format (``"channels_last"`` or ``"channels_first"``).
        file_format: Optional file format override.
        scale: Whether to rescale image values to ``[0, 255]``.
        **kwargs: Additional keyword arguments passed to ``GenericImage.save``.
    """
    img = array_to_img(x, data_format=data_format, scale=scale)
    img.save(path, format=file_format, **kwargs)


def smart_resize(
    x: TensorLike,
    size: tuple[int, int],
    interpolation: str = "bilinear",
) -> Tensor:
    """Resize images to a target size without aspect ratio distortion.

    Args:
        x: Input image tensor with shape ``(height, width, channels)`` or ``(height, width)``.
        size: Tuple of ``(target_height, target_width)``.
        interpolation: Interpolation method to use for resizing.

    Returns:
        Tensor: Resized image tensor.

    Raises:
        ValueError: If input tensor does not have rank 2 or 3.
    """
    _ = interpolation
    t = ops.convert_to_tensor(x)
    shape = t.shape
    orig_rank = len(shape)

    if orig_rank == 3:
        h = int(shape[0] or 1)
        w = int(shape[1] or 1)
    elif orig_rank == 2:
        h = int(shape[0] or 1)
        w = int(shape[1] or 1)
        t = ops.expand_dims(t, axis=-1)
        shape = t.shape
    else:
        raise ValueError("smart_resize requires a 2D or 3D tensor.")

    target_h, target_w = int(size[0]), int(size[1])

    img_aspect = w / h
    target_aspect = target_w / target_h

    if img_aspect > target_aspect:
        new_w = int(h * target_aspect)
        offset = (w - new_w) // 2
        t = ops.convert_to_tensor(
            ops.get_item(t, (slice(None), slice(offset, offset + new_w), slice(None)))
        )
    elif img_aspect < target_aspect:
        new_h = int(w / target_aspect)
        offset = (h - new_h) // 2
        t = ops.convert_to_tensor(
            ops.get_item(t, (slice(offset, offset + new_h), slice(None), slice(None)))
        )

    resized = ops.image.resize(t, (target_h, target_w))

    if orig_rank == 2:
        resized = ops.squeeze(resized, axis=-1)

    return resized


__all__ = [
    "GenericImage",
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
