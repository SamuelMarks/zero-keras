"""Dataset utilities shim."""

from ml_switcheroo_compiler.utils.dataset_utils import (
    audio_dataset_from_directory as msc_audio,
    image_dataset_from_directory as msc_image,
    text_dataset_from_directory as msc_text,
    timeseries_dataset_from_array as msc_timeseries,
    DatasetConfig,
    _get_files_and_labels,
    pack_x_y_sample_weight,
    pad_sequences,
    split_dataset,
    unpack_x_y_sample_weight,
    NumpyDataset as msc_numpydataset,
)


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
        labels=labels,
        label_mode=label_mode,
        class_names=class_names,
        batch_size=batch_size,
        sampling_rate=sampling_rate,
        output_sequence_length=output_sequence_length,
        shuffle=shuffle,
        seed=seed,
        validation_split=validation_split,
        subset=subset,
        follow_links=follow_links,
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
        labels=labels,
        label_mode=label_mode,
        class_names=class_names,
        color_mode=color_mode,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        seed=seed,
        validation_split=validation_split,
        subset=subset,
        interpolation=interpolation,
        follow_links=follow_links,
        crop_to_aspect_ratio=crop_to_aspect_ratio,
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
        labels=labels,
        label_mode=label_mode,
        class_names=class_names,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=shuffle,
        seed=seed,
        validation_split=validation_split,
        subset=subset,
        follow_links=follow_links,
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
        sequence_stride=sequence_stride,
        sampling_rate=sampling_rate,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        start_index=start_index,
        end_index=end_index,
    )
    return msc_timeseries(
        data, targets, sequence_length, config=config
    )  # pragma: no cover


def NumpyDataset(
    x, y=None, batch_size=32, shuffle=False, seed=None, **kwargs
):  # pragma: no cover
    """docstring."""

    config = DatasetConfig(batch_size=batch_size, shuffle=shuffle, seed=seed)
    return msc_numpydataset(x, y=y, config=config)  # pragma: no cover


__all__ = [
    "audio_dataset_from_directory",
    "image_dataset_from_directory",
    "text_dataset_from_directory",
    "timeseries_dataset_from_array",
    "pack_x_y_sample_weight",
    "pad_sequences",
    "split_dataset",
    "unpack_x_y_sample_weight",
    "NumpyDataset",
    "_get_files_and_labels",
]


def array_to_img(x, data_format=None, scale=True, dtype=None):  # pragma: no cover
    """array_to_img docstring."""
    return x  # pragma: no cover


def img_to_array(img, data_format=None, dtype=None):  # pragma: no cover
    """img_to_array docstring."""
    return img  # pragma: no cover


def load_img(
    path,
    grayscale=False,
    color_mode="rgb",
    target_size=None,
    interpolation="nearest",
    keep_aspect_ratio=False,
):  # pragma: no cover
    """load_img docstring."""
    return path  # pragma: no cover


def save_img(
    path, x, data_format=None, file_format=None, scale=True, **kwargs
):  # pragma: no cover
    """save_img docstring."""
    pass  # pragma: no cover


def smart_resize(x, size, interpolation="bilinear"):  # pragma: no cover
    """smart_resize docstring."""
    return x  # pragma: no cover
