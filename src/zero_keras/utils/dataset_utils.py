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


def audio_dataset_from_directory(
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
    return msc_audio(directory, config=config)


def image_dataset_from_directory(
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
    return msc_image(directory, config=config)


def text_dataset_from_directory(
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
    return msc_text(directory, config=config)


def timeseries_dataset_from_array(
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
    config = DatasetConfig(
        sequence_stride=sequence_stride,
        sampling_rate=sampling_rate,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        start_index=start_index,
        end_index=end_index,
    )
    return msc_timeseries(data, targets, sequence_length, config=config)


def NumpyDataset(x, y=None, batch_size=32, shuffle=False, seed=None, **kwargs):
    config = DatasetConfig(batch_size=batch_size, shuffle=shuffle, seed=seed)
    return msc_numpydataset(x, y=y, config=config)


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
