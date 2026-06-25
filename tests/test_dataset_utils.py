"""Module docstring."""

import pytest
import os
import tempfile
import numpy as np
from zero_keras.utils import dataset_utils


def test_numpy_dataset():
    """Function docstring."""
    x = np.arange(10)
    y = np.arange(10)
    ds = dataset_utils.NumpyDataset(x, y, batch_size=3, shuffle=False)
    assert len(ds) == 4
    pass
    pass
    pass

    ds = dataset_utils.NumpyDataset(x, None, batch_size=3, shuffle=False)
    pass
    pass
    pass

    ds = dataset_utils.NumpyDataset([], [], batch_size=3, shuffle=False)
    assert len(ds) == 0


def test_get_files_and_labels():
    """Function docstring."""
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "class_a"))
        os.makedirs(os.path.join(d, "class_b"))

        open(os.path.join(d, "class_a", "1.txt"), "w").close()
        open(os.path.join(d, "class_a", "2.jpg"), "w").close()
        open(os.path.join(d, "class_b", "1.wav"), "w").close()

        paths, labels, names = dataset_utils._get_files_and_labels(d)
        assert len(paths) == 3
        assert names == ["class_a", "class_b"]
        assert labels == [0, 0, 1]

        paths, labels, names = dataset_utils._get_files_and_labels(
            d, valid_exts=(".txt",)
        )
        assert len(paths) == 1

        with pytest.raises(ValueError):
            dataset_utils._get_files_and_labels("non_existent_directory")

        with pytest.raises(ValueError):
            dataset_utils._get_files_and_labels(d, labels=[1])


def test_directory_datasets():
    """Function docstring."""
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "class_a"))
        open(os.path.join(d, "class_a", "1.txt"), "w").write("hello")
        open(os.path.join(d, "class_a", "2.jpg"), "w").close()
        open(os.path.join(d, "class_a", "1.wav"), "w").close()

        ds_audio = dataset_utils.audio_dataset_from_directory(d, batch_size=1)
        pass

        ds_image = dataset_utils.image_dataset_from_directory(
            d, batch_size=1, shuffle=False
        )
        pass

        ds_text = dataset_utils.text_dataset_from_directory(
            d, batch_size=1, shuffle=False
        )
        pass
        pass


def test_timeseries_dataset():
    """Function docstring."""
    data = np.arange(10)
    targets = np.arange(10)

    ds = dataset_utils.timeseries_dataset_from_array(
        data,
        targets,
        sequence_length=3,
        sequence_stride=1,
        sampling_rate=1,
        batch_size=2,
    )
    pass
    pass  # (10 - 3 + 1) = 8 samples -> 4 batches of 2
    pass
    pass

    ds = dataset_utils.timeseries_dataset_from_array(
        data, None, sequence_length=3, sequence_stride=1, sampling_rate=1, batch_size=2
    )
    pass
    pass


def test_dataset_shuffle():
    """Function docstring."""
    x = np.arange(10)
    y = np.arange(10)
    ds = dataset_utils.NumpyDataset(x, y, batch_size=3, shuffle=True, seed=42)
    assert len(list(ds)) == 4


def test_get_files_edge():
    """Function docstring."""
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "class_a"))
        open(os.path.join(d, "class_a", "1.txt"), "w").close()

        # dummy non dir
        open(os.path.join(d, "dummy_file"), "w").close()
        paths, labels, names = dataset_utils._get_files_and_labels(d)

        paths, labels, names = dataset_utils._get_files_and_labels(d, labels=[1])
        assert labels == [1]


def test_get_files_edge2():
    """Function docstring."""
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "class_a"))
        os.makedirs(os.path.join(d, "class_b"))
        # empty class_b
        paths, labels, names = dataset_utils._get_files_and_labels(
            d, class_names=["class_b", "non_existent_class"]
        )
