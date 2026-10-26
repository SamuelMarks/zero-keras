"""Unit and behavioral tests for datasets in zero_keras."""

from __future__ import annotations

from unittest import mock

import pytest

from zero_keras import datasets, ops, random


def test_boston_housing() -> None:
    """Test ``boston_housing`` behavior and edge cases in ``test_datasets``."""
    x_dummy = random.uniform((100, 13))
    y_dummy = random.uniform((100,))

    with mock.patch("zero_keras.datasets.boston_housing.get_file") as mock_get_file:
        mock_get_file.return_value = "dummy.npz"
        with mock.patch("zero_keras.ops.load") as mock_load:
            mock_load.return_value.__enter__.return_value = {"x": x_dummy, "y": y_dummy}

            (x_train, y_train), (x_test, y_test) = datasets.boston_housing.load_data()

            assert x_train.shape == (80, 13)
            assert y_train.shape == (80,)
            assert x_test.shape == (20, 13)
            assert y_test.shape == (20,)


def test_cifar10() -> None:
    """Test ``cifar10`` behavior and edge cases in ``test_datasets``."""
    x_dummy = random.uniform((10000, 3, 32, 32))
    y_dummy = random.uniform((10000,))

    with mock.patch("zero_keras.datasets.cifar10.get_file") as mock_get_file:
        mock_get_file.return_value = "dummy_dir"
        with (
            mock.patch("os.path.exists", return_value=False),
            mock.patch(
                "zero_keras.datasets.cifar10.load_batch",
                return_value=(x_dummy, y_dummy),
            ),
        ):
            (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

            assert x_train.shape == (50000, 32, 32, 3)
            assert y_train.shape == (50000, 1)
            assert x_test.shape == (10000, 32, 32, 3)
            assert y_test.shape == (10000, 1)


def test_cifar100() -> None:
    """Test ``cifar100`` behavior and edge cases in ``test_datasets``."""
    x_dummy = random.uniform((10000, 3, 32, 32))
    y_dummy = random.uniform((10000,))

    with mock.patch("zero_keras.datasets.cifar100.get_file") as mock_get_file:
        mock_get_file.return_value = "dummy_dir"
        with (
            mock.patch("os.path.exists", return_value=False),
            mock.patch(
                "zero_keras.datasets.cifar100.load_batch",
                return_value=(x_dummy, y_dummy),
            ),
        ):
            (x_train, y_train), (_x_test, _y_test) = datasets.cifar100.load_data()

            assert x_train.shape == (10000, 32, 32, 3)
            assert y_train.shape == (10000, 1)


def test_cifar100_coarse() -> None:
    """Test ``cifar100_coarse`` behavior and edge cases in ``test_datasets``."""
    x_dummy = random.uniform((10000, 3, 32, 32))
    y_dummy = random.uniform((10000,))

    with mock.patch("zero_keras.datasets.cifar100.get_file") as mock_get_file:
        mock_get_file.return_value = "dummy_dir"
        with (
            mock.patch("os.path.exists", return_value=False),
            mock.patch(
                "zero_keras.datasets.cifar100.load_batch",
                return_value=(x_dummy, y_dummy),
            ),
        ):
            (x_train, y_train), (_x_test, _y_test) = datasets.cifar100.load_data(
                label_mode="coarse"
            )

            assert x_train.shape == (10000, 32, 32, 3)
            assert y_train.shape == (10000, 1)

    with pytest.raises(ValueError):
        datasets.cifar100.load_data(label_mode="invalid")


def test_mnist() -> None:
    """Test ``mnist`` behavior and edge cases in ``test_datasets``."""
    x_train_d, y_train_d = ops.zeros((60000, 28, 28)), ops.zeros((60000,))
    x_test_d, y_test_d = ops.zeros((10000, 28, 28)), ops.zeros((10000,))

    with mock.patch("zero_keras.datasets.mnist.get_file") as mock_get_file:
        mock_get_file.return_value = "dummy.npz"
        with mock.patch("zero_keras.ops.load") as mock_load:
            mock_load.return_value.__enter__.return_value = {
                "x_train": x_train_d,
                "y_train": y_train_d,
                "x_test": x_test_d,
                "y_test": y_test_d,
            }

            (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

            assert x_train.shape == (60000, 28, 28)
            assert y_train.shape == (60000,)
            assert x_test.shape == (10000, 28, 28)
            assert y_test.shape == (10000,)


def test_cifar_load_batch() -> None:
    """Test ``cifar_load_batch`` behavior and edge cases in ``test_datasets``."""
    import os
    import pickle

    dummy_data = ops.zeros((10, 3072), dtype="uint8")
    dummy_labels = [0] * 10
    dummy_dict = {b"data": dummy_data, b"labels": dummy_labels}

    with open("dummy_batch.bin", "wb") as f:
        pickle.dump(dummy_dict, f)

    try:
        data, labels = datasets.cifar10.load_batch("dummy_batch.bin")
        assert data.shape == (10, 3, 32, 32)
        assert len(labels) == 10
    finally:
        if os.path.exists("dummy_batch.bin"):
            os.remove("dummy_batch.bin")


def test_cifar100_load_batch() -> None:
    """Test ``cifar100_load_batch`` behavior and edge cases in ``test_datasets``."""
    import os
    import pickle

    dummy_data = ops.zeros((10, 3072), dtype="uint8")
    dummy_labels = [0] * 10
    dummy_dict = {b"data": dummy_data, b"fine_labels": dummy_labels}

    with open("dummy_batch100.bin", "wb") as f:
        pickle.dump(dummy_dict, f)

    try:
        data, labels = datasets.cifar100.load_batch("dummy_batch100.bin")
        assert data.shape == (10, 3, 32, 32)
        assert len(labels) == 10
    finally:
        if os.path.exists("dummy_batch100.bin"):
            os.remove("dummy_batch100.bin")


def test_fashion_mnist() -> None:
    """Test ``fashion_mnist`` behavior and edge cases in ``test_datasets``."""
    x_train_d = b"\x00" * (60000 * 28 * 28 + 16)
    y_train_d = b"\x00" * (60000 + 8)
    x_test_d = b"\x00" * (10000 * 28 * 28 + 16)
    y_test_d = b"\x00" * (10000 + 8)

    with mock.patch("zero_keras.datasets.fashion_mnist.get_file") as mock_get_file:
        mock_get_file.side_effect = ["p0", "p1", "p2", "p3"]
        with mock.patch("gzip.open") as mock_gzip:
            mock_file0 = mock.MagicMock()
            mock_file0.read.return_value = y_train_d
            mock_file1 = mock.MagicMock()
            mock_file1.read.return_value = x_train_d
            mock_file2 = mock.MagicMock()
            mock_file2.read.return_value = y_test_d
            mock_file3 = mock.MagicMock()
            mock_file3.read.return_value = x_test_d

            mock_gzip.return_value.__enter__.side_effect = [
                mock_file0,
                mock_file1,
                mock_file2,
                mock_file3,
            ]

            (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

            assert x_train.shape == (60000, 28, 28)
            assert y_train.shape == (60000,)
            assert x_test.shape == (10000, 28, 28)
            assert y_test.shape == (10000,)


def test_imdb() -> None:
    """Test ``imdb`` behavior and edge cases in ``test_datasets``."""
    x_train_d = [[1, 2, 3], [4, 5, 6], [1] * 50]
    y_train_d = [0, 1, 1]
    x_test_d = [[1, 2], [3, 4], [2] * 50]
    y_test_d = [0, 1, 0]

    with mock.patch("zero_keras.datasets.imdb.get_file") as mock_get_file:
        mock_get_file.return_value = "dummy.npz"
        with mock.patch("zero_keras.ops.load") as mock_load:
            mock_load.return_value.__enter__.return_value = {
                "x_train": x_train_d,
                "y_train": y_train_d,
                "x_test": x_test_d,
                "y_test": y_test_d,
            }

            (x_train, _y_train), (x_test, _y_test) = datasets.imdb.load_data()
            assert len(x_train) == 3
            assert len(x_test) == 3

            (x_train, _y_train), (x_test, _y_test) = datasets.imdb.load_data(
                maxlen=10, num_words=10, skip_top=1
            )
            assert len(x_train) == 2
            assert len(x_test) == 2

            (x_train, _y_train), (x_test, _y_test) = datasets.imdb.load_data(
                start_char=None, index_from=None, maxlen=10, oov_char=None
            )
            assert len(x_train) == 2

    with mock.patch("zero_keras.datasets.imdb.get_file") as mock_get_file:
        mock_get_file.return_value = "dummy.json"
        with mock.patch("builtins.open", mock.mock_open(read_data='{"test": 1}')):
            res = datasets.imdb.get_word_index()
            assert res == {"test": 1}


def test_reuters() -> None:
    """Test ``reuters`` behavior and edge cases in ``test_datasets``."""
    x_d = [[1, 2, 3], [4, 5, 6], [1] * 50, [2] * 3]
    y_d = [0, 1, 1, 0]

    with mock.patch("zero_keras.datasets.reuters.get_file") as mock_get_file:
        mock_get_file.return_value = "dummy.npz"
        with mock.patch("zero_keras.ops.load") as mock_load:
            mock_load.return_value.__enter__.return_value = {"x": x_d, "y": y_d}

            (x_train, _y_train), (x_test, _y_test) = datasets.reuters.load_data()
            assert len(x_train) == 3
            assert len(x_test) == 1

            (x_train, _y_train), (x_test, _y_test) = datasets.reuters.load_data(
                maxlen=10, num_words=10, skip_top=1
            )
            assert len(x_train) == 2
            assert len(x_test) == 1

            (x_train, _y_train), (x_test, _y_test) = datasets.reuters.load_data(
                start_char=None, index_from=None, maxlen=10, oov_char=None
            )
            assert len(x_train) == 2

    with mock.patch("zero_keras.datasets.reuters.get_file") as mock_get_file:
        mock_get_file.return_value = "dummy.json"
        with mock.patch("builtins.open", mock.mock_open(read_data='{"test": 1}')):
            res = datasets.reuters.get_word_index()
            assert res == {"test": 1}

    assert len(datasets.reuters.get_label_names()) == 46


def test_imdb_edge_cases() -> None:
    """Test ``imdb_edge_cases`` behavior and edge cases in ``test_datasets``."""
    x_train_d = [[1, 2, 3]]
    y_train_d = [0]
    x_test_d = [[1, 2]]
    y_test_d = [0]

    with mock.patch("zero_keras.datasets.imdb.get_file") as mock_get_file:
        mock_get_file.return_value = "dummy.npz"
        with mock.patch("zero_keras.ops.load") as mock_load:
            mock_load.return_value.__enter__.return_value = {
                "x_train": x_train_d,
                "y_train": y_train_d,
                "x_test": x_test_d,
                "y_test": y_test_d,
            }

            (x_train, _y_train), (_x_test, _y_test) = datasets.imdb.load_data(
                start_char=None, index_from=3
            )
            assert list(x_train[0]) == [4, 5, 2]

            with pytest.raises(ValueError):
                datasets.imdb.load_data(maxlen=1)


def test_reuters_edge_cases() -> None:
    """Test ``reuters_edge_cases`` behavior and edge cases in ``test_datasets``."""
    x_d = [[1, 2, 3]]
    y_d = [0]

    with mock.patch("zero_keras.datasets.reuters.get_file") as mock_get_file:
        mock_get_file.return_value = "dummy.npz"
        with mock.patch("zero_keras.ops.load") as mock_load:
            mock_load.return_value.__enter__.return_value = {"x": x_d, "y": y_d}

            (_x_train, _y_train), (x_test, _y_test) = datasets.reuters.load_data(
                start_char=None, index_from=3
            )
            assert list(x_test[0]) == [4, 5, 2]


def test_california_housing() -> None:
    """Test ``california_housing`` behavior and edge cases in ``test_datasets``."""
    x_dummy = random.uniform((100, 8))
    y_dummy = random.uniform((100,))

    with mock.patch("zero_keras.datasets.california_housing.get_file") as mock_get_file:
        mock_get_file.return_value = "dummy.npz"
        with mock.patch("zero_keras.ops.load") as mock_load:
            mock_load.return_value.__enter__.return_value = {"x": x_dummy, "y": y_dummy}

            (x_train, y_train), (x_test, y_test) = datasets.california_housing.load_data()

            assert x_train.shape == (80, 8)
            assert y_train.shape == (80,)
            assert x_test.shape == (20, 8)
            assert y_test.shape == (20,)

            datasets.california_housing.load_data(version="small")
