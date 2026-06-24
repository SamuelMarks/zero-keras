"""Module docstring."""

import pytest
from unittest import mock
import numpy as np
from zero_keras import datasets


def test_boston_housing():
    """Function docstring."""
    x_dummy = np.random.rand(100, 13)
    y_dummy = np.random.rand(100)

    with mock.patch("zero_keras.datasets.boston_housing.get_file") as mock_get_file:
        mock_get_file.return_value = "dummy.npz"
        with mock.patch("numpy.load") as mock_load:
            mock_load.return_value.__enter__.return_value = {"x": x_dummy, "y": y_dummy}

            (x_train, y_train), (x_test, y_test) = datasets.boston_housing.load_data()

            assert x_train.shape == (80, 13)
            assert y_train.shape == (80,)
            assert x_test.shape == (20, 13)
            assert y_test.shape == (20,)


def test_cifar10():
    """Function docstring."""
    x_dummy = np.random.rand(10000, 3, 32, 32)
    y_dummy = np.random.rand(10000)

    with mock.patch("zero_keras.datasets.cifar10.get_file") as mock_get_file:
        mock_get_file.return_value = "dummy_dir"
        with mock.patch("os.path.exists", return_value=False):
            with mock.patch(
                "zero_keras.datasets.cifar10.load_batch",
                return_value=(x_dummy, y_dummy),
            ):
                (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

                assert x_train.shape == (50000, 32, 32, 3)
                assert y_train.shape == (50000, 1)
                assert x_test.shape == (10000, 32, 32, 3)
                assert y_test.shape == (10000, 1)


def test_cifar100():
    """Function docstring."""
    x_dummy = np.random.rand(10000, 3, 32, 32)
    y_dummy = np.random.rand(10000)

    with mock.patch("zero_keras.datasets.cifar100.get_file") as mock_get_file:
        mock_get_file.return_value = "dummy_dir"
        with mock.patch("os.path.exists", return_value=False):
            with mock.patch(
                "zero_keras.datasets.cifar100.load_batch",
                return_value=(x_dummy, y_dummy),
            ):
                (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()

                assert x_train.shape == (
                    10000,
                    32,
                    32,
                    3,
                )  # Wait, cifar100 load_batch should return 50k for train
                assert y_train.shape == (10000, 1)


def test_cifar100_coarse():
    """Function docstring."""
    x_dummy = np.random.rand(10000, 3, 32, 32)
    y_dummy = np.random.rand(10000)

    with mock.patch("zero_keras.datasets.cifar100.get_file") as mock_get_file:
        mock_get_file.return_value = "dummy_dir"
        with mock.patch("os.path.exists", return_value=False):
            with mock.patch(
                "zero_keras.datasets.cifar100.load_batch",
                return_value=(x_dummy, y_dummy),
            ):
                (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data(
                    label_mode="coarse"
                )

                assert x_train.shape == (10000, 32, 32, 3)
                assert y_train.shape == (10000, 1)

    with pytest.raises(ValueError):
        datasets.cifar100.load_data(label_mode="invalid")


def test_mnist():
    """Function docstring."""
    x_train_d, y_train_d = np.zeros((60000, 28, 28)), np.zeros((60000,))
    x_test_d, y_test_d = np.zeros((10000, 28, 28)), np.zeros((10000,))

    with mock.patch("zero_keras.datasets.mnist.get_file") as mock_get_file:
        mock_get_file.return_value = "dummy.npz"
        with mock.patch("numpy.load") as mock_load:
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


def test_cifar_load_batch():
    """Function docstring."""
    import pickle
    import os

    # Create dummy data
    dummy_data = np.zeros((10, 3072), dtype="uint8")
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


def test_cifar100_load_batch():
    """Function docstring."""
    import pickle
    import os

    # Create dummy data
    dummy_data = np.zeros((10, 3072), dtype="uint8")
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


def test_fashion_mnist():
    """Function docstring."""
    x_train_d = np.zeros(60000 * 28 * 28 + 16, dtype=np.uint8).tobytes()
    y_train_d = np.zeros(60000 + 8, dtype=np.uint8).tobytes()
    x_test_d = np.zeros(10000 * 28 * 28 + 16, dtype=np.uint8).tobytes()
    y_test_d = np.zeros(10000 + 8, dtype=np.uint8).tobytes()

    with mock.patch("zero_keras.datasets.fashion_mnist.get_file") as mock_get_file:
        mock_get_file.side_effect = ["p0", "p1", "p2", "p3"]
        with mock.patch("gzip.open") as mock_gzip:
            # We mock the return value of gzip.open().read() sequentially
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


def test_imdb():
    """Function docstring."""
    x_train_d = np.array([[1, 2, 3], [4, 5, 6], [1] * 50], dtype="object")
    y_train_d = np.array([0, 1, 1])
    x_test_d = np.array([[1, 2], [3, 4], [2] * 50], dtype="object")
    y_test_d = np.array([0, 1, 0])

    with mock.patch("zero_keras.datasets.imdb.get_file") as mock_get_file:
        mock_get_file.return_value = "dummy.npz"
        with mock.patch("numpy.load") as mock_load:
            mock_load.return_value.__enter__.return_value = {
                "x_train": x_train_d,
                "y_train": y_train_d,
                "x_test": x_test_d,
                "y_test": y_test_d,
            }

            # test standard
            (x_train, y_train), (x_test, y_test) = datasets.imdb.load_data()
            assert len(x_train) == 3
            assert len(x_test) == 3

            # test kwargs (maxlen)
            (x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(
                maxlen=10, num_words=10, skip_top=1
            )
            assert len(x_train) == 2
            assert len(x_test) == 2

            # test no index_from
            (x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(
                start_char=None, index_from=None, maxlen=10, oov_char=None
            )
            assert len(x_train) == 2

    with mock.patch("zero_keras.datasets.imdb.get_file") as mock_get_file:
        mock_get_file.return_value = "dummy.json"
        with mock.patch("builtins.open", mock.mock_open(read_data='{"test": 1}')):
            res = datasets.imdb.get_word_index()
            assert res == {"test": 1}


def test_reuters():
    """Function docstring."""
    x_d = np.array([[1, 2, 3], [4, 5, 6], [1] * 50, [2] * 3], dtype="object")
    y_d = np.array([0, 1, 1, 0])

    with mock.patch("zero_keras.datasets.reuters.get_file") as mock_get_file:
        mock_get_file.return_value = "dummy.npz"
        with mock.patch("numpy.load") as mock_load:
            mock_load.return_value.__enter__.return_value = {"x": x_d, "y": y_d}

            # test standard
            (x_train, y_train), (x_test, y_test) = datasets.reuters.load_data()
            assert len(x_train) == 3
            assert len(x_test) == 1

            # test kwargs (maxlen)
            (x_train, y_train), (x_test, y_test) = datasets.reuters.load_data(
                maxlen=10, num_words=10, skip_top=1
            )
            assert len(x_train) == 2
            assert len(x_test) == 1

            # test no index_from
            (x_train, y_train), (x_test, y_test) = datasets.reuters.load_data(
                start_char=None, index_from=None, maxlen=10, oov_char=None
            )
            assert len(x_train) == 2

    with mock.patch("zero_keras.datasets.reuters.get_file") as mock_get_file:
        mock_get_file.return_value = "dummy.json"
        with mock.patch("builtins.open", mock.mock_open(read_data='{"test": 1}')):
            res = datasets.reuters.get_word_index()
            assert res == {"test": 1}

    assert len(datasets.reuters.get_label_names()) == 46


def test_imdb_edge_cases():
    """Function docstring."""
    x_train_d = np.array([[1, 2, 3]], dtype="object")
    y_train_d = np.array([0])
    x_test_d = np.array([[1, 2]], dtype="object")
    y_test_d = np.array([0])

    with mock.patch("zero_keras.datasets.imdb.get_file") as mock_get_file:
        mock_get_file.return_value = "dummy.npz"
        with mock.patch("numpy.load") as mock_load:
            mock_load.return_value.__enter__.return_value = {
                "x_train": x_train_d,
                "y_train": y_train_d,
                "x_test": x_test_d,
                "y_test": y_test_d,
            }

            # test start_char=None but index_from > 0
            (x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(
                start_char=None, index_from=3
            )
            assert list(x_train[0]) == [4, 5, 2]

            # test value error
            with pytest.raises(ValueError):
                datasets.imdb.load_data(maxlen=1)


def test_reuters_edge_cases():
    """Function docstring."""
    x_d = np.array([[1, 2, 3]], dtype="object")
    y_d = np.array([0])

    with mock.patch("zero_keras.datasets.reuters.get_file") as mock_get_file:
        mock_get_file.return_value = "dummy.npz"
        with mock.patch("numpy.load") as mock_load:
            mock_load.return_value.__enter__.return_value = {"x": x_d, "y": y_d}

            # test start_char=None but index_from > 0
            (x_train, y_train), (x_test, y_test) = datasets.reuters.load_data(
                start_char=None, index_from=3
            )
            # test_split=0.2 so 1 sample -> 1 * 0.8 = 0 for train, 1 for test
            assert list(x_test[0]) == [4, 5, 2]
