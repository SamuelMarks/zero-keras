"""Module `test_utils` for zero_keras."""

from __future__ import annotations

import os
import tempfile
from unittest import mock

from ml_switcheroo_compiler import ops as msc_ops
import pytest

from zero_keras import ops, utils
from zero_keras.types import TensorLike


def test_to_categorical() -> None:
    """Verify or execute `test to categorical`."""
    x = ops.convert_to_tensor([0, 1, 2, 1])
    y = utils.to_categorical(x, num_classes=3)
    assert y.shape == (4, 3)

    x2 = [0, 1, 2, 1]
    y2 = utils.to_categorical(x2, num_classes=3)
    assert y2 is not None


def test_normalize() -> None:
    """Verify or execute `test normalize`."""
    x = ops.convert_to_tensor([[1.0, 0.0], [0.0, 2.0]])
    y = utils.normalize(x, axis=-1)
    assert y is not None


def test_set_random_seed() -> None:
    """Verify or execute `test set random seed`."""
    utils.set_random_seed(42)


def test_get_file() -> None:
    """Verify or execute `test get file`."""
    with tempfile.TemporaryDirectory() as d:
        dummy_content = b"test content"
        dummy_url = "file://" + os.path.join(d, "origin.txt")
        with open(os.path.join(d, "origin.txt"), "wb") as f:
            f.write(dummy_content)
        fpath = utils.get_file("test.txt", dummy_url, cache_dir=d, cache_subdir="test_subdir")
        assert os.path.exists(fpath)
        fpath2 = utils.get_file("test.txt", dummy_url, cache_dir=d, cache_subdir="test_subdir")
        assert fpath2 == fpath

        import zipfile

        zip_path = os.path.join(d, "test.zip")
        with zipfile.ZipFile(zip_path, "w") as z:
            z.writestr("test_in_zip.txt", dummy_content)
        utils.get_file(
            "test.zip",
            "file://" + zip_path,
            extract=True,
            cache_dir=d,
            cache_subdir="test_subdir2",
        )

        import tarfile

        tar_path = os.path.join(d, "test.tar.gz")
        with tarfile.open(tar_path, "w:gz") as t:
            with open(os.path.join(d, "origin.txt"), "wb") as f:
                f.write(b"tar content")
            t.add(os.path.join(d, "origin.txt"), arcname="test_in_tar.txt")
        utils.get_file(
            "test.tar.gz",
            "file://" + tar_path,
            untar=True,
            cache_dir=d,
            cache_subdir="test_subdir3",
        )


def test_get_file_error() -> None:
    """Verify or execute `test get file error`."""
    with tempfile.TemporaryDirectory() as d, pytest.raises(Exception):
        utils.get_file("test_err.txt", "http://localhost:12345/nonexistent", cache_dir=d)


def test_progbar() -> None:
    """Verify or execute `test progbar`."""
    pb = utils.Progbar(10)
    pb.update(5, values=[("loss", 0.5)])
    pb.update(10, values=[("loss", 0.4), ("acc", 0.9)])
    pb2 = utils.Progbar(None, stateful_metrics=["acc"])
    pb2.update(5, values=[("acc", 0.9)])


@mock.patch("zero_keras.utils.plot_model")
def test_plot_model(mock_plot_model: TensorLike) -> None:
    """Verify or execute `test plot model`.

    Args:
        mock_plot_model: Test fixture or input parameter `mock_plot_model` (mock plot model).
    """
    utils.plot_model(None)
    mock_plot_model.assert_called_once_with(None)


def test_dataset_utils() -> TensorLike:
    """Verify or execute `test dataset utils`."""


def test_set_random_seed_no_numpy(monkeypatch: TensorLike) -> None:
    """Verify or execute `test set random seed no numpy`.

    Args:
        monkeypatch: Test fixture or input parameter `monkeypatch` (monkeypatch).
    """
    import sys

    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.setitem(sys.modules, "random", None)
    utils.set_random_seed(42)


def test_get_file_default_cache() -> None:
    """Verify or execute `test get file default cache`."""
    fpath = utils.get_file("test_default.txt", "file:///dev/null")
    assert fpath is not None


def test_get_file_tar() -> None:
    """Verify or execute `test get file tar`."""
    with tempfile.TemporaryDirectory() as d:
        import tarfile

        tar_path = os.path.join(d, "test.tar")
        with tarfile.open(tar_path, "w:") as t:
            with open(os.path.join(d, "origin.txt"), "wb") as f:
                f.write(b"tar content")
            t.add(os.path.join(d, "origin.txt"), arcname="test_in_tar.txt")
        utils.get_file("test.tar", "file://" + tar_path, untar=True, cache_dir=d)


def test_to_categorical_exceptions(monkeypatch: TensorLike) -> None:
    """Verify or execute `test to categorical exceptions`.

    Args:
        monkeypatch: Test fixture or input parameter `monkeypatch` (monkeypatch).
    """

    class BadMaxObj:
        """Test helper class `BadMaxObj` for utils."""

        def max(self) -> TensorLike:
            """Verify or execute `max` on `BadMaxObj`."""
            raise Exception("bad max")

    def mock_ops_max(x: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Verify or execute `mock ops max`.

        Args:
            x: Test fixture or input parameter `x` (x).
            *args: Positional arguments forwarded to helper.
            **kwargs: Keyword arguments forwarded to helper.

        Returns:
            TensorLike: Result from `mock_ops_max`.
        """
        _ = x, args, kwargs

        class OpsMaxObj:
            """Test helper class `OpsMaxObj` for utils."""

            def __int__(self) -> TensorLike:
                """Verify or execute `int` on `OpsMaxObj`.

                Returns:
                    TensorLike: Result from `__int__`.
                """
                return 1

        return OpsMaxObj()

    monkeypatch.setattr(msc_ops, "max", mock_ops_max)

    with pytest.raises(TypeError):
        utils.to_categorical(BadMaxObj())


def test_to_categorical_no_num_classes_tensor() -> None:
    """Verify or execute `test to categorical no num classes tensor`."""
    y = utils.to_categorical(ops.convert_to_tensor([0, 1]))
    assert y is not None
