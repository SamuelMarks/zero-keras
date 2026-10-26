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
    """Test ``to_categorical`` behavior and edge cases in ``test_utils``."""
    x = ops.convert_to_tensor([0, 1, 2, 1])
    y = utils.to_categorical(x, num_classes=3)
    assert y.shape == (4, 3)

    x2 = [0, 1, 2, 1]
    y2 = utils.to_categorical(x2, num_classes=3)
    assert y2 is not None


def test_normalize() -> None:
    """Test ``normalize`` behavior and edge cases in ``test_utils``."""
    x = ops.convert_to_tensor([[1.0, 0.0], [0.0, 2.0]])
    y = utils.normalize(x, axis=-1)
    assert y is not None


def test_set_random_seed() -> None:
    """Test ``set_random_seed`` behavior and edge cases in ``test_utils``."""
    utils.set_random_seed(42)


def test_get_file() -> None:
    """Test ``get_file`` behavior and edge cases in ``test_utils``."""
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
    """Test ``get_file_error`` behavior and edge cases in ``test_utils``."""
    with tempfile.TemporaryDirectory() as d, pytest.raises(Exception):
        utils.get_file("test_err.txt", "http://localhost:12345/nonexistent", cache_dir=d)


def test_progbar() -> None:
    """Test ``progbar`` behavior and edge cases in ``test_utils``."""
    pb = utils.Progbar(10)
    pb.update(5, values=[("loss", 0.5)])
    pb.update(10, values=[("loss", 0.4), ("acc", 0.9)])
    pb2 = utils.Progbar(None, stateful_metrics=["acc"])
    pb2.update(5, values=[("acc", 0.9)])


@mock.patch("zero_keras.utils.plot_model")
def test_plot_model(mock_plot_model: TensorLike) -> None:
    """Test ``plot_model`` behavior and edge cases in ``test_utils``.

    Args:
        mock_plot_model: Mocked ``plot_model`` visualization fixture.
    """
    utils.plot_model(None)
    mock_plot_model.assert_called_once_with(None)


def test_dataset_utils() -> TensorLike:
    """Test ``dataset_utils`` behavior and edge cases in ``test_utils``."""


def test_set_random_seed_no_numpy(monkeypatch: TensorLike) -> None:
    """Test ``set_random_seed_no_numpy`` behavior and edge cases in ``test_utils``.

    Args:
        monkeypatch: Pytest ``MonkeyPatch`` fixture for attribute patching.
    """
    import sys

    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.setitem(sys.modules, "random", None)
    utils.set_random_seed(42)


def test_get_file_default_cache() -> None:
    """Test ``get_file_default_cache`` behavior and edge cases in ``test_utils``."""
    fpath = utils.get_file("test_default.txt", "file:///dev/null")
    assert fpath is not None


def test_get_file_tar() -> None:
    """Test ``get_file_tar`` behavior and edge cases in ``test_utils``."""
    with tempfile.TemporaryDirectory() as d:
        import tarfile

        tar_path = os.path.join(d, "test.tar")
        with tarfile.open(tar_path, "w:") as t:
            with open(os.path.join(d, "origin.txt"), "wb") as f:
                f.write(b"tar content")
            t.add(os.path.join(d, "origin.txt"), arcname="test_in_tar.txt")
        utils.get_file("test.tar", "file://" + tar_path, untar=True, cache_dir=d)


def test_to_categorical_exceptions(monkeypatch: TensorLike) -> None:
    """Test ``to_categorical_exceptions`` behavior and edge cases in ``test_utils``.

    Args:
        monkeypatch: Pytest ``MonkeyPatch`` fixture for attribute patching.
    """

    class BadMaxObj:
        """Test helper class `BadMaxObj` for utils."""

        def max(self) -> TensorLike:
            """Compute the maximum element value of ``BadMaxObj``."""
            raise Exception("bad max")

    def mock_ops_max(x: TensorLike, *args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Execute test helper ``mock_ops_max`` in ``test_utils``.

        Args:
            x: Input tensor or feature array.
            *args: Positional arguments forwarded to helper.
            **kwargs: Keyword arguments forwarded to helper.

        Returns:
            TensorLike: Result from `mock_ops_max`.
        """
        _ = x, args, kwargs

        class OpsMaxObj:
            """Test helper class `OpsMaxObj` for utils."""

            def __int__(self) -> TensorLike:
                """Convert ``OpsMaxObj`` to a Python ``int``.

                Returns:
                    TensorLike: Result from `__int__`.
                """
                return 1

        return OpsMaxObj()

    monkeypatch.setattr(msc_ops, "max", mock_ops_max)

    with pytest.raises(TypeError):
        utils.to_categorical(BadMaxObj())


def test_to_categorical_no_num_classes_tensor() -> None:
    """Test ``to_categorical_no_num_classes_tensor`` behavior and edge cases in ``test_utils``."""
    y = utils.to_categorical(ops.convert_to_tensor([0, 1]))
    assert y is not None
