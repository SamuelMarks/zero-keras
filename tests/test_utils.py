"""Module docstring."""

import pytest
import numpy as np
import os
import tempfile
from zero_keras import utils
import ml_switcheroo_compiler.ops as ops


def test_to_categorical():
    """Function docstring."""
    x = np.array([0, 1, 2, 1])
    y = utils.to_categorical(x, num_classes=3)
    assert y.shape == (4, 3)

    x2 = [0, 1, 2, 1]
    y2 = utils.to_categorical(x2, num_classes=3)
    assert y2 is not None


def test_normalize():
    """Function docstring."""
    x = np.array([[1.0, 0.0], [0.0, 2.0]])
    y = utils.normalize(x, axis=-1)
    assert y is not None


def test_set_random_seed():
    """Function docstring."""
    utils.set_random_seed(42)


def test_get_file():
    """Function docstring."""
    with tempfile.TemporaryDirectory() as d:
        dummy_content = b"test content"
        dummy_url = "file://" + os.path.join(d, "origin.txt")
        with open(os.path.join(d, "origin.txt"), "wb") as f:
            f.write(dummy_content)
        fpath = utils.get_file(
            "test.txt", dummy_url, cache_dir=d, cache_subdir="test_subdir"
        )
        assert os.path.exists(fpath)
        fpath2 = utils.get_file(
            "test.txt", dummy_url, cache_dir=d, cache_subdir="test_subdir"
        )
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


def test_get_file_error():
    """Function docstring."""
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(Exception):
            utils.get_file(
                "test_err.txt", "http://localhost:12345/nonexistent", cache_dir=d
            )


def test_progbar():
    """Function docstring."""
    pb = utils.Progbar(10)
    pb.update(5, values=[("loss", 0.5)])
    pb.update(10, values=[("loss", 0.4), ("acc", 0.9)])
    pb2 = utils.Progbar(None, stateful_metrics=["acc"])
    pb2.update(5, values=[("acc", 0.9)])


def test_plot_model():
    """Function docstring."""
    utils.plot_model(None)


def test_dataset_utils():
    """Function docstring."""


def test_set_random_seed_no_numpy(monkeypatch):
    """Function docstring.

    Args:
        monkeypatch: Description.
    """
    import sys

    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.setitem(sys.modules, "random", None)
    utils.set_random_seed(42)


def test_get_file_default_cache():
    """Function docstring."""
    fpath = utils.get_file("test_default.txt", "file:///dev/null")
    assert fpath is not None


def test_get_file_tar():
    """Function docstring."""
    with tempfile.TemporaryDirectory() as d:
        import tarfile

        tar_path = os.path.join(d, "test.tar")
        with tarfile.open(tar_path, "w:") as t:
            with open(os.path.join(d, "origin.txt"), "wb") as f:
                f.write(b"tar content")
            t.add(os.path.join(d, "origin.txt"), arcname="test_in_tar.txt")
        utils.get_file("test.tar", "file://" + tar_path, untar=True, cache_dir=d)


def test_to_categorical_exceptions(monkeypatch):
    """Function docstring.

    Args:
        monkeypatch: Description.
    """

    class BadMaxObj:
        """Class docstring."""

        def max(self):
            """Function docstring."""
            raise Exception("bad max")  # pragma: no cover

    original_ops_max = ops.max

    def mock_ops_max(x, *args, **kwargs):
        """Function docstring.

        Args:
            x: Description.
            args: Description.
            kwargs: Description.
        """

        class OpsMaxObj:
            """Class docstring."""

            def __int__(self):
                """Function docstring."""
                return 1

        return OpsMaxObj()

    monkeypatch.setattr(ops, "max", mock_ops_max)

    with pytest.raises(TypeError):
        utils.to_categorical(BadMaxObj())


def test_to_categorical_no_num_classes_tensor():
    """Function docstring."""
    y = utils.to_categorical(np.array([0, 1]))
    assert y is not None
