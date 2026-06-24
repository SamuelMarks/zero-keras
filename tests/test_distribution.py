"""Module docstring."""

from zero_keras.distribution import (
    DeviceMesh,
    LayoutMap,
    Distribution,
    DataParallel,
    ModelParallel,
    distribute_tensor,
    distribution,
)


def test_distribution():
    """Function docstring."""
    mesh = DeviceMesh((2, 2), ("x", "y"))
    assert mesh.shape == (2, 2)
    assert mesh.axis_names == ("x", "y")

    lm = LayoutMap(mesh)
    lm["a"] = "b"
    assert lm["a"] == "b"
    assert lm.get("c") is None

    from ml_switcheroo_compiler.core import config

    config.eager_mode = False

    try:
        dist = Distribution(mesh)
        with dist.scope():
            assert distribution() is dist
            # When eager_mode is false, distribute_tensor will call shard_tensor.
            import unittest.mock as mock

            with mock.patch(
                "ml_switcheroo_compiler.ops.distributed.shard_tensor"
            ) as mock_shard:

                class FakeData:
                    """Class docstring."""

                    id = 1

                class FakeTensor:
                    """Class docstring."""

                    data = FakeData()
                    shape = (2, 2)
                    dtype = "float32"

                class FakeInput:
                    """Class docstring."""

                    _tensor = FakeTensor()
                    shape = (2, 2)

                mock_shard.return_value = FakeTensor()
                res = distribute_tensor(FakeInput(), None)
                assert res is not None
                assert mock_shard.called
    finally:
        config.eager_mode = True

    assert distribution() is None
    assert distribute_tensor(1, "x") == 1

    with dist.scope():
        assert distribute_tensor(2, "x") == 2

    dp = DataParallel(mesh)
    assert dp is not None
    assert dp is not None
    mp = ModelParallel(mesh, lm, "batch")
    assert mp.layout_map is lm
