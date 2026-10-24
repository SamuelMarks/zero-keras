"""Comprehensive tests for ``zero_keras.distribution``."""

from __future__ import annotations

import pytest

from zero_keras import config, distribution


def test_distribution_methods() -> None:
    """Verify DataParallel, DeviceMesh, LayoutMap, ModelParallel, and TensorLayout."""
    from zero_keras.distribution import (
        DataParallel,
        DeviceMesh,
        LayoutMap,
        ModelParallel,
        TensorLayout,
    )

    mesh_valid = DeviceMesh(shape=(2,), axis_names=["batch"], devices=["cpu:0", "cpu:1"])
    assert mesh_valid.shape == (2,)
    assert mesh_valid.axis_names == ("batch",)
    assert mesh_valid.devices == ("cpu:0", "cpu:1")
    assert mesh_valid.backend_mesh is None

    with pytest.raises(ValueError, match="Shape length"):
        DeviceMesh(shape=(2, 2), axis_names=["batch"])
    with pytest.raises(ValueError, match="Device count"):
        DeviceMesh(shape=(2, 2), axis_names=["a", "b"], devices=["cpu:0"])

    dp = DataParallel(device_mesh=mesh_valid)
    assert dp.batch_dim_name == "batch"
    assert dp.device_mesh is mesh_valid
    assert dp.distribute_dataset("dataset") == "dataset"
    assert dp.get_data_layout((1,)) is None
    assert dp.get_tensor_layout((1,)) is None
    assert dp.get_variable_layout(None) is None

    # Scope activates and restores distribution
    distribution.set_distribution(None)
    assert distribution.distribution() is None
    with dp.scope():
        assert distribution.distribution() is dp
    assert distribution.distribution() is None

    dm = DeviceMesh()
    assert dm.axis_names is None
    assert dm.backend_mesh is None
    assert dm.devices is None
    assert dm.shape is None

    lm = LayoutMap(device_mesh=mesh_valid)
    assert list(iter(lm)) == []
    assert len(lm) == 0
    assert lm.device_mesh is mesh_valid

    tl_explicit = TensorLayout(axes=["batch", None], device_mesh=mesh_valid)
    lm["dense.*kernel"] = tl_explicit
    lm["dense.*bias"] = ["batch"]
    assert lm["dense.*kernel"] is tl_explicit
    assert lm["dense_1/kernel"] is tl_explicit
    assert isinstance(lm["dense_1/bias"], TensorLayout)
    assert lm["unmatched/path"] is None

    assert len(lm) == 2
    assert len(list(lm.items())) == 2
    assert len(list(lm.keys())) == 2
    assert len(list(lm.values())) == 2
    assert lm.setdefault("dense.*kernel", None) is tl_explicit
    assert lm.setdefault("new_key", tl_explicit) is tl_explicit
    assert lm.pop("new_key") is tl_explicit
    assert lm.pop("missing_key") is None
    popped = lm.popitem()
    assert popped[0] in ("dense.*kernel", "dense.*bias")
    lm.clear()
    assert len(lm) == 0
    assert lm.setdefault("empty_default", None) is None
    lm.update({"conv.*": tl_explicit})
    assert lm["conv_1/weights"] is tl_explicit

    mp = ModelParallel(device_mesh=mesh_valid, layout_map=lm, batch_dim_name="batch")
    assert mp.batch_dim_name == "batch"
    assert mp.device_mesh is mesh_valid
    assert mp.layout_map is lm
    assert mp.distribute_dataset("dataset") == "dataset"
    assert mp.get_data_layout((1,)) is None
    assert mp.get_tensor_layout("conv_1/weights") is tl_explicit
    assert mp.get_tensor_layout((1,)) is None
    assert mp.get_variable_layout(None) is None
    with mp.scope():
        assert distribution.distribution() is mp

    tl = TensorLayout()
    assert tl.axes is None
    assert tl.backend_layout is None
    assert tl.device_mesh is None


def test_all_distribution_coverage() -> None:
    """Verify module-level functions and backend fallback in distribution."""
    dp = distribution.DataParallel()
    assert dp.batch_dim_name is None
    assert dp.device_mesh is None
    assert dp.distribute_dataset(None) is None
    assert dp.get_data_layout(None) is None
    assert dp.get_tensor_layout(None) is None
    assert dp.get_variable_layout(None) is None

    assert distribution.distribute_tensor("a", None) == "a"
    distribution.set_distribution(None)
    assert distribution.distribution() is None

    config.set_backend("jax")
    assert isinstance(distribution.distribution(), distribution.DataParallel)
    config.set_backend("numpy")

    distribution.initialize()
    assert distribution.list_devices() == []
    distribution.set_distribution(None)
