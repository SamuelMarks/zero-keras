"""Unit and branch coverage tests for utils, random, constraints, schedules, distribution, dtype_policies, activations, and initializers."""

from __future__ import annotations

from pathlib import Path
import struct
from unittest import mock

from ml_switcheroo_compiler.core.dtype import DType

from zero_keras import (
    activations,
    config,
    constraints,
    distribution,
    dtype_policies,
    initializers,
    layers,
    models,
    ops,
    random,
)
from zero_keras.core_layers import KerasTensor
from zero_keras.optimizers import schedules
from zero_keras.types import TensorLike
from zero_keras.utils import dataset_utils, generic_utils, vis_utils


def test_generic_utils_and_vis_utils_coverage(tmp_path: Path) -> None:
    """Cover all classes, functions, and branches in ``generic_utils`` and ``vis_utils``.

    Args:
        tmp_path: Temporary directory fixture.
    """

    class MyDataset(generic_utils.PyDataset):
        """Concrete PyDataset implementation."""

        def __getitem__(self, index: int) -> TensorLike:
            """Return batch item.

            Args:
                index: Batch index.

            Returns:
                TensorLike: Batch value.
            """
            return index * 10

        def __len__(self) -> int:
            """Return dataset length.

            Returns:
                int: Number of batches.
            """
            return 2

    ds = MyDataset(workers=2, use_multiprocessing=False, max_queue_size=5)
    assert list(ds) == [0, 10]
    assert ds.on_epoch_end() is None

    fs = generic_utils.FeatureSpace(
        features={"a": "float"},
        output_mode="concat",
        crosses=[("a", "b")],
    )
    fs.adapt([1, 2, 3])
    assert fs.built is True
    assert fs({"a": 1}) == {"a": 1}
    cfg = fs.get_config()
    fs2 = generic_utils.FeatureSpace.from_config(cfg)
    assert fs2.name == "feature_space"

    kt = KerasTensor((2, 2))
    assert generic_utils.is_keras_tensor(kt) is True
    assert generic_utils.is_keras_tensor(123) is False

    generic_utils.clear_session()
    assert generic_utils.get_uid("layer") == 1
    assert generic_utils.get_uid("layer") == 2

    # vis_utils: DotGraph, model_to_dot, plot_model
    dot_empty = vis_utils.model_to_dot(None, dpi=None)
    assert "rankdir" in dot_empty.to_string()

    seq = models.Sequential([layers.Dense(4, name="d1"), layers.Dense(2, name="d2")], name="seq")
    dot_seq = vis_utils.model_to_dot(seq, show_layer_names=True, dpi=96)
    assert "d1" in dot_seq.to_string()
    dot_no_names = vis_utils.model_to_dot(seq, show_layer_names=False)
    assert "Dense" in dot_no_names.to_string()

    class UnnamedLayer:
        """Layer without name property."""

    class CustomContainer:
        """Container with unnamed layer."""

        name = "c"
        layers = [UnnamedLayer()]

    dot_custom = vis_utils.model_to_dot(CustomContainer())  # type: ignore[arg-type]
    assert "layer_0" in dot_custom.to_string()

    out_plot = tmp_path / "subdir" / "model.png"
    vis_utils.plot_model(seq, to_file=out_plot)
    assert out_plot.exists()
    vis_utils.plot_model(seq, to_file="model_local.png")
    Path("model_local.png").unlink(missing_ok=True)
    vis_utils.plot_model(seq, to_file="")


def test_dataset_utils_coverage(tmp_path: Path) -> None:
    """Cover all dataset loaders and pure-Python image parsing branches in ``dataset_utils``.

    Args:
        tmp_path: Temporary directory fixture.
    """
    with mock.patch("zero_keras.utils.dataset_utils.msc_audio", return_value="audio_ds"):
        assert dataset_utils.audio_dataset_from_directory(tmp_path, class_names=["a"]) == "audio_ds"
    with mock.patch("zero_keras.utils.dataset_utils.msc_image", return_value="img_ds"):
        assert dataset_utils.image_dataset_from_directory(tmp_path, class_names=["a"]) == "img_ds"
    with mock.patch("zero_keras.utils.dataset_utils.msc_text", return_value="txt_ds"):
        assert dataset_utils.text_dataset_from_directory(tmp_path, class_names=["a"]) == "txt_ds"
    with mock.patch("zero_keras.utils.dataset_utils.msc_timeseries", return_value="ts_ds"):
        assert dataset_utils.timeseries_dataset_from_array([1, 2, 3], None, 2) == "ts_ds"

    nds = dataset_utils.NumpyDataset([[1.0, 2.0]], [1.0], batch_size=1, seed=42)
    assert nds is not None

    # GenericImage 1D / non-3D shape & convert & thumbnail branches
    gi_1d = dataset_utils.GenericImage([1.0, 2.0], mode="RGB")
    assert gi_1d.size == (0, 0)
    assert gi_1d.convert("L").mode == "L"
    gi_1d.thumbnail((10, 10))

    gi_3d = dataset_utils.GenericImage(ops.zeros((4, 6, 3)), mode="RGB")
    assert gi_3d.convert("RGB").mode == "RGB"
    assert gi_3d.convert("L").mode == "L"
    assert gi_3d.convert("RGBA").mode == "RGBA"
    gi_3d.thumbnail((2, 2))

    # _parse_image_dimensions for PNG, GIF, BMP, JPEG, and fallback
    png_bytes = (
        bytes((0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A))
        + b"\x00" * 8
        + struct.pack(">II", 12, 8)
        + b"\x08\x02"
    )
    assert dataset_utils._parse_image_dimensions(png_bytes) == (12, 8, 3)

    gif_bytes = b"GIF89a" + struct.pack("<HH", 16, 10)
    assert dataset_utils._parse_image_dimensions(gif_bytes) == (16, 10, 3)

    bmp_bytes = b"BM" + b"\x00" * 16 + struct.pack("<ii", 20, -15)
    assert dataset_utils._parse_image_dimensions(bmp_bytes) == (20, 15, 3)

    jpeg_sof = (
        b"\xff\xd8"
        + b"\x00"  # non-0xff byte
        + b"\xff\xff"  # padded 0xff
        + b"\xff\xd0"  # RST0 standalone marker
        + b"\xff\xe0"
        + struct.pack(">H", 4)
        + b"\x00\x00"  # APP0 segment
        + b"\xff\xc0"
        + struct.pack(">H", 11)
        + b"\x08"
        + struct.pack(">HHB", 24, 32, 1)
        + b"\x00" * 10
    )
    assert dataset_utils._parse_image_dimensions(jpeg_sof) == (32, 24, 1)

    jpeg_trunc = b"\xff\xd8" + b"\xff\xe0" + b"\x00" * 10
    assert dataset_utils._parse_image_dimensions(jpeg_trunc) == (256, 256, 3)
    jpeg_short_seg = b"\xff\xd8\x00\x00\x00\x00\xff\xc0\x00"
    assert dataset_utils._parse_image_dimensions(jpeg_short_seg) == (256, 256, 3)

    assert dataset_utils._parse_image_dimensions(b"unknown") == (256, 256, 3)

    # array_to_img & img_to_array & load_img branches
    img_2d = dataset_utils.array_to_img(ops.zeros((4, 4)), scale=False)
    assert img_2d.mode == "L"
    arr_from_2d = dataset_utils.img_to_array(ops.zeros((4, 4)), data_format="channels_first")
    assert arr_from_2d.shape == (1, 4, 4)

    gray_path = tmp_path / "gray.zk"
    gray_header = struct.pack(">4sHHH", b"ZKIM", 8, 8, 1)
    gray_path.write_bytes(gray_header)
    loaded_rgb = dataset_utils.load_img(gray_path, color_mode="rgb")
    assert loaded_rgb.mode == "RGB"
    loaded_gray = dataset_utils.load_img(gray_path, color_mode="grayscale")
    assert loaded_gray.mode == "L"
    loaded_rgba_already = dataset_utils.load_img(gray_path, color_mode="rgba")
    assert loaded_rgba_already.mode == "RGBA"

    rgba_path = tmp_path / "rgba.zk"
    rgba_path.write_bytes(struct.pack(">4sHHH", b"ZKIM", 8, 8, 4))
    assert dataset_utils.load_img(rgba_path, color_mode="rgba").mode == "RGBA"
    assert dataset_utils.load_img(rgba_path, color_mode="other", target_size=(8, 8)).mode == "RGBA"
    rgb_path = tmp_path / "rgb.zk"
    rgb_path.write_bytes(struct.pack(">4sHHH", b"ZKIM", 8, 8, 3))
    assert dataset_utils.load_img(rgb_path, color_mode="rgb").mode == "RGB"
    assert dataset_utils.load_img(
        rgb_path, color_mode="rgb", target_size=(4, 4), keep_aspect_ratio=True
    ).size == (4, 4)

    # smart_resize aspect == target_aspect and aspect < target_aspect
    same_aspect = dataset_utils.smart_resize(ops.zeros((10, 10, 3)), (5, 5))
    assert same_aspect.shape == (5, 5, 3)
    tall_aspect = dataset_utils.smart_resize(ops.zeros((20, 10, 3)), (5, 5))
    assert tall_aspect.shape == (5, 5, 3)


def test_random_constraints_schedules_distribution_dtype_activations_initializers() -> None:
    """Cover remaining lines/branches in random, constraints, schedules, distribution, dtype_policies, activations, and initializers."""
    # random
    rs = random.RandomState(42)
    items: list[int] = [1, 2, 3]
    rs.shuffle(items)  # type: ignore[arg-type]
    sg = random.SeedGenerator(seed=10)
    assert sg.next().shape == (2,)
    assert random._resolve_shape(None) == (1,)
    assert random._resolve_shape(4) == (4,)

    # constraints
    assert isinstance(constraints.max_norm(), constraints.MaxNorm)
    assert isinstance(constraints.min_max_norm(), constraints.MinMaxNorm)
    assert isinstance(constraints.non_neg(), constraints.NonNeg)
    assert isinstance(constraints.unit_norm(), constraints.UnitNorm)
    c = constraints.Constraint.from_config({})
    assert c(10) == 10
    assert constraints.serialize(c) == {"class_name": "Constraint", "config": {}}
    assert constraints.deserialize(c) is c
    assert constraints.get(c) is c

    # schedules
    base_sched = schedules.LearningRateSchedule.from_config({})
    assert base_sched.get_config() == {}
    assert schedules.deserialize(base_sched) is base_sched
    assert schedules.serialize(base_sched) is base_sched

    # distribution
    lm = distribution.LayoutMap()
    layout = distribution.TensorLayout(["data"])
    assert lm.setdefault("w", layout) is layout
    assert lm.setdefault("w", None) is layout
    assert lm.setdefault("missing", None) is None
    assert lm.popitem() == ("w", layout)

    prev_backend = config.backend()
    config.set_backend("jax")
    assert isinstance(distribution.distribution(), distribution.DataParallel)
    config.set_backend(prev_backend)

    # dtype_policies
    pm = dtype_policies.DTypePolicyMap(default_policy=None)
    assert pm.get("nonexistent", "fallback_val") == "fallback_val"

    # activations
    assert activations.deserialize(123) is None
    try:
        activations._to_tensor(None)
    except ValueError:
        pass

    class HasUnderlyingTensor:
        """Wrapper with _tensor attribute."""

        def __init__(self) -> None:
            """Initialize wrapper."""
            self._tensor = ops.zeros((1, 2))

    assert activations._to_tensor(HasUnderlyingTensor()) is not None  # type: ignore[arg-type]

    # initializers
    class SymbolicMock:
        """Mock symbolic tensor with id on data."""

        shape = (2, 2)
        dtype = "float32"

        class data:
            """Inner node with id."""

            id = "node_1"

    assert isinstance(initializers._wrap(SymbolicMock()), KerasTensor)  # type: ignore[arg-type]
    assert initializers._wrap(123) == 123
    assert initializers._normalize_shape(3) == (3,)
    z = initializers.Zeros()
    assert isinstance(z.clone(), initializers.Zeros)
    assert z.get_config() == {}
    assert isinstance(initializers.get({"class_name": "Zeros", "config": {}}), initializers.Zeros)
    assert initializers.deserialize("nonexistent_init") == "nonexistent_init"
    assert isinstance(
        initializers.deserialize(
            {"class_name": "Custom", "config": {}}, custom_objects={"Custom": initializers.Zeros}
        ),
        initializers.Zeros,
    )
    assert initializers.deserialize(
        {"class_name": "Custom", "config": {}}, custom_objects={"Custom": "not_callable"}
    ) == {
        "class_name": "Custom",
        "config": {},
    }
    assert initializers.deserialize({"class_name": "Custom", "config": {}}, custom_objects={}) == {
        "class_name": "Custom",
        "config": {},
    }
    assert initializers._get_dtype(DType.Float32) == DType.Float32

    # dataset_utils extra branches
    assert dataset_utils.array_to_img(ops.zeros((3, 4, 4)), data_format="channels_first").size == (
        4,
        4,
    )
    try:
        dataset_utils.array_to_img(ops.zeros((4,)))
    except ValueError:
        pass
    try:
        dataset_utils.img_to_array(ops.zeros((4,)))
    except ValueError:
        pass
    rgb_path = Path("tmp_rgb_test.zk")
    rgb_path.write_bytes(struct.pack(">4sHHH", b"ZKIM", 8, 8, 3))
    assert dataset_utils.load_img(rgb_path, color_mode="rgb").mode == "RGB"
    rgb_path.unlink(missing_ok=True)
    jpeg_short_end = b"\xff\xd8\xff\xe0\x00\x05\x00\x00\x00\xff\xe0\x00"
    assert dataset_utils._parse_image_dimensions(jpeg_short_end) == (256, 256, 3)
