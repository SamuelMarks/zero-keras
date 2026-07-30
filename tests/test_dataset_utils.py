import os
from unittest import mock

import pytest

from zero_keras import ops
from zero_keras.utils import dataset_utils


def test_array_to_img():
    # RGB
    arr = ops.zeros((32, 32, 3))
    img = dataset_utils.array_to_img(arr, scale=True)
    assert isinstance(img, dataset_utils.GenericImage)
    assert img.mode == "RGB"

    # Grayscale
    arr_l = ops.zeros((32, 32, 1))
    img_l = dataset_utils.array_to_img(arr_l, scale=True)
    assert img_l.mode == "L"

    # RGBA
    arr_rgba = ops.zeros((32, 32, 4))
    img_rgba = dataset_utils.array_to_img(arr_rgba, scale=True)
    assert img_rgba.mode == "RGBA"

    # Not scaled, const array
    arr_c = ops.zeros((32, 32, 3))
    img_c = dataset_utils.array_to_img(arr_c, scale=False)
    assert img_c.mode == "RGB"

    # 2D case reshaped
    arr_2d = ops.zeros((32, 32))
    img_2d = dataset_utils.array_to_img(arr_2d, scale=True)
    assert img_2d.mode == "L"

    with pytest.raises(ValueError):
        dataset_utils.array_to_img(ops.zeros((32,)))


def test_img_to_array():
    arr = ops.zeros((32, 32, 3))
    img = dataset_utils.GenericImage(arr, mode="RGB")
    res = dataset_utils.img_to_array(img)
    assert res.shape == (32, 32, 3)

    # Test passing tensor directly to img_to_array
    res2 = dataset_utils.img_to_array(arr)
    assert res2.shape == (32, 32, 3)

    # Test 2D expanding
    arr_2d = ops.zeros((32, 32))
    res3 = dataset_utils.img_to_array(arr_2d)
    assert res3.shape == (32, 32, 1)


def test_load_img(tmpdir):
    path = str(tmpdir.join("test_img.jpg"))

    from PIL import Image

    Image.new("RGB", (256, 256)).save(path)

    # test load
    loaded = dataset_utils.load_img(path)
    assert loaded.size == (256, 256)
    assert loaded.mode == "RGB"

    # test grayscale
    loaded_l = dataset_utils.load_img(path, grayscale=True)
    assert loaded_l.mode == "L"

    # test color_mode
    loaded_rgba = dataset_utils.load_img(path, color_mode="rgba")
    assert loaded_rgba.mode == "RGBA"

    # test target size
    loaded_rs = dataset_utils.load_img(path, target_size=(32, 32))
    assert loaded_rs.size == (32, 32)


def test_generic_image(tmpdir):
    arr = ops.zeros((32, 32, 3))
    img = dataset_utils.GenericImage(arr, mode="RGB")
    img2 = img.convert("L")
    assert img2.mode == "L"

    # MOCK ops.image.resize to avoid ml_switcheroo_compiler bug
    with mock.patch("zero_keras.ops.image.resize", return_value=ops.zeros((16, 16, 3))):
        img3 = img.resize((16, 16))
        assert img3.size == (16, 16)

    path = str(tmpdir.join("test_save.jpg"))
    img.save(path)
    assert os.path.exists(path)

    img_empty = dataset_utils.GenericImage(ops.zeros((32,)))
    assert img_empty.size == (0, 0)


def test_save_img(tmpdir):
    path = str(tmpdir.join("test_save2.jpg"))
    arr = ops.zeros((32, 32, 3))
    dataset_utils.save_img(path, arr, file_format="jpeg")
    assert os.path.exists(path)


def test_smart_resize():
    # MOCK ops.slice to avoid ml_switcheroo_compiler bug and ops.image.resize
    with (
        mock.patch("zero_keras.ops.slice", return_value=ops.zeros((50, 50, 3))),
        mock.patch("zero_keras.ops.image.resize", return_value=ops.zeros((50, 50, 3))),
    ):
        arr = ops.zeros((100, 200, 3))
        res = dataset_utils.smart_resize(arr, (50, 50))
        assert res.shape == (50, 50, 3)

        arr_tall = ops.zeros((200, 100, 3))
        res_tall = dataset_utils.smart_resize(arr_tall, (50, 50))
        assert res_tall.shape == (50, 50, 3)

    with (
        mock.patch("zero_keras.ops.slice", return_value=ops.zeros((50, 50, 1))),
        mock.patch("zero_keras.ops.image.resize", return_value=ops.zeros((50, 50, 1))),
    ):
        # 2D case
        arr_2d = ops.zeros((100, 200))
        res_2d = dataset_utils.smart_resize(arr_2d, (50, 50))
        assert res_2d.shape == (50, 50)

    with pytest.raises(ValueError):
        dataset_utils.smart_resize(ops.zeros((100,)), (50, 50))
