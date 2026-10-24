"""Comprehensive tests for ``zero_keras.utils.bounding_boxes`` and ``zero_keras.utils.legacy``."""

from __future__ import annotations

import numpy as np
import pytest

from zero_keras import ops, utils


def test_utils_bounding() -> None:
    """Verify bounding box format conversions, IoU/CIoU, transforms, and deltas."""
    bboxes = utils.bounding_boxes

    # Scalar fallback and format validation
    assert bboxes.affine_transform(1, 1) == 1
    assert bboxes.clip_to_image_size(1, 1) == 1
    assert bboxes.compute_ciou(1, 1) == 1
    assert bboxes.compute_iou(1, 1) == 1
    assert bboxes.convert_format(1, "xyxy", "xywh") == 1
    assert bboxes.crop(1, 1, 1) == 1
    assert bboxes.decode_deltas_to_boxes(1, 1) == 1
    assert bboxes.encode_box_to_deltas(1, 1) == 1
    assert bboxes.pad(1, 1) == 1

    with pytest.raises(ValueError, match="Unsupported format conversion"):
        bboxes.convert_format(1, "invalid_fmt", "xyxy")
    with pytest.raises(ValueError, match="Unsupported bounding box format"):
        bboxes.convert_format([[0.0, 0.0, 10.0, 10.0]], "invalid_src", "xyxy")
    with pytest.raises(ValueError, match="Unsupported bounding box format"):
        bboxes.convert_format([[0.0, 0.0, 10.0, 10.0]], "xyxy", "invalid_tgt")

    # Convert between xyxy, yxyx, xywh, center_xywh, and rel_* formats
    box_xyxy = [[10.0, 20.0, 30.0, 60.0]]
    box_xywh = bboxes.convert_format(box_xyxy, "xyxy", "xywh")
    np.testing.assert_allclose(ops.convert_to_numpy(box_xywh), [[10.0, 20.0, 20.0, 40.0]])

    box_cxywh = bboxes.convert_format(box_xywh, "xywh", "center_xywh")
    np.testing.assert_allclose(ops.convert_to_numpy(box_cxywh), [[20.0, 40.0, 20.0, 40.0]])

    box_yxyx = bboxes.convert_format(box_cxywh, "center_xywh", "yxyx")
    np.testing.assert_allclose(ops.convert_to_numpy(box_yxyx), [[20.0, 10.0, 60.0, 30.0]])

    box_rel = bboxes.convert_format(box_yxyx, "yxyx", "rel_xyxy", height=100, width=100)
    np.testing.assert_allclose(ops.convert_to_numpy(box_rel), [[0.1, 0.2, 0.3, 0.6]])

    box_back = bboxes.convert_format(box_rel, "rel_xyxy", "xyxy", height=100, width=100)
    np.testing.assert_allclose(ops.convert_to_numpy(box_back), box_xyxy)

    # clip_to_image_size (tuple, int, explicit height/width)
    out_of_bounds = [[-5.0, -10.0, 120.0, 80.0]]
    clipped = bboxes.clip_to_image_size(out_of_bounds, image_size=(50, 100))
    np.testing.assert_allclose(ops.convert_to_numpy(clipped), [[0.0, 0.0, 100.0, 50.0]])
    clipped_int = bboxes.clip_to_image_size(out_of_bounds, image_size=60)
    np.testing.assert_allclose(ops.convert_to_numpy(clipped_int), [[0.0, 0.0, 60.0, 60.0]])
    clipped_none = bboxes.clip_to_image_size([[0.2, 0.3, 0.8, 0.9]])
    np.testing.assert_allclose(ops.convert_to_numpy(clipped_none), [[0.2, 0.3, 0.8, 0.9]])

    # compute_iou & compute_ciou
    b1 = [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]
    b2 = [[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0]]
    iou = ops.convert_to_numpy(bboxes.compute_iou(b1, b2))
    assert iou.shape == (2, 2)
    np.testing.assert_allclose(iou[0, 0], 1.0, rtol=1e-5)
    np.testing.assert_allclose(iou[0, 1], 50.0 / 150.0, rtol=1e-5)
    np.testing.assert_allclose(iou[1, 0], 0.0, atol=1e-6)

    ciou = ops.convert_to_numpy(bboxes.compute_ciou(b1, b2, image_shape=(100, 100)))
    assert ciou.shape == (2, 2)
    np.testing.assert_allclose(ciou[0, 0], 1.0, rtol=1e-5)
    assert float(ciou[0, 1]) < float(iou[0, 1])

    # affine_transform (matrix and parametric)
    trans_mat = [[1.0, 0.0, 5.0], [0.0, 1.0, 10.0]]
    transformed = bboxes.affine_transform([[10.0, 20.0, 30.0, 40.0]], transform=trans_mat)
    np.testing.assert_allclose(ops.convert_to_numpy(transformed), [[15.0, 30.0, 35.0, 50.0]])

    transformed_param = bboxes.affine_transform(
        [[10.0, 20.0, 30.0, 40.0]], translate_x=5.0, translate_y=-5.0
    )
    np.testing.assert_allclose(ops.convert_to_numpy(transformed_param), [[15.0, 15.0, 35.0, 35.0]])

    # crop & pad
    cropped = bboxes.crop([[10.0, 20.0, 50.0, 60.0]], top=5, left=10, height=30, width=25)
    np.testing.assert_allclose(ops.convert_to_numpy(cropped), [[0.0, 15.0, 25.0, 30.0]])
    cropped_rect = bboxes.crop([[10.0, 20.0, 50.0, 60.0]], (100, 100), [5, 10, 30, 25])
    np.testing.assert_allclose(ops.convert_to_numpy(cropped_rect), [[0.0, 15.0, 25.0, 30.0]])
    cropped_no_hw = bboxes.crop([[10.0, 20.0, 50.0, 60.0]], top=5, left=10)
    np.testing.assert_allclose(ops.convert_to_numpy(cropped_no_hw), [[0.0, 15.0, 40.0, 55.0]])

    padded = bboxes.pad([[10.0, 20.0, 30.0, 40.0]], top=5, left=15)
    np.testing.assert_allclose(ops.convert_to_numpy(padded), [[25.0, 25.0, 45.0, 45.0]])
    padded_seq = bboxes.pad([[10.0, 20.0, 30.0, 40.0]], [5, 15])
    np.testing.assert_allclose(ops.convert_to_numpy(padded_seq), [[25.0, 25.0, 45.0, 45.0]])

    # encode_box_to_deltas & decode_deltas_to_boxes round-trip
    anchors = [[10.0, 10.0, 30.0, 30.0]]
    target_boxes = [[12.0, 14.0, 52.0, 54.0]]
    deltas = bboxes.encode_box_to_deltas(anchors, target_boxes, variance=[0.1, 0.1, 0.2, 0.2])
    decoded = bboxes.decode_deltas_to_boxes(anchors, deltas, variance=[0.1, 0.1, 0.2, 0.2])
    np.testing.assert_allclose(ops.convert_to_numpy(decoded), target_boxes, rtol=1e-5, atol=1e-5)

    deltas_novar = bboxes.encode_box_to_deltas(anchors, target_boxes)
    decoded_novar = bboxes.decode_deltas_to_boxes(anchors, deltas_novar)
    np.testing.assert_allclose(
        ops.convert_to_numpy(decoded_novar), target_boxes, rtol=1e-5, atol=1e-5
    )

    with pytest.raises(ValueError, match="variance"):
        bboxes.encode_box_to_deltas(anchors, target_boxes, variance=[0.1, 0.2])
    with pytest.raises(ValueError, match="variance"):
        bboxes.decode_deltas_to_boxes(anchors, deltas, variance=[0.1, 0.2])

    # Legacy serialization shim
    assert utils.legacy.deserialize_keras_object(1) == 1

    class Dummy:
        """Dummy class for legacy serialization testing."""

    assert utils.legacy.serialize_keras_object(Dummy()) == {"class_name": "Dummy"}
