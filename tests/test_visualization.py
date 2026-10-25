"""Comprehensive tests for ``zero_keras.visualization``."""

from __future__ import annotations

from zero_keras import ops, visualization

from .conftest import assert_allclose_zero


def test_visualization() -> None:
    """Verify bounding box drawing, segmentation mask blending, and gallery tiling."""
    assert visualization.draw_bounding_boxes(1, 1, 1) == 1
    assert visualization.draw_segmentation_masks(1, 1, 1) == 1
    assert visualization.plot_bounding_box_gallery(1, 1, 1) == 1
    assert visualization.plot_image_gallery(1) == 1
    assert visualization.plot_segmentation_mask_gallery(1, 1, 1) == 1

    vec = ops.zeros((4,), dtype="float32")
    assert tuple(visualization.draw_bounding_boxes(vec, [[0, 0, 1, 1]]).shape) == (4,)
    img_3d = ops.zeros((8, 8, 3), dtype="float32")
    assert tuple(visualization.draw_bounding_boxes(img_3d, 1).shape) == (8, 8, 3)

    boxes = [[[1.0, 1.0, 5.0, 5.0], [6.0, 2.0, 2.0, 6.0]]]
    drawn_3d = ops.convert_to_numpy(
        visualization.draw_bounding_boxes(img_3d, {"boxes": boxes[0]}, color=(200, 100, 50))
    )
    assert drawn_3d.shape == (8, 8, 3)
    assert_allclose_zero(drawn_3d[1, 1], [200.0, 100.0, 50.0])
    assert_allclose_zero(drawn_3d[5, 5], [200.0, 100.0, 50.0])
    assert_allclose_zero(drawn_3d[3, 3], [0.0, 0.0, 0.0])

    imgs_4d = ops.zeros((2, 6, 6, 3), dtype="float32")
    drawn_4d = ops.convert_to_numpy(visualization.draw_bounding_boxes(imgs_4d, boxes, color=128))
    assert drawn_4d.shape == (2, 6, 6, 3)
    assert_allclose_zero(drawn_4d[0, 1, 1], [128.0, 128.0, 128.0])

    masks = ops.ones((2, 6, 6), dtype="float32")
    masked = ops.convert_to_numpy(
        visualization.draw_segmentation_masks(imgs_4d, masks, color=None, alpha=0.5)
    )
    assert masked.shape == (2, 6, 6, 3)
    assert_allclose_zero(masked, 127.5)

    masked_4d = ops.convert_to_numpy(
        visualization.draw_segmentation_masks(
            imgs_4d, ops.ones((2, 6, 6, 1), dtype="float32"), color=(100, 100, 100), alpha=0.5
        )
    )
    assert_allclose_zero(masked_4d, 50.0)

    assert tuple(visualization.plot_image_gallery(img_3d).shape) == (8, 8, 3)
    imgs_3 = ops.ones((3, 4, 5, 3), dtype="float32")
    gal = ops.convert_to_numpy(visualization.plot_image_gallery(imgs_3, rows=2, cols=2))
    assert gal.shape == (8, 10, 3)

    bbox_gal = ops.convert_to_numpy(
        visualization.plot_bounding_box_gallery(
            imgs_4d,
            y_true=boxes,
            y_pred=boxes,
            rows=1,
            cols=2,
        )
    )
    assert bbox_gal.shape == (6, 12, 3)

    seg_gal = ops.convert_to_numpy(
        visualization.plot_segmentation_mask_gallery(
            imgs_4d,
            y_true=masks,
            y_pred=masks,
            rows=1,
            cols=2,
        )
    )
    assert seg_gal.shape == (6, 12, 3)
