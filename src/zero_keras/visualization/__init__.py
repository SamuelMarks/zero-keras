"""Visualization gallery and overlay functions for zero_keras."""

from __future__ import annotations

import math

from zero_keras import ops
from zero_keras.activations import _to_tensor, _wrap
from zero_keras.types import ConfigValue, TensorLike
from zero_keras.utils import bounding_boxes as bbox_utils


def _resolve_rgb(color: tuple[int, int, int] | int | None) -> tuple[float, float, float]:
    """Normalize an RGB color specification into a 3-tuple of floats.

    Args:
        color: Optional RGB tuple or grayscale integer.

    Returns:
        tuple[float, float, float]: Normalized ``(r, g, b)`` floats.
    """
    if isinstance(color, tuple) and len(color) == 3:
        return float(color[0]), float(color[1]), float(color[2])
    if isinstance(color, int):
        c = float(color)
        return c, c, c
    return 255.0, 0.0, 0.0


def draw_bounding_boxes(
    images: TensorLike,
    bounding_boxes: TensorLike | dict[str, TensorLike],
    color: tuple[int, int, int] | int = (255, 0, 0),
    bounding_box_format: str = "xyxy",
    name: str | None = None,
) -> TensorLike:
    """Draw bounding box outlines on ``images``.

    Args:
        images: Batch of input images of shape ``(H, W, C)`` or ``(B, H, W, C)``.
        bounding_boxes: Bounding box coordinates tensor of shape ``(..., N, 4)``
            or dictionary containing a ``"boxes"`` key.
        color: RGB color tuple or integer intensity for box outlines.
        bounding_box_format: Format of ``bounding_boxes`` coordinates.
        name: Optional operation name.

    Returns:
        TensorLike: Copy of ``images`` with drawn bounding box outlines.
    """
    _ = name
    if isinstance(images, (int, float)):
        return images
    img_t = ops.cast(_to_tensor(images), "float32")
    if len(img_t.shape) < 3:
        return _wrap(ops.copy(img_t))

    raw_boxes: TensorLike
    if isinstance(bounding_boxes, dict):
        raw_boxes = bounding_boxes.get("boxes", [])
    else:
        raw_boxes = bounding_boxes
    if isinstance(raw_boxes, (int, float)):
        return _wrap(ops.copy(img_t))

    is_3d = len(img_t.shape) == 3
    batch_t = ops.expand_dims(img_t, axis=0) if is_3d else ops.copy(img_t)
    b_size = int(batch_t.shape[0])
    h = int(batch_t.shape[1])
    w = int(batch_t.shape[2])
    c = int(batch_t.shape[3])

    boxes_xyxy = _to_tensor(
        bbox_utils.convert_format(
            raw_boxes,
            source=bounding_box_format,
            target="xyxy",
            height=h,
            width=w,
        )
    )
    if len(boxes_xyxy.shape) == 2:
        boxes_xyxy = ops.expand_dims(boxes_xyxy, axis=0)

    rgb = _resolve_rgb(color)
    color_pixel = ops.convert_to_tensor([rgb[i % 3] for i in range(c)], dtype="float32")
    color_pixel = ops.reshape(color_pixel, (1, 1, c))

    ys = ops.reshape(ops.arange(h, dtype="int32"), (h, 1, 1))
    xs = ops.reshape(ops.arange(w, dtype="int32"), (1, w, 1))

    n_boxes = int(boxes_xyxy.shape[1])
    box_batches = min(b_size, int(boxes_xyxy.shape[0]))
    out_batches = []
    for b_idx in range(b_size):
        cur_img = batch_t[b_idx]
        if b_idx < box_batches:
            for k in range(n_boxes):
                box_k = ops.convert_to_numpy(boxes_xyxy[b_idx, k])
                x1 = max(0, min(w - 1, int(round(float(box_k[0])))))
                y1 = max(0, min(h - 1, int(round(float(box_k[1])))))
                x2 = max(x1, min(w - 1, int(round(float(box_k[2])))))
                y2 = max(y1, min(h - 1, int(round(float(box_k[3])))))
                in_x = ops.logical_and(ops.greater_equal(xs, x1), ops.less_equal(xs, x2))
                in_y = ops.logical_and(ops.greater_equal(ys, y1), ops.less_equal(ys, y2))
                on_h_edge = ops.logical_and(
                    in_x,
                    ops.logical_or(ops.equal(ys, y1), ops.equal(ys, y2)),
                )
                on_v_edge = ops.logical_and(
                    in_y,
                    ops.logical_or(ops.equal(xs, x1), ops.equal(xs, x2)),
                )
                border_mask = ops.logical_or(on_h_edge, on_v_edge)
                cur_img = ops.where(border_mask, color_pixel, cur_img)
        out_batches.append(cur_img)

    out_t = ops.stack(out_batches, axis=0)
    if is_3d:
        out_t = ops.squeeze(out_t, axis=0)
    return _wrap(out_t)


def draw_segmentation_masks(
    images: TensorLike,
    masks: TensorLike,
    num_classes: int | None = None,
    color: tuple[int, int, int] | int | None = None,
    alpha: float = 0.5,
    name: str | None = None,
) -> TensorLike:
    """Alpha-blend segmentation ``masks`` onto ``images``.

    Args:
        images: Batch of input images.
        masks: Segmentation mask tensor.
        num_classes: Optional number of segmentation classes.
        color: Optional overlay color specification.
        alpha: Blending factor in ``[0.0, 1.0]``.
        name: Optional operation name.

    Returns:
        TensorLike: Images blended with segmentation masks.
    """
    _ = num_classes, name
    if isinstance(images, (int, float)):
        return images
    img_t = ops.cast(_to_tensor(images), "float32")
    mask_t = ops.cast(_to_tensor(masks), "float32")
    if len(mask_t.shape) < len(img_t.shape):
        mask_t = ops.expand_dims(mask_t, axis=-1)

    rgb = _resolve_rgb(color)
    overlay_val = rgb[0] if color is not None else 255.0
    active = ops.cast(ops.greater(mask_t, 0.0), "float32")
    blended = ops.add(
        ops.multiply(img_t, ops.subtract(1.0, ops.multiply(active, float(alpha)))),
        ops.multiply(ops.multiply(active, overlay_val), float(alpha)),
    )
    return _wrap(blended)


def _tile_gallery(
    images: TensorLike,
    rows: int | None = None,
    cols: int | None = None,
) -> TensorLike:
    """Tile a 4D batch of images ``(B, H, W, C)`` into a single gallery grid tensor.

    Args:
        images: Batch of images or scalar fallback.
        rows: Optional explicit row count.
        cols: Optional explicit column count.

    Returns:
        TensorLike: Tiled gallery image of shape ``(rows * H, cols * W, C)``.
    """
    if isinstance(images, (int, float)):
        return images
    t = ops.cast(_to_tensor(images), "float32")
    if len(t.shape) != 4:
        return _wrap(t)
    b, h, w, c = int(t.shape[0]), int(t.shape[1]), int(t.shape[2]), int(t.shape[3])
    grid_rows = rows if rows is not None else max(1, int(math.ceil(math.sqrt(b))))
    grid_cols = cols if cols is not None else max(1, int(math.ceil(b / grid_rows)))

    row_tensors = []
    for r in range(grid_rows):
        col_tensors = []
        for col in range(grid_cols):
            idx = r * grid_cols + col
            if idx < b:
                col_tensors.append(t[idx])
            else:
                col_tensors.append(ops.zeros((h, w, c), dtype="float32"))
        row_tensors.append(ops.concatenate(col_tensors, axis=1))
    return _wrap(ops.concatenate(row_tensors, axis=0))


def plot_image_gallery(
    images: TensorLike,
    value_range: tuple[float, float] = (0, 255),
    rows: int | None = None,
    cols: int | None = None,
    name: str | None = None,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Plot a gallery grid of ``images``.

    Args:
        images: Batch of images to display.
        value_range: Pixel intensity range ``(min, max)``.
        rows: Optional gallery row count.
        cols: Optional gallery column count.
        name: Optional operation name.
        **kwargs: Additional display options.

    Returns:
        TensorLike: Rendered gallery tensor.
    """
    _ = value_range, name, kwargs
    return _tile_gallery(images, rows=rows, cols=cols)


def plot_bounding_box_gallery(
    images: TensorLike,
    value_range: tuple[float, float] | int = (0, 255),
    bounding_box_format: str | int = "xyxy",
    y_true: TensorLike | dict[str, TensorLike] | None = None,
    y_pred: TensorLike | dict[str, TensorLike] | None = None,
    rows: int | None = None,
    cols: int | None = None,
    name: str | None = None,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Plot a gallery of images with bounding box annotations.

    Args:
        images: Batch of images to plot.
        value_range: Pixel intensity value range.
        bounding_box_format: Format of bounding box coordinates.
        y_true: Optional ground-truth bounding boxes.
        y_pred: Optional predicted bounding boxes.
        rows: Optional gallery row count.
        cols: Optional gallery column count.
        name: Optional operation name.
        **kwargs: Additional plotting options.

    Returns:
        TensorLike: Rendered gallery tensor.
    """
    _ = value_range, name, kwargs
    fmt = bounding_box_format if isinstance(bounding_box_format, str) else "xyxy"
    out = images
    if y_true is not None:
        out = draw_bounding_boxes(out, y_true, color=(0, 255, 0), bounding_box_format=fmt)
    if y_pred is not None:
        out = draw_bounding_boxes(out, y_pred, color=(255, 0, 0), bounding_box_format=fmt)
    return _tile_gallery(out, rows=rows, cols=cols)


def plot_segmentation_mask_gallery(
    images: TensorLike,
    value_range: tuple[float, float] | int = (0, 255),
    num_classes: int = 2,
    y_true: TensorLike | None = None,
    y_pred: TensorLike | None = None,
    rows: int | None = None,
    cols: int | None = None,
    name: str | None = None,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Plot a gallery of images with segmentation masks.

    Args:
        images: Batch of images.
        value_range: Pixel intensity range.
        num_classes: Number of segmentation classes.
        y_true: Optional ground-truth segmentation masks.
        y_pred: Optional predicted segmentation masks.
        rows: Optional gallery row count.
        cols: Optional gallery column count.
        name: Optional operation name.
        **kwargs: Additional plotting options.

    Returns:
        TensorLike: Rendered segmentation gallery.
    """
    _ = value_range, name, kwargs
    out = images
    if y_true is not None:
        out = draw_segmentation_masks(out, y_true, num_classes=num_classes)
    if y_pred is not None:
        out = draw_segmentation_masks(out, y_pred, num_classes=num_classes)
    return _tile_gallery(out, rows=rows, cols=cols)


__all__ = [
    "draw_bounding_boxes",
    "draw_segmentation_masks",
    "plot_bounding_box_gallery",
    "plot_image_gallery",
    "plot_segmentation_mask_gallery",
]
