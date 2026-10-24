"""Bounding box geometry and format conversion utilities for zero_keras."""

from __future__ import annotations

from collections.abc import Sequence
import math

from zero_keras import ops
from zero_keras.activations import _to_tensor, _wrap
from zero_keras.types import ShapeLike, TensorLike

_VALID_FORMATS: frozenset[str] = frozenset(
    {
        "xyxy",
        "yxyx",
        "xywh",
        "center_xywh",
        "rel_xyxy",
        "rel_yxyx",
        "rel_xywh",
        "rel_center_xywh",
    }
)


def _extract_hw(
    image_size: ShapeLike | None = None,
    height: float | int | None = None,
    width: float | int | None = None,
) -> tuple[float, float]:
    """Resolve ``(height, width)`` floats from ``image_size`` or explicit arguments.

    Args:
        image_size: Optional ``(height, width)`` or ``(height, width, channels)`` shape.
        height: Optional explicit image height.
        width: Optional explicit image width.

    Returns:
        tuple[float, float]: Resolved ``(height, width)`` pair.
    """
    if height is not None and width is not None:
        return float(height), float(width)
    if isinstance(image_size, int):
        return float(image_size), float(image_size)
    if isinstance(image_size, Sequence) and len(image_size) >= 2:
        h_val = image_size[0]
        w_val = image_size[1]
        return (
            float(h_val) if h_val is not None else 1.0,
            float(w_val) if w_val is not None else 1.0,
        )
    return (
        float(height) if height is not None else 1.0,
        float(width) if width is not None else 1.0,
    )


def _split_coords(boxes: TensorLike) -> tuple[TensorLike, TensorLike, TensorLike, TensorLike]:
    """Split a bounding box tensor of shape ``(..., 4)`` into 4 coordinate tensors.

    Args:
        boxes: Bounding box tensor with last dimension 4.

    Returns:
        tuple[TensorLike, TensorLike, TensorLike, TensorLike]: Coordinate slices
            ``(c0, c1, c2, c3)`` with shape ``(..., 1)``.
    """
    t = _to_tensor(boxes)
    parts = ops.split(t, 4, axis=-1)
    return parts[0], parts[1], parts[2], parts[3]


def _to_xyxy(
    boxes: TensorLike,
    fmt: str,
    h: float = 1.0,
    w: float = 1.0,
) -> tuple[TensorLike, TensorLike, TensorLike, TensorLike]:
    """Convert boxes from ``fmt`` into absolute ``(x1, y1, x2, y2)`` components.

    Args:
        boxes: Input bounding boxes of shape ``(..., 4)``.
        fmt: Source bounding box format string.
        h: Image height for relative formats.
        w: Image width for relative formats.

    Returns:
        tuple[TensorLike, TensorLike, TensorLike, TensorLike]: ``(x1, y1, x2, y2)``.

    Raises:
        ValueError: If ``fmt`` is not a recognized bounding box format.
    """
    fmt_lower = fmt.lower()
    if fmt_lower not in _VALID_FORMATS:
        raise ValueError(f"Unsupported bounding box format: {fmt!r}")
    c0, c1, c2, c3 = _split_coords(boxes)
    is_rel = fmt_lower.startswith("rel_")
    base_fmt = fmt_lower[4:] if is_rel else fmt_lower

    if base_fmt == "xyxy":
        x1, y1, x2, y2 = c0, c1, c2, c3
    elif base_fmt == "yxyx":
        y1, x1, y2, x2 = c0, c1, c2, c3
    elif base_fmt == "xywh":
        x1 = c0
        y1 = c1
        x2 = ops.add(c0, c2)
        y2 = ops.add(c1, c3)
    else:  # center_xywh
        half_w = ops.multiply(c2, 0.5)
        half_h = ops.multiply(c3, 0.5)
        x1 = ops.subtract(c0, half_w)
        y1 = ops.subtract(c1, half_h)
        x2 = ops.add(c0, half_w)
        y2 = ops.add(c1, half_h)

    if is_rel:
        x1 = ops.multiply(x1, w)
        x2 = ops.multiply(x2, w)
        y1 = ops.multiply(y1, h)
        y2 = ops.multiply(y2, h)

    return x1, y1, x2, y2


def _from_xyxy(
    x1: TensorLike,
    y1: TensorLike,
    x2: TensorLike,
    y2: TensorLike,
    fmt: str,
    h: float = 1.0,
    w: float = 1.0,
) -> TensorLike:
    """Convert ``(x1, y1, x2, y2)`` coordinate tensors into target format ``fmt``.

    Args:
        x1: Left x-coordinate tensor ``(..., 1)``.
        y1: Top y-coordinate tensor ``(..., 1)``.
        x2: Right x-coordinate tensor ``(..., 1)``.
        y2: Bottom y-coordinate tensor ``(..., 1)``.
        fmt: Target bounding box format string.
        h: Image height for relative formats.
        w: Image width for relative formats.

    Returns:
        TensorLike: Combined bounding box tensor of shape ``(..., 4)``.

    Raises:
        ValueError: If ``fmt`` is not a recognized bounding box format.
    """
    fmt_lower = fmt.lower()
    if fmt_lower not in _VALID_FORMATS:
        raise ValueError(f"Unsupported bounding box format: {fmt!r}")
    is_rel = fmt_lower.startswith("rel_")
    base_fmt = fmt_lower[4:] if is_rel else fmt_lower

    if is_rel:
        x1 = ops.divide(x1, w)
        x2 = ops.divide(x2, w)
        y1 = ops.divide(y1, h)
        y2 = ops.divide(y2, h)

    if base_fmt == "xyxy":
        out = ops.concatenate([x1, y1, x2, y2], axis=-1)
    elif base_fmt == "yxyx":
        out = ops.concatenate([y1, x1, y2, x2], axis=-1)
    elif base_fmt == "xywh":
        bw = ops.subtract(x2, x1)
        bh = ops.subtract(y2, y1)
        out = ops.concatenate([x1, y1, bw, bh], axis=-1)
    else:  # center_xywh
        bw = ops.subtract(x2, x1)
        bh = ops.subtract(y2, y1)
        cx = ops.add(x1, ops.multiply(bw, 0.5))
        cy = ops.add(y1, ops.multiply(bh, 0.5))
        out = ops.concatenate([cx, cy, bw, bh], axis=-1)

    return _wrap(out)


def convert_format(
    boxes: TensorLike,
    source: str,
    target: str,
    height: float | int | None = None,
    width: float | int | None = None,
    dtype: str = "float32",
) -> TensorLike:
    """Convert bounding boxes from ``source`` format to ``target`` format.

    Args:
        boxes: Input bounding boxes of shape ``(..., 4)``.
        source: Source box format string (e.g., ``"xyxy"``, ``"xywh"``).
        target: Target box format string (e.g., ``"xywh"``, ``"center_xywh"``).
        height: Optional image height for relative coordinate conversions.
        width: Optional image width for relative coordinate conversions.
        dtype: Output tensor dtype.

    Returns:
        TensorLike: Converted bounding boxes.
    """
    if isinstance(boxes, (int, float)):
        if source.lower() not in _VALID_FORMATS or target.lower() not in _VALID_FORMATS:
            raise ValueError(f"Unsupported format conversion: {source!r} -> {target!r}")
        return boxes
    h, w = _extract_hw(height=height, width=width)
    t = ops.cast(_to_tensor(boxes), dtype)
    x1, y1, x2, y2 = _to_xyxy(t, source, h=h, w=w)
    return _from_xyxy(x1, y1, x2, y2, target, h=h, w=w)


def clip_to_image_size(
    boxes: TensorLike,
    image_size: ShapeLike | None = None,
    height: int | float | None = None,
    width: int | float | None = None,
    bounding_box_format: str = "xyxy",
) -> TensorLike:
    """Clip bounding boxes to fit within ``image_size`` or ``(height, width)``.

    Args:
        boxes: Bounding boxes tensor of shape ``(..., 4)``.
        image_size: Optional image ``(height, width)`` bounds.
        height: Optional explicit image height.
        width: Optional explicit image width.
        bounding_box_format: Format of ``boxes``.

    Returns:
        TensorLike: Clipped bounding boxes in ``bounding_box_format``.
    """
    if isinstance(boxes, (int, float)):
        return boxes
    h, w = _extract_hw(image_size=image_size, height=height, width=width)
    t = ops.cast(_to_tensor(boxes), "float32")
    x1, y1, x2, y2 = _to_xyxy(t, bounding_box_format, h=h, w=w)
    x1_c = ops.clip(x1, 0.0, w)
    x2_c = ops.clip(x2, 0.0, w)
    y1_c = ops.clip(y1, 0.0, h)
    y2_c = ops.clip(y2, 0.0, h)
    return _from_xyxy(x1_c, y1_c, x2_c, y2_c, bounding_box_format, h=h, w=w)


def compute_iou(
    boxes1: TensorLike,
    boxes2: TensorLike,
    bounding_box_format: str = "xyxy",
    use_masking: bool = False,
    mask_val: float = -1.0,
) -> TensorLike:
    """Compute pairwise Intersection over Union (IoU) between ``boxes1`` and ``boxes2``.

    Args:
        boxes1: First set of bounding boxes of shape ``(N, 4)`` or ``(..., N, 4)``.
        boxes2: Second set of bounding boxes of shape ``(M, 4)`` or ``(..., M, 4)``.
        bounding_box_format: Format of input boxes.
        use_masking: Whether to mask invalid boxes.
        mask_val: Sentinel value used when ``use_masking=True``.

    Returns:
        TensorLike: Pairwise IoU matrix of shape ``(..., N, M)``.
    """
    _ = use_masking, mask_val
    if isinstance(boxes1, (int, float)):
        return boxes1
    t1 = ops.cast(_to_tensor(boxes1), "float32")
    t2 = ops.cast(_to_tensor(boxes2), "float32")
    x1_1, y1_1, x2_1, y2_1 = _to_xyxy(t1, bounding_box_format)
    x1_2, y1_2, x2_2, y2_2 = _to_xyxy(t2, bounding_box_format)

    # Expand dims for pairwise broadcast: (..., N, 1) vs (..., 1, M)
    b1_x1 = ops.expand_dims(ops.squeeze(x1_1, axis=-1), axis=-1)
    b1_y1 = ops.expand_dims(ops.squeeze(y1_1, axis=-1), axis=-1)
    b1_x2 = ops.expand_dims(ops.squeeze(x2_1, axis=-1), axis=-1)
    b1_y2 = ops.expand_dims(ops.squeeze(y2_1, axis=-1), axis=-1)

    b2_x1 = ops.expand_dims(ops.squeeze(x1_2, axis=-1), axis=-2)
    b2_y1 = ops.expand_dims(ops.squeeze(y1_2, axis=-1), axis=-2)
    b2_x2 = ops.expand_dims(ops.squeeze(x2_2, axis=-1), axis=-2)
    b2_y2 = ops.expand_dims(ops.squeeze(y2_2, axis=-1), axis=-2)

    inter_x1 = ops.maximum(b1_x1, b2_x1)
    inter_y1 = ops.maximum(b1_y1, b2_y1)
    inter_x2 = ops.minimum(b1_x2, b2_x2)
    inter_y2 = ops.minimum(b1_y2, b2_y2)

    inter_w = ops.maximum(ops.subtract(inter_x2, inter_x1), 0.0)
    inter_h = ops.maximum(ops.subtract(inter_y2, inter_y1), 0.0)
    inter_area = ops.multiply(inter_w, inter_h)

    area1 = ops.multiply(
        ops.maximum(ops.subtract(b1_x2, b1_x1), 0.0),
        ops.maximum(ops.subtract(b1_y2, b1_y1), 0.0),
    )
    area2 = ops.multiply(
        ops.maximum(ops.subtract(b2_x2, b2_x1), 0.0),
        ops.maximum(ops.subtract(b2_y2, b2_y1), 0.0),
    )
    union_area = ops.subtract(ops.add(area1, area2), inter_area)
    iou = ops.divide(inter_area, ops.add(union_area, 1e-7))
    return _wrap(iou)


def compute_ciou(
    boxes1: TensorLike,
    boxes2: TensorLike,
    bounding_box_format: str = "xyxy",
    image_shape: ShapeLike | None = None,
) -> TensorLike:
    """Compute Complete Intersection over Union (CIoU) between two box sets.

    Args:
        boxes1: First set of bounding boxes of shape ``(N, 4)``.
        boxes2: Second set of bounding boxes of shape ``(M, 4)``.
        bounding_box_format: Format of input boxes.
        image_shape: Optional image shape for relative formats.

    Returns:
        TensorLike: Pairwise CIoU tensor of shape ``(..., N, M)``.
    """
    if isinstance(boxes1, (int, float)):
        return boxes1
    h, w = _extract_hw(image_size=image_shape)
    t1 = ops.cast(_to_tensor(boxes1), "float32")
    t2 = ops.cast(_to_tensor(boxes2), "float32")
    x1_1, y1_1, x2_1, y2_1 = _to_xyxy(t1, bounding_box_format, h=h, w=w)
    x1_2, y1_2, x2_2, y2_2 = _to_xyxy(t2, bounding_box_format, h=h, w=w)

    b1_x1 = ops.expand_dims(ops.squeeze(x1_1, axis=-1), axis=-1)
    b1_y1 = ops.expand_dims(ops.squeeze(y1_1, axis=-1), axis=-1)
    b1_x2 = ops.expand_dims(ops.squeeze(x2_1, axis=-1), axis=-1)
    b1_y2 = ops.expand_dims(ops.squeeze(y2_1, axis=-1), axis=-1)

    b2_x1 = ops.expand_dims(ops.squeeze(x1_2, axis=-1), axis=-2)
    b2_y1 = ops.expand_dims(ops.squeeze(y1_2, axis=-1), axis=-2)
    b2_x2 = ops.expand_dims(ops.squeeze(x2_2, axis=-1), axis=-2)
    b2_y2 = ops.expand_dims(ops.squeeze(y2_2, axis=-1), axis=-2)

    w1 = ops.maximum(ops.subtract(b1_x2, b1_x1), 1e-7)
    h1 = ops.maximum(ops.subtract(b1_y2, b1_y1), 1e-7)
    w2 = ops.maximum(ops.subtract(b2_x2, b2_x1), 1e-7)
    h2 = ops.maximum(ops.subtract(b2_y2, b2_y1), 1e-7)

    inter_w = ops.maximum(
        ops.subtract(ops.minimum(b1_x2, b2_x2), ops.maximum(b1_x1, b2_x1)),
        0.0,
    )
    inter_h = ops.maximum(
        ops.subtract(ops.minimum(b1_y2, b2_y2), ops.maximum(b1_y1, b2_y1)),
        0.0,
    )
    inter_area = ops.multiply(inter_w, inter_h)
    union_area = ops.add(
        ops.subtract(ops.add(ops.multiply(w1, h1), ops.multiply(w2, h2)), inter_area),
        1e-7,
    )
    iou = ops.divide(inter_area, union_area)

    cx1 = ops.multiply(ops.add(b1_x1, b1_x2), 0.5)
    cy1 = ops.multiply(ops.add(b1_y1, b1_y2), 0.5)
    cx2 = ops.multiply(ops.add(b2_x1, b2_x2), 0.5)
    cy2 = ops.multiply(ops.add(b2_y1, b2_y2), 0.5)
    rho2 = ops.add(ops.square(ops.subtract(cx1, cx2)), ops.square(ops.subtract(cy1, cy2)))

    enc_x1 = ops.minimum(b1_x1, b2_x1)
    enc_y1 = ops.minimum(b1_y1, b2_y1)
    enc_x2 = ops.maximum(b1_x2, b2_x2)
    enc_y2 = ops.maximum(b1_y2, b2_y2)
    c2 = ops.add(
        ops.add(ops.square(ops.subtract(enc_x2, enc_x1)), ops.square(ops.subtract(enc_y2, enc_y1))),
        1e-7,
    )

    atan_diff = ops.subtract(ops.arctan(ops.divide(w2, h2)), ops.arctan(ops.divide(w1, h1)))
    v = ops.multiply(4.0 / (math.pi**2), ops.square(atan_diff))
    alpha = ops.divide(v, ops.add(ops.subtract(1.0, iou), ops.add(v, 1e-7)))
    ciou = ops.subtract(ops.subtract(iou, ops.divide(rho2, c2)), ops.multiply(alpha, v))
    return _wrap(ciou)


def affine_transform(
    boxes: TensorLike,
    transform: TensorLike | None = None,
    angle: float = 0.0,
    translate_x: float = 0.0,
    translate_y: float = 0.0,
    scale: float = 1.0,
    shear_x: float = 0.0,
    shear_y: float = 0.0,
    height: int | float | None = None,
    width: int | float | None = None,
    bounding_box_format: str = "xyxy",
) -> TensorLike:
    """Apply a 2D affine transformation to bounding boxes.

    Args:
        boxes: Bounding boxes tensor of shape ``(..., 4)``.
        transform: Optional ``(2, 3)`` or ``(3, 3)`` affine matrix or 6-element vector.
        angle: Rotation angle in degrees if ``transform`` is None.
        translate_x: Horizontal translation if ``transform`` is None.
        translate_y: Vertical translation if ``transform`` is None.
        scale: Isotropic scale factor if ``transform`` is None.
        shear_x: Horizontal shear factor if ``transform`` is None.
        shear_y: Vertical shear factor if ``transform`` is None.
        height: Optional image height for relative formats.
        width: Optional image width for relative formats.
        bounding_box_format: Format of ``boxes``.

    Returns:
        TensorLike: Transformed axis-aligned bounding boxes.
    """
    if isinstance(boxes, (int, float)):
        return boxes
    h, w = _extract_hw(height=height, width=width)
    t = ops.cast(_to_tensor(boxes), "float32")
    x1, y1, x2, y2 = _to_xyxy(t, bounding_box_format, h=h, w=w)

    if transform is not None and not isinstance(transform, (int, float)):
        mat = ops.reshape(ops.cast(_to_tensor(transform), "float32"), (-1,))
        a0 = mat[0]
        a1 = mat[1]
        a2 = mat[2]
        b0 = mat[3]
        b1 = mat[4]
        b2 = mat[5]
    else:
        rad = math.radians(float(angle))
        cos_a = math.cos(rad) * float(scale)
        sin_a = math.sin(rad) * float(scale)
        a0 = cos_a
        a1 = -sin_a + float(shear_x)
        a2 = float(translate_x)
        b0 = sin_a + float(shear_y)
        b1 = cos_a
        b2 = float(translate_y)

    # Transform all 4 corners: (x1,y1), (x2,y1), (x1,y2), (x2,y2)
    cx = [x1, x2, x1, x2]
    cy = [y1, y1, y2, y2]
    tx = [
        ops.add(ops.add(ops.multiply(px, a0), ops.multiply(py, a1)), a2) for px, py in zip(cx, cy)
    ]
    ty = [
        ops.add(ops.add(ops.multiply(px, b0), ops.multiply(py, b1)), b2) for px, py in zip(cx, cy)
    ]

    nx1 = ops.minimum(ops.minimum(tx[0], tx[1]), ops.minimum(tx[2], tx[3]))
    ny1 = ops.minimum(ops.minimum(ty[0], ty[1]), ops.minimum(ty[2], ty[3]))
    nx2 = ops.maximum(ops.maximum(tx[0], tx[1]), ops.maximum(tx[2], tx[3]))
    ny2 = ops.maximum(ops.maximum(ty[0], ty[1]), ops.maximum(ty[2], ty[3]))

    return _from_xyxy(nx1, ny1, nx2, ny2, bounding_box_format, h=h, w=w)


def crop(
    boxes: TensorLike,
    top: int | float | ShapeLike = 0,
    left: int | float | TensorLike = 0,
    height: int | float | None = None,
    width: int | float | None = None,
    bounding_box_format: str = "xyxy",
) -> TensorLike:
    """Adjust bounding boxes for an image crop region.

    Args:
        boxes: Input bounding boxes of shape ``(..., 4)``.
        top: Top crop offset (or original ``image_size`` when called with 3 args).
        left: Left crop offset (or ``crop_rect`` ``(top, left, height, width)``).
        height: Optional crop height.
        width: Optional crop width.
        bounding_box_format: Format of ``boxes``.

    Returns:
        TensorLike: Cropped bounding boxes.
    """
    if isinstance(boxes, (int, float)):
        return boxes
    if isinstance(left, Sequence) and len(left) == 4:
        t_off = float(left[0]) if isinstance(left[0], (int, float)) else 0.0
        l_off = float(left[1]) if isinstance(left[1], (int, float)) else 0.0
        c_h = float(left[2]) if isinstance(left[2], (int, float)) else None
        c_w = float(left[3]) if isinstance(left[3], (int, float)) else None
    else:
        t_off = float(top) if isinstance(top, (int, float)) else 0.0
        l_off = float(left) if isinstance(left, (int, float)) else 0.0
        c_h = float(height) if height is not None else None
        c_w = float(width) if width is not None else None

    t = ops.cast(_to_tensor(boxes), "float32")
    x1, y1, x2, y2 = _to_xyxy(t, bounding_box_format)
    x1 = ops.subtract(x1, l_off)
    x2 = ops.subtract(x2, l_off)
    y1 = ops.subtract(y1, t_off)
    y2 = ops.subtract(y2, t_off)

    if c_w is not None:
        x1 = ops.clip(x1, 0.0, c_w)
        x2 = ops.clip(x2, 0.0, c_w)
    if c_h is not None:
        y1 = ops.clip(y1, 0.0, c_h)
        y2 = ops.clip(y2, 0.0, c_h)

    return _from_xyxy(x1, y1, x2, y2, bounding_box_format)


def pad(
    boxes: TensorLike,
    top: int | float | TensorLike = 0,
    left: int | float = 0,
    bounding_box_format: str = "xyxy",
) -> TensorLike:
    """Shift bounding boxes according to top and left image padding offsets.

    Args:
        boxes: Input bounding boxes of shape ``(..., 4)``.
        top: Top padding offset or ``(top, left, ...)`` sequence.
        left: Left padding offset.
        bounding_box_format: Format of ``boxes``.

    Returns:
        TensorLike: Padded bounding boxes.
    """
    if isinstance(boxes, (int, float)):
        return boxes
    if isinstance(top, Sequence) and len(top) >= 2:
        t_off = float(top[0]) if isinstance(top[0], (int, float)) else 0.0
        l_off = float(top[1]) if isinstance(top[1], (int, float)) else 0.0
    else:
        t_off = float(top) if isinstance(top, (int, float)) else 0.0
        l_off = float(left)

    t = ops.cast(_to_tensor(boxes), "float32")
    x1, y1, x2, y2 = _to_xyxy(t, bounding_box_format)
    x1 = ops.add(x1, l_off)
    x2 = ops.add(x2, l_off)
    y1 = ops.add(y1, t_off)
    y2 = ops.add(y2, t_off)
    return _from_xyxy(x1, y1, x2, y2, bounding_box_format)


def encode_box_to_deltas(
    anchors: TensorLike,
    boxes: TensorLike,
    anchor_format: str = "xyxy",
    box_format: str = "xyxy",
    variance: Sequence[float] | None = None,
) -> TensorLike:
    """Encode ground-truth ``boxes`` into regression deltas relative to ``anchors``.

    Args:
        anchors: Anchor boxes tensor of shape ``(..., 4)``.
        boxes: Target bounding boxes tensor of shape ``(..., 4)``.
        anchor_format: Format of ``anchors``.
        box_format: Format of ``boxes``.
        variance: Optional 4-element variance scaling sequence.

    Returns:
        TensorLike: Encoded ``(dx, dy, dw, dh)`` box regression deltas.

    Raises:
        ValueError: If ``variance`` does not have length 4.
    """
    if isinstance(boxes, (int, float)):
        return boxes
    if variance is not None and len(variance) != 4:
        raise ValueError(f"`variance` must have length 4, got {len(variance)}.")

    a_cxywh = convert_format(anchors, anchor_format, "center_xywh")
    b_cxywh = convert_format(boxes, box_format, "center_xywh")
    ax, ay, aw, ah = _split_coords(a_cxywh)
    bx, by, bw, bh = _split_coords(b_cxywh)

    aw_safe = ops.maximum(aw, 1e-7)
    ah_safe = ops.maximum(ah, 1e-7)
    bw_safe = ops.maximum(bw, 1e-7)
    bh_safe = ops.maximum(bh, 1e-7)

    dx = ops.divide(ops.subtract(bx, ax), aw_safe)
    dy = ops.divide(ops.subtract(by, ay), ah_safe)
    dw = ops.log(ops.divide(bw_safe, aw_safe))
    dh = ops.log(ops.divide(bh_safe, ah_safe))

    if variance is not None:
        dx = ops.divide(dx, float(variance[0]))
        dy = ops.divide(dy, float(variance[1]))
        dw = ops.divide(dw, float(variance[2]))
        dh = ops.divide(dh, float(variance[3]))

    return _wrap(ops.concatenate([dx, dy, dw, dh], axis=-1))


def decode_deltas_to_boxes(
    anchors: TensorLike,
    boxes_delta: TensorLike,
    anchor_format: str = "xyxy",
    box_format: str = "xyxy",
    variance: Sequence[float] | None = None,
) -> TensorLike:
    """Decode relative box regression ``boxes_delta`` relative to ``anchors``.

    Args:
        anchors: Anchor boxes tensor of shape ``(..., 4)``.
        boxes_delta: Encoded box deltas tensor of shape ``(..., 4)``.
        anchor_format: Format of ``anchors``.
        box_format: Desired output format of decoded boxes.
        variance: Optional 4-element variance scaling sequence.

    Returns:
        TensorLike: Decoded bounding boxes in ``box_format``.

    Raises:
        ValueError: If ``variance`` does not have length 4.
    """
    if isinstance(boxes_delta, (int, float)):
        return boxes_delta
    if variance is not None and len(variance) != 4:
        raise ValueError(f"`variance` must have length 4, got {len(variance)}.")

    a_cxywh = convert_format(anchors, anchor_format, "center_xywh")
    ax, ay, aw, ah = _split_coords(a_cxywh)
    dx, dy, dw, dh = _split_coords(ops.cast(_to_tensor(boxes_delta), "float32"))

    if variance is not None:
        dx = ops.multiply(dx, float(variance[0]))
        dy = ops.multiply(dy, float(variance[1]))
        dw = ops.multiply(dw, float(variance[2]))
        dh = ops.multiply(dh, float(variance[3]))

    bx = ops.add(ops.multiply(dx, aw), ax)
    by = ops.add(ops.multiply(dy, ah), ay)
    bw = ops.multiply(ops.exp(dw), aw)
    bh = ops.multiply(ops.exp(dh), ah)

    decoded_cxywh = ops.concatenate([bx, by, bw, bh], axis=-1)
    return convert_format(decoded_cxywh, "center_xywh", box_format)


__all__ = [
    "affine_transform",
    "clip_to_image_size",
    "compute_ciou",
    "compute_iou",
    "convert_format",
    "crop",
    "decode_deltas_to_boxes",
    "encode_box_to_deltas",
    "pad",
]
