"""utils.bounding_boxes API."""


def affine_transform(boxes, transform):
    """affine_transform docstring."""
    return boxes


def clip_to_image_size(boxes, image_size):
    """clip_to_image_size docstring."""
    return boxes


def compute_ciou(boxes1, boxes2):
    """compute_ciou docstring."""
    return boxes1


def compute_iou(boxes1, boxes2):
    """compute_iou docstring."""
    return boxes1


def convert_format(boxes, source, target):
    """convert_format docstring."""
    return boxes


def crop(boxes, image_size, crop_rect):
    """crop docstring."""
    return boxes


def decode_deltas_to_boxes(anchors, deltas):
    """decode_deltas_to_boxes docstring."""
    return deltas


def encode_box_to_deltas(anchors, boxes):
    """encode_box_to_deltas docstring."""
    return boxes


def pad(boxes, pad_rect):
    """pad docstring."""
    return boxes


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
