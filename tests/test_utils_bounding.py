from zero_keras import utils


def test_utils_bounding():
    """Test utils bounding boxes."""
    utils.bounding_boxes.affine_transform(1, 1)
    utils.bounding_boxes.clip_to_image_size(1, 1)
    utils.bounding_boxes.compute_ciou(1, 1)
    utils.bounding_boxes.compute_iou(1, 1)
    utils.bounding_boxes.convert_format(1, 1, 1)
    utils.bounding_boxes.crop(1, 1, 1)
    utils.bounding_boxes.decode_deltas_to_boxes(1, 1)
    utils.bounding_boxes.encode_box_to_deltas(1, 1)
    utils.bounding_boxes.pad(1, 1)

    assert utils.legacy.deserialize_keras_object(1) == 1

    class Dummy:
        pass

    assert utils.legacy.serialize_keras_object(Dummy()) == {"class_name": "Dummy"}
