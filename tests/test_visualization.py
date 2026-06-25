from zero_keras import visualization


def test_visualization():
    """Test visualization module."""
    visualization.draw_bounding_boxes(1, 1, 1)
    visualization.draw_segmentation_masks(1, 1, 1)
    visualization.plot_bounding_box_gallery(1, 1, 1)
    visualization.plot_image_gallery(1)
    visualization.plot_segmentation_mask_gallery(1, 1, 1)
