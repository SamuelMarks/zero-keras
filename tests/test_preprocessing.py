from zero_keras import preprocessing


def test_preprocessing():
    """Test preprocessing module."""
    # The functions are just imported, checking their existence is enough to cover the __init__.py files
    assert hasattr(preprocessing, "image")
    assert hasattr(preprocessing, "sequence")
    assert hasattr(preprocessing.image, "array_to_img")
    assert hasattr(preprocessing.image, "img_to_array")
    assert hasattr(preprocessing.image, "load_img")
    assert hasattr(preprocessing.image, "save_img")
    assert hasattr(preprocessing.image, "smart_resize")

    assert hasattr(preprocessing.sequence, "pad_sequences")
    assert hasattr(preprocessing, "image_dataset_from_directory")
    assert hasattr(preprocessing, "text_dataset_from_directory")
    assert hasattr(preprocessing, "timeseries_dataset_from_array")
