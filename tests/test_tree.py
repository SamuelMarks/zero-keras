from zero_keras import tree


def test_tree():
    """Test tree module."""
    assert tree.MAP_TO_NONE is not None
    tree.assert_same_paths(1, 1)
    tree.assert_same_structure(1, 1)
    assert tree.flatten(1) == [1]
    assert tree.flatten_with_path(1) == [((), 1)]
    assert tree.is_nested(1) is False
    assert tree.lists_to_tuples(1) == 1
    assert tree.map_shape_structure(lambda x: x, 1) == 1
    assert tree.map_structure(lambda x: x, 1) == 1
    assert tree.map_structure_up_to(1, lambda x: x, 1) == 1
    assert tree.pack_sequence_as(1, [1]) == 1
    tree.traverse(lambda x: x, 1)
