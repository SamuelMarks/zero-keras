"""Comprehensive tests for ``zero_keras.tree`` PyTree utilities."""

from __future__ import annotations

import pytest

from zero_keras import tree
from zero_keras.types import NestedTensor


def test_tree() -> None:
    """Verify recursive PyTree operations, path flattening, and structure validation."""
    assert tree.MAP_TO_NONE is not None
    assert tree.is_nested(1) is False
    assert tree.is_nested([1, 2]) is True
    assert tree.is_nested((1, 2)) is True
    assert tree.is_nested({"a": 1}) is True

    nested: NestedTensor = {"b": [1, (2, 3)], "a": 0}
    assert tree.flatten(nested) == [0, 1, 2, 3]
    assert tree.flatten_with_path(nested) == [
        (("a",), 0),
        (("b", 0), 1),
        (("b", 1, 0), 2),
        (("b", 1, 1), 3),
    ]

    # assert_same_structure & assert_same_paths
    tree.assert_same_structure(nested, {"a": 10, "b": [20, (30, 40)]})
    tree.assert_same_paths(nested, {"a": 10, "b": [20, (30, 40)]})
    tree.assert_same_structure([1, 2], (1, 2), check_types=False)

    with pytest.raises(ValueError, match="Structures do not match"):
        tree.assert_same_structure(1, [1])
    with pytest.raises(TypeError, match="Container types do not match"):
        tree.assert_same_structure({"a": 1}, [1], check_types=True)
    with pytest.raises(ValueError, match="Structures do not match"):
        tree.assert_same_structure({"a": 1}, [1], check_types=False)
    with pytest.raises(ValueError, match="Dictionary keys do not match"):
        tree.assert_same_structure({"a": 1}, {"b": 1})
    with pytest.raises(TypeError, match="Sequence types do not match"):
        tree.assert_same_structure([1, 2], (1, 2), check_types=True)
    with pytest.raises(ValueError, match="Sequence lengths do not match"):
        tree.assert_same_structure([1, 2], [1, 2, 3])
    with pytest.raises(ValueError, match="Structure paths do not match"):
        tree.assert_same_paths({"a": 1}, {"b": 1})

    # lists_to_tuples
    assert tree.lists_to_tuples({"k": [1, [2, 3]], "t": (4, [5])}) == {
        "k": (1, (2, 3)),
        "t": (4, (5,)),
    }

    # map_structure
    mapped = tree.map_structure(lambda x, y: x + y, {"a": [1, 2]}, {"a": (10, 20)})
    assert mapped == {"a": [11, 22]}
    assert tree.map_structure(lambda x: tree.MAP_TO_NONE, (1, 2)) == (None, None)
    with pytest.raises(ValueError, match="Must provide at least one structure"):
        tree.map_structure(lambda x: x)

    # map_shape_structure
    shapes: NestedTensor = {"img": (None, 32, 32, 3), "seq": [(10, 4), ((5, 2),)]}
    batch_sizes = tree.map_shape_structure(lambda s: len(s), shapes)
    assert batch_sizes == {"img": 4, "seq": [2, (2,)]}
    assert tree.map_shape_structure(lambda x: x, 1) == 1
    with pytest.raises(ValueError, match="Must provide at least one structure"):
        tree.map_shape_structure(lambda s: s)

    # map_structure_up_to
    shallow: NestedTensor = {"a": 0, "b": [0, (0,)]}
    deep: NestedTensor = {"a": [1, 2], "b": [[3, 4], ([5, 6],)]}
    up_to_res = tree.map_structure_up_to(shallow, lambda x: len(x), deep)
    assert up_to_res == {"a": 2, "b": [2, (2,)]}
    with pytest.raises(ValueError, match="Must provide at least one structure"):
        tree.map_structure_up_to(shallow, lambda x: x)
    with pytest.raises(ValueError, match="missing key"):
        tree.map_structure_up_to({"a": 1}, lambda x: x, {"b": 1})
    with pytest.raises(ValueError, match="does not match shallow sequence"):
        tree.map_structure_up_to([1, 2], lambda x: x, [1])

    # pack_sequence_as
    repacked = tree.pack_sequence_as(nested, [10, 20, 30, 40])
    assert repacked == {"a": 10, "b": [20, (30, 40)]}
    with pytest.raises(ValueError, match="expected more elements"):
        tree.pack_sequence_as([1, 2, 3], [10])
    with pytest.raises(ValueError, match="flat_sequence has 3 elements"):
        tree.pack_sequence_as([1], [10, 20, 30])

    # traverse top_down and bottom_up
    visited_td: list[NestedTensor] = []

    def visitor_td(x: NestedTensor) -> NestedTensor | tree._MapToNoneSentinel | None:
        """Record node in top-down traversal.

        Args:
            x: Current node.

        Returns:
            NestedTensor | tree._MapToNoneSentinel | None: Transformation result.
        """
        visited_td.append(x)
        if x == 99:
            return tree.MAP_TO_NONE
        if x == 42:
            return 100
        return None

    t_out = tree.traverse(visitor_td, {"k": [99, (42, 1)]}, top_down=True)
    assert t_out == {"k": [None, (100, 1)]}

    visited_bu: list[NestedTensor] = []

    def visitor_bu(x: NestedTensor) -> NestedTensor | tree._MapToNoneSentinel | None:
        """Record node in bottom-up traversal.

        Args:
            x: Current node.

        Returns:
            NestedTensor | tree._MapToNoneSentinel | None: Transformation result.
        """
        visited_bu.append(x)
        if x == 99:
            return tree.MAP_TO_NONE
        if x == 42:
            return 100
        return None

    t_bu = tree.traverse(visitor_bu, {"k": [99, (42, 1)]}, top_down=False)
    assert t_bu == {"k": [None, (100, 1)]}
