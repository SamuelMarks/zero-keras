"""tree API."""

MAP_TO_NONE = object()


def assert_same_paths(a, b):
    """assert_same_paths docstring."""
    pass


def assert_same_structure(a, b, check_types=True):
    """assert_same_structure docstring."""
    pass


def flatten(structure):
    """flatten docstring."""
    return [structure]


def flatten_with_path(structure):
    """flatten_with_path docstring."""
    return [((), structure)]


def is_nested(structure):
    """is_nested docstring."""
    return False


def lists_to_tuples(structure):
    """lists_to_tuples docstring."""
    return structure


def map_shape_structure(func, *structures):
    """map_shape_structure docstring."""
    return structures[0]


def map_structure(func, *structures, **kwargs):
    """map_structure docstring."""
    return structures[0]


def map_structure_up_to(shallow_structure, func, *structures, **kwargs):
    """map_structure_up_to docstring."""
    return structures[0]


def pack_sequence_as(structure, flat_sequence):
    """pack_sequence_as docstring."""
    return structure


def traverse(func, structure, **kwargs):
    """traverse docstring."""
    pass


__all__ = [
    "MAP_TO_NONE",
    "assert_same_paths",
    "assert_same_structure",
    "flatten",
    "flatten_with_path",
    "is_nested",
    "lists_to_tuples",
    "map_shape_structure",
    "map_structure",
    "map_structure_up_to",
    "pack_sequence_as",
    "traverse",
]
