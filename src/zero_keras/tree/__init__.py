"""Nested structure (PyTree) manipulation utilities for zero_keras."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from zero_keras.types import ConfigValue, NestedTensor


class _MapToNoneSentinel:
    """Sentinel indicating that a mapped leaf should be replaced with None."""


MAP_TO_NONE: _MapToNoneSentinel = _MapToNoneSentinel()


def is_nested(structure: NestedTensor) -> bool:
    """Return whether ``structure`` is a nested container (list, tuple, or dict).

    Args:
        structure: Candidate value to inspect.

    Returns:
        bool: True if ``structure`` is a list, tuple, or dict.
    """
    return isinstance(structure, (list, tuple, dict))


def flatten_with_path(
    structure: NestedTensor,
) -> list[tuple[tuple[str | int, ...], NestedTensor]]:
    """Flatten a nested structure into ``(path, leaf)`` pairs.

    Args:
        structure: Nested structure to flatten.

    Returns:
        list[tuple[tuple[str | int, ...], NestedTensor]]: Path-qualified leaf list.
    """
    results: list[tuple[tuple[str | int, ...], NestedTensor]] = []

    def _walk(node: NestedTensor, path: tuple[str | int, ...]) -> None:
        """Recursively collect ``(path, leaf)`` pairs from ``node``.

        Args:
            node: Current node in the structure.
            path: Tuple of keys/indices leading to ``node``.
        """
        if isinstance(node, dict):
            for k in sorted(node.keys()):
                _walk(node[k], (*path, str(k)))
        elif isinstance(node, (list, tuple)):
            for idx, item in enumerate(node):
                _walk(item, (*path, idx))
        else:
            results.append((path, node))

    _walk(structure, ())
    return results


def flatten(structure: NestedTensor) -> list[NestedTensor]:
    """Flatten a nested structure into a flat list of leaves.

    Args:
        structure: Nested structure to flatten.

    Returns:
        list[NestedTensor]: Flat list of leaf elements.
    """
    return [leaf for _, leaf in flatten_with_path(structure)]


def assert_same_structure(
    a: NestedTensor,
    b: NestedTensor,
    check_types: bool = True,
) -> None:
    """Assert that two nested structures have the same layout.

    Args:
        a: First nested structure.
        b: Second nested structure.
        check_types: Whether container types (list vs tuple) must match exactly.

    Raises:
        TypeError: If ``check_types=True`` and container types differ.
        ValueError: If nesting structure, sequence lengths, or dict keys differ.
    """
    a_nested = is_nested(a)
    b_nested = is_nested(b)
    if a_nested != b_nested:
        raise ValueError(f"Structures do not match: {a!r} vs {b!r}")
    if not a_nested:
        return
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            if check_types:
                raise TypeError(f"Container types do not match: {type(a)} vs {type(b)}")
            raise ValueError(f"Structures do not match: {a!r} vs {b!r}")
        if set(a.keys()) != set(b.keys()):
            raise ValueError(
                f"Dictionary keys do not match: {sorted(a.keys())} vs {sorted(b.keys())}"
            )
        for k in sorted(a.keys()):
            assert_same_structure(a[k], b[k], check_types=check_types)
        return
    if check_types and (isinstance(a, list) != isinstance(b, list)):
        raise TypeError(f"Sequence types do not match: {type(a)} vs {type(b)}")
    seq_a = a if isinstance(a, (list, tuple)) else ()
    seq_b = b if isinstance(b, (list, tuple)) else ()
    if len(seq_a) != len(seq_b):
        raise ValueError(f"Sequence lengths do not match: {len(seq_a)} vs {len(seq_b)}")
    for item_a, item_b in zip(seq_a, seq_b):
        assert_same_structure(item_a, item_b, check_types=check_types)


def assert_same_paths(a: NestedTensor, b: NestedTensor) -> None:
    """Assert that two nested structures share identical key/index paths.

    Args:
        a: First nested structure.
        b: Second nested structure.

    Raises:
        ValueError: If leaf paths of ``a`` and ``b`` differ.
    """
    paths_a = [p for p, _ in flatten_with_path(a)]
    paths_b = [p for p, _ in flatten_with_path(b)]
    if paths_a != paths_b:
        raise ValueError(f"Structure paths do not match: {paths_a} vs {paths_b}")


def lists_to_tuples(structure: NestedTensor) -> NestedTensor:
    """Convert all lists in ``structure`` into tuples recursively.

    Args:
        structure: Input nested structure.

    Returns:
        NestedTensor: Structure with lists replaced by tuples.
    """
    if isinstance(structure, dict):
        return {k: lists_to_tuples(v) for k, v in structure.items()}
    if isinstance(structure, (list, tuple)):
        return tuple(lists_to_tuples(item) for item in structure)
    return structure


def _normalize_mapped_leaf(val: NestedTensor | _MapToNoneSentinel) -> NestedTensor:
    """Convert ``MAP_TO_NONE`` sentinel results into ``None``.

    Args:
        val: Raw mapped leaf value.

    Returns:
        NestedTensor: ``None`` if ``val is MAP_TO_NONE``, else ``val``.
    """
    if isinstance(val, _MapToNoneSentinel):
        return None
    return val


def map_structure(
    func: Callable[..., NestedTensor | _MapToNoneSentinel],
    *structures: NestedTensor,
    none_is_leaf: bool = False,
    **kwargs: ConfigValue,
) -> NestedTensor:
    """Apply ``func`` to each leaf across ``structures``.

    Args:
        func: Callable applied to corresponding leaves.
        *structures: One or more nested structures with matching layout.
        none_is_leaf: Whether ``None`` is treated as a leaf value.
        **kwargs: Additional configuration options.

    Returns:
        NestedTensor: New nested structure containing mapped leaves.

    Raises:
        ValueError: If no structures are provided or structures do not match.
    """
    _ = none_is_leaf, kwargs
    if not structures:
        raise ValueError("Must provide at least one structure to map_structure.")
    first = structures[0]
    for other in structures[1:]:
        assert_same_structure(first, other, check_types=False)

    if isinstance(first, dict):
        out_dict: dict[str, NestedTensor] = {}
        for k in sorted(first.keys()):
            child_args = [s[k] for s in structures if isinstance(s, dict)]
            out_dict[k] = map_structure(func, *child_args)
        return out_dict
    if isinstance(first, list):
        return [
            map_structure(
                func,
                *[s[idx] for s in structures if isinstance(s, (list, tuple))],
            )
            for idx in range(len(first))
        ]
    if isinstance(first, tuple):
        return tuple(
            map_structure(
                func,
                *[s[idx] for s in structures if isinstance(s, (list, tuple))],
            )
            for idx in range(len(first))
        )
    return _normalize_mapped_leaf(func(*structures))


def _is_shape_tuple(node: NestedTensor) -> bool:
    """Return True if ``node`` is a tuple of ints/Nones representing a tensor shape.

    Args:
        node: Candidate structure node.

    Returns:
        bool: True if ``node`` is a shape tuple.
    """
    return isinstance(node, tuple) and all(isinstance(x, int) or x is None for x in node)


def map_shape_structure(
    func: Callable[..., NestedTensor],
    *structures: NestedTensor,
) -> NestedTensor:
    """Apply ``func`` to shape tuples (or leaves) within ``structures``.

    Args:
        func: Mapping function to apply to each shape tuple.
        *structures: Input nested structures of shapes.

    Returns:
        NestedTensor: Mapped shape structure.

    Raises:
        ValueError: If no structures are provided.
    """
    if not structures:
        raise ValueError("Must provide at least one structure to map_shape_structure.")
    first = structures[0]
    if _is_shape_tuple(first) or not is_nested(first):
        return func(*structures)
    if isinstance(first, dict):
        return {
            k: map_shape_structure(
                func,
                *[s[k] for s in structures if isinstance(s, dict)],
            )
            for k in sorted(first.keys())
        }
    if isinstance(first, list):
        return [
            map_shape_structure(
                func,
                *[s[idx] for s in structures if isinstance(s, (list, tuple))],
            )
            for idx in range(len(first))
        ]
    first_seq = first if isinstance(first, tuple) else ()
    return tuple(
        map_shape_structure(
            func,
            *[s[idx] for s in structures if isinstance(s, (list, tuple))],
        )
        for idx in range(len(first_seq))
    )


def map_structure_up_to(
    shallow_structure: NestedTensor,
    func: Callable[..., NestedTensor],
    *structures: NestedTensor,
    **kwargs: ConfigValue,
) -> NestedTensor:
    """Apply ``func`` up to the nesting depth of ``shallow_structure``.

    Args:
        shallow_structure: Template structure defining recursion depth.
        func: Callable applied at shallow leaves.
        *structures: Input structures to map.
        **kwargs: Additional keyword arguments.

    Returns:
        NestedTensor: Mapped structure matching ``shallow_structure`` depth.

    Raises:
        ValueError: If no structures are provided or shallow layout mismatches.
    """
    _ = kwargs
    if not structures:
        raise ValueError("Must provide at least one structure to map_structure_up_to.")
    if not is_nested(shallow_structure):
        return func(*structures)
    if isinstance(shallow_structure, dict):
        out_dict: dict[str, NestedTensor] = {}
        for k in sorted(shallow_structure.keys()):
            child_args: list[NestedTensor] = []
            for s in structures:
                if not isinstance(s, dict) or k not in s:
                    raise ValueError(f"Structure {s!r} missing key {k!r}")
                child_args.append(s[k])
            out_dict[k] = map_structure_up_to(shallow_structure[k], func, *child_args)
        return out_dict
    shallow_seq = shallow_structure if isinstance(shallow_structure, (list, tuple)) else ()
    items: list[NestedTensor] = []
    for idx, sub_shallow in enumerate(shallow_seq):
        child_args = []
        for s in structures:
            if not isinstance(s, (list, tuple)) or len(s) != len(shallow_seq):
                raise ValueError(
                    f"Structure {s!r} does not match shallow sequence {shallow_structure!r}"
                )
            child_args.append(s[idx])
        items.append(map_structure_up_to(sub_shallow, func, *child_args))
    return tuple(items) if isinstance(shallow_structure, tuple) else items


def pack_sequence_as(
    structure: NestedTensor,
    flat_sequence: Sequence[NestedTensor],
) -> NestedTensor:
    """Pack ``flat_sequence`` into the layout of ``structure``.

    Args:
        structure: Target structure template.
        flat_sequence: Sequence of leaf elements.

    Returns:
        NestedTensor: Repacked nested structure.

    Raises:
        ValueError: If ``flat_sequence`` has fewer or more elements than ``structure``.
    """
    cursor = [0]

    def _pack(template: NestedTensor) -> NestedTensor:
        """Consume elements from ``flat_sequence`` to populate ``template``.

        Args:
            template: Current template node.

        Returns:
            NestedTensor: Populated structure node.
        """
        if isinstance(template, dict):
            out: dict[str, NestedTensor] = {}
            for k in sorted(template.keys()):
                out[k] = _pack(template[k])
            return out
        if isinstance(template, list):
            return [_pack(item) for item in template]
        if isinstance(template, tuple):
            return tuple(_pack(item) for item in template)
        if cursor[0] >= len(flat_sequence):
            raise ValueError(
                f"Could not pack sequence: expected more elements than {len(flat_sequence)}."
            )
        val = flat_sequence[cursor[0]]
        cursor[0] += 1
        return val

    packed = _pack(structure)
    if cursor[0] != len(flat_sequence):
        raise ValueError(
            f"Could not pack sequence: structure has {cursor[0]} leaves, "
            f"but flat_sequence has {len(flat_sequence)} elements."
        )
    return packed


def traverse(
    func: Callable[[NestedTensor], NestedTensor | _MapToNoneSentinel | None],
    structure: NestedTensor,
    top_down: bool = True,
) -> NestedTensor:
    """Traverse ``structure`` applying visitor ``func``.

    Args:
        func: Visitor callable returning ``None`` to continue, ``MAP_TO_NONE`` to
            replace with ``None``, or a replacement subtree.
        structure: Nested structure to traverse.
        top_down: Whether to apply ``func`` before (True) or after (False) children.

    Returns:
        NestedTensor: Traversed and potentially transformed structure.
    """
    if top_down:
        res = func(structure)
        if isinstance(res, _MapToNoneSentinel):
            return None
        if res is not None:
            return res

    if isinstance(structure, dict):
        traversed: NestedTensor = {
            k: traverse(func, structure[k], top_down=top_down) for k in sorted(structure.keys())
        }
    elif isinstance(structure, list):
        traversed = [traverse(func, item, top_down=top_down) for item in structure]
    elif isinstance(structure, tuple):
        traversed = tuple(traverse(func, item, top_down=top_down) for item in structure)
    else:
        traversed = structure

    if not top_down:
        res_post = func(traversed)
        if isinstance(res_post, _MapToNoneSentinel):
            return None
        if res_post is not None:
            return res_post
    return traversed


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
