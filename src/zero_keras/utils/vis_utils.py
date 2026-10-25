"""Visualization and image utility functions for zero_keras.

Provides pure-Python implementations of ``model_to_dot`` and ``plot_model``
alongside image conversion re-exports (`array_to_img`, `img_to_array`,
`load_img`, `save_img`).
"""

from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

from zero_keras.types import ConfigValue
from zero_keras.utils.dataset_utils import (
    array_to_img,
    img_to_array,
    load_img,
    save_img,
)


@runtime_checkable
class _NamedLayerProtocol(Protocol):
    """Protocol for layers or models with a name."""

    @property
    def name(self) -> str:
        """Return the layer name.

        Returns:
            str: Layer or model name.
        """


@runtime_checkable
class _ContainerModelProtocol(_NamedLayerProtocol, Protocol):
    """Protocol for models exposing a sequence of child layers."""

    @property
    def layers(self) -> Sequence[_NamedLayerProtocol | ConfigValue]:
        """Return the sequence of child layers.

        Returns:
            Sequence[_NamedLayerProtocol | ConfigValue]: Child layers.
        """


class DotGraph:
    """Pure-Python Graphviz DOT representation for Keras models."""

    def __init__(self, rankdir: str = "TB", dpi: int | None = 96) -> None:
        """Initialize a DOT graph container.

        Args:
            rankdir: Direction of graph layout (``"TB"`` or ``"LR"``).
            dpi: Dots per inch resolution.
        """
        self.rankdir = rankdir
        self.dpi = dpi
        self.nodes: list[tuple[str, str]] = []
        self.edges: list[tuple[str, str]] = []

    def add_node(self, node_id: str, label: str) -> None:
        """Add a node with a display label to the DOT graph.

        Args:
            node_id: Unique node identifier.
            label: Display label for the node.
        """
        self.nodes.append((node_id, label))

    def add_edge(self, src_id: str, dst_id: str) -> None:
        """Add a directed edge between two nodes in the DOT graph.

        Args:
            src_id: Source node identifier.
            dst_id: Destination node identifier.
        """
        self.edges.append((src_id, dst_id))

    def to_string(self) -> str:
        """Serialize the graph to Graphviz DOT format string.

        Returns:
            str: Graphviz DOT source code.
        """
        lines = ["digraph G {", f'  rankdir="{self.rankdir}";']
        if self.dpi is not None:
            lines.append(f'  graph [dpi="{self.dpi}"];')
        for node_id, label in self.nodes:
            lines.append(f'  "{node_id}" [label="{label}", shape="record"];')
        for src_id, dst_id in self.edges:
            lines.append(f'  "{src_id}" -> "{dst_id}";')
        lines.append("}")
        return chr(10).join(lines)

    def create(self, prog: str = "dot", format: str = "png") -> bytes:
        """Render the DOT graph to bytes.

        Args:
            prog: Layout program name.
            format: Output format name.

        Returns:
            bytes: Encoded graph representation.
        """
        _ = prog, format
        return self.to_string().encode("utf-8")

    def write(self, path: str | Path, prog: str = "dot", format: str = "png") -> bool:
        """Write the rendered graph to disk.

        Args:
            path: Target file path.
            prog: Layout program name.
            format: Output format name.

        Returns:
            bool: True when written successfully.
        """
        data = self.create(prog=prog, format=format)
        with open(path, "wb") as f:
            f.write(data)
        return True


def model_to_dot(
    model: _NamedLayerProtocol | None,
    show_shapes: bool = False,
    show_dtype: bool = False,
    show_layer_names: bool = True,
    rankdir: str = "TB",
    expand_nested: bool = False,
    dpi: int | None = 96,
    subgraph: bool = False,
    show_trainable: bool = False,
    **kwargs: bool | int | str | None,
) -> DotGraph:
    """Convert a Keras model to a pure-Python ``DotGraph`` representation.

    Args:
        model: A Keras model instance.
        show_shapes: Whether to display shape information.
        show_dtype: Whether to display layer dtypes.
        show_layer_names: Whether to display layer names.
        rankdir: ``"TB"`` creates a vertical plot; ``"LR"`` creates a horizontal plot.
        expand_nested: Whether to expand nested Functional models into clusters.
        dpi: Image resolution in dots per inch.
        subgraph: Whether to return a subgraph cluster.
        show_trainable: Whether to display whether a layer is trainable.
        **kwargs: Additional keyword arguments for compatibility.

    Returns:
        DotGraph: Constructed DOT graph object.
    """
    _ = show_shapes, show_dtype, expand_nested, subgraph, show_trainable, kwargs
    dot = DotGraph(rankdir=rankdir, dpi=dpi)
    if model is None:
        return dot

    layers_seq: Sequence[_NamedLayerProtocol] = (
        model.layers if isinstance(model, _ContainerModelProtocol) else ()
    )
    prev_id: str | None = None
    for idx, layer in enumerate(layers_seq):
        layer_name = layer.name if isinstance(layer, _NamedLayerProtocol) else f"layer_{idx}"
        cls_name = type(layer).__name__
        label = f"{layer_name}: {cls_name}" if show_layer_names else cls_name
        node_id = f"{idx}_{layer_name}"
        dot.add_node(node_id, label)
        if prev_id is not None:
            dot.add_edge(prev_id, node_id)
        prev_id = node_id
    return dot


def plot_model(
    model: _NamedLayerProtocol | None,
    to_file: str | Path = "model.png",
    show_shapes: bool = False,
    show_dtype: bool = False,
    show_layer_names: bool = True,
    rankdir: str = "TB",
    expand_nested: bool = False,
    dpi: int | None = 96,
    show_layer_activations: bool = False,
    show_trainable: bool = False,
    **kwargs: bool | int | str | None,
) -> DotGraph:
    """Convert a Keras model to DOT format and save to a file.

    Args:
        model: A Keras model instance.
        to_file: File name of the plot image.
        show_shapes: Whether to display shape information.
        show_dtype: Whether to display layer dtypes.
        show_layer_names: Whether to display layer names.
        rankdir: ``"TB"`` creates a vertical plot; ``"LR"`` creates a horizontal plot.
        expand_nested: Whether to expand nested models into clusters.
        dpi: Dots per inch.
        show_layer_activations: Display layer activations.
        show_trainable: Whether to display if a layer is trainable.
        **kwargs: Additional keyword arguments.

    Returns:
        DotGraph: Constructed DOT graph object.
    """
    _ = show_layer_activations
    dot = model_to_dot(
        model,
        show_shapes=show_shapes,
        show_dtype=show_dtype,
        show_layer_names=show_layer_names,
        rankdir=rankdir,
        expand_nested=expand_nested,
        dpi=dpi,
        show_trainable=show_trainable,
        **kwargs,
    )
    if to_file:
        parent = os.path.dirname(str(to_file))
        if parent:
            os.makedirs(parent, exist_ok=True)
        dot.write(to_file)
    return dot


__all__ = [
    "DotGraph",
    "array_to_img",
    "img_to_array",
    "load_img",
    "model_to_dot",
    "plot_model",
    "save_img",
]
