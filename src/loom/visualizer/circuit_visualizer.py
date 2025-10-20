"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.
"""

from collections import defaultdict

import numpy as np
import networkx as nx
import plotly.graph_objs as go

from loom.eka import Circuit

from .plotting_utils import convert_circuit_to_nx_graph


# pylint: disable=too-many-locals
def hierarchy_layout(graph, root=None, x_spacing=1.0, y_spacing=1.5):
    """
    Pure-Python replacement for graphviz_layout (hierarchical layout).

    Parameters
    ----------
    graph : nx.DiGraph
        Directed acyclic graph to layout.
    root : node or None
        Optional root node. If None, the first node with in-degree 0 is used.
    x_spacing : float
        Horizontal spacing between sibling nodes.
    y_spacing : float
        Vertical spacing between levels.

    Returns
    -------
    dict
        Mapping of node -> (x, y) positions.
    """

    if not nx.is_directed_acyclic_graph(graph):
        graph = nx.DiGraph(graph)

    # Choose a root if none provided
    if root is None:
        roots = [n for n, deg in graph.in_degree() if deg == 0]
        root = roots[0] if roots else list(graph.nodes())[0]

    # BFS/topological sort to determine depth/layer of each node
    layers = defaultdict(list)
    for node in nx.topological_sort(graph):
        preds = list(graph.predecessors(node))
        depth = (
            0 if not preds else 1 + max(graph.nodes[p].get("_depth", 0) for p in preds)
        )
        graph.nodes[node]["_depth"] = depth
        layers[depth].append(node)

    # Assign x, y positions
    pos = {}
    max_width = max(len(nodes) for nodes in layers.values())
    for depth, nodes in layers.items():
        width = len(nodes)
        for i, node in enumerate(nodes):
            x = (i - (width - 1) / 2) * x_spacing * (max_width / width)
            y = -depth * y_spacing
            pos[node] = (x, y)
    return pos


def plot_circuit_tree(  # pylint: disable=too-many-locals
    circuit: Circuit,
    max_layer: int | None = None,
    layer_colors: list[str] | None = None,
    layer_labels: list[str] | None = None,
    num_layers_with_text: int | None = 1,
) -> go.Figure:
    """
    Plot the tree structure of a `Circuit` object.

    Parameters
    ----------
    circuit: Circuit
        Circuit which should be plotted
    max_layer: int | None
        Maximum layer up to which the tree should be plotted.
        If None is provided, all layers are plotted.
    layer_colors: list[str] | None
        Array of colors for the markers in different layers.
    layer_labels: list[str] | None
        Array of labels for the different layers.
        If None is provided, they are labeled by their number.
    num_layers_with_text: int | None
        Number of layers where their name is written on top of the markers

    Returns
    -------
    go.Figure
        Interactive `go.Figure` object for the circuit tree
    """

    # Default colors for the different layers
    if layer_colors is None:
        layer_colors = [
            "#054352",
            "#536070",
            "#857972",
            "#94AD72",
            "#D1E071",
            "#B39F67",
            "#946F52",
            "#D16F4D",
            "#E64040",
            "#990538",
            "#61105B",
            "#3F2882",
            "#005FA8",
        ]

    # Get an nx graph representing the circuit and a list of labels for all nodes
    graph, labels_nodes = convert_circuit_to_nx_graph(circuit)

    # Tree-like layout (pure Python, no Graphviz required)
    positions = hierarchy_layout(graph)

    nodes_x_coords = [x for x, y in positions.values()]
    nodes_y_coords = [y for x, y in positions.values()]

    # Create mapping from node ID to coordinates
    node_positions = {node: (x, y) for node, (x, y) in positions.items()}
    # Create lists of edge coordinates
    edge_x_coords = []
    edge_y_coords = []
    for u, v in graph.edges():
        x0, y0 = node_positions[u]
        x1, y1 = node_positions[v]
        edge_x_coords.append([x0, x1, None])
        edge_y_coords.append([y0, y1, None])

    # Sorted list of unique y-coordinates for layers
    nodes_y_coords_set = sorted({y for _, y in node_positions.values()}, reverse=True)

    # Check whether provided layer labels are valid
    if layer_labels is None:
        layer_labels = [
            f"Layer {layer_idx+1}" for layer_idx in range(len(nodes_y_coords_set))
        ]
    else:
        if len(layer_labels) != len(nodes_y_coords_set):
            raise ValueError(
                "Number of layer labels does not match the "
                f"number of layers. {len(layer_labels)} layer "
                f"labels were provided while there are "
                f"{len(nodes_y_coords_set)} layers."
            )

    fig = go.Figure()

    # This dummy trace is not visible but needed for the correct ordering of layers
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            name="Layer 1",
            mode="lines",
            line={"color": "rgb(210,210,210)", "width": 1},
            hoverinfo="none",
            legendgroup="layer_1",
            showlegend=False,
        )
    )

    # Create the condition for two elements close on the y axis
    def is_close_in_y(y1, y2, tol=1e-3):
        """Check if two y-coordinates are close to each other."""
        return np.abs(y1 - y2) < tol

    # Draw lines from a circuit to its subcircuits
    for layer_idx, y_val in enumerate(nodes_y_coords_set):
        if max_layer is not None and layer_idx >= max_layer - 1:
            break
        points_in_this_layer = np.array(
            [
                (xs, ys)
                for xs, ys in zip(edge_x_coords, edge_y_coords, strict=True)
                if is_close_in_y(ys[0], y_val, tol=1e-3)
            ]
        )
        if len(points_in_this_layer) > 0:
            fig.add_trace(
                go.Scatter(
                    x=points_in_this_layer[:, 0].flatten(),
                    y=points_in_this_layer[:, 1].flatten(),
                    name=f"Layer {layer_idx+2}",
                    mode="lines",
                    line={"color": "rgb(210,210,210)", "width": 1},
                    hoverinfo="none",
                    legendgroup=f"layer_{layer_idx+2}",
                    showlegend=False,
                )
            )

    # Plot nodes
    for layer_idx, y_val in enumerate(nodes_y_coords_set):
        if max_layer is not None and layer_idx >= max_layer:
            break
        background_color = layer_colors[layer_idx % len(layer_colors)]
        points_in_this_layer = np.array(
            [
                (x, y, label)
                for x, y, label in zip(
                    nodes_x_coords, nodes_y_coords, labels_nodes, strict=True
                )
                if is_close_in_y(y, y_val, tol=1e-3)
            ]
        )
        if num_layers_with_text is not None and layer_idx < num_layers_with_text:
            mode = "markers+text"
            hoverinfo = "none"
        else:
            mode = "markers"
            hoverinfo = "text"
        fig.add_trace(
            go.Scatter(
                x=points_in_this_layer[:, 0],
                y=points_in_this_layer[:, 1],
                mode=mode,
                name=layer_labels[layer_idx],
                marker={
                    "symbol": "circle-dot",
                    "size": 18,
                    "color": background_color,
                    "line": {"color": "rgb(50,50,50)", "width": 1},
                },
                text=points_in_this_layer[:, 2],
                textposition="top center",
                hoverinfo=hoverinfo,
                opacity=0.8,
                legendgroup=f"layer_{layer_idx+1}",
            )
        )

    fig.update_layout(
        xaxis={
            "showgrid": False,
            "zeroline": False,
            "visible": False,
        },
        yaxis={
            "showgrid": False,
            "visible": False,
        },
    )

    return fig
