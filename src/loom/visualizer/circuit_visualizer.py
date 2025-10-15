"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import numpy as np
import plotly.graph_objs as go

from loom.eka import Circuit

from .plotting_utils import convert_circuit_to_igraph


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

    # Get an igraph object representing the circuit and a list of labels for all nodes
    graph, labels_nodes = convert_circuit_to_igraph(circuit)

    # Calculate the coordinates of the nodes in the tree
    nr_vertices = graph.vcount()

    # Get the tree layout using the Reingold-Tilford method.
    # 'layout' is a list of tuples with the x and y coordinates of the nodes
    layout = graph.layout("rt")
    positions = {k: layout[k] for k in range(nr_vertices)}  # Create dict for positions

    # Create separate lists for the x and y coordinates of the nodes.
    # The y coordinates are flipped to have the root node at the top
    max_y = max(layout[k][1] for k in range(nr_vertices))
    nodes_x_coords = [positions[k][0] for k in range(nr_vertices)]
    nodes_y_coords = [max_y - positions[k][1] for k in range(nr_vertices)]

    # Create lists of edge coordinates
    # Flip the y-coordinates of the edges in the same way as the nodes
    edge_coords = [e.tuple for e in graph.es]  # List of edges
    edge_x_coords = []
    edge_y_coords = []
    for edge in edge_coords:
        edge_x_coords += [[positions[edge[0]][0], positions[edge[1]][0], None]]
        edge_y_coords += [
            [max_y - positions[edge[0]][1], max_y - positions[edge[1]][1], None]
        ]

    # Sorted list of unique y-coordinates of the nodes
    nodes_y_coords_set = sorted(set(nodes_y_coords), reverse=True)

    # Check whether provided layer labels are valid
    # and generate default labels if none are provided
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
                    showlegend=False,  # Dont't show this label. Instead
                    # the labels from the dots (circuit elements) are used
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
