"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import numpy as np
import plotly.graph_objs as go
import networkx as nx

from loom.eka import Circuit


def hex_to_rgb(hex_color: str) -> list[int]:
    """
    Converts a hexadecimal color code to a list of RGB values between 0 and 255.

    Parameters
    ----------
    hex_color : str
        The hexadecimal color code (e.g., "#RRGGBB").

    Returns
    -------
    list[int]
        RGB values between 0 and 255 representing the color.
    """
    # Remove "#" if present
    if hex_color.startswith("#"):
        hex_color = hex_color[1:]

    # Convert hex color to RGB components
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return [r, g, b]


def rgb_to_hex(rgb_color: list[int]) -> str:
    """
    Converts a color from RGB format to HEX format.

    Parameters
    ----------
    rgb_color : list[int]
        RGB values between 0 and 255 representing the color.

    Returns
    -------
    str
        The hexadecimal color code (e.g., "#RRGGBB").
    """
    return f"#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}"


def average_color_hex(color_list: list[str]) -> str:
    """
    Calculate the average color from a list of colors in HEX format.

    Parameters
    ----------
    color_list : list[str]
        List of hexadecimal colors

    Returns
    -------
    str
        The average color in HEX format
    """
    rgb_colors = [hex_to_rgb(color) for color in color_list]
    avg_rgb = tuple(
        int(sum(color_channel) / len(color_list))
        for color_channel in zip(*rgb_colors, strict=True)
    )
    return rgb_to_hex(avg_rgb)


def change_color_brightness(hex_color: str, factor: float) -> str:
    """
    Change the brightness of a color by a given factor.

    Parameters
    ----------
    hex_color : str
        The hexadecimal color code (e.g., "#RRGGBB").
    factor : float
        The factor by which the brightness of the color should be multiplied.

    Returns
    -------
    str
        The new color in HEX format.
    """
    if factor < 0:
        raise ValueError("Factor must be greater than or equal to 0.")
    rgb_color = hex_to_rgb(hex_color)
    new_rgb_color = [min(int(np.round(color * factor)), 255) for color in rgb_color]
    new_hex_color = rgb_to_hex(new_rgb_color)
    return new_hex_color


def get_font_color(background_color: str) -> str:
    """
    Determine font color based on background color. The font color is white for dark
    backgrounds and gray or black for bright backgrounds.

    Parameters
    ----------
    background_color : str
        Background color in hexadecimal format (e.g., '#RRGGBB')

    Returns
    -------
    str
        Font color for a good contrast with the background color
    """
    # Convert hexadecimal color to RGB
    r = int(background_color[1:3], 16)
    g = int(background_color[3:5], 16)
    b = int(background_color[5:], 16)

    # Calculate luminance (YIQ formula)
    luminance = (r * 0.299 + g * 0.587 + b * 0.114) / 255

    # Determine font color based on luminance
    if luminance > 0.7:
        return "#000000"
    if luminance > 0.5:
        return "#222222"
    return "#ffffff"


def point_in_polygon(
    x: float,
    y: float,
    polygon: list[tuple[float, float]],
) -> bool:
    """
    Check if a point (x, y) is inside a polygon defined by its corners. This
    algorithm is based on the ray casting algorithm. Note that the behaviour for points
    on the edges or very close to them is undefined.

    Parameters
    ----------
    x : float
        x coordinate of the point to check
    y : float
        y coordinate of the point to check
    polygon : list[tuple[float, float]]
        List of tuples representing the corners of the polygon

    Returns
    -------
    bool
        True if the point is inside the polygon, False otherwise
    """
    # How the ray casting algorithm works:
    # We consider a horizontal line starting from the point to check and going to the
    # right. We count how many times this line intersects with the edges of the polygon.
    # If the number of intersections is odd, the point is inside the polygon. If it is
    # even, the point is outside the polygon.
    # In the code, instead of an integer counter, we use a boolean variable is_inside
    # which is toggled every time the line intersects with an edge.
    is_inside = False
    for i, corner in enumerate(polygon):  # Iterate over each edge of the polygon
        # corner = polygon[i] is the current corner
        # polygon[j] is the next corner of the polygon
        j = (i + 1) % len(polygon)
        next_corner = polygon[j]

        # Check if the y coordinate of the point is between the upper and lower y
        # coordinates of the edge
        point_is_in_y_range = (corner[1] > y) != (next_corner[1] > y)
        # If the point is not in the y range, we can skip this edge since there cannot
        # be an intersection
        if not point_is_in_y_range:
            continue

        # Check if the x coordinate of the point is to the left of the point on the edge
        # which has the same y coordinate as the given point
        if abs(next_corner[1] - corner[1]) > 1e-9:  # Avoid division by zero
            inverse_slope = (next_corner[0] - corner[0]) / (
                next_corner[1] - corner[1]
            )  # Inverse slope of line going from (x1, y1) to (x2, y2) is (x2 - x1) / (y2 - y1)

            point_is_left_of_edge = x < corner[0] + inverse_slope * (y - corner[1])

            # Check if the horizontal line going to the right intersects with the edge
            if point_is_left_of_edge:
                is_inside = not is_inside
        else:
            # This is the case where the edge is practically horizontal and the point
            # might or might not lie on the edge. As stated in the docstring, the
            # behaviour for points on the edges or very close to them is undefined.
            # Therefore, we just skip this edge.
            continue

    return is_inside


def interpolate_values(
    point: tuple[float, float],
    interpolation_points: list[tuple[float, float]],
    interpolation_values: list[float] | list[list[float]],
    interpolation_power: float | None = None,
    min_distance: float | None = 1e-3,
) -> np.ndarray:
    """
    Perform interpolation of the values in `interpolation_values` which are
    associated to the points in `interpolation_points`, based on the distances to a
    given point. The values in `interpolation_values` are either floats or lists of
    floats. If they are a list, the interpolation is done element- wise. E.g. every
    interpolation point could have a tuple (r, g, b) of three integers, representing an
    RGB color. Then the weighted average of these RGB tuples is calculated with weights
    determined by the distances to these points.

    Parameters
    ----------
    point : tuple[float, float]
        Tuple representing the coordinates of the given point
    interpolation_points : list[tuple[float, float]]
        List of points at which the values in interpolation_values are given
    interpolation_values : list[float] | list[list[float]]
        List of values associated to the points in interpolation_points. The values can
        be floats or lists of floats. If they are lists, the interpolation is done
        element-wise.
    interpolation_power : float | None
        The interpolation_power parameter determines how rapidly the interpolated value
        changes with increasing distance. The weights with which the values of the n
        points are interpolated are given as 1 / distance**interpolation_power.
        The default value is 2.
    min_distance : float | None
        Points which are closer to one of the given points than min_distance will get a
        value equal to the value of this point. This is to prevent division by zero when
        calculating the weights.

    Returns
    -------
    np.ndarray
        Interpolated value(s) based on the distances from the given point. The length of
        the array is equal to the length of the values in interpolation_values.
    """
    if interpolation_power is None:
        interpolation_power = 2

    # Make sure the number of interpolation points and interpolation values is the same
    if len(interpolation_points) != len(interpolation_values):
        raise ValueError(
            "The number of interpolation points and interpolation values must be the same."
        )

    # Make sure every element of interpolation_values is a list.
    # If it is a single value, convert it to a list of length 1.
    for i, val in enumerate(interpolation_values):
        if not isinstance(val, (tuple, list)):
            interpolation_values[i] = [val]

    # Check if all interpolation_values have the same dimension
    if any(len(val) != len(interpolation_values[0]) for val in interpolation_values):
        raise ValueError(
            "All values in interpolation_values must have the same dimension."
        )

    # Check that all points in interpolation_points are tuples of length 2
    if any(
        not (isinstance(point, tuple)) or len(point) != 2
        for point in interpolation_points
    ):
        raise ValueError(
            "All points in interpolation_points must be tuples of length 2."
        )

    total_value = np.array([0] * len(interpolation_values[0]), dtype=float)
    total_weight = 0

    # Calculate the weighted average of the values of the n points
    for pt, value in zip(interpolation_points, interpolation_values, strict=True):
        distance = ((point[0] - pt[0]) ** 2 + (point[1] - pt[1]) ** 2) ** 0.5
        if distance < min_distance:
            return value

        weight = 1 / distance**interpolation_power
        total_value += np.array(value) * weight
        total_weight += weight

    return total_value / total_weight


def center_of_points(points: list[tuple[float, float]]) -> np.ndarray:
    """Calculate the center of a list of points."""
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    return np.array([np.mean(x_coords), np.mean(y_coords)])


def center_of_scatter_plot(scatter_plot: go.Scatter) -> np.ndarray:
    """Calculate the center of the points contained in a scatter plot."""
    coords = np.stack((scatter_plot.x, scatter_plot.y), axis=1)
    return np.average(coords, axis=0)


def get_angle_from_x_axis(
    point1: tuple[float, float], point2: tuple[float, float]
) -> float:
    """
    Calculate the angle between the vector from point1 to point2 and the x-axis.

    E.g. for point1 == [0,0] and point2 == [1,0], the angle is 0. For point1 == [0,0]
    and point2 == [0,1], the angle is pi/2 (90 degree).
    """
    delta = np.array(point2) - np.array(point1)
    return np.arctan2(delta[1], delta[0])


def order_points_counterclockwise(
    points: list[tuple[float, float, list[any]]],
) -> list[tuple[float, float, list[any]]]:
    """
    Order the provided list of points counterclockwise around the center point.

    Parameters
    ----------
    points : list[tuple[float, float, list[any]]]
        List of points to be ordered. Every element in the list is a tuple which contains
        the x and y coordinate of the point as the first two elements. The third element
        is a list which can store additional metadata.

    Returns
    -------
    list[tuple[float, float, list[any]]]
        List of points ordered counterclockwise around the center point of the provided
        list. The list still contains the original metadata. As an additional metadata,
        the angle for the every point was added to its respective list.
    """
    center_point = center_of_points(points)
    points_with_angles = [list(point) for point in points]
    for point, point_with_angle in zip(points, points_with_angles, strict=True):
        # If the points have no additional metadata, add an empty list to add the angle
        if len(point_with_angle) == 2:
            point_with_angle.append([])

        point_with_angle[2].append(
            get_angle_from_x_axis(center_point, np.array([point[0], point[1]]))
        )
    sorted_points = sorted(points_with_angles, key=lambda x: x[2][-1])
    return sorted_points


# pylint: disable=too-many-arguments, too-many-positional-arguments
def draw_half_circle(
    center: list[float],
    r: float,
    direction: float | None = 0,
    name: str | None = "",
    text: str | None = "",
    fillcolor: str | None = "white",
    line: dict | None = None,
    showlegend: bool = True,
    legendgroup: str | None = None,
) -> go.Scatter:
    """
    Generate a scatter plot for a half circle.

    Parameters
    ----------
    center : list[float]
        Center point around which the half circle should be drawn
    r : float
        Radius of the half circle
    direction : float | None
        Direction in which the half circle should point. E.g.,

        - If direction == 0, half circle points upwards.
        - If direction == pi/2, half circle points right.
        - If direction == - pi/2, half circle points left.
    name : str | None
        Name parameter of the scatter plot
    text : str | None
        Text parameter of the scatter plot
    fillcolor : str | None
        Fill color of the half circle
    line : dict | None
        Line parameter of the scatter plot
    showlegend : bool, optional
        If True (the default), the stabilizers are shown in the legend.
    legendgroup: str | None
        Legend group to which the stabilizers belong.

    Returns
    -------
    go.Scatter
        Scatter plot with the half circle
    """
    if line is None:
        line = {}
    theta = np.linspace(0, np.pi, 100)

    # Calculate x and y coordinates for half circle
    x = center[0] + r * np.cos(-direction + theta)
    y = center[1] + r * np.sin(-direction + theta)

    # Add the starting point to have a closed shape
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    # Add half circle as scatter plot
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=name,
        fill="toself",
        fillcolor=fillcolor,
        line=line,
        text=text,
        hoverinfo="text",
        showlegend=showlegend,
        legendgroup=legendgroup,
    )


def get_label_for_circuit(circ: Circuit) -> str:
    """
    Generate a label for a circuit. If the circuit has subcircuits, the label is
    simply the name attribute of the circuit. If the circuit has no subcircuits, the
    label contains the circuit's name (specifying the gate) and the labels of the
    channels that are involved in the circuit.

    Parameters
    ----------
    circ : Circuit
        Circuit for which the label should be generated

    Returns
    -------
    str
        Label for the circuit
    """
    if len(circ.circuit) == 0:
        qb_list_str = ",".join(ch.label for ch in circ.channels)
        return f"{circ.name}({qb_list_str})"

    return circ.name


def convert_circuit_to_nx_graph(circ: Circuit) -> tuple[nx.DiGraph, list[str]]:
    """
    Construct a NetworkX directed graph (DiGraph) from a circuit. The nodes of the graph are all the
    subcircuits contained in the circuit object. The edges are directed from every
    circuit to its subcircuits.

    Parameters
    ----------
    circ : Circuit
        Circuit from which the graph should be constructed

    Returns
    -------
    nx.DiGraph: Directed graph representing the circuit
    list[str]: List of labels for the nodes in BFS traversal order
    """
    graph = nx.DiGraph()
    labels_nodes = []
    graph.add_node(circ.id)
    labels_nodes.append(get_label_for_circuit(circ))

    # This is the Breadth First Search (BFS) traversal of a tree:
    # https://en.wikipedia.org/wiki/Tree_traversal#Breadth-first_search
    queue = [circ]
    while len(queue) > 0:
        next_circuit = queue.pop(0)

        for c in next_circuit.circuit:  # Iterate over all subcircuits
            if len(c) > 0:  # Skip empty tuples
                graph.add_node(c[0].id)  # Add node to graph
                graph.add_edge(
                    next_circuit.id, c[0].id
                )  # Add edge from parent circuit to this subcircuit
                # Add label for the plot
                labels_nodes.append(get_label_for_circuit(c[0]))

                # If this subcircuit contains more subcircuits which have to be added to
                # the graph, add it to the queue so that it will be processed later in
                # the right order
                if len(c[0].circuit) > 0:
                    queue.append(c[0])

    return graph, labels_nodes
