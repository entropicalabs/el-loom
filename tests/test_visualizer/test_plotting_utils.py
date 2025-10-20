"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest

import networkx as nx
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np

from loom.eka import Circuit, Channel, ChannelType
from loom.visualizer.plotting_utils import (
    hex_to_rgb,
    rgb_to_hex,
    average_color_hex,
    change_color_brightness,
    get_font_color,
    point_in_polygon,
    interpolate_values,
    center_of_points,
    center_of_scatter_plot,
    get_angle_from_x_axis,
    order_points_counterclockwise,
    draw_half_circle,
    get_label_for_circuit,
    convert_circuit_to_nx_graph,
)


class TestPlottingUtils(unittest.TestCase):
    """Unit tests for the plotting utilities in the visualizer module."""

    def setUp(self):
        # Create example circuits
        self.circ1 = Circuit(
            "H", channels=[Channel(type=ChannelType.QUANTUM, label="Q1")]
        )
        self.circ2 = Circuit(
            "CNOT",
            channels=[
                Channel(type=ChannelType.QUANTUM, label="D1"),
                Channel(type=ChannelType.QUANTUM, label="D2"),
            ],
        )
        self.circ3 = Circuit(
            "syndrome_extraction",
            circuit=(
                (self.circ1),
                (self.circ2),
            ),
        )

    # pylint: disable=invalid-name
    def assertAlmostEqualVector(self, vec_a, vec_b):
        """Element-wise comparison of two vectors with 'AlmostEqual'."""
        self.assertEqual(len(vec_a), len(vec_b))
        for a, b in zip(vec_a, vec_b, strict=True):
            self.assertAlmostEqual(a, b)

    def test_hex_to_rgb(self):
        """Test the conversion from hex to RGB."""
        self.assertEqual([0, 0, 0], hex_to_rgb("#000000"))
        self.assertEqual([255, 255, 255], hex_to_rgb("#ffffff"))
        self.assertEqual([242, 160, 76], hex_to_rgb("#f2a04c"))

    def test_rgb_to_hex(self):
        """Test the conversion from RGB to hex."""
        self.assertEqual("#000000", rgb_to_hex([0, 0, 0]))
        self.assertEqual("#ffffff", rgb_to_hex([255, 255, 255]))
        self.assertEqual("#f2a04c", rgb_to_hex([242, 160, 76]))

    def test_average_color_hex(self):
        """Test the average color calculation in hex format."""
        self.assertEqual("#000000", average_color_hex(["#000000", "#000000"]))
        self.assertEqual("#7f7f7f", average_color_hex(["#000000", "#ffffff"]))
        self.assertEqual(
            "#200003", average_color_hex(["#000000", "#600000", "#000009"])
        )

    def test_change_color_brightness(self):
        """Test the change of color brightness."""
        self.assertEqual("#34f2b5", change_color_brightness("#34f2b5", 1))
        self.assertEqual("#000000", change_color_brightness("#34f2b5", 0))
        self.assertEqual("#ffffff", change_color_brightness("#34f2b5", 255))
        self.assertEqual("#404040", change_color_brightness("#808080", 0.5))
        self.assertEqual("#040000", change_color_brightness("#0c0000", 0.3333333))
        self.assertEqual("#040000", change_color_brightness("#0c0000", 0.34))

    def test_get_font_color(self):
        """Test the font color selection based on background color."""
        self.assertEqual("#000000", get_font_color("#ffffff"))
        self.assertEqual("#ffffff", get_font_color("#000000"))
        self.assertEqual("#000000", get_font_color("#ABCDEF"))
        self.assertEqual("#222222", get_font_color("#ABABAB"))

    def test_point_in_polygon(self):
        """Test the function to check if a point is inside a polygon."""
        e = 1e-5
        polygon = [(0, 0), (0, 1), (1, 1), (1, 0)]
        self.assertEqual(True, point_in_polygon(0.5, 0.5, polygon))
        self.assertEqual(False, point_in_polygon(1.5, 0.5, polygon))
        self.assertEqual(True, point_in_polygon(0 + e, 0 + e, polygon))
        self.assertEqual(True, point_in_polygon(0 + e, 1 - e, polygon))
        self.assertEqual(True, point_in_polygon(1 - e, 1 - e, polygon))
        self.assertEqual(True, point_in_polygon(1 - e, 0 + e, polygon))

        polygon = [(0, 0), (0, 1), (1, 1), (0.3, 0.7)]
        self.assertEqual(False, point_in_polygon(0.4, 0.6, polygon))
        self.assertEqual(True, point_in_polygon(0.2, 0.8, polygon))

    def test_interpolation_list_ints(self):
        """
        Test the interpolation function for a list of values at a point.
        """
        interpolation_points = [
            (0, 0),
            (0, 1),
            (1, 1),
            (1, 0),
        ]
        interpolation_values = [
            [100, -100, 10],
            [0, -50, 20],
            [0, 100, 30],
            [100, 50, 40],
        ]
        vals = interpolate_values(
            (0.5, 0.5), interpolation_points, interpolation_values
        ).tolist()
        expected_vals = [50, 0, 25]
        for val, expected_val in zip(vals, expected_vals, strict=True):
            self.assertAlmostEqual(expected_val, val)

    def test_interpolation_ints(self):
        """Test the interpolation function for a single value at a point."""
        interpolation_points = [
            (0, 0),
            (0, 1),
            (1, 1),
            (1, 0),
        ]
        interpolation_values = [
            -10,
            -10,
            20,
            20,
        ]
        vals = interpolate_values(
            (0.5, 0.5), interpolation_points, interpolation_values
        ).tolist()
        expected_val = 5
        self.assertEqual(len(vals), 1)
        self.assertAlmostEqual(expected_val, vals[0])

    def test_interpolation_exceptions(self):
        """Test the exceptions raised by the interpolation function."""
        # Different number of points and values
        with self.assertRaises(ValueError):
            interpolate_values((0.5, 0.5), [(0, 0), (1, 1)], [1])

        # Different number of points and values
        with self.assertRaises(ValueError):
            interpolate_values((0.5, 0.5), [(0, 0)], [1, 2])

        # Invalid interpolation point (not 2 coordinates)
        with self.assertRaises(ValueError):
            interpolate_values((0.5, 0.5), [(0, 0), (1)], [1, 2])

        # Invalid interpolation values (not all elements have the same dimension)
        with self.assertRaises(ValueError):
            interpolate_values((0.5, 0.5), [(0, 0), (1, 1)], [[1], [1, 2]])

    def test_center_of_points(self):
        """Test the function to calculate the center of a list of points."""
        self.assertAlmostEqualVector((0, 0), center_of_points([(0, 0), (0, 0)]))
        self.assertAlmostEqualVector((0.5, 0.5), center_of_points([(0, 0), (1, 1)]))
        self.assertAlmostEqualVector((0.5, 0.5), center_of_points([(0, 1), (1, 0)]))
        self.assertAlmostEqualVector(
            (-0.5, 100), center_of_points([(0, 100), (-1, 200), (0, -300), (-1, 400)])
        )

    def test_center_of_scatter_plot(self):
        """Test the function to calculate the center of a scatter plot."""
        scatter_plot = go.Scatter(
            x=[1, 2, 3],
            y=[-2.2, -4.2, -6.2],
        )

        center = center_of_scatter_plot(scatter_plot)
        self.assertAlmostEqualVector([2, -4.2], center)

    def test_get_angle(self):
        """Test the function to calculate the angle from the x-axis."""
        self.assertAlmostEqual(0, get_angle_from_x_axis([0, 0], [1, 0]))
        self.assertAlmostEqual(np.pi / 2, get_angle_from_x_axis([0, 0], [0, 1]))
        self.assertAlmostEqual(-np.pi / 4, get_angle_from_x_axis([0, 0], [1, -1]))

    def test_order_points_counterclockwise(self):
        """Test the function that orders points counterclockwise."""
        points = [[0, 0], [0, 1], [1, 1], [1, 0, ["additional_metadata"]]]
        ordered_points = order_points_counterclockwise(points)
        expected_points = [[0, 0], [1, 0, ["additional_metadata"]], [1, 1], [0, 1]]
        for ordered, expected in zip(ordered_points, expected_points, strict=True):
            self.assertEqual(expected[0], ordered[0])
            self.assertEqual(expected[1], ordered[1])
            if len(ordered[2]) > 1:
                self.assertAlmostEqualVector(expected[2], ordered[2][:-1])

    def test_draw_half_circle(self):
        """Test the function that draws a half-circle scatter plot."""
        # Generate the scatter plot, add it to a figure, and check that no error is produced
        scatter_plot = draw_half_circle([0, 0], 1)
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(scatter_plot)
        del fig

    def test_get_label_for_circuit(self):
        """Test the function that generates a label for a circuit."""
        self.assertEqual("h(Q1)", get_label_for_circuit(self.circ1))
        self.assertEqual("cnot(D1,D2)", get_label_for_circuit(self.circ2))
        self.assertEqual("syndrome_extraction", get_label_for_circuit(self.circ3))

    def test_convert_circuit_to_nx_graph(self):
        """Test the conversion of a circuit to a NetworkX DiGraph."""
        graph, labels_nodes = convert_circuit_to_nx_graph(self.circ3)
        adj_matrix = nx.to_numpy_array(graph, dtype=int).tolist()
        self.assertEqual([[0, 1, 1], [0, 0, 0], [0, 0, 0]], adj_matrix)
        # Node labels
        self.assertEqual(["syndrome_extraction", "h(Q1)", "cnot(D1,D2)"], labels_nodes)


if __name__ == "__main__":
    unittest.main()
