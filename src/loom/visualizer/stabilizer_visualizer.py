"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from loom.eka import Stabilizer, Lattice, Block, PauliOperator

from .plotting_utils import (
    order_points_counterclockwise,
    hex_to_rgb,
    draw_half_circle,
    center_of_scatter_plot,
    get_font_color,
    average_color_hex,
)


@dataclass()
class StabilizerPlot:  # pylint: disable=too-many-instance-attributes
    """
    Class for plotting stabilizers, pauli strings, and individual qubits of a Eka.

    Parameters
    ----------
    lattice : Lattice
        Lattice for which the plot should be created.
    stabilizers : list[Stabilizer] | None
        List of stabilizers which should be plotted
    width : int | None
        Width of the plotly figure. Default is 800.
    height : int | None
        Height of the plotly figure. Default is 600.
    rescale_coords : bool | None
        In case the image is drawn with a canvas for stabilizers containing different
        pauli operators, this parameter has to be set to True. Default is False. When
        True, the mixed-type stabilizers are drawn pixel by pixel for a canvas of the
        given width and height. The coordinates of the data qubits are rescaled to fit
        the image.
    padding_relative : float | None
        In case the image is drawn with a canvas for stabilizers containing different
        pauli operators, this parameter specifies the padding around the drawings,
        relative to the width of the image.
    xmin : float | None
        Minimum x-coordinate of the data qubit locations. Only needed if
        `rescale_coords` is True. Since objects might be added to the figure later,
        the x and y ranges have to be defined already at the beginning.
    xmax : float | None
        Maximum x-coordinate of the data qubit locations. Only needed if
        `rescale_coords` is True. Since objects might be added to the figure later,
        the x and y ranges have to be defined already at the beginning.
    ymin : float | None
        Minimum y-coordinate of the data qubit locations. Only needed if
        `rescale_coords` is True. Since objects might be added to the figure later,
        the x and y ranges have to be defined already at the beginning.
    ymax : float | None
        Maximum y-coordinate of the data qubit locations. Only needed if
        `rescale_coords` is True. Since objects might be added to the figure later,
        the x and y ranges have to be defined already at the beginning.
    title : str | None
        Title of the figure.
    opacity_stabs : float | None
        Opacity for the stabilizer colors. Range: 0 to 1 with 0 being
        completely transparent and 1 being completely opaque.
    show_grid : bool | None
        If set to False (default), the x and y axes are not shown, the zero line is
        hidden, and the grid is not displayed.
    fill_colors : dict[str, str] | None
        Dict for the fill colors of X, Y, and Z stabilizers.
    line_colors : dict[str, str] | None
        Dict for the line colors of X, Y, and Z stabilizers.
    dqb_plot_indices : bool | None
        If set to True (default), the data qubit indices are plotted inside the markers.
    dqb_marker_size : int | list[int] | None
        Marker size for the data qubits. Either a single integer is provided which is
        used for all data qubits, or a list of integers is provided which specifies the
        marker size for each data qubit individually.
    dqb_marker_color : str | list[str] | None
        Color for the data qubit markers. Either a single color is provided which is
        used for all data qubits, or a list of colors is provided which specifies the
        marker color for each data qubit individually.
    """

    lattice: Lattice
    stabilizers: list[Stabilizer] | None = Field(default=None, validate_default=True)
    width: int | None = 800
    height: int | None = 600
    rescale_coords: bool | None = False
    padding_relative: float | None = 0.05
    xmin: float | None = None
    xmax: float | None = None
    ymin: float | None = None
    ymax: float | None = None
    title: str | None = None
    opacity_stabs: float | None = 0.8
    show_grid: bool | None = False
    fill_colors: dict[str, str] | None = Field(
        default_factory=lambda: {
            "X": "#f2a04c",
            "Y": "#e95cb5",
            "Z": "#49bbc2",
        }
    )
    line_colors: dict[str, str] | None = Field(
        default_factory=lambda: {
            "X": "#7f6668",
            "Y": "#7f6668",
            "Z": "#7f6668",
        }
    )
    dqb_plot_indices: bool | None = True
    dqb_marker_size: int | list[int] | None = 25
    dqb_marker_color: str | list[str] | None = "#ec6f6d"

    def __post_init__(self) -> None:
        if self.rescale_coords:
            if None in [self.xmin, self.xmax, self.ymin, self.ymax]:
                raise ValueError(
                    "If `rescale_coords` is set to True, `xmin`, `xmax`, `ymin`, and "
                    "`ymax` must be provided."
                )

        # Create the figure object
        self._fig = make_subplots(rows=1, cols=1)
        self.canvas = np.ones((self.height, self.width, 3), dtype=np.uint8)
        self.canvas[:, :, 0] = 229
        self.canvas[:, :, 1] = 236
        self.canvas[:, :, 2] = 246

        # Calculate coordinates of data qubits and store them both in a list and a dict
        self.dqubit_coords = [self.get_qb_coord(qb) for qb in self.lattice.all_qubits()]
        self.dqubit_coordinates_map = {
            qb: self.get_qb_coord(qb) for qb in self.lattice.all_qubits()
        }

        # Make sure dqb_marker_color is a list of colors
        if isinstance(self.dqb_marker_color, str):
            self.dqb_marker_color = [self.dqb_marker_color] * len(self.dqubit_coords)
        else:
            if len(self.dqb_marker_color) != len(self.dqubit_coords):
                raise ValueError(
                    "The length of the dqb_marker_color list should be equal "
                    "to the number of data qubits. The list has a "
                    f"length of {len(self.dqb_marker_color)} but there are "
                    f"{len(self.dqubit_coords)} data qubits."
                )

        # Color of data qubit markers in the legend
        self.legend_dqb_marker_color = "#000000"

        # Dummy plot of data qubits such that they appear first in the legend
        self._fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name="Data qubits",
                marker={
                    "size": self.dqb_marker_size,
                    "color": self.dqb_marker_color[0],
                    "line": {
                        "width": 1,
                        "color": "#9f4b49",
                    },
                },
                legendgroup="data_qubits",
                showlegend=True,
            )
        )

        # Settings
        self._fig.update_layout(
            width=self.width,
            height=self.height,
            xaxis_title="X-axis",
            yaxis_title="Y-axis",
            showlegend=True,
            xaxis={
                "showgrid": self.show_grid,
                "zeroline": False,
                "visible": self.show_grid,
            },
            yaxis={
                "showgrid": self.show_grid,
                "visible": self.show_grid,
                "scaleanchor": "x",
                "scaleratio": 1,
                "autorange": "reversed",  # Otherwise y axis would be reversed
            },
            hoverdistance=1,  # With this setting, the hover text will only show up when
            # exactly hovering above an object and not when one is close to it. This is
            # important for overlapping objects where the wrong hover text might be
            # shown.
        )
        if self.title is not None:
            self._fig.update_layout(title=self.title)

    def _scale_coords(self, x: float, y: float) -> tuple[float, float]:
        """
        If `self.rescale_coords == True`, any coordinates are rescaled to fit the
        canvas dimensions. This is used when drawing stabilizers with different pauli
        operators with a color gradient pixel by pixel.

        NOTE: In most cases, `self.rescale_coords == False` and this functions does
        nothing.

        Parameters
        ----------
        x : float
            x coordinate of the pixel to be rescaled
        y : float
            y coordinate of the pixel to be rescaled

        Returns
        -------
        tuple[float, float]
            x and y coordinate of the rescaled point
        """
        if self.rescale_coords:
            # For a good visualization, we do not want to plot objects exactly until the
            # border of the canvas but keep some distance (padding) to the border.
            # By default, this padding is 5% of the width of the canvas.
            padding_absolute = self.padding_relative * self.width
            # New x coordinate
            # Previously values were in the range [xmin, xmax] and now we want to scale
            # them to the range [padding_absolute, width - padding_absolute].
            # This corresponds to scaling by a factor of (width - 2 * padding_absolute) /
            # (xmax - xmin) and then shifting by padding_absolute.
            x_new = padding_absolute + (x - self.xmin) * (
                self.width - 2 * padding_absolute
            ) / (self.xmax - self.xmin)
            # New y coordinate
            # Calculation is analogous as for x
            y_new = padding_absolute + (y - self.ymin) * (
                self.height - 2 * padding_absolute
            ) / (self.ymax - self.ymin)
            return x_new, y_new

        return x, y

    def show(self, **kwargs) -> None:
        """Show the plotly figure."""
        self._fig.show(**kwargs)

    def get_qb_coord(self, qb: tuple[int, ...]) -> tuple[float, ...]:
        """
        Get the coordinates of a data qubit. The coordinates of data qubit (x, y, a)
        are calculated by the formula x * l[0] + y * l[1] + a * b[a], where l is the
        list of lattice vectors and b is the list of basis vectors. We put all these
        terms into a list and sum them up in the very end.
        """
        list_vectors = [
            np.array(self.lattice.lattice_vectors[i]) * qb[i]
            for i in range(len(self.lattice.lattice_vectors))
        ]
        # If there are more entries in the qubit index than lattice vectors, the last
        # element in the tuple denotes which basis vector in the unit cell the qubit
        # belongs to
        if len(qb) > len(self.lattice.lattice_vectors):
            list_vectors.append(self.lattice.basis_vectors[qb[-1]])
        else:
            # If there is no such entry, the qubit is assumed to belong to the first
            # basis vector in the unit cell
            list_vectors.append(self.lattice.basis_vectors[0])
        return np.sum(np.array(list_vectors), axis=0)

    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    def get_stabilizer_traces(
        self,
        stabilizers: list[Stabilizer],
        fill_colors: dict[str, str] | None = None,
        fill: str = "toself",
        opacity_stabs: float | None = None,
        showlegend: bool = True,
        legendgroup: str | None = None,
    ) -> list[go.Scatter]:
        """
        Generate the plotly traces for the stabilizers.

        Parameters
        ----------
        stabilizers : list[Stabilizer]
            Stabilizers which should be plotted.
        fill_colors : dict[str, str] | None
            Colors for the fill of the stabilizers. If None is provided, the default
            colors from the class are taken.
        fill : str
            Defines how the area of the stabilizers is filled. Default is 'toself' which
            fills the area of the polygon. Another option is 'none' (no fill) for having
            only the border line but no filling.
        opacity_stabs : float | None
            Opacity for the stabilizer colors. If If None is provided, the default
            opacity from the class is taken.
        showlegend : bool, optional
            If True (the default), the stabilizers are shown in the legend.
        legendgroup: str | None
            Legend group to which the stabilizers belong.

        Returns
        -------
        list[go.Scatter]
            List of scatter plot traces for the stabilizers.
        """
        stab_traces = []
        for stab in stabilizers:
            polygon_corners = [
                self.dqubit_coordinates_map[qb] for qb in stab.data_qubits
            ]

            # Order the corners of the polygon such that plotting the polygon looks nice
            ordered_corners = order_points_counterclockwise(polygon_corners)
            ordered_corners_scaled = []
            for corner in ordered_corners:
                corner_x_scaled, corner_y_scaled = self._scale_coords(
                    corner[0], corner[1]
                )
                ordered_corners_scaled.append([corner_x_scaled, corner_y_scaled])

            stab_name = ""
            for pauli, data_qubit in zip(stab.pauli, stab.data_qubits, strict=True):
                stab_name += pauli + "<sub>" + str(data_qubit) + "</sub>"

            # Line color
            line_color = average_color_hex(
                [self.line_colors[pauli] for pauli in stab.pauli]
            )

            # Start with self.fill_colors and update it with values provided in the
            # fill_colors argument of this function
            stab_fill_colors = self.fill_colors
            if fill_colors is not None:
                stab_fill_colors.update(fill_colors)

            if stab.pauli in stab_fill_colors:
                fill_color = stab_fill_colors[stab.pauli]
            else:
                fill_color = average_color_hex(
                    [stab_fill_colors[pauli] for pauli in stab.pauli]
                )

            if opacity_stabs is None:
                opacity_stabs = self.opacity_stabs
            background_color_arr = hex_to_rgb(fill_color)
            background_color = "rgba("
            for color in background_color_arr:
                background_color += str(color) + ","
            background_color = background_color + str(opacity_stabs) + ")"

            if (
                len(stab.data_qubits) == 2
            ):  # Weight-2 stabilizer --> Plot stabilizer as half circle
                (x1, y1) = self.dqubit_coordinates_map[stab.data_qubits[0]]
                (x2, y2) = self.dqubit_coordinates_map[stab.data_qubits[1]]
                x1_scaled, y1_scaled = self._scale_coords(x1, y1)
                x2_scaled, y2_scaled = self._scale_coords(x2, y2)
                dist = np.linalg.norm([x2_scaled - x1_scaled, y2_scaled - y1_scaled])
                direction = np.arctan2(y2 - y1, x2 - x1)

                # Generate two half circles for the two possible directions in which
                # it could point. Then check which of the two is better.
                # For codes with dimension > 1, the half circles will face away
                # from the bulk of the code block.
                # In the special case of repetition codes, the half circles will
                # be turned into different directions alternatingly.
                xscaled, yscaled = self._scale_coords((x1 + x2) / 2, (y1 + y2) / 2)
                scatter1 = draw_half_circle(
                    [xscaled, yscaled],
                    dist / 2,
                    direction,
                    name=stab_name,
                    fillcolor=background_color,
                    line={"color": line_color},
                    text=stab_name,
                    showlegend=showlegend,
                    legendgroup=legendgroup,
                )
                scatter2 = draw_half_circle(
                    [xscaled, yscaled],
                    dist / 2,
                    direction + np.pi,
                    name=stab_name,
                    fillcolor=background_color,
                    line={"color": line_color},
                    text=stab_name,
                    showlegend=showlegend,
                    legendgroup=legendgroup,
                )
                center = [q for stab in stabilizers for q in stab.data_qubits]
                center_of_block = np.average(np.array(center), axis=0)
                scaled_center = self._scale_coords(
                    center_of_block[0], center_of_block[1]
                )
                center1 = center_of_scatter_plot(scatter1)
                center2 = center_of_scatter_plot(scatter2)
                dist1 = np.linalg.norm(
                    [scaled_center[0] - center1[0], scaled_center[1] - center1[1]]
                )
                dist2 = np.linalg.norm(
                    [scaled_center[0] - center2[0], scaled_center[1] - center2[1]]
                )
                if dist1 < dist2:
                    stab_traces.append(scatter2)
                else:
                    stab_traces.append(scatter1)
            else:  # Weight > 2 --> Plot stabilizer as polygon
                ordered_corners_scaled.append(ordered_corners_scaled[0])
                ordered_corners_scaled_x = [
                    coord[0] for coord in ordered_corners_scaled
                ]
                ordered_corners_scaled_y = [
                    coord[1] for coord in ordered_corners_scaled
                ]

                stab_traces.append(
                    go.Scatter(
                        x=ordered_corners_scaled_x,
                        y=ordered_corners_scaled_y,
                        mode="lines",
                        name=stab_name,
                        fill=fill,  # Fill the area of the polygon
                        fillcolor=background_color,
                        line={"color": line_color},
                        text=stab_name,
                        hoverinfo="text",
                        showlegend=showlegend,
                        legendgroup=legendgroup,
                    )
                )

        return stab_traces

    def add_stabilizers(
        self,
        stabilizers: list[Stabilizer],
        **kwargs,
    ) -> None:
        """
        Adds stabilizers to the plot.

        Parameters
        ----------
        stabilizers : list[Stabilizer]
            Stabilizers which should be plotted.
        """
        stabilizers_traces = self.get_stabilizer_traces(stabilizers, **kwargs)

        for stab in stabilizers_traces:
            self._fig.add_trace(stab)

    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals, too-many-branches
    def get_dqubit_traces(
        self,
        dqb_show: list[bool] | None = None,
        labels: list[str] | None = None,
        marker_mode: str | None = None,
        marker_style: dict | None = None,
        legendgroup: str | None = None,
        marker_opacity: float | None = 1,
    ) -> list[go.Scatter]:
        """
        Generate the plotly traces for the data qubit markers.

        Parameters
        ----------
        dqb_show : list[bool] | None
            List of booleans, indicating whether the i-th data qubit should be
            shown or not. If None, all data qubits are shown.
        labels : list[str] | None
            Labels which should be displayed inside the data qubit markers or on hover.
        marker_mode : str | None
            Mode in which the markers are displayed, e.g. `markers` or `markers+text`
        marker_style : dict | None
            Define the style of the markers such as size, color, shape, etc.
        legendgroup : str | None
            Legend group to which the data qubit markers belong.
        marker_opacity : float | None
            Opacity of the data qubit markers.

        Returns
        -------
        list[go.Scatter]
            List of scatter plot traces for the data qubit markers.
        """
        # Check inputs and transform them
        if dqb_show is None:
            dqb_show = [True] * len(self.dqubit_coordinates_map)  # Plot all data qubits
        else:
            if len(dqb_show) != len(self.dqubit_coordinates_map):
                raise ValueError(
                    "The length of the dqb_show list should be equal "
                    "to the number of data qubits. The list has a "
                    f"length of {len(dqb_show)} but there are "
                    f"{len(self.dqubit_coordinates_map)} data qubits."
                )

        # Legend group
        if legendgroup is None:
            legendgroup = "data_qubits"

        # Marker type
        if marker_mode is None:
            if self.dqb_plot_indices:
                marker_mode = "markers+text"
            else:
                marker_mode = "markers"

        dqubit_traces = []
        # Iterate over all data qubits and (if dqb_show[i] is True for the data qubit
        # with index i) create a trace for this data qubit marker.
        # We create separate traces for the data qubits to be able to change the style
        # and text of each data qubit individually.
        for i, dqubit_coord in enumerate(self.dqubit_coords):
            # Check if this data qubit should be plotted or not
            if dqb_show[i] is not True:
                continue
            # Label
            if labels is not None:
                text = labels[i]
            else:
                converted_coords = tuple(
                    (
                        coord.item()
                        if isinstance(coord, (np.integer, np.floating))
                        else coord
                    )
                    for coord in list(self.dqubit_coordinates_map.keys())[i]
                )
                if self.dqb_plot_indices:
                    text = str(converted_coords)
                else:
                    text = f"Data qubit {converted_coords}"

            # Marker style
            marker_color_hex = self.dqb_marker_color[i]
            marker_color = "rgba("
            for c in hex_to_rgb(marker_color_hex):
                marker_color += str(c) + ","
            marker_color += str(round(marker_opacity, 3)) + ")"

            if marker_style is None:
                dqb_marker_style = {
                    "size": self.dqb_marker_size,
                    "color": marker_color,
                    "line": {
                        "width": 1,
                        "color": "#9f4b49",
                    },
                }
                textfont_color = get_font_color(marker_color_hex)
            else:
                dqb_marker_style = marker_style
                if "color" not in dqb_marker_style:
                    raise ValueError(
                        "The marker style dict must contain the color of the markers."
                    )
                if len(dqb_marker_style["color"]) == 7:  # Already HEX format
                    textfont_color = get_font_color(dqb_marker_style["color"])
                else:
                    raise ValueError("Only HEX values supported at the moment.")

            xnew, ynew = self._scale_coords(dqubit_coord[0], dqubit_coord[1])
            dqubit_traces.append(
                go.Scatter(
                    x=[xnew],
                    y=[ynew],
                    mode=marker_mode,
                    marker=dqb_marker_style,
                    text=text,
                    textfont_color=textfont_color,
                    textposition="middle center",
                    hoverinfo="text",
                    legendgroup=legendgroup,
                    showlegend=False,
                )
            )
        return dqubit_traces

    def add_dqubit_traces(
        self,
        dqb_show: list[bool] | None = None,
        **kwargs,
    ) -> None:
        """
        Add data qubit markers to the figure.

        Parameters
        ----------
        dqb_show : list[bool] | None
            List of booleans, indicating whether the i-th data qubit should be
            shown or not. If None, all data qubits are shown.
        """
        dqubit_traces = self.get_dqubit_traces(dqb_show, **kwargs)

        for dqb_trace in dqubit_traces:
            self._fig.add_trace(dqb_trace)

    def get_pauli_string_traces(
        self,
        operation: PauliOperator,
        dqb_marker_size: int | None = 25,
        connecting_line: bool | None = True,
        log_op_fill_colors: dict[str, str] | None = None,
        log_op_line_colors: dict[str, str] | None = None,
        legend_name: str | None = "",
        legendgroup: str = "data_qubits_log_op",
        shift: tuple[float, float] = (0, 0),
    ) -> list[go.Scatter]:
        """
        Generate the plotly traces for visualizing a Pauli string.

        Parameters
        ----------
        operation : PauliOperator
            Pauli string which should be plotted
        dqb_marker_size : int | None
            Marker size for the data qubits contained in the Pauli string
        connecting_line : bool | None
            If True, the data qubits in the Pauli string will be connected by a line
        log_op_fill_colors: dict[str, str] | None
            Dict containing the colors for the X, Y, and Z markers of the logical
            operators. If None is provided, the same colors are taken as for the
            stabilizers.
        log_op_line_colors : dict[str, str] | None
            Dict containing the border colors for the X, Y, and Z markers of the logical
            operators. If None is provided, the same colors are taken as for the
            stabilizers.
        legend_name : str | None
            Name of this Pauli string which should appear in the legend.
        legendgroup : str, optional
            Legend group to which the markers belong
        shift : tuple[float, float]
            2D vector by which the data qubit markers should be shifted, relative to the
            original position of the data qubits.

        Returns
        -------
        list[go.Scatter]
            List of scatter plot traces for the data qubit markers contained in the
            pauli string as well as (if specified) a connecting line between them.
        """
        if log_op_fill_colors is None:
            log_op_fill_colors = self.fill_colors
        if log_op_line_colors is None:
            log_op_line_colors = self.line_colors

        pauli_dqubit_coords = np.array(
            [
                np.array(self.dqubit_coordinates_map[qb]) + shift
                for qb in operation.data_qubits
            ]
        )  # List of coordinates of the data qubits that are involved in the Pauli string

        pauli_string_traces = []

        # Connecting line
        if connecting_line is True:
            connecting_line_color = average_color_hex(
                [log_op_fill_colors[pauli] for pauli in operation.pauli]
            )
            pauli_string_traces.append(
                go.Scatter(
                    x=pauli_dqubit_coords[:, 0],
                    y=pauli_dqubit_coords[:, 1],
                    mode="lines",
                    hoverinfo="text",
                    line={"color": connecting_line_color, "width": 15},
                    showlegend=False,
                    legendgroup=legendgroup,
                )
            )

        # Legend name for the pauli operator
        if legend_name == "":
            for i, qb in enumerate(operation.data_qubits):
                legend_name += operation.pauli[i] + "<sub>" + str(qb) + "</sub>"

        # Create traces for the data qubit markers containing the pauli operator
        for i, data_qubit in enumerate(operation.data_qubits):
            label_pauli_with_number = f"{operation.pauli[i]}{data_qubit}"

            pauli_string_traces.append(
                go.Scatter(
                    x=[pauli_dqubit_coords[i, 0]],
                    y=[pauli_dqubit_coords[i, 1]],
                    mode="markers+text",
                    name=legend_name,
                    marker={
                        "size": dqb_marker_size * 1.25,
                        "color": log_op_fill_colors[operation.pauli[i]],
                        "line": {
                            "width": 1,
                            "color": log_op_line_colors[operation.pauli[i]],
                        },
                        "symbol": "square",
                    },
                    text=f"{operation.pauli[i]}",
                    textposition="middle center",
                    hoverinfo="text",
                    hovertemplate=label_pauli_with_number,
                    legendgroup=legendgroup,
                    showlegend=False,
                )
            )

        return pauli_string_traces

    def plot_pauli_string(
        self,
        operation: PauliOperator,
        **kwargs,
    ) -> None:
        """
        Add a Pauli string plotting to the figure.

        Parameters
        ----------
        operation : PauliOperator
            Pauli string which should be plotted
        """
        for trace in self.get_pauli_string_traces(operation, **kwargs):
            self._fig.add_trace(trace)

    def plot_pauli_charges(
        self,
        block: Block,
        legendgroup: str | None = None,
    ) -> None:
        """
        Plot the Pauli charges of the data qubits of the given block.

        Parameters
        ----------
        block: Block
            Block for which the pauli charges should be plotted
        legendgroup : str | None
            Legend group to which the markers belong
        """
        # Construct a PauliOperator containing the charge of each data qubit respectively
        op = PauliOperator(
            pauli="".join(
                [charge for charge in block.pauli_charges.values() if charge != "_"]
            ),
            data_qubits=[
                qb for qb, charge in block.pauli_charges.items() if charge != "_"
            ],
        )

        pauli_traces = self.get_pauli_string_traces(
            op,
            connecting_line=False,
            legend_name="Pauli charges",
            legendgroup=legendgroup,
        )

        for tr in pauli_traces:
            self._fig.add_trace(tr)

    def plot_blocks(
        self,
        blocks: Block | list[Block],
        plot_logical_operators: bool = True,
        plot_pauli_charges: bool = False,
        log_ops_shift: float = 0.03,
    ) -> None:
        """
        Plot all the stabilizers of a `Block` or multiple `Block` objects.

        Parameters
        ----------
        blocks : Block | list[Block]
            A block or a list of blocks to be plotted
        plot_logical_operators : bool
            Whether the logical operators should be plotted, defaults to True
        plot_pauli_charges : bool
            Whether the pauli charges should be plotted, defaults to False
        log_ops_shift: float = 0.03
            Amount by which the logical X operator is shifted to the bottom left and the
            logical Z operator to the top right. This is done such that one can see both
            boxes at the qubit(s) on which both operators act. Defaults to 0.03.
        """
        if isinstance(blocks, Block):
            blocks = [blocks]

        for block in blocks:
            legendgroup = f"block_{block.unique_label}"

            # Dummy plot for the block to have a nice label in the legend
            self._fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    name=block.unique_label,
                    marker={
                        "size": self.dqb_marker_size,
                        "color": "#555555",
                        "line": {
                            "width": 1,
                            "color": "#333333",
                        },
                    },
                    legendgroup=legendgroup,
                    showlegend=True,
                )
            )

            # Stabilizers
            self.add_stabilizers(
                block.stabilizers, legendgroup=legendgroup, showlegend=False
            )

            # Pauli charges
            if plot_pauli_charges:
                self.plot_pauli_charges(block, legendgroup=legendgroup)

            # Logical X and Z operators
            if plot_logical_operators:
                for log_x in block.logical_x_operators:
                    self.plot_pauli_string(
                        log_x,
                        shift=(-log_ops_shift, -log_ops_shift),
                        legendgroup=legendgroup,
                    )
                for log_z in block.logical_z_operators:
                    self.plot_pauli_string(
                        log_z,
                        shift=(log_ops_shift, log_ops_shift),
                        legendgroup=legendgroup,
                    )
