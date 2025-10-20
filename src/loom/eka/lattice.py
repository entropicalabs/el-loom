"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from enum import Enum
import itertools

from pydantic.dataclasses import dataclass
from pydantic import field_validator, ValidationInfo
import numpy as np

from .utilities import (
    retrieve_field,
    dataclass_params,
)


class LatticeType(str, Enum):
    """Defines the types of lattices available."""

    LINEAR = "linear"
    TRIANGLE_2D = "triangle_2d"
    SQUARE_2D = "square_2d"
    SQUARE_2D_0ANC = "square_2d_0anc"
    SQUARE_2D_2ANC = "square_2d_2anc"
    CAIRO_PENT_2D = "cairo_pent_2d"
    HEX_2D = "hex_2d"
    OCT_2D = "oct_2d"
    POLY_2D = "poly_2D"
    CUBE_3D = "cube_3d"
    CUSTOM = "custom"


@dataclass(**dataclass_params)
class Lattice:
    """
    A lattice object contains information about the lattice structure. The lattice
    defines the indexing system of qubits, since qubits are indexed by their coordinates
    in terms of the lattice and basis vectors.

    Note that this lattice on which stabilizers, logical qubit blocks, and lattice
    surgery operations are defined, is a priori only an abstract lattice which makes
    defining all these things and handling qubit indices convenient. It does not have to
    be the actual lattice of the physical hardware.

    Qubit indices have the form `(x_0, ..., x_{n-1}, b)` where `x_0, ..., x_{n-1}`
    denote the unit cell with `n` the dimension of the lattice and `b` denotes which
    qubit in the unit cell it is.

    E.g. for a 2D lattice with a 2-qubit unit cell, the qubit indices are `(x, y, b)`
    which means that it is the `x`-th unit cell along the first lattice vector, the
    `y`-th unit cell along the second lattice vector, and the `b`-th qubit in the unit
    cell (b is either 0 or 1 in the case of a 2-qubit unit cell).

    Parameters
    ----------
    basis_vectors : tuple[tuple[float, ...], ...]
        Tuple of basis vectors that define the unit cell. If there is more than one
        qubit in the unit cell, one has to specify their position inside the unit cell
        using the basis vectors.
    lattice_vectors : tuple[tuple[float, ...], ...]
        Tuple of lattice vectors that define the lattice. The whole lattice is obtained
        by translating the unit cell along these lattice vectors by integer amounts.
    size : tuple[int, ...] | None
        The size of the lattice in each dimension. If set to None, the lattice is
        assumed to be infinitely large.
    lattice_type : LatticeType
        Type of the lattice. This is useful for storing the lattice as well as for some
        function such as `Block` creation which behave differently for different lattice
        types. Default is LatticeType.CUSTOM which means the lattice is created by the
        user.
    """

    basis_vectors: tuple[tuple[float, ...], ...]
    lattice_vectors: tuple[tuple[float, ...], ...]
    size: tuple[int, ...] | None = None
    lattice_type: LatticeType = LatticeType.CUSTOM

    @property
    def n_dimensions(self) -> int:
        """
        The dimension of the lattice, i.e. the number of lattice vectors.
        """
        return len(self.lattice_vectors)

    @property
    def unit_cell_size(self) -> int:
        """
        The size of the unit cell, i.e. the number of basis vectors.
        """
        return len(self.basis_vectors)

    # Validation
    @field_validator("basis_vectors", mode="before")
    @classmethod
    def format_basis_vectors(cls, basis_vectors: tuple):
        """
        Format the basis_vector input to also accept [] for 1-qubit unit cells.
        """
        if len(basis_vectors) == 0:
            return [[0, 0]]
        return basis_vectors

    @field_validator("basis_vectors", mode="after")
    @classmethod
    def basis_vectors_same_length(cls, basis_vectors: tuple):
        """
        Validate that all basis vectors have the same length.
        """
        if any(
            len(basis_vector) != len(basis_vectors[0]) for basis_vector in basis_vectors
        ):
            raise ValueError("All basis vectors must have the same length.")
        return basis_vectors

    @field_validator("lattice_vectors", mode="after")
    @classmethod
    def lattice_vectors_same_length(cls, lattice_vectors: tuple):
        """
        Validate that all lattice vectors have the same length.
        """
        if any(
            len(lattice_vector) != len(lattice_vectors[0])
            for lattice_vector in lattice_vectors
        ):
            raise ValueError("All lattice vectors must have the same length.")
        return lattice_vectors

    @field_validator("size", mode="after")
    @classmethod
    def size_right_dimension(cls, size: int | None, values: ValidationInfo):
        """
        Validate that `size` has the right dimension, i.e. the size of the tuple has
        to be equal to the number of lattice vectors.
        """
        # Allow infinite lattices where size is set to None
        if size is None:
            return size

        # If the lattice is not infinite, check that the size has the right dimension
        n_dim = len(retrieve_field("lattice_vectors", values))
        if len(size) != n_dim:
            raise ValueError(
                f"The given `size` is invalid. `size` has {len(size)} "
                f"elements, but the lattice has {n_dim} dimensions."
            )
        return size

    @field_validator("size", mode="after")
    @classmethod
    def size_not_negative(cls, size):
        """Validate that `size` is not negative."""
        if size is not None and any(x < 0 for x in size):
            raise ValueError("Size cannot be negative.")
        return size

    # Methods
    def all_unit_cells(
        self, size: tuple[int | None, ...] | None = None
    ) -> list[tuple[int, ...]]:
        """
        Get a list of all unit cells of the lattice. They are given in terms of the
        lattice vectors.

        Parameters
        ----------
        size : tuple[int  |  None, ...] | None, optional
            If only a part of the lattice is needed, the size of the wanted section can
            be specified. If None is provided, all unit cells of the lattice will be
            returned. If the lattice has infinite size, the `size` parameter must be
            provided.

        Returns
        -------
        list[tuple[int, ...]]
            Unit cells of the lattice for the given dimensions. They are given as a list
            of tuples where each tuple contains the coordinates of the unit cell in
            terms of the lattice vectors.
        """
        # Validation
        if size is None and self.size is None:
            raise ValueError("Please specify the `size` parameter.")
        if self.size is None and any(x is None for x in size):
            raise ValueError("Please specify all dimensions in the `size` parameter.")
        if size is not None and len(size) != self.n_dimensions:
            raise ValueError(
                f"The given `size` is invalid. `size` has {len(size)} "
                f"elements, but the lattice has {self.n_dimensions} dimensions."
            )

        # Combine the size of the lattice with the given size
        if size is None:
            adapted_size = self.size
        else:
            adapted_size = [
                size[i] if size[i] is not None else self.size[i]
                for i in range(self.n_dimensions)
            ]
        # Get all unit cells inside the specified region
        grids = np.meshgrid(*[range(x) for x in adapted_size], indexing="ij")
        return list(zip(*[map(int, grid.flatten()) for grid in grids], strict=True))

    def all_qubits(
        self,
        size: tuple[int | None, ...] | None = None,
        force_including_basis: bool = False,
    ) -> list[tuple[int, ...]]:
        """
        Get a list of qubits in the lattice inside the region specified by `size`.

        Parameters
        ----------
        size : tuple[int  |  None, ...] | None, optional
            If only a part of the lattice is needed, the size of the wanted section can
            be specified. If None is provided, all qubits of the lattice will be
            returned. If the lattice has infinite size, the `size` parameter must be
            provided.
        force_including_basis : bool, optional
            By default, the basis parameter is included in the qubit indices only if
            `size > 1`. To force it to appear for `size == 1`, set this boolean to
            `True`.

        Returns
        -------
        list[tuple[int, ...]]
            List of all qubits inside the specified region.
        """
        unit_cells = self.all_unit_cells(size)
        if self.unit_cell_size == 1 and force_including_basis is False:
            return unit_cells

        bases_in_unit_cell = range(self.unit_cell_size)
        all_combinations = [
            tuple_ + (int_,)
            for tuple_, int_ in itertools.product(unit_cells, bases_in_unit_cell)
        ]
        return all_combinations

    # Default lattices

    @staticmethod
    def _points_on_circle(
        n_points: int,
        radius: float | int,
        offset: float | int = -0.5 * np.pi,
        disp: tuple[int | float, int | float] = (0, 0),
    ) -> list[list[float]]:
        """Generate points on a circle of given radius and number of points. Default
        offset places first point at the top of the circle for polygons with odd number
        of points, and disp shifts the points.

        Parameters
        ----------
        radius : float | int
            Radius of the circle on which the points are placed.
        n_points : int
            Number of points to be placed on the circle.
        offset : float | int, optional
            Offset to be applied to the angle of the points on the circle. This is
            useful for placing the first point at a specific angle. The default values
            for this parameter are accurate to the plotting convention of
            StabilizerPlot(), which inverts the y-axis:
                top of circle = -0.5 * np.pi
                right of circle = 0
                bot of circle = +0.5 * np.pi
                etc.
        disp : tuple[int | float, int | float], optional
            Displacement of the points from the center of the circle. This is useful
            for placing the points at a specific position in the 2D plane.

        Returns
        -------
        list[list[float]]
            List of points on the circle, each point represented as a list of two
            floats [x, y] coordinates.
        """

        # Validation
        if not isinstance(n_points, int) or n_points < 1:
            raise ValueError(
                "Number of points must be an integer that is at least 1. "
                f"Received {n_points} of type {type(n_points)}."
            )
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise ValueError(
                "Radius must be a positive number (int or float). "
                f"Received {radius} of type {type(radius)}."
            )
        if not isinstance(offset, (int, float)):
            raise TypeError(
                "Offset must be a number (int or float). "
                f"Received {offset} of type {type(offset)}."
            )
        if (
            not isinstance(disp, tuple)
            or not all(isinstance(x, (int, float)) for x in disp)
            or len(disp) != 2
        ):
            raise TypeError(
                "Disp must be a tuple of two numbers (x, y). "
                f"Received {disp} of type {type(disp)}."
            )

        if n_points == 1:
            # If only one point is requested, return it at the specified displacement
            return [[disp[0], disp[1]]]

        return [
            [
                float(radius * np.cos(2 * np.pi * i / n_points + offset) + disp[0]),
                float(radius * np.sin(2 * np.pi * i / n_points + offset) + disp[1]),
            ]
            for i in range(n_points)
        ]

    @classmethod
    def linear(
        cls,
        lattice_size: tuple[int] | None = None,
    ):
        """Generates a linear lattice in 1D. The lattice is defined by a single lattice
        vector that is parallel to the x-axis. The unit cell is defined by set of two
        basis vectors. The unit cell contains 2 points (1 qubit, 1 ancilla) arranged on
        a straight line. The lattice vector tiles the plane in a linear pattern.

        Parameters
        ----------
        lattice_size : tuple[int] | None, optional
            The size of the lattice in terms of unit cells. If None, the lattice is
            assumed to be infinitely large.

        Returns
        -------
        Lattice
            A Lattice object representing the linear lattice.
        """
        lattice_type = LatticeType.LINEAR
        # Vectors are artificially expanded to 2D for plotting purposes
        basis_vectors = [[0, 0], [0.5, 0]]
        lattice_vectors = [[1, 0]]
        return cls(basis_vectors, lattice_vectors, lattice_size, lattice_type)

    @classmethod
    def triangle_2d(
        cls,
        lattice_size: tuple[int, int] | None = None,
    ):
        """Generates a triangular lattice in 2D. The lattice tiles the plane, and is
        defined by two lattice vectors that are 60 degrees to each other. The unit cell
        is defined by a set of three basis vectors. The unit cell contains 3 points
        (1 qubits, 2 ancilla) arranged in a downwards sloping line. The lattice vectors
        tile the plane in a triangular pattern.

        Basis vector 0 represents the data qubit. Basis vector 1 represents ancilla
        qubit for triangles with vertices pointing downwards, and basis vector 2
        represents ancilla qubit for triangles with vertices pointing upwards.

        Parameters
        ----------
        lattice_size : tuple[int, int] | None, optional
            The size of the lattice in terms of unit cells. If None, the lattice is
            assumed to be infinitely large.

        Returns
        -------
        Lattice
            A Lattice object representing the triangular lattice.
        """
        lattice_type = LatticeType.TRIANGLE_2D
        basis_vectors = [[0, 0], [0.5, 1 / (2 * np.sqrt(3))], [1, 1 / np.sqrt(3)]]
        lattice_vectors = [[1, 0], [0.5, 0.5 * np.sqrt(3)]]
        return cls(basis_vectors, lattice_vectors, lattice_size, lattice_type)

    @classmethod
    def square_2d_0anc(cls, lattice_size: tuple[int, int] | None = None):
        """Generates a square lattice in 2D. The lattice tiles the plane, and is
        defined by two lattice vectors that are orthogonal to each other. The unit
        cell is defined by one basis vector. The unit cell contains 1 point (1 qubit).
        The lattice vectors tile the plane in a square pattern.

        Basis vector 0 represents the data qubit.

        NOTE: This lattice was defined for the specific use of the HGP code class,
        because the product structure automatically allocates two dimensional
        coordinates to all qubits in the code block (both datas and checks)

        Parameters
        ----------
        lattice_size : tuple[int, int] | None, optional
            The size of the lattice in terms of unit cells. If None, the lattice is
            assumed to be infinitely large.

        Returns
        -------
        Lattice
            A Lattice object representing the square lattice.
        """
        lattice_type = LatticeType.SQUARE_2D_0ANC
        basis_vectors = [[0, 0]]
        lattice_vectors = [[1, 0], [0, 1]]
        return cls(basis_vectors, lattice_vectors, lattice_size, lattice_type)

    @classmethod
    def square_2d(
        cls,
        lattice_size: tuple[int, int] | None = None,
    ):
        """Generates a square lattice in 2D. The lattice tiles the plane, and is
        defined by two lattice vectors that are orthogonal to each other. The unit
        cell is defined by a set of two basis vectors. The unit cell contains 2 points
        (1 qubits, 1 ancilla) arranged in a downwards sloping line. The lattice vectors
        tile the plane in a square pattern.

        Basis vector 0 represents the data qubit, and basis vector 1 represents the
        ancilla qubit.

        Parameters
        ----------
        lattice_size : tuple[int, int] | None, optional
            The size of the lattice in terms of unit cells. If None, the lattice is
            assumed to be infinitely large.

        Returns
        -------
        Lattice
            A Lattice object representing the square lattice.
        """
        lattice_type = LatticeType.SQUARE_2D
        basis_vectors = [[0, 0], [-0.5, -0.5]]
        lattice_vectors = [[1, 0], [0, 1]]
        return cls(basis_vectors, lattice_vectors, lattice_size, lattice_type)

    @classmethod
    def square_2d_2anc(cls, lattice_size: tuple[int, int] | None = None, shift=0.15):
        """Generates a square lattice in 2D. The lattice tiles the plane, and is
        defined by two lattice vectors that are orthogonal to each other. The unit
        cell is defined by a set of two basis vectors. The unit cell contains 3 points
        (1 qubits, 2 ancilla) arranged such that the ancillas lie on the same
        line, but with a slight separation. The lattice vectors tile the plane in a
        square pattern.
        Basis vector 0 represents the data qubit, and basis vectors 1 and 2 represent
        the first ancilla and second ancilla qubit.

        NOTE: This lattice was defined for the specific use of the 488 Color code, where
        the ancillas are arranged to be slightly shifted from the center of a polygon
        and where the datas are placed at the vertices of the given polygon.

        Parameters
        ----------
        lattice_size : tuple[int, int] | None, optional
            The size of the lattice in terms of unit cells. If None, the lattice is
            assumed to be infinitely large.
        shift : float, optional
            The shift of the ancilla qubits from the center of the unit cell.
            Default is 0.15.
        Returns
        -------
        Lattice
            A Lattice object representing the square lattice.
        """
        lattice_type = LatticeType.SQUARE_2D_2ANC
        basis_vectors = [[0, 0], [0, -shift], [0, shift]]
        lattice_vectors = [[1, 0], [0, 1]]
        return cls(basis_vectors, lattice_vectors, lattice_size, lattice_type)

    @classmethod
    def cairo_pent_2d(
        cls,
        lattice_size: tuple[int, int] | None = None,
    ):
        """Generates a catalan pentagonal lattice which is a regular tiling of the
        plane. Although there are an infinite number of ways to tile the plane with
        pentagons, this implementation uses a bilaterally symmetric pentagon with 4
        long edges and 1 short edge in the ratio of 1 : sqrt(3) - 1. The angles of
        this pentagon (starting from top and moving clockwise) are 120, 90, 120, 120,
        and 90 degrees.

        Each unit cell contains 6 points. The first 5 points are the vertices of a
        pentagon, and the 6th point is a point on another pentagon that is adjacent to
        the first one. The points are arranged such that they tile the plane via the
        lattice vectors, though the tiling will not be intuitive due to the current
        unit cell definition.
        Returns
        -------
        Lattice
            A Lattice object representing the catalan pentagonal lattice.
        """

        lattice_type = LatticeType.CAIRO_PENT_2D
        basis_vectors = [
            [0, 0],
            [0.5 * np.sqrt(3), 0.5],
            [0.5 * (np.sqrt(3) - 1), 0.5 * (1 + np.sqrt(3))],
            [0.5 * (1 - np.sqrt(3)), 0.5 * (1 + np.sqrt(3))],
            [-0.5 * np.sqrt(3), 0.5],
            [-np.sqrt(3), 1],
        ]
        lattice_vectors = [[2 * np.sqrt(3), 0], [np.sqrt(3), np.sqrt(3)]]
        return cls(basis_vectors, lattice_vectors, lattice_size, lattice_type)

    @classmethod
    def hex_2d(
        cls,
        lattice_size: tuple[int, int] | None = None,
    ):
        """Generates a hexagonal lattice in 2D. The lattice tiles the plane, and is
        defined by two lattice vectors that are 60 degrees to each other. The unit
        cell contains 2 points (2 qubits) arranged in a downwards sloping line.
        The lattice vectors tile the plane in a hexagonal pattern.

        Parameters
        ----------
        lattice_size : tuple[int, int] | None, optional
            The size of the lattice in terms of unit cells. If None, the lattice is
            assumed to be infinitely large.

        Returns
        -------
        Lattice
            A Lattice object representing the hexagonal lattice.
        """

        lattice_type = LatticeType.HEX_2D
        basis_vectors = [[0, 0], [0, 1 / np.sqrt(3)]]
        lattice_vectors = [[1, 0], [0.5, 0.5 * np.sqrt(3)]]
        return cls(basis_vectors, lattice_vectors, lattice_size, lattice_type)

    @classmethod
    def oct_2d(
        cls,
        lattice_size: tuple[int, int] | None = None,
        r: float = 0.5,
        anc: int = 0,
    ):
        """Generates a semi-regular octagonal lattice with a specified radius `r` for
        the points on the circle. The lattice tiles the plane. Spaces between adjacent
        octagons are tiled by squares.

        Each unit cell contains 8 qubits + 4 sets of ancilla. The number of ancilla in
        each set is determined by the `anc` parameter, and is 0 by default. The data
        qubits are placed on a circle of radius `r`. The ancilla qubits are placed on a
        circle that is 1 / 2 the radius of the squares.

        First set with indices [8 : 8 + anc]:
            Ancilla for primary octagon. Placed on a circle of radius
            `r * sin(pi / 8) / 2`.
        Second set with indices [8 + anc : 8 + 2 * anc]:
            Ancilla for square below primary octagon. Placed on a circle of the same
            radius, shifted by half the distance between adjacent unit cells in the
            y-direction.
        Third set with indices [8 + 2 * anc : 8 + 3 * anc]:
            Ancilla for square to right of primary octagon. Placed on a circle of the
            same radius, shifted by half the distance between adjacent unit cells in
            the y-direction.
        Fourth set with indices [8 + 3 * anc : 8 + 4 * anc]:
            Ancilla for secondary octagon, diagonal down from primary octagon. Placed
            on a circle of the same radius, shifted by half the distance between
            adjacent unit cells in both the x and y directions.

        Parameters
        ----------
        lattice_size : tuple[int, int] | None, optional
            The size of the lattice in terms of unit cells. If None, the lattice is
            assumed to be infinitely large.
        r : float, optional
            The radius of the circle on which the points are placed. Default is 0.5.
        anc : int, optional
            The number of ancilla qubits in the unit cell. Default is 0. If anc is set
            to 1, ancilla will be placed at the centre of each octagonal and square
            tiles.

        Returns
        -------
        Lattice
            A Lattice object representing the octagonal lattice.
        """
        # Validation
        if not isinstance(anc, int) or anc < 0:
            raise ValueError("Number of ancilla qubits must be a non-negative integer.")

        lattice_type = LatticeType.OCT_2D
        anc_radius = r * np.sin(np.pi / 8) / 2
        cell_dist = (
            2 * r * (np.sin(np.pi / 8) + np.cos(np.pi / 8))
        )  # Ensures squares fit between octagons
        basis_vectors = cls._points_on_circle(8, r, -5 * np.pi / 8)

        if anc > 0:
            basis_vectors = basis_vectors + (
                cls._points_on_circle(anc, anc_radius)
                + cls._points_on_circle(anc, anc_radius, disp=(0, 0.5 * cell_dist))
                + cls._points_on_circle(anc, anc_radius, disp=(0.5 * cell_dist, 0))
                + cls._points_on_circle(
                    anc, anc_radius, disp=(0.5 * cell_dist, 0.5 * cell_dist)
                )
            )
        lattice_vectors = [[0, cell_dist], [cell_dist, 0]]
        return cls(basis_vectors, lattice_vectors, lattice_size, lattice_type)

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    @classmethod
    def poly_2d(
        cls,
        lattice_size: tuple[int, int] | None = None,
        n: int | None = None,
        poly_radius: float | int = 0.5,
        poly_offset: float | int = -0.5 * np.pi,
        cell_dist_factor: float | int = (1 + np.sqrt(3)),
        anc: int = 1,
        anc_radius: float | int = 0.2,
        anc_offset: float | int = -0.5 * np.pi,
        anc_disp: tuple[int | float, int | float] = (0, 0),
    ):
        """
        Generates a lattice of n-sided polygons. The points are placed on a circle
        of radius `poly_radius` and arranged in a polygonal pattern. The lattice does
        not tile the plane, but is useful for certain applications where polygonal
        symmetry is desired.

        Lattice vectors are defined such that the distance between adjacent unit cells
        is `poly_radius * cell_dist_factor`. Each unit cell contains n + anc points
        (`n` qubits, `anc` ancilla). The `anc` points are placed on a circle of radius
        `anc_radius`. If `anc` is set to 1, ancilla will be placed at the center of the
        polygon.

        Parameters
        ----------
        lattice_size : tuple[int, int] | None, optional
            The size of the lattice in terms of unit cells. If None, the lattice is
            assumed to be infinitely large.
        n : int, optional
            The number of sides of the polygon. Default is 5 (pentagon).
        poly_radius : float | int, optional
            The radius of the circle on which the points are placed. Default is 0.5.
        poly_offset : float | int, optional
            The offset to be applied to the angle of the points on the circle. This
            is useful for placing the first point at a specific angle. Default is
            -0.5 * np.pi (top of circle).
        cell_dist_factor : float | int, optional
            The factor by which the distance between adjacent unit cells is multiplied.
            This is useful for adjusting the distance between the polygons. Default is
            (1 + np.sqrt(3)) as a purely aesthetic choice to ensure that the
            polygons are not too close together.
        anc : int, optional
            The number of ancilla qubits in the unit cell. Default is 1. If anc is set
            to 1, ancilla will be placed at the center of the polygon and at the center
            of the edges of the polygon.
        anc_radius : float | int, optional
            The radius of the circle on which the ancilla points are placed. Default is
            0.2.
        anc_offset : float | int, optional
            The offset to be applied to the angle of the ancilla points on the circle.
            This is useful for placing the first ancilla point at a specific angle.
            Default is -0.5 * np.pi (top of circle).
        anc_disp : tuple[int | float, int | float], optional
            The displacement of the ancilla points from the center of the circle. This
            is useful for placing the ancilla points at a specific position in the 2D
            plane. Default is (0, 0).

        Returns
        -------
        Lattice
            A Lattice object representing the polygonal lattice.
        """
        # Validation
        if n is None:
            raise ValueError("Please specify the number of sides `n` for the polygon.")
        if not isinstance(n, int) or n < 3:
            raise ValueError("Number of sides must be an integer that is at least 3.")
        if not isinstance(anc, int) or anc < 0:
            raise ValueError("Number of ancilla qubits must be a non-negative integer.")

        lattice_type = LatticeType.POLY_2D
        basis_vectors = cls._points_on_circle(n, poly_radius, poly_offset)
        if anc > 0:
            basis_vectors = basis_vectors + cls._points_on_circle(
                anc, anc_radius, anc_offset, anc_disp
            )
        cell_dist = poly_radius * cell_dist_factor
        lattice_vectors = [[0, cell_dist], [cell_dist, 0]]
        return cls(basis_vectors, lattice_vectors, lattice_size, lattice_type)

    @classmethod
    def cube_3d(
        cls,
        lattice_size: tuple[int, int, int] | None = None,
    ):
        """
        Generates a cubic lattice in 3D. The lattice is defined by three lattice
        vectors that are orthogonal to each other tiling the 3D space in a cubic
        pattern.

        Parameters
        ----------
        lattice_size : tuple[int, int, int] | None, optional
            The size of the lattice in terms of unit cells. If None, the lattice is
            assumed to be infinitely large.

        Returns
        -------
        Lattice
            A Lattice object representing the cubic lattice.
        """
        lattice_type = LatticeType.CUBE_3D
        basis_vectors = []
        lattice_vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        return cls(basis_vectors, lattice_vectors, lattice_size, lattice_type)
