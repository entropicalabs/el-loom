"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest
import numpy as np

from loom.eka import Lattice, LatticeType


# pylint: disable=invalid-name, too-many-instance-attributes, unnecessary-lambda-assignment, protected-access
class TestLattice(unittest.TestCase):
    """Test suite for the Lattice class. These tests cover the creation of custom and
    default lattices, validation of inputs, and the behavior of various helper methods.

    To add tests for new default lattices:

        1) In `setUp()`, add lattice to `self.default_list` dictionary.

        2) If applicable, add validation tests for lattice in
        `test_default_lattice_creation_validation()`

        3) In `test_default_lattice_creation()`, add new lattice to
        `default_attr_list` dictionary

        4) In `test_default_lattice_all_qubits()`, add new lattice to
        `qubits_force_false` and `qubits_force_true` dictionaries.

    To add new tests, use the following keywords in the method name:

        - "validation": Tests for function's input validations only

        - "custom_lattice": Tests for custom defined lattices that use the Lattice
        object

        - "default_lattice": Tests for default lattices like triangle_2d, square_2d etc
    """

    def setUp(self):
        self.pts_on_circ = lambda n, r, offset, disp: tuple(
            (
                float(r * np.cos(2 * np.pi * i / n + offset) + disp[0]),
                float(r * np.sin(2 * np.pi * i / n + offset) + disp[1]),
            )
            for i in range(n)
        )

        self.unit_cells_1D = lambda x: [(i,) for i in range(x)]
        self.unit_cells_2D = lambda x, y: [(i, j) for i in range(x) for j in range(y)]
        self.unit_cells_3D = lambda x, y, z: [
            (i, j, k) for i in range(x) for j in range(y) for k in range(z)
        ]

        self.size_1D = (10,)
        self.size_2D = (10, 10)
        self.size_3D = (10, 10, 10)
        self.size_inf = None

        self.basis_vec = ((0, 0), (0.5, 0.5))
        self.lat_vec = ((1, 0), (0, 1))

        self.n = np.random.randint(3, 100)
        self.anc = int(np.random.choice([i for i in range(0, 100) if i != 1]))

        # Custom lattices
        self.fin_lattice = Lattice(self.basis_vec, self.lat_vec, size=self.size_2D)
        self.inf_lattice = Lattice(self.basis_vec, self.lat_vec, size=self.size_inf)

        # Default lattices
        self.default_list = {
            "1D_linear": [
                Lattice.linear(self.size_1D),
                Lattice.linear(self.size_inf),
            ],
            "2D_triangle": [
                Lattice.triangle_2d(self.size_2D),
                Lattice.triangle_2d(self.size_inf),
            ],
            "2D_square": [
                Lattice.square_2d(self.size_2D),
                Lattice.square_2d(self.size_inf),
            ],
            "2D_square_2d_0anc": [
                Lattice.square_2d_0anc(self.size_2D),
                Lattice.square_2d_0anc(self.size_inf),
            ],
            "2D_square_2d_2anc": [
                Lattice.square_2d_2anc(self.size_2D, shift=0.2),
                Lattice.square_2d_2anc(self.size_inf, shift=0.2),
            ],
            "2D_cairo_pent": [
                Lattice.cairo_pent_2d(self.size_2D),
                Lattice.cairo_pent_2d(self.size_inf),
            ],
            "2D_hex": [
                Lattice.hex_2d(self.size_2D),
                Lattice.hex_2d(self.size_inf),
            ],
            "2D_oct": [
                Lattice.oct_2d(self.size_2D, r=1, anc=1),
                Lattice.oct_2d(self.size_inf, r=1, anc=1),
            ],
            "2D_poly_fixed": [
                Lattice.poly_2d(self.size_2D, n=5, poly_radius=1),
                Lattice.poly_2d(self.size_inf, n=5, poly_radius=1),
            ],
            "2D_poly_rand": [
                Lattice.poly_2d(self.size_2D, n=self.n, poly_radius=1, anc=self.anc),
                Lattice.poly_2d(self.size_inf, n=self.n, poly_radius=1, anc=self.anc),
            ],
            "3D_cube_3d": [
                Lattice.cube_3d(self.size_3D),
                Lattice.cube_3d(self.size_inf),
            ],
        }

    def test_custom_lattice_creation_validation(self):
        """Test whether exceptions are raised for invalid inputs when creating custom
        lattices of finite and infinite size."""
        err_basis_vec = ((0, 0), (0.5, 0.5, 42))
        err_lat_vec = ((1, 0), (0, 1, 42))
        err_size = (10, 10, 5)

        # Test basis vectors don't have the same length
        err_msg = "All basis vectors must have the same length."
        with self.assertRaises(ValueError) as cm:
            _ = Lattice(err_basis_vec, self.lat_vec, self.size_2D)
        self.assertIn(err_msg, str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            _ = Lattice(err_basis_vec, self.lat_vec, self.size_inf)
        self.assertIn(err_msg, str(cm.exception))

        # Test lattice vectors don't have the same length
        err_msg = "All lattice vectors must have the same length."
        with self.assertRaises(ValueError) as cm:
            _ = Lattice(self.basis_vec, err_lat_vec, self.size_2D)
        self.assertIn(err_msg, str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            _ = Lattice(self.basis_vec, err_lat_vec, self.size_inf)
        self.assertIn(err_msg, str(cm.exception))

        # Test wrong size dimensions (this error does not occur for infinite lattices)
        err_msg = (
            "The given `size` is invalid. `size` has 3 elements, "
            "but the lattice has 2 dimensions."
        )
        with self.assertRaises(ValueError) as cm:
            _ = Lattice(self.basis_vec, self.lat_vec, err_size)
        self.assertIn(err_msg, str(cm.exception))

    def test_custom_lattice_creation(self):
        """Test whether custom lattices of finite and infinite size can be created
        successfully."""
        # Create list of custom lattices of finite and infinite size
        lattice_list = [self.fin_lattice, self.inf_lattice]

        # Test finite and infinite lattice creation
        for ind, lat in enumerate(lattice_list):
            fail_msg = ("Infinite " if ind else "Finite ") + "lattice creation failed."

            self.assertEqual(lat.size, (None if ind else (10, 10)), fail_msg)
            self.assertEqual(lat.lattice_type, LatticeType.CUSTOM, fail_msg)
            self.assertEqual(lat.basis_vectors, self.basis_vec, fail_msg)
            self.assertEqual(lat.lattice_vectors, self.lat_vec, fail_msg)
            self.assertEqual(lat.n_dimensions, 2, fail_msg)
            self.assertEqual(lat.unit_cell_size, 2, fail_msg)

    def test_default_lattice_creation_validation(self):
        """Test whether exceptions are raised for invalid inputs when creating default
        lattices of finite and infinite size. Currently (25/7/25), this only applies to
        oct_2d and poly_2d lattices."""
        # Test oct_2d number of ancilla qubits is not a non-negative integer
        for err_anc in [-1, -2, -3.5, "ancilla"]:
            err_msg = "Number of ancilla qubits must be a non-negative integer."
            with self.assertRaises(ValueError) as cm:
                _ = Lattice.oct_2d(self.size_2D, anc=err_anc)
            self.assertIn(err_msg, str(cm.exception))

            with self.assertRaises(ValueError) as cm:
                _ = Lattice.oct_2d(self.size_inf, anc=err_anc)
            self.assertIn(err_msg, str(cm.exception))

        # Test poly_2d number of sides not specified
        err_msg = "Please specify the number of sides `n` for the polygon."
        with self.assertRaises(ValueError) as cm:
            _ = Lattice.poly_2d(self.size_2D)
        self.assertIn(err_msg, str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            _ = Lattice.poly_2d(self.size_inf)
        self.assertIn(err_msg, str(cm.exception))

        # Test poly_2d number of sides is not an integer >= 3
        for err_n in [-1, 0, 1, 2, 2.5, "five"]:
            err_msg = "Number of sides must be an integer that is at least 3."
            with self.assertRaises(ValueError) as cm:
                _ = Lattice.poly_2d(self.size_2D, n=err_n)
            self.assertIn(err_msg, str(cm.exception))

            with self.assertRaises(ValueError) as cm:
                _ = Lattice.poly_2d(self.size_inf, n=err_n)
            self.assertIn(err_msg, str(cm.exception))

        # Test poly_2d number of ancilla qubits is not a non-negative integer
        for err_anc in [-1, -2, -3.5, "ancilla"]:
            err_msg = "Number of ancilla qubits must be a non-negative integer."
            with self.assertRaises(ValueError) as cm:
                _ = Lattice.poly_2d(self.size_2D, n=5, anc=err_anc)
            self.assertIn(err_msg, str(cm.exception))

            with self.assertRaises(ValueError) as cm:
                _ = Lattice.poly_2d(self.size_inf, n=5, anc=err_anc)
            self.assertIn(err_msg, str(cm.exception))

    def test_default_lattice_creation(self):
        """Test whether the provided default lattices of finite and infinite size can be
        created correctly."""
        # Create list of attributes for all default lattices of finite and infinite size
        half_cell_dist = np.sqrt(1 / np.sqrt(2) + 1)

        default_attr_list = {
            "1D_linear": {
                "size": self.size_1D,
                "type": LatticeType.LINEAR,
                "basis_vec": ((0, 0), (0.5, 0)),
                "lat_vec": ((1, 0),),
                "n_dimensions": 1,
                "cell_size": 2,
            },
            "2D_triangle": {
                "size": self.size_2D,
                "type": LatticeType.TRIANGLE_2D,
                "basis_vec": (
                    (0, 0),
                    (0.5, 1 / (2 * np.sqrt(3))),
                    (1, 1 / np.sqrt(3)),
                ),
                "lat_vec": ((1, 0), (0.5, 0.5 * np.sqrt(3))),
                "n_dimensions": 2,
                "cell_size": 3,
            },
            "2D_square": {
                "size": self.size_2D,
                "type": LatticeType.SQUARE_2D,
                "basis_vec": ((0, 0), (-0.5, -0.5)),
                "lat_vec": ((1, 0), (0, 1)),
                "n_dimensions": 2,
                "cell_size": 2,
            },
            "2D_square_2d_0anc": {
                "size": self.size_2D,
                "type": LatticeType.SQUARE_2D_0ANC,
                "basis_vec": ((0, 0),),
                "lat_vec": ((1, 0), (0, 1)),
                "n_dimensions": 2,
                "cell_size": 1,
            },
            "2D_square_2d_2anc": {
                "size": self.size_2D,
                "type": LatticeType.SQUARE_2D_2ANC,
                "basis_vec": ((0, 0), (0, -0.2), (0, 0.2)),
                "lat_vec": ((1, 0), (0, 1)),
                "n_dimensions": 2,
                "cell_size": 3,
            },
            "2D_cairo_pent": {
                "size": self.size_2D,
                "type": LatticeType.CAIRO_PENT_2D,
                "basis_vec": (
                    (0, 0),
                    (0.5 * np.sqrt(3), 0.5),
                    (0.5 * (np.sqrt(3) - 1), 0.5 * (1 + np.sqrt(3))),
                    (0.5 * (1 - np.sqrt(3)), 0.5 * (1 + np.sqrt(3))),
                    (-0.5 * np.sqrt(3), 0.5),
                    (-np.sqrt(3), 1),
                ),
                "lat_vec": ((2 * np.sqrt(3), 0), (np.sqrt(3), np.sqrt(3))),
                "n_dimensions": 2,
                "cell_size": 6,  # Not 5 bc pt overlap constraints
            },
            "2D_hex": {
                "size": self.size_2D,
                "type": LatticeType.HEX_2D,
                "basis_vec": ((0, 0), (0, 1 / np.sqrt(3))),
                "lat_vec": ((1, 0), (0.5, 0.5 * np.sqrt(3))),
                "n_dimensions": 2,
                "cell_size": 2,
            },
            "2D_oct": {
                "size": self.size_2D,
                "type": LatticeType.OCT_2D,
                "basis_vec": self.pts_on_circ(8, 1, -5 * np.pi / 8, (0, 0))
                + (
                    (0, 0),
                    (0, half_cell_dist),
                    (half_cell_dist, 0),
                    (half_cell_dist, half_cell_dist),
                ),
                "lat_vec": ((0, 2 * half_cell_dist), (2 * half_cell_dist, 0)),
                "n_dimensions": 2,
                "cell_size": 8 + 4 * 1,  # = n + 4 * anc
            },
            "2D_poly_fixed": {
                "size": self.size_2D,
                "type": LatticeType.POLY_2D,
                "basis_vec": self.pts_on_circ(5, 1, -0.5 * np.pi, (0, 0)) + ((0, 0),),
                "lat_vec": ((0, 1 + np.sqrt(3)), (1 + np.sqrt(3), 0)),
                "n_dimensions": 2,
                "cell_size": 5 + 1,  # = n + anc
            },
            "2D_poly_rand": {
                "size": self.size_2D,
                "type": LatticeType.POLY_2D,
                "basis_vec": self.pts_on_circ(self.n, 1, -0.5 * np.pi, (0, 0))
                + self.pts_on_circ(self.anc, 0.2, -0.5 * np.pi, (0, 0)),
                "lat_vec": ((0, 1 + np.sqrt(3)), (1 + np.sqrt(3), 0)),
                "n_dimensions": 2,
                "cell_size": self.n + self.anc,
            },
            "3D_cube_3d": {
                "size": self.size_3D,
                "type": LatticeType.CUBE_3D,
                "basis_vec": ((0, 0),),
                "lat_vec": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                "n_dimensions": 3,
                "cell_size": 1,
            },
        }

        # Test default finite and infinite lattice creation
        len_attr = len(default_attr_list["1D_linear"])
        for name, attr in default_attr_list.items():
            self.assertTrue(len(attr) == len_attr, f"Incomplete attributes for {name}.")

            _msg = f"{name} creation failed"
            if name == "2D_poly_rand":
                _msg = _msg + f" with n = {self.n}, anc = {self.anc}"

            for ind, lat in enumerate(self.default_list[name]):
                fail_msg = ("Infinite " if ind else "Finite ") + _msg

                self.assertTrue(lat.size is None if ind else attr["size"], fail_msg)
                self.assertTrue(lat.lattice_type == attr["type"], fail_msg)
                self.assertTrue(lat.basis_vectors == attr["basis_vec"], fail_msg)
                self.assertTrue(lat.lattice_vectors == attr["lat_vec"], fail_msg)
                self.assertTrue(lat.n_dimensions == attr["n_dimensions"], fail_msg)
                self.assertTrue(lat.unit_cell_size == attr["cell_size"], fail_msg)

    def test_custom_lattice_all_unit_cells_validation(self):
        """Tests whether exceptions are raised for invalid inputs to `.all_unit_cells()`
        method in custom lattices of finite and infinite size."""
        # Test missing size parameter (only occurs for infinite lattices)
        err_msg = "Please specify the `size` parameter."
        with self.assertRaises(ValueError) as cm:
            _ = self.inf_lattice.all_unit_cells()
        self.assertIn(err_msg, str(cm.exception))

        # Test incomplete size parameter (only occurs for infinite lattices)
        err_msg = "Please specify all dimensions in the `size` parameter."
        with self.assertRaises(ValueError) as cm:
            _ = self.inf_lattice.all_unit_cells((None,))
        self.assertIn(err_msg, str(cm.exception))

        # Test invalid size parameter
        for err_size in [(10,), (10, 10, 5), (10, 10, 10, 10)]:
            err_msg = (
                f"The given `size` is invalid. `size` has {len(err_size)} "
                "elements, but the lattice has 2 dimensions."
            )
            with self.assertRaises(ValueError) as cm:
                _ = self.fin_lattice.all_unit_cells(err_size)
            self.assertIn(err_msg, str(cm.exception))

            with self.assertRaises(ValueError) as cm:
                _ = self.inf_lattice.all_unit_cells(err_size)
            self.assertIn(err_msg, str(cm.exception))

    def test_custom_lattice_all_unit_cells(self):
        """Tests whether correct unit cells for custom lattices of finite and infinite
        size are returned."""
        unit_cells = lambda x, y: [(i, j) for i in range(x) for j in range(y)]

        # Test custom finite lattice all unit cells
        self.assertEqual(self.fin_lattice.all_unit_cells(), unit_cells(10, 10))
        self.assertEqual(self.fin_lattice.all_unit_cells((2, None)), unit_cells(2, 10))
        self.assertEqual(self.fin_lattice.all_unit_cells((None, 5)), unit_cells(10, 5))

        # Test custom infinite lattice all unit cells
        self.assertEqual(self.inf_lattice.all_unit_cells((3, 3)), unit_cells(3, 3))

    def test_default_lattice_all_unit_cells_validation(self):
        """Tests whether exceptions are raised for invalid inputs to `.all_unit_cells()`
        method in default lattices of finite and infinite size."""

        for name, (fin_lat, inf_lat) in self.default_list.items():
            # Test missing size parameter (only occurs for infinite lattices)
            err_msg = "Please specify the `size` parameter."
            with self.assertRaises(ValueError) as cm:
                inf_lat.all_unit_cells()
            self.assertIn(err_msg, str(cm.exception), f"{name} infinite lattice failed")

            # Test incomplete size parameter (only occurs for infinite lattices)
            err_msg = "Please specify all dimensions in the `size` parameter."
            with self.assertRaises(ValueError) as cm:
                inf_lat.all_unit_cells((None,))
            self.assertIn(err_msg, str(cm.exception), f"{name} infinite lattice failed")

            # Test invalid size parameter
            err_msg = (
                "The given `size` is invalid. `size` has 4 "
                f"elements, but the lattice has {fin_lat.n_dimensions} dimensions."
            )
            with self.assertRaises(ValueError) as cm:
                _ = fin_lat.all_unit_cells((10, 10, 10, 10))
            self.assertIn(err_msg, str(cm.exception), f"{name} finite lattice failed")

            with self.assertRaises(ValueError) as cm:
                _ = inf_lat.all_unit_cells((10, 10, 10, 10))
            self.assertIn(err_msg, str(cm.exception), f"{name} infinite lattice failed")

    def test_default_lattice_all_unit_cells(self):
        """Tests whether correct unit cells for default lattices of finite and infinite
        size are returned."""
        # Create unit cells helper function for multiple dimensions and
        # mapping between inputs and expected outputs

        fin_map = {
            "1D": [
                (None, self.unit_cells_1D(*self.size_1D)),
                ((3,), self.unit_cells_1D(3)),
            ],
            "2D": [
                (None, self.unit_cells_2D(*self.size_2D)),
                ((3, None), self.unit_cells_2D(3, 10)),
                ((None, 3), self.unit_cells_2D(10, 3)),
                ((3, 3), self.unit_cells_2D(3, 3)),
            ],
            "3D": [
                (None, self.unit_cells_3D(*self.size_3D)),
                ((3, None, None), self.unit_cells_3D(3, 10, 10)),
                ((None, 3, None), self.unit_cells_3D(10, 3, 10)),
                ((None, None, 3), self.unit_cells_3D(10, 10, 3)),
                ((3, 3, None), self.unit_cells_3D(3, 3, 10)),
                ((3, None, 3), self.unit_cells_3D(3, 10, 3)),
                ((None, 3, 3), self.unit_cells_3D(10, 3, 3)),
                ((3, 3, 3), self.unit_cells_3D(3, 3, 3)),
            ],
        }

        inf_map = {
            "1D": ((3,), self.unit_cells_1D(3)),
            "2D": ((3, 3), self.unit_cells_2D(3, 3)),
            "3D": ((3, 3, 3), self.unit_cells_3D(3, 3, 3)),
        }

        # Test default lattice all unit cells
        for name, (fin_lat, inf_lat) in self.default_list.items():
            fail_msg = f"{name} lattice failed"
            if name == "2D_poly_rand":
                fail_msg = fail_msg + f" with n = {self.n}, anc = {self.anc}"

            for fin_inp, fin_out in fin_map[name[:2]]:
                self.assertEqual(fin_lat.all_unit_cells(fin_inp), fin_out, fail_msg)

            inf_inp, inf_out = inf_map[name[:2]]
            self.assertEqual(inf_lat.all_unit_cells(inf_inp), inf_out, fail_msg)

    def test_custom_lattice_all_qubits(self):
        """Tests whether the correct qubits for custom lattices of finite and infinite
        size are returned.

        For lattices with unit cell size = 1, the output of `all_qubits()` will not
        contain a third index for specifying the qubit inside the unit cell. If
        `force_including_basis` is set to True, the output will contain the third
        index for specifying the qubit inside the unit cell (always 0 in this case).

        For lattices with unit cell size > 1, the output of `all_qubits()` and
        `all_qubits(force_including_basis=True)` will be the same, since there is
        always a third index for specifying the qubit inside the unit cell.

        Currently (25/7/25), every lattice except cube_3d have unit cell size > 1.
        """
        # Create custom 1D, 2D, 3D lattices with one qubit per unit cell
        custom_lattices = {
            "1D_custom": Lattice(
                basis_vectors=((0,),),
                lattice_vectors=((1,),),
                size=self.size_1D,
            ),
            "2D_custom": Lattice(
                basis_vectors=((0, 0),),
                lattice_vectors=((1, 0), (0, 1)),
                size=self.size_2D,
            ),
            "3D_custom": Lattice(
                basis_vectors=((0, 0, 0),),
                lattice_vectors=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                size=self.size_3D,
            ),
        }

        qubits_force_false = {  # force_including_basis=False
            "1D_custom": self.unit_cells_1D(*self.size_1D),
            "2D_custom": self.unit_cells_2D(*self.size_2D),
            "3D_custom": self.unit_cells_3D(*self.size_3D),
        }

        qubits_force_true = {  # force_including_basis=True
            "1D_custom": [tup + (0,) for tup in qubits_force_false["1D_custom"]],
            "2D_custom": [tup + (0,) for tup in qubits_force_false["2D_custom"]],
            "3D_custom": [tup + (0,) for tup in qubits_force_false["3D_custom"]],
        }

        # Test custom lattice all qubits
        for name, custom_lat in custom_lattices.items():
            fail_msg = f"{name} lattice failed"

            self.assertEqual(
                custom_lat.all_qubits(force_including_basis=False),
                qubits_force_false[name],
                fail_msg + " with force_including_basis=False",
            )
            self.assertEqual(
                custom_lat.all_qubits(force_including_basis=True),
                qubits_force_true[name],
                fail_msg + " with force_including_basis=True",
            )

    def test_default_lattice_all_qubits(self):
        """Tests whether the correct qubits for default lattices are returned (only
        occurs for finite lattices).

        For lattices with unit cell size = 1, the output of `all_qubits()` will not
        contain a third index for specifying the qubit inside the unit cell. If
        `force_including_basis` is set to True, the output will contain the third
        index for specifying the qubit inside the unit cell (always 0 in this case).

        For lattices with unit cell size > 1, the output of `all_qubits()` and
        `all_qubits(force_including_basis=True)` will be the same, since there is
        always a third index for specifying the qubit inside the unit cell.

        Currently (25/7/25), every lattice except 2D_square_2d_0anc and cube_3d have
        unit cell size > 1.
        """
        all_qubits_1D = lambda x, cell: [(i, j) for i in range(x) for j in range(cell)]
        all_qubits_2D = lambda x, y, cell: [
            (i, j, k) for i in range(x) for j in range(y) for k in range(cell)
        ]
        all_qubits_3D = lambda x, y, z, cell: [
            (i, j, k, l)
            for i in range(x)
            for j in range(y)
            for k in range(z)
            for l in range(cell)
        ]

        qubits_force_false = {  # force_including_basis=False
            "1D_linear": all_qubits_1D(*self.size_1D, 2),
            "2D_triangle": all_qubits_2D(*self.size_2D, 3),
            "2D_square": all_qubits_2D(*self.size_2D, 2),
            "2D_square_2d_0anc": [
                (x, y) for x, y, _ in all_qubits_2D(*self.size_2D, 1)
            ],
            "2D_square_2d_2anc": all_qubits_2D(*self.size_2D, 3),
            "2D_cairo_pent": all_qubits_2D(*self.size_2D, 6),
            "2D_hex": all_qubits_2D(*self.size_2D, 2),
            "2D_oct": all_qubits_2D(*self.size_2D, 8 + 1 * 4),
            "2D_poly_fixed": all_qubits_2D(*self.size_2D, 5 + 1),
            "2D_poly_rand": all_qubits_2D(*self.size_2D, self.n + self.anc),
            "3D_cube_3d": [(x, y, z) for x, y, z, _ in all_qubits_3D(*self.size_3D, 1)],
        }

        qubits_force_true = {  # force_including_basis=True
            "1D_linear": qubits_force_false["1D_linear"],
            "2D_triangle": qubits_force_false["2D_triangle"],
            "2D_square": qubits_force_false["2D_square"],
            "2D_square_2d_0anc": all_qubits_2D(*self.size_2D, 1),
            "2D_square_2d_2anc": qubits_force_false["2D_square_2d_2anc"],
            "2D_cairo_pent": qubits_force_false["2D_cairo_pent"],
            "2D_hex": qubits_force_false["2D_hex"],
            "2D_oct": qubits_force_false["2D_oct"],
            "2D_poly_fixed": qubits_force_false["2D_poly_fixed"],
            "2D_poly_rand": qubits_force_false["2D_poly_rand"],
            "3D_cube_3d": all_qubits_3D(*self.size_3D, 1),
        }

        # Test default lattice all qubits (finite only)
        for name, (fin_lat, _) in self.default_list.items():
            fail_msg = f"{name} lattice failed"
            if name == "2D_poly_rand":
                fail_msg = fail_msg + f" with n = {self.n}, anc = {self.anc}"

            self.assertEqual(
                fin_lat.all_qubits(force_including_basis=False),
                qubits_force_false[name],
                fail_msg + " with force_including_basis=False",
            )
            self.assertEqual(
                fin_lat.all_qubits(force_including_basis=True),
                qubits_force_true[name],
                fail_msg + " with force_including_basis=True",
            )

    def test__points_on_circle_validation(self):
        """Test whether exceptions are raised for invalid inputs to _points_on_circle,
        the helper function that generates points on a circle."""
        # Test whether exceptions are raised for invalid inputs.
        n_points = 10
        radius = 5.0
        offset = 0.0
        disp = (0.0, 0.0)

        # Test radius
        err_radius = [-1.0, 0, "five"]
        for err_r in err_radius:
            err_msg = (
                "Radius must be a positive number (int or float). "
                f"Received {err_r} of type {type(err_r)}."
            )
            with self.assertRaises(ValueError) as cm:
                _ = Lattice._points_on_circle(n_points, err_r, offset, disp)
            self.assertIn(err_msg, str(cm.exception))

        # Test n_points
        err_points = [0, 10.5]
        for err_n in err_points:
            err_msg = (
                "Number of points must be an integer that is at least 1. "
                f"Received {err_n} of type {type(err_n)}."
            )
            with self.assertRaises(ValueError) as cm:
                _ = Lattice._points_on_circle(err_n, radius, offset, disp)
            self.assertIn(err_msg, str(cm.exception))

        # Test offset
        err_offset = ["zero", 0.0j, None]
        for err_o in err_offset:
            err_msg = (
                "Offset must be a number (int or float). "
                f"Received {err_o} of type {type(err_o)}."
            )
            with self.assertRaises(TypeError) as cm:
                _ = Lattice._points_on_circle(n_points, radius, err_o, disp)
            self.assertIn(err_msg, str(cm.exception))

        # Test disp
        err_disp = [(0.0, "zero"), (0.0, 0.0j), (None, 0.0), (0.0, 0.0, 0.0)]
        for err_d in err_disp:
            err_msg = (
                "Disp must be a tuple of two numbers (x, y). "
                f"Received {err_d} of type {type(err_d)}."
            )
            with self.assertRaises(TypeError) as cm:
                _ = Lattice._points_on_circle(n_points, radius, offset, err_d)
            self.assertIn(err_msg, str(cm.exception))

    def test__points_on_circle(self):
        """Test whether correct outputs are returned by _points_on_circle, the helper
        function that generates points on a circle."""
        # Test with fixed parameters
        n = 10
        r = 5.0
        offset = 0.01
        disp = (0.0, 0.0)
        expected_points = self.pts_on_circ(n, r, offset, disp)
        actual_points = tuple(
            tuple(i) for i in Lattice._points_on_circle(n, r, offset, disp)
        )
        self.assertEqual(expected_points, actual_points)

        # Test with random parameters
        n = np.random.randint(3, 100)
        r = np.random.uniform(0.1, 100)
        offset = np.random.uniform(-np.pi, np.pi)
        disp = (np.random.uniform(-10, 10), np.random.uniform(-10, 10))
        expected_points = self.pts_on_circ(n, r, offset, disp)
        actual_points = tuple(
            tuple(i) for i in Lattice._points_on_circle(n, r, offset, disp)
        )
        self.assertEqual(
            expected_points,
            actual_points,
            f"Assert failed with n={n}, r={r}, offset={offset}, disp={disp}",
        )


if __name__ == "__main__":
    unittest.main()
