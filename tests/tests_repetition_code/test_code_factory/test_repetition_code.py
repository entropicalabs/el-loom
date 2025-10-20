"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest
from itertools import product

from loom.eka import Stabilizer, PauliOperator, Lattice
from loom.eka.utilities import Direction
from loom_repetition_code.code_factory import RepetitionCode


class TestRepetitionCodeBlock(unittest.TestCase):
    """
    Test suite for the RepetitionCode block.
    """

    def setUp(self):
        self.linear_lattice = Lattice.linear()
        self.distances = [3, 5, 7]
        self.check_types = ["X", "Z"]

    def test_repetition_code_input_validation(self):
        """Test the input validation for the creation of a repetition code block."""

        # Check distance is a positive integer
        invalid_distances = [0, -1, 0.5, "invalid"]
        for d in invalid_distances:
            err_msg = f"`d` must be a positive integer. Got '{d}' instead."
            with self.assertRaises(ValueError) as cm:
                RepetitionCode.create(
                    d, check_type="X", lattice=self.linear_lattice, unique_label="q1"
                )
            self.assertEqual(str(cm.exception), err_msg)

        # Check check_type is either "X" or "Z"
        invalid_check_types = ["Y", "invalid", "x"]
        for check_type in invalid_check_types:
            err_msg = (
                f"`check_type` must be either 'X' or 'Z'. Got '{check_type}' instead."
            )
            with self.assertRaises(ValueError) as cm:
                RepetitionCode.create(
                    d=3,
                    check_type=check_type,
                    lattice=self.linear_lattice,
                    unique_label="q1",
                )
            self.assertEqual(str(cm.exception), err_msg)

        # Lattice for which the creation of repetition codes is not supported
        cube_3d = Lattice.cube_3d()
        err_msg = (
            "The creation of repetition chains is currently only supported for "
            f"linear lattices. Instead the lattice is of type {cube_3d.lattice_type}."
        )
        with self.assertRaises(ValueError) as cm:
            _ = RepetitionCode.create(
                d=11, check_type="Z", lattice=cube_3d, unique_label="q1"
            )
        self.assertEqual(str(cm.exception), err_msg)

        # Invalid input for the position
        positions = ["invalid", (0.55,)]
        for position in positions:
            err_msg = (
                f"`position` must be a tuple of integers. Got '{position}' instead."
            )
            with self.assertRaises(ValueError) as cm:
                _ = RepetitionCode.create(
                    d=9,
                    check_type="X",
                    lattice=self.linear_lattice,
                    unique_label="q1",
                    position=position,
                )
            self.assertEqual(str(cm.exception), err_msg)

        # Position tuple has the wrong length (!= lattice dimension)
        position = (0, 9, 2)
        err_msg = (
            f"`position` has length {len(position)} while length "
            f"{self.linear_lattice.n_dimensions} is required to match the lattice "
            "dimension."
        )
        with self.assertRaises(ValueError) as cm:
            RepetitionCode.create(
                d=9,
                lattice=self.linear_lattice,
                check_type="X",
                unique_label="q1",
                position=position,
            )
        self.assertEqual(str(cm.exception), err_msg)

        # Logical operator length does not match distance
        err_msg = "Support of input X logical should be equal to distance"
        with self.assertRaises(ValueError) as cm:
            RepetitionCode.create(
                d=9,
                lattice=self.linear_lattice,
                check_type="Z",
                unique_label="q1",
                logical_x_operator=PauliOperator(
                    pauli="XXX", data_qubits=[(0, 0), (1, 0), (2, 0)]
                ),
            )
        self.assertEqual(str(cm.exception), err_msg)

        err_msg = "Support of input Z logical should be equal to distance"
        with self.assertRaises(ValueError) as cm:
            RepetitionCode.create(
                d=9,
                lattice=self.linear_lattice,
                check_type="X",
                unique_label="q1",
                logical_z_operator=PauliOperator(pauli="Z", data_qubits=[(0, 0)]),
            )
        self.assertEqual(str(cm.exception), err_msg)

    def test_repetition_code_check_types(self):
        """Test the check type properties for repetition code."""

        for check in self.check_types:
            block = RepetitionCode.create(
                d=9,
                check_type=check,
                lattice=self.linear_lattice,
                unique_label="q1",
                position=(1,),
            )
            self.assertEqual(block.check_type, check)

    def test_repetition_code_boundary_qubits(self):
        """Test the extraction of boundary qubits for repetition code."""

        # Test for several distances and both check types
        # Shift the block every time to check the boundary qubits
        for d, check_type in product(self.distances, self.check_types):
            block = RepetitionCode.create(
                d=d,
                check_type=check_type,
                lattice=self.linear_lattice,
                unique_label="q1",
                position=(d // 2,),
            )

            left_boundary_qubit = (d // 2, 0)
            right_boundary_qubit = (d // 2 + d - 1, 0)

            for left in ["left", Direction.LEFT]:
                self.assertEqual(block.boundary_qubits(left), left_boundary_qubit)
            for right in ["right", Direction.RIGHT]:
                self.assertEqual(block.boundary_qubits(right), right_boundary_qubit)

        # Test for invalid directions
        invalid_directions = ["top", "bottom", Direction.BOTTOM, Direction.TOP]

        for direction in invalid_directions:
            err_msg = (
                f"Invalid direction '{direction}'. "
                "Only 'left' and 'right' are supported."
            )

            block = RepetitionCode.create(
                d=9,
                check_type="X",
                lattice=self.linear_lattice,
                unique_label="q1",
                position=(1,),
            )

            with self.assertRaises(ValueError) as cm:
                _ = block.boundary_qubits(direction)
            self.assertEqual(str(cm.exception), err_msg)

    def test_repetition_code_shifted_logical(self):  # pylint: disable=too-many-locals
        """Test the shifting of a logical operator located on the left boundary"""

        left_boundary = 2

        for d, check_type in product(self.distances, self.check_types):

            # Place small logical in the middle of the chain
            short_logical = PauliOperator(pauli=check_type, data_qubits=[(d // 2, 0)])
            x_logical, z_logical = (
                (short_logical, None) if check_type == "X" else (None, short_logical)
            )

            block = RepetitionCode.create(
                d=d,
                check_type=check_type,
                lattice=self.linear_lattice,
                unique_label="q1",
                logical_x_operator=x_logical,
                logical_z_operator=z_logical,
                position=(left_boundary,),
            )

            # Position of the small logical after block initialization
            current_position = d // 2 + left_boundary

            # Test shift to the left
            for shift_position in range(left_boundary, current_position):

                correct_shifted_logical = PauliOperator(
                    pauli=check_type, data_qubits=[(shift_position, 0)]
                )

                correct_required_stabilizers = tuple(
                    Stabilizer(
                        pauli=check_type * 2,
                        data_qubits=[
                            (i, 0),
                            (i + 1, 0),
                        ],
                        ancilla_qubits=[(i, 1)],
                    )
                    for i in range(shift_position, current_position)
                )

                new_logical_qubit = (shift_position, 0)
                shifted_logical, required_stabilizers = (
                    block.get_shifted_equivalent_logical_operator(new_logical_qubit)
                )

                self.assertEqual(shifted_logical, correct_shifted_logical)
                self.assertEqual(required_stabilizers, correct_required_stabilizers)

            # Test shift to the right
            for shift_position in range(current_position + 1, left_boundary + d):

                correct_shifted_logical = PauliOperator(
                    pauli=check_type, data_qubits=[(shift_position, 0)]
                )

                correct_required_stabilizers = tuple(
                    Stabilizer(
                        pauli=check_type * 2,
                        data_qubits=[
                            (i, 0),
                            (i + 1, 0),
                        ],
                        ancilla_qubits=[(i, 1)],
                    )
                    for i in range(current_position, shift_position)
                )

                new_logical_qubit = (shift_position, 0)
                shifted_logical, required_stabilizers = (
                    block.get_shifted_equivalent_logical_operator(new_logical_qubit)
                )

                self.assertEqual(shifted_logical, correct_shifted_logical)
                self.assertEqual(required_stabilizers, correct_required_stabilizers)

        # Check for invalid inputs
        for d, check_type in product(self.distances, self.check_types):
            block = RepetitionCode.create(
                d=d,
                check_type=check_type,
                lattice=self.linear_lattice,
                unique_label="q1",
                position=(left_boundary,),
            )

            invalid_positions = [
                left_boundary - 1,
                d + left_boundary,
                d + left_boundary + 1,
            ]

            for shift_position in invalid_positions:
                new_logical_qubit = (shift_position, 0)
                err_msg = (
                    f"New logical position {new_logical_qubit} "
                    "is not part of the data qubits"
                )
                with self.assertRaises(ValueError) as cm:
                    shifted_logical, required_stabilizers = (
                        block.get_shifted_equivalent_logical_operator(new_logical_qubit)
                    )
                self.assertEqual(str(cm.exception), err_msg)

    def test_repetition_code_stabilizer_definition(self):
        """Test the definition of stabilizers for a repetition code block."""

        # Test for several distances and both check types
        for d, check_type in product(self.distances, self.check_types):
            block = RepetitionCode.create(
                d,
                check_type=check_type,
                lattice=self.linear_lattice,
                unique_label="q1",
            )

            stabilizers = tuple(
                Stabilizer(
                    pauli=check_type * 2,
                    data_qubits=[(i, 0), (i + 1, 0)],
                    ancilla_qubits=[(i, 1)],
                )
                for i in range(d - 1)
            )

            self.assertEqual(block.stabilizers, stabilizers)

    def test_repetition_code_logical_operators_definition(self):
        """Test the definition of logical operators for a repetition code block."""

        # Test for several distances and both check types
        for d, check_type in product(self.distances, self.check_types):
            block = RepetitionCode.create(
                d,
                check_type=check_type,
                lattice=self.linear_lattice,
                unique_label="q1",
            )

            if check_type == "X":
                logical_z_operator = PauliOperator(
                    pauli="Z" * d, data_qubits=[(i, 0) for i in range(d)]
                )
                logical_x_operator = PauliOperator(pauli="X", data_qubits=[(0, 0)])
            else:
                logical_x_operator = PauliOperator(
                    pauli="X" * d, data_qubits=[(i, 0) for i in range(d)]
                )
                logical_z_operator = PauliOperator(pauli="Z", data_qubits=[(0, 0)])

            self.assertEqual(block.logical_x_operators, (logical_x_operator,))
            self.assertEqual(block.logical_z_operators, (logical_z_operator,))

    def test_repetition_code_creation(self):
        """Test the creation of a repetition code block."""

        # Test for several distances and both check types
        for d, check_type in product(self.distances, self.check_types):

            # Define block through create method
            block = RepetitionCode.create(
                d,
                check_type=check_type,
                lattice=self.linear_lattice,
                unique_label="q1",
                position=(1,),
            )

            # Define stabilizers
            stabilizers = [
                Stabilizer(
                    pauli=check_type * 2,
                    data_qubits=[(i, 0), (i + 1, 0)],
                    ancilla_qubits=[(i, 1)],
                )
                for i in range(d - 1)
            ]

            # Define logicals
            if check_type == "X":
                logical_z_operator = PauliOperator(
                    pauli="Z" * d, data_qubits=[(i, 0) for i in range(d)]
                )
                logical_x_operator = PauliOperator(pauli="X", data_qubits=[(0, 0)])
            else:
                logical_x_operator = PauliOperator(
                    pauli="X" * d, data_qubits=[(i, 0) for i in range(d)]
                )
                logical_z_operator = PauliOperator(pauli="Z", data_qubits=[(0, 0)])

            # Define block manually
            manual_block = RepetitionCode(
                unique_label="q1",
                stabilizers=stabilizers,
                logical_x_operators=[logical_x_operator],
                logical_z_operators=[logical_z_operator],
            ).shift((1,))

            self.assertEqual(block, manual_block)


if __name__ == "__main__":
    unittest.main()
