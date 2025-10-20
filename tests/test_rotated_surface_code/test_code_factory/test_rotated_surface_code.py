"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=duplicate-code
import unittest
import itertools
import logging

from loom.eka import (
    Block,
    Channel,
    Circuit,
    Stabilizer,
    SyndromeCircuit,
    PauliOperator,
    Lattice,
)
from loom.eka.utilities import Direction, Orientation

from loom_rotated_surface_code.code_factory import RotatedSurfaceCode
from loom_rotated_surface_code.utilities import FourBodySchedule

# Set up logging
logging.getLogger().setLevel(logging.DEBUG)


class TestSurfaceCodeBlock(unittest.TestCase):
    """
    Test the functionalities of the RotatedSurfaceCode class, which is a
    subclass of the Block class.
    """

    def setUp(self):
        self.lattice_2d_square = Lattice.square_2d()
        self.rsc = RotatedSurfaceCode.create(
            dx=3, dz=3, lattice=self.lattice_2d_square, unique_label="q1"
        )

    def test_rotated_surface_code_creation(self):
        """
        Test the creation of rotated surface codes using the
        `RotatedSurfaceCode.create`.
        """
        block = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice_2d_square,
            unique_label="q1",
        )

        # Define the Block manually, to compare with the one created by the function
        # `create()`
        manual_block = RotatedSurfaceCode(
            stabilizers=(
                Stabilizer(
                    pauli="ZZZZ",
                    data_qubits=((1, 0, 0), (0, 0, 0), (1, 1, 0), (0, 1, 0)),
                    ancilla_qubits=((1, 1, 1),),
                    uuid=block.stabilizers[0].uuid,
                ),
                Stabilizer(
                    pauli="ZZZZ",
                    data_qubits=((2, 1, 0), (1, 1, 0), (2, 2, 0), (1, 2, 0)),
                    ancilla_qubits=((2, 2, 1),),
                    uuid=block.stabilizers[1].uuid,
                ),
                Stabilizer(
                    pauli="XXXX",
                    data_qubits=((1, 1, 0), (1, 2, 0), (0, 1, 0), (0, 2, 0)),
                    ancilla_qubits=((1, 2, 1),),
                    uuid=block.stabilizers[2].uuid,
                ),
                Stabilizer(
                    pauli="XXXX",
                    data_qubits=((2, 0, 0), (2, 1, 0), (1, 0, 0), (1, 1, 0)),
                    ancilla_qubits=((2, 1, 1),),
                    uuid=block.stabilizers[3].uuid,
                ),
                Stabilizer(
                    pauli="XX",
                    data_qubits=((0, 0, 0), (0, 1, 0)),
                    ancilla_qubits=((0, 1, 1),),
                    uuid=block.stabilizers[4].uuid,
                ),
                Stabilizer(
                    pauli="XX",
                    data_qubits=((2, 1, 0), (2, 2, 0)),
                    ancilla_qubits=((3, 2, 1),),
                    uuid=block.stabilizers[5].uuid,
                ),
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((2, 0, 0), (1, 0, 0)),
                    ancilla_qubits=((2, 0, 1),),
                    uuid=block.stabilizers[6].uuid,
                ),
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((1, 2, 0), (0, 2, 0)),
                    ancilla_qubits=((1, 3, 1),),
                    uuid=block.stabilizers[7].uuid,
                ),
            ),
            logical_x_operators=[
                PauliOperator(
                    pauli="XXX", data_qubits=((0, 0, 0), (1, 0, 0), (2, 0, 0))
                )
            ],
            logical_z_operators=[
                PauliOperator(
                    pauli="ZZZ", data_qubits=((0, 0, 0), (0, 1, 0), (0, 2, 0))
                )
            ],
            unique_label="q1",
            syndrome_circuits=block.syndrome_circuits,
            stabilizer_to_circuit=block.stabilizer_to_circuit,
        )

        self.assertEqual(block, manual_block)

    def test_rotated_surface_code_variations(self):
        """Test whether the rotated surface code can be successfully created for
        different dimensions (1 dimensional, i.e. a repetition code, even & odd distance
        etc.) and different variations of boundaries and syndrome extraction schedules.
        """
        dimensions = [
            # Different repetition codes
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 10),
            (2, 1),
            (3, 1),
            (11, 1),
            # Different surface codes
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (2, 9),
            (3, 8),
            (4, 5),
            (7, 2),
            (5, 4),
        ]

        for dim, x_boundary, first_row, weight_4_x_schedule in list(
            itertools.product(
                dimensions, ["horizontal", "vertical"], [True, False], ["N", "Z"]
            )
        ):
            _ = RotatedSurfaceCode.create(
                dx=dim[0],
                dz=dim[1],
                lattice=self.lattice_2d_square,
                unique_label="q1",
                x_boundary=x_boundary,
                weight_2_stab_is_first_row=first_row,
                weight_4_x_schedule=weight_4_x_schedule,
            )

    def test_rotated_surface_code_creation_input_validation(self):
        """
        Test the input validation of the `RotatedSurfaceCode.create` function.
        """
        hex_2d = Lattice.hex_2d()

        # Lattice for which the creation of rotated surface codes is not supported
        with self.assertRaises(ValueError):
            RotatedSurfaceCode.create(dx=3, dz=3, lattice=hex_2d, unique_label="q1")

        # Invalid input for the position
        with self.assertRaises(ValueError):
            RotatedSurfaceCode.create(
                dx=3,
                dz=3,
                lattice=self.lattice_2d_square,
                unique_label="q1",
                position="invalid",
            )

        # Position tuple has the wrong length (!= lattice dimension)
        with self.assertRaises(ValueError):
            RotatedSurfaceCode.create(
                dx=3,
                dz=3,
                lattice=self.lattice_2d_square,
                unique_label="q1",
                position=(0, 0, 0),
            )

    def test_rotated_surface_code_different_schedule(self):
        """Test whether the rotated surface can successfully be created by using N
        schedule for Z stabilizers."""
        block = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice_2d_square,
            unique_label="q1",
            weight_4_x_schedule="N",
        )

        manual_block = Block(
            stabilizers=(
                Stabilizer(
                    pauli="ZZZZ",
                    data_qubits=((1, 0, 0), (0, 0, 0), (1, 1, 0), (0, 1, 0)),
                    ancilla_qubits=((1, 1, 1),),
                    uuid=block.stabilizers[0].uuid,
                ),
                Stabilizer(
                    pauli="ZZZZ",
                    data_qubits=((2, 1, 0), (1, 1, 0), (2, 2, 0), (1, 2, 0)),
                    ancilla_qubits=((2, 2, 1),),
                    uuid=block.stabilizers[1].uuid,
                ),
                Stabilizer(
                    pauli="XXXX",
                    data_qubits=((1, 1, 0), (1, 2, 0), (0, 1, 0), (0, 2, 0)),
                    ancilla_qubits=((1, 2, 1),),
                    uuid=block.stabilizers[2].uuid,
                ),
                Stabilizer(
                    pauli="XXXX",
                    data_qubits=((2, 0, 0), (2, 1, 0), (1, 0, 0), (1, 1, 0)),
                    ancilla_qubits=((2, 1, 1),),
                    uuid=block.stabilizers[3].uuid,
                ),
                Stabilizer(
                    pauli="XX",
                    data_qubits=((0, 0, 0), (0, 1, 0)),
                    ancilla_qubits=((0, 1, 1),),
                    uuid=block.stabilizers[4].uuid,
                ),
                Stabilizer(
                    pauli="XX",
                    data_qubits=((2, 1, 0), (2, 2, 0)),
                    ancilla_qubits=((3, 2, 1),),
                    uuid=block.stabilizers[5].uuid,
                ),
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((2, 0, 0), (1, 0, 0)),
                    ancilla_qubits=((2, 0, 1),),
                    uuid=block.stabilizers[6].uuid,
                ),
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((1, 2, 0), (0, 2, 0)),
                    ancilla_qubits=((1, 3, 1),),
                    uuid=block.stabilizers[7].uuid,
                ),
            ),
            logical_x_operators=[
                PauliOperator(
                    pauli="XXX", data_qubits=((0, 0, 0), (1, 0, 0), (2, 0, 0))
                )
            ],
            logical_z_operators=[
                PauliOperator(
                    pauli="ZZZ", data_qubits=((0, 0, 0), (0, 1, 0), (0, 2, 0))
                )
            ],
            unique_label="q1",
            syndrome_circuits=block.syndrome_circuits,
            stabilizer_to_circuit=block.stabilizer_to_circuit,
        )

        self.assertEqual(block, manual_block)

    def test_rotated_surface_code_default_circuits(self):
        """Tests that the default syndrome circuits and the mapping are correctly
        created for a 3x3 rotated surface code."""
        # Check that there are as many stabilizers-to-SyndromeCircuit mappings as there
        # are stabilizers
        self.assertEqual(len(self.rsc.stabilizers), len(self.rsc.stabilizer_to_circuit))
        # Check that there are 6 SyndromeCircuits (left-XX, right-XX, top-ZZ, top-ZZ,
        #  XXXX, ZZZZ)
        self.assertEqual(len(self.rsc.syndrome_circuits), 6)

        # Check that uuids exist and pauli strings match
        syndrome_circs_dict = {circ.uuid: circ for circ in self.rsc.syndrome_circuits}
        stabilizer_dict = {stab.uuid: stab for stab in self.rsc.stabilizers}
        for stab_uuid, syndrome_circ_uuid in self.rsc.stabilizer_to_circuit.items():
            self.assertTrue(stab_uuid in stabilizer_dict.keys())
            self.assertTrue(syndrome_circ_uuid in syndrome_circs_dict.keys())
            self.assertEqual(
                stabilizer_dict[stab_uuid].pauli,
                syndrome_circs_dict[syndrome_circ_uuid].pauli,
            )

    def test_rotated_surface_code_anc_qubit_assignment(
        self,
    ):  # pylint: disable=too-many-branches
        """
        Test that the ancilla qubits of each stabilizer are correctly assigned.
        Ancilla qubits are denoted by the last element of the tuple representing the
        co-ordinate of that qubit. (Data qubits are represented by a 0, ancilla qubits
        by a 1).
        """
        # Create a 3x3 Block
        block = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice_2d_square,
        )

        # Check that the qubit type booleans are correctly assigned within the
        # Stabilizers.
        self.assertEqual(len(block.stabilizers), 8)
        for each_stab in block.stabilizers:
            for each_qubit in each_stab.data_qubits:
                self.assertEqual(each_qubit[-1], 0)
            for each_qubit in each_stab.ancilla_qubits:
                self.assertEqual(each_qubit[-1], 1)

            # Each Stabilizer has only 1 ancilla qubit
            self.assertEqual(len(each_stab.ancilla_qubits), 1)

            # Check that the ancilla qubits are correctly assigned across the Block.
            if len(each_stab.data_qubits) == 2:
                if each_stab.data_qubits == ((0, 0, 0), (0, 1, 0)):
                    self.assertEqual((0, 1, 1), each_stab.ancilla_qubits[0])  # Left
                elif each_stab.data_qubits == ((0, 2, 0), (1, 2, 0)):
                    self.assertEqual((1, 3, 1), each_stab.ancilla_qubits[0])  # Bottom
                elif each_stab.data_qubits == ((1, 0, 0), (2, 0, 0)):
                    self.assertEqual((2, 0, 1), each_stab.ancilla_qubits[0])  # Top
                elif each_stab.data_qubits == ((2, 1, 0), (2, 2, 0)):
                    self.assertEqual((3, 2, 1), each_stab.ancilla_qubits[0])  # Right

            # Check for 4-Body Stabilizers
            elif len(each_stab.data_qubits) == 4:
                if each_stab.data_qubits == (
                    (0, 0, 0),
                    (0, 1, 0),
                    (1, 1, 0),
                    (1, 0, 0),
                ):
                    self.assertEqual((1, 1, 1), each_stab.ancilla_qubits[0])
                elif each_stab.data_qubits == (
                    (0, 1, 0),
                    (0, 2, 0),
                    (1, 2, 0),
                    (1, 1, 0),
                ):
                    self.assertEqual((1, 2, 1), each_stab.ancilla_qubits[0])
                elif each_stab.data_qubits == (
                    (1, 0, 0),
                    (1, 1, 0),
                    (2, 1, 0),
                    (2, 0, 0),
                ):
                    self.assertEqual((2, 1, 1), each_stab.ancilla_qubits[0])
                elif each_stab.data_qubits == (
                    (1, 1, 0),
                    (1, 2, 0),
                    (2, 2, 0),
                    (2, 1, 0),
                ):
                    self.assertEqual((2, 2, 1), each_stab.ancilla_qubits[0])

            else:
                raise ValueError("Stabilizer has an unexpected number of data qubits.")

    def test_rotated_surface_code_size_property(self):
        """Test whether the size property of rotated surface codes correctly returns the
        size of the block."""
        dimensions = [
            # Different repetition codes
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (2, 1),
            (3, 1),
            # Different surface codes
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (7, 7),
            (3, 8),
            (4, 5),
            (5, 4),
        ]
        for dim in dimensions:
            block = RotatedSurfaceCode.create(
                dx=dim[0],
                dz=dim[1],
                lattice=self.lattice_2d_square,
                unique_label="q1",
            )
            self.assertTrue(block.size == dim)

    def test_rotated_surface_code_boundary_qubits(self):
        """Test the `boundary_qubits` function of the rotated surface code block
        class."""
        dx = 5
        dz = 4
        start_x = 3
        start_z = 6
        displaced_block = RotatedSurfaceCode.create(
            dx=dx,
            dz=dz,
            lattice=self.lattice_2d_square,
            unique_label="q1",
            position=(start_x, start_z),
        )
        self.assertEqual(
            set(displaced_block.boundary_qubits("top")),
            set((i, start_z, 0) for i in range(start_x, start_x + dx)),
        )
        self.assertEqual(
            set(displaced_block.boundary_qubits("bottom")),
            set((i, start_z + dz - 1, 0) for i in range(start_x, start_x + dx)),
        )
        self.assertEqual(
            set(displaced_block.boundary_qubits("left")),
            set((start_x, i, 0) for i in range(start_z, start_z + dz)),
        )
        self.assertEqual(
            set(displaced_block.boundary_qubits("right")),
            set((start_x + dx - 1, i, 0) for i in range(start_z, start_z + dz)),
        )

    def test_boundary_stabilizers(self):
        """Test the `boundary_stabilizers` function of the rotated surface code block
        and the property `all_boundary_stabilizers`."""
        dx = 5
        dz = 4
        start_x = 3
        start_z = 6
        displaced_block = RotatedSurfaceCode.create(
            dx=dx,
            dz=dz,
            lattice=self.lattice_2d_square,
            unique_label="q1",
            position=(start_x, start_z),
        )
        self.assertEqual(
            set(displaced_block.boundary_stabilizers("top")),
            set(
                top_stabs := (
                    Stabilizer(
                        "ZZ",
                        data_qubits=((5, 6, 0), (4, 6, 0)),
                        ancilla_qubits=((5, 6, 1),),
                    ),
                    Stabilizer(
                        "ZZ",
                        data_qubits=((7, 6, 0), (6, 6, 0)),
                        ancilla_qubits=((7, 6, 1),),
                    ),
                )
            ),
        )
        self.assertEqual(
            set(displaced_block.boundary_stabilizers("bottom")),
            set(
                bottom_stabs := (
                    Stabilizer(
                        "ZZ",
                        data_qubits=((5, 9, 0), (4, 9, 0)),
                        ancilla_qubits=((5, 10, 1),),
                    ),
                    Stabilizer(
                        "ZZ",
                        data_qubits=((7, 9, 0), (6, 9, 0)),
                        ancilla_qubits=((7, 10, 1),),
                    ),
                )
            ),
        )
        self.assertEqual(
            set(displaced_block.boundary_stabilizers("left")),
            set(
                left_stabs := (
                    Stabilizer(
                        "XX",
                        data_qubits=((3, 6, 0), (3, 7, 0)),
                        ancilla_qubits=((3, 7, 1),),
                    ),
                    Stabilizer(
                        "XX",
                        data_qubits=((3, 8, 0), (3, 9, 0)),
                        ancilla_qubits=((3, 9, 1),),
                    ),
                )
            ),
        )
        self.assertEqual(
            set(displaced_block.boundary_stabilizers("right")),
            set(
                right_stabs := (
                    Stabilizer(
                        "XX",
                        data_qubits=((7, 7, 0), (7, 8, 0)),
                        ancilla_qubits=((8, 8, 1),),
                    ),
                )
            ),
        )

        # Test all_boundary_stabilizers property
        self.assertEqual(
            set(displaced_block.all_boundary_stabilizers),
            set(top_stabs + bottom_stabs + left_stabs + right_stabs),
        )

    def test_rotated_surface_code_bulk_stabilizers(self):
        """Test the `bulk_stabilizers` property of the rotated surface code block
        class."""
        dx = 5
        dz = 4
        start_x = 3
        start_z = 2
        displaced_block = RotatedSurfaceCode.create(
            dx=dx,
            dz=dz,
            lattice=self.lattice_2d_square,
            unique_label="q1",
            position=(start_x, start_z),
        )
        expected_bulk_stabilizers = tuple(
            stab for stab in displaced_block.stabilizers if len(stab.data_qubits) == 4
        )
        self.assertEqual(displaced_block.bulk_stabilizers, expected_bulk_stabilizers)

    def test_rotated_surface_code_boundary_type(self):
        """Test the `boundary_type` function of the rotated surface code block class."""
        dx = 5
        dz = 4
        start_x = 3
        start_z = 6
        block = RotatedSurfaceCode.create(
            dx=dx,
            dz=dz,
            lattice=self.lattice_2d_square,
            unique_label="q1",
            position=(start_x, start_z),
        )
        self.assertEqual(block.boundary_type("top"), "X")
        self.assertEqual(block.boundary_type("bottom"), "X")
        self.assertEqual(block.boundary_type("left"), "Z")
        self.assertEqual(block.boundary_type("right"), "Z")

        # Test that the boundary types do not change if the weight-2 stabilizers start
        # only in the second instead of the first row at the left boundary
        block = RotatedSurfaceCode.create(
            dx=dx,
            dz=dz,
            lattice=self.lattice_2d_square,
            unique_label="q1",
            position=(start_x, start_z),
            weight_2_stab_is_first_row=False,
        )
        self.assertEqual(block.boundary_type("top"), "X")
        self.assertEqual(block.boundary_type("bottom"), "X")
        self.assertEqual(block.boundary_type("left"), "Z")
        self.assertEqual(block.boundary_type("right"), "Z")

        # Test that the boundary types change if the x_boundary is set to "vertical"
        # instead of the default "horizontal"
        block = RotatedSurfaceCode.create(
            dx=dx,
            dz=dz,
            lattice=self.lattice_2d_square,
            unique_label="q1",
            position=(start_x, start_z),
            x_boundary="vertical",
        )
        self.assertEqual(block.boundary_type("top"), "Z")
        self.assertEqual(block.boundary_type("bottom"), "Z")
        self.assertEqual(block.boundary_type("left"), "X")
        self.assertEqual(block.boundary_type("right"), "X")

    def test_rotated_surface_code_get_shifted_equivalent_logical_operator(self):
        """Test the `get_shifted_equivalent_logical_operator` function of the rotated
        surface code block class."""
        block = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice_2d_square,
            unique_label="q1",
        )
        initial_logical_x = block.logical_x_operators[0]
        initial_logical_z = block.logical_z_operators[0]

        # Test that the logical X operator is shifted correctly
        shifted_logical_x, stabilizers_required = (
            block.get_shifted_equivalent_logical_operator(initial_logical_x, (0, 2, 0))
        )

        expected_shifted_logical_x = PauliOperator(
            pauli="XXX", data_qubits=((0, 2, 0), (1, 2, 0), (2, 2, 0))
        )
        expected_x_stabilizers = tuple(
            stab for stab in block.stabilizers if stab.pauli[0] == "X"
        )
        self.assertEqual(shifted_logical_x, expected_shifted_logical_x)
        self.assertEqual(stabilizers_required, expected_x_stabilizers)

        # Test that the logical Z operator is shifted correctly
        shifted_logical_z, stabilizers_required = (
            block.get_shifted_equivalent_logical_operator(initial_logical_z, (2, 0, 0))
        )

        expected_shifted_logical_z = PauliOperator(
            pauli="ZZZ", data_qubits=((2, 0, 0), (2, 1, 0), (2, 2, 0))
        )
        expected_z_stabilizers = tuple(
            stab for stab in block.stabilizers if stab.pauli[0] == "Z"
        )
        self.assertEqual(shifted_logical_z, expected_shifted_logical_z)
        self.assertEqual(stabilizers_required, expected_z_stabilizers)

        # Test that the behaviour is correct for the same position
        shifted_logical_x, stabilizers_required = (
            block.get_shifted_equivalent_logical_operator(initial_logical_x, (0, 0, 0))
        )
        self.assertEqual(shifted_logical_x, initial_logical_x)
        self.assertEqual(stabilizers_required, ())

        # Test for a bigger block
        big_block = RotatedSurfaceCode.create(
            dx=7,
            dz=3,
            lattice=self.lattice_2d_square,
            unique_label="q1",
        )
        initial_logical_z = big_block.logical_z_operators[0]
        # Shift the logical Z operator to the right by 4
        shifted_logical_z, stabilizers_required = (
            big_block.get_shifted_equivalent_logical_operator(
                initial_logical_z, (4, 0, 0)
            )
        )
        expected_shifted_logical_z = PauliOperator(
            pauli="ZZZ", data_qubits=((4, 0, 0), (4, 1, 0), (4, 2, 0))
        )
        expected_z_stabilizers = tuple(
            stab
            for stab in big_block.stabilizers
            if stab.pauli[0] == "Z" and all(q[0] <= 4 for q in stab.data_qubits)
        )
        self.assertEqual(shifted_logical_z, expected_shifted_logical_z)
        self.assertEqual(stabilizers_required, expected_z_stabilizers)

    def test_rotated_surface_code_topological_corners(self):
        """Test the topological_corners property of the RotatedSurfaceCode class."""

        # Standard case: 3x3 rotated surface code
        expected_standard_corners = {
            (0, 0, 0),
            (2, 0, 0),
            (0, 2, 0),
            (2, 2, 0),
        }
        self.assertEqual(set(self.rsc.topological_corners), expected_standard_corners)

        # Test for a more complex block
        grown_block = RotatedSurfaceCode.create(
            dx=5,
            dz=9,
            lattice=self.lattice_2d_square,
            unique_label="q1",
            weight_2_stab_is_first_row=False,
            x_boundary=Orientation.VERTICAL,
        )
        rotated_stabs = (
            grown_block.bulk_stabilizers
            + grown_block.boundary_stabilizers(Direction.TOP)
            + grown_block.boundary_stabilizers(Direction.LEFT)
            + tuple(
                [  # Add the new right boundary stabs
                    Stabilizer(
                        "ZZ", ((4, 0, 0), (4, 1, 0)), ancilla_qubits=((5, 1, 1),)
                    ),
                    Stabilizer(
                        "ZZ", ((4, 2, 0), (4, 3, 0)), ancilla_qubits=((5, 3, 1),)
                    ),
                    Stabilizer(
                        "XX", ((4, 5, 0), (4, 6, 0)), ancilla_qubits=((5, 6, 1),)
                    ),
                    Stabilizer(
                        "XX", ((4, 7, 0), (4, 8, 0)), ancilla_qubits=((5, 8, 1),)
                    ),
                ]
            )
            + tuple(
                [  # Add the new bottom boundary stabs
                    Stabilizer(
                        "ZZ", ((1, 8, 0), (0, 8, 0)), ancilla_qubits=((1, 9, 1),)
                    ),
                    Stabilizer(
                        "ZZ", ((3, 8, 0), (2, 8, 0)), ancilla_qubits=((3, 9, 1),)
                    ),
                ]
            )
        )
        z_op = PauliOperator(
            "ZZZZZ", ((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0))
        )
        x_op = PauliOperator(
            "XXXXYZZZZ",
            (
                (4, 0, 0),
                (4, 1, 0),
                (4, 2, 0),
                (4, 3, 0),
                (4, 4, 0),
                (3, 4, 0),
                (2, 4, 0),
                (1, 4, 0),
                (0, 4, 0),
            ),
        )
        twisted_rsc_block = RotatedSurfaceCode(
            stabilizers=rotated_stabs,
            logical_x_operators=[x_op],
            logical_z_operators=[z_op],
            unique_label="twist",
        )

        expected_twisted_corners = {
            (0, 0, 0),
            (4, 0, 0),
            (4, 4, 0),
            (4, 8, 0),
        }
        self.assertEqual(
            set(twisted_rsc_block.topological_corners), expected_twisted_corners
        )

    def test_rotated_surface_code_find_padding(self):
        """Test the `find_padding` function of the rotated surface code block class."""

        test_args_and_expected = [
            ({"boundary": Direction.TOP, "schedule": FourBodySchedule.N}, (0, 2)),
            ({"boundary": Direction.TOP, "schedule": FourBodySchedule.Z}, (0, 1)),
            ({"boundary": Direction.RIGHT, "schedule": FourBodySchedule.N}, (0, 1)),
            ({"boundary": Direction.RIGHT, "schedule": FourBodySchedule.Z}, (0, 2)),
            ({"boundary": Direction.BOTTOM, "schedule": FourBodySchedule.N}, (1, 3)),
            ({"boundary": Direction.BOTTOM, "schedule": FourBodySchedule.Z}, (2, 3)),
            ({"boundary": Direction.LEFT, "schedule": FourBodySchedule.N}, (2, 3)),
            ({"boundary": Direction.LEFT, "schedule": FourBodySchedule.Z}, (1, 3)),
        ]

        for args, expected in test_args_and_expected:
            output = RotatedSurfaceCode.find_padding(**args)
            self.assertEqual(output, expected)

    def test_rotated_surface_code_generate_syndrome_circuits(self):
        """Test the `generate_syndrome_circuits` function of the rotated surface code
        block class."""

        d_channels = [Channel(label=f"d{i}") for i in range(4)]
        a_channels = [Channel(label=f"a{i}", type="quantum") for i in range(1)]
        c_channels = [Channel(label=f"c{i}", type="classical") for i in range(1)]
        test_args_and_expected = [
            (
                {"pauli": "ZZZZ", "padding": (), "name": "bulk-zzzz"},
                SyndromeCircuit(
                    pauli="ZZZZ",
                    name="bulk-zzzz",
                    circuit=Circuit(
                        name="bulk-zzzz",
                        circuit=[
                            [Circuit("Reset_0", channels=a_channels)],
                            [Circuit("H", channels=a_channels)],
                            [Circuit("CZ", channels=[a_channels[0], d_channels[0]])],
                            [Circuit("CZ", channels=[a_channels[0], d_channels[1]])],
                            [Circuit("CZ", channels=[a_channels[0], d_channels[2]])],
                            [Circuit("CZ", channels=[a_channels[0], d_channels[3]])],
                            [Circuit("H", channels=a_channels)],
                            [Circuit("Measurement", channels=a_channels + c_channels)],
                        ],
                    ),
                ),
            ),
            (
                {"pauli": "XX", "padding": (1, 3), "name": "left-xx"},
                SyndromeCircuit(
                    pauli="XX",
                    name="left-xx",
                    circuit=Circuit(
                        name="left-xx",
                        circuit=[
                            [Circuit("Reset_0", channels=a_channels)],
                            [Circuit("H", channels=a_channels)],
                            [Circuit("CX", channels=[a_channels[0], d_channels[0]])],
                            [],
                            [Circuit("CX", channels=[a_channels[0], d_channels[1]])],
                            [],
                            [Circuit("H", channels=a_channels)],
                            [Circuit("Measurement", channels=a_channels + c_channels)],
                        ],
                    ),
                ),
            ),
            (
                {"pauli": "XYZ", "padding": (2,), "name": "fancy-xyz"},
                SyndromeCircuit(
                    pauli="XYZ",
                    name="fancy-xyz",
                    circuit=Circuit(
                        name="fancy-xyz",
                        circuit=[
                            [Circuit("Reset_0", channels=a_channels)],
                            [Circuit("H", channels=a_channels)],
                            [Circuit("CX", channels=[a_channels[0], d_channels[0]])],
                            [Circuit("CY", channels=[a_channels[0], d_channels[1]])],
                            [],
                            [Circuit("CZ", channels=[a_channels[0], d_channels[2]])],
                            [Circuit("H", channels=a_channels)],
                            [Circuit("Measurement", channels=a_channels + c_channels)],
                        ],
                    ),
                ),
            ),
        ]

        for args, expected in test_args_and_expected:
            syndrome_circuit = RotatedSurfaceCode.generate_syndrome_circuit(**args)
            self.assertEqual(syndrome_circuit, expected)


if __name__ == "__main__":
    unittest.main()
