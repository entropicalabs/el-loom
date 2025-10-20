"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=duplicate-code, too-many-lines
from copy import deepcopy
import unittest

from loom.eka import (
    Circuit,
    Channel,
    Lattice,
    Stabilizer,
    PauliOperator,
    SyndromeCircuit,
)
from loom.eka.utilities import Direction, Orientation, DiagonalDirection
from loom.interpreter import InterpretationStep, Syndrome

from loom_rotated_surface_code.code_factory import RotatedSurfaceCode
from loom_rotated_surface_code.applicator.move_corners import (
    move_corner,
    move_corners,
    get_associated_boundaries,
    find_new_boundary_direction,
    is_2_body_included,
    cut_corner_stabilizer,
    cut_corner_syndrome_circuit,
    generate_updated_2_body_stabilizers,
    find_new_boundary_stabilizers,
    move_corner_logical_operators,
    move_corner_circuit,
)
from loom_rotated_surface_code.applicator.utilities import (
    generate_syndrome_extraction_circuits,
)


class TestRotatedSurfaceCodeMoveCorners(unittest.TestCase):
    """
    Tests the move_corners function and its associated functions.
    """

    def setUp(self):
        self.square_2d_lattice = Lattice.square_2d((10, 20))
        self.rot_surf_code_1 = RotatedSurfaceCode.create(
            dx=3,
            dz=5,
            lattice=self.square_2d_lattice,
            unique_label="q1",
            weight_2_stab_is_first_row=True,
        )
        self.base_step = InterpretationStep(
            block_history=((self.rot_surf_code_1,),),
            syndromes=tuple(
                Syndrome(
                    stabilizer=stabilizer.uuid,
                    measurements=((f"c_{stabilizer.ancilla_qubits[0]}", 0),),
                    block=self.rot_surf_code_1.uuid,
                    round=0,
                    corrections=(),
                )
                for stabilizer in self.rot_surf_code_1.stabilizers
            ),
        )
        self.grown_block = RotatedSurfaceCode.create(
            dx=5,
            dz=9,
            lattice=self.square_2d_lattice,
            unique_label="q1",
            weight_2_stab_is_first_row=False,
            x_boundary=Orientation.VERTICAL,
        )
        self.rotated_stabs = (
            self.grown_block.bulk_stabilizers
            + self.grown_block.boundary_stabilizers(Direction.TOP)
            + self.grown_block.boundary_stabilizers(Direction.LEFT)
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
        # pylint: enable=duplicate-code
        self.twisted_rsc_block = RotatedSurfaceCode(
            stabilizers=self.rotated_stabs,
            logical_x_operators=[x_op],
            logical_z_operators=[z_op],
            unique_label="twist",
        )

        # For reference, the twisted block is the following:
        #
        #            X                   X
        #    *(0,0) --- (1,0) --- (2,0) --- (3,0) --- (4,0)*
        #       |         |         |         |         |
        #       |    Z    |    X    |    Z    |    X    |  Z
        #       |         |         |         |         |
        #     (0,1) --- (1,1) --- (2,1) --- (3,1) --- (4,1)
        #       |         |         |         |         |
        #    Z  |    X    |    Z    |    X    |    Z    |
        #       |         |         |         |         |
        #     (0,2) --- (1,2) --- (2,2) --- (3,2) --- (4,2)
        #       |         |         |         |         |
        #       |    Z    |    X    |    Z    |    X    |  Z
        #       |         |         |         |         |
        #     (0,3) --- (1,3) --- (2,3) --- (3,3) --- (4,3)
        #       |         |         |         |         |
        #    Z  |    X    |    Z    |    X    |    Z    |
        #       |         |         |         |         |
        #     (0,4) --- (1,4) --- (2,4) --- (3,4) --- (4,4)*
        #       |         |         |         |         |
        #       |    Z    |    X    |    Z    |    X    |
        #       |         |         |         |         |
        #     (0,5) --- (1,5) --- (2,5) --- (3,5) --- (4,5)
        #       |         |         |         |         |
        #    Z  |    X    |    Z    |    X    |    Z    |  X
        #       |         |         |         |         |
        #     (0,6) --- (1,6) --- (2,6) --- (3,6) --- (4,6)
        #       |         |         |         |         |
        #       |    Z    |    X    |    Z    |    X    |
        #       |         |         |         |         |
        #     (0,7) --- (1,7) --- (2,7) --- (3,7) --- (4,7)
        #       |         |         |         |         |
        #    Z  |    X    |    Z    |    X    |    Z    |  X
        #       |         |         |         |         |
        #     (0,8) --- (1,8) --- (2,8) --- (3,8) --- (4,8)*
        #            Z                   Z
        #
        # The topological corners are the starred qubits

    def test_get_associated_boundaries(self):
        """Tests the get_corner_qubit function."""
        # Test a standard rsc block
        expected_corners_direction = [
            ((0, 0, 0), {"top", "left"}),
            ((0, 4, 0), {"bottom", "left"}),
            ((2, 0, 0), {"right", "top"}),
            ((2, 4, 0), {"bottom", "right"}),
        ]
        for corner_qubit, expected_directions in expected_corners_direction:
            corner_directions = get_associated_boundaries(
                self.rot_surf_code_1, corner_qubit
            )
            self.assertEqual(set(corner_directions), expected_directions)

        # Test a non standard rsc block (used for the S gate)
        expected_corners_direction = [
            ((0, 0, 0), {"top", "left"}),
            ((4, 0, 0), {"right", "top"}),
            ((4, 4, 0), {"right"}),
            ((4, 8, 0), {"bottom", "right"}),
        ]
        for corner_qubit, expected_directions in expected_corners_direction:
            corner_directions = get_associated_boundaries(
                self.twisted_rsc_block, corner_qubit
            )
            self.assertEqual(set(corner_directions), expected_directions)

        # Test that the right errror is thrown if the corner is not a topological corner
        with self.assertRaises(ValueError) as cm:
            _ = get_associated_boundaries(self.twisted_rsc_block, (1, 1, 0))
        err_msg = (
            "The selected corner qubit (1, 1, 0) is not a topological corner of"
            + " the block `twist`"
        )
        self.assertIn(err_msg, str(cm.exception))

    def test_find_new_boundary_direction(self):
        """
        Tests the find_new_boundary_direction function.
        """
        # Test generic cases
        args_expected_outcome = [
            (((Direction.TOP, Direction.LEFT), Direction.RIGHT), Direction.TOP),
            (((Direction.TOP, Direction.LEFT), Direction.BOTTOM), Direction.LEFT),
            (((Direction.TOP, Direction.RIGHT), Direction.LEFT), Direction.TOP),
            (((Direction.TOP, Direction.RIGHT), Direction.BOTTOM), Direction.RIGHT),
            (((Direction.BOTTOM, Direction.LEFT), Direction.RIGHT), Direction.BOTTOM),
            (((Direction.BOTTOM, Direction.LEFT), Direction.TOP), Direction.LEFT),
            (((Direction.BOTTOM, Direction.RIGHT), Direction.LEFT), Direction.BOTTOM),
            (((Direction.BOTTOM, Direction.RIGHT), Direction.TOP), Direction.RIGHT),
        ]
        for args, expected_outcome in args_expected_outcome:
            outcome = find_new_boundary_direction(*args)
            self.assertEqual(outcome, expected_outcome)

        # Test that an error is raised if there are too many directions
        with self.assertRaises(ValueError) as cm:
            _ = find_new_boundary_direction(
                (Direction.TOP, Direction.LEFT, Direction.RIGHT), Direction.BOTTOM
            )
        err_msg = "Invalid number of corner boundaries: 3, must be either 1 or 2"
        self.assertIn(err_msg, str(cm.exception))

        # Test that an error is raised if the directions are not orthogonal
        with self.assertRaises(ValueError) as cm:
            _ = find_new_boundary_direction(
                (Direction.TOP, Direction.BOTTOM), Direction.TOP
            )
        err_msg = (
            "Invalid corner boundaries: (<Direction.TOP: 'top'>, <Direction.BOTTOM: "
            + "'bottom'>), they must be orthogonal, e.g. (TOP, LEFT) or a single "
            + "direction."
        )
        self.assertIn(err_msg, str(cm.exception))

    def test_is_2_body_included(self):
        """
        Tests the is_2_body_included function.
        """
        # Test generic cases
        args_expected_outcome = [
            (((0, 0, 0), Direction.LEFT), True),
            (((0, 0, 0), Direction.TOP), False),
            (((0, 4, 0), Direction.BOTTOM), True),
            (((0, 4, 0), Direction.LEFT), False),
            (((2, 0, 0), Direction.TOP), True),
            (((2, 0, 0), Direction.RIGHT), False),
            (((2, 4, 0), Direction.BOTTOM), False),
            (((2, 4, 0), Direction.RIGHT), True),
        ]
        for args, expected_outcome in args_expected_outcome:
            outcome = is_2_body_included(self.rot_surf_code_1, *args)
            self.assertEqual(outcome, expected_outcome)

        # Test the for the twisted block
        args_expected_outcome = [
            (((0, 0, 0), Direction.TOP), True),
            (((0, 0, 0), Direction.LEFT), False),
            (((4, 0, 0), Direction.TOP), False),
            (((4, 0, 0), Direction.RIGHT), True),
            (
                ((4, 4, 0), Direction.RIGHT),
                False,
            ),  # Can be moved both to the top and bottom with no 2body stab
            (((4, 8, 0), Direction.RIGHT), True),
            (((4, 8, 0), Direction.BOTTOM), False),
        ]
        for args, expected_outcome in args_expected_outcome:
            outcome = is_2_body_included(self.twisted_rsc_block, *args)
            self.assertEqual(outcome, expected_outcome)

        # Test that an error is raised if direction is invalid
        with self.assertRaises(ValueError) as cm:
            _ = is_2_body_included(self.rot_surf_code_1, (0, 0, 0), Direction.BOTTOM)
        err_msg = "The corner qubit (0, 0, 0) is not part of the bottom boundary"
        self.assertIn(err_msg, str(cm.exception))

    def test_cut_corner_stabilizer(self):
        """
        Test the cut_corner_stabilizer function.
        """
        # upper left corner
        expected_stab = Stabilizer(
            pauli="ZZZ",
            data_qubits=[(1, 0, 0), (1, 1, 0), (0, 1, 0)],
            ancilla_qubits=[(1, 1, 1)],
        )
        cut_stab = cut_corner_stabilizer(self.rot_surf_code_1, (0, 0, 0))
        self.assertEqual(cut_stab, expected_stab)

        # upper right corner
        expected_stab = Stabilizer(
            pauli="XXX",
            data_qubits=[(2, 1, 0), (1, 0, 0), (1, 1, 0)],
            ancilla_qubits=[(2, 1, 1)],
        )
        cut_stab = cut_corner_stabilizer(self.rot_surf_code_1, (2, 0, 0))
        self.assertEqual(cut_stab, expected_stab)

        # lower left corner
        expected_stab = Stabilizer(
            pauli="XXX",
            data_qubits=[(1, 3, 0), (1, 4, 0), (0, 3, 0)],
            ancilla_qubits=[(1, 4, 1)],
        )
        cut_stab = cut_corner_stabilizer(self.rot_surf_code_1, (0, 4, 0))
        self.assertEqual(cut_stab, expected_stab)

        # lower right corner
        expected_stab = Stabilizer(
            pauli="ZZZ",
            data_qubits=[(2, 3, 0), (1, 3, 0), (1, 4, 0)],
            ancilla_qubits=[(2, 4, 1)],
        )
        cut_stab = cut_corner_stabilizer(self.rot_surf_code_1, (2, 4, 0))
        self.assertEqual(cut_stab, expected_stab)

    def test_cut_corner_syndrome_circuit(self):
        """
        Test the cut_corner_syndrome_circuit function.
        """

        # Select the bottom left corner to remove
        corner_qubit = (0, 4, 0)
        stab_cut = Stabilizer(
            pauli="XXX",
            data_qubits=[(1, 3, 0), (0, 3, 0), (1, 4, 0)],
            ancilla_qubits=[(1, 4, 1)],
        )
        output_syndrome_circuit = cut_corner_syndrome_circuit(
            block=self.rot_surf_code_1,
            cut_stabilizer=stab_cut,
            corner_qubit=corner_qubit,
            which_corner=(Direction.LEFT, Direction.BOTTOM),
        )
        q_channels = [Channel("quantum") for _ in range(3)]
        a_channels = [Channel("quantum")]
        c_channels = [Channel("classical")]
        expected_circuit = Circuit(
            name="bottom-left-XXX",
            circuit=[
                [Circuit("Reset_0", channels=a_channels)],
                [Circuit("H", channels=a_channels)],
                [Circuit("CX", channels=[a_channels[0], q_channels[0]])],
                [Circuit("CX", channels=[a_channels[0], q_channels[1]])],
                [Circuit("CX", channels=[a_channels[0], q_channels[2]])],
                [],
                [Circuit("H", channels=a_channels)],
                [Circuit("Measurement", channels=a_channels + c_channels)],
            ],
        )
        expected_syndrome_circuit = SyndromeCircuit(
            pauli="XXX",
            name="bottom-left-XXX",
            circuit=expected_circuit,
        )
        self.assertEqual(output_syndrome_circuit, expected_syndrome_circuit)

    def test_generate_updated_2_body_stabilizers(self):
        """
        Test the generate_updated_2_body_stabilizers function.
        """
        # For reference we will use a 5x5 rotated surface code
        # Test all combinations of corners and move_directions
        test_args_and_expected = [
            (
                {  # Set of parameters to move the top left corner down
                    "old_corner_qubit": (0, 0, 0),
                    "new_boundary_direction": Direction.LEFT,
                    "unit_vector": (0, 1),
                    "how_far": 3,  # Odd, the corner is cut
                    "pauli_type": "Z",  # We extend the Z stabilizer boundary
                },
                [Stabilizer("ZZ", [(0, 1, 0), (0, 2, 0)], ancilla_qubits=[(0, 2, 1)])],
            ),
            (
                {  # Set of parameters to move the top left corner to the right
                    "old_corner_qubit": (0, 0, 0),
                    "new_boundary_direction": Direction.TOP,
                    "unit_vector": (1, 0),
                    "how_far": 2,
                    "pauli_type": "X",  # We extend the X stabilizer boundary
                },
                [Stabilizer("XX", [(1, 0, 0), (0, 0, 0)], ancilla_qubits=[(1, 0, 1)])],
            ),
            (
                {  # Set of parameters to move the bottom left corner up
                    "old_corner_qubit": (0, 4, 0),
                    "new_boundary_direction": Direction.LEFT,
                    "unit_vector": (0, -1),
                    "how_far": 2,
                    "pauli_type": "Z",  # We extend the Z stabilizer boundary
                },
                [Stabilizer("ZZ", [(0, 3, 0), (0, 4, 0)], ancilla_qubits=[(0, 4, 1)])],
            ),
            (
                {  # Set of parameters to move the bottom left corner to the right
                    "old_corner_qubit": (0, 4, 0),
                    "new_boundary_direction": Direction.BOTTOM,
                    "unit_vector": (0, 1),
                    "how_far": 1,
                    "pauli_type": "X",  # We extend the X stabilizer boundary
                },
                [],
            ),
            (
                {  # Set of parameters to move the top right corner to the left
                    "old_corner_qubit": (4, 0, 0),
                    "new_boundary_direction": Direction.TOP,
                    "unit_vector": (-1, 0),
                    "how_far": 3,
                    "pauli_type": "X",  # We extend the X stabilizer boundary
                },
                [Stabilizer("XX", [(3, 0, 0), (2, 0, 0)], ancilla_qubits=[(3, 0, 1)])],
            ),
            (
                {  # Set of parameters to move the top right corner down
                    "old_corner_qubit": (4, 0, 0),
                    "new_boundary_direction": Direction.RIGHT,
                    "unit_vector": (0, 1),
                    "how_far": 2,
                    "pauli_type": "Z",  # We extend the Z stabilizer boundary
                },
                [Stabilizer("ZZ", [(4, 0, 0), (4, 1, 0)], ancilla_qubits=[(5, 1, 1)])],
            ),
            (
                {  # Set of parameters to move the bottom right corner to the left
                    "old_corner_qubit": (4, 4, 0),
                    "new_boundary_direction": Direction.BOTTOM,
                    "unit_vector": (-1, 0),
                    "how_far": 2,
                    "pauli_type": "X",  # We extend the X stabilizer boundary
                },
                [Stabilizer("XX", [(4, 4, 0), (3, 4, 0)], ancilla_qubits=[(4, 5, 1)])],
            ),
            (
                {  # Set of parameters to move the bottom right corner up
                    "old_corner_qubit": (4, 4, 0),
                    "new_boundary_direction": Direction.RIGHT,
                    "unit_vector": (0, -1),
                    "how_far": 3,
                    "pauli_type": "Z",  # We extend the Z stabilizer boundary
                },
                [Stabilizer("ZZ", [(4, 2, 0), (4, 3, 0)], ancilla_qubits=[(5, 3, 1)])],
            ),
        ]
        for args, expected_output in test_args_and_expected:
            output = generate_updated_2_body_stabilizers(**args)
            self.assertEqual(output, expected_output)

    def test_find_new_boundary_stabilizers(self):
        """
        Test the find_new_boundary_stabilizers function.
        """
        # We use a 5x5 rotated surface code
        big_block = RotatedSurfaceCode.create(
            dx=5,
            dz=5,
            lattice=self.square_2d_lattice,
            unique_label="q1",
        )

        test_args_and_expected = [
            (
                {  # Set of parameters to move the top left corner down
                    "corner_qubit": (0, 0, 0),
                    "modified_boundary_direction": Direction.LEFT,
                    "unit_vector": (0, 1),
                    "how_far": 3,  # Odd because the corner is cut
                    "two_body_is_included": True,
                },
                (
                    [
                        stab
                        for stab in big_block.all_boundary_stabilizers
                        if not all(
                            q in [(0, 0, 0), (0, 2, 0), (0, 1, 0), (0, 3, 0)]
                            for q in stab.data_qubits
                        )
                    ],
                    [
                        Stabilizer(
                            "ZZ", [(0, 1, 0), (0, 2, 0)], ancilla_qubits=[(0, 2, 1)]
                        )
                    ],
                    Stabilizer(
                        "ZZZ",
                        [(1, 0, 0), (1, 1, 0), (0, 1, 0)],
                        ancilla_qubits=[(1, 1, 1)],
                    ),
                ),
            ),
            (
                {  # Set of parameters to move the top left corner to the right
                    "corner_qubit": (0, 0, 0),
                    "modified_boundary_direction": Direction.TOP,
                    "unit_vector": (1, 0),
                    "how_far": 2,  # Even because the corner is not cut
                    "two_body_is_included": False,
                },
                (
                    [
                        stab
                        for stab in big_block.all_boundary_stabilizers
                        if not all(
                            q in [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
                            for q in stab.data_qubits
                        )
                    ],
                    [
                        Stabilizer(
                            "XX", [(1, 0, 0), (0, 0, 0)], ancilla_qubits=[(1, 0, 1)]
                        )
                    ],
                    None,
                ),
            ),
            (
                {  # Set of parameters to move the top right corner to the left
                    "corner_qubit": (4, 0, 0),
                    "modified_boundary_direction": Direction.TOP,
                    "unit_vector": (-1, 0),
                    "how_far": 1,  # Odd because the corner is cut
                    "two_body_is_included": True,
                },
                (
                    [
                        stab
                        for stab in big_block.all_boundary_stabilizers
                        if not all(
                            q in [(4, 0, 0), (3, 0, 0)] for q in stab.data_qubits
                        )
                    ],
                    [],
                    Stabilizer(
                        "XXX",
                        [(4, 1, 0), (3, 0, 0), (3, 1, 0)],
                        ancilla_qubits=[(4, 1, 1)],
                    ),
                ),
            ),
            (
                {  # Set of parameters to move the top right corner down
                    "corner_qubit": (4, 0, 0),
                    "modified_boundary_direction": Direction.RIGHT,
                    "unit_vector": (0, 1),
                    "how_far": 2,  # Even because the corner is not cut
                    "two_body_is_included": False,
                },
                (
                    [
                        stab
                        for stab in big_block.all_boundary_stabilizers
                        if not all(
                            q in [(4, 0, 0), (4, 1, 0), (4, 2, 0)]
                            for q in stab.data_qubits
                        )
                    ],
                    [
                        Stabilizer(
                            "ZZ", [(4, 0, 0), (4, 1, 0)], ancilla_qubits=[(5, 1, 1)]
                        )
                    ],
                    None,
                ),
            ),
            (
                {  # Set of parameters to move the bottom left corner up
                    "corner_qubit": (0, 4, 0),
                    "modified_boundary_direction": Direction.LEFT,
                    "unit_vector": (0, -1),
                    "how_far": 2,  # Even because the corner is not cut
                    "two_body_is_included": False,
                },
                (
                    [
                        stab
                        for stab in big_block.all_boundary_stabilizers
                        if not all(
                            q in [(0, 4, 0), (0, 3, 0), (0, 2, 0)]
                            for q in stab.data_qubits
                        )
                    ],
                    [
                        Stabilizer(
                            "ZZ", [(0, 3, 0), (0, 4, 0)], ancilla_qubits=[(0, 4, 1)]
                        )
                    ],
                    None,
                ),
            ),
            (
                {  # Set of parameters to move the bottom left corner to the right
                    "corner_qubit": (0, 4, 0),
                    "modified_boundary_direction": Direction.BOTTOM,
                    "unit_vector": (1, 0),
                    "how_far": 3,  # Odd because the corner is cut
                    "two_body_is_included": True,
                },
                (
                    [
                        stab
                        for stab in big_block.all_boundary_stabilizers
                        if not all(
                            q in [(0, 4, 0), (2, 4, 0), (1, 4, 0), (3, 4, 0)]
                            for q in stab.data_qubits
                        )
                    ],
                    [
                        Stabilizer(
                            "XX", [(2, 4, 0), (1, 4, 0)], ancilla_qubits=[(2, 5, 1)]
                        )
                    ],
                    Stabilizer(
                        "XXX",
                        [(1, 3, 0), (1, 4, 0), (0, 3, 0)],
                        ancilla_qubits=[(1, 4, 1)],
                    ),
                ),
            ),
            (
                {  # Set of parameters to move the bottom right corner up
                    "corner_qubit": (4, 4, 0),
                    "modified_boundary_direction": Direction.RIGHT,
                    "unit_vector": (0, -1),
                    "how_far": 1,  # Odd because the corner is cut
                    "two_body_is_included": True,
                },
                (
                    [
                        stab
                        for stab in big_block.all_boundary_stabilizers
                        if not all(
                            q in [(4, 4, 0), (4, 3, 0)] for q in stab.data_qubits
                        )
                    ],
                    [],
                    Stabilizer(
                        "ZZZ",
                        [(4, 3, 0), (3, 3, 0), (3, 4, 0)],
                        ancilla_qubits=[(4, 4, 1)],
                    ),
                ),
            ),
            (
                {  # Set of parameters to move the bottom right corner left
                    "corner_qubit": (4, 4, 0),
                    "modified_boundary_direction": Direction.BOTTOM,
                    "unit_vector": (-1, 0),
                    "how_far": 2,  # Even because the corner is not cut
                    "two_body_is_included": False,
                },
                (
                    [
                        stab
                        for stab in big_block.all_boundary_stabilizers
                        if not all(
                            q in [(4, 4, 0), (3, 4, 0), (2, 4, 0)]
                            for q in stab.data_qubits
                        )
                    ],
                    [
                        Stabilizer(
                            "XX", [(4, 4, 0), (3, 4, 0)], ancilla_qubits=[(4, 5, 1)]
                        )
                    ],
                    None,
                ),
            ),
        ]
        for args, expected_output in test_args_and_expected:
            kept_stabs, new_stabs, cut_stab, stab_evol = find_new_boundary_stabilizers(
                block=big_block, **args
            )
            output = (kept_stabs, new_stabs, cut_stab)
            self.assertEqual(output, expected_output)
            if isinstance(cut_stab, Stabilizer):  # Need to generate the evolution dict
                stab_to_cut = next(
                    stab
                    for stab in big_block.stabilizers
                    if len(stab.pauli) == 4 and args["corner_qubit"] in stab.data_qubits
                )
                self.assertEqual(stab_evol, {cut_stab.uuid: (stab_to_cut.uuid,)})
            else:  # Test for empty evolution dict
                self.assertFalse(args["two_body_is_included"])
                self.assertEqual(cut_stab, None)
                self.assertEqual(stab_evol, {})

    def test_move_corner_logical_operators(self):  # pylint: disable=too-many-locals
        """
        Test the move_corners_logical_operators function.
        """
        # For reference we will use a 5x5 rotated surface code with standard logical ops
        big_block = RotatedSurfaceCode.create(
            dx=5,
            dz=5,
            lattice=self.square_2d_lattice,
            unique_label="q5x5",
        )

        test_args_and_expected = [
            (
                {  # Set of parameters to move the top left corner down :
                    # case 3 for X, case 2 for Z
                    "corner_qubit": (0, 0, 0),
                    "unit_vector": (0, 1),
                    "how_far": 3,  # Odd because the corner is cut
                    "old_stabs_to_be_removed": (
                        old_stabs_to_be_removed := [
                            Stabilizer(
                                "XX", ((0, 0, 0), (0, 1, 0)), ancilla_qubits=[(0, 1, 1)]
                            ),
                            Stabilizer(
                                "XX", ((0, 2, 0), (0, 3, 0)), ancilla_qubits=[(0, 3, 1)]
                            ),
                        ]
                    ),  # we extend the top boundary to the left
                    "new_stabs_to_be_added": (
                        new_stabs_to_be_added := [
                            Stabilizer(
                                "ZZ", ((0, 1, 0), (0, 2, 0)), ancilla_qubits=[(0, 2, 1)]
                            ),
                        ]
                    ),
                },
                (
                    PauliOperator(
                        "XXXXXXX",
                        (
                            (1, 0, 0),
                            (2, 0, 0),
                            (3, 0, 0),
                            (4, 0, 0),
                            (0, 1, 0),
                            (0, 2, 0),
                            (0, 3, 0),
                        ),
                    ),
                    PauliOperator("ZZ", ((0, 3, 0), (0, 4, 0))),
                    # Required stabilizers to modify the operator
                    old_stabs_to_be_removed,
                    # Required stabilizers to modify the operator
                    new_stabs_to_be_added,
                ),
            ),
            (
                {  # Set of parameters to move the top left corner to the right:
                    # case 2 for X, case 3 for Z
                    "corner_qubit": (0, 0, 0),
                    "unit_vector": (1, 0),
                    "how_far": 2,  # Even because the corner is not cut
                    "old_stabs_to_be_removed": (
                        old_stabs_to_be_removed := [
                            Stabilizer(
                                "ZZ", ((2, 0, 0), (1, 0, 0)), ancilla_qubits=[(2, 0, 1)]
                            ),
                        ]
                    ),  # we extend the left boundary to the top
                    "new_stabs_to_be_added": (
                        new_stabs_to_be_added := [
                            Stabilizer(
                                "XX", ((1, 0, 0), (0, 0, 0)), ancilla_qubits=[(1, 0, 1)]
                            ),
                        ]
                    ),
                },
                (
                    PauliOperator("XXX", ((2, 0, 0), (3, 0, 0), (4, 0, 0))),
                    PauliOperator(
                        "ZZZZZZZ",
                        (
                            (0, 0, 0),
                            (0, 1, 0),
                            (0, 2, 0),
                            (0, 3, 0),
                            (0, 4, 0),
                            (1, 0, 0),
                            (2, 0, 0),
                        ),
                    ),
                    # Required stabilizers to modify the operator
                    new_stabs_to_be_added,
                    # Required stabilizers to modify the operator
                    old_stabs_to_be_removed,
                ),
            ),
            (
                {  # Set of parameters to move the top right corner to the left:
                    # case 2 for X, case 1 for Z
                    "corner_qubit": (4, 0, 0),
                    "unit_vector": (-1, 0),
                    "how_far": 1,  # Odd because the corner is cut
                    "old_stabs_to_be_removed": (
                        old_stabs_to_be_removed := [
                            Stabilizer(
                                "ZZ", ((4, 0, 0), (3, 0, 0)), ancilla_qubits=[(3, 0, 1)]
                            ),
                        ]
                    ),  # we extend the right boundary to the top
                    "new_stabs_to_be_added": [],
                },
                (
                    PauliOperator("XXXX", ((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0))),
                    PauliOperator(
                        "ZZZZZ", ((0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0), (0, 4, 0))
                    ),
                    tuple(),
                    tuple(),
                ),
            ),
            (
                {  # Set of parameters to move the top right corner down:
                    # case 3 for X, case 1 for Z
                    "corner_qubit": (4, 0, 0),
                    "unit_vector": (0, 1),
                    "how_far": 2,  # Even because the corner is not cut
                    "old_stabs_to_be_removed": (
                        old_stabs_to_be_removed := [
                            Stabilizer(
                                "XX", ((4, 1, 0), (4, 2, 0)), ancilla_qubits=[(5, 2, 1)]
                            )
                        ]
                    ),  # we extend the top boundary to the right
                    "new_stabs_to_be_added": [
                        Stabilizer(
                            "ZZ", ((4, 0, 0), (4, 1, 0)), ancilla_qubits=[(5, 1, 1)]
                        )
                    ],
                },
                (
                    PauliOperator(
                        "XXXXXXX",
                        (
                            (0, 0, 0),
                            (1, 0, 0),
                            (2, 0, 0),
                            (3, 0, 0),
                            (4, 0, 0),
                            (4, 1, 0),
                            (4, 2, 0),
                        ),
                    ),
                    PauliOperator(
                        "ZZZZZ", ((0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0), (0, 4, 0))
                    ),
                    # Required stabilizers to modify the operator
                    old_stabs_to_be_removed,
                    tuple(),
                ),
            ),
            (
                {  # Set of parameters to move the bottom left corner up:
                    # case 1 for X, case 2 for Z
                    "corner_qubit": (0, 4, 0),
                    "unit_vector": (0, -1),
                    "how_far": 2,  # Even because the corner is not cut
                    "old_stabs_to_be_removed": (
                        old_stabs_to_be_removed := [
                            Stabilizer(
                                "XX", ((0, 2, 0), (0, 3, 0)), ancilla_qubits=[(0, 3, 1)]
                            )
                        ]
                    ),  # we extend the bottom boundary to the left
                    "new_stabs_to_be_added": (
                        new_stabs_to_be_added := [
                            Stabilizer(
                                "ZZ", ((0, 3, 0), (0, 4, 0)), ancilla_qubits=[(0, 4, 1)]
                            )
                        ]
                    ),
                },
                (
                    PauliOperator(
                        "XXXXX", ((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0))
                    ),
                    PauliOperator(
                        "ZZZ",
                        ((0, 0, 0), (0, 1, 0), (0, 2, 0)),
                    ),
                    (),
                    # Required stabilizers to modify the operator
                    new_stabs_to_be_added,
                ),
            ),
            (
                {  # Set of parameters to move the bottom left corner to the right:
                    # case 1 for X, case 3 for Z
                    "corner_qubit": (0, 4, 0),
                    "unit_vector": (1, 0),
                    "how_far": 3,  # Odd because the corner is cut
                    "old_stabs_to_be_removed": (
                        old_stabs_to_be_removed := [
                            Stabilizer(
                                "ZZ", ((1, 4, 0), (0, 4, 0)), ancilla_qubits=[(1, 5, 1)]
                            ),
                            Stabilizer(
                                "ZZ", ((3, 4, 0), (2, 4, 0)), ancilla_qubits=[(3, 5, 1)]
                            ),
                        ]
                    ),  # we extend the left boundary to the bottom
                    "new_stabs_to_be_added": [
                        Stabilizer(
                            "XX", ((2, 4, 0), (1, 4, 0)), ancilla_qubits=[(2, 5, 1)]
                        )
                    ],
                },
                (
                    PauliOperator(
                        "XXXXX", ((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0))
                    ),
                    PauliOperator(
                        "ZZZZZZZ",
                        (
                            (0, 0, 0),
                            (0, 1, 0),
                            (0, 2, 0),
                            (0, 3, 0),
                            (1, 4, 0),
                            (2, 4, 0),
                            (3, 4, 0),
                        ),
                    ),
                    (),
                    # Required stabilizers to modify the operator
                    old_stabs_to_be_removed,
                ),
            ),
            # NOTE we do not test the bottom right corner as no operator will be
            # modified
        ]
        for args, expected_output in test_args_and_expected:
            expected_x, expected_z, x_op_stabs, z_op_stabs = expected_output
            (new_x, new_z, x_evolution, z_evolution) = move_corner_logical_operators(
                block=big_block, **args
            )
            self.assertEqual(new_x, expected_x)
            self.assertEqual(new_z, expected_z)

            if big_block.logical_x_operators[0] == new_x:
                self.assertEqual(x_evolution, {})
            else:
                expected_x_evol = {
                    new_x.uuid: (big_block.logical_x_operators[0].uuid,)
                    + tuple(stab.uuid for stab in x_op_stabs)
                }
                self.assertEqual(x_evolution, expected_x_evol)

            if big_block.logical_z_operators[0] == new_z:
                self.assertEqual(z_evolution, {})
            else:
                expected_z_evol = {
                    new_z.uuid: (big_block.logical_z_operators[0].uuid,)
                    + tuple(stab.uuid for stab in z_op_stabs)
                }
                self.assertEqual(z_evolution, expected_z_evol)

    def test_move_corner_logical_operators_alternative_operator(self):
        """
        Test the move_corners_logical_operators function with an alternative logical
        X operator.
        """
        alt_block = RotatedSurfaceCode.create(
            dx=5,
            dz=5,
            lattice=self.square_2d_lattice,
            unique_label="q5x5",
            logical_x_operator=PauliOperator(
                "XXXXX",
                ((0, 2, 0), (1, 2, 0), (2, 2, 0), (3, 2, 0), (4, 2, 0)),
            ),
        )
        old_stabs_to_be_removed = [
            Stabilizer("XX", ((0, 0, 0), (0, 1, 0)), ancilla_qubits=[(0, 1, 1)]),
            Stabilizer("XX", ((0, 2, 0), (0, 3, 0)), ancilla_qubits=[(0, 3, 1)]),
        ]
        new_stabs_to_be_added = [
            Stabilizer("ZZ", ((0, 1, 0), (0, 2, 0)), ancilla_qubits=[(0, 2, 1)]),
        ]
        # Set of parameters to move the top left corner down :
        # case 3 for X, case 2 for Z
        (new_x, new_z, x_evolution, z_evolution) = move_corner_logical_operators(
            alt_block,
            corner_qubit=(0, 0, 0),
            unit_vector=(0, 1),
            how_far=3,
            old_stabs_to_be_removed=old_stabs_to_be_removed,
            new_stabs_to_be_added=new_stabs_to_be_added,
        )
        expected_x = PauliOperator(
            "XXXXX",
            (
                (1, 2, 0),
                (2, 2, 0),
                (3, 2, 0),
                (4, 2, 0),
                (0, 3, 0),
            ),
        )
        expected_z = PauliOperator("ZZ", ((0, 3, 0), (0, 4, 0)))
        # First stab is not included in the transformation
        included_old_stabs = old_stabs_to_be_removed[1:]
        expected_x_evol = {
            new_x.uuid: (alt_block.logical_x_operators[0].uuid,)
            + tuple()
            + tuple(stab.uuid for stab in included_old_stabs)
        }
        expected_z_evol = {
            new_z.uuid: (alt_block.logical_z_operators[0].uuid,)
            + tuple()
            + tuple(stab.uuid for stab in new_stabs_to_be_added)
        }
        self.assertEqual(new_x, expected_x)
        self.assertEqual(new_z, expected_z)
        self.assertEqual(x_evolution, expected_x_evol)
        self.assertEqual(z_evolution, expected_z_evol)

    def test_move_corner_circuit(self):
        """
        Test the move_corner_circuit function.
        """
        # For reference we will use a 5x5 rotated surface code with standard logical ops
        big_block = RotatedSurfaceCode.create(
            dx=5,
            dz=5,
            lattice=self.square_2d_lattice,
            unique_label="q5x5",
        )
        test_args_and_expected = [
            (
                {  # Set of parameters to move the top left corner down
                    "corner_qubit": (0, 0, 0),
                    "cut_stabilizer": Stabilizer(
                        pauli="ZZZ",
                        data_qubits=[(1, 0, 0), (1, 1, 0), (0, 1, 0)],
                        ancilla_qubits=[(1, 1, 1)],
                    ),
                    "how_far": 3,  # Odd because the corner is cut
                    "move_direction": Direction.BOTTOM,
                },
                (
                    ("c_(0, 0, 0)", 0),
                    Circuit(
                        name="moving corner (0, 0, 0) to bottom by 3",
                        circuit=[
                            [
                                Circuit(
                                    "measure",
                                    channels=[
                                        Channel(label="(0, 0, 0)"),
                                        Channel(
                                            label="c_(0, 0, 0)_0", type="classical"
                                        ),
                                    ],
                                )
                            ]
                        ],
                    ),  # Expected circuit
                ),
            ),
            (
                {  # Set of parameters to move the top left corner to the right
                    "corner_qubit": (0, 0, 0),
                    "cut_stabilizer": None,
                    "how_far": 3,  # Odd because the corner is cut
                    "move_direction": Direction.RIGHT,
                },
                None,
            ),
            (
                {  # Set of parameters to move the top right corner to the left
                    "corner_qubit": (4, 0, 0),
                    "cut_stabilizer": Stabilizer(
                        pauli="XXX",
                        data_qubits=[(3, 0, 0), (4, 1, 0), (3, 1, 0)],
                        ancilla_qubits=[(4, 1, 1)],
                    ),
                    "how_far": 1,  # Odd because the corner is cut
                    "move_direction": Direction.LEFT,
                },
                (
                    ("c_(4, 0, 0)", 0),
                    Circuit(
                        name="moving corner (4, 0, 0) to left by 1",
                        circuit=[
                            [
                                Circuit(
                                    "measure_x",
                                    channels=[
                                        Channel(label="(4, 0, 0)"),
                                        Channel(
                                            label="c_(4, 0, 0)_0", type="classical"
                                        ),
                                    ],
                                )
                            ]
                        ],
                    ),  # Expected circuit
                ),
            ),
        ]
        for args, expected_output in test_args_and_expected:
            output = move_corner_circuit(
                self.base_step,
                big_block,
                **args,
            )
            if expected_output is not None:
                expected_cbit, expected_circuit = expected_output
                circuit, cbit = output
                self.assertEqual(circuit, expected_circuit)
                self.assertEqual(cbit, expected_cbit)
            else:
                self.assertEqual(output, expected_output)

    def test_move_corner(self):  # pylint: disable=too-many-locals,too-many-statements
        """Test the move_corner function. The two examples are: move the upper left
        corner down by 3 and move the upper left corner right by 2.
        """
        big_block = RotatedSurfaceCode.create(
            dx=5,
            dz=5,
            lattice=self.square_2d_lattice,
            unique_label="q5x5",
        )
        int_step = InterpretationStep(
            block_history=((big_block,),),
            syndromes=tuple(
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=((f"c_{stab.ancilla_qubits[0]}", 0),),
                    block=big_block.uuid,
                    round=0,
                    corrections=(),
                )
                for stab in big_block.stabilizers
            ),
        )

        # I - Move the upper left corner down by 3
        # Create the expected block components
        # Stabilizers
        stabilizers = [
            stab
            for stab in big_block.bulk_stabilizers
            if (0, 0, 0) not in stab.data_qubits
        ] + list(
            big_block.boundary_stabilizers(Direction.TOP)
            + big_block.boundary_stabilizers(Direction.RIGHT)
            + big_block.boundary_stabilizers(Direction.BOTTOM)
        )
        cut_stabilizer = Stabilizer(
            "ZZZ", [(1, 0, 0), (1, 1, 0), (0, 1, 0)], ancilla_qubits=[(1, 1, 1)]
        )
        new_boundary_stab = Stabilizer(
            "ZZ", ((0, 1, 0), (0, 2, 0)), ancilla_qubits=[(0, 2, 1)]
        )
        stabilizers += [cut_stabilizer, new_boundary_stab]
        # Logical operators
        logical_x = PauliOperator(
            "XXXXXXX",
            (
                (1, 0, 0),
                (2, 0, 0),
                (3, 0, 0),
                (4, 0, 0),
                (0, 1, 0),
                (0, 2, 0),
                (0, 3, 0),
            ),
        )
        stabs_to_remove = [
            stab
            for stab in big_block.stabilizers
            if all(
                q in ((0, 0, 0), (0, 2, 0), (0, 1, 0), (0, 3, 0))
                for q in stab.data_qubits
            )
        ]
        logical_z = PauliOperator("ZZ", ((0, 3, 0), (0, 4, 0)))
        # Syndrome circuits and mapping
        syndrome_circuits = big_block.syndrome_circuits + (
            RotatedSurfaceCode.generate_syndrome_circuit("ZZZ", [1], "top-left-ZZZ"),
            RotatedSurfaceCode.generate_syndrome_circuit("ZZ", [1, 3], "left-ZZ"),
        )
        stab_to_circuit = {
            stab_id: synd_circ_id
            for stab_id, synd_circ_id in big_block.stabilizer_to_circuit.items()
            if stab_id in [stab.uuid for stab in stabilizers]
        } | {
            cut_stabilizer.uuid: syndrome_circuits[-2].uuid,
            new_boundary_stab.uuid: syndrome_circuits[-1].uuid,
        }
        # Remove the syndrome circuits that are not used anymore
        syndrome_circuits = tuple(
            sc for sc in syndrome_circuits if sc.uuid in stab_to_circuit.values()
        )

        expected_block = RotatedSurfaceCode(
            unique_label="q5x5",
            stabilizers=stabilizers,
            logical_x_operators=[logical_x],
            logical_z_operators=[logical_z],
            syndrome_circuits=syndrome_circuits,
            stabilizer_to_circuit=stab_to_circuit,
        )

        # The example is: move the upper left corner down by 3
        output_int_step, past_updates, future_updates = move_corner(
            interpretation_step=deepcopy(int_step),
            block=big_block,
            corner_qubit=(0, 0, 0),
            move_direction=Direction.BOTTOM,
            how_far=3,
        )
        new_block = output_int_step.get_block("q5x5")
        self.assertEqual(new_block, expected_block)

        logical_x_evol = {
            new_block.logical_x_operators[0].uuid: (
                big_block.logical_x_operators[0].uuid,
            )
            + tuple(stab.uuid for stab in stabs_to_remove)
        }
        logical_z_evol = {
            new_block.logical_z_operators[0].uuid: (
                big_block.logical_z_operators[0].uuid,
            )
            + (
                next(
                    stab.uuid
                    for stab in new_block.stabilizers
                    if stab == new_boundary_stab
                ),
            )
        }
        expected_stab_evolution = {
            next(stab for stab in new_block.stabilizers if stab.pauli == "ZZZ").uuid: (
                next(
                    stab
                    for stab in big_block.stabilizers
                    if (1, 1, 1) in stab.ancilla_qubits
                ).uuid,
            )
        }
        self.assertEqual(output_int_step.logical_x_evolution, logical_x_evol)
        self.assertEqual(output_int_step.logical_z_evolution, logical_z_evol)
        self.assertEqual(output_int_step.stabilizer_evolution, expected_stab_evolution)

        cbit = ("c_(0, 0, 0)", 0)
        expected_stab_updates = {
            next(stab for stab in new_block.stabilizers if stab.pauli == "ZZZ").uuid: (
                cbit,
            ),
        }
        expected_log_x_updates = {
            new_block.logical_x_operators[0].uuid: (cbit,),
        }
        expected_log_z_updates = {
            new_block.logical_z_operators[0].uuid: (cbit,),
        }
        self.assertEqual(output_int_step.stabilizer_updates, expected_stab_updates)
        self.assertEqual(
            output_int_step.logical_x_operator_updates, expected_log_x_updates
        )
        self.assertEqual(
            output_int_step.logical_z_operator_updates, expected_log_z_updates
        )

        expected_past_udpates = (tuple(stabs_to_remove), ())
        self.assertEqual(past_updates, expected_past_udpates)
        expected_future_updates = (
            (),
            (new_boundary_stab,),
        )
        self.assertEqual(future_updates, expected_future_updates)

        # II - Move the upper left corner right by 2
        # Create the expected block components
        # Stabilizers
        stabilizers = (
            list(big_block.bulk_stabilizers)  # We keep the bulk stabilizers
            + list(
                big_block.boundary_stabilizers(Direction.RIGHT)
                + big_block.boundary_stabilizers(Direction.LEFT)
                + big_block.boundary_stabilizers(Direction.BOTTOM)
            )
            + [
                stab
                for stab in big_block.boundary_stabilizers(Direction.TOP)
                if (1, 0, 0) not in stab.data_qubits  # Remove part of the top boundary
            ]
        )
        new_boundary_stab = Stabilizer(
            "XX", ((1, 0, 0), (0, 0, 0)), ancilla_qubits=[(1, 0, 1)]
        )
        stabilizers += [new_boundary_stab]
        # Logical operators
        logical_x = PauliOperator(
            "XXX",
            (
                (2, 0, 0),
                (3, 0, 0),
                (4, 0, 0),
            ),
        )
        stabs_to_remove = [
            stab
            for stab in big_block.stabilizers
            if all(q in ((0, 0, 0), (1, 0, 0), (2, 0, 0)) for q in stab.data_qubits)
        ]
        logical_z = PauliOperator(
            "ZZZZZZZ",
            (
                (0, 0, 0),
                (0, 1, 0),
                (0, 2, 0),
                (0, 3, 0),
                (0, 4, 0),
                (1, 0, 0),
                (2, 0, 0),
            ),
        )
        # Syndrome circuits and mapping
        syndrome_circuits = big_block.syndrome_circuits + (
            RotatedSurfaceCode.generate_syndrome_circuit("XX", [0, 2], "top-XX"),
        )
        stab_to_circuit = {
            stab_id: synd_circ_id
            for stab_id, synd_circ_id in big_block.stabilizer_to_circuit.items()
            if stab_id in [stab.uuid for stab in stabilizers]
        } | {
            new_boundary_stab.uuid: syndrome_circuits[-1].uuid,
        }

        expected_block = RotatedSurfaceCode(
            unique_label="q5x5",
            stabilizers=stabilizers,
            logical_x_operators=[logical_x],
            logical_z_operators=[logical_z],
            syndrome_circuits=syndrome_circuits,
            stabilizer_to_circuit=stab_to_circuit,
        )

        # The example is: move the upper left corner right by 2
        output_int_step, past_updates, future_updates = move_corner(
            interpretation_step=deepcopy(int_step),
            block=big_block,
            corner_qubit=(0, 0, 0),
            move_direction=Direction.RIGHT,
            how_far=2,
        )
        new_block = output_int_step.get_block("q5x5")
        self.assertEqual(new_block, expected_block)

        logical_x_evol = {
            new_block.logical_x_operators[0].uuid: (
                big_block.logical_x_operators[0].uuid,
            )
            + (
                next(
                    stab.uuid
                    for stab in new_block.stabilizers
                    if stab == new_boundary_stab
                ),
            )
        }
        logical_z_evol = {
            new_block.logical_z_operators[0].uuid: (
                big_block.logical_z_operators[0].uuid,
            )
            + tuple(stab.uuid for stab in stabs_to_remove)
        }
        expected_stab_evolution = {}  # No stab is morphed this time
        self.assertEqual(output_int_step.logical_x_evolution, logical_x_evol)
        self.assertEqual(output_int_step.logical_z_evolution, logical_z_evol)
        self.assertEqual(output_int_step.stabilizer_evolution, expected_stab_evolution)

        # There is no update since we perform no measurement
        expected_stab_updates = {}
        expected_log_x_updates = {}
        expected_log_z_updates = {}
        self.assertEqual(output_int_step.stabilizer_updates, expected_stab_updates)
        self.assertEqual(
            output_int_step.logical_x_operator_updates, expected_log_x_updates
        )
        self.assertEqual(
            output_int_step.logical_z_operator_updates, expected_log_z_updates
        )

        expected_past_udpates = ((), tuple(stabs_to_remove))
        self.assertEqual(past_updates, expected_past_udpates)
        expected_future_updates = ((new_boundary_stab,), ())
        self.assertEqual(future_updates, expected_future_updates)

    def test_move_corners(self):  # pylint: disable=too-many-statements
        """
        Test the move_corners function. This is a wrapper for using multiple times
        the move_corner function in a row. Since it also measures the syndromes, we
        will get additional syndromes and detectors
        """
        # I - Perform the move required for a Hadamard

        # For reference we will use a 5x9 rotated surface code with standard logical ops
        # This example is used when performing a logical Hadamard. 6 separate moves are
        # performed to rotate the topological corners

        # The initial block is the following:
        #
        #                      Z                   Z
        #    *(0,0) --- (1,0) --- (2,0) --- (3,0) --- (4,0)*
        #       |         |         |         |         |
        #    X  |    Z    |    X    |    Z    |    X    |
        #       |         |         |         |         |
        #     (0,1) --- (1,1) --- (2,1) --- (3,1) --- (4,1)
        #       |         |         |         |         |
        #       |    X    |    Z    |    X    |    Z    |  X
        #       |         |         |         |         |
        #     (0,2) --- (1,2) --- (2,2) --- (3,2) --- (4,2)
        #       |         |         |         |         |
        #    X  |    Z    |    X    |    Z    |    X    |
        #       |         |         |         |         |
        #     (0,3) --- (1,3) --- (2,3) --- (3,3) --- (4,3)
        #       |         |         |         |         |
        #       |    X    |    Z    |    X    |    Z    |  X
        #       |         |         |         |         |
        #     (0,4) --- (1,4) --- (2,4) --- (3,4) --- (4,4)
        #       |         |         |         |         |
        #    X  |    Z    |    X    |    Z    |    X    |
        #       |         |         |         |         |
        #     (0,5) --- (1,5) --- (2,5) --- (3,5) --- (4,5)
        #       |         |         |         |         |
        #       |    X    |    Z    |    X    |    Z    |  X
        #       |         |         |         |         |
        #     (0,6) --- (1,6) --- (2,6) --- (3,6) --- (4,6)
        #       |         |         |         |         |
        #    X  |    Z    |    X    |    Z    |    X    |
        #       |         |         |         |         |
        #     (0,7) --- (1,7) --- (2,7) --- (3,7) --- (4,7)
        #       |         |         |         |         |
        #       |    X    |    Z    |    X    |    Z    |  X
        #       |         |         |         |         |
        #    *(0,8) --- (1,8) --- (2,8) --- (3,8) --- (4,8)*
        #            Z                   Z
        #
        # The final block after 6 moves is:
        #
        #            X                   X
        #    *(0,0) --- (1,0) --- (2,0) --- (3,0) --- (4,0)*
        #       |         |         |         |         |
        #       |    Z    |    X    |    Z    |    X    |  Z
        #       |         |         |         |         |
        #     (0,1) --- (1,1) --- (2,1) --- (3,1) --- (4,1)
        #       |         |         |         |         |
        #    Z  |    X    |    Z    |    X    |    Z    |
        #       |         |         |         |         |
        #     (0,2) --- (1,2) --- (2,2) --- (3,2) --- (4,2)
        #       |         |         |         |         |
        #       |    Z    |    X    |    Z    |    X    |  Z
        #       |         |         |         |         |
        #     (0,3) --- (1,3) --- (2,3) --- (3,3) --- (4,3)
        #       |         |         |         |         |
        #    Z  |    X    |    Z    |    X    |    Z    |
        #       |         |         |         |         |
        #     (0,4) --- (1,4) --- (2,4) --- (3,4) --- (4,4)
        #       |         |         |         |         |
        #       |    Z    |    X    |    Z    |    X    |  Z
        #       |         |         |         |         |
        #     (0,5) --- (1,5) --- (2,5) --- (3,5) --- (4,5)
        #       |         |         |         |         |
        #    Z  |    X    |    Z    |    X    |    Z    |
        #       |         |         |         |         |
        #     (0,6) --- (1,6) --- (2,6) --- (3,6) --- (4,6)
        #       |         |         |         |         |
        #       |    Z    |    X    |    Z    |    X    |  Z
        #       |         |         |         |         |
        #     (0,7) --- (1,7) --- (2,7) --- (3,7) --- (4,7)
        #       |         |         |         |         |
        #    Z  |    X    |    Z    |    X    |    Z    |
        #       |         |         |         |         |
        #    *(0,8) --- (1,8) --- (2,8) --- (3,8) --- (4,8)*
        #                      X                   X

        big_block = RotatedSurfaceCode.create(
            dx=5,
            dz=9,
            lattice=self.square_2d_lattice,
            unique_label="q1",
        )
        corner_args = (
            ((4, 0, 0), Direction.BOTTOM, 4),
            ((0, 8, 0), Direction.TOP, 4),
            ((4, 8, 0), Direction.LEFT, 4),
            ((0, 0, 0), Direction.RIGHT, 4),
            ((4, 4, 0), Direction.BOTTOM, 4),
            ((0, 4, 0), Direction.TOP, 4),
        )
        int_step = InterpretationStep(
            block_history=((big_block,),),
            syndromes=tuple(
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=((f"c_{stab.ancilla_qubits[0]}", i),),
                    block=big_block.uuid,
                    round=i,
                    corrections=(),
                )
                for stab in big_block.stabilizers
                for i in range(3)
            ),
            logical_x_operator_updates={
                big_block.logical_x_operators[0].uuid: (("dummy_X", 0),)
            },  # Check for update propagation
            logical_z_operator_updates={
                big_block.logical_z_operators[0].uuid: (("dummy_Z", 0),)
            },  # Check for update propagation
        )
        interpretation_step = move_corners(
            interpretation_step=int_step,
            block=big_block,
            corner_args=corner_args,
            same_timeslice=False,
            debug_mode=True,
        )
        new_block = interpretation_step.get_block("q1")

        mock_expected_block = RotatedSurfaceCode.create(
            dx=5,
            dz=9,
            lattice=self.square_2d_lattice,
            unique_label="q1",
            weight_2_stab_is_first_row=False,
            weight_4_x_schedule="N",
            x_boundary=Orientation.VERTICAL,
            logical_x_operator=PauliOperator(
                "XXXXXXXXX",
                (
                    (4, 0, 0),
                    (4, 1, 0),
                    (4, 2, 0),
                    (4, 3, 0),
                    (4, 4, 0),
                    (4, 5, 0),
                    (4, 6, 0),
                    (4, 7, 0),
                    (4, 8, 0),
                ),
            ),
            logical_z_operator=PauliOperator(
                "ZZZZZ",
                (
                    (0, 0, 0),
                    (1, 0, 0),
                    (2, 0, 0),
                    (3, 0, 0),
                    (4, 0, 0),
                ),
            ),
        )

        # Populate with appropriate syndrome circuits
        synd_circs, new_stab_to_circuit = generate_syndrome_extraction_circuits(
            mock_expected_block, DiagonalDirection.TOP_RIGHT
        )
        expected_block = RotatedSurfaceCode(
            unique_label=mock_expected_block.unique_label,
            stabilizers=mock_expected_block.stabilizers,
            logical_x_operators=mock_expected_block.logical_x_operators,
            logical_z_operators=mock_expected_block.logical_z_operators,
            syndrome_circuits=synd_circs,
            stabilizer_to_circuit=new_stab_to_circuit,
        )
        self.assertEqual(new_block, expected_block)

        # Check that the stabilizer evolution is empty (no stab is cut)
        self.assertEqual(interpretation_step.stabilizer_evolution, {})

        # Check that the final logical operators have the right updates
        self.assertEqual(
            interpretation_step.logical_x_operator_updates[
                new_block.logical_x_operators[0].uuid
            ],
            (
                ("c_(5, 6, 1)", 2),
                ("c_(5, 8, 1)", 2),
                ("c_(5, 2, 1)", 2),
                ("c_(5, 4, 1)", 2),
                ("dummy_X", 0),
                ("c_(1, 0, 1)", 0),
                ("c_(3, 0, 1)", 0),
            ),
        )
        self.assertEqual(
            interpretation_step.logical_z_operator_updates[
                new_block.logical_z_operators[0].uuid
            ],
            (
                ("c_(2, 0, 1)", 2),
                ("c_(4, 0, 1)", 2),
                ("dummy_Z", 0),
                ("c_(0, 6, 1)", 0),
                ("c_(0, 8, 1)", 0),
                ("c_(0, 2, 1)", 0),
                ("c_(0, 4, 1)", 0),
            ),
        )

        # II - Change the block's boundaries orientation (rotate by 90 degrees)

        # For reference we will use a 5x5 rotated surface code with standard logical
        # ops. It takes 8 moves to perform the complete rotation of the boundaries.

        # The initial block is the following:
        #
        #                      Z                   Z
        #    *(0,0) --- (1,0) --- (2,0) --- (3,0) --- (4,0)*
        #       |         |         |         |         |
        #    X  |    Z    |    X    |    Z    |    X    |
        #       |         |         |         |         |
        #     (0,1) --- (1,1) --- (2,1) --- (3,1) --- (4,1)
        #       |         |         |         |         |
        #       |    X    |    Z    |    X    |    Z    |  X
        #       |         |         |         |         |
        #     (0,2) --- (1,2) --- (2,2) --- (3,2) --- (4,2)
        #       |         |         |         |         |
        #    X  |    Z    |    X    |    Z    |    X    |
        #       |         |         |         |         |
        #     (0,3) --- (1,3) --- (2,3) --- (3,3) --- (4,3)
        #       |         |         |         |         |
        #       |    X    |    Z    |    X    |    Z    |  X
        #       |         |         |         |         |
        #    *(0,4) --- (1,4) --- (2,4) --- (3,4) --- (4,4)*
        #            Z                   Z
        #
        # The final block after 8 moves is:
        #
        #            X                   X
        #    *(0,0) --- (1,0) --- (2,0) --- (3,0) --- (4,0)*
        #       |         |         |         |         |
        #       |    Z    |    X    |    Z    |    X    |  Z
        #       |         |         |         |         |
        #     (0,1) --- (1,1) --- (2,1) --- (3,1) --- (4,1)
        #       |         |         |         |         |
        #    Z  |    X    |    Z    |    X    |    Z    |
        #       |         |         |         |         |
        #     (0,2) --- (1,2) --- (2,2) --- (3,2) --- (4,2)
        #       |         |         |         |         |
        #       |    Z    |    X    |    Z    |    X    |  Z
        #       |         |         |         |         |
        #     (0,3) --- (1,3) --- (2,3) --- (3,3) --- (4,3)
        #       |         |         |         |         |
        #    Z  |    X    |    Z    |    X    |    Z    |
        #       |         |         |         |         |
        #    *(0,4) --- (1,4) --- (2,4) --- (3,4) --- (4,4)*
        #                      X                   X
        square_block = RotatedSurfaceCode.create(
            dx=5,
            dz=5,
            lattice=self.square_2d_lattice,
            unique_label="q2",
        )
        square_corner_args = (
            ((0, 0, 0), Direction.RIGHT, 2),
            ((4, 0, 0), Direction.BOTTOM, 2),
            ((0, 4, 0), Direction.TOP, 2),
            ((4, 4, 0), Direction.LEFT, 2),
            ((2, 0, 0), Direction.RIGHT, 2),
            ((4, 2, 0), Direction.BOTTOM, 2),
            ((0, 2, 0), Direction.TOP, 2),
            ((2, 4, 0), Direction.LEFT, 2),
        )
        int_square_step = InterpretationStep(
            block_history=((square_block,),),
            syndromes=tuple(
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=((f"c_{stab.ancilla_qubits[0]}", i),),
                    block=square_block.uuid,
                    round=i,
                    corrections=(),
                )
                for stab in square_block.stabilizers
                for i in range(3)
            ),
            logical_x_operator_updates={
                square_block.logical_x_operators[0].uuid: (("dummy_X", 1),)
            },  # Check for update propagation
            logical_z_operator_updates={
                square_block.logical_z_operators[0].uuid: (("dummy_Z", 1),)
            },  # Check for update propagation
        )
        interpretation_step = move_corners(
            interpretation_step=int_square_step,
            block=square_block,
            corner_args=square_corner_args,
            same_timeslice=False,
            debug_mode=True,
        )
        new_square_block = interpretation_step.get_block("q2")

        mock_expected_square_block = RotatedSurfaceCode.create(
            dx=5,
            dz=5,
            lattice=self.square_2d_lattice,
            unique_label="q2",
            weight_4_x_schedule="N",
            weight_2_stab_is_first_row=False,
            x_boundary=Orientation.VERTICAL,
            logical_x_operator=PauliOperator(
                "XXXXX",
                (
                    (4, 0, 0),
                    (4, 1, 0),
                    (4, 2, 0),
                    (4, 3, 0),
                    (4, 4, 0),
                ),
            ),
            logical_z_operator=PauliOperator(
                "ZZZZZ",
                (
                    (0, 0, 0),
                    (1, 0, 0),
                    (2, 0, 0),
                    (3, 0, 0),
                    (4, 0, 0),
                ),
            ),
        )

        # Populate with appropriate syndrome circuits
        synd_circs, new_stab_to_circuit = generate_syndrome_extraction_circuits(
            mock_expected_square_block, DiagonalDirection.TOP_RIGHT
        )
        expected_square_block = RotatedSurfaceCode(
            unique_label=mock_expected_square_block.unique_label,
            stabilizers=mock_expected_square_block.stabilizers,
            logical_x_operators=mock_expected_square_block.logical_x_operators,
            logical_z_operators=mock_expected_square_block.logical_z_operators,
            syndrome_circuits=synd_circs,
            stabilizer_to_circuit=new_stab_to_circuit,
        )

        self.assertEqual(new_square_block, expected_square_block)

        # Check that the stabilizer evolution is empty (no stab is cut)
        self.assertEqual(interpretation_step.stabilizer_evolution, {})

        # Check the final logical operators have the right updates
        self.assertEqual(
            interpretation_step.logical_x_operator_updates[
                new_square_block.logical_x_operators[0].uuid
            ],
            (
                ("c_(5, 4, 1)", 2),
                ("c_(5, 2, 1)", 2),
                ("dummy_X", 1),
                ("c_(1, 0, 1)", 0),
                ("c_(3, 0, 1)", 0),
            ),
        )
        self.assertEqual(
            interpretation_step.logical_z_operator_updates[
                new_square_block.logical_z_operators[0].uuid
            ],
            (
                ("c_(4, 0, 1)", 2),
                ("c_(2, 0, 1)", 2),
                ("dummy_Z", 1),
                ("c_(0, 4, 1)", 0),
                ("c_(0, 2, 1)", 0),
            ),
        )


if __name__ == "__main__":
    unittest.main()
