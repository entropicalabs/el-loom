"""
Copyright 2024 Entropica Labs Pte Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

# pylint: disable=duplicate-code, too-many-lines
import unittest

from loom.eka import Lattice, Stabilizer, PauliOperator
from loom.eka.utilities import Orientation, Direction
from loom.eka.operations import MeasureBlockSyndromes
from loom.interpreter import InterpretationStep, Syndrome
from loom.interpreter.applicator import measureblocksyndromes

from loom_rotated_surface_code.applicator.y_wall_out import y_wall_out
from loom_rotated_surface_code.applicator.y_wall_out.y_wall_out import (
    y_wall_out_consistency_check,
)
from loom_rotated_surface_code.applicator.y_wall_out.y_wall_measurement_hadamard import (  # pylint: disable=line-too-long
    find_new_x_logical_operator,
)
from loom_rotated_surface_code.applicator.move_corners import move_corners
from loom_rotated_surface_code.code_factory import RotatedSurfaceCode


class TestRotatedSurfaceCodeYWallOut(
    unittest.TestCase
):  # pylint: disable=too-many-instance-attributes
    """
    Tests the Y wall out operation on a RotatedSurfaceCode.
    """

    def setUp(self):
        self.square_2d_lattice = Lattice.square_2d((10, 20))

        # MAKE A VERTICAL BLOCK
        # distance: 5, top-left bulk stabilizer: Z
        self.big_block_v5z = RotatedSurfaceCode.create(
            dx=5,
            dz=10,
            lattice=self.square_2d_lattice,
            unique_label="q1",
            weight_2_stab_is_first_row=False,
            x_boundary=Orientation.VERTICAL,
        )

        # Get the twisted block v5z by moving the topological corner instead
        self.move_corner_step_v5z = move_corners(
            interpretation_step=InterpretationStep(
                block_history=((self.big_block_v5z,),),
                syndromes=tuple(
                    Syndrome(
                        stabilizer=stab.uuid,
                        measurements=((f"c_{stab.ancilla_qubits[0]}", 0),),
                        block=self.big_block_v5z.unique_label,
                        round=0,
                        corrections=[],
                    )
                    for stab in self.big_block_v5z.stabilizers
                ),
            ),
            block=self.big_block_v5z,
            corner_args=(((0, 9, 0), Direction.TOP, 4),),
            same_timeslice=False,
            debug_mode=True,
        )
        self.twisted_rsc_block_v5z = self.move_corner_step_v5z.get_block(
            self.big_block_v5z.unique_label
        )

        # For reference, the twisted block v5z is the following:
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
        #    *(0,5) --- (1,5) --- (2,5) --- (3,5) --- (4,5)
        #       |         |         |         |         |
        #       |    X    |    Z    |    X    |    Z    |
        #       |         |         |         |         |
        #     (0,6) --- (1,6) --- (2,6) --- (3,6) --- (4,6)
        #       |         |         |         |         |
        #    X  |    Z    |    X    |    Z    |    X    |  Z
        #       |         |         |         |         |
        #     (0,7) --- (1,7) --- (2,7) --- (3,7) --- (4,7)
        #       |         |         |         |         |
        #       |    X    |    Z    |    X    |    Z    |
        #       |         |         |         |         |
        #     (0,8) --- (1,8) --- (2,8) --- (3,8) --- (4,8)
        #       |         |         |         |         |
        #    X  |    Z    |    X    |    Z    |    X    |  Z
        #       |         |         |         |         |
        #     (0,9) --- (1,9) --- (2,9) --- (3,9) --- (4,9)*
        #            X                   X
        #
        # The topological corners are the starred qubits

        # MAKE A SECOND VERTICAL BLOCK
        # distance: 5, top-left bulk stabilizer: X
        self.big_block_v5x = RotatedSurfaceCode.create(
            dx=5,
            dz=10,
            lattice=self.square_2d_lattice,
            unique_label="q2",
            weight_2_stab_is_first_row=True,
            x_boundary=Orientation.VERTICAL,
        )

        self.twisted_rsc_block_v5x_with_wrong_x_logical = move_corners(
            interpretation_step=InterpretationStep(
                block_history=((self.big_block_v5x,),),
                syndromes=tuple(
                    Syndrome(
                        stabilizer=stab.uuid,
                        measurements=((f"c_{stab.ancilla_qubits[0]}", 0),),
                        block=self.big_block_v5x.unique_label,
                        round=0,
                        corrections=[],
                    )
                    for stab in self.big_block_v5x.stabilizers
                ),
            ),
            block=self.big_block_v5x,
            corner_args=(((4, 9, 0), Direction.TOP, 4),),
            same_timeslice=False,
            debug_mode=True,
        ).get_block(self.big_block_v5x.unique_label)

        # Rewrite the block but with the X logical operator on the right side so that
        # it's on the correct side
        block = self.twisted_rsc_block_v5x_with_wrong_x_logical
        new_x_logical_operator = PauliOperator(
            "X" * 6, [qub for qub in block.boundary_qubits("right") if qub[1] <= 5]
        )
        self.twisted_rsc_block_v5x = RotatedSurfaceCode(
            stabilizers=block.stabilizers,
            logical_x_operators=(new_x_logical_operator,),
            logical_z_operators=block.logical_z_operators,
            syndrome_circuits=block.syndrome_circuits,
            stabilizer_to_circuit=block.stabilizer_to_circuit,
            unique_label=block.unique_label,
            skip_validation=False,
        )
        # For reference, the twisted block v5x is the following:
        #
        #                      X                   X
        #    *(0,0) --- (1,0) --- (2,0) --- (3,0) --- (4,0)*
        #       |         |         |         |         |
        #    Z  |    X    |    Z    |    X    |    Z    |
        #       |         |         |         |         |
        #     (0,1) --- (1,1) --- (2,1) --- (3,1) --- (4,1)
        #       |         |         |         |         |
        #       |    Z    |    X    |    Z    |    X    |  Z
        #       |         |         |         |         |
        #     (0,2) --- (1,2) --- (2,2) --- (3,2) --- (4,2)
        #       |         |         |         |         |
        #    Z  |    X    |    Z    |    X    |    Z    |
        #       |         |         |         |         |
        #     (0,3) --- (1,3) --- (2,3) --- (3,3) --- (4,3)
        #       |         |         |         |         |
        #       |    Z    |    X    |    Z    |    X    |  Z
        #       |         |         |         |         |
        #     (0,4) --- (1,4) --- (2,4) --- (3,4) --- (4,4)
        #       |         |         |         |         |
        #    Z  |    X    |    Z    |    X    |    Z    |
        #       |         |         |         |         |
        #     (0,5) --- (1,5) --- (2,5) --- (3,5) --- (4,5)*
        #       |         |         |         |         |
        #       |    Z    |    X    |    Z    |    X    |
        #       |         |         |         |         |
        #     (0,6) --- (1,6) --- (2,6) --- (3,6) --- (4,6)
        #       |         |         |         |         |
        #    Z  |    X    |    Z    |    X    |    Z    |  X
        #       |         |         |         |         |
        #     (0,7) --- (1,7) --- (2,7) --- (3,7) --- (4,7)
        #       |         |         |         |         |
        #       |    Z    |    X    |    Z    |    X    |
        #       |         |         |         |         |
        #     (0,8) --- (1,8) --- (2,8) --- (3,8) --- (4,8)
        #       |         |         |         |         |
        #    Z  |    X    |    Z    |    X    |    Z    |  X
        #       |         |         |         |         |
        #    *(0,9) --- (1,9) --- (2,9) --- (3,9) --- (4,9)
        #                      X                   X

        # MAKE A HORIZONTAL BLOCK
        # The block is the same as the v5z block but rotated 90 degrees
        # counter-clockwise
        # The topological corners will be at:
        # (0, 0), (0, 4), (9, 0) and (5, 4)
        # Also note that the top left bulk stabilizer is going to be an X stabilizer

        self.big_block_h5x = RotatedSurfaceCode.create(
            dx=10,
            dz=5,
            lattice=self.square_2d_lattice,
            unique_label="q1",
            weight_2_stab_is_first_row=False,
            x_boundary=Orientation.HORIZONTAL,
        )

        self.twisted_rsc_block_h5x_with_wrong_x_logical = move_corners(
            interpretation_step=InterpretationStep(
                block_history=((self.big_block_h5x,),),
                syndromes=tuple(
                    Syndrome(
                        stabilizer=stab.uuid,
                        measurements=((f"c_{stab.ancilla_qubits[0]}", 0),),
                        block=self.big_block_h5x.unique_label,
                        round=0,
                        corrections=[],
                    )
                    for stab in self.big_block_h5x.stabilizers
                ),
            ),
            block=self.big_block_h5x,
            corner_args=(((9, 4, 0), Direction.LEFT, 4),),
            same_timeslice=False,
            debug_mode=True,
        ).get_block(self.big_block_h5x.unique_label)

        # Rewrite the block but with the X logical operator on the right side so that
        # it's on the correct side
        block = self.twisted_rsc_block_h5x_with_wrong_x_logical
        new_x_logical_operator = PauliOperator(
            "X" * 6, [qub for qub in block.boundary_qubits("bottom") if qub[0] <= 5]
        )
        self.twisted_rsc_block_h5x = RotatedSurfaceCode(
            stabilizers=block.stabilizers,
            logical_x_operators=(new_x_logical_operator,),
            logical_z_operators=block.logical_z_operators,
            syndrome_circuits=block.syndrome_circuits,
            stabilizer_to_circuit=block.stabilizer_to_circuit,
            unique_label=block.unique_label,
            skip_validation=False,
        )

        # MAKE A VERTICAL BLOCK
        # distance: 3, top-left bulk stabilizer: Z
        self.big_block_v3z = RotatedSurfaceCode.create(
            dx=3,
            dz=6,
            lattice=self.square_2d_lattice,
            unique_label="q3",
            weight_2_stab_is_first_row=False,
            x_boundary=Orientation.VERTICAL,
        )

        self.move_corner_step_v3z = move_corners(
            interpretation_step=InterpretationStep(
                block_history=((self.big_block_v3z,),),
                syndromes=tuple(
                    Syndrome(
                        stabilizer=stab.uuid,
                        measurements=((f"c_{stab.ancilla_qubits[0]}", 0),),
                        block=self.big_block_v3z.unique_label,
                        round=0,
                        corrections=[],
                    )
                    for stab in self.big_block_v3z.stabilizers
                ),
            ),
            block=self.big_block_v3z,
            corner_args=(((0, 5, 0), Direction.TOP, 2),),
            same_timeslice=False,
            debug_mode=True,
        )
        self.twisted_rsc_block_v3z = self.move_corner_step_v3z.get_block(
            self.big_block_v3z.unique_label
        )

    @staticmethod
    def base_interpretation_step(block):
        """Create a base interpretation step for the given block."""
        return InterpretationStep(
            block_history=((block,),),
            syndromes=[
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=[("mock_register", i)],
                    block=block.uuid,
                    round=-1,
                )
                for i, stab in enumerate(block.stabilizers)
            ],
        )

    def test_consistency_check(self):
        """Test the y_wall_out_consistency_check function."""
        # Test the vertical block v1
        # These should run without raising any exceptions
        y_wall_out_consistency_check(
            self.twisted_rsc_block_v5z, 5, Orientation.HORIZONTAL
        )
        y_wall_out_consistency_check(
            self.twisted_rsc_block_v5x, 5, Orientation.HORIZONTAL
        )
        y_wall_out_consistency_check(
            self.twisted_rsc_block_h5x, 5, Orientation.VERTICAL
        )
        y_wall_out_consistency_check(
            self.twisted_rsc_block_v3z, 3, Orientation.HORIZONTAL
        )

        with self.assertRaises(ValueError) as cm:
            y_wall_out_consistency_check(
                self.twisted_rsc_block_v5z, 4, Orientation.HORIZONTAL
            )
        expected_msg = (
            "The wall position must be such that the block on the bottom/right side "
            "of the wall is a square."
        )
        self.assertIn(expected_msg, str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            y_wall_out_consistency_check(
                self.twisted_rsc_block_v5z, 5, Orientation.VERTICAL
            )
        expected_msg = "The block and the wall must have perpendicular orientations."
        self.assertIn(expected_msg, str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            y_wall_out_consistency_check(self.big_block_v5z, 5, Orientation.HORIZONTAL)
        expected_msg = (
            "The block does not have 3 topological corners located at the geometric "
            "corners."
        )

        with self.assertRaises(ValueError) as cm:
            y_wall_out_consistency_check(
                self.twisted_rsc_block_v5x_with_wrong_x_logical,
                5,
                Orientation.HORIZONTAL,
            )
        expected_msg = (
            "The X logical operator is not located at the expected position. It needs "
            "to be straight and with the smallest distance possible given the geometry "
            "of the block."
        )
        self.assertIn(expected_msg, str(cm.exception))

    def test_y_wall_out_debug(self):
        """Test the y_wall_out function in debug mode."""
        for twisted_block, args in [
            (self.twisted_rsc_block_v5z, (5, Orientation.HORIZONTAL)),
            (self.twisted_rsc_block_v5x, (5, Orientation.HORIZONTAL)),
            (self.twisted_rsc_block_h5x, (5, Orientation.VERTICAL)),
            (self.twisted_rsc_block_v3z, (3, Orientation.HORIZONTAL)),
        ]:
            y_wall_out(
                self.base_interpretation_step(twisted_block),
                twisted_block,
                *args,
                same_timeslice=False,
                debug_mode=True,
            )

    def test_y_wall_out_block(self):
        """Test whether the y_wall_out function returns the correct block."""
        y_wall_out_block_v5z = y_wall_out(
            self.base_interpretation_step(self.twisted_rsc_block_v5z),
            self.twisted_rsc_block_v5z,
            5,
            Orientation.HORIZONTAL,
            same_timeslice=False,
            debug_mode=True,
        ).get_block(self.twisted_rsc_block_v5z.unique_label)

        # Get the expected Stabilizers
        expected_stabilizers_args = [
            # idling stabilizers
            (
                "XX",
                ((1, 0, 0), (0, 0, 0)),
                ((1, 0, 1),),
            ),
            (
                "XX",
                ((3, 0, 0), (2, 0, 0)),
                ((3, 0, 1),),
            ),
            (
                "ZZ",
                ((0, 1, 0), (0, 2, 0)),
                ((0, 2, 1),),
            ),
            (
                "ZZ",
                ((0, 3, 0), (0, 4, 0)),
                ((0, 4, 1),),
            ),
            (
                "ZZ",
                ((4, 0, 0), (4, 1, 0)),
                ((5, 1, 1),),
            ),
            (
                "ZZ",
                ((4, 2, 0), (4, 3, 0)),
                ((5, 3, 1),),
            ),
            (
                "ZZZZ",
                ((1, 0, 0), (1, 1, 0), (0, 0, 0), (0, 1, 0)),
                ((1, 1, 1),),
            ),
            (
                "ZZZZ",
                ((1, 2, 0), (1, 3, 0), (0, 2, 0), (0, 3, 0)),
                ((1, 3, 1),),
            ),
            (
                "ZZZZ",
                ((2, 1, 0), (2, 2, 0), (1, 1, 0), (1, 2, 0)),
                ((2, 2, 1),),
            ),
            (
                "ZZZZ",
                ((2, 3, 0), (2, 4, 0), (1, 3, 0), (1, 4, 0)),
                ((2, 4, 1),),
            ),
            (
                "ZZZZ",
                ((3, 0, 0), (3, 1, 0), (2, 0, 0), (2, 1, 0)),
                ((3, 1, 1),),
            ),
            (
                "ZZZZ",
                ((3, 2, 0), (3, 3, 0), (2, 2, 0), (2, 3, 0)),
                ((3, 3, 1),),
            ),
            (
                "ZZZZ",
                ((4, 1, 0), (4, 2, 0), (3, 1, 0), (3, 2, 0)),
                ((4, 2, 1),),
            ),
            (
                "ZZZZ",
                ((4, 3, 0), (4, 4, 0), (3, 3, 0), (3, 4, 0)),
                ((4, 4, 1),),
            ),
            (
                "XXXX",
                ((1, 1, 0), (0, 1, 0), (1, 2, 0), (0, 2, 0)),
                ((1, 2, 1),),
            ),
            (
                "XXXX",
                ((1, 3, 0), (0, 3, 0), (1, 4, 0), (0, 4, 0)),
                ((1, 4, 1),),
            ),
            (
                "XXXX",
                ((2, 0, 0), (1, 0, 0), (2, 1, 0), (1, 1, 0)),
                ((2, 1, 1),),
            ),
            (
                "XXXX",
                ((2, 2, 0), (1, 2, 0), (2, 3, 0), (1, 3, 0)),
                ((2, 3, 1),),
            ),
            (
                "XXXX",
                ((3, 1, 0), (2, 1, 0), (3, 2, 0), (2, 2, 0)),
                ((3, 2, 1),),
            ),
            (
                "XXXX",
                ((3, 3, 0), (2, 3, 0), (3, 4, 0), (2, 4, 0)),
                ((3, 4, 1),),
            ),
            (
                "XXXX",
                ((4, 0, 0), (3, 0, 0), (4, 1, 0), (3, 1, 0)),
                ((4, 1, 1),),
            ),
            (
                "XXXX",
                ((4, 2, 0), (3, 2, 0), (4, 3, 0), (3, 3, 0)),
                ((4, 3, 1),),
            ),
            # new hadamard stabilizers
            (
                "ZZ",
                ((1, 8, 0), (0, 8, 0)),
                ((1, 9, 1),),
            ),
            (
                "ZZ",
                ((3, 8, 0), (2, 8, 0)),
                ((3, 9, 1),),
            ),
            (
                "XX",
                ((4, 5, 0), (4, 6, 0)),
                ((5, 6, 1),),
            ),
            (
                "XX",
                ((4, 7, 0), (4, 8, 0)),
                ((5, 8, 1),),
            ),
            (
                "ZZ",
                ((0, 5, 0), (0, 6, 0)),
                ((0, 6, 1),),
            ),
            (
                "ZZ",
                ((0, 7, 0), (0, 8, 0)),
                ((0, 8, 1),),
            ),
            (
                "XXXX",
                ((1, 5, 0), (0, 5, 0), (1, 6, 0), (0, 6, 0)),
                ((1, 6, 1),),
            ),
            (
                "XXXX",
                ((1, 7, 0), (0, 7, 0), (1, 8, 0), (0, 8, 0)),
                ((1, 8, 1),),
            ),
            (
                "XXXX",
                ((2, 4, 0), (1, 4, 0), (2, 5, 0), (1, 5, 0)),
                ((2, 5, 1),),
            ),
            (
                "XXXX",
                ((2, 6, 0), (1, 6, 0), (2, 7, 0), (1, 7, 0)),
                ((2, 7, 1),),
            ),
            (
                "XXXX",
                ((3, 5, 0), (2, 5, 0), (3, 6, 0), (2, 6, 0)),
                ((3, 6, 1),),
            ),
            (
                "XXXX",
                ((3, 7, 0), (2, 7, 0), (3, 8, 0), (2, 8, 0)),
                ((3, 8, 1),),
            ),
            (
                "XXXX",
                ((4, 4, 0), (3, 4, 0), (4, 5, 0), (3, 5, 0)),
                ((4, 5, 1),),
            ),
            (
                "XXXX",
                ((4, 6, 0), (3, 6, 0), (4, 7, 0), (3, 7, 0)),
                ((4, 7, 1),),
            ),
            (
                "ZZZZ",
                ((1, 4, 0), (1, 5, 0), (0, 4, 0), (0, 5, 0)),
                ((1, 5, 1),),
            ),
            (
                "ZZZZ",
                ((1, 6, 0), (1, 7, 0), (0, 6, 0), (0, 7, 0)),
                ((1, 7, 1),),
            ),
            (
                "ZZZZ",
                ((2, 5, 0), (2, 6, 0), (1, 5, 0), (1, 6, 0)),
                ((2, 6, 1),),
            ),
            (
                "ZZZZ",
                ((2, 7, 0), (2, 8, 0), (1, 7, 0), (1, 8, 0)),
                ((2, 8, 1),),
            ),
            (
                "ZZZZ",
                ((3, 4, 0), (3, 5, 0), (2, 4, 0), (2, 5, 0)),
                ((3, 5, 1),),
            ),
            (
                "ZZZZ",
                ((3, 6, 0), (3, 7, 0), (2, 6, 0), (2, 7, 0)),
                ((3, 7, 1),),
            ),
            (
                "ZZZZ",
                ((4, 5, 0), (4, 6, 0), (3, 5, 0), (3, 6, 0)),
                ((4, 6, 1),),
            ),
            (
                "ZZZZ",
                ((4, 7, 0), (4, 8, 0), (3, 7, 0), (3, 8, 0)),
                ((4, 8, 1),),
            ),
        ]

        # Construct the expected stabilizers
        expected_stabilizers = [
            Stabilizer(pauli=args[0], data_qubits=args[1], ancilla_qubits=args[2])
            for args in expected_stabilizers_args
        ]

        # Check that the stabilizers are the same
        self.assertEqual(
            set(y_wall_out_block_v5z.stabilizers), set(expected_stabilizers)
        )

        # Check the number of syndrome circuits to be 7:
        # - 2 for the bulk stabilizers
        # - 1 for each of boundaries
        # - 1 extra because one topological corner is not at the geometric corner
        # and thus that boundary has both X and Z boundary stabilizers
        self.assertEqual(len(y_wall_out_block_v5z.syndrome_circuits), 9)

        # Check that the names of the syndrome circuits are correct
        synd_circuit_names = {
            synd_circ.name for synd_circ in y_wall_out_block_v5z.syndrome_circuits
        }
        synd_circuit_names_expected = {
            "non_triangular-bulk-xxxx",
            "non_triangular-bulk-zzzz",
            "non_triangular-left-zz",
            "non_triangular-right-zz",
            "non_triangular-top-xx",
            "triangular-bottom-zz",
            "triangular-bulk-xxxx",
            "triangular-bulk-zzzz",
            "triangular-right-xx",
        }

        self.assertEqual(synd_circuit_names, synd_circuit_names_expected)

    def test_y_wall_out_log_operator_evolution(self):
        """Test the log operator evolution of the interpretation step."""
        y_wall_out_interpretation_step = y_wall_out(
            self.base_interpretation_step(self.twisted_rsc_block_v3z),
            self.twisted_rsc_block_v3z,
            3,
            Orientation.HORIZONTAL,
            same_timeslice=False,
            debug_mode=True,
        )

        output_block = y_wall_out_interpretation_step.get_block(
            self.twisted_rsc_block_v3z.unique_label
        )

        # Obtain the log operator evolution
        log_x_evolution = y_wall_out_interpretation_step.logical_x_evolution
        log_z_evolution = y_wall_out_interpretation_step.logical_z_evolution

        # X logical operator should have evolved 3 times
        self.assertEqual(len(log_x_evolution), 3)
        # Z logical operator should have evolved 2 times
        self.assertEqual(len(log_z_evolution), 2)

        # Get the initial and final log x operators
        final_log_x_operator = output_block.logical_x_operators[0]
        _, stabs_for_operator_jump = find_new_x_logical_operator(
            self.twisted_rsc_block_v3z,
            is_top_left_bulk_stab_x=False,
            qubits_to_idle=[(i, j, 0) for i in range(3) for j in range(3)],
        )

        # Find the expected ones
        expected_stabs_args = (
            # The X stabilizers above the wall
            ("XX", [(1, 0, 0), (0, 0, 0)], [(1, 0, 1)]),
            ("XXXX", [(1, 1, 0), (0, 1, 0), (1, 2, 0), (0, 2, 0)], [(1, 2, 1)]),
            ("XXXX", [(2, 0, 0), (1, 0, 0), (2, 1, 0), (1, 1, 0)], [(2, 1, 1)]),
            ("XXXX", [(2, 2, 0), (1, 2, 0), (2, 3, 0), (1, 3, 0)], [(2, 3, 1)]),
            # The Z stabilizers above the wall
            ("ZZ", [(2, 2, 0), (2, 3, 0)], [(3, 3, 1)]),
            ("ZZ", [(2, 0, 0), (2, 1, 0)], [(3, 1, 1)]),
            ("ZZZZ", [(1, 2, 0), (1, 3, 0), (0, 2, 0), (0, 3, 0)], [(1, 3, 1)]),
            ("ZZZZ", [(2, 1, 0), (2, 2, 0), (1, 1, 0), (1, 2, 0)], [(2, 2, 1)]),
            ("ZZZZ", [(1, 0, 0), (1, 1, 0), (0, 0, 0), (0, 1, 0)], [(1, 1, 1)]),
            ("ZZ", [(0, 1, 0), (0, 2, 0)], [(0, 2, 1)]),
        )

        expected_stabs_for_operator_jump = [
            Stabilizer(pauli=args[0], data_qubits=args[1], ancilla_qubits=args[2])
            for args in expected_stabs_args
        ]
        expected_output_log_x_operator = PauliOperator(
            "X" * 3, [(2, 0, 0), (2, 1, 0), (2, 2, 0)]
        )

        # Check that the final log x operator and the stabilizers are correct
        self.assertEqual(final_log_x_operator, expected_output_log_x_operator)
        self.assertEqual(
            set(stabs_for_operator_jump), set(expected_stabs_for_operator_jump)
        )

    def test_final_syndrome_circuit_compilation(self):
        """
        Check that the final syndrome circuits can be used to measure syndromes.
        This is done by running the measureblocksyndromes function before and after
        the y_wall_out function and ensuring that no errors are raised.
        """
        for block, args in [
            (self.twisted_rsc_block_v5z, (5, Orientation.HORIZONTAL)),
            (self.twisted_rsc_block_v5x, (5, Orientation.HORIZONTAL)),
            (self.twisted_rsc_block_h5x, (5, Orientation.VERTICAL)),
            (self.twisted_rsc_block_v3z, (3, Orientation.HORIZONTAL)),
        ]:
            int_step = InterpretationStep(
                block_history=((block,),),
            )
            # Measure the syndromes of the initial block
            int_step = measureblocksyndromes(
                int_step,
                MeasureBlockSyndromes(block.unique_label),
                same_timeslice=False,
                debug_mode=True,
            )
            # Apply y_wall_out
            int_step = y_wall_out(
                int_step,
                block,
                *args,
                same_timeslice=False,
                debug_mode=True,
            )
            # Measure the syndromes again. This should not raise an error.
            int_step = measureblocksyndromes(
                int_step,
                MeasureBlockSyndromes(block.unique_label),
                same_timeslice=False,
                debug_mode=True,
            )

    def test_stabilizer_updates(self):
        """Test the stabilizer updates of the interpretation step for the y_wall_out
        operation."""

        y_wall_out_interpretation_step = y_wall_out(
            self.base_interpretation_step(self.twisted_rsc_block_v3z),
            self.twisted_rsc_block_v3z,
            3,
            Orientation.HORIZONTAL,
            same_timeslice=False,
            debug_mode=True,
        )

        # Expected stabilizer updates
        expected_stabilizers_to_update = [
            {(0, 2, 0), (1, 2, 0), (0, 3, 0), (1, 3, 0)},
            {(1, 2, 0), (2, 2, 0), (1, 3, 0), (2, 3, 0)},
        ]
        # Expected updates
        expected_stabilizer_updates = [
            (("c_(0, 3, 0)", 0), ("c_(1, 3, 0)", 0), 1),
            (("c_(1, 3, 0)", 0), ("c_(2, 3, 0)", 0), 1),
        ]

        # Stabilizer updates
        stabilizer_updates = y_wall_out_interpretation_step.stabilizer_updates

        for stab_uuid, cbits in stabilizer_updates.items():
            output_stab = y_wall_out_interpretation_step.stabilizers_dict[stab_uuid]

            # Check that the stabilizer is in the expected list
            self.assertIn(set(output_stab.data_qubits), expected_stabilizers_to_update)

            # Find the index
            idx = expected_stabilizers_to_update.index(set(output_stab.data_qubits))

            # Check that the cbits are correct
            self.assertEqual(cbits, expected_stabilizer_updates[idx])

    def test_logical_x_operator_updates_d3(self):
        """Test the logical x operator updates of the interpretation step for
        distance 3."""
        # Distance 3 - no parity change needed for

        base_int_step = self.base_interpretation_step(self.twisted_rsc_block_v3z)
        base_int_step.logical_x_operator_updates[
            self.twisted_rsc_block_v3z.logical_x_operators[0].uuid
        ] = (("mock_log_x_register", 0),)
        base_int_step.logical_z_operator_updates[
            self.twisted_rsc_block_v3z.logical_z_operators[0].uuid
        ] = (
            ("mock_log_z_register", 0),
            ("mock_log_z_register", 1),
        )

        y_wall_out_interpretation_step = y_wall_out(
            base_int_step,
            self.twisted_rsc_block_v3z,
            3,
            Orientation.HORIZONTAL,
            same_timeslice=False,
            debug_mode=True,
        )

        output_block = y_wall_out_interpretation_step.get_block(
            self.twisted_rsc_block_v3z.unique_label
        )

        # Get the logical operator updates
        output_logical_x_operator_updates = (
            y_wall_out_interpretation_step.logical_x_operator_updates[
                output_block.logical_x_operators[0].uuid
            ]
        )

        # Get the expected cbits for the logical x operator jump
        expected_cbits_for_logical_x_operator_jump = [
            cbit
            for measurements in [
                synd.measurements
                for synd in self.move_corner_step_v3z.syndromes
                if all(
                    # From the wall and above (idling part)
                    dq[1] <= 3
                    for dq in self.move_corner_step_v3z.stabilizers_dict[
                        synd.stabilizer
                    ].data_qubits
                )
                # The rounds are 0 as the first round of syndrome measurement
                and synd.round == 0
            ]
            for cbit in measurements
        ]

        expected_logical_x_operator_updates = (
            # Z contributions
            ("mock_log_z_register", 0),
            ("mock_log_z_register", 1),
            # wall bits
            ("c_(0, 3, 0)", 0),
            ("c_(1, 3, 0)", 0),
            ("c_(2, 3, 0)", 0),
            # potential parity change
            0,
            # X contributions from inheritance
            ("mock_log_x_register", 0),
            # contributions from teleportation circuits
            ("c_(2, 0, 0)", 0),
            ("c_(3, 3, 1)", 1),
        ) + tuple(expected_cbits_for_logical_x_operator_jump)

        # There should be only one logical operator update
        self.assertEqual(
            set(output_logical_x_operator_updates),
            set(expected_logical_x_operator_updates),
        )

    def test_logical_x_operator_updates_d5(self):
        """Test the logical x operator updates of the interpretation step for
        distance 5."""
        # Distance 5 - parity change needed for

        base_int_step = self.base_interpretation_step(self.twisted_rsc_block_v5z)
        base_int_step.logical_x_operator_updates[
            self.twisted_rsc_block_v5z.logical_x_operators[0].uuid
        ] = (("mock_x_log_register", 0),)
        base_int_step.logical_z_operator_updates[
            self.twisted_rsc_block_v5z.logical_z_operators[0].uuid
        ] = (
            ("mock_z_log_register", 0),
            ("mock_z_log_register", 1),
        )
        y_wall_out_interpretation_step = y_wall_out(
            base_int_step,
            self.twisted_rsc_block_v5z,
            5,
            Orientation.HORIZONTAL,
            same_timeslice=False,
            debug_mode=True,
        )

        output_block = y_wall_out_interpretation_step.get_block(
            self.twisted_rsc_block_v5z.unique_label
        )

        # Get the logical operator updates
        output_logical_x_operator_updates = (
            y_wall_out_interpretation_step.logical_x_operator_updates[
                output_block.logical_x_operators[0].uuid
            ]
        )

        # Get the cbits for the logical x operator jump
        expected_cbits_for_logical_x_operator_jump = [
            cbit
            for measurements in [
                synd.measurements
                for synd in self.move_corner_step_v5z.syndromes
                if all(
                    # From the wall and above (idling part)
                    dq[1] <= 5
                    for dq in self.move_corner_step_v5z.stabilizers_dict[
                        synd.stabilizer
                    ].data_qubits
                )
                # The rounds are 0 as the first round of syndrome measurement
                and synd.round == 0
            ]
            for cbit in measurements
        ]

        # Get the cbits for the logical x operator jump
        expected_logical_x_operator_updates = (
            # Z contributions
            ("mock_z_log_register", 0),
            ("mock_z_log_register", 1),
            # wall bits
            ("c_(0, 5, 0)", 0),
            ("c_(1, 5, 0)", 0),
            ("c_(2, 5, 0)", 0),
            ("c_(3, 5, 0)", 0),
            ("c_(4, 5, 0)", 0),
            # potential parity change
            1,
            # X contributions from inheritance
            ("mock_x_log_register", 0),
            # contributions from teleportation circuits
            ("c_(4, 0, 0)", 0),
            ("c_(5, 5, 1)", 1),
        ) + tuple(expected_cbits_for_logical_x_operator_jump)

        # There should be only one logical operator update
        self.assertEqual(
            set(output_logical_x_operator_updates),
            set(expected_logical_x_operator_updates),
        )


if __name__ == "__main__":
    unittest.main()
