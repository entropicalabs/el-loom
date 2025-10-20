"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest
from itertools import product


from loom.eka import Lattice
from loom.eka.utilities import Orientation, Direction
from loom.interpreter import InterpretationStep, Syndrome

from loom_rotated_surface_code.operations import LogicalPhaseViaYwall
from loom_rotated_surface_code.code_factory import RotatedSurfaceCode
from loom_rotated_surface_code.applicator import logical_phase_via_ywall, move_corners


class TestLogicalPhaseViaYwall(unittest.TestCase):
    """
    Test the applicator for the logical phase via Y wall operation of RotatedSurfaceCode
    blocks.
    """

    def setUp(self):
        # We need to first measure the block syndromes to get the necessary syndromes
        # for the logical phase via y-wall operation.
        self.base_int_step = lambda block: InterpretationStep(
            block_history=((block,),),
            syndromes=tuple(
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=((f"c_{stab.ancilla_qubits[0]}", 0),),
                    block=block.uuid,
                    round=0,
                    corrections=(),
                )
                for stab in block.stabilizers
            ),
        )

    def test_consistency_check(self):
        """
        Test the consistency check for the logical phase via y-wall operation."
        """
        # Create a block with non-square or even dimensions
        dx_dz_invalid = [(3, 5), (5, 3), (4, 4)]
        for dx, dz in dx_dz_invalid:
            block = RotatedSurfaceCode.create(
                dx=dx,
                dz=dz,
                lattice=Lattice.square_2d(),
            )

            with self.assertRaises(ValueError) as context:
                logical_phase_via_ywall(
                    self.base_int_step(block),
                    LogicalPhaseViaYwall(block.unique_label, Direction.RIGHT),
                    same_timeslice=False,
                    debug_mode=True,
                )
            self.assertIn(
                "Block must be square and have odd dimensions for the logical phase via"
                " y-wall operation.",
                str(context.exception),
            )

        growth_direction = Direction.TOP
        x_boundary = Orientation.HORIZONTAL
        # Test block with invalid Growth direction
        block = RotatedSurfaceCode.create(
            dx=5,
            dz=5,
            lattice=Lattice.square_2d(),
            x_boundary=x_boundary,
        )
        with self.assertRaises(ValueError) as context:
            logical_phase_via_ywall(
                self.base_int_step(block),
                LogicalPhaseViaYwall(block.unique_label, Direction.TOP),
                same_timeslice=False,
                debug_mode=True,
            )
        self.assertIn(
            f"The growth direction ({growth_direction.name}) must be parallel to the x "
            f"boundary ({x_boundary.name}) of the block for the logical phase via "
            "y-wall operation.",
            str(context.exception),
        )

        # Test block with moved corners
        block = RotatedSurfaceCode.create(
            dx=5,
            dz=5,
            lattice=Lattice.square_2d(),
        )

        current_int_step = move_corners(
            interpretation_step=self.base_int_step(block),
            block=block,
            corner_args=(
                (
                    (0, 0, 0),
                    Direction.RIGHT,
                    2,
                ),
            ),
            same_timeslice=False,
            debug_mode=True,
        )

        with self.assertRaises(ValueError) as context:
            logical_phase_via_ywall(
                current_int_step,
                LogicalPhaseViaYwall(block.unique_label, Direction.RIGHT),
                same_timeslice=False,
                debug_mode=True,
            )
        self.assertIn(
            "Topological corners must coincide with geometric corners for "
            "the logical phase via y-wall operation.",
            str(context.exception),
        )

    def test_valid_logical_phase(self):
        """Test that for a valid block, the logical phase via y-wall operation
        is applied correctly and returns the initial block defined the same.
        """

        distances = [(3, 3), (5, 5)]
        weight_2_stab_is_first_row_values = [False, True]

        for (dx, dz), growth_direction, weight_2_stab_is_first_row in product(
            distances, Direction, weight_2_stab_is_first_row_values
        ):

            input_block = RotatedSurfaceCode.create(
                dx=dx,
                dz=dz,
                lattice=Lattice.square_2d(),
                x_boundary=growth_direction.to_orientation(),
                weight_2_stab_is_first_row=weight_2_stab_is_first_row,
            )

            # Apply the logical phase via y-wall operation
            interpretation_step = logical_phase_via_ywall(
                self.base_int_step(input_block),
                LogicalPhaseViaYwall(input_block.unique_label, growth_direction),
                same_timeslice=False,
                debug_mode=True,
            )

            # Get output block
            output_block = interpretation_step.get_block(input_block.unique_label)

            # Check that the output block is the same as the input block
            self.assertEqual(input_block, output_block)


if __name__ == "__main__":
    unittest.main()
