"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest
from copy import deepcopy

from loom.eka.utilities import Direction, DiagonalDirection
from loom.eka import Lattice
from loom.interpreter import InterpretationStep, cleanup_final_step
from loom_rotated_surface_code.operations import MoveBlock
from loom_rotated_surface_code.code_factory import RotatedSurfaceCode
from loom_rotated_surface_code.applicator.move_block import (
    move_block,
    check_valid_move,
    direction_to_coord,
    composite_direction,
    update_qubit_coords,
)


class TestRotatedSurfaceCodeMoveBlock(unittest.TestCase):
    """
    Test class for the Rotated Surface Code MoveBlock Operation
    """

    def setUp(self):
        self.square_2d_lattice = Lattice.square_2d((12, 12))
        self.rot_surf_code_1 = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.square_2d_lattice,
            unique_label="q1",
            position=(4, 4),
        )
        self.rot_surf_code_2 = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.square_2d_lattice,
            unique_label="q2",
            position=(1, 1),
        )
        self.base_step = InterpretationStep(
            block_history=((self.rot_surf_code_1,),),
        )

    def test_check_valid_move(self):
        """
        For any direction
        """
        all_directions = [
            Direction.RIGHT,
            Direction.LEFT,
            Direction.TOP,
            Direction.BOTTOM,
        ]

        # Case 1: Simple validation check

        check_valid_move(
            occupied_qubits=((0, 2, 0), (1, 1, 0), (1, 1, 1), (0, 2, 1)),
            moving_qubits=((0, 1, 0), (0, 1, 1)),
            direction=Direction.TOP,
        )  # TOP
        check_valid_move(
            occupied_qubits=((0, 2, 0), (0, 0, 0), (0, 0, 1), (0, 2, 1)),
            moving_qubits=((0, 1, 0), (0, 1, 1)),
            direction=Direction.RIGHT,
        )  # RIGHT
        check_valid_move(
            occupied_qubits=((1, 1, 0), (2, 0, 0), (2, 0, 1), (1, 1, 1)),
            moving_qubits=((1, 0, 0), (1, 0, 1)),
            direction=Direction.LEFT,
        )  # LEFT
        check_valid_move(
            occupied_qubits=((1, 0, 0), (1, 0, 1)),
            moving_qubits=((0, 0, 0), (0, 0, 1)),
            direction=Direction.BOTTOM,
        )  # BOTTOM

        with self.assertRaises(ValueError) as message:
            check_valid_move(
                occupied_qubits=((0, 2, 0), (0, 2, 1)),
                moving_qubits=((0, 1, 0), (0, 1, 1)),
                direction=Direction.BOTTOM,
            )  # BOTTOM (Fail)

        # Check error message
        self.assertEqual(
            str(message.exception),
            "The move operation is invalid. The following qubit, (0, 1, 0), is moving "
            "to an occupied qubit, (0, 2, 0).",
        )

        # Case 1b: Simple validation check for Diagonal Movements
        check_valid_move(
            occupied_qubits=((0, 0, 1), (0, 1, 1), (1, 1, 1)),
            moving_qubits=((0, 0, 0),),
            direction=DiagonalDirection.TOP_RIGHT,
        )  # TOP RIGHT (Sub-Lattice = 0)
        check_valid_move(
            occupied_qubits=((0, 0, 0), (0, 1, 0), (1, 1, 0)),
            moving_qubits=((1, 1, 1),),
            direction=DiagonalDirection.TOP_RIGHT,
        )  # TOP RIGHT (Sub-Lattice = 1)
        check_valid_move(
            occupied_qubits=((0, 1, 1), (1, 1, 1), (1, 0, 1)),
            moving_qubits=((0, 0, 0),),
            direction=DiagonalDirection.TOP_LEFT,
        )  # TOP LEFT (Sub-Lattice = 0)
        check_valid_move(
            occupied_qubits=((1, 1, 0), (1, 0, 0), (0, 1, 0)),
            moving_qubits=((1, 1, 1),),
            direction=DiagonalDirection.TOP_LEFT,
        )  # TOP LEFT (Sub-Lattice = 1)

        check_valid_move(
            occupied_qubits=((0, 0, 1), (1, 0, 1), (0, 1, 1)),
            moving_qubits=((0, 0, 0),),
            direction=DiagonalDirection.BOTTOM_RIGHT,
        )  # BOTTOM RIGHT (Sub-Lattice = 0)
        check_valid_move(
            occupied_qubits=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
            moving_qubits=((1, 1, 1),),
            direction=DiagonalDirection.BOTTOM_RIGHT,
        )  # BOTTOM RIGHT (Sub-Lattice = 1)

        check_valid_move(
            occupied_qubits=((0, 0, 1), (1, 1, 1), (1, 0, 1)),
            moving_qubits=((0, 0, 0),),
            direction=DiagonalDirection.BOTTOM_LEFT,
        )  # BOTTOM LEFT (Sub-Lattice = 0)

        check_valid_move(
            occupied_qubits=((0, 0, 0), (1, 0, 0), (1, 1, 0)),
            moving_qubits=((1, 1, 1),),
            direction=DiagonalDirection.BOTTOM_LEFT,
        )  # BOTTOM LEFT (Sub-Lattice = 1)

        with self.assertRaises(ValueError) as message:
            check_valid_move(
                occupied_qubits=((1, 1, 1), (0, 2, 1)),
                moving_qubits=((0, 0, 0), (1, 1, 1)),
                direction=DiagonalDirection.BOTTOM_RIGHT,
            )  # BOTTOM LEFT (Fail)

        # Check error message
        self.assertEqual(
            str(message.exception),
            "The move operation is invalid. The following qubit, (0, 0, 0), is moving "
            "to an occupied qubit, (1, 1, 1).",
        )

        # Case 2: Full Block Clash check (Multiple qubits)
        starting_positions = [(7, 4), (1, 4), (4, 1), (4, 7)]

        for each_direction, each_starting_position in zip(
            all_directions, starting_positions, strict=True
        ):
            # Check that a False boolean is raised if the move cannot be made
            # (Clashing Block)
            clashing_block = RotatedSurfaceCode.create(
                dx=3,
                dz=3,
                lattice=self.square_2d_lattice,
                unique_label="q3",
                position=each_starting_position,
            )
            with self.assertRaises(ValueError):
                check_valid_move(
                    occupied_qubits=clashing_block.qubits,
                    moving_qubits=self.rot_surf_code_1.qubits,
                    direction=each_direction,
                )

            # Valid move (No Clashing Block)
            check_valid_move(
                occupied_qubits=self.rot_surf_code_2.qubits,
                moving_qubits=self.rot_surf_code_1.qubits,
                direction=each_direction,
            )

        # Case 4: ValueError raised if direction is not a Direction enum.
        with self.assertRaises(ValueError) as message:
            check_valid_move(
                occupied_qubits=((0, 0, 0),),
                moving_qubits=((0, 1, 0),),
                direction="top",
            )
        # Check error message
        self.assertEqual(
            str(message.exception),
            "Invalid direction. direction must be type Direction or set of Direction",
        )

    def test_composite_direction(self):
        """
        Correct decomposition
        """
        self.assertEqual(
            composite_direction("right"),
            (DiagonalDirection.TOP_RIGHT, DiagonalDirection.BOTTOM_RIGHT),
        )
        self.assertEqual(
            composite_direction("left"),
            (DiagonalDirection.TOP_LEFT, DiagonalDirection.BOTTOM_LEFT),
        )
        self.assertEqual(
            composite_direction("top"),
            (DiagonalDirection.TOP_LEFT, DiagonalDirection.TOP_RIGHT),
        )
        self.assertEqual(
            composite_direction("bottom"),
            (DiagonalDirection.BOTTOM_LEFT, DiagonalDirection.BOTTOM_RIGHT),
        )

        # Random Input (Error raised by Enum)
        with self.assertRaises(KeyError) as message:
            composite_direction("hello world")

        # Check error message
        self.assertEqual(str(message.exception), "'hello world'")

    def test_update_qubit_coords(self):
        """
        Check that the coordinates are updated correctly depending on the direction
        and their sub-lattice indices.
        """
        initial_coords = ((0, 1, 0), (1, 1, 1))
        direction = [
            Direction.TOP,
            Direction.BOTTOM,
            Direction.RIGHT,
            Direction.LEFT,
            DiagonalDirection.TOP_LEFT,
            DiagonalDirection.TOP_RIGHT,
            DiagonalDirection.BOTTOM_LEFT,
            DiagonalDirection.BOTTOM_RIGHT,
        ]
        test_coords = (
            ((0, 0, 0), (1, 0, 1)),
            ((0, 2, 0), (1, 2, 1)),
            ((1, 1, 0), (2, 1, 1)),
            ((-1, 1, 0), (0, 1, 1)),
            ((0, 1, 1), (0, 0, 0)),
            ((1, 1, 1), (1, 0, 0)),
            ((0, 2, 1), (0, 1, 0)),
            ((1, 2, 1), (1, 1, 0)),
        )

        for each_index, each_direction in enumerate(direction):
            self.assertEqual(
                update_qubit_coords(initial_coords, each_direction),
                test_coords[each_index],
            )

    def test_direction_to_coord(self):
        """
        Test that all Directions are properly translated into coordinates.
        """
        # For horizontal and vertical directions
        all_directions = [
            Direction.TOP,
            Direction.BOTTOM,
            Direction.RIGHT,
            Direction.LEFT,
        ]
        test_coords = [(0, -1, 0), (0, 1, 0), (1, 0, 0), (-1, 0, 0)]

        for each_index, each_direction in enumerate(all_directions):
            self.assertEqual(
                direction_to_coord(each_direction), test_coords[each_index]
            )

        # For diagonal directions (sub-lattice index = 0)
        all_directions = [
            DiagonalDirection.TOP_LEFT,
            DiagonalDirection.TOP_RIGHT,
            DiagonalDirection.BOTTOM_LEFT,
            DiagonalDirection.BOTTOM_RIGHT,
        ]
        test_coords_0 = [(0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
        test_coords_1 = [(-1, -1, -1), (0, -1, -1), (-1, 0, -1), (0, 0, -1)]

        for each_index, each_direction in enumerate(all_directions):
            self.assertEqual(
                direction_to_coord(each_direction, sublattice_index=0),
                test_coords_0[each_index],
            )
            self.assertEqual(
                direction_to_coord(each_direction, sublattice_index=1),
                test_coords_1[each_index],
            )

    def test_move_block(self):
        """
        The block is moved to the top in this test.
        The syndrome extraction circuits assign the qubits as follows:
        1. The first set of syndrome extraction rounds assigns the qubits as follows:
                     a
            x --- x --- x
         a  |  a  |  a  |
            x --- x --- x
            |  a  |  a  |  a
            x --- x --- x
               a
        where x are data channels and a are ancilla channels.
        2. The second set of syndrome extraction rounds creates a new set of channels
        BASED on how the channels are first created, in most cases as ancilla qubits of
        either the teleport circuits or that of the syndrome extraction rounds during
        the move.
        As such the new set of qubits are assigned as follows:
            a     a     a
               x --- x --- x
            a  |  a  |  a  |
               x --- x --- x
            a  |  a  |  a  |  a
               x --- x --- x
                  a

        2b. Following step:
               a     a     a
            a --- a --- a
         a  |  x  |  x  |  x
            a --- a --- a
            |  x  |  x  |  x
            a --- a --- a     a
               x     x     x
                  a

        2c. Final Expected Configuration:
                        s
               a --- a --- a
            a  |  a  |  a  |
         a     x --- x --- x
            a  |  a  |  a  |  s
               x --- x --- x
            a     a     a     a
               x     x     x
                  a
        where s are qubits assigned to a stabilizer but not part of the syndrome
        extraction/teleport circuits of the QEC-SWAP.
        These qubits labelled s will be used in the next syndrome extraction round.
        """
        # Only checking that the circuits are appended for the 4-body stabilizers.
        # Case 1: Move Block to the top
        final_step = move_block(
            deepcopy(self.base_step),
            MoveBlock("q1", direction=Direction.TOP),
            same_timeslice=False,
            debug_mode=True,
        )
        new_step = cleanup_final_step(final_step)

        # Change in the circuit
        self.assertNotEqual(
            new_step.final_circuit, cleanup_final_step(self.base_step).final_circuit
        )

        # In 1st move: 4-body stabilizer = 1 bit each (4 for 3x3 Block)
        #                   [SWAPs 4 data qubits]
        #              Teleport qubits = 1 bit each (3 for 3x3 Block)
        #                   [Teleports 3 data qubits]
        #              2-body stabilizer = 1 bit each (Only 2 relevant in 3x3 Block)
        #                   [SWAPs 2 data qubits]
        #                   [Stabilizers in the direction of movement help with swapping
        #                       data and ancilla qubit.]
        #              Total = 9 bits for "moving" data qubits and 2 bits for just
        #                   measuring ancilla qubits
        # 2 moves: 11 bits * 2 = 22 bits
        self.assertEqual(
            len(
                [
                    each_channel
                    for each_channel in new_step.final_circuit.channels
                    if each_channel.is_classical()
                ]
            ),
            22,
        )
        # There should be 24 qubits in total. 9 data qubits and 15 ancilla qubits.
        # Check description for why 9 data channels.
        self.assertEqual(
            len(
                [
                    each_channel
                    for each_channel in new_step.final_circuit.channels
                    if each_channel.is_quantum()
                ]
            ),
            24,
        )

        all_channel_labels = [
            each_channel.label for each_channel in new_step.final_circuit.channels
        ]
        self.assertEqual(len(all_channel_labels), len(set(all_channel_labels)))
        # Check for identical channel labels (No overlap channels.
        # Same label different types.)

        # Check updated Block
        updated_block = new_step.block_history[-1][0]
        original_block = new_step.block_history[0][0]
        self.assertNotEqual(original_block, updated_block)

        # Test against the expected Blocks
        expected_block = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.square_2d_lattice,
            unique_label="q1",
            position=(4, 3),
        )
        self.assertEqual(updated_block, expected_block)

    def test_move_block_all_dir(self):
        """
        Check that all directions can be applied with no errors:
        Move Block to the top, left, right, bottom
        """
        directions = [Direction.RIGHT, Direction.LEFT, Direction.TOP, Direction.BOTTOM]
        direction_vectors = [(1, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 1, 0)]

        for each_direction, each_direction_vector in zip(
            directions, direction_vectors, strict=True
        ):
            final_step = move_block(
                deepcopy(self.base_step),
                MoveBlock("q1", direction=each_direction),
                same_timeslice=False,
                debug_mode=True,
            )
            new_step = cleanup_final_step(final_step)

            # Check updated Block
            updated_block = new_step.block_history[-1][0]
            original_block = new_step.block_history[0][0]
            self.assertNotEqual(original_block, updated_block)

            # Test against the expected Block
            new_starting_position = tuple(
                a + b for a, b in zip((4, 4), each_direction_vector[:-1], strict=True)
            )
            expected_block = RotatedSurfaceCode.create(
                dx=3,
                dz=3,
                lattice=self.square_2d_lattice,
                unique_label="q1",
                position=new_starting_position,
            )
            self.assertEqual(updated_block, expected_block)
