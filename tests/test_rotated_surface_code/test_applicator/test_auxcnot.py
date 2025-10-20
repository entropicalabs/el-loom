"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=duplicate-code
import unittest
from copy import deepcopy

from loom.eka import Eka
from loom.eka.utilities import Orientation, Direction, loads, dumps
from loom.eka.operations import (
    Merge,
    Grow,
    Shrink,
    Split,
    MeasureBlockSyndromes,
    Operation,
)
from loom.eka.lattice import Lattice
from loom.eka.utilities import uuid_error
from loom.interpreter import InterpretationStep, interpret_eka, Syndrome

from loom_rotated_surface_code.applicator.auxcnot import (
    auxcnot,
    auxcnot_consistency_check,
    get_grow_shrink_directions,
    auxcnot_grow_control,
    auxcnot_split_control,
    auxcnot_merge_aux_target,
    auxcnot_shrink_target,
)
from loom_rotated_surface_code.code_factory import RotatedSurfaceCode
from loom_rotated_surface_code.operations import AuxCNOT


class TestRotatedSurfaceCodeAuxCNOT(unittest.TestCase):
    """
    Test cases for the AuxCNOT operation in the Rotated Surface Code.
    """

    def setUp(self):
        self.lattice = Lattice.square_2d((8, 8))
        self.control = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice,
            unique_label="c",
        )
        self.target = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            position=(4, 4),
            lattice=self.lattice,
            unique_label="t",
        )
        self.base_step = InterpretationStep()
        self.base_step.block_history = ((self.control, self.target),)

    def test_op_aux_cnot(self):
        """Test the creation of an AuxCNOT operation"""
        # pylint: disable=protected-access
        # Test the creation of an AuxCNOT operation
        aux_cnot = AuxCNOT(input_blocks_name=["q1", "q2"])
        self.assertEqual(aux_cnot.input_blocks_name, ("q1", "q2"))
        self.assertEqual(aux_cnot.__class__.__name__, "AuxCNOT")

        # Test the loads/dumps both using the right class and the abstract base class
        self.assertEqual(aux_cnot, loads(AuxCNOT, dumps(aux_cnot)))
        self.assertEqual(aux_cnot, loads(Operation, dumps(aux_cnot)))

    def test_auxcnot_invalid_size(self):
        """Test that the AuxCNOT operation raises an error for invalid sizes."""
        # This target has invalid size
        target_invalid_size = RotatedSurfaceCode.create(
            dx=2,  # Invalid size
            dz=3,
            position=(4, 4),
            lattice=self.lattice,
            unique_label="t",
        )
        with self.assertRaises(ValueError) as cm:
            auxcnot_consistency_check(
                c_block=self.control,
                t_block=target_invalid_size,
            )
        err_msg = (
            f"The blocks must have the same size to perform the auxiliary CNOT "
            f"operation. The sizes of the blocks are {self.control.size}, "
            f"{target_invalid_size.size}."
        )
        self.assertIn(err_msg, str(cm.exception))

    def test_auxcnot_invalid_boundary(self):
        """
        Test that the AuxCNOT operation raises an error for invalid boundary
        orientations.
        """
        # This target has an invalid boundary orientation
        target_invalid_boundary = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            position=(4, 4),
            lattice=self.lattice,
            unique_label="t",
            x_boundary=Orientation.VERTICAL,  # Invalid boundary orientation
        )
        with self.assertRaises(ValueError) as cm:
            auxcnot_consistency_check(
                c_block=self.control,
                t_block=target_invalid_boundary,
            )
        err_msg = (
            "The blocks must have the same boundary orientations to perform the "
            "auxiliary CNOT operation. The X boundary orientations are "
            f"{self.control.x_boundary} and {target_invalid_boundary.x_boundary}."
        )
        self.assertIn(err_msg, str(cm.exception))

    def test_auxcnot_invalid_positions(self):
        """Test that the AuxCNOT operation raises an error for invalid positions."""
        # With these blocks an error should be raised because the positions are invalid
        target_invalid_position = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            position=(5, 0),  # Incorrect position
            lattice=self.lattice,
            unique_label="t",
        )
        with self.assertRaises(ValueError) as cm:
            auxcnot_consistency_check(
                c_block=self.control,
                t_block=target_invalid_position,
            )
        err_msg = (
            "The blocks are not in the correct configuration for the auxiliary CNOT "
            "operation. The upper left corners of the blocks must satisfy the "
            "following relations: \n"
            "|t_block.upper_left_qubit[0] - c_block.upper_left_qubit[0]| = "
            "c_block.size[0] + 1, |t_block.upper_left_qubit[1] - "
            "c_block.upper_left_qubit[1]| = c_block.size[1] + 1"
            f"\nGot |{target_invalid_position.upper_left_qubit[0]} - "
            f"{self.control.upper_left_qubit[0]}| = {self.control.size[0] + 1}, "
            f"|{target_invalid_position.upper_left_qubit[1]} - "
            f"{self.control.upper_left_qubit[1]}| = {self.control.size[1] + 1} instead."
        )
        self.assertIn(err_msg, str(cm.exception))

    def test_auxcnot_get_directions(self):
        """
        Test that the get_grow_shrink_directions function returns the correct
        grow and shrink directions based on the control and target blocks.
        """
        # Get the grow and shrink directions for the regular target and control blocks
        grow_direction, shrink_direction = get_grow_shrink_directions(
            control=self.control,
            target=self.target,
        )
        # Check that the directions are correct
        self.assertEqual(grow_direction, Direction.RIGHT)
        self.assertEqual(shrink_direction, Direction.TOP)

        # Get the grow and shrink directions when inverting the inputs
        grow_direction, shrink_direction = get_grow_shrink_directions(
            control=self.target,
            target=self.control,
        )
        # Check that the directions are correct
        self.assertEqual(grow_direction, Direction.LEFT)
        self.assertEqual(shrink_direction, Direction.BOTTOM)

    def test_auxcnot_grow_control(self):
        """
        Test that the auxcnot_grow_control function correctly grows the control
        block.
        """
        # Apply the grow function
        new_step = auxcnot_grow_control(
            interpretation_step=deepcopy(self.base_step),
            control=self.control,
            target=self.target,
            grow_direction=Direction.RIGHT,
            same_timeslice=False,
            debug_mode=True,
        )
        # Check that the control block has been grown correctly
        self.assertEqual(len(new_step.block_history), 2)
        grown_control = new_step.get_block("c")
        untouched_target = new_step.get_block("t")
        expected_grown_control = RotatedSurfaceCode.create(
            dx=7,
            dz=3,
            lattice=self.lattice,
            unique_label="c",
        )
        self.assertEqual(grown_control, expected_grown_control)
        self.assertEqual(untouched_target, self.target)

    def test_auxcnot_split_control(self):
        """
        Test that the auxcnot_split_control function correctly splits the grown
        control block into control and auxiliary blocks.
        """
        # Create a grown control block
        grown_control = RotatedSurfaceCode.create(
            dx=7,
            dz=3,
            lattice=self.lattice,
            unique_label="c",
        )
        grown_step = InterpretationStep(
            block_history=((grown_control, self.target),),
            syndromes=tuple(
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=((f"c_{stab.ancilla_qubits[0]}", 0),),
                    block=grown_control.uuid,
                    round=0,
                )
                for stab in grown_control.stabilizers
            ),
        )
        # Apply the split function
        new_step = auxcnot_split_control(
            interpretation_step=grown_step,
            aux_unique_label="aux",
            initial_control=self.control,
            initial_target=self.target,
            grown_control=grown_control,
            same_timeslice=False,
            debug_mode=True,
        )
        # Check that the control and auxiliary blocks have been created correctly
        control_block = new_step.get_block("c")
        aux_block = new_step.get_block("aux")
        expected_control_block = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice,
            unique_label="c",
        )
        expected_aux_block = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice,
            position=(4, 0),
            unique_label="aux",
        )
        self.assertEqual(control_block, expected_control_block)
        self.assertEqual(aux_block, expected_aux_block)

    def test_auxcnot_merge_aux_target(self):
        """
        Test that the auxcnot_merge_aux_target function correctly merges the
        auxiliary block with the target block.
        """
        # Create an auxiliary block
        aux_block = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice,
            position=(4, 0),
            unique_label="aux",
        )
        split_step = InterpretationStep(
            block_history=((aux_block, self.target),),
            syndromes=tuple(
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=((f"c_{stab.ancilla_qubits[0]}", 0),),
                    block=aux_block.uuid,
                    round=0,
                )
                for stab in aux_block.stabilizers
            ),
        )
        # Apply the merge function
        new_step = auxcnot_merge_aux_target(
            interpretation_step=split_step,
            aux=aux_block,
            target=self.target,
            same_timeslice=False,
            debug_mode=True,
        )
        # Check that the auxiliary block has been merged with the target block
        merged_target = new_step.get_block("t")
        expected_target = RotatedSurfaceCode.create(
            dx=3,
            dz=7,
            lattice=self.lattice,
            position=(4, 0),
            unique_label="t",
        )
        self.assertEqual(merged_target, expected_target)

    def test_auxcnot_shrink_target(self):
        """
        Test that the auxcnot_shrink_target function correctly shrinks the target
        block.
        """
        # Create a merged target block
        merged_target = RotatedSurfaceCode.create(
            dx=3,
            dz=7,
            lattice=self.lattice,
            position=(4, 0),
            unique_label="t",
        )
        merge_step = InterpretationStep(
            block_history=((self.control, merged_target),),
            syndromes=tuple(
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=((f"c_{stab.ancilla_qubits[0]}", 0),),
                    block=merged_target.uuid,
                    round=0,
                )
                for stab in merged_target.stabilizers
            ),
        )
        # Apply the shrink function
        new_step = auxcnot_shrink_target(
            interpretation_step=merge_step,
            initial_target=self.target,
            merged_target=merged_target,
            shrink_direction=Direction.TOP,
            same_timeslice=False,
            debug_mode=True,
        )
        # Check that the target block has been shrunk correctly
        shrunk_target = new_step.get_block("t")
        expected_shrunk_target = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice,
            position=(4, 4),
            unique_label="t",
        )
        self.assertEqual(shrunk_target, expected_shrunk_target)

    def test_valid_auxcnot_circuit(self):
        """
        Test that a valid AuxCNOT operation generates the correct circuit.
        """
        # Get the AuxCNOT operation circuit
        aux_cnot = AuxCNOT(["c", "t"])
        auxcnot_eka = Eka(
            self.lattice,
            blocks=[self.control, self.target],
            operations=[aux_cnot],
        )
        auxcnot_int_step = interpret_eka(auxcnot_eka)
        auxcnot_circuit = auxcnot_int_step.final_circuit

        # Get the expected circuit
        expected_eka = Eka(
            self.lattice,
            blocks=[self.control, self.target],
            operations=[
                [Grow("c", length=4, direction="right")],
                [MeasureBlockSyndromes("c", 3), MeasureBlockSyndromes("t", 3)],
                [Split("c", ["c", "c_aux"], orientation="vertical", split_position=3)],
                [
                    MeasureBlockSyndromes("c", 1),
                    MeasureBlockSyndromes("c_aux", 1),
                    MeasureBlockSyndromes("t", 1),
                ],
                [Merge(["t", "c_aux"], "t"), MeasureBlockSyndromes("c", 1)],
                [MeasureBlockSyndromes("c", 2), MeasureBlockSyndromes("t", 2)],
                [Shrink("t", direction="top", length=4)],
            ],
        )
        expected_int_step = interpret_eka(expected_eka)
        expected_circuit = expected_int_step.final_circuit

        # Check that the circuits are the same
        self.assertEqual(auxcnot_circuit.circuit[0][0], expected_circuit)

        # Check that the duration of the AuxCNOT operation is correct
        # 1 for grow, 8 for syndrome measurements, 2 for split (x-basis),
        # 8 for syndrome measurements, 1 for merge, 8 for syndrome measurement after
        # merge, and 1 for shrink for a total is 1 + 8*3 + 2 + 8 + 1 + 8*3 + 1 = 61
        self.assertEqual(auxcnot_circuit.circuit[0][0].duration, 61)

        # Check that the blocks are still defined the same way after the AuxCNOT
        for block in [self.control, self.target]:
            self.assertEqual(block, auxcnot_int_step.get_block(block.unique_label))

    def test_parallel_auxcnot(self):
        """
        Test that AuxCNOT operations can be executed in parallel and be scheduled in
        the correct order.
        """
        bigger_lattice = Lattice.square_2d((12, 8))

        extra_block = RotatedSurfaceCode.create(
            dx=3, dz=7, position=(8, 0), lattice=bigger_lattice, unique_label="extra"
        )

        aux_cnot = AuxCNOT(["c", "t"])
        single_measurement = MeasureBlockSyndromes("extra", n_cycles=4)

        eka = Eka(
            bigger_lattice,
            blocks=[self.control, self.target, extra_block],
            operations=[(aux_cnot, single_measurement)],
        )

        final_step = interpret_eka(eka)
        parallel_circuit = final_step.final_circuit

        # Check that the parallel circuit implements the two operations in the same
        # timestep
        self.assertEqual(
            [circ.name for circ in parallel_circuit.circuit[0]],
            [
                "auxcnot between c and t",
                "measure extra syndromes 4 time(s)",
            ],
        )

    def test_auxcnot_wrapper(self):
        """Test the auxcnot function for given input updates and check that they
        propagate correctly."""
        new_lattice = Lattice.square_2d((11, 11))
        pos_control = (4, 4)
        d = 3
        for c_to_t_vector in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            pos_target = (
                pos_control[0] + c_to_t_vector[0] * (d + 1),
                pos_control[1] + c_to_t_vector[1] * (d + 1),
            )
            new_control = RotatedSurfaceCode.create(
                d, d, new_lattice, "new_c", position=pos_control
            )
            new_target = RotatedSurfaceCode.create(
                d, d, new_lattice, "new_t", position=pos_target
            )
            new_step = InterpretationStep(block_history=((new_control, new_target),))
            new_step.logical_x_operator_updates = {
                new_control.logical_x_operators[0].uuid: (("dummy_X_control", 0),),
                new_target.logical_x_operators[0].uuid: (("dummy_X_target", 0),),
            }
            new_step.logical_z_operator_updates = {
                new_control.logical_z_operators[0].uuid: (("dummy_Z_control", 0),),
                new_target.logical_z_operators[0].uuid: (("dummy_Z_target", 0),),
            }

            op = AuxCNOT(input_blocks_name=("new_c", "new_t"))
            new_int_step = auxcnot(new_step, op, same_timeslice=False, debug_mode=True)

            new_target = new_int_step.get_block("new_t")
            new_control = new_int_step.get_block("new_c")

            def err_message(p, s, c_to_t_vector):
                message = (
                    f"\n{p} update for {s} not found for c_to_t_vector: "
                    f"{c_to_t_vector}"
                )
                return message

            # Check that the previous Z updates of the control is propagated
            self.assertIn(
                ("dummy_Z_control", 0),
                new_int_step.logical_z_operator_updates[
                    new_control.logical_z_operators[0].uuid
                ],
                err_message("Z", "control", c_to_t_vector),
            )
            # Check that all previous Z updates are propagated to the new target
            for previous_z_update in (("dummy_Z_target", 0), ("dummy_Z_control", 0)):
                self.assertIn(
                    previous_z_update,
                    new_int_step.logical_z_operator_updates[
                        new_target.logical_z_operators[0].uuid
                    ],
                    err_message("Z", "target", c_to_t_vector),
                )
            # Check that all previous X updates are propagated to the new control
            for previous_x_update in (("dummy_X_control", 0), ("dummy_X_target", 0)):
                self.assertIn(
                    previous_x_update,
                    new_int_step.logical_x_operator_updates[
                        new_control.logical_x_operators[0].uuid
                    ],
                    err_message("X", "control", c_to_t_vector),
                )
            # Check that the previous X updates of the control is propagated
            self.assertIn(
                ("dummy_X_target", 0),
                new_int_step.logical_x_operator_updates[
                    new_target.logical_x_operators[0].uuid
                ],
                err_message("X", "target", c_to_t_vector),
            )

    def test_auxcnot_renaming_aux(self):
        """
        Test that the AuxCNOT operation correctly renames the auxiliary block if a block
        with the same name already exists.
        """
        # Create a new AuxCNOT operation with a different auxiliary block name
        aux_cnot = AuxCNOT(["c", "t"])
        lattice = Lattice.square_2d((20, 20))
        same_name_block = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            position=(10, 0),
            lattice=lattice,
            unique_label="c_aux",
        )
        eka = Eka(
            lattice,
            blocks=[self.control, self.target, same_name_block],
            operations=[aux_cnot],
        )

        final_step = interpret_eka(eka)

        # Check that the auxiliary block is named with a uuid if the name already exists
        expected_uuid_name = next(
            block.unique_label
            for block in final_step.block_history[2]
            if block.unique_label
            not in (
                self.control.unique_label,
                self.target.unique_label,
                f"{self.control.unique_label}_aux",
            )
        )
        # Check that the auxiliary block has a valid uuid
        uuid_error(expected_uuid_name)


if __name__ == "__main__":
    unittest.main()
