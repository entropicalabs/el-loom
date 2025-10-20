"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest
from copy import deepcopy
from itertools import product

from loom.eka import Lattice
from loom.eka.operations import Shrink
from loom.interpreter import InterpretationStep, Syndrome, Detector
from loom.eka.utilities import Orientation, Direction

from loom_rotated_surface_code.code_factory import RotatedSurfaceCode
from loom_rotated_surface_code.applicator.shrink import (
    shrink,
    shrink_consistency_checks,
)


class TestRotatedSurfaceCodeShrink(unittest.TestCase):
    """
    Tests the shrink operation on a RotatedSurfaceCode.
    """

    def setUp(self):
        self.square_2d_lattice = Lattice.square_2d((10, 20))
        self.rot_surf_code_1 = RotatedSurfaceCode.create(
            dx=3, dz=3, lattice=self.square_2d_lattice, unique_label="q1"
        )
        self.rot_surf_code_2 = RotatedSurfaceCode.create(
            dx=5, dz=5, lattice=self.square_2d_lattice, unique_label="q2"
        )
        self.base_step = InterpretationStep(
            block_history=((self.rot_surf_code_1,),),
            syndromes=[
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=[(f"c_{stab.ancilla_qubits[0]}", 0)],
                    block=self.rot_surf_code_1.uuid,
                    round=0,
                )
                for stab in self.rot_surf_code_1.stabilizers
            ],
            logical_x_operator_updates={
                self.rot_surf_code_1.logical_x_operators[0].uuid: (("dummy_X", 0),)
            },
            logical_z_operator_updates={
                self.rot_surf_code_1.logical_z_operators[0].uuid: (("dummy_Z", 0),)
            },
        )
        # Create an initial step with a different block
        self.other_base_step = InterpretationStep(
            block_history=((self.rot_surf_code_2,),),
            syndromes=[
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=[(f"c_{stab.ancilla_qubits[0]}", 0)],
                    block=self.rot_surf_code_2.uuid,
                    round=0,
                )
                for stab in self.rot_surf_code_2.stabilizers
            ],
            logical_x_operator_updates={
                self.rot_surf_code_2.logical_x_operators[0].uuid: (("dummy_X", 0),)
            },
            logical_z_operator_updates={
                self.rot_surf_code_2.logical_z_operators[0].uuid: (("dummy_Z", 0),)
            },
        )

    def test_shrink_consistency_checks(self):
        """Tests that the shrink operation is consistent with the block."""
        block = self.rot_surf_code_1
        int_step = InterpretationStep(
            block_history=((block,),),
        )

        for direction, add_length in product(
            ("left", "right", "top", "bottom"), (-1, 0, 6)
        ):
            block_length = block.size[
                (
                    0
                    if Orientation.from_direction(Direction(direction))
                    == Orientation.HORIZONTAL
                    else 1
                )
            ]
            shrink_length = block_length + add_length
            shrink_op = Shrink(
                input_block_name=block.unique_label,
                direction=direction,
                length=shrink_length,
            )
            # Check that the shrink operation is consistent with the block
            err_msg = (
                f"Shrink length {shrink_op.length} is not valid. "
                f"Must be between 1 and {block_length - 2} for the selected block."
            )
            with self.assertRaises(ValueError) as cm:
                shrink_consistency_checks(int_step, shrink_op)
            self.assertIn(err_msg, str(cm.exception))

    def test_applicator_shrink(self):
        """Tests that the shrink operation is correctly applied."""
        shrink_op = Shrink(
            input_block_name=self.rot_surf_code_1.unique_label,
            direction="right",
            length=1,
        )
        final_step = shrink(
            deepcopy(self.base_step), shrink_op, same_timeslice=False, debug_mode=True
        )
        final_block = final_step.get_block(self.rot_surf_code_1.unique_label)

        # Check that the syndromes are correct
        fully_measured_stabs = tuple(
            stab
            for stab in self.rot_surf_code_1.stabilizers
            if (
                all(q not in final_block.data_qubits for q in stab.data_qubits)
                and set(stab.pauli) == {"X"}
            )
        )
        expected_syndromes = tuple(
            Syndrome(
                stabilizer=stab.uuid,
                measurements=tuple((f"c_{q}", 0) for q in stab.data_qubits),
                corrections=(),
                block=self.rot_surf_code_1.uuid,
                round=0,
            )
            for stab in fully_measured_stabs
        )
        self.assertEqual(
            self.base_step.syndromes + expected_syndromes, final_step.syndromes
        )
        # Check that the detectors are correct
        expected_detectors = tuple(
            Detector(
                tuple(
                    syndrome
                    for syndrome in final_step.syndromes
                    if syndrome.stabilizer == stab.uuid
                )
            )
            for stab in fully_measured_stabs
        )
        self.assertEqual(expected_detectors, final_step.detectors)

        # Check that the final block is as expected
        expected_block = RotatedSurfaceCode.create(
            dx=2,
            dz=3,
            lattice=self.square_2d_lattice,
            unique_label="q1",
        )
        self.assertEqual(final_block, expected_block)

        self.assertEqual(
            final_step.intermediate_circuit_sequence[0][0].name,
            "shrink q1 by 1 from right",
        )
        # First there should be 3 Hadamard gates to measure the data qubits in the
        # right basis
        self.assertEqual(
            {
                (
                    gate.name,
                    tuple(channel.label for channel in gate.channels),
                )
                for gate in final_step.intermediate_circuit_sequence[0][0].circuit[0]
            },
            {("h", ("(2, 0, 0)",)), ("h", ("(2, 1, 0)",)), ("h", ("(2, 2, 0)",))},
        )
        # Then there should be 3 measurements of the three right most data qubits
        self.assertEqual(
            {
                (
                    gate.name,
                    tuple(channel.label for channel in gate.channels),
                )
                for gate in final_step.intermediate_circuit_sequence[0][0].circuit[1]
            },
            {
                ("measurement", ("(2, 0, 0)", "c_(2, 0, 0)_0")),
                ("measurement", ("(2, 1, 0)", "c_(2, 1, 0)_0")),
                ("measurement", ("(2, 2, 0)", "c_(2, 2, 0)_0")),
            },
        )

        # The Z operator didn't change, therefore it's update is the same as the
        # previous one
        self.assertEqual(
            final_step.logical_z_operator_updates,
            self.base_step.logical_z_operator_updates,
        )

        # One data qubit of the X operator was measured which should therefore be
        # included in the update list.
        self.assertEqual(
            final_step.logical_x_operator_updates[
                final_block.logical_x_operators[0].uuid
            ],
            (("dummy_X", 0), ("c_(2, 0, 0)", 0)),
        )

    def test_applicator_shrink_on_logical_op(self):
        """Test that the shrink operation is correctly applied to a block when we need
        to move the logical operator within the block"""
        shrink_op = Shrink(
            input_block_name=self.rot_surf_code_2.unique_label,
            direction="left",
            length=2,
        )
        final_step = shrink(
            deepcopy(self.other_base_step),
            shrink_op,
            same_timeslice=False,
            debug_mode=True,
        )
        final_block = final_step.get_block(self.rot_surf_code_2.unique_label)
        expected_x_op_evolution = {
            final_block.logical_x_operators[0].uuid: (
                self.rot_surf_code_2.logical_x_operators[0].uuid,
            )
        }
        # Check that the syndromes are correct
        fully_measured_stabs = tuple(
            stab
            for stab in self.rot_surf_code_2.stabilizers
            if (
                all(q not in final_block.data_qubits for q in stab.data_qubits)
                and set(stab.pauli) == {"X"}
            )
        )
        expected_syndromes = tuple(
            Syndrome(
                stabilizer=stab.uuid,
                measurements=tuple((f"c_{q}", 0) for q in stab.data_qubits),
                corrections=(),
                block=self.rot_surf_code_2.uuid,
                round=0,
            )
            for stab in fully_measured_stabs
        )
        self.assertEqual(
            self.other_base_step.syndromes + expected_syndromes, final_step.syndromes
        )
        # Check that the detectors are correct
        expected_detectors = tuple(
            Detector(
                tuple(
                    syndrome
                    for syndrome in final_step.syndromes
                    if syndrome.stabilizer == stab.uuid
                )
            )
            for stab in fully_measured_stabs
        )
        self.assertEqual(expected_detectors, final_step.detectors)

        # The equivalent stabilizers are located on the border that we shrink
        _, eq_z_stabs = self.rot_surf_code_2.get_shifted_equivalent_logical_operator(
            self.rot_surf_code_2.logical_z_operators[0], new_upleft_qubit=(2, 0, 0)
        )
        expected_z_op_evolution = {
            final_block.logical_z_operators[0].uuid: (
                self.rot_surf_code_2.logical_z_operators[0].uuid,
            )
            + tuple(stab.uuid for stab in eq_z_stabs)
        }
        # Check the operator evolution and updates
        self.assertEqual(final_step.logical_x_evolution, expected_x_op_evolution)
        self.assertEqual(
            final_step.logical_x_operator_updates,
            {
                final_block.logical_x_operators[0].uuid: (
                    ("dummy_X", 0),
                    ("c_(0, 0, 0)", 0),
                    ("c_(1, 0, 0)", 0),
                )
            },
        )
        self.assertEqual(final_step.logical_z_evolution, expected_z_op_evolution)
        expected_z_updates = {
            self.rot_surf_code_2.logical_z_operators[0].uuid: (("dummy_Z", 0),),
            final_block.logical_z_operators[0].uuid: (
                ("c_(1, 1, 1)", 0),
                ("c_(1, 3, 1)", 0),
                ("c_(2, 2, 1)", 0),
                ("c_(2, 4, 1)", 0),
                ("c_(2, 0, 1)", 0),
                ("c_(1, 5, 1)", 0),
                ("dummy_Z", 0),
            ),
        }
        self.assertEqual(final_step.logical_z_operator_updates, expected_z_updates)


if __name__ == "__main__":
    unittest.main()
