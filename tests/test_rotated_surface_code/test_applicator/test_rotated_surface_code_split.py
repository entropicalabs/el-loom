"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest
from copy import deepcopy
from itertools import product

from loom.eka import Circuit, Channel, Lattice, Stabilizer, PauliOperator
from loom.eka.utilities import Direction, Orientation
from loom.eka.operations import Split
from loom.interpreter import InterpretationStep, Syndrome

from loom_rotated_surface_code.code_factory import RotatedSurfaceCode
from loom_rotated_surface_code.applicator.split import (
    split_consistency_check,
    create_split_circuit,
    split_stabilizers,
    find_split_stabilizer_to_circuit_mappings,
    split_logical_operators,
    split,
)


class TestRotatedSurfaceCodeSplit(unittest.TestCase):
    """
    Test the applicator for the split operation of RotatedSurfaceCode block.
    """

    def setUp(self):
        self.lattice = Lattice.square_2d((7, 7))
        self.block_1 = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice,
            unique_label="q1",
        )
        self.block_2 = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice,
            unique_label="q2",
        ).shift((4, 0))
        self.big_block = RotatedSurfaceCode.create(
            dx=7,
            dz=3,
            lattice=self.lattice,
            unique_label="big_block",
        )
        self.base_step = InterpretationStep(
            block_history=((self.big_block,),),
            syndromes=[
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=[(f"c_{stab.ancilla_qubits[0]}", 0)],
                    block=self.big_block.uuid,
                    round=0,
                )
                for stab in self.big_block.stabilizers
            ],
        )
        self.big_base_step = InterpretationStep(
            block_history=((self.big_block,),),
            syndromes=[
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=[(f"c_{stab.ancilla_qubits[0]}", 0)],
                    block=self.big_block.uuid,
                    round=0,
                )
                for stab in self.big_block.stabilizers
            ],
            logical_x_operator_updates={
                self.big_block.logical_x_operators[0].uuid: (("dummy_X", 0),),
            },
            logical_z_operator_updates={
                self.big_block.logical_z_operators[0].uuid: (("dummy_Z", 0),),
            },
        )

    def test_split_consistency_check(self):
        "Test that the split_consistency_check function raises all the expected errors."

        # 1 - Test a horizontal split where the split position is larger than the width
        # of the block
        interpretation_step = InterpretationStep(
            block_history=((self.big_block,),),
        )
        split_op = Split("big_block", ("q1", "q2"), Orientation.HORIZONTAL, 4)
        with self.assertRaises(ValueError) as cm:
            split_consistency_check(interpretation_step, split_op)
        self.assertIn(
            "Split position 4 is larger than the width of the block (3).",
            str(cm.exception),
        )

        # 2 - Test a horizontal split where the split position is at the edge of the
        # block
        split_op = Split("big_block", ("q1", "q2"), Orientation.HORIZONTAL, 2)
        with self.assertRaises(ValueError) as cm:
            split_consistency_check(interpretation_step, split_op)
        self.assertIn(
            "Split position cannot be at the edge of the block.", str(cm.exception)
        )

        # 3 - Test a vertical split where the split position is larger than the height
        # of the block
        split_op = Split("big_block", ("q1", "q2"), Orientation.VERTICAL, 9)
        with self.assertRaises(ValueError) as cm:
            split_consistency_check(interpretation_step, split_op)
        self.assertIn(
            "Split position 9 is larger than the height of the block (7).",
            str(cm.exception),
        )

        # 4 - Test a vertical split where the split position is at the edge of the block
        split_op = Split("big_block", ("q1", "q2"), Orientation.VERTICAL, 6)
        with self.assertRaises(ValueError) as cm:
            split_consistency_check(interpretation_step, split_op)
        self.assertIn(
            "Split position cannot be at the edge of the block.", str(cm.exception)
        )

    def test_create_split_circuit(self):
        """Test that the create_split_circuit function returns the expected circuit and
        cbit list."""

        # Test the generation of the circuit for a vertical split
        split_op = Split("big_block", ("q1", "q2"), Orientation.VERTICAL, 3)
        qubits_to_measure = ((3, 0, 0), (3, 1, 0), (3, 2, 0))
        base_step = deepcopy(self.base_step)
        circuit, cbits = create_split_circuit(
            base_step, self.big_block, split_op, qubits_to_measure, "Z"
        )

        # Check that the circuit is as expected
        q_channels = [Channel(label=f"{q}") for q in qubits_to_measure]
        c_channels = [Channel(label=f"c_{q}_0") for q in qubits_to_measure]
        expected_circuit = Circuit(
            name="split big_block at column 3",
            circuit=[
                [Circuit("H", channels=[chan]) for chan in q_channels],
                [
                    Circuit("Measurement", channels=[q, c])
                    for q, c in zip(q_channels, c_channels, strict=True)
                ],
            ],
        )
        expected_cbits = [(f"c_{q}", 0) for q in qubits_to_measure]
        self.assertEqual(circuit, expected_circuit)
        self.assertEqual(cbits, expected_cbits)

        # Test the generation of the circuit for a horizontal split
        _ = RotatedSurfaceCode.create(
            dx=3,
            dz=7,
            lattice=Lattice.square_2d((4, 7)),
            unique_label="big_block",
        )
        split_op = Split("big_block_vert", ("q1", "q2"), Orientation.HORIZONTAL, 3)
        qubits_to_measure = ((0, 3, 0), (1, 3, 0), (2, 3, 0))
        base_step = deepcopy(self.base_step)
        circuit, cbits = create_split_circuit(
            base_step, self.big_block, split_op, qubits_to_measure, "X"
        )

        # Check that the circuit is as expected
        q_channels = [Channel(label=f"{q}") for q in qubits_to_measure]
        c_channels = [Channel(label=f"c_{q}_0") for q in qubits_to_measure]
        expected_circuit = Circuit(
            name="split big_block_vert at row 3",
            circuit=[
                [
                    Circuit("Measurement", channels=[q, c])
                    for q, c in zip(q_channels, c_channels, strict=True)
                ]
            ],
        )
        expected_cbits = [(f"c_{q}", 0) for q in qubits_to_measure]
        self.assertEqual(circuit, expected_circuit)
        self.assertEqual(cbits, expected_cbits)

    def test_split_stabilizers(self):
        "Test the split_stabilizers function."
        split_op = Split("big_block", ("q1", "q2"), Orientation.VERTICAL, 3)
        qubits_to_measure = ((3, 0, 0), (3, 1, 0), (3, 2, 0))
        stabs_block1, stabs_block2, old_stab_to_remove, new_stabs_replace = (
            split_stabilizers(self.big_block, split_op, qubits_to_measure)
        )

        # Check that the stabilizers are as expected
        expected_stabs_block1 = self.block_1.stabilizers
        expected_stabs_block2 = self.block_2.stabilizers
        expected_new_stabs_replace = (
            Stabilizer("XX", ((2, 1, 0), (2, 2, 0)), ancilla_qubits=((3, 2, 1),)),
            Stabilizer("XX", ((4, 0, 0), (4, 1, 0)), ancilla_qubits=((4, 1, 1),)),
        )
        expected_old_stabs_to_remove = (
            Stabilizer(
                "XXXX",
                ((3, 1, 0), (3, 2, 0), (2, 1, 0), (2, 2, 0)),
                ancilla_qubits=((3, 2, 1),),
            ),
            Stabilizer(
                "XXXX",
                ((4, 0, 0), (4, 1, 0), (3, 0, 0), (3, 1, 0)),
                ancilla_qubits=((4, 1, 1),),
            ),
        )

        self.assertEqual(set(stabs_block1), set(expected_stabs_block1))
        self.assertEqual(set(stabs_block2), set(expected_stabs_block2))
        self.assertEqual(set(new_stabs_replace), set(expected_new_stabs_replace))
        self.assertEqual(set(old_stab_to_remove), set(expected_old_stabs_to_remove))

    def test_find_split_stabilizer_to_circuit_mappings(self):
        "Test the find_split_stabilizer_to_circuit_mappings function."
        # Test the mapping for the first block
        new_stabs_block1 = tuple(
            stab
            for stab in self.big_block.stabilizers
            if all(coord < 3 for q in stab.data_qubits for coord in q)
        )
        new_boundary_stabs_block1 = (
            Stabilizer("XX", ((2, 1, 0), (2, 2, 0)), ancilla_qubits=((3, 2, 1),)),
        )
        new_stabs_block1 += new_boundary_stabs_block1

        new_stab_to_circ = find_split_stabilizer_to_circuit_mappings(
            self.big_block, new_stabs_block1, Direction.RIGHT
        )
        expected_mapping_block1 = {
            stab_id: synd_id
            for stab_id, synd_id in self.big_block.stabilizer_to_circuit.items()
            for new_stab in new_stabs_block1
            if stab_id == new_stab.uuid
        } | {
            new_boundary_stabs_block1[0].uuid: next(
                synd_circ.uuid
                for synd_circ in self.big_block.syndrome_circuits
                if synd_circ.name == "right-xx"
            )
        }
        self.assertEqual(new_stab_to_circ, expected_mapping_block1)

        # Test the mapping for the second block
        new_stabs_block2 = tuple(
            stab
            for stab in self.big_block.stabilizers
            if all(coord > 3 for q in stab.data_qubits for coord in q)
        )
        new_boundary_stabs_block2 = (
            Stabilizer("XX", ((4, 1, 0), (4, 2, 0)), ancilla_qubits=((4, 2, 1),)),
        )
        new_stabs_block2 += new_boundary_stabs_block2

        new_stab_to_circ = find_split_stabilizer_to_circuit_mappings(
            self.big_block, new_stabs_block2, Direction.LEFT
        )
        expected_mapping_block2 = {
            stab_id: synd_id
            for stab_id, synd_id in self.big_block.stabilizer_to_circuit.items()
            for new_stab in new_stabs_block2
            if stab_id == new_stab.uuid
        } | {
            new_boundary_stabs_block2[0].uuid: next(
                synd_circ.uuid
                for synd_circ in self.big_block.syndrome_circuits
                if synd_circ.name == "left-xx"
            )
        }
        self.assertEqual(new_stab_to_circ, expected_mapping_block2)

    def test_split_logical_operators(self):
        "Test the split_logical_operators function."
        # 1 - Test vertical split
        split_op = Split("big_block", ("q1", "q2"), Orientation.VERTICAL, 3)
        initial_x_op = self.big_block.logical_x_operators[0]
        initial_z_op = self.big_block.logical_z_operators[0]
        stabilizers_to_z_eq = tuple(
            stab
            for stab in self.big_block.stabilizers
            if all(q[0] <= 4 for q in stab.data_qubits) and stab.pauli[0] == "Z"
        )

        qubits_to_measure = ((3, 0, 0), (3, 1, 0), (3, 2, 0))
        (
            int_step,
            (logical_x_1, logical_z_1),
            (logical_x_2, logical_z_2),
            x_op_updates,
            z_op_updates,
        ) = split_logical_operators(
            self.big_base_step,
            self.big_block,
            qubits_to_measure,
            split_op,
            (0, 0, 0),
            (4, 0, 0),
        )
        # Check that the logical operators are as expected
        self.assertEqual(logical_x_1, self.block_1.logical_x_operators[0])
        self.assertEqual(logical_z_1, self.block_1.logical_z_operators[0])
        self.assertEqual(logical_x_2, self.block_2.logical_x_operators[0])
        self.assertEqual(logical_z_2, self.block_2.logical_z_operators[0])

        # Check that the logical operator evolution is correct
        self.assertEqual(
            int_step.logical_x_evolution[logical_x_1.uuid], (initial_x_op.uuid,)
        )
        self.assertEqual(
            int_step.logical_z_evolution[logical_z_1.uuid], (initial_z_op.uuid,)
        )
        self.assertEqual(
            int_step.logical_x_evolution[logical_x_2.uuid], (initial_x_op.uuid,)
        )
        self.assertEqual(
            int_step.logical_z_evolution[logical_z_2.uuid],
            (initial_z_op.uuid,) + tuple(stab.uuid for stab in stabilizers_to_z_eq),
        )

        # Check that the logical operator updates are correct (from syndrome
        # measurements only)
        # Only the vertical Z operator of the second block is affected
        expected_z_updates = {
            logical_z_1.uuid: (),
            logical_z_2.uuid: (
                ("c_(1, 1, 1)", 0),
                ("c_(2, 2, 1)", 0),
                ("c_(3, 1, 1)", 0),
                ("c_(4, 2, 1)", 0),
                ("c_(2, 0, 1)", 0),
                ("c_(4, 0, 1)", 0),
                ("c_(1, 3, 1)", 0),
                ("c_(3, 3, 1)", 0),
            ),
        }
        expected_x_updates = {}

        self.assertEqual(x_op_updates, expected_x_updates)
        self.assertEqual(z_op_updates, expected_z_updates)

        # 2 - Test horizontal split, the logical operator is defined on the measured
        # qubits, this results in both blocks having a displaced logical X operator
        split_op = Split("big_block", ("q1", "q2"), Orientation.HORIZONTAL, 3)
        big_block = RotatedSurfaceCode.create(
            dx=3,
            dz=7,
            lattice=self.lattice,
            unique_label="big_block",
            logical_x_operator=PauliOperator(
                "XXX",
                ((0, 3, 0), (1, 3, 0), (2, 3, 0)),  # Custom X operator
            ),
        )
        horizontal_base_step = InterpretationStep(
            block_history=((big_block,),),
            syndromes=[
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=[(f"c_{stab.ancilla_qubits[0]}", 0)],
                    block=big_block.uuid,
                    round=0,
                )
                for stab in big_block.stabilizers
            ],
        )
        new_block_1 = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice,
            unique_label="q1",
        )
        new_block_2 = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice,
            unique_label="q1",
            position=(0, 4),
        )
        initial_x_op = big_block.logical_x_operators[0]
        initial_z_op = big_block.logical_z_operators[0]

        # Both top and bottom blocks will have a displaced logical X operator
        # The top block's X operator requires these stabilizers:
        stabs_required_x_top = tuple(
            stab
            for stab in big_block.stabilizers
            if all(q[1] <= 3 for q in stab.data_qubits) and stab.pauli[0] == "X"
        )
        # The bottom block's X operator requires these stabilizers:
        stabs_required_x_bottom = tuple(
            stab
            for stab in big_block.stabilizers
            if all(q[1] <= 4 and q[1] >= 3 for q in stab.data_qubits)
            and stab.pauli[0] == "X"
        )

        qubits_to_measure = ((0, 3, 0), (1, 3, 0), (2, 3, 0))
        (
            int_step,
            (logical_x_1, logical_z_1),
            (logical_x_2, logical_z_2),
            x_op_updates,
            z_op_updates,
        ) = split_logical_operators(
            horizontal_base_step,
            big_block,
            qubits_to_measure,
            split_op,
            (0, 0, 0),
            (0, 4, 0),
        )
        # Check that the logical operators are as expected
        self.assertEqual(logical_x_1, new_block_1.logical_x_operators[0])
        self.assertEqual(logical_z_1, new_block_1.logical_z_operators[0])
        self.assertEqual(logical_x_2, new_block_2.logical_x_operators[0])
        self.assertEqual(logical_z_2, new_block_2.logical_z_operators[0])

        # Check that the logical operator evolution is correct
        self.assertEqual(
            int_step.logical_x_evolution[logical_x_1.uuid],
            (initial_x_op.uuid,) + tuple(stab.uuid for stab in stabs_required_x_top),
        )
        self.assertEqual(
            int_step.logical_x_evolution[logical_x_2.uuid],
            (initial_x_op.uuid,) + tuple(stab.uuid for stab in stabs_required_x_bottom),
        )
        self.assertEqual(
            int_step.logical_z_evolution[logical_z_1.uuid], (initial_z_op.uuid,)
        )
        self.assertEqual(
            int_step.logical_z_evolution[logical_z_2.uuid], (initial_z_op.uuid,)
        )

        # Check that the logical operator updates are correct (from syndrome
        # measurements only)
        # Only the horizontal X operator of the second block is affected
        expected_x_updates = {
            logical_x_1.uuid: tuple(
                (f"c_{stab.ancilla_qubits[0]}", 0) for stab in stabs_required_x_top
            ),
            logical_x_2.uuid: tuple(
                (f"c_{stab.ancilla_qubits[0]}", 0) for stab in stabs_required_x_bottom
            ),
        }
        expected_z_updates = {}

        self.assertEqual(x_op_updates, expected_x_updates)
        self.assertEqual(z_op_updates, expected_z_updates)

    def test_applicator_split(self):  # pylint: disable=too-many-locals
        """Tests that the split operation is correctly applied for a simple case."""
        initial_block = self.big_block
        split_op = Split(
            input_block_name="big_block",
            output_blocks_name=["q1", "q2"],
            orientation=Orientation.VERTICAL,
            split_position=3,
        )
        final_step = split(
            self.big_base_step, split_op, same_timeslice=False, debug_mode=True
        )
        (q2, q3) = final_step.block_history[-1]

        # Test that the blocks have been split correctly
        expected_block_1 = self.block_1
        expected_block_2 = self.block_2
        self.assertEqual(q2, expected_block_1)
        self.assertEqual(q3, expected_block_2)

        # Test the block history
        expected_block_history = (
            (initial_block,),
            (expected_block_1, expected_block_2),
        )
        self.assertEqual(final_step.block_history, expected_block_history)

        # Test that the circuit generated is correct
        q_channels = [Channel("quantum", f"(3, {i}, 0)") for i in range(3)]
        expected_circuit = Circuit(
            "split big_block at column 3",
            circuit=(
                (
                    Circuit("H", channels=[q_channels[0]]),
                    Circuit("H", channels=[q_channels[1]]),
                    Circuit("H", channels=[q_channels[2]]),
                ),
                (
                    Circuit(
                        "Measurement",
                        channels=[
                            q_channels[0],
                            Channel("classical", "c_(3, 0, 0)_0"),
                        ],
                    ),
                    Circuit(
                        "Measurement",
                        channels=[
                            q_channels[1],
                            Channel("classical", "c_(3, 1, 0)_0"),
                        ],
                    ),
                    Circuit(
                        "Measurement",
                        channels=[
                            q_channels[2],
                            Channel("classical", "c_(3, 2, 0)_0"),
                        ],
                    ),
                ),
            ),
        )
        self.assertEqual(
            final_step.intermediate_circuit_sequence[0][0], expected_circuit
        )

        # Test that the evolutions are correct
        expected_stab_evolution = {
            next(
                stab
                for stab in q2.stabilizers
                if stab.data_qubits == ((2, 1, 0), (2, 2, 0))
            ).uuid: (
                next(
                    stab
                    for stab in initial_block.stabilizers
                    if stab.data_qubits == ((3, 1, 0), (3, 2, 0), (2, 1, 0), (2, 2, 0))
                ).uuid,
            ),
            next(
                stab
                for stab in q3.stabilizers
                if stab.data_qubits == ((4, 0, 0), (4, 1, 0))
            ).uuid: (
                next(
                    stab
                    for stab in initial_block.stabilizers
                    if stab.data_qubits == ((4, 0, 0), (4, 1, 0), (3, 0, 0), (3, 1, 0))
                ).uuid,
            ),
        }
        self.assertEqual(final_step.stabilizer_evolution, expected_stab_evolution)

        expected_x_evolution = {
            q2.logical_x_operators[0].uuid: (
                (initial_block.logical_x_operators[0].uuid,)
            ),
            q3.logical_x_operators[0].uuid: (
                (initial_block.logical_x_operators[0].uuid,)
            ),
        }
        # This recreates an operator, only compare the stabilizers exctracted
        _, required_stabs = initial_block.get_shifted_equivalent_logical_operator(
            initial_block.logical_z_operators[0], (4, 0, 0)
        )
        expected_z_evolution = {
            q2.logical_z_operators[0].uuid: (
                initial_block.logical_z_operators[0].uuid,
            ),
            q3.logical_z_operators[0].uuid: (initial_block.logical_z_operators[0].uuid,)
            + tuple(stab.uuid for stab in required_stabs),
        }

        self.assertEqual(final_step.logical_x_evolution, expected_x_evolution)
        self.assertEqual(final_step.logical_z_evolution, expected_z_evolution)

        # Test that the updates are correct
        expected_stab_updates = {
            next(
                stab
                for stab in q2.stabilizers
                if stab.data_qubits == ((2, 1, 0), (2, 2, 0))
            ).uuid: (("c_(3, 1, 0)", 0), ("c_(3, 2, 0)", 0)),
            next(
                stab
                for stab in q3.stabilizers
                if stab.data_qubits == ((4, 0, 0), (4, 1, 0))
            ).uuid: (("c_(3, 0, 0)", 0), ("c_(3, 1, 0)", 0)),
        }
        self.assertEqual(final_step.stabilizer_updates, expected_stab_updates)

        # The X operators are modified by the data qubits measurements
        expected_x_updates = {
            q2.logical_x_operators[0].uuid: (("c_(3, 0, 0)", 0), ("dummy_X", 0)),
        } | self.big_base_step.logical_x_operator_updates
        # The Z operators are modified by the most recent stabilizer measurements
        # Only the right block is affected since its logical operator is the one that is
        # displaced
        expected_z_updates = {
            q2.logical_z_operators[0].uuid: (("dummy_Z", 0),),
            q3.logical_z_operators[0].uuid: (
                ("c_(1, 1, 1)", 0),
                ("c_(2, 2, 1)", 0),
                ("c_(3, 1, 1)", 0),
                ("c_(4, 2, 1)", 0),
                ("c_(2, 0, 1)", 0),
                ("c_(4, 0, 1)", 0),
                ("c_(1, 3, 1)", 0),
                ("c_(3, 3, 1)", 0),
                ("dummy_Z", 0),
            ),
        } | self.big_base_step.logical_z_operator_updates
        self.assertEqual(final_step.logical_x_operator_updates, expected_x_updates)
        self.assertEqual(final_step.logical_z_operator_updates, expected_z_updates)

    def test_split_not_at_0_0(self):  # pylint: disable=too-many-locals
        """Tests that the split operation is correctly applied for a simple case"""
        lattice_inf = Lattice.square_2d()
        initial_block = RotatedSurfaceCode.create(
            dx=7,
            dz=7,
            position=(1, 2),
            lattice=lattice_inf,
            unique_label="big_block_not_at_0_0",
        )
        split_op = Split(
            input_block_name="big_block_not_at_0_0",
            output_blocks_name=["q1", "q2"],
            orientation=Orientation.VERTICAL,
            split_position=3,
        )

        int_step = InterpretationStep(
            block_history=((initial_block,),),
            syndromes=[
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=[(f"c_{stab.ancilla_qubits[0]}", 0)],
                    block=initial_block.uuid,
                    round=0,
                )
                for stab in initial_block.stabilizers
            ],
        )

        # Test that it runs without throwing an error
        final_step = split(int_step, split_op, same_timeslice=False, debug_mode=True)
        (left_block, right_block) = final_step.block_history[-1]

        expected_left_block = RotatedSurfaceCode.create(
            dx=3,
            dz=7,
            lattice=lattice_inf,
            unique_label="q1",
            position=(1, 2),
        )
        expected_right_block = RotatedSurfaceCode.create(
            dx=3,
            dz=7,
            lattice=lattice_inf,
            unique_label="q2",
            position=(5, 2),
        )

        # Test that the blocks have been split correctly
        self.assertEqual(left_block, expected_left_block)
        self.assertEqual(right_block, expected_right_block)

        # Test the circuit generated
        q_channels = [Channel("quantum", f"(3, {i}, 0)") for i in range(7)]
        c_channels = [Channel("classical", f"c_(3, {i}, 0)_0") for i in range(7)]
        expected_circuit = Circuit(
            "split big_block_not_at_0_0 at column 3",
            circuit=(
                tuple(Circuit("H", channels=[channel]) for channel in q_channels),
                tuple(
                    Circuit("Measurement", channels=[q, c])
                    for q, c in zip(q_channels, c_channels, strict=True)
                ),
            ),
            channels=q_channels + c_channels,
        )
        self.assertEqual(
            final_step.intermediate_circuit_sequence[0][0], expected_circuit
        )

        # Test that the stabilizers evolutions are correct
        # All possible ancilla qubits close to the measured qubits
        included_left_ancilla_qubits = [(4, 2 + i, 1) for i in range(7)]
        included_right_ancilla_qubits = [(5, 2 + i, 1) for i in range(7)]

        expected_stab_evolution = {
            stab1.uuid: (stab2.uuid,)
            for stab1, stab2 in product(
                left_block.stabilizers, initial_block.stabilizers
            )
            for ancilla_qubit in included_left_ancilla_qubits
            if ancilla_qubit in stab1.ancilla_qubits
            and ancilla_qubit in stab2.ancilla_qubits
        } | {
            stab1.uuid: (stab2.uuid,)
            for stab1, stab2 in product(
                right_block.stabilizers, initial_block.stabilizers
            )
            for ancilla_qubit in included_right_ancilla_qubits
            if ancilla_qubit in stab1.ancilla_qubits
            and ancilla_qubit in stab2.ancilla_qubits
        }
        self.assertEqual(final_step.stabilizer_evolution, expected_stab_evolution)

        # Test that the logical X operator evolution is correct
        expected_x_evolution = {
            left_block.logical_x_operators[0].uuid: (
                initial_block.logical_x_operators[0].uuid,
            ),
            right_block.logical_x_operators[0].uuid: (
                initial_block.logical_x_operators[0].uuid,
            ),
        }
        self.assertEqual(final_step.logical_x_evolution, expected_x_evolution)

        # Test that the logical Z operator evolution is correct
        _, required_z_stabs = initial_block.get_shifted_equivalent_logical_operator(
            initial_block.logical_z_operators[0], (5, 2, 0)
        )
        expected_z_evolution = {
            left_block.logical_z_operators[0].uuid: (
                initial_block.logical_z_operators[0].uuid,
            ),
            right_block.logical_z_operators[0].uuid: (
                initial_block.logical_z_operators[0].uuid,
            )
            + tuple(stab.uuid for stab in required_z_stabs),
        }
        self.assertEqual(final_step.logical_z_evolution, expected_z_evolution)

        # Test that the stabilizer updates are correct
        expected_stab_updates = {
            stab1.uuid: tuple(
                (f"c_{data_qubit}", 0)
                for data_qubit in stab2.data_qubits
                if data_qubit not in stab1.data_qubits
            )
            for stab1, stab2 in product(
                left_block.stabilizers, initial_block.stabilizers
            )
            for ancilla_qubit in included_left_ancilla_qubits
            if ancilla_qubit in stab1.ancilla_qubits
            and ancilla_qubit in stab2.ancilla_qubits
        } | {
            stab1.uuid: tuple(
                (f"c_{data_qubit}", 0)
                for data_qubit in stab2.data_qubits
                if data_qubit not in stab1.data_qubits
            )
            for stab1, stab2 in product(
                right_block.stabilizers, initial_block.stabilizers
            )
            for ancilla_qubit in included_right_ancilla_qubits
            if ancilla_qubit in stab1.ancilla_qubits
            and ancilla_qubit in stab2.ancilla_qubits
        }
        self.assertEqual(final_step.stabilizer_updates, expected_stab_updates)

        # Test that the logical operator updates are correct
        expected_z_updates = {
            right_block.logical_z_operators[0].uuid: tuple(
                (f"c_{stab.ancilla_qubits[0]}", 0) for stab in required_z_stabs
            ),
        }
        self.assertEqual(final_step.logical_z_operator_updates, expected_z_updates)


if __name__ == "__main__":
    unittest.main()
