"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=duplicate-code
import unittest
from itertools import product
from copy import deepcopy

from loom.eka import Block, Circuit, Lattice, PauliOperator, Stabilizer
from loom.eka.operations import Merge
from loom.eka.utilities import Direction, Orientation
from loom.interpreter import InterpretationStep
from loom_repetition_code.applicator.merge import (
    merge_consistency_check,
    get_new_data_qubits_info,
    find_merge_circuit,
    find_new_stabilizers,
    get_logical_operator_and_evolution,
    merge,
)
from loom_repetition_code.code_factory import RepetitionCode


class TestRepetitionCodeMerge(
    unittest.TestCase
):  # pylint: disable=too-many-instance-attributes
    """
    Test the applicator for the merge operation of RepetitionCode blocks.
    """

    def setUp(self):
        self.position = 3
        self.distance = 7

        self.linear_lattice = Lattice.linear((20,))

        self.bitflip_code = RepetitionCode.create(
            d=self.distance,
            check_type="Z",
            lattice=self.linear_lattice,
            unique_label="bitflip_qubit",
            position=(self.position,),
        )

        self.phaseflip_code = RepetitionCode.create(
            d=self.distance,
            check_type="X",
            logical_x_operator=PauliOperator(pauli="X", data_qubits=[(6, 0)]),
            lattice=self.linear_lattice,
            unique_label="phaseflip_qubit",
            position=(self.position,),
        )

        # Properties to iterate over during tests
        self.merge_spacings = [1, 2, 3, 4]
        self.check_types = ["X", "Z"]
        self.properties_iteration = tuple(
            product(self.check_types, self.merge_spacings)
        )

        self.rep_code_dict = {"X": self.phaseflip_code, "Z": self.bitflip_code}

        shifted_bitflip_codes = {
            i: RepetitionCode.create(
                d=self.distance,
                check_type="Z",
                logical_z_operator=PauliOperator(pauli="Z", data_qubits=[(6, 0)]),
                lattice=self.linear_lattice,
                unique_label="q2",
                position=(self.position + self.distance + i,),
            )
            for i in self.merge_spacings
        }

        shifted_phaseflip_codes = {
            i: RepetitionCode.create(
                d=self.distance,
                check_type="X",
                logical_x_operator=PauliOperator(pauli="X", data_qubits=[(6, 0)]),
                lattice=self.linear_lattice,
                unique_label="q2",
                position=(self.position + self.distance + i,),
            )
            for i in self.merge_spacings
        }

        self.shifted_rep_codes_dict = {
            "X": shifted_phaseflip_codes,
            "Z": shifted_bitflip_codes,
        }

        self.new_qubits_dict = {
            spacing: [
                (i, 0)
                for i in range(
                    self.position + self.distance,
                    self.position + self.distance + spacing,
                )
            ]
            for spacing in self.merge_spacings
        }

        self.base_step_dict = {
            (check, spacing): InterpretationStep(
                block_history=(
                    (
                        rc_1 := self.rep_code_dict[check],
                        rc_2 := self.shifted_rep_codes_dict[check][spacing],
                    ),
                ),
                logical_x_operator_updates={
                    rc_1.logical_x_operators[0].uuid: (("dummy_X", 0),),
                    rc_2.logical_x_operators[0].uuid: (("dummy_X", 1),),
                },
                logical_z_operator_updates={
                    rc_1.logical_z_operators[0].uuid: (("dummy_Z", 0),),
                    rc_2.logical_z_operators[0].uuid: (("dummy_Z", 1),),
                },
            )
            for check, spacing in self.properties_iteration
        }

    def test_applicator_merge_consistency_check(self):
        """Test consistency check for merging operation."""

        # Test merging with invalid block types
        invalid_block = Block(
            stabilizers=[Stabilizer("ZZ", ((0, 0), (1, 0)))],
            logical_x_operators=[PauliOperator("XX", ((0, 0), (1, 0)))],
            logical_z_operators=[PauliOperator("Z", ((0, 0),))],
            unique_label="q1",
        )

        valid_block = RepetitionCode.create(
            d=3,
            check_type="Z",
            lattice=Lattice.linear((20,)),
            unique_label="q2",
        )

        block_order = [(invalid_block, valid_block), (valid_block, invalid_block)]

        for block1, block2 in block_order:

            merge_op = Merge(
                input_blocks_name=[block1.unique_label, block2.unique_label],
                output_block_name="out_q1",
                orientation=Orientation.HORIZONTAL,
            )
            input_step = InterpretationStep(
                block_history=((block1, block2),),
            )

            err_msg_type = (
                f"This merge operation is not supported"
                f" for {(type(invalid_block),)} blocks."
            )

            with self.assertRaises(TypeError) as cm:
                _ = merge_consistency_check(input_step, merge_op)
            self.assertEqual(str(cm.exception), err_msg_type)

        # Test merging with repetition codes of different check types
        block1 = self.rep_code_dict["X"]
        block2 = self.rep_code_dict["Z"]
        input_step = InterpretationStep(
            block_history=((block1, block2),),
        )

        merge_op = Merge(
            input_blocks_name=[block1.unique_label, block2.unique_label],
            output_block_name="out_q1",
            orientation=Orientation.HORIZONTAL,
        )

        err_msg = (
            f"Cannot merge blocks with different check types:"
            f" {block1.check_type} and {block2.check_type}"
        )

        with self.assertRaises(TypeError) as cm:
            _ = merge_consistency_check(input_step, merge_op)
        self.assertEqual(str(cm.exception), err_msg)

        # Test overlapping blocks
        block1 = self.rep_code_dict["Z"]

        block2 = RepetitionCode.create(
            d=self.distance,
            check_type="Z",
            logical_z_operator=PauliOperator(pauli="Z", data_qubits=[(2, 0)]),
            lattice=self.linear_lattice,
            unique_label="q1",
            position=(self.position + 1,),
        )
        input_step = InterpretationStep(
            block_history=((block1, block2),),
        )

        err_msg = "Cannot merge blocks that overlap."
        merge_op = Merge(
            input_blocks_name=[block1.unique_label, block2.unique_label],
            output_block_name="out_q1",
            orientation=Orientation.HORIZONTAL,
        )
        with self.assertRaises(ValueError) as cm:
            _ = merge_consistency_check(input_step, merge_op)
        self.assertEqual(str(cm.exception), err_msg)

        # Test merging in vertical orientation
        block1 = RepetitionCode.create(
            d=self.distance,
            check_type="Z",
            logical_z_operator=PauliOperator(pauli="Z", data_qubits=[(2, 0)]),
            lattice=self.linear_lattice,
            unique_label="q1",
            position=(0,),
        )

        block2 = RepetitionCode.create(
            d=self.distance,
            check_type="Z",
            logical_z_operator=PauliOperator(pauli="Z", data_qubits=[(2, 0)]),
            lattice=self.linear_lattice,
            unique_label="q2",
            position=(self.distance + 2,),
        )
        input_step = InterpretationStep(
            block_history=((block1, block2),),
        )

        err_msg = (
            "Repetition code does not support merging in the vertical orientation."
        )

        merge_op = Merge(
            input_blocks_name=[block1.unique_label, block2.unique_label],
            output_block_name="out_q1",
            orientation=Orientation.VERTICAL,
        )
        with self.assertRaises(ValueError) as cm:
            _ = merge_consistency_check(input_step, merge_op)
        self.assertEqual(str(cm.exception), err_msg)

    def test_applicator_merge_new_qubits(self):
        """Test extraction of new data qubits after merging."""

        for check_type, spacing in self.properties_iteration:

            block1, block2 = (
                self.rep_code_dict[check_type],
                self.shifted_rep_codes_dict[check_type][spacing],
            )

            correct_new_data_qubits = self.new_qubits_dict[spacing]

            # Extract data qubits
            new_data_qubits = get_new_data_qubits_info([block1, block2])

            # Ensure correctness
            self.assertEqual(set(correct_new_data_qubits), set(new_data_qubits))

    def test_applicator_merge_circuit(self):
        """Test creation of merge circuit."""

        for check_type, spacing in self.properties_iteration:

            new_data_qubits = self.new_qubits_dict[spacing]
            base_step = deepcopy(self.base_step_dict[(check_type, spacing)])
            circuit_name = "skibidi_prra_prra"

            reset_state = "0" if check_type == "X" else "+"
            _ = [(new_data_qubits[0][0] - 1, 1)] + [
                (qb[0], 1) for qb in new_data_qubits
            ]

            reset_sequence = [
                [
                    Circuit(
                        name=f"reset_{reset_state}",
                        channels=base_step.get_channel_MUT(q, "quantum"),
                    )
                    for q in new_data_qubits
                ]
            ]

            # Create correct circuit
            correct_circuit = Circuit(
                name=circuit_name,
                circuit=reset_sequence,
            )

            # Generate circuit
            circuit = find_merge_circuit(
                base_step, check_type, new_data_qubits, circuit_name
            )

            # Compare circuits
            self.assertEqual(circuit, correct_circuit)

    def test_applicator_merge_new_stabilizers(self):
        """Test extraction of new stabilizers after merging."""

        for check_type, spacing in self.properties_iteration:

            block1, block2 = (
                self.rep_code_dict[check_type],
                self.shifted_rep_codes_dict[check_type][spacing],
            )

            correct_new_stabilizers = [
                Stabilizer(
                    pauli=check_type * 2,
                    data_qubits=[(i, 0), (i + 1, 0)],
                    ancilla_qubits=[(i, 1)],
                )
                for i in range(
                    self.position, self.position + 2 * self.distance + spacing - 1
                )
            ]

            # Extract new stabilizers
            new_stabilizers = find_new_stabilizers([block1, block2], check_type)

            # Ensure correctness
            self.assertEqual(set(new_stabilizers), set(correct_new_stabilizers))

    def test_applicator_merge_logicals_and_evolution(
        self,
    ):  # pylint: disable=too-many-locals
        """Test extraction of logical operators and evolution after merging."""

        for check_type, spacing in self.properties_iteration:

            other_check_type = "X" if check_type == "Z" else "Z"

            block1, block2 = (
                self.rep_code_dict[check_type],
                self.shifted_rep_codes_dict[check_type][spacing],
            )

            old_logs_1 = [block1.logical_x_operators[0], block1.logical_z_operators[0]]
            old_short_log_1, old_long_log_1 = sorted(
                old_logs_1, key=lambda x: len(x.data_qubits)
            )
            old_logs_2 = [block2.logical_x_operators[0], block2.logical_z_operators[0]]
            _, old_long_log_2 = sorted(old_logs_2, key=lambda x: len(x.data_qubits))

            new_qubits = self.new_qubits_dict[spacing]

            correct_long_logical = PauliOperator(
                pauli=other_check_type * (2 * self.distance + spacing),
                data_qubits=list(block1.data_qubits)
                + new_qubits
                + list(block2.data_qubits),
            )

            correct_short_logical, stabs_required = (
                block1.get_shifted_equivalent_logical_operator(
                    block1.boundary_qubits(Direction.LEFT)
                )
            )

            # Extract logical operators and updates
            logicals, log_evolution = get_logical_operator_and_evolution(
                [block1, block2], check_type, new_qubits
            )

            correct_long_log_evolution = {
                logicals[0][0].uuid: (old_long_log_1.uuid, old_long_log_2.uuid)
            }

            if stabs_required:
                id_stabs_required = [stab.uuid for stab in stabs_required]
                correct_short_log_evolution = {
                    logicals[1][0].uuid: tuple(
                        [old_short_log_1.uuid] + id_stabs_required
                    )
                }

            else:
                correct_short_log_evolution = {}

            # Ensure correctness
            self.assertEqual(logicals[0][0], correct_long_logical)
            self.assertEqual(logicals[1][0], correct_short_logical)
            self.assertEqual(log_evolution[0], correct_long_log_evolution)
            self.assertEqual(log_evolution[1], correct_short_log_evolution)

    def test_applicator_merge(self):  # pylint: disable=too-many-locals
        """Test the merge applicator for the repetition code."""

        for check_type, spacing in self.properties_iteration:

            block1, block2 = (
                self.rep_code_dict[check_type],
                self.shifted_rep_codes_dict[check_type][spacing],
            )

            # Create merge operation and obtain merged block
            merge_op = Merge(
                input_blocks_name=[block1.unique_label, block2.unique_label],
                output_block_name="out_q1",
                orientation=Orientation.HORIZONTAL,
            )

            base_step = deepcopy(self.base_step_dict[(check_type, spacing)])

            final_step = merge(
                base_step, merge_op, same_timeslice=False, debug_mode=True
            )
            final_block = final_step.get_block(merge_op.output_block_name)

            ### Check block is correct by manually creating a merged block
            ### and comparing it with the computed one

            # Create attributes for the manually merged block
            manual_distance = 2 * self.distance + spacing
            manual_stabilizers = [
                Stabilizer(
                    pauli=check_type * 2,
                    data_qubits=[(i, 0), (i + 1, 0)],
                    ancilla_qubits=[(i, 1)],
                )
                for i in range(manual_distance - 1)
            ]

            manual_logical_x_operators = [
                (
                    PauliOperator(pauli=check_type, data_qubits=[(0, 0)])
                    if check_type == "X"
                    else PauliOperator(
                        pauli="X" * manual_distance,
                        data_qubits=[(i, 0) for i in range(manual_distance)],
                    )
                )
            ]
            manual_logical_z_operators = [
                (
                    PauliOperator(pauli=check_type, data_qubits=[(0, 0)])
                    if check_type == "Z"
                    else PauliOperator(
                        pauli="Z" * manual_distance,
                        data_qubits=[(i, 0) for i in range(manual_distance)],
                    )
                )
            ]

            # Create manually merged block
            manual_merged_block = RepetitionCode(
                stabilizers=manual_stabilizers,
                logical_x_operators=manual_logical_x_operators,
                logical_z_operators=manual_logical_z_operators,
                unique_label=merge_op.output_block_name,
            )
            manual_merged_block = manual_merged_block.shift((self.position,))

            # Check that computed Path matched the manually generated one
            self.assertEqual(final_block, manual_merged_block)

            # Check that the circuit implementing the merge operation has the
            # appropriate name
            correct_circuit_name = (
                f"merge {block1.unique_label} and "
                f"{block2.unique_label} into {merge_op.output_block_name}"
            )

            self.assertEqual(
                final_step.intermediate_circuit_sequence[0][0].name,
                correct_circuit_name,
            )

            ### Check circuit is correct by recreating the expected merging circuit
            reset_state = "0" if check_type == "X" else "+"
            new_data_qubits = self.new_qubits_dict[spacing]

            reset_datas = [
                (f"reset_{reset_state}", (str(qb),)) for qb in new_data_qubits
            ]
            self.assertEqual(
                {
                    (
                        gate.name,
                        tuple(channel.label for channel in gate.channels),
                    )
                    for gate in final_step.intermediate_circuit_sequence[0][0].circuit[
                        0
                    ]
                },
                set(reset_datas),
            )

            # Check logical operator evolutions are correct by comparing the computed
            # and the expected ones for both cases depending on the check type

            _, stabs_required = block1.get_shifted_equivalent_logical_operator(
                block1.boundary_qubits(Direction.LEFT)
            )

            if check_type == "Z":
                correct_x_evolution = {
                    final_block.logical_x_operators[0].uuid: (
                        block1.logical_x_operators[0].uuid,
                        block2.logical_x_operators[0].uuid,
                    )
                }

                correct_z_evolution = (
                    {
                        final_block.logical_z_operators[0].uuid: (
                            block1.logical_z_operators[0].uuid,
                        )
                        + tuple(stab.uuid for stab in stabs_required)
                    }
                    if stabs_required
                    else {}
                )

            else:
                correct_z_evolution = {
                    final_block.logical_z_operators[0].uuid: (
                        block1.logical_z_operators[0].uuid,
                        block2.logical_z_operators[0].uuid,
                    )
                }

                correct_x_evolution = (
                    {
                        final_block.logical_x_operators[0].uuid: (
                            block1.logical_x_operators[0].uuid,
                        )
                        + tuple(stab.uuid for stab in stabs_required)
                    }
                    if stabs_required
                    else {}
                )
            self.assertEqual(final_step.logical_x_evolution, correct_x_evolution)
            self.assertEqual(final_step.logical_z_evolution, correct_z_evolution)

            # Check the logical updates - the long logical should bear the merge
            # measurement and the two previous updates,
            # the new short will only bear updates of the conserved logical
            correct_logical_update = {
                "X": {
                    final_block.logical_x_operators[0].uuid: (
                        tuple(("dummy_X", i) for i in range(2))
                        if check_type == "Z"
                        else (("dummy_X", 0),)
                    )
                },
                "Z": {
                    final_block.logical_z_operators[0].uuid: (
                        tuple(("dummy_Z", i) for i in range(2))
                        if check_type == "X"
                        else (("dummy_Z", 0),)
                    )
                },
            }

            self.assertEqual(
                final_step.logical_x_operator_updates[
                    final_block.logical_x_operators[0].uuid
                ],
                correct_logical_update["X"][final_block.logical_x_operators[0].uuid],
            )
            self.assertEqual(
                final_step.logical_z_operator_updates[
                    final_block.logical_z_operators[0].uuid
                ],
                correct_logical_update["Z"][final_block.logical_z_operators[0].uuid],
            )


if __name__ == "__main__":
    unittest.main()
