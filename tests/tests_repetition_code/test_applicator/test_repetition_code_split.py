"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=duplicate-code
import unittest
import itertools
from copy import deepcopy

from loom.eka import Block, Channel, Circuit, Eka, Lattice, PauliOperator, Stabilizer
from loom.eka.operations import Split, MeasureBlockSyndromes
from loom.eka.utilities import Orientation
from loom.interpreter import InterpretationStep, Syndrome, interpret_eka

from loom_repetition_code.code_factory import RepetitionCode
from loom_repetition_code.applicator.split import (
    split_consistency_check,
    find_qubit_to_measure,
    create_split_circuit,
    find_new_stabilizers,
    get_logical_operator_and_updates,
    split,
)


class TestRepetitionCodeSplit(
    unittest.TestCase
):  # pylint: disable=too-many-instance-attributes
    """
    Test the applicator for the split operation of RepetitionCode blocks.
    """

    def setUp(self):
        self.position = 3
        self.distance = 7

        self.linear_lattice = Lattice.linear((20,))

        self.bitflip_code = RepetitionCode.create(
            d=self.distance,
            check_type="Z",
            logical_z_operator=PauliOperator(pauli="Z", data_qubits=[(2, 0)]),
            lattice=self.linear_lattice,
            unique_label="q1",
            position=(self.position,),
        )

        self.phaseflip_code = RepetitionCode.create(
            d=self.distance,
            check_type="X",
            logical_x_operator=PauliOperator(pauli="X", data_qubits=[(6, 0)]),
            lattice=self.linear_lattice,
            unique_label="q1",
            position=(self.position,),
        )
        self.generic_left_boundary = (self.position, 0)

        # Properties to iterate over during tests
        self.split_positions = list(range(2, self.distance - 2))
        self.check_types = ["X", "Z"]
        self.properties_iteration = tuple(
            itertools.product(self.check_types, self.split_positions)
        )

        self.rep_code_dict = {"X": self.phaseflip_code, "Z": self.bitflip_code}

        self.base_step_dict = {
            check: InterpretationStep(
                block_history=((code,),),
                syndromes=tuple(
                    Syndrome(
                        stabilizer=stab.uuid,
                        measurements=((f"c_{stab.ancilla_qubits[0]}", 0),),
                        block=code.uuid,
                        round=0,
                    )
                    for stab in code.stabilizers
                ),
                logical_x_operator_updates={
                    code.logical_x_operators[0].uuid: (("dummy_X", 0),)
                },
                logical_z_operator_updates={
                    code.logical_z_operators[0].uuid: (("dummy_Z", 0),)
                },
            )
            for check, code in self.rep_code_dict.items()
        }

        self.qubit_to_measure_list = [
            (split_position + self.position, 0)
            for split_position in self.split_positions
        ]

    def test_applicator_split_consistency_check(self):
        """Test consistency check for the split operation."""

        # Test splitting with invalid blocks
        invalid_block = Block(
            stabilizers=[Stabilizer("ZZ", ((0, 0), (1, 0)))],
            logical_x_operators=[PauliOperator("XX", ((0, 0), (1, 0)))],
            logical_z_operators=[PauliOperator("Z", ((0, 0),))],
            unique_label="q1",
        )
        input_step = InterpretationStep(
            block_history=((invalid_block,),),
        )

        split_op = Split(
            input_block_name=invalid_block.unique_label,
            output_blocks_name=["out_q1", "out_q2"],
            orientation=Orientation.VERTICAL,
            split_position=1,
        )

        err_msg_type = (
            f"This split operation is not supported for {type(invalid_block)} blocks."
        )
        with self.assertRaises(ValueError) as cm:
            _ = split_consistency_check(input_step, split_op)
        self.assertEqual(str(cm.exception), err_msg_type)

        # Test splitting in the wrong positions
        outside_chain_positions = [self.distance + self.position]
        boundary_positions = [self.distance - 1]
        single_qubit_position = [1, self.distance - 2]
        input_step = InterpretationStep(
            block_history=((self.bitflip_code,),),
        )

        for split_position in outside_chain_positions:
            err_msg = (
                f"Split position {split_position+self.position} is outside the chain."
            )
            split_op = Split(
                input_block_name=self.bitflip_code.unique_label,
                output_blocks_name=["out_q1", "out_q2"],
                orientation=Orientation.VERTICAL,
                split_position=split_position,
            )

            with self.assertRaises(ValueError) as cm:
                _ = split_consistency_check(input_step, split_op)
            self.assertEqual(str(cm.exception), err_msg)

        for split_position in boundary_positions:
            input_step = InterpretationStep(
                block_history=((self.bitflip_code,),),
            )
            err_msg = "Split position cannot be at the edge of the chain."
            split_op = Split(
                input_block_name=self.bitflip_code.unique_label,
                output_blocks_name=["out_q1", "out_q2"],
                orientation=Orientation.VERTICAL,
                split_position=split_position,
            )
            with self.assertRaises(ValueError) as cm:
                _ = split_consistency_check(input_step, split_op)
            self.assertEqual(str(cm.exception), err_msg)

        for split_position in single_qubit_position:
            input_step = InterpretationStep(
                block_history=((self.bitflip_code,),),
            )
            err_msg = "Split has to partition chain in units of at least two qubits."
            split_op = Split(
                input_block_name=self.bitflip_code.unique_label,
                output_blocks_name=["out_q1", "out_q2"],
                orientation=Orientation.VERTICAL,
                split_position=split_position,
            )
            with self.assertRaises(ValueError) as cm:
                _ = split_consistency_check(input_step, split_op)
            self.assertEqual(str(cm.exception), err_msg)

        # Check that operation goes along vertical direction
        err_msg_orientation = "Only vertical splits are supported."
        split_op = Split(
            input_block_name=self.bitflip_code.unique_label,
            output_blocks_name=["out_q1", "out_q2"],
            orientation=Orientation.HORIZONTAL,
            split_position=2,
        )
        input_step = InterpretationStep(
            block_history=((self.bitflip_code,),),
        )
        with self.assertRaises(ValueError) as cm:
            _ = split_consistency_check(input_step, split_op)
        self.assertEqual(str(cm.exception), err_msg_orientation)

    def test_applicator_split_qubit_to_measure(self):
        """Test correct qubit is chosen for measurement during split."""

        for check_type, split_position in self.properties_iteration:
            repetition_code = self.rep_code_dict[check_type]

            # Check that the qubits to measure are correct
            qubits_to_measure = find_qubit_to_measure(repetition_code, split_position)
            expected_qubit_to_measure = self.qubit_to_measure_list[
                self.split_positions.index(split_position)
            ]

            self.assertEqual(qubits_to_measure, expected_qubit_to_measure)

    def test_applicator_split_circuit(self):  # pylint: disable=too-many-locals
        """Test the correct creation of the circuit executing split operation."""

        for check_type, split_position in self.properties_iteration:
            repetition_code = self.rep_code_dict[check_type]
            base_step = deepcopy(self.base_step_dict[check_type])

            circuit_name = f"Split {repetition_code.unique_label} at {split_position}"
            qubit_to_measure = self.qubit_to_measure_list[
                self.split_positions.index(split_position)
            ]

            split_circuit, cbit = create_split_circuit(
                base_step, check_type, qubit_to_measure, circuit_name
            )

            # Check circuit
            q_channel = Channel(label=f"{qubit_to_measure}")
            c_channel = Channel(label=f"c_{qubit_to_measure}_0")
            measurement_layer = [
                [Circuit("Measurement", channels=[q_channel, c_channel])]
            ]

            if check_type == "Z":
                hadamard_layer = [[Circuit("H", channels=[q_channel])]]
                circuit = hadamard_layer + measurement_layer
            else:
                circuit = measurement_layer

            correct_circuit = Circuit(name=circuit_name, circuit=circuit)
            self.assertEqual(split_circuit, correct_circuit)

            # Check cbit
            correct_cbit = (f"c_{qubit_to_measure}", 0)
            self.assertEqual(cbit, correct_cbit)

    def test_applicator_split_new_stabilizers(self):
        """Test the correct creation of the new stabilizers after split operation."""

        for check_type, split_position in self.properties_iteration:
            repetition_code = self.rep_code_dict[check_type]

            qubit_to_measure = self.qubit_to_measure_list[
                self.split_positions.index(split_position)
            ]

            new_stabilizers_1, new_stabilizers_2 = find_new_stabilizers(
                repetition_code, qubit_to_measure
            )

            # Check new stabilizers
            correct_new_stabilizers_1 = [
                Stabilizer(
                    pauli=check_type * 2,
                    data_qubits=[(i, 0), (i + 1, 0)],
                    ancilla_qubits=[(i, 1)],
                )
                for i in range(self.position, qubit_to_measure[0] - 1)
            ]

            correct_new_stabilizers_2 = [
                Stabilizer(
                    pauli=check_type * 2,
                    data_qubits=[(i, 0), (i + 1, 0)],
                    ancilla_qubits=[(i, 1)],
                )
                for i in range(
                    qubit_to_measure[0] + 1, self.distance + self.position - 1
                )
            ]

            self.assertEqual(new_stabilizers_1, correct_new_stabilizers_1)
            self.assertEqual(new_stabilizers_2, correct_new_stabilizers_2)

    def test_applicator_split_logical_operator_and_updates(
        self,
    ):  # pylint: disable=too-many-locals
        """Test the correct creation of the logicals and updates after split."""

        # Complementary Pauli type
        complementary = {"X": "Z", "Z": "X"}

        for check_type, split_position in self.properties_iteration:

            repetition_code = self.rep_code_dict[check_type]
            qubit_to_measure = self.qubit_to_measure_list[
                self.split_positions.index(split_position)
            ]
            cbit = (f"c_{qubit_to_measure}", 0)

            base_step = self.base_step_dict[check_type]
            (
                new_logs_1,
                new_logs_2,
                log_evolution_1,
                log_evolution_2,
                log_updates_1,
                log_updates_2,
            ) = get_logical_operator_and_updates(
                base_step, repetition_code, check_type, qubit_to_measure, cbit
            )

            # Check logical operators
            correct_data_qubits_1 = [
                qb for qb in repetition_code.data_qubits if qb[0] < qubit_to_measure[0]
            ]
            correct_left_boundary_1 = min(correct_data_qubits_1, key=lambda x: x[0])
            correct_data_qubits_2 = [
                qb for qb in repetition_code.data_qubits if qb[0] > qubit_to_measure[0]
            ]
            correct_left_boundary_2 = min(correct_data_qubits_2, key=lambda x: x[0])

            correct_short_logical_1 = PauliOperator(
                pauli=check_type,
                data_qubits=[correct_left_boundary_1],
            )
            correct_long_logical_1 = PauliOperator(
                pauli=complementary[check_type] * len(correct_data_qubits_1),
                data_qubits=correct_data_qubits_1,
            )

            self.assertEqual(
                new_logs_1, [[correct_long_logical_1], [correct_short_logical_1]]
            )

            correct_short_logical_2 = PauliOperator(
                pauli=check_type,
                data_qubits=[correct_left_boundary_2],
            )
            correct_long_logical_2 = PauliOperator(
                pauli=complementary[check_type] * len(correct_data_qubits_2),
                data_qubits=correct_data_qubits_2,
            )

            self.assertEqual(
                new_logs_2, [[correct_long_logical_2], [correct_short_logical_2]]
            )

            # Check logical evolution
            old_x_logical = repetition_code.logical_x_operators[0]
            old_z_logical = repetition_code.logical_z_operators[0]
            old_long_logical = old_x_logical if check_type == "Z" else old_z_logical
            old_short_logical = old_z_logical if check_type == "Z" else old_x_logical

            _, stabs_required_1 = (
                repetition_code.get_shifted_equivalent_logical_operator(
                    correct_left_boundary_1
                )
            )
            id_stabs_required_1 = [stab.uuid for stab in stabs_required_1]

            correct_long_logical_evolution_1 = {
                new_logs_1[0][0].uuid: (old_long_logical.uuid,)
            }
            if id_stabs_required_1 == []:
                correct_short_logical_evolution_1 = {}
            else:
                correct_short_logical_evolution_1 = {
                    new_logs_1[1][0].uuid: tuple(
                        [old_short_logical.uuid] + id_stabs_required_1
                    )
                }

            self.assertEqual(log_evolution_1[0], correct_long_logical_evolution_1)
            self.assertEqual(log_evolution_1[1], correct_short_logical_evolution_1)

            _, stabs_required_2 = (
                repetition_code.get_shifted_equivalent_logical_operator(
                    correct_left_boundary_2
                )
            )
            id_stabs_required_2 = [stab.uuid for stab in stabs_required_2]

            correct_long_logical_evolution_2 = {
                new_logs_2[0][0].uuid: (old_long_logical.uuid,)
            }
            if id_stabs_required_2 == []:
                correct_short_logical_evolution_2 = {}
            else:
                correct_short_logical_evolution_2 = {
                    new_logs_2[1][0].uuid: tuple(
                        [old_short_logical.uuid] + id_stabs_required_2
                    )
                }

            self.assertEqual(log_evolution_2[0], correct_long_logical_evolution_2)
            self.assertEqual(log_evolution_2[1], correct_short_logical_evolution_2)

            # Check logical updates
            correct_long_logical_update_1 = {new_logs_1[0][0].uuid: (cbit,)}
            correct_short_logical_update_1 = {
                new_logs_1[1][0].uuid: tuple(
                    (f"c_{stab.ancilla_qubits[0]}", 0) for stab in stabs_required_1
                )
            }

            self.assertEqual(log_updates_1[0], correct_long_logical_update_1)
            self.assertEqual(log_updates_1[1], correct_short_logical_update_1)

            correct_long_logical_update_2 = {}  # Only the first operator is updated
            correct_short_logical_update_2 = {
                new_logs_2[1][0].uuid: tuple(
                    (f"c_{stab.ancilla_qubits[0]}", 0) for stab in stabs_required_2
                )
            }

            self.assertEqual(log_updates_2[0], correct_long_logical_update_2)
            self.assertEqual(log_updates_2[1], correct_short_logical_update_2)

    def test_applicator_split(
        self,
    ):  # pylint: disable=too-many-locals, too-many-statements
        """Test the proper action of the applicator for the split operation."""

        for check_type, split_position in self.properties_iteration:
            repetition_code = self.rep_code_dict[check_type]

            split_op = Split(
                input_block_name=repetition_code.unique_label,
                output_blocks_name=[
                    f"out_{repetition_code.unique_label}_1",
                    f"out_{repetition_code.unique_label}_2",
                ],
                orientation=Orientation.VERTICAL,
                split_position=split_position,
            )

            base_step = deepcopy(self.base_step_dict[check_type])

            final_step = split(
                base_step, split_op, same_timeslice=False, debug_mode=True
            )

            # Check new blocks are correct

            # First block
            manual_distance_1 = split_position

            manual_stabilizers_1 = [
                Stabilizer(
                    pauli=check_type * 2,
                    data_qubits=[(i, 0), (i + 1, 0)],
                    ancilla_qubits=[(i, 1)],
                )
                for i in range(manual_distance_1 - 1)
            ]

            manual_logical_x_operators_1 = [
                (
                    PauliOperator(pauli=check_type, data_qubits=[(0, 0)])
                    if check_type == "X"
                    else PauliOperator(
                        pauli="X" * manual_distance_1,
                        data_qubits=[(i, 0) for i in range(manual_distance_1)],
                    )
                )
            ]

            manual_logical_z_operators_1 = [
                (
                    PauliOperator(pauli=check_type, data_qubits=[(0, 0)])
                    if check_type == "Z"
                    else PauliOperator(
                        pauli="Z" * manual_distance_1,
                        data_qubits=[(i, 0) for i in range(manual_distance_1)],
                    )
                )
            ]

            manual_block_1 = RepetitionCode(
                stabilizers=manual_stabilizers_1,
                logical_x_operators=manual_logical_x_operators_1,
                logical_z_operators=manual_logical_z_operators_1,
                unique_label=repetition_code.unique_label,
            )

            manual_split_block_1 = manual_block_1.shift(
                position=(self.position,),
                new_label=f"out_{repetition_code.unique_label}_1",
            )

            # Second block
            manual_distance_2 = self.distance - split_position - 1

            manual_stabilizers_2 = [
                Stabilizer(
                    pauli=check_type * 2,
                    data_qubits=[(i, 0), (i + 1, 0)],
                    ancilla_qubits=[(i, 1)],
                )
                for i in range(manual_distance_2 - 1)
            ]

            manual_logical_x_operators_2 = [
                (
                    PauliOperator(pauli=check_type, data_qubits=[(0, 0)])
                    if check_type == "X"
                    else PauliOperator(
                        pauli="X" * manual_distance_2,
                        data_qubits=[(i, 0) for i in range(manual_distance_2)],
                    )
                )
            ]

            manual_logical_z_operators_2 = [
                (
                    PauliOperator(pauli=check_type, data_qubits=[(0, 0)])
                    if check_type == "Z"
                    else PauliOperator(
                        pauli="Z" * manual_distance_2,
                        data_qubits=[(i, 0) for i in range(manual_distance_2)],
                    )
                )
            ]

            manual_block_2 = RepetitionCode(
                stabilizers=manual_stabilizers_2,
                logical_x_operators=manual_logical_x_operators_2,
                logical_z_operators=manual_logical_z_operators_2,
                unique_label=repetition_code.unique_label,
            )

            manual_split_block_2 = manual_block_2.shift(
                position=(self.position + manual_distance_1 + 1,),
                new_label=f"out_{repetition_code.unique_label}_2",
            )

            (split_block_1, split_block_2) = final_step.block_history[-1]

            self.assertEqual(split_block_1, manual_split_block_1)
            self.assertEqual(split_block_2, manual_split_block_2)

            # Check generated circuit has appropriate name
            correct_circuit_name = (
                f"split {repetition_code.unique_label} at {split_position}"
            )
            self.assertEqual(
                final_step.intermediate_circuit_sequence[0][0].name,
                correct_circuit_name,
            )

            # Check the circuit is correct
            qubit_to_measure = list(
                set(repetition_code.data_qubits)
                - set(split_block_1.data_qubits + split_block_2.data_qubits)
            )[0]

            q_chan = Channel(label=str(qubit_to_measure))
            c_chan = Channel(label="c_" + str(qubit_to_measure) + "_0")

            if check_type == "Z":
                circ_seq = [
                    [Circuit("h", channels=[q_chan])],
                ]
            else:
                circ_seq = []
            circ_seq += [[Circuit("Measurement", channels=[q_chan, c_chan])]]

            expected_circ = Circuit(name=correct_circuit_name, circuit=circ_seq)
            self.assertEqual(
                expected_circ, final_step.intermediate_circuit_sequence[0][0]
            )

            # Check evolutions and updates are correct
            cbit = (f"c_{qubit_to_measure}", 0)
            _, stabs_required_1 = (
                repetition_code.get_shifted_equivalent_logical_operator(
                    new_qubit=(self.position, 0)
                )
            )
            id_stabs_required_1 = [stab.uuid for stab in stabs_required_1]

            _, stabs_required_2 = (
                repetition_code.get_shifted_equivalent_logical_operator(
                    new_qubit=(self.position + manual_distance_1 + 1, 0)
                )
            )
            id_stabs_required_2 = [stab.uuid for stab in stabs_required_2]

            if check_type == "Z":
                # Long logical: Only the first logical inherits
                # the dummy update and the measurement
                correct_x_logical_updates = {
                    split_block_1.logical_x_operators[0].uuid: (
                        cbit,
                        ("dummy_X", 0),
                    ),
                } | base_step.logical_x_operator_updates
                correct_x_evolution = {
                    split_block_1.logical_x_operators[0].uuid: (
                        repetition_code.logical_x_operators[0].uuid,
                    ),
                    split_block_2.logical_x_operators[0].uuid: (
                        repetition_code.logical_x_operators[0].uuid,
                    ),
                }
                # Short logical: both inherit the dummy update
                correct_z_logical_updates = {
                    split_block_1.logical_z_operators[0].uuid: tuple(
                        (f"c_{stab.ancilla_qubits[0]}", 0) for stab in stabs_required_1
                    )
                    + (("dummy_Z", 0),),
                    split_block_2.logical_z_operators[0].uuid: tuple(
                        (f"c_{stab.ancilla_qubits[0]}", 0) for stab in stabs_required_2
                    )
                    + (("dummy_Z", 0),),
                } | base_step.logical_z_operator_updates
                correct_z_evolution = {
                    split_block_1.logical_z_operators[0].uuid: (
                        repetition_code.logical_z_operators[0].uuid,
                    )
                    + tuple(id_stabs_required_1),
                    split_block_2.logical_z_operators[0].uuid: (
                        repetition_code.logical_z_operators[0].uuid,
                    )
                    + tuple(id_stabs_required_2),
                }

            else:
                # Short logical:  both inherit the dummy update
                correct_x_logical_updates = {
                    split_block_1.logical_x_operators[0].uuid: tuple(
                        (f"c_{stab.ancilla_qubits[0]}", 0) for stab in stabs_required_1
                    )
                    + (("dummy_X", 0),),
                    split_block_2.logical_x_operators[0].uuid: tuple(
                        (f"c_{stab.ancilla_qubits[0]}", 0) for stab in stabs_required_2
                    )
                    + (("dummy_X", 0),),
                } | base_step.logical_x_operator_updates
                correct_x_evolution = {
                    split_block_1.logical_x_operators[0].uuid: (
                        repetition_code.logical_x_operators[0].uuid,
                    )
                    + tuple(id_stabs_required_1),
                    split_block_2.logical_x_operators[0].uuid: (
                        repetition_code.logical_x_operators[0].uuid,
                    )
                    + tuple(id_stabs_required_2),
                }
                # Long logical: Only the first logical inherits
                # the dummy update and the measurement
                correct_z_logical_updates = {
                    split_block_1.logical_z_operators[0].uuid: (
                        cbit,
                        ("dummy_Z", 0),
                    ),
                } | base_step.logical_z_operator_updates
                correct_z_evolution = {
                    split_block_1.logical_z_operators[0].uuid: (
                        repetition_code.logical_z_operators[0].uuid,
                    ),
                    split_block_2.logical_z_operators[0].uuid: (
                        repetition_code.logical_z_operators[0].uuid,
                    ),
                }

            self.assertEqual(final_step.logical_x_evolution, correct_x_evolution)
            self.assertEqual(final_step.logical_z_evolution, correct_z_evolution)

            self.assertEqual(
                final_step.logical_x_operator_updates, correct_x_logical_updates
            )
            self.assertEqual(
                final_step.logical_z_operator_updates, correct_z_logical_updates
            )

    def test_within_eka(self):
        """Test that the operation is correctly applied within the Eka class."""
        split_position = 4
        repetition_code = self.bitflip_code
        output_names = ("qout_1", "q_out2")

        op = Split(
            repetition_code.unique_label,
            output_names,
            Orientation.VERTICAL,
            split_position,
        )
        meas_op = MeasureBlockSyndromes(repetition_code.unique_label)

        # Apply operation using Eka
        eka = Eka(
            self.linear_lattice, blocks=[repetition_code], operations=[meas_op, op]
        )
        final_step_eka = interpret_eka(eka)

        # Apply operation manually
        base_step = deepcopy(self.base_step_dict["Z"])
        final_step_applicator = split(
            base_step, op, same_timeslice=False, debug_mode=True
        )

        # Assert output blocks are equal
        for block in output_names:
            self.assertEqual(
                final_step_eka.get_block(block),
                final_step_applicator.get_block(block),
            )


if __name__ == "__main__":
    unittest.main()
