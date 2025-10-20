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

from loom.eka import Channel, Circuit, Block, Eka, Lattice, PauliOperator, Stabilizer
from loom.eka.operations import Shrink, MeasureBlockSyndromes
from loom.eka.utilities import Direction
from loom.interpreter import InterpretationStep, Syndrome, interpret_eka

from loom_repetition_code.code_factory import RepetitionCode
from loom_repetition_code.applicator.shrink import (
    shrink_consistency_check,
    get_qubits_to_measure,
    find_shrink_circuit,
    find_new_stabilizers,
    get_logical_operator_and_updates,
    shrink,
)


class TestRepetitionCodeShrink(
    unittest.TestCase
):  # pylint: disable=too-many-instance-attributes
    """
    Test the applicator for the shrink operation of RepetitionCode blocks.
    """

    def setUp(self):
        self.position = 3
        self.distance = 7

        self.linear_lattice = Lattice.linear((20,))

        self.bitflip_code = RepetitionCode.create(
            d=self.distance,
            check_type="Z",
            lattice=self.linear_lattice,
            unique_label="q1",
            position=(self.position,),
        )

        self.phaseflip_code = RepetitionCode.create(
            d=self.distance,
            check_type="X",
            lattice=self.linear_lattice,
            unique_label="q1",
            position=(self.position,),
        )
        self.generic_left_boundary = (self.position, 0)

        # Properties to iterate over during tests
        self.lengths = list(range(1, self.distance - 1))
        self.directions = ["left", "right"]
        self.check_types = ["X", "Z"]
        self.properties_iteration = tuple(
            product(self.check_types, self.directions, self.lengths)
        )

        self.rep_code_dict = {"X": self.phaseflip_code, "Z": self.bitflip_code}

        self.base_step_dict = {
            check: InterpretationStep(
                block_history=((code,),),
                syndromes=(
                    Syndrome(
                        stabilizer=stab.uuid,
                        measurements=[(f"c_{stab.ancilla_qubits[0]}", 0)],
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

        self.qubits_to_measure_dict = {
            "left": [
                [(self.position + i, 0) for i in range(length)]
                for length in self.lengths
            ],
            "right": [
                [
                    (self.position + i, 0)
                    for i in range(self.distance - length, self.distance)
                ]
                for length in self.lengths
            ],
        }

        self.new_left_boundary_qubit_dict = {
            "right": [self.generic_left_boundary for _ in self.lengths],
            "left": [(self.position + i, 0) for i in self.lengths],
        }

    def test_applicator_shrink_consistency_check(self):
        """Test that the shrink consistency check raises the correct errors"""

        # Test shrinking with invalid blocks
        invalid_block = Block(
            stabilizers=[Stabilizer("ZZ", ((0, 0), (1, 0)))],
            logical_x_operators=[PauliOperator("XX", ((0, 0), (1, 0)))],
            logical_z_operators=[PauliOperator("Z", ((0, 0),))],
            unique_label="q1",
        )
        input_step = InterpretationStep(
            block_history=((invalid_block,),),
        )

        shrink_op = Shrink(
            input_block_name=invalid_block.unique_label,
            direction=Direction.RIGHT,
            length=1,
        )

        err_msg_type = (
            f"This shrink operation is not supported for {type(invalid_block)} blocks."
        )
        with self.assertRaises(ValueError) as cm:
            _ = shrink_consistency_check(input_step, shrink_op)
        self.assertEqual(str(cm.exception), err_msg_type)

        # Test shrinking with invalid lengths (larger than allowed)
        invalid_lengths = [self.distance, self.distance + 1, self.distance + 1000]

        for length in invalid_lengths:
            shrink_op = Shrink(
                input_block_name=self.bitflip_code.unique_label,
                direction=Direction.LEFT,
                length=length,
            )
            input_step = InterpretationStep(
                block_history=((self.bitflip_code,),),
            )

            err_msg_length = "Shrink size is too large."
            with self.assertRaises(ValueError) as cm:
                _ = shrink_consistency_check(input_step, shrink_op)
            self.assertEqual(str(cm.exception), err_msg_length)

        # Test shrinking in invalid directions
        invalid_directions = [Direction.TOP, Direction.BOTTOM]

        for direction in invalid_directions:
            shrink_op = Shrink(
                input_block_name=self.bitflip_code.unique_label,
                direction=direction,
                length=1,
            )
            input_step = InterpretationStep(
                block_history=((self.bitflip_code,),),
            )

            err_msg_direction = (
                "Repetition code does not support "
                f"shrinking in the {direction} direction."
            )
            with self.assertRaises(ValueError) as cm:
                _ = shrink_consistency_check(input_step, shrink_op)
            self.assertEqual(str(cm.exception), err_msg_direction)

    def test_applicator_shrink_data_qubits(self):
        """Test correct generation of qubits to be measured during shrink."""

        for check_type, direction, length in self.properties_iteration:
            repetition_code = self.rep_code_dict[check_type]
            qubits_to_measure = get_qubits_to_measure(
                repetition_code, direction=direction, length=length
            )

            # Check correct qubits to measure
            correct_qubits_to_measure = self.qubits_to_measure_dict[direction][
                length - 1
            ]
            self.assertEqual(set(qubits_to_measure), set(correct_qubits_to_measure))

    def test_applicator_shrink_circuit(self):  # pylint: disable=too-many-locals
        """Test the correct generation of the shrink circuit."""

        for check_type, direction, length in self.properties_iteration:

            base_step = deepcopy(self.base_step_dict[check_type])
            repetition_code = self.rep_code_dict[check_type]

            circuit_name = (
                f"shrink {repetition_code.unique_label} by {length} from {direction}"
            )
            qubits_to_measure = self.qubits_to_measure_dict[direction][length - 1]

            # Compute shrink circuit and cbits
            shrink_circuit, cbits = find_shrink_circuit(
                base_step, check_type, qubits_to_measure, circuit_name
            )

            # Check circuit
            q_channels = [Channel(label=f"{q}") for q in qubits_to_measure]
            c_channels = [Channel(label=f"c_{q}_0") for q in qubits_to_measure]
            measurement_layer = [
                [
                    Circuit("Measurement", channels=[q, c])
                    for q, c in zip(q_channels, c_channels)
                ]
            ]

            if check_type == "Z":
                hadamard_layer = [
                    [Circuit("H", channels=[chan]) for chan in q_channels]
                ]
                circuit = hadamard_layer + measurement_layer
            else:
                circuit = measurement_layer

            correct_circuit = Circuit(name=circuit_name, circuit=circuit)
            self.assertEqual(shrink_circuit, correct_circuit)

            # Check cbits
            correct_cbits = [(f"c_{q}", 0) for q in qubits_to_measure]
            self.assertEqual(set(cbits), set(correct_cbits))

    def test_applicator_shrink_stabilizers(self):
        """Test the correct generation of new stabilizers after shrinking."""

        for check_type, direction, length in self.properties_iteration:

            # Compute new stabilizers
            repetition_code = self.rep_code_dict[check_type]
            qubits_to_measure = self.qubits_to_measure_dict[direction][length - 1]
            new_stabilizers = find_new_stabilizers(repetition_code, qubits_to_measure)

            # Compute correct stabilizers
            remaining_data_qubits = sorted(
                list(set(repetition_code.data_qubits) - set(qubits_to_measure)),
                key=lambda x: x[0],
            )
            correct_stabilizers = [
                Stabilizer(
                    pauli=check_type * 2,
                    data_qubits=remaining_data_qubits[i : i + 2],
                    ancilla_qubits=[(remaining_data_qubits[i][0], 1)],
                )
                for i in range(len(remaining_data_qubits) - 1)
            ]

            # Check correct stabilizers are generated
            self.assertEqual(set(new_stabilizers), set(correct_stabilizers))

    def test_applicator_shrink_logical_operator_and_updates(
        self,
    ):  # pylint: disable=too-many-locals
        """Test the generation of logical operators and updates after shrinking."""

        # Complementary Pauli type
        complementary = {"X": "Z", "Z": "X"}

        for check_type, direction, length in self.properties_iteration:

            repetition_code = self.rep_code_dict[check_type]
            new_left_boundary_qubit = self.new_left_boundary_qubit_dict[direction][
                length - 1
            ]
            is_left = direction == "left"
            qubits_to_measure = self.qubits_to_measure_dict[direction][length - 1]
            cbits = [(f"c_{q}", 0) for q in qubits_to_measure]

            # Compute uuids if left logical needs to be shifted
            stabs_to_remove = [
                stab
                for stab in repetition_code.stabilizers
                if any(qb in qubits_to_measure for qb in stab.data_qubits)
            ]
            id_stabs_required = (
                [stab.uuid for stab in stabs_to_remove] if is_left else []
            )

            # Generate new logicals and updates
            new_logs, log_evolution, log_updates = get_logical_operator_and_updates(
                self.base_step_dict[check_type],
                repetition_code,
                check_type,
                is_left,
                qubits_to_measure,
                cbits,
            )

            # Check logical operators
            remaining_data_qubits = list(
                set(repetition_code.data_qubits) - set(qubits_to_measure)
            )
            correct_long_logical = PauliOperator(
                pauli=complementary[check_type] * len(remaining_data_qubits),
                data_qubits=remaining_data_qubits,
            )
            correct_short_logical = PauliOperator(
                pauli=check_type, data_qubits=[new_left_boundary_qubit]
            )
            self.assertEqual(new_logs[0][0], correct_long_logical)
            self.assertEqual(new_logs[1][0], correct_short_logical)

            # Check logical evolution
            old_x_logical = repetition_code.logical_x_operators[0]
            old_z_logical = repetition_code.logical_z_operators[0]
            old_long_logical = old_x_logical if check_type == "Z" else old_z_logical
            old_short_logical = old_z_logical if check_type == "Z" else old_x_logical

            correct_long_log_evolution = {new_logs[0][0].uuid: (old_long_logical.uuid,)}
            correct_short_log_evolution = (
                {
                    new_logs[1][0].uuid: (old_short_logical.uuid,)
                    + tuple(id_stabs_required)
                }
                if is_left
                else {}
            )

            self.assertEqual(log_evolution[0], correct_long_log_evolution)
            self.assertEqual(log_evolution[1], correct_short_log_evolution)

            # Check logical updates
            correct_long_log_updates = {new_logs[0][0].uuid: tuple(cbits)}
            correct_short_log_updates = (
                {
                    new_logs[1][0].uuid: tuple(
                        (f"c_{stab.ancilla_qubits[0]}", 0) for stab in stabs_to_remove
                    )
                }
                if is_left
                else {}
            )
            self.assertEqual(log_updates[0], correct_long_log_updates)
            self.assertEqual(log_updates[1], correct_short_log_updates)

    def test_applicator_shrink(
        self,
    ):  # pylint: disable=too-many-statements, too-many-locals
        """Tests that the shrink operation is correctly applied. We test for shrinking
        in both directions, for several lengths, and for both X and Z checks."""

        for check_type, direction, length in self.properties_iteration:
            repetition_code = self.rep_code_dict[check_type]
            is_left = direction == "left"

            shrink_op = Shrink(
                input_block_name=repetition_code.unique_label,
                direction=direction,
                length=length,
            )

            base_step = deepcopy(self.base_step_dict[check_type])

            final_step = shrink(
                base_step, shrink_op, same_timeslice=False, debug_mode=True
            )
            final_block = final_step.get_block(repetition_code.unique_label)

            ### Check block is correct

            # Create manual shrunk block
            manual_distance = self.distance - length
            manual_stabilizers = [
                Stabilizer(
                    pauli=check_type * 2,
                    data_qubits=[(i, 0), (i + 1, 0)],
                    ancilla_qubits=[(i, 1)],
                )
                for i in range(manual_distance - 1)
            ]

            complementary = {"X": "Z", "Z": "X"}
            manual_logical_operators = {
                check_type: PauliOperator(pauli=check_type, data_qubits=[(0, 0)]),
                complementary[check_type]: PauliOperator(
                    pauli=complementary[check_type] * manual_distance,
                    data_qubits=[(i, 0) for i in range(manual_distance)],
                ),
            }

            manual_shrunk_block = RepetitionCode(
                stabilizers=manual_stabilizers,
                logical_x_operators=[manual_logical_operators["X"]],
                logical_z_operators=[manual_logical_operators["Z"]],
                unique_label=repetition_code.unique_label,
            )
            manual_shrunk_block = manual_shrunk_block.shift(
                (self.position + length * is_left,)
            )

            # Check block is correct
            self.assertEqual(final_block, manual_shrunk_block)

            # Check generated circuit has appropriate name
            correct_circuit_name = (
                f"shrink {repetition_code.unique_label} by {length} from {direction}"
            )
            self.assertEqual(
                final_step.intermediate_circuit_sequence[0][0].name,
                correct_circuit_name,
            )

            ### Check circuit is correct

            # Extract removed data qubits - sorted for comparing lists
            removed_data_qubits = sorted(
                list(set(repetition_code.data_qubits) - set(final_block.data_qubits)),
                key=lambda x: x[0],
                reverse=(direction == Direction.RIGHT),
            )

            # The time step in the circuit we check for
            operation_time_step = 0

            # If necessary, check Hadamard layer is correctly incorporated
            if check_type == "Z":
                hadamard_operations = [("h", (str(qb),)) for qb in removed_data_qubits]
                self.assertEqual(
                    {
                        (
                            gate.name,
                            tuple(channel.label for channel in gate.channels),
                        )
                        for gate in final_step.intermediate_circuit_sequence[0][
                            0
                        ].circuit[operation_time_step]
                    },
                    set(hadamard_operations),
                )
                operation_time_step += 1

            # Check for correct measurement operations
            measurement_operations = [
                ("measurement", (str(qb), "c_" + str(qb) + "_0"))
                for qb in removed_data_qubits
            ]
            self.assertEqual(
                {
                    (
                        gate.name,
                        tuple(channel.label for channel in gate.channels),
                    )
                    for gate in final_step.intermediate_circuit_sequence[0][0].circuit[
                        operation_time_step
                    ]
                },
                set(measurement_operations),
            )

            # Check logical operator evolutions
            removed_stabs = sorted(
                list(set(repetition_code.stabilizers) - set(final_block.stabilizers)),
                key=lambda x: x.data_qubits[0][0],
            )
            id_stabs_required = [stab.uuid for stab in removed_stabs] if is_left else []
            cbits = [(f"c_{q}", 0) for q in removed_data_qubits]

            if check_type == "Z":
                calculated_long_logical = final_block.logical_x_operators
                repetition_code_long_logical = repetition_code.logical_x_operators
                initial_long_updates = base_step.logical_x_operator_updates
                calculated_short_logical = final_block.logical_z_operators
                repetition_code_short_logical = repetition_code.logical_z_operators
                initial_short_updates = base_step.logical_z_operator_updates
            else:
                calculated_long_logical = final_block.logical_z_operators
                repetition_code_long_logical = repetition_code.logical_z_operators
                initial_long_updates = base_step.logical_z_operator_updates
                calculated_short_logical = final_block.logical_x_operators
                repetition_code_short_logical = repetition_code.logical_x_operators
                initial_short_updates = base_step.logical_x_operator_updates

            correct_long_logical_evolution = {
                calculated_long_logical[0].uuid: (repetition_code_long_logical[0].uuid,)
            }
            correct_long_logical_updates = {
                calculated_long_logical[0].uuid: tuple(cbits)
                + (("dummy_" + "".join("X" if check_type == "Z" else "Z"), 0),)
            } | initial_long_updates  # Add the initial state of the dictionary

            correct_short_logical_evolution = (
                {
                    calculated_short_logical[0].uuid: (
                        repetition_code_short_logical[0].uuid,
                    )
                    + tuple(id_stabs_required)
                }
                if is_left
                else {}
            )
            correct_short_logical_updates = (
                {
                    calculated_short_logical[0].uuid: tuple(
                        (f"c_{stab.ancilla_qubits[0]}", 0) for stab in removed_stabs
                    )
                    + ((f"dummy_{check_type}", 0),)
                }
                if is_left
                else {}
            ) | initial_short_updates  # Add the initial state of the dictionary

            if check_type == "Z":
                correct_x_evolution = correct_long_logical_evolution
                correct_x_logical_updates = correct_long_logical_updates
                correct_z_evolution = correct_short_logical_evolution
                correct_z_logical_updates = correct_short_logical_updates
            else:
                correct_x_evolution = correct_short_logical_evolution
                correct_x_logical_updates = correct_short_logical_updates
                correct_z_evolution = correct_long_logical_evolution
                correct_z_logical_updates = correct_long_logical_updates

            for calculated, correct in [
                (final_step.logical_x_evolution, correct_x_evolution),
                (final_step.logical_z_evolution, correct_z_evolution),
                (final_step.logical_x_operator_updates, correct_x_logical_updates),
                (final_step.logical_z_operator_updates, correct_z_logical_updates),
            ]:
                self.assertEqual(calculated, correct)

    def test_within_eka(self):
        """Test that the operation is correctly applied within the Eka class."""
        direction = Direction.LEFT
        length = 2
        repetition_code = self.bitflip_code

        shrink_op = Shrink(repetition_code.unique_label, direction, length)
        ops = [
            MeasureBlockSyndromes(repetition_code.unique_label),
            shrink_op,
        ]

        # Apply operation using Eka
        eka = Eka(self.linear_lattice, blocks=[repetition_code], operations=ops)
        output_block_eka = interpret_eka(eka).get_block(repetition_code.unique_label)

        # Apply operation manually
        base_step = deepcopy(self.base_step_dict["Z"])
        final_step = shrink(base_step, shrink_op, same_timeslice=False, debug_mode=True)
        final_block_applicator = final_step.get_block(repetition_code.unique_label)

        # Assert blocks are equal
        self.assertEqual(output_block_eka, final_block_applicator)


if __name__ == "__main__":
    unittest.main()
