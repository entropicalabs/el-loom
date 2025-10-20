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

from loom.eka import Channel, Circuit, Eka, Lattice, PauliOperator, Stabilizer, Block
from loom.eka.operations import Grow
from loom.eka.utilities import Direction
from loom.interpreter import InterpretationStep, interpret_eka
from loom_repetition_code.applicator.grow import (
    grow_consistency_check,
    get_new_data_qubits_info,
    create_grow_circuit,
    find_new_stabilizers,
    get_logical_operator_and_evolution,
    grow,
)
from loom_repetition_code.code_factory import RepetitionCode


class TestRepetitionCodeGrow(
    unittest.TestCase
):  # pylint: disable=too-many-instance-attributes
    """
    Test the applicator for the grow operation of RepetitionCode blocks.
    """

    def setUp(self):
        self.position = 6
        self.distance = 7

        self.linear_lattice = Lattice.linear((30,))

        self.bitflip_code = RepetitionCode.create(
            d=self.distance,
            check_type="Z",
            lattice=self.linear_lattice,
            unique_label="q1",
            position=(self.position,),
        )
        self.bitflip_int_step = InterpretationStep(
            block_history=((self.bitflip_code,),)
        )

        self.phaseflip_code = RepetitionCode.create(
            d=self.distance,
            check_type="X",
            lattice=self.linear_lattice,
            unique_label="q1",
            position=(self.position,),
        )
        self.phaseflip_int_step = InterpretationStep(
            block_history=((self.phaseflip_code,),)
        )
        self.generic_left_boundary = (self.position, 0)

        # Properties to iterate over during tests
        self.lengths = list(range(1, self.position - 1))
        self.directions = ["left", "right"]
        self.check_types = ["X", "Z"]
        self.properties_iteration = tuple(
            product(self.check_types, self.directions, self.lengths)
        )

        self.rep_code_dict = {"X": self.phaseflip_code, "Z": self.bitflip_code}

        self.base_step_dict = {
            check: InterpretationStep(
                block_history=((code,),),
                logical_x_operator_updates={
                    code.logical_x_operators[0].uuid: (("dummy_X", 0),)
                },
                logical_z_operator_updates={
                    code.logical_z_operators[0].uuid: (("dummy_Z", 0),)
                },
            )
            for check, code in self.rep_code_dict.items()
        }

        self.new_left_boundary_qubit_dict = {
            "right": [self.generic_left_boundary for _ in self.lengths],
            "left": [(self.position - i, 0) for i in self.lengths],
        }

        self.new_data_qubit_dict = {
            "right": [
                [(i + self.distance + self.position, 0) for i in range(l)]
                for l in self.lengths
            ],
            "left": [
                [(self.position - i - 1, 0) for i in range(l)] for l in self.lengths
            ],
        }

    def test_applicator_grow_consistency_check(self):
        """Test that the grow consistency check raises the correct errors"""

        # Test growing with invalid blocks
        invalid_block = Block(
            stabilizers=[Stabilizer("ZZ", ((0, 0), (1, 0)))],
            logical_x_operators=[PauliOperator("XX", ((0, 0), (1, 0)))],
            logical_z_operators=[PauliOperator("Z", ((0, 0),))],
            unique_label="q1",
        )
        input_step = InterpretationStep(block_history=((invalid_block,),))

        grow_op = Grow(
            input_block_name=invalid_block.unique_label,
            direction=Direction.RIGHT,
            length=1,
        )

        err_msg_type = (
            f"This grow operation is not supported for {type(invalid_block)} blocks."
        )
        with self.assertRaises(ValueError) as cm:
            _ = grow_consistency_check(input_step, grow_op)
        self.assertEqual(str(cm.exception), err_msg_type)

        # Test growing from the left beyond the lattice boundary
        invalid_lengths = [self.position + i for i in range(1, 4)]

        for length in invalid_lengths:
            grow_op = Grow(
                input_block_name=self.bitflip_code.unique_label,
                direction=Direction.LEFT,
                length=length,
            )

            err_msg_length = "Cannot grow beyond the boundary of the lattice."
            with self.assertRaises(ValueError) as cm:
                _ = grow_consistency_check(self.bitflip_int_step, grow_op)
            self.assertEqual(str(cm.exception), err_msg_length)

        # Test growing in invalid directions
        invalid_directions = [Direction.TOP, Direction.BOTTOM]

        for direction in invalid_directions:
            grow_op = Grow(
                input_block_name=self.bitflip_code.unique_label,
                direction=direction,
                length=1,
            )

            err_msg_direction = (
                "Repetition code does not support "
                f"growing in the {direction} direction."
            )
            with self.assertRaises(ValueError) as cm:
                _ = grow_consistency_check(self.bitflip_int_step, grow_op)
            self.assertEqual(str(cm.exception), err_msg_direction)

    def test_applicator_grow_data_qubits(self):
        """Test correct generation of qubits to be measured during grow."""

        for check_type, direction, length in self.properties_iteration:
            repetition_code = self.rep_code_dict[check_type]
            new_data_qubits = get_new_data_qubits_info(
                repetition_code, direction=direction, length=length
            )

            # Check correct qubits to measure
            correct_new_data_qubits = self.new_data_qubit_dict[direction][length - 1]
            self.assertEqual(set(new_data_qubits), set(correct_new_data_qubits))

    def test_applicator_grow_circuit(self):
        """Test correct generation of circuit for growing the repetition code."""

        for check_type, direction, length in self.properties_iteration:
            repetition_code = self.rep_code_dict[check_type]
            base_step = self.base_step_dict[check_type]
            new_data_qubits = self.new_data_qubit_dict[direction][length - 1]
            new_ancilla_qubits = [(q[0], 1) for q in new_data_qubits]

            reset_type = "0" if check_type == "X" else "+"

            circuit_name = (
                f"grow {repetition_code.unique_label} by {length} to the {direction}"
            )

            circuit = create_grow_circuit(
                base_step, check_type, new_data_qubits, circuit_name
            )

            q_channels = [
                Channel(label=f"{q}", type="quantum") for q in new_ancilla_qubits
            ]

            correct_sequence = [
                Circuit(name=f"reset_{reset_type}", channels=ch) for ch in q_channels
            ]

            correct_circuit = Circuit(name=circuit_name, circuit=[correct_sequence])

            # Check the circuit
            self.assertEqual(circuit.name, correct_circuit.name)
            self.assertEqual(circuit, correct_circuit)

    def test_applicator_grow_stabilizers(self):
        """Test the correct generation of new stabilizers after grow."""

        for check_type, direction, length in self.properties_iteration:

            # Compute new stabilizers
            repetition_code = self.rep_code_dict[check_type]
            is_left = direction == "left"
            new_data_qubits = self.new_data_qubit_dict[direction][length - 1]
            new_stabilizers = find_new_stabilizers(
                repetition_code, check_type, is_left, new_data_qubits
            )

            # Compute correct stabilizers
            all_data_qubits = sorted(
                list(repetition_code.data_qubits) + new_data_qubits
            )
            correct_stabilizers = [
                Stabilizer(
                    pauli=check_type * 2,
                    data_qubits=all_data_qubits[i : i + 2],
                    ancilla_qubits=[(all_data_qubits[i][0], 1)],
                )
                for i in range(len(all_data_qubits) - 1)
            ]

            # Check correct stabilizers are generated
            self.assertEqual(set(new_stabilizers), set(correct_stabilizers))

    def test_applicator_grow_logical_operator_and_evolution(
        self,
    ):  # pylint: disable=too-many-locals
        """Test the generation of logical operators and evolution after growing."""

        # Complementary Pauli type
        complementary = {"X": "Z", "Z": "X"}

        for check_type, direction, length in self.properties_iteration:

            repetition_code = self.rep_code_dict[check_type]
            new_data_qubits = self.new_data_qubit_dict[direction][length - 1]

            # Generate new logicals and updates
            new_logs, log_evolution = get_logical_operator_and_evolution(
                repetition_code,
                check_type,
                new_data_qubits,
            )

            # Extract old logicals
            old_x_logical = repetition_code.logical_x_operators[0]
            old_z_logical = repetition_code.logical_z_operators[0]
            old_long_logical = old_x_logical if check_type == "Z" else old_z_logical
            old_short_logical = old_z_logical if check_type == "Z" else old_x_logical

            # Check logical operators
            all_data_qubits = list(repetition_code.data_qubits) + new_data_qubits

            correct_long_logical = PauliOperator(
                pauli=complementary[check_type] * len(all_data_qubits),
                data_qubits=all_data_qubits,
            )
            correct_short_logical = old_short_logical

            self.assertEqual(new_logs[0][0], correct_long_logical)
            self.assertEqual(new_logs[1][0], correct_short_logical)

            # Check logical evolution
            correct_long_log_evolution = {new_logs[0][0].uuid: (old_long_logical.uuid,)}
            correct_short_log_evolution = {}

            self.assertEqual(log_evolution[0], correct_long_log_evolution)
            self.assertEqual(log_evolution[1], correct_short_log_evolution)

    def test_applicator_grow(self):  # pylint: disable=too-many-locals
        """Tests that the grow operation is correctly applied. We test for growing
        in both directions, for several lengths, and for both X and Z checks."""

        for check_type, direction, length in self.properties_iteration:
            repetition_code = self.rep_code_dict[check_type]
            is_left = direction == "left"

            grow_op = Grow(
                input_block_name=repetition_code.unique_label,
                direction=direction,
                length=length,
            )

            base_step = deepcopy(self.base_step_dict[check_type])

            final_step = grow(base_step, grow_op, same_timeslice=False, debug_mode=True)
            final_block = final_step.get_block(repetition_code.unique_label)

            ### Check block is correct
            # Create manual grown block
            manual_distance = self.distance + length
            manual_stabilizers = [
                Stabilizer(
                    pauli=check_type * 2,
                    data_qubits=[(i, 0), (i + 1, 0)],
                    ancilla_qubits=[(i, 1)],
                )
                for i in range(manual_distance - 1)
            ]

            short_logical_data = (length * is_left, 0)

            manual_logical_x_operators = [
                (
                    PauliOperator(pauli=check_type, data_qubits=[short_logical_data])
                    if check_type == "X"
                    else PauliOperator(
                        pauli="X" * manual_distance,
                        data_qubits=[(i, 0) for i in range(manual_distance)],
                    )
                )
            ]
            manual_logical_z_operators = [
                (
                    PauliOperator(pauli=check_type, data_qubits=[short_logical_data])
                    if check_type == "Z"
                    else PauliOperator(
                        pauli="Z" * manual_distance,
                        data_qubits=[(i, 0) for i in range(manual_distance)],
                    )
                )
            ]
            manual_grown_block = RepetitionCode(
                stabilizers=manual_stabilizers,
                logical_x_operators=manual_logical_x_operators,
                logical_z_operators=manual_logical_z_operators,
                unique_label=repetition_code.unique_label,
            )
            manual_grown_block = manual_grown_block.shift(
                (self.position - length * is_left,)
            )

            # Check block is correct
            self.assertEqual(final_block, manual_grown_block)

            ### Check circuit is correct
            new_data_qubits = self.new_data_qubit_dict[direction][length - 1]
            reset_type = "0" if check_type == "X" else "+"

            correct_circuit_name = (
                f"grow {repetition_code.unique_label} by {length} to the {direction}"
            )
            circuit = [
                Circuit(
                    name=f"reset_{reset_type}",
                    channels=[Channel(label=str(q), type="quantum")],
                )
                for q in new_data_qubits
            ]

            correct_circuit = Circuit(name=correct_circuit_name, circuit=[circuit])
            self.assertEqual(
                final_step.intermediate_circuit_sequence[0][0], correct_circuit
            )

            # Check logical operator evolutions
            correct_x_evolution = (
                {
                    final_block.logical_x_operators[0].uuid: (
                        repetition_code.logical_x_operators[0].uuid,
                    )
                }
                if check_type == "Z"
                else {}
            )

            correct_z_evolution = (
                {
                    final_block.logical_z_operators[0].uuid: (
                        repetition_code.logical_z_operators[0].uuid,
                    )
                }
                if check_type == "X"
                else {}
            )

            self.assertEqual(final_step.logical_x_evolution, correct_x_evolution)
            self.assertEqual(final_step.logical_z_evolution, correct_z_evolution)

            # Check logical updates are propagated correctly
            self.assertEqual(
                final_step.logical_x_operator_updates[
                    final_block.logical_x_operators[0].uuid
                ],
                (("dummy_X", 0),),
            )
            self.assertEqual(
                final_step.logical_z_operator_updates[
                    final_block.logical_z_operators[0].uuid
                ],
                (("dummy_Z", 0),),
            )

    def test_within_eka(self):
        """Test that the grow operation is correctly applied within the Eka class."""
        direction = Direction.RIGHT
        length = 7
        repetition_code = self.bitflip_code

        # Apply grow operation using Eka
        grow_op = Grow(repetition_code.unique_label, direction, length)
        eka = Eka(self.linear_lattice, blocks=[repetition_code], operations=[grow_op])
        output_block_eka = interpret_eka(eka).get_block(repetition_code.unique_label)

        # Apply grow operation manually
        base_step = deepcopy(self.base_step_dict["Z"])
        final_step = grow(base_step, grow_op, same_timeslice=False, debug_mode=True)
        final_block_applicator = final_step.get_block(repetition_code.unique_label)

        # Assert blocks are equal
        self.assertEqual(output_block_eka, final_block_applicator)


if __name__ == "__main__":
    unittest.main()
