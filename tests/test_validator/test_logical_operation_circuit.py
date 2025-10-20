"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest

from loom.eka import (
    Circuit,
    Channel,
    ChannelType,
    LogicalState,
    Stabilizer,
    PauliOperator,
    Block,
)
from loom.validator import (
    is_circuit_valid,
    is_logical_operation_circuit_valid,
    logical_state_transformations_to_check,
)


# pylint: disable=duplicate-code
class TestLogicalOperationValidator(unittest.TestCase):
    """
    Test cases for validating logical operation circuits using the Validator module.
    """

    def setUp(self) -> None:

        # Define rotated surface code and circuit for CNOT
        self.rsc1_block = Block(
            stabilizers=(
                Stabilizer(
                    "ZZZZ",
                    ((1, 0, 0), (0, 0, 0), (1, 1, 0), (0, 1, 0)),
                    ancilla_qubits=((1, 1, 1),),
                ),
                Stabilizer(
                    "ZZZZ",
                    ((2, 1, 0), (1, 1, 0), (2, 2, 0), (1, 2, 0)),
                    ancilla_qubits=((2, 2, 1),),
                ),
                Stabilizer(
                    "XXXX",
                    ((1, 1, 0), (1, 2, 0), (0, 1, 0), (0, 2, 0)),
                    ancilla_qubits=((1, 2, 1),),
                ),
                Stabilizer(
                    "XXXX",
                    ((2, 0, 0), (2, 1, 0), (1, 0, 0), (1, 1, 0)),
                    ancilla_qubits=((2, 1, 1),),
                ),
                Stabilizer("XX", ((0, 0, 0), (0, 1, 0)), ancilla_qubits=((0, 1, 1),)),
                Stabilizer("XX", ((2, 1, 0), (2, 2, 0)), ancilla_qubits=((3, 2, 1),)),
                Stabilizer("ZZ", ((2, 0, 0), (1, 0, 0)), ancilla_qubits=((2, 0, 1),)),
                Stabilizer("ZZ", ((1, 2, 0), (0, 2, 0)), ancilla_qubits=((1, 3, 1),)),
            ),
            logical_x_operators=[
                PauliOperator(
                    pauli="XXX", data_qubits=tuple((i, 0, 0) for i in range(3))
                )
            ],
            logical_z_operators=[
                PauliOperator(
                    pauli="ZZZ", data_qubits=((0, 0, 0), (0, 1, 0), (0, 2, 0))
                )
            ],
        )
        self.rsc2_block = Block(
            stabilizers=(
                Stabilizer(
                    "ZZZZ",
                    ((4, 0, 0), (3, 0, 0), (4, 1, 0), (3, 1, 0)),
                    ancilla_qubits=((4, 1, 1),),
                ),
                Stabilizer(
                    "ZZZZ",
                    ((5, 1, 0), (4, 1, 0), (5, 2, 0), (4, 2, 0)),
                    ancilla_qubits=((5, 2, 1),),
                ),
                Stabilizer(
                    "XXXX",
                    ((4, 1, 0), (4, 2, 0), (3, 1, 0), (3, 2, 0)),
                    ancilla_qubits=((4, 2, 1),),
                ),
                Stabilizer(
                    "XXXX",
                    ((5, 0, 0), (5, 1, 0), (4, 0, 0), (4, 1, 0)),
                    ancilla_qubits=((5, 1, 1),),
                ),
                Stabilizer("XX", ((3, 0, 0), (3, 1, 0)), ancilla_qubits=((3, 1, 1),)),
                Stabilizer("XX", ((5, 1, 0), (5, 2, 0)), ancilla_qubits=((6, 2, 1),)),
                Stabilizer("ZZ", ((5, 0, 0), (4, 0, 0)), ancilla_qubits=((5, 0, 1),)),
                Stabilizer("ZZ", ((4, 2, 0), (3, 2, 0)), ancilla_qubits=((4, 3, 1),)),
            ),
            logical_x_operators=[
                PauliOperator(
                    pauli="XXX", data_qubits=((3, 0, 0), (4, 0, 0), (5, 0, 0))
                )
            ],
            logical_z_operators=[
                PauliOperator(
                    pauli="ZZZ", data_qubits=((3, 0, 0), (3, 1, 0), (3, 2, 0))
                )
            ],
        )

        # Define the qubit channels
        dq_channels = {
            qub: Channel(type=ChannelType.QUANTUM, label=str(qub))
            for qub in self.rsc1_block.data_qubits + self.rsc2_block.data_qubits
        }

        # Define the circuit for a logical CNOT operation
        # Because the rotated surface code is a CSS code, we can implement the CNOT gate
        # transversally
        self.rsc12_cnot = Circuit(
            "rot_surface_code_cnot",
            circuit=[
                Circuit(
                    "CNOT",
                    channels=[dq_channels[(i, j, 0)], dq_channels[(i + 3, j, 0)]],
                )
                for i in range(3)
                for j in range(3)
            ],
        )

        self.steane_block = Block(
            stabilizers=(
                Stabilizer(
                    "XXXX",
                    ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)),
                    ancilla_qubits=((0, 0, 1),),
                ),
                Stabilizer(
                    "XXXX",
                    ((1, 0, 0), (1, 1, 0), (2, 1, 0), (2, 0, 0)),
                    ancilla_qubits=((1, 0, 1),),
                ),
                Stabilizer(
                    "XXXX",
                    ((2, 1, 0), (1, 2, 0), (0, 1, 0), (1, 1, 0)),
                    ancilla_qubits=((2, 0, 1),),
                ),
                Stabilizer(
                    "ZZZZ",
                    ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)),
                    ancilla_qubits=((0, 1, 1),),
                ),
                Stabilizer(
                    "ZZZZ",
                    ((1, 0, 0), (1, 1, 0), (2, 1, 0), (2, 0, 0)),
                    ancilla_qubits=((0, 2, 1),),
                ),
                Stabilizer(
                    "ZZZZ",
                    ((2, 1, 0), (1, 2, 0), (0, 1, 0), (1, 1, 0)),
                    ancilla_qubits=((0, 3, 1),),
                ),
            ),
            logical_x_operators=[
                PauliOperator(
                    pauli="XXX", data_qubits=((0, 0, 0), (1, 0, 0), (2, 0, 0))
                )
            ],
            logical_z_operators=[
                PauliOperator(
                    pauli="ZZZ", data_qubits=((0, 0, 0), (1, 0, 0), (2, 0, 0))
                )
            ],
        )

        # Define the circuit
        steane_dqubits = [
            Channel(type=ChannelType.QUANTUM, label=str(q))
            for q in self.steane_block.data_qubits
        ]
        self.steane_h_logical = Circuit(
            "steane_code_logical_hadamard",
            circuit=[Circuit("H", channels=steane_dqubits[i]) for i in range(7)],
        )

    def test_steane_code_logical_hadamard_valid(self):
        """Test the validation of logical Hadamard operation on the Steane code."""
        # Define logical state transformations
        # Hadamard gate should map logical states:
        # |0> -> |+> and |+> -> |0>
        logical_state_transformations = [
            (LogicalState("+Z0"), [LogicalState("+X0")]),
            (LogicalState("+X0"), [LogicalState("+Z0")]),
        ]

        debug_data = is_circuit_valid(
            circuit=self.steane_h_logical,
            input_block=self.steane_block,
            output_block=self.steane_block,
            output_stabilizers_parity={},
            output_stabilizers_with_any_value=[],
            logical_state_transformations_with_parity={},
            logical_state_transformations=logical_state_transformations,
            measurement_to_input_stabilizer_map={},
        )

        self.assertTrue(debug_data.valid)

    def test_steane_code_logical_identity_invalid_multi(self):
        """Test the validation of logical identity operation on the Steane code with an
        invalid circuit that implements a logical Hadamard operation."""
        # Define a transformation that would be expected from a logical identity
        identity_logical_state_transformations = [
            (LogicalState("+Z0"), [LogicalState("+Z0")]),
        ]
        # Run Validator validation using the logical hadamard circuit
        debug_data = is_circuit_valid(
            circuit=self.steane_h_logical,
            input_block=self.steane_block,
            output_block=self.steane_block,
            output_stabilizers_parity={},
            output_stabilizers_with_any_value=[],
            logical_state_transformations_with_parity={},
            logical_state_transformations=identity_logical_state_transformations,
            measurement_to_input_stabilizer_map={},
        )

        # The circuit is not correct
        self.assertFalse(debug_data.valid)

        # +Z0 logical operator was not transformed as expected. The expected output
        # was +Z0 but instead it was transformed to +X0.
        expected_output = (
            (LogicalState(("+Z0",)), (LogicalState(("+Z0",)),), LogicalState(("+X0",))),
        )
        logical_output = debug_data.checks.logical_operators.output
        self.assertEqual(
            logical_output.input_vs_expected_vs_actual_logicals_multi, expected_output
        )

    def test_steane_code_logical_identity_invalid_parity(self):
        """Test the validation of logical identity operation on the Steane code with a
        invalid circuit that implements a logical Hadamard operation."""
        # Define a transformation that would be expected from a logical identity
        log_transf_with_parity = {
            LogicalState("+Z0"): (LogicalState("+Z0"), {}),
        }
        # Run Validator validation using the logical identity circuit
        debug_data = is_circuit_valid(
            circuit=self.steane_h_logical,
            input_block=self.steane_block,
            output_block=self.steane_block,
            output_stabilizers_parity={},
            output_stabilizers_with_any_value=[],
            logical_state_transformations_with_parity=log_transf_with_parity,
            logical_state_transformations={},
            measurement_to_input_stabilizer_map={},
        )
        # The circuit is not correct
        self.assertFalse(debug_data.valid)
        # +Z0 logical operator was not transformed as expected. The expected output
        # was +Z0 but instead it was transformed to +X0.
        expected_output = (
            (
                LogicalState(["+Z0"]),
                (LogicalState(["+Z0"]), (0,)),
                LogicalState(["+X0"]),
            ),
        )
        logical_output = debug_data.checks.logical_operators.output
        self.assertEqual(
            logical_output.input_vs_expected_vs_actual_logicals_with_parity,
            expected_output,
        )

    def test_4qubit_code_logical_swap(self):
        """Test the validation of a logical SWAP operation on the 4-qubit code."""
        # Define 4qubit code
        # Stabilizers
        four_qubit_code_stabilizers = [
            Stabilizer("ZZZZ", [(i, 0) for i in range(4)]),
            Stabilizer("XXXX", [(i, 0) for i in range(4)]),
        ]

        # Logical operator set
        four_qubit_code_logical_operator_set = {
            "Z": [
                PauliOperator("ZZ", [(0, 0), (1, 0)]),
                PauliOperator("ZZ", [(0, 0), (2, 0)]),
            ],
            "X": [
                PauliOperator("XX", [(0, 0), (2, 0)]),
                PauliOperator("XX", [(0, 0), (1, 0)]),
            ],
        }
        # Block
        four_qubit_code_block = Block(
            stabilizers=four_qubit_code_stabilizers,
            logical_x_operators=four_qubit_code_logical_operator_set["X"],
            logical_z_operators=four_qubit_code_logical_operator_set["Z"],
        )

        # Define the data qubit channels
        dqubits = [
            Channel(type=ChannelType.QUANTUM, label=str((q, 0))) for q in range(4)
        ]

        # Swapping qubits 1 and 2 should be equivalent to a logical SWAP operation
        circuit = Circuit(
            "4qubit_code_logical_swap",
            circuit=Circuit(
                "SWAP",
                channels=[dqubits[1], dqubits[2]],
            ),
        )

        # Define logical state transformations
        # SWAP gate should map logical states:
        # |00> -> |00> and |++> -> |++>
        # |0+> -> |+0> and |+0> -> |0+>
        logical_state_transformations = [
            (LogicalState(["+Z0", "+Z1"]), [LogicalState(["+Z0", "+Z1"])]),
            (LogicalState(["+X0", "+X1"]), [LogicalState(["+X0", "+X1"])]),
            (LogicalState(["+Z0", "+X1"]), [LogicalState(["+X0", "+Z1"])]),
            (LogicalState(["+X0", "+Z1"]), [LogicalState(["+Z0", "+X1"])]),
        ]

        debug_data = is_circuit_valid(
            circuit=circuit,
            input_block=four_qubit_code_block,
            output_block=four_qubit_code_block,
            output_stabilizers_parity={},
            output_stabilizers_with_any_value=[],
            logical_state_transformations_with_parity={},
            logical_state_transformations=logical_state_transformations,
            measurement_to_input_stabilizer_map={},
        )

        self.assertTrue(debug_data.valid)

    def test_steane_code_x_log_measurement(self):
        """Test the validation of logical X measurement operation on the Steane code."""
        # Define the Steane code block
        steane_block = self.steane_block
        # Define the data qubit channels and an extra auxiliary channel
        dq_channels = [
            Channel(type=ChannelType.QUANTUM, label=str(q))
            for q in steane_block.data_qubits
        ]
        aux_channel = Channel(type=ChannelType.QUANTUM, label="quantum")

        # Define the measurement operation as a projection of the logical operator
        # onto the auxiliary qubit
        circuit = Circuit(
            "steane_code_x_log_measurement",
            circuit=[Circuit("H", channels=[aux_channel])]
            + [
                Circuit("CNOT", channels=[aux_channel, dq_chan])
                for dq_chan in dq_channels
            ]
            + [Circuit("H", channels=[aux_channel])]
            + [
                Circuit(
                    "measurement",
                    channels=[
                        aux_channel,
                        Channel(label="c0", type=ChannelType.CLASSICAL),
                    ],
                )
            ],
        )

        # Define logical state transformations
        # Measurement of X should map logical states:
        # |0> -> |+> or |-> , |+> -> |+>  and |-> -> |->
        logical_state_transformations = [
            (
                LogicalState("+Z0"),
                (LogicalState("+X0"), LogicalState("-X0")),
            ),
            (LogicalState("+X0"), (LogicalState("+X0"),)),
            (LogicalState("-X0"), (LogicalState("-X0"),)),
        ]

        debug_data = is_circuit_valid(
            circuit=circuit,
            input_block=steane_block,
            output_block=steane_block,
            output_stabilizers_parity={},
            output_stabilizers_with_any_value=[],
            logical_state_transformations_with_parity={},
            logical_state_transformations=logical_state_transformations,
            measurement_to_input_stabilizer_map={},
        )

        self.assertTrue(debug_data.valid)

    def test_rot_surface_code_cnot(self):
        """Test the validation of a CNOT operation on the rotated surface code in an
        explicit manner."""
        # CNOT should map the logical operators:
        # ZI -> ZI, IZ -> ZZ, XI -> XX, IX -> IX
        logical_state_transformations = [
            # |00> -> |00>
            (LogicalState(["+Z0", "+Z1"]), [LogicalState(["+Z0", "+Z0Z1"])]),
            # |++> -> |++>
            (LogicalState(["+X0", "+X1"]), [LogicalState(["+X0X1", "+X1"])]),
            # |0+> -> |0+>
            (LogicalState(["+Z0", "+X1"]), [LogicalState(["+Z0", "+X1"])]),
            # |+0> -> |00> + |11> (bell pair)
            (LogicalState(["+X0", "+Z1"]), [LogicalState(["+X0X1", "+Z0Z1"])]),
        ]
        debug_data = is_circuit_valid(
            circuit=self.rsc12_cnot,
            input_block=(self.rsc1_block, self.rsc2_block),
            output_block=(self.rsc1_block, self.rsc2_block),
            output_stabilizers_parity={},
            output_stabilizers_with_any_value=[],
            logical_state_transformations_with_parity={},
            logical_state_transformations=logical_state_transformations,
            measurement_to_input_stabilizer_map={},
        )
        self.assertTrue(debug_data.valid)

    def test_rot_surface_code_cnot_invalid(self):
        """
        Test the validation of a CNOT operation on the rotated surface code in an
        explicit manner with an invalid logical state transformation.
        """
        # Try with an invalid logical state transformation
        invalid_logical_state_transformations = [
            (LogicalState(["+Z0", "+Z1"]), [LogicalState(["+Z0", "+X1"])]),
        ]
        debug_data_invalid = is_circuit_valid(
            circuit=self.rsc12_cnot,
            input_block=(self.rsc1_block, self.rsc2_block),
            output_block=(self.rsc1_block, self.rsc2_block),
            output_stabilizers_parity={},
            output_stabilizers_with_any_value=[],
            logical_state_transformations_with_parity={},
            logical_state_transformations=invalid_logical_state_transformations,
            measurement_to_input_stabilizer_map={},
        )
        # The circuit is not correct
        self.assertFalse(debug_data_invalid.valid)
        # Only the logical operators were not transformed as expected
        self.assertTrue(debug_data_invalid.checks.code_stabilizers.valid)
        self.assertFalse(debug_data_invalid.checks.logical_operators.valid)
        self.assertTrue(debug_data_invalid.checks.stabilizers_measured.valid)

    def test_rot_surface_code_cnot_using_transformations_generator(self):
        """
        Test the validation of a CNOT operation on the rotated surface code using the
        logical state transformations generator.
        """
        # CNOT should map the logical operators:
        logical_state_transformations = logical_state_transformations_to_check(
            ["X0X1", "X1"],  # X0 -> X0X1, X1 -> X1
            ["Z0", "Z1Z0"],  # Z0 -> Z0, Z1 -> Z1Z0
        )
        debug_data = is_circuit_valid(
            circuit=self.rsc12_cnot,
            input_block=(self.rsc1_block, self.rsc2_block),
            output_block=(self.rsc1_block, self.rsc2_block),
            output_stabilizers_parity={},
            output_stabilizers_with_any_value=[],
            logical_state_transformations_with_parity={},
            logical_state_transformations=logical_state_transformations,
            measurement_to_input_stabilizer_map={},
        )
        self.assertTrue(debug_data.valid)

    def test_logical_state_transformations_to_check_invalid_inputs(self):
        """
        Test the validation of the logical state transformations generator.
        """
        # check invalid inputs to logical_state_transformations_to_check
        error_msg1 = (
            "The number of X and Z operators should be the same for the logical state "
            "transformations."
        )
        with self.assertRaises(ValueError) as context1:
            logical_state_transformations_to_check(["X0X1", "X1"], ["Z0", "Z1Z0", "Z1"])
        self.assertIn(error_msg1, str(context1.exception))

        error_msg2 = (
            "The transformed X and Z operators do not generate a valid tableau when "
            "stacked. Check the input sparse Pauli string maps."
        )
        with self.assertRaises(ValueError) as context2:
            logical_state_transformations_to_check(["X0X1", "X1"], ["Z0", "Z0"])
        self.assertIn(error_msg2, str(context2.exception))

    def test_is_logical_operation_circuit_valid(self):
        """
        Test the validation of a logical operation circuit using the wrapper function.
        """
        debug_data = is_logical_operation_circuit_valid(
            circuit=self.rsc12_cnot,
            input_block=(self.rsc1_block, self.rsc2_block),
            x_operators_sparse_pauli_map=["X0X1", "X1"],
            z_operators_sparse_pauli_map=["Z0", "Z0Z1"],
        )
        self.assertTrue(debug_data.valid)


if __name__ == "__main__":
    unittest.main()
