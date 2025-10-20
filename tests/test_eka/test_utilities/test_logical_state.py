"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest

from loom.eka import (
    Block,
    Stabilizer,
    PauliOperator,
    LogicalState,
)

from loom.eka.utilities import (
    is_tableau_valid,
    is_stabarray_equivalent,
    StabArray,
    SignedPauliOp,
)


class TestLogicalState(unittest.TestCase):
    """Tests for the LogicalState class."""

    def setUp(self) -> None:
        # Define a full 4-qubit code with two logical qubits.
        stabilizers = [
            Stabilizer("ZZZZ", ((0,), (1,), (2,), (3,))),
            Stabilizer("XXXX", ((0,), (1,), (2,), (3,))),
        ]

        x_operators = [
            PauliOperator("XX", ((0,), (1,))),
            PauliOperator("XX", ((0,), (2,))),
        ]
        z_operators = [
            PauliOperator("ZZ", ((0,), (2,))),
            PauliOperator("ZZ", ((0,), (1,))),
        ]

        self.full_4q_code_block = Block(
            stabilizers=stabilizers,
            logical_x_operators=x_operators,
            logical_z_operators=z_operators,
        )

        self.repc = Block(
            stabilizers=(
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((0, 0, 0), (0, 1, 0)),
                    ancilla_qubits=((0, 1, 1),),
                ),
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((0, 1, 0), (0, 2, 0)),
                    ancilla_qubits=((0, 2, 1),),
                ),
            ),
            logical_x_operators=[
                PauliOperator(
                    pauli="XXX", data_qubits=((0, 0, 0), (0, 1, 0), (0, 2, 0))
                )
            ],
            logical_z_operators=[PauliOperator(pauli="Z", data_qubits=((0, 0, 0),))],
            unique_label="q1",
        )

    def test_logical_state_init(self):
        """Test simple initializations of LogicalState."""
        # Test initialization of LogicalState with a single logical qubit.
        ls0 = LogicalState(("+Z0",))
        self.assertEqual(ls0.sparse_logical_paulistrings, ("+Z0",))
        # Should also work without the tuple.
        ls1 = LogicalState("+Y0")
        self.assertEqual(ls1.sparse_logical_paulistrings, ("+Y0",))

        # Test initialization of LogicalState with multiple logical qubits.
        ls2 = LogicalState(("+Z0", "+Z1"))
        self.assertEqual(ls2.sparse_logical_paulistrings, ("+Z0", "+Z1"))
        ls3 = LogicalState(("+Z0X1", "+Z1X0", "-Y2"))
        self.assertEqual(ls3.sparse_logical_paulistrings, ("+Z0X1", "+Z1X0", "-Y2"))

    def test_logical_state_init_invalid(self):
        """Test invalid initializations of LogicalState."""
        # Test initialization of LogicalState with an invalid logical operator.
        with self.assertRaises(ValueError) as cm:
            LogicalState(("+Z0X0", "+Z1"))
        self.assertIn(
            "Qubit indices {0} appear more than once in the Pauli string.",
            str(cm.exception),
        )

        # Test non irreducible set
        with self.assertRaises(ValueError) as cm:
            LogicalState(("+Z0Z1", "+Z1Z2", "+Z2Z0"))
        self.assertIn(
            "The set of logical paulistrings is not irreducible.", str(cm.exception)
        )

        # Test initialization of LogicalState with missing stabilizers
        with self.assertRaises(ValueError) as cm:
            LogicalState(("+Z0Z1", "+X2"))
        self.assertIn(
            "The number of sparse stabilizers (2) does not "
            "match the maximum logical qubits required by the operators "
            "(3).",
            str(cm.exception),
        )

    def test_tableau_generation1(self):
        """Test the generation of the tableau of a |00> LogicalState."""
        logical_state = LogicalState(("+Z0", "+Z1"))
        tableau = logical_state.get_tableau(self.full_4q_code_block)
        # Isolate the code generators and logical operators.
        tableau_stabilizers_stabarr = StabArray(tableau[4:])
        correct_stabilizers_stabarr = StabArray.from_signed_pauli_ops(
            # the code generators
            self.full_4q_code_block.reduced_stabarray[:]
            + [
                # Z0_L = Z0Z2
                SignedPauliOp.from_sparse_string("+Z0Z2", nqubits=4),
                # Z1_L = Z0Z1
                SignedPauliOp.from_sparse_string("+Z0Z1", nqubits=4),
            ]
        )

        # check that the stabilizer part that defines the state is correct
        self.assertTrue(
            is_stabarray_equivalent(
                tableau_stabilizers_stabarr, correct_stabilizers_stabarr
            )
        )
        # check that the tableau is valid, i.e. that the defined state is accompanied
        # by a valid destabilizer array
        self.assertTrue(is_tableau_valid(tableau))

    def test_tableau_generation2(self):
        """Test the generation of the tableau of an entangled LogicalState."""
        logical_state = LogicalState(("+Y0X1", "+Z0Z1"))
        tableau = logical_state.get_tableau(self.full_4q_code_block)
        # Isolate the code generators and logical operators.
        tableau_stabilizers_stabarr = StabArray(tableau[4:])

        correct_tableau_stabarr = StabArray.from_signed_pauli_ops(
            # the code generators
            self.full_4q_code_block.reduced_stabarray[:]
            + [
                # Y0_L = i X0_L Z0_L = i X0X1 Z0Z2 = Y0X1Z2    and    X1_L = X0 X2
                # so:
                # Y0_L * X1_L = Y0X1Z2 * X0 X2 = + Z0X1Y2
                SignedPauliOp.from_sparse_string("+Z0X1Y2", nqubits=4),
                # Z0_L = Z0 Z2    and    Z1_L = Z0 Z1
                # so:
                # Z0_L * Z1_L = Z0Z2 * Z0Z1 = + Z1 Z2
                SignedPauliOp.from_sparse_string("+Z1Z2", nqubits=4),
            ]
        )

        # check that the stabilizer part that defines the state is correct
        self.assertTrue(
            is_stabarray_equivalent(
                tableau_stabilizers_stabarr, correct_tableau_stabarr
            )
        )
        # check that the tableau is valid, i.e. that the defined state is accompanied
        # by a valid destabilizer array
        self.assertTrue(is_tableau_valid(tableau))

    def test_rot_surface_code(self):
        """Test the generation of the tableau of a rotated surface code block."""
        log_state = LogicalState("+Z0")
        # test that the tableau can be generated
        tab = log_state.get_tableau(self.repc)
        # test that the tableau is valid
        self.assertTrue(is_tableau_valid(tab))


if __name__ == "__main__":
    unittest.main()
