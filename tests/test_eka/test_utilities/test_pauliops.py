"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest

from loom.eka.utilities import UnsignedPauliOp, SignedPauliOp, pauliops_anti_commute


class TestPauliOps(unittest.TestCase):
    """
    Test for Eka utilities.
    """

    def test_basic_unsignedpauliop(self):
        "Test basic UnsignedPauliOp initialization"
        pauli_op0 = UnsignedPauliOp([0, 1, 1, 0])
        pauli_op1 = UnsignedPauliOp.from_string("ZX")
        pauli_op2 = UnsignedPauliOp([1, 1, 0, 0])

        # test nqubits
        self.assertEqual(pauli_op0.nqubits, 2)

        # test equality
        self.assertTrue(pauli_op0 == pauli_op1 != pauli_op2)

    def test_basic_signedpauliop(self):
        """
        Test basic SignedPauliOp initialization.
        """
        pauli_op0 = SignedPauliOp([1, 1, 1, 0, 1])
        pauli_op1 = SignedPauliOp.from_string("-YX")
        pauli_op2 = SignedPauliOp([1, 1, 0, 0, 0])

        # test nqubits
        self.assertEqual(pauli_op0.nqubits, 2)

        # test equality
        self.assertTrue(pauli_op0 == pauli_op1 != pauli_op2)

    def test_signed_pauli_op_from_sparse_string(self):
        """
        Test SignedPauliOp initialization from sparse string.
        """
        # check initialization from sparse string vs dense string
        # Test case 1:
        self.assertEqual(
            SignedPauliOp.from_sparse_string("-X4Y2Z3"),
            SignedPauliOp.from_string("-IIYZX"),
        )
        # Test case 2:
        self.assertEqual(
            SignedPauliOp.from_sparse_string("+Z4Y3", nqubits=6),
            SignedPauliOp.from_string("+IIIYZI"),
        )

        # Test case 3: (with a double digit qubit index)
        self.assertEqual(
            SignedPauliOp.from_sparse_string("+Z1Z10Y0", nqubits=12),
            SignedPauliOp.from_string("+YZ" + "I" * 8 + "ZI"),
        )

        # Test case 4: With lower case letters
        self.assertEqual(
            SignedPauliOp.from_sparse_string("+z1z10y0", nqubits=12),
            SignedPauliOp.from_string("+YZ" + "I" * 8 + "ZI"),
        )

        # Check invalid inputs

        # qubit index out of range
        with self.assertRaises(ValueError) as cm:
            SignedPauliOp.from_sparse_string("+X4Y2Z3", nqubits=3)
        self.assertIn("Qubit index 4 is out of range for 3 qubits.", str(cm.exception))

        # doubly indexed qubit
        with self.assertRaises(ValueError) as cm:
            SignedPauliOp.from_sparse_string("+X4Z4")
        self.assertIn(
            "Qubit indices {4} appear more than once in the Pauli string.",
            str(cm.exception),
        )

        # missing sign
        with self.assertRaises(ValueError) as cm:
            SignedPauliOp.from_sparse_string("X4")
        self.assertIn(
            "The first character of the a Pauli string should be '+' or '-'.",
            str(cm.exception),
        )

        # invalid characters
        with self.assertRaises(ValueError) as cm:
            SignedPauliOp.from_sparse_string("+Q4Y3")
        self.assertIn("Invalid elements in the Pauli string: Q4.", str(cm.exception))

    def test_as_sparse_string(self):
        """
        Test SignedPauliOp as_sparse_string method.
        """
        for sparse_str in ["+Z1Z10Y0", "-X4Y2Z3", "+Z4Y3"]:
            # Use the sparse string to initialize a SignedPauliOp object
            pauili_op1 = SignedPauliOp.from_sparse_string(sparse_str)
            as_sparse_str = pauili_op1.as_sparse_string()
            pauli_op2 = SignedPauliOp.from_sparse_string(as_sparse_str)
            self.assertEqual(pauili_op1, pauli_op2)

    def test_anti_commutation(self):
        """
        Test pauliop anti-commutation.
        """
        pauli_op0 = SignedPauliOp.from_string("-YXZ")
        pauli_op1 = SignedPauliOp.from_string("+ZXI")
        pauli_op2 = SignedPauliOp.from_string("+ZZI")

        self.assertTrue(pauliops_anti_commute(pauli_op0, pauli_op1))
        self.assertFalse(pauliops_anti_commute(pauli_op0, pauli_op2))
        self.assertTrue(pauliops_anti_commute(pauli_op1, pauli_op2))

    def test_pauliop_multiplication(self):
        """
        Test multiplication of Pauli operators.
        """
        nqubits = 3
        pauli_op0 = SignedPauliOp.from_string("-YXZ")
        pauli_op1 = SignedPauliOp.from_string("+ZZI")
        pauli_op2 = SignedPauliOp.from_string("-XYI")

        # Check multiplication results
        self.assertEqual(pauli_op0 * pauli_op1, SignedPauliOp.from_string("-XYZ"))
        self.assertEqual(pauli_op0 * pauli_op2, SignedPauliOp.from_string("+ZZZ"))
        self.assertEqual(pauli_op1 * pauli_op2, SignedPauliOp.from_string("-YXI"))

        # Check identity multiplication
        for pauli_op in [pauli_op0, pauli_op1, pauli_op2]:
            self.assertEqual(SignedPauliOp.identity(nqubits) * pauli_op, pauli_op)

    def test_a_comm_pauliop_multiplication(self):
        """
        Test multiplication of Pauli operators that anti-commute.
        """
        pauli_op0 = SignedPauliOp.from_string("-YXZ")
        pauli_op1 = SignedPauliOp.from_string("+ZXI")
        pauli_op2 = SignedPauliOp.from_string("-XYX")

        # Check multiplication results
        # i * (-YXZ) * (+ZXI) = -i (iX) I Z = + XIZ
        self.assertEqual(
            pauli_op0.multiply_with_anticommuting_operator(pauli_op1),
            SignedPauliOp.from_string("+XIZ"),
        )
        # invert the order and the sign changes
        self.assertEqual(
            pauli_op1.multiply_with_anticommuting_operator(pauli_op0),
            SignedPauliOp.from_string("-XIZ"),
        )
        # i(-YXZ)(-XYX)= i(YX)(XY)(ZX) = i(-iZ)(iZ)(iY)= -ZZY
        self.assertEqual(
            pauli_op0.multiply_with_anticommuting_operator(pauli_op2),
            SignedPauliOp.from_string("-ZZY"),
        )
        # invert the order and the sign changes
        self.assertEqual(
            pauli_op2.multiply_with_anticommuting_operator(pauli_op0),
            SignedPauliOp.from_string("+ZZY"),
        )

        # check that commuting operators cannot be multiplied
        with self.assertRaises(ValueError):
            pauli_op1.multiply_with_anticommuting_operator(pauli_op2)

    def test_sign_flipping(self):
        """
        Test basic SignedPauliOp sign flipping.
        """
        self.assertEqual(
            SignedPauliOp.from_string("-YX").with_flipped_sign(),
            SignedPauliOp.from_string("+YX"),
        )

    def test_signed_pauli_op_reindex(self):
        """
        Test reindexing of SignedPauliOp.
        """
        pauli_op = SignedPauliOp.from_string("-YXZ")

        # Check reindexing for SignedPauliOp
        self.assertEqual(
            pauli_op.reindexed([3, 0, 4], nqubits=5),
            SignedPauliOp.from_string("-XIIYZ"),
        )

        # Check invalid reindexing due to qubit index out of range
        with self.assertRaises(ValueError):
            pauli_op.reindexed([3, 0, 4], nqubits=4)

        # Check invalid reindexing due to repeated qubit index
        with self.assertRaises(ValueError):
            pauli_op.reindexed([0, 0, 1])


if __name__ == "__main__":
    unittest.main()
