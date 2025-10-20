"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest

from loom.eka import PauliOperator
from loom.eka.utilities import dumps, loads, SignedPauliOp


class TestPauliOperator(unittest.TestCase):
    """
    Test for the PauliOperator class.
    """

    def test_creation_pauli_operator(self):
        """
        Test the creation of a PauliOperator object.
        """
        pauli_operator = PauliOperator(pauli="XXX", data_qubits=((0,), (1,), (2,)))

        self.assertEqual(pauli_operator.pauli, "XXX")
        self.assertEqual(pauli_operator.data_qubits, ((0,), (1,), (2,)))

    def test_creation_pauli_operator_from_list(self):
        """
        Test the creation of a PauliOperator object.
        """
        pauli_operator = PauliOperator(pauli="XXX", data_qubits=[[0], [1], [2]])

        self.assertEqual(pauli_operator.pauli, "XXX")
        self.assertEqual(pauli_operator.data_qubits, ((0,), (1,), (2,)))

    def test_wrong_nr_of_qubits(self):
        """
        Test that an error is raised if the number of qubits is not equal to
        the length of the pauli string.
        """
        with self.assertRaises(ValueError):
            PauliOperator(
                pauli="XXX",
                data_qubits=(
                    (0,),
                    (1,),
                ),
            )

    def test_non_unique_qubits(self):
        """
        Test that an error is raised if the qubits are not unique.
        """
        with self.assertRaises(ValueError):
            PauliOperator(
                pauli="XXX",
                data_qubits=(
                    (0,),
                    (1,),
                    (0,),
                ),
            )

    def test_commutes_with_paulioperator(self):
        """
        Tests the PauliOperator().commutes_with method.
        """
        op1 = PauliOperator(
            pauli="XZ",
            data_qubits=((0,), (1,)),
        )
        op2 = PauliOperator(
            pauli="X",
            data_qubits=((0,),),
        )
        op3 = PauliOperator(
            pauli="Z",
            data_qubits=((0,),),
        )
        op4 = PauliOperator(
            pauli="Z",
            data_qubits=((1,),),
        )

        self.assertTrue(op1.commutes_with(op1))
        self.assertTrue(op2.commutes_with(op2))
        self.assertTrue(op3.commutes_with(op3))
        self.assertTrue(op4.commutes_with(op4))
        self.assertTrue(op1.commutes_with(op2))
        self.assertTrue(not op1.commutes_with(op3))
        self.assertTrue(op1.commutes_with(op4))
        self.assertTrue(not op2.commutes_with(op3))
        self.assertTrue(op2.commutes_with(op4))
        self.assertTrue(op3.commutes_with(op4))

    def test_as_signed_pauli_op(self):
        """
        Test that the as_signed_pauli_op method works correctly.
        """
        op = PauliOperator(pauli="XZ", data_qubits=((0,), (2,)))
        signed_pauli_op = op.as_signed_pauli_op(((0,), (1,), (2,), (3,)))
        reference_pauli_op = SignedPauliOp.from_string("+XIZI")

        self.assertEqual(signed_pauli_op, reference_pauli_op)

        # Test as_signed_pauli_op with a different qubit ordering
        op = PauliOperator(pauli="XZ", data_qubits=((2,), (0,)))
        signed_pauli_op = op.as_signed_pauli_op(((3,), (1,), (2,), (0,)))
        reference_pauli_op = SignedPauliOp.from_string("+IIXZ")

        self.assertEqual(signed_pauli_op, reference_pauli_op)

        # Test as_signed_pauli_op with a list of lists input
        op = PauliOperator(pauli="XZ", data_qubits=[[0], [1]])
        signed_pauli_op = op.as_signed_pauli_op(([0], [1], [2], [3]))
        reference_pauli_op = SignedPauliOp.from_string("+XZII")

        self.assertEqual(signed_pauli_op, reference_pauli_op)

    def test_from_signed_pauli_op(self):
        """
        Test that the from_signed_pauli_op method works correctly.
        """
        signed_pauli_op = SignedPauliOp.from_string("+XIZI")
        op = PauliOperator.from_signed_pauli_op(
            signed_pauli_op, {0: (0,), 1: (1,), 2: (2,), 3: (3,)}
        )

        self.assertEqual(op.pauli, "XZ")
        self.assertEqual(op.data_qubits, ((0,), (2,)))

        # Test from_signed_pauli_op with a different qubit ordering
        signed_pauli_op = SignedPauliOp.from_string("+IZXY")
        op = PauliOperator.from_signed_pauli_op(
            signed_pauli_op, {0: (3,), 1: (1,), 2: (2,), 3: (0,)}
        )

        self.assertEqual(op.pauli, "ZXY")
        self.assertEqual(op.data_qubits, ((1,), (2,), (0,)))

        # Test missing qubit coordinates
        signed_pauli_op = SignedPauliOp.from_string("+IZXY")
        with self.assertRaises(ValueError) as cm:
            PauliOperator.from_signed_pauli_op(signed_pauli_op, {0: (3,), 1: (1,)})
        exp_err_msg = "Missing qubit coordinates for indices [2, 3]."
        self.assertIn(exp_err_msg, str(cm.exception))

    def test_coordinate_length(self):
        """
        Test that an error is raised if the qubits do not have the same length.
        """
        with self.assertRaises(ValueError) as cm:
            PauliOperator(pauli="XXX", data_qubits=((0,), (1,), (0, 1)))
        self.assertIn("Length of coordinates must be consistent.", str(cm.exception))

    def test_loads_dumps(self):
        """
        Test that the loads and dumps functions work correctly.
        """
        pauli_op = PauliOperator(pauli="XZ", data_qubits=((0, 0), (1, 0)))

        pauli_op_json = dumps(pauli_op)
        loaded_pauli_op = loads(PauliOperator, pauli_op_json)

        self.assertEqual(loaded_pauli_op, pauli_op)


if __name__ == "__main__":
    unittest.main()
