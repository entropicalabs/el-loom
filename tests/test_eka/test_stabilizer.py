"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest
from itertools import combinations

from loom.eka import Stabilizer, PauliOperator
from loom.eka.utilities import SignedPauliOp, loads, dumps


class TestStabilizer(unittest.TestCase):
    """
    Test for the Stabilizer class.
    """

    def test_creation_stabilizer(self):
        """
        Tests the creation of a Stabilizer object.
        """
        stab = Stabilizer(
            pauli="XYZ",
            data_qubits=(
                (0, 0),
                (0, 1),
                (1, 0),
            ),
            ancilla_qubits=((1, 1),),
        )

        self.assertEqual(stab.pauli, "XYZ")
        self.assertEqual(
            stab.data_qubits,
            (
                (0, 0),
                (0, 1),
                (1, 0),
            ),
        )
        self.assertEqual(stab.ancilla_qubits, ((1, 1),))

    def test_creation_stabilizer_using_lists(self):
        """
        Tests the creation of a Stabilizer object using a list of coordinates.
        """
        stab = Stabilizer(
            pauli="XYZ",
            data_qubits=[
                [0, 0],
                [0, 1],
                [1, 0],
            ],
            ancilla_qubits=[
                [1, 1],
            ],
        )
        # Check that the data_qubits and ancilla_qubits are converted to tuples
        self.assertEqual(stab.pauli, "XYZ")
        self.assertEqual(
            stab.data_qubits,
            (
                (0, 0),
                (0, 1),
                (1, 0),
            ),
        )
        self.assertEqual(stab.ancilla_qubits, ((1, 1),))

    def test_wrong_nr_of_data_qubits(self):
        """
        Tests that an error is raised if the number of data qubits is not equal to
        the length of the pauli string.
        """
        with self.assertRaises(ValueError):
            Stabilizer(
                pauli="XYZ",
                data_qubits=(
                    (0, 0),
                    (0, 1),
                ),
                ancilla_qubits=((1, 1),),
            )

    def test_non_unique_qubits(self):
        """
        Tests that an error is raised if the qubits are not unique.
        """
        with self.assertRaises(ValueError):
            Stabilizer(
                pauli="XYZ",
                data_qubits=(
                    (0, 0),
                    (0, 1),
                    (0, 0),
                ),
                ancilla_qubits=((1, 1),),
            )
        with self.assertRaises(ValueError):
            Stabilizer(
                pauli="XYZ",
                data_qubits=(
                    (0, 0),
                    (0, 1),
                    (1, 1),
                ),
                ancilla_qubits=((1, 1), (1, 1)),
            )

    def test_commutes_with_stabilizer(self):
        """
        Tests the Stabilizer().commutes_with method.
        """
        stab1 = Stabilizer(
            pauli="XZ",
            data_qubits=((0,), (1,)),
            ancilla_qubits=((5,),),
        )
        stab2 = Stabilizer(
            pauli="X",
            data_qubits=((0,),),
            ancilla_qubits=((6,),),
        )
        stab3 = Stabilizer(
            pauli="Z",
            data_qubits=((0,),),
            ancilla_qubits=((7,),),
        )
        stab4 = Stabilizer(
            pauli="Z",
            data_qubits=((1,),),
            ancilla_qubits=((8,),),
        )

        self.assertTrue(stab1.commutes_with(stab1))
        self.assertTrue(stab2.commutes_with(stab2))
        self.assertTrue(stab3.commutes_with(stab3))
        self.assertTrue(stab4.commutes_with(stab4))
        self.assertTrue(stab1.commutes_with(stab2))
        self.assertTrue(not stab1.commutes_with(stab3))
        self.assertTrue(stab1.commutes_with(stab4))
        self.assertTrue(not stab2.commutes_with(stab3))
        self.assertTrue(stab2.commutes_with(stab4))
        self.assertTrue(stab3.commutes_with(stab4))

    def test_commutes_with_paulioperator(self):
        """
        Test that we can use commutes_with between Stabilizer and PauliOperator.
        """
        stab1 = Stabilizer(
            pauli="XZ",
            data_qubits=((0,), (1,)),
            ancilla_qubits=((5,),),
        )
        stab2 = Stabilizer(
            pauli="Z",
            data_qubits=((0,),),
            ancilla_qubits=((6,),),
        )
        op1 = PauliOperator(
            pauli="XZ",
            data_qubits=((0,), (1,)),
        )
        op2 = PauliOperator(
            pauli="Z",
            data_qubits=((0,),),
        )

        self.assertTrue(stab1.commutes_with(stab1))
        self.assertTrue(stab2.commutes_with(stab2))
        self.assertTrue(op1.commutes_with(op1))
        self.assertTrue(op2.commutes_with(op2))
        self.assertTrue(not stab1.commutes_with(stab2))
        self.assertTrue(stab1.commutes_with(op1))
        self.assertTrue(not stab1.commutes_with(op2))
        self.assertTrue(not stab2.commutes_with(op1))
        self.assertTrue(stab2.commutes_with(op2))
        self.assertTrue(not op1.commutes_with(op2))

    def test_commutes_with_stabilizer_order_preserved(self):
        """
        Test that all the stabilizers commute with each other. This example comes
        from the rotated surface code with an extra stabilizer that is a multiple of two
        existing stabilizers. All the stabilizers commute with each other.
        """
        stabs = [
            Stabilizer("ZZZZ", ((0, 0), (0, 1), (1, 1), (1, 0))),
            Stabilizer("ZZZZ", ((1, 1), (1, 2), (2, 2), (2, 1))),
            Stabilizer("XXXX", ((0, 1), (0, 2), (1, 2), (1, 1))),
            Stabilizer("XXXX", ((1, 0), (1, 1), (2, 1), (2, 0))),
            Stabilizer("XX", ((0, 0), (0, 1))),
            Stabilizer("XX", ((2, 1), (2, 2))),
            Stabilizer("ZZ", ((1, 0), (2, 0))),
            Stabilizer("ZZ", ((0, 2), (1, 2))),
            Stabilizer("ZZXX", ((0, 2), (1, 2), (2, 1), (2, 2))),
        ]
        for a, b in combinations(stabs, 2):
            self.assertTrue(a.commutes_with(b))

    def test_as_signed_pauli_op(self):
        """
        Test that the as_signed_pauli_op method works correctly.
        """
        op = Stabilizer(pauli="XZ", data_qubits=((0,), (2,)), ancilla_qubits=[])
        signed_pauli_op = op.as_signed_pauli_op(((0,), (1,), (2,), (3,)))
        reference_pauli_op = SignedPauliOp.from_string("+XIZI")

        self.assertEqual(signed_pauli_op, reference_pauli_op)

        # Test as_signed_pauli_op with a different qubit ordering
        op = Stabilizer(pauli="XZ", data_qubits=((2,), (0,)), ancilla_qubits=[])
        signed_pauli_op = op.as_signed_pauli_op(((3,), (1,), (2,), (0,)))
        reference_pauli_op = SignedPauliOp.from_string("+IIXZ")

        self.assertEqual(signed_pauli_op, reference_pauli_op)

        # Test as_signed_pauli_op with a list of lists input
        op = Stabilizer(pauli="XZ", data_qubits=[[0], [1]], ancilla_qubits=[])
        signed_pauli_op = op.as_signed_pauli_op(([0], [1], [2], [3]))
        reference_pauli_op = SignedPauliOp.from_string("+XZII")

        self.assertEqual(signed_pauli_op, reference_pauli_op)

    def test_coordinate_length(self):
        """
        Test that an error is raised if the qubits do not have the same length.
        """
        with self.assertRaises(ValueError) as cm:
            Stabilizer(pauli="XZX", data_qubits=((0,), (1,), (0, 1)), ancilla_qubits=[])
        self.assertIn("Length of coordinates must be consistent.", str(cm.exception))

    def test_loads_dumps(self):
        """
        Test that the loads and dumps functions work correctly.
        """
        stab = Stabilizer(
            pauli="XY",
            data_qubits=(
                (0, 0),
                (1, 0),
            ),
            ancilla_qubits=((2, 0),),
        )

        stab_json = dumps(stab)
        loaded_stab = loads(Stabilizer, stab_json)

        self.assertEqual(loaded_stab, stab)


if __name__ == "__main__":
    unittest.main()
