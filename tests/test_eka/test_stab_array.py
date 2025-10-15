"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest
from itertools import product
from functools import reduce

import numpy as np

from loom.eka.utilities import (
    SignedPauliOp,
    StabArray,
    stabarray_bge,
    pauliops_anti_commute,
    find_destabarray,
    is_stabarray_equivalent,
    merge_stabarrays,
    stabarray_bge_with_bookkeeping,
    reduce_stabarray_with_bookkeeping,
    invert_bookkeeping_matrix,
    paulichar_to_xz,
    is_tableau_valid,
    AntiCommutationError,
)


class TestStabArray(unittest.TestCase):
    """
    Test for Eka utilities.
    """

    @classmethod
    def setUpClass(cls) -> None:  # pylint: disable=too-many-locals
        """
        Initializes a dictionary containing all two and three qubit operators.
        """
        # Get all 2 and 3 qubit pauli operators
        cls.p_ops: dict[str, np.ndarray] = {}

        for nqubits in [2, 3]:
            for pauli_tuple in product(["_", "X", "Y", "Z"], repeat=nqubits):
                pauli_vector = np.zeros(2 * nqubits, dtype=np.int8)
                # initialize pauli vector without the sign
                for i, p in enumerate(pauli_tuple):
                    x, z = paulichar_to_xz(p)
                    pauli_vector[i] = x
                    pauli_vector[i + nqubits] = z

                # add sign and store it in the dictionary
                p_string = "".join(pauli_tuple)
                cls.p_ops[f"+{p_string}"] = SignedPauliOp(np.append(pauli_vector, 0))
                cls.p_ops[f"-{p_string}"] = SignedPauliOp(np.append(pauli_vector, 1))

        # Manually define the stabilizer arrays for different codes
        stabs_steane_code = [
            "111100000000000",
            "011011000000000",
            "001101100000000",
            "000000011110000",
            "000000001101100",
            "000000000110110",
        ]
        stabs_five_qubit_code = [
            "10010011000",
            "01001001100",
            "10100000110",
            "01010100010",
        ]
        stabs_repetition_code_5_qubits_z_checks = [
            "00000110000",
            "00000011000",
            "00000001100",
            "00000000110",
        ]
        stabs_repetition_code_5_qubits_x_checks = [
            "11000000000",
            "01100000000",
            "00110000000",
            "00011000000",
        ]
        stabs_repetition_code_5_qubits_y_checks = [
            "11000110000",
            "01100011000",
            "00110001100",
            "00011000110",
        ]
        stabs_bacon_shor_code_5x5_qubits = [
            "111111111100000000000000000000000000000000000000000",
            "000001111111111000000000000000000000000000000000000",
            "000000000011111111110000000000000000000000000000000",
            "000000000000000111111111100000000000000000000000000",
            "000000000000000000000000011000110001100011000110000",
            "000000000000000000000000001100011000110001100011000",
            "000000000000000000000000000110001100011000110001100",
            "000000000000000000000000000011000110001100011000110",
        ]
        stabs_rotated_surface_code_3x3_qubits = [
            "0000000001101100000",
            "0110110000000000000",
            "0001101100000000000",
            "0000000000000110110",
            "1001000000000000000",
            "0000010010000000000",
            "0000000000000001100",
            "0000000000110000000",
        ]
        stabs_color_code_488_distance_5 = [
            "01110100000000000000000000000000000",
            "00000000000000000011101000000000000",
            "00011110000000000000000000000000000",
            "00000000000000000000111100000000000",
            "11011000000000000000000000000000000",
            "00000000000000000110110000000000000",
            "00000001110100000000000000000000000",
            "00000000000000000000000011101000000",
            "00000000011110000000000000000000000",
            "00000000000000000000000000111100000",
            "00000000000001111000000000000000000",
            "00000000000000000000000000000011110",
            "00100111011001010000000000000000000",
            "00000000000000000001001110110010100",
            "00000000001010011000000000000000000",
            "00000000000000000000000000010100110",
        ]
        stabs_xzzx_surface_code_3x5_qubits = [
            "1000001000000000100010000000000",
            "0100000100000000010001000000000",
            "0010000010000000001000100000000",
            "0001000001000000000100010000000",
            "0000010000010000000001000100000",
            "0000001000001000000000100010000",
            "0000000100000100000000010001000",
            "0000000010000010000000001000100",
            "0000010000000001000000000000000",
            "0000000001000000000000000000010",
            "0000000000100000000000000010000",
            "0010000000000000100000000000000",
            "0000000000001000000000000000100",
            "0000100000000000001000000000000",
        ]

        stab_array_steane_code = StabArray(
            np.array(
                [list(s) for s in stabs_steane_code],
                dtype=int,
            )
        )
        stab_array_five_qubit_code = StabArray(
            np.array(
                [list(s) for s in stabs_five_qubit_code],
                dtype=int,
            )
        )
        stab_array_repetition_code_5_qubits_z_checks = StabArray(
            np.array(
                [list(s) for s in stabs_repetition_code_5_qubits_z_checks],
                dtype=int,
            )
        )
        stab_array_repetition_code_5_qubits_x_checks = StabArray(
            np.array(
                [list(s) for s in stabs_repetition_code_5_qubits_x_checks],
                dtype=int,
            )
        )
        stab_array_repetition_code_5_qubits_y_checks = StabArray(
            np.array(
                [list(s) for s in stabs_repetition_code_5_qubits_y_checks],
                dtype=int,
            )
        )
        stab_array_bacon_shor_code_5x5_qubits = StabArray(
            np.array(
                [list(s) for s in stabs_bacon_shor_code_5x5_qubits],
                dtype=int,
            )
        )
        stab_array_rotated_surface_code_3x3_qubits = StabArray(
            np.array(
                [list(s) for s in stabs_rotated_surface_code_3x3_qubits],
                dtype=int,
            )
        )
        stab_array_color_code_488_distance_5 = StabArray(
            np.array(
                [list(s) for s in stabs_color_code_488_distance_5],
                dtype=int,
            )
        )
        stab_array_xzzx_surface_code_3x5_qubits = StabArray(
            np.array(
                [list(s) for s in stabs_xzzx_surface_code_3x5_qubits],
                dtype=int,
            )
        )
        cls.code_check_stab_arrays = [
            stab_array_steane_code,
            stab_array_five_qubit_code,
            stab_array_repetition_code_5_qubits_z_checks,
            stab_array_repetition_code_5_qubits_x_checks,
            stab_array_repetition_code_5_qubits_y_checks,
            stab_array_bacon_shor_code_5x5_qubits,
            stab_array_rotated_surface_code_3x3_qubits,
            stab_array_color_code_488_distance_5,
            stab_array_xzzx_surface_code_3x5_qubits,
        ]

    def test_stabarray_init(self):
        """
        Tests the initialization of stabarrays.
        """
        StabArray.from_signed_pauli_ops([self.p_ops["+_Z"], self.p_ops["+Z_"]])

        # faulty stabarray initializations
        with self.assertRaises(AntiCommutationError):
            StabArray.from_signed_pauli_ops([self.p_ops["+_Z"], self.p_ops["+YY"]])
        with self.assertRaises(ValueError):
            StabArray.from_signed_pauli_ops([self.p_ops["+_Z"], self.p_ops["+_YX"]])

    def test_stab_array_bge(self):
        """
        Test stabilizer BGE function.
        """
        # two arrays that correspond to the same stabilizer set
        stabarray0 = StabArray.from_signed_pauli_ops(
            (self.p_ops["+Z_"], self.p_ops["+_X"])
        )
        stabarray1 = StabArray.from_signed_pauli_ops(
            (self.p_ops["+ZX"], self.p_ops["+_X"])
        )

        # after stabilizer_bge they should be the same
        stabarray0_bge = stabarray_bge(stabarray0)
        stabarray1_bge = stabarray_bge(stabarray1)
        self.assertTrue(np.all(stabarray0_bge.array == stabarray1_bge.array))

        # a third one that is not the same
        stab_array2 = StabArray.from_signed_pauli_ops(
            (self.p_ops["+ZX"], self.p_ops["-_X"])
        )
        stab_array2_bge = stabarray_bge(stab_array2)
        self.assertFalse(np.all(stabarray0_bge.array == stab_array2_bge.array))

    def test_op_anti_commute(self):
        """
        Test stabilizer operator anti-commutation.
        """
        # identity testing
        for p in ["+XZ", "+__", "-__", "+Y_"]:
            a_comm = pauliops_anti_commute(self.p_ops["+__"], self.p_ops[p])
            self.assertEqual(a_comm, 0)

        # test anti-commuting
        a_comm = pauliops_anti_commute(self.p_ops["+ZZ"], self.p_ops["+_Y"])
        self.assertEqual(a_comm, 1)

        # test commuting
        a_comm = pauliops_anti_commute(self.p_ops["+ZZ"], self.p_ops["+XY"])
        self.assertEqual(a_comm, 0)

    def test_destabilizer_array_simple(self):
        """
        Test destabilizer array finding for a simple case.
        """
        # We know that the all zeros state has:
        stab_array = StabArray.from_signed_pauli_ops(
            (self.p_ops["+Z_"], self.p_ops["+_Z"])
        )
        destab_array = StabArray.from_signed_pauli_ops(
            (self.p_ops["+X_"], self.p_ops["+_X"])
        )

        # find destab array via method
        destab_array_test = find_destabarray(stab_array)

        # assert that they are the same
        self.assertTrue(is_stabarray_equivalent(destab_array, destab_array_test))

    def test_destabilizer_array_complicated(self):
        """
        Test destabilizer array finding for a more complicated state.
        """
        # We know that the all zeros state has:
        stab_array = StabArray.from_signed_pauli_ops(
            (self.p_ops["+XX"], self.p_ops["+ZZ"])
        )

        # find destab array via method
        destab_array = find_destabarray(stab_array)

        # assert that the result can generate a valid tableau
        self.assertTrue(
            is_tableau_valid(np.vstack((destab_array.array, stab_array.array)))
        )

    def test_destabilizer_array_edge_case(self):
        """
        Test destabilizer array finding for an edge case. That is the repetition code
        with X stabilizers for distance 65. As a logical operator we use the -Y operator
        """
        check_array = np.zeros((65, 131), dtype=np.int8)
        # Set the XX stabilizers
        for i in range(64):
            check_array[i, i] = 1
            check_array[i, i + 1] = 1
        # Set the -Y logical operator in the end
        check_array[64, :] = np.ones(131, dtype=np.int8)

        # Create the StabArray object
        stab_array = StabArray(check_array)
        # Find the destabilizer array
        destab_array = find_destabarray(stab_array)
        # Check that the result can generate a valid tableau
        self.assertTrue(
            is_tableau_valid(np.vstack((destab_array.array, stab_array.array)))
        )

    def test_tableau_validity(self):
        """
        Test tableau validity.
        """
        # We know that the all zeros state has:
        stab_array = StabArray.from_signed_pauli_ops(
            (self.p_ops["+Z_"], self.p_ops["+_Z"])
        )
        destab_array = StabArray.from_signed_pauli_ops(
            (self.p_ops["+X_"], self.p_ops["+_X"])
        )

        # form tableau
        tab = np.vstack((destab_array.array, stab_array.array))
        self.assertTrue(is_tableau_valid(tab))

        # form some invalid tableaus
        # invalid tableau 1
        invalid_tab = np.vstack((stab_array.array, stab_array.array))
        self.assertFalse(is_tableau_valid(invalid_tab))

        # invalid tableau 2
        invalid_tab = np.vstack((destab_array.array, destab_array.array))
        self.assertFalse(is_tableau_valid(invalid_tab))

    def test_stab_set_from_array(self):
        """
        Test stabilizer_set from array.
        """
        stabs = ["+Z_", "+_Y"]
        stab_array = StabArray.from_signed_pauli_ops([self.p_ops[ps] for ps in stabs])
        self.assertEqual(stab_array.as_paulistrings, stabs)

    def test_stabarray_bookkeeping(self):
        """
        Test the bookkeeping of the stabilizer array.
        """
        for stab_array in self.code_check_stab_arrays:
            # Get the row echelon form of the stabilizer array with the bookkeeping
            stab_array_bge, bookkeeping = stabarray_bge_with_bookkeeping(stab_array)

            # Reconstruct the bge stabilizer array from the bookkeeping and the original
            # stabilizer array
            stab_array_bge_reconstructed = StabArray.from_signed_pauli_ops(
                [
                    reduce(
                        lambda x, y: x * y,
                        stab_array[bookkeeping[i]],
                        SignedPauliOp.identity(stab_array.nqubits),
                    )
                    for i in range(len(bookkeeping))
                ]
            )
            # Check that the bge stabilizer arrays are the same
            self.assertTrue(
                np.array_equal(stab_array_bge.array, stab_array_bge_reconstructed.array)
            )

            # Inverse the row echelon form
            bookkeeping_inverse = invert_bookkeeping_matrix(bookkeeping)
            stab_array_reconstructed = StabArray.from_signed_pauli_ops(
                [
                    reduce(
                        lambda x, y: x * y,
                        stab_array_bge[bookkeeping_inverse[i]],
                        SignedPauliOp.identity(stab_array.nqubits),
                    )
                    for i in range(len(bookkeeping_inverse))
                ]
            )

            # Check that the reconstructed stabilizer array is the same as the original
            self.assertTrue(
                np.array_equal(stab_array_reconstructed.array, stab_array.array)
            )

    def test_stabarray_reduced_bookkeeping(self):
        """
        Test the bookkeeping of the stabilizer array with bookkeeping when reducing.
        """
        for stab_array in self.code_check_stab_arrays:
            # Make it into a reducible stabilizer array
            stab_array_reducible = merge_stabarrays(
                (
                    stab_array,
                    # Add a stabilizer that is the product of the first two stabilizers
                    StabArray.from_signed_pauli_ops([stab_array[0] * stab_array[1]]),
                )
            )

            # Reduce the stabilizer array with bookkeeping
            stab_array_reduced, bookkeeping = reduce_stabarray_with_bookkeeping(
                stab_array_reducible
            )

            # Check that the bookkeeping and the reduced stabilizer array are consistent
            self.assertLess(stab_array_reduced.nstabs, stab_array_reducible.nstabs)

            # Reconstruct the reduced stabilizer array from the bookkeeping and the
            # original stabilizer array
            stab_array_reduced_reconstructed = StabArray.from_signed_pauli_ops(
                [
                    reduce(
                        lambda x, y: x * y,
                        stab_array_reducible[bookkeeping[i]],
                        SignedPauliOp.identity(stab_array.nqubits),
                    )
                    for i in range(len(bookkeeping))
                ]
            )
            # Remove the trivial stabilizers since it's a reducible stabilizer array
            stab_array_reduced_reconstructed = StabArray.from_signed_pauli_ops(
                [
                    p_op
                    for p_op in stab_array_reduced_reconstructed
                    if not p_op.is_trivial
                ]
            )

            # Check that the reduced stabilizer arrays are the same
            self.assertTrue(
                np.array_equal(
                    stab_array_reduced.array, stab_array_reduced_reconstructed.array
                )
            )

            # Inverse the reduction
            bookkeeping_inverse = invert_bookkeeping_matrix(bookkeeping)
            stab_array_reconstructed = StabArray.from_signed_pauli_ops(
                [
                    reduce(
                        lambda x, y: x * y,
                        stab_array_reduced[bookkeeping_inverse[i]],
                        SignedPauliOp.identity(stab_array.nqubits),
                    )
                    for i in range(len(bookkeeping_inverse))
                ]
            )

            # Check that the reconstructed stabilizer array is the same as the original
            # reducible stabilizer array
            self.assertTrue(
                np.array_equal(
                    stab_array_reconstructed.array, stab_array_reducible.array
                )
            )


if __name__ == "__main__":
    unittest.main()
