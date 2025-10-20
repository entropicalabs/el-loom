"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest
from itertools import product

import numpy as np

from loom.eka.utilities import (
    paulixz_to_char,
    paulichar_to_xz,
    paulichar_to_xz_npfunc,
    paulixz_to_char_npfunc,
    g,
    g_npfunc,
    paulis_anti_commute,
)


class TestPauliStrFunctions(unittest.TestCase):
    """
    Test for Eka utilities.
    """

    def setUp(self):
        self.pauli_char_dict = {"_": (0, 0), "X": (1, 0), "Z": (0, 1), "Y": (1, 1)}

        self.anti_comm_val = {
            "_": lambda p: 0,
            "X": lambda p: 0 if p in ["_", "X"] else 1,
            "Z": lambda p: 0 if p in ["_", "Z"] else 1,
            "Y": lambda p: 0 if p in ["_", "Y"] else 1,
        }

    def test_pauli_char_functions(self):
        """
        Test correctness of pauli char functions.
        """
        for p, tup in self.pauli_char_dict.items():
            self.assertTrue(p == paulixz_to_char(*tup))
            self.assertTrue(tup == paulichar_to_xz(p))

        # test lower case which is also valid
        self.assertTrue(
            paulichar_to_xz("i") == paulichar_to_xz("_") == paulichar_to_xz("I")
        )
        self.assertTrue(paulichar_to_xz("x") == paulichar_to_xz("X"))
        self.assertTrue(paulichar_to_xz("z") == paulichar_to_xz("Z"))
        self.assertTrue(paulichar_to_xz("y") == paulichar_to_xz("Y"))

        # test invalid inputs
        for char in ["P", "1"]:
            self.assertRaises(ValueError, paulichar_to_xz, char)
        for tup in [(0, 2), (2, 0), (-1, 0), (5, 5)]:
            self.assertRaises(ValueError, paulixz_to_char, *tup)

    def test_pauli_char_functions_vec(self):
        """
        Test vectorized versions of the pauli char functions.
        """
        array_size = 10
        # initialize a randon initial array
        init_pauli_array = np.random.choice(
            np.array(list(self.pauli_char_dict.keys())), size=array_size
        )

        # use vectorized version of function
        x, z = paulichar_to_xz_npfunc(init_pauli_array)

        # test that the output is correct for every element
        for i, (px, pz) in enumerate(zip(x, z, strict=True)):
            self.assertTrue(init_pauli_array[i], paulixz_to_char(px, pz))

        # use the reverse vectorized method and check that we get the initial array
        final_pauli_arr = paulixz_to_char_npfunc(x, z)
        self.assertTrue(np.all(final_pauli_arr == init_pauli_array))

    def test_g_function(self):
        """
        Tests the g function by explicitly expressing it as it's explicitly
        described in https://arxiv.org/abs/quant-ph/0406196.
        """
        # dict keys ar x1 and z1
        # the value is the method acting on x2,z2
        g_result = {
            (0, 0): lambda x2, z2: 0,
            (0, 1): lambda x2, z2: x2 * (1 - 2 * z2),
            (1, 0): lambda x2, z2: z2 * (2 * x2 - 1),
            (1, 1): lambda x2, z2: z2 - x2,
        }

        four_bit_pairs = list(product([0, 1], repeat=4))

        for x1, z1, x2, z2 in four_bit_pairs:
            g_func = g_result[x1, z1]

            test_g_value = g_func(x2, z2)
            self.assertEqual(test_g_value, g(x1, z1, x2, z2))

    def test_g_function_vectorized_version(self):
        """
        Tests the g function by explicitly expressing it as it's explicitly
        described in https://arxiv.org/abs/quant-ph/0406196.
        """
        # dict keys ar x1 and z1
        # the value is the method acting on x2,z2
        g_result = {
            (0, 0): lambda x2, z2: 0,
            (0, 1): lambda x2, z2: x2 * (1 - 2 * z2),
            (1, 0): lambda x2, z2: z2 * (2 * x2 - 1),
            (1, 1): lambda x2, z2: z2 - x2,
        }
        array_size = 100
        x1_array = np.random.randint(0, 2, size=array_size)
        x2_array = np.random.randint(0, 2, size=array_size)
        z1_array = np.random.randint(0, 2, size=array_size)
        z2_array = np.random.randint(0, 2, size=array_size)

        g_array = g_npfunc(x1_array, z1_array, x2_array, z2_array)

        # verify that the result is correct
        for i, g_val in enumerate(g_array):
            x1 = x1_array[i]
            x2 = x2_array[i]
            z1 = z1_array[i]
            z2 = z2_array[i]
            self.assertEqual(g_val, g_result[x1, z1](x2, z2))

    def test_anti_commute(self):
        """
        Test all pauli anti_commutation combinations.
        """
        #
        for p1, p2 in product(["_", "X", "Z", "Y"], repeat=2):
            x1, z1 = paulichar_to_xz(p1)
            x2, z2 = paulichar_to_xz(p2)

            self.assertEqual(
                self.anti_comm_val[p1](p2), paulis_anti_commute(x1, z1, x2, z2)
            )


if __name__ == "__main__":
    unittest.main()
