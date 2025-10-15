"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest

from loom.eka.utilities import apply_to_nested


class TestUtilities(unittest.TestCase):
    """
    Test for Eka utilities.
    """

    def test_apply_to_nested(self):
        """
        Tests the apply_to_nested() utility function.
        """

        def mult_2(nr, mult=2):
            return mult * nr

        def check_nested(in_list):
            if isinstance(in_list, list):
                _ = [check_nested(el) for el in in_list]
            elif in_list != 4:
                raise ValueError("apply_to_nested() didn't apply function correctly.")

        test_case1 = [2, [2, [2, [2, 2]]]]
        test_case2 = [[[[2]]], [[2]], [2], 2]
        test_case3 = [4, [4, [4, [2, 4]]]]

        check_nested(apply_to_nested(test_case1, mult_2, 2))
        check_nested(apply_to_nested(test_case2, mult_2))
        with self.assertRaises(ValueError):
            check_nested(test_case3)
