"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest

from loom.eka import Block, Eka, Lattice, Stabilizer, PauliOperator
from loom_five_qubit_perfect_code.applicator import FiveQubitPerfectCodeApplicator


class TestFiveQubitPerfectCodeApplicator(unittest.TestCase):
    """
    Test the functionalities of the FiveQubitPerfectCodeApplicator class.
    """

    def test_valid_input_block_type(self):
        """
        Test that an error is raised when an invalid block type is input during an
        operation.
        """

        # Test using invalid block (not FiveQubitPerfectCode)
        invalid_block = Block(
            stabilizers=[Stabilizer("XX", ((0, 0), (1, 0)))],
            logical_x_operators=[PauliOperator("ZZ", ((0, 0), (1, 0)))],
            logical_z_operators=[PauliOperator("X", ((0, 0),))],
            unique_label="q1",
        )
        eka = Eka(
            Lattice.poly_2d(n=5, anc=4),
            blocks=[invalid_block],
            operations=[],
        )

        err_msg_type = "All blocks must be of type FiveQubitPerfectCode."
        with self.assertRaises(ValueError) as cm:
            _ = FiveQubitPerfectCodeApplicator(eka)
        self.assertEqual(str(cm.exception), err_msg_type)


if __name__ == "__main__":
    unittest.main()
