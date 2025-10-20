"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest

from loom.eka import Block, Eka, Lattice, Stabilizer, PauliOperator
from loom_shor_code.applicator import ShorCodeApplicator


# pylint: disable=duplicate-code
class TestShorCodeApplicator(unittest.TestCase):
    """
    Test the functionalities of the ShorCodeApplicator class.
    """

    def test_valid_input_block_type(self):
        """
        Test that an invalid block type raises an error when performing an operation.
        """

        # Test using invalid blocks (not ShorCode)
        invalid_block = Block(
            stabilizers=[Stabilizer("XX", ((0, 0), (1, 0)))],
            logical_x_operators=[PauliOperator("ZZ", ((0, 0), (1, 0)))],
            logical_z_operators=[PauliOperator("X", ((0, 0),))],
            unique_label="q1",
        )
        eka = Eka(
            Lattice.linear((5,)),
            blocks=[invalid_block],
            operations=[],
        )

        err_msg_type = "All blocks must be of type ShorCode."
        with self.assertRaises(ValueError) as cm:
            _ = ShorCodeApplicator(eka)
        self.assertEqual(str(cm.exception), err_msg_type)


if __name__ == "__main__":
    unittest.main()
