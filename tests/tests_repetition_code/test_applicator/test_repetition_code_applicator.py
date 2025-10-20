"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=duplicate-code
import unittest

from loom.eka import Block, Eka, Lattice, Stabilizer, PauliOperator
from loom_repetition_code.applicator import RepetitionCodeApplicator


class TestRepetitionCodeApplicator(unittest.TestCase):
    """
    Test the functionalities of the RepetitionCodeApplicator class.
    """

    def test_valid_input_block_type(self):
        """Test input an invalid block type when performing an operation"""

        # Test shrinking with invalid block (not RepetitionCode)
        invalid_block = Block(
            stabilizers=[Stabilizer("ZZ", ((0, 0), (1, 0)))],
            logical_x_operators=[PauliOperator("XX", ((0, 0), (1, 0)))],
            logical_z_operators=[PauliOperator("Z", ((0, 0),))],
            unique_label="q1",
        )
        eka = Eka(
            Lattice.square_2d((20, 20)),
            blocks=[invalid_block],
            operations=[],
        )

        err_msg_type = "All blocks must be of type RepetitionCode."
        with self.assertRaises(ValueError) as cm:
            _ = RepetitionCodeApplicator(eka)
        self.assertEqual(str(cm.exception), err_msg_type)


if __name__ == "__main__":
    unittest.main()
