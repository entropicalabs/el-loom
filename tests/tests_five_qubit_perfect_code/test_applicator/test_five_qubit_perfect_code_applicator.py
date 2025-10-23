"""
Copyright 2024 Entropica Labs Pte Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

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
