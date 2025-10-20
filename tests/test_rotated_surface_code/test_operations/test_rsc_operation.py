"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=duplicate-code
import unittest
from loom.eka.utilities.serialization import loads, dumps
from loom.eka.operations import Operation

from loom_rotated_surface_code.operations import AuxCNOT  # TransversalHadamard


# Note: tests for grow, shrink, merge, split, state injection are in
# loom test_operation_code.py
class TestCodeOperation(unittest.TestCase):
    """
    Test the creation of code operations for Rotated Surface Code
    """

    def test_aux_cnot(self):  # pylint: disable=protected-access
        """Test the creation of an AuxCNOT operation"""
        # pylint: disable=protected-access
        # Test the creation of an AuxCNOT operation
        aux_cnot = AuxCNOT(input_blocks_name=["q1", "q2"])
        self.assertEqual(aux_cnot.input_blocks_name, ("q1", "q2"))
        self.assertEqual(aux_cnot.__class__.__name__, "AuxCNOT")

        # Test the loads/dumps both using the right class and the abstract base class
        self.assertEqual(aux_cnot, loads(AuxCNOT, dumps(aux_cnot)))
        self.assertEqual(aux_cnot, loads(Operation, dumps(aux_cnot)))


if __name__ == "__main__":
    unittest.main()
