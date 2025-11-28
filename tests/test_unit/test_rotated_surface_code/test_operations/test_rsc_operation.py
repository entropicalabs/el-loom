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
