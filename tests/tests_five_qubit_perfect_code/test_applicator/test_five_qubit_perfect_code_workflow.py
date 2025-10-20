"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest

from loom.eka import Eka, Lattice
from loom.interpreter import interpret_eka
from loom.eka.operations.code_operation import (
    ResetAllDataQubits,
    MeasureLogicalX,
    MeasureBlockSyndromes,
)
from loom_five_qubit_perfect_code.code_factory import FiveQubitPerfectCode


class TestFiveQubitPerfectCodeWorkflows(unittest.TestCase):
    """
    Test the correct workflow generation for the FiveQubitPerfectCode class.
    """

    def setUp(self):
        self.lattice = Lattice.poly_2d(n=5, anc=4)

    def test_measure_syndrome_and_logical(self):
        """Test that the measurement of syndromes and logical operators is performed
        without warnings or errors.
        """
        block = FiveQubitPerfectCode.create(
            lattice=self.lattice, unique_label="q1", position=(5, 0)
        )

        operations = [
            ResetAllDataQubits(block.unique_label),
            MeasureBlockSyndromes(block.unique_label, n_cycles=2),
            MeasureLogicalX(block.unique_label),
        ]

        final_step = interpret_eka(Eka(self.lattice, [block], operations), True)
        final_block = final_step.get_block(block.unique_label)

        # Obtain the observable cbits
        cbits = final_step.logical_observables[0].measurements

        # Expected cbits
        data_indices = final_block.logical_x_operators[0].data_qubits
        expected_cbits = [(f"c_{d}", 0) for d in data_indices]

        # Check that the cbits are as expected
        # (sort the lists since the order of the cbits is not guaranteed)
        self.assertEqual(sorted(cbits), sorted(expected_cbits))


if __name__ == "__main__":
    unittest.main()
