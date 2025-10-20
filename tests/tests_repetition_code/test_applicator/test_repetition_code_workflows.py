"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=duplicate-code
import unittest

from loom.eka import Eka, Lattice, Stabilizer
from loom.eka.operations.code_operation import (
    Grow,
    Shrink,
    MeasureLogicalZ,
    MeasureBlockSyndromes,
)
from loom.eka.utilities import Direction
from loom.interpreter import interpret_eka

from loom_repetition_code.code_factory import RepetitionCode


class TestRepetitionCodeWorkflows(unittest.TestCase):
    """
    Test the correct workflow generation for the RepetitionCode class.
    """

    def setUp(self):
        self.lattice = Lattice.linear()

    def test_grow_shrink_measure_caterpillar(self):
        """Test that the accumulation of cbits is correct when growing, shrinking and
        measuring a repetition code. This is the so-called caterpillar experiment.
        We also check that the output stabilizers are as expected.
        """
        block = RepetitionCode.create(
            d=3, check_type="Z", lattice=self.lattice, unique_label="q1", position=(5,)
        )

        direction = Direction.RIGHT
        length = 2

        operations = [
            Grow(block.unique_label, direction, length),
            MeasureBlockSyndromes(block.unique_label),
            Shrink(block.unique_label, direction.opposite(), length),
            MeasureLogicalZ(block.unique_label),
        ]

        final_step = interpret_eka(Eka(self.lattice, [block], operations), True)
        final_block = final_step.get_block(block.unique_label)

        # Obtain the observable cbits
        cbits = final_step.logical_observables[0].measurements

        # Expected cbits
        expected_cbits = [
            # The stabilizer cbits
            # - Z5Z6
            ("c_(5, 1)", 0),
            # - Z6Z7
            ("c_(6, 1)", 0),
            # And the data qubit(s) of the logical operator
            ("c_(7, 0)", 0),
        ]

        # Check that the cbits are as expected
        # (sort the lists since the order of the cbits is not guaranteed)
        self.assertEqual(sorted(cbits), sorted(expected_cbits))

        # Verify that the output stabilizers of the caterpillar experiment is as
        # expected
        output_block_stabilizers = [
            # Z7Z8
            Stabilizer("ZZ", ((7, 0), (8, 0)), ancilla_qubits=((7, 1),)),
            # Z8Z9
            Stabilizer("ZZ", ((8, 0), (9, 0)), ancilla_qubits=((8, 1),)),
        ]
        self.assertEqual(set(final_block.stabilizers), set(output_block_stabilizers))


if __name__ == "__main__":
    unittest.main()
