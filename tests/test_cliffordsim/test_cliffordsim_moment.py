"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest
import numpy as np

from loom.cliffordsim.tableau import Tableau
from loom.cliffordsim.data_store import DataStore
from loom.cliffordsim.moments.base_moment import Moment
from loom.cliffordsim.moments.instruction import (
    IdentityInstruction,
    HadamardDecorator,
    PhaseDecorator,
    CNOTDecorator,
)
from loom.cliffordsim.operations import Hadamard, Phase, CNOT


class TestCliffordSimMoment(unittest.TestCase):
    """
    We test the Moment class and its associated functionality.
    """

    def test_created_instructions(self):
        """We check that for a set of input Operations, the corresponding set
        of Instructions are correctly created."""
        parallelized_ops = (Hadamard(0), Phase(1), CNOT(2, 3))
        time_step = 0
        moment_obj = Moment(parallelized_ops, time_step)

        moment_ins = moment_obj.instruction
        moment_wrapped_ins = moment_ins.wrapped_instruction
        moment_2_wrapped_ins = moment_wrapped_ins.wrapped_instruction

        self.assertEqual(moment_obj.root_operations, parallelized_ops)

        self.assertEqual(type(moment_ins), CNOTDecorator)
        self.assertEqual(moment_ins.input_operation.operating_qubit, [2, 3])
        self.assertEqual(type(moment_wrapped_ins), PhaseDecorator)
        self.assertEqual(
            moment_wrapped_ins.input_operation.operating_qubit,
            [1],
        )
        self.assertEqual(
            type(moment_2_wrapped_ins),
            HadamardDecorator,
        )
        self.assertEqual(
            moment_2_wrapped_ins.input_operation.operating_qubit,
            [0],
        )

    def test_instruction_equality(self):
        """Ensure that the wrapped instruction object within the Moment does
        the same transformation as if the instruction was created outside of
        the Moment."""
        init_inst = IdentityInstruction()
        first_inst = HadamardDecorator(init_inst, Hadamard(0))
        second_inst = PhaseDecorator(first_inst, Phase(1))
        final_inst = CNOTDecorator(second_inst, CNOT(2, 3))

        parallelized_ops = (Hadamard(0), Phase(1), CNOT(2, 3))
        time_step = 0
        moment_obj = Moment(parallelized_ops, time_step)

        manual_test_er = Tableau(4)
        data_store = DataStore()
        manual_inst_er, _, _ = final_inst.apply(manual_test_er, data_store)

        moment_test_er = Tableau(4)
        data_store = DataStore()
        moment_inst_er, _, _ = moment_obj.transform_tab(moment_test_er, data_store)

        self.assertIsNotNone(manual_inst_er)
        self.assertIsNotNone(moment_inst_er)
        self.assertTrue(np.array_equal(manual_inst_er.tableau, moment_inst_er.tableau))


if __name__ == "__main__":
    unittest.main()
