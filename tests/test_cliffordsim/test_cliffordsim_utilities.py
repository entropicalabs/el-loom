"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest
import numpy as np

from loom.cliffordsim.engine import Engine
from loom.cliffordsim.operations import Hadamard, CNOT, Measurement, UpdateTableau
from loom.cliffordsim.tableau import compare_stabilizer_set
from loom.eka.utilities import stabarray_bge, StabArray

# pylint: disable=import-error, wrong-import-order
from utilities import random_list_of_gates


class TestCliffordSimUtilities(unittest.TestCase):
    """
    We test the utilities used in cliffordsim.
    """

    def test_stabilizer_bge(self):
        """Tests the stabilizer binary gaussian elimination for some
        specific cases.
        """
        # Stabilizer set: [-IZI,+IIZ,+ZZZ]
        stab_array = [
            [0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 1, 0],
        ]
        # In row echelon form : [-ZII,-IZI,+IIZ]
        expected_ref = [
            [0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 0],
        ]

        stab_array = np.array(stab_array)
        expected_ref = np.array(expected_ref)

        stab_array_ref = stabarray_bge(StabArray(stab_array))

        self.assertTrue(np.array_equal(stab_array_ref.array, expected_ref))

    def test_stabilizer_set_equivalence(self):
        """Check that the stabilizer sets of 2 equivalent circuits, even though
        different, stabilize the same states.
        """

        nqubits = 2
        seed = 5
        operation_list_1 = [Hadamard(0), CNOT(0, 1), Measurement(0)]
        operation_list_2 = [Hadamard(0), CNOT(0, 1), Measurement(1)]

        cliffordsim_engine_1 = Engine(operation_list_1, nqubits, seed)
        cliffordsim_engine_2 = Engine(operation_list_2, nqubits, seed)

        cliffordsim_engine_1.run()
        cliffordsim_engine_2.run()

        engine_rep_1 = cliffordsim_engine_1.tableau_w_scratch
        engine_rep_2 = cliffordsim_engine_2.tableau_w_scratch

        self.assertNotEqual(engine_rep_1.stabilizer_set, engine_rep_2.stabilizer_set)

        self.assertTrue(compare_stabilizer_set(engine_rep_1, engine_rep_2))

    def test_non_equivalent_stabilizers(self):
        "Test that a bell pair and the 00 state are not equal."
        nqubits = 2
        operation_list_1 = [Hadamard(0), CNOT(0, 1)]
        operation_list_2 = []

        cliffordsim_engine_1 = Engine(operation_list_1, nqubits)
        cliffordsim_engine_2 = Engine(operation_list_2, nqubits)

        cliffordsim_engine_1.run()
        cliffordsim_engine_2.run()

        engine_rep_1 = cliffordsim_engine_1.tableau_w_scratch
        engine_rep_2 = cliffordsim_engine_2.tableau_w_scratch

        self.assertFalse(compare_stabilizer_set(engine_rep_1, engine_rep_2))

    def test_run_from_state_inverted_process(self):
        """
        The goal of this test is to make sure that an arbitrary tableau can be inverted.
        The test starts by applying a random set of hermitian gates to the trivial
        tableau, and from the resulting tableau, the inverse process is then applied.
        The test is successful if at the end we are back with the trivial tableau.
        """
        nqubits = 3
        ngates = 15

        for _ in range(3):
            gate_list = random_list_of_gates(
                nqubits, ngates, include_non_hermitian=False
            )
            cat_engine = Engine(gate_list, nqubits)

            # get initial and final tableau
            init_tableau = cat_engine.tableau_w_scratch.tableau.copy()
            cat_engine.run()
            final_tableau = cat_engine.tableau_w_scratch.tableau.copy()

            # invert the process
            gate_list.append(UpdateTableau(np.array(final_tableau), True))
            gate_list_inv = gate_list[::-1]
            cat_engine = Engine(gate_list_inv, nqubits)
            # run it from the final tableau
            cat_engine.run()

            inv_process_final_tableau = cat_engine.tableau_w_scratch.tableau

            self.assertTrue(np.array_equal(inv_process_final_tableau, init_tableau))


if __name__ == "__main__":
    unittest.main()
