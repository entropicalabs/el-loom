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

from loom.eka import (
    Lattice,
    Eka,
)
from loom.eka.operations import MeasureBlockSyndromes
from loom.interpreter import InterpretationStep, interpret_eka
from loom_five_qubit_perfect_code.code_factory import FiveQubitPerfectCode


# pylint: disable=duplicate-code
class TestInterpreterFiveQubitPerfectCode(unittest.TestCase):
    """
    Tests the integration between this plugin and the interpreter API, interpret_eka.
    """

    def setUp(self):
        self.lattice = Lattice.poly_2d(n=5, anc=4)

        # Five Qubit Perfect Code Blocks
        self.five_qubit_code_1 = FiveQubitPerfectCode.create(
            lattice=self.lattice,
            position=(0, 0),
            unique_label="q1",
        )

        self.five_qubit_code_2 = FiveQubitPerfectCode.create(
            lattice=self.lattice,
            position=(20, 0),
            unique_label="q2",
        )

    def test_run_interpreter(self):
        """
        Tests where the interpreter runs without an error if we include operations.
        """
        operations = [
            MeasureBlockSyndromes(input_block_name="q1", n_cycles=2),
        ]
        eka_with_ops = Eka(
            self.lattice,
            blocks=[self.five_qubit_code_1],
            operations=operations,
        )
        final_step = interpret_eka(eka_with_ops)
        self.assertTrue(isinstance(final_step, InterpretationStep))

    def test_run_interpreter_parallel_operations(self):
        """
        Tests the interpretation for multiple operations happening in parallel.
        """
        operations = [
            [MeasureBlockSyndromes("q1", 2), MeasureBlockSyndromes("q2", 4)],
        ]
        eka_w_ops = Eka(
            self.lattice,
            blocks=[self.five_qubit_code_1, self.five_qubit_code_2],
            operations=operations,
        )
        new_step = interpret_eka(eka_w_ops)
        circ_meas_block_1 = new_step.final_circuit.circuit[0][0]
        circ_meas_block_2 = new_step.final_circuit.circuit[0][1]
        # Check that the right circuits are generated
        self.assertEqual(circ_meas_block_1.name, "measure q1 syndromes 2 time(s)")
        self.assertEqual(circ_meas_block_2.name, "measure q2 syndromes 4 time(s)")
        # Check that both operations are indeed done in parallel
        self.assertEqual(len(new_step.final_circuit.circuit), 48)


if __name__ == "__main__":
    unittest.main()
