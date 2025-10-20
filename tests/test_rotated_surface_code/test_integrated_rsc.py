"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=duplicate-code
import unittest

from loom.eka import (
    Lattice,
    Eka,
)
from loom.eka.operations import MeasureBlockSyndromes, Grow, Merge, Split, Shrink
from loom.interpreter import InterpretationStep, interpret_eka
from loom_rotated_surface_code.code_factory import RotatedSurfaceCode
from loom_rotated_surface_code.operations import (
    AuxCNOT,
)  # , TransversalHadamard, MoveBlock


class TestInterpreterRSC(unittest.TestCase):
    """
    Tests the integration between this plugin and the interpreter API, interpret_eka.
    """

    def setUp(self):
        self.lattice = Lattice.square_2d((10, 20))
        # These Blocks are Rotated Surface Code Blocks
        self.rsc_code_1 = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            lattice=self.lattice,
            position=(0, 0),
            unique_label="q1",
        )
        self.rsc_code_2 = self.rsc_code_1.shift(position=(4, 0), new_label="q2")
        self.rsc_code_big = RotatedSurfaceCode.create(
            dx=6,
            dz=3,
            lattice=self.lattice,
            position=(0, 0),
            unique_label="q3",
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
            blocks=[self.rsc_code_1],
            operations=operations,
        )
        final_step = interpret_eka(eka_with_ops)
        self.assertTrue(isinstance(final_step, InterpretationStep))

    def test_run_interpreter_parallel_operations(self):
        """
        Tests the interpretation for multiple operations happening in parallel.
        """
        operations = [
            [MeasureBlockSyndromes("q1", 2), MeasureBlockSyndromes("q2", 2)],
        ]
        eka_w_ops = Eka(
            self.lattice,
            blocks=[self.rsc_code_1, self.rsc_code_2],
            operations=operations,
        )
        new_step = interpret_eka(eka_w_ops)
        circ_meas_block_1 = new_step.final_circuit.circuit[0][0]
        circ_meas_block_2 = new_step.final_circuit.circuit[0][1]
        # Check that the right circuits are generated
        self.assertEqual(circ_meas_block_1.name, "measure q1 syndromes 2 time(s)")
        self.assertEqual(circ_meas_block_2.name, "measure q2 syndromes 2 time(s)")
        # Check that both operations are indeed done in parallel
        self.assertEqual(len(new_step.final_circuit.circuit), 16)

    def test_run_interpreter_parallel_operations_different_lengths(self):
        """
        Tests the interpretation for multiple operations happening in parallel but
        they have different lengths.
        """
        operations = [
            [
                MeasureBlockSyndromes("q2", 2),
                MeasureBlockSyndromes("q1", 1),
            ],
        ]
        eka_w_ops = Eka(
            self.lattice,
            blocks=[self.rsc_code_1, self.rsc_code_2],
            operations=operations,
        )
        new_step = interpret_eka(eka_w_ops)
        circ_meas_block_2 = new_step.final_circuit.circuit[0][0]
        circ_meas_block_1 = new_step.final_circuit.circuit[0][1]
        # Check that the right circuits are generated
        self.assertEqual(circ_meas_block_2.name, "measure q2 syndromes 2 time(s)")
        self.assertEqual(circ_meas_block_1.name, "measure q1 syndromes 1 time(s)")
        # Check that both operations are indeed done in parallel
        self.assertEqual(len(new_step.final_circuit.circuit), 16)

    def test_run_interpreter_with_rsc_extended_operations(self):
        """
        Tests the interpretation for operations that are specific to the Rotated Surface
        Code.
        """
        # Test Grow operation
        operations = [
            Grow(input_block_name="q1", direction="right", length=2),
            MeasureBlockSyndromes(input_block_name="q1", n_cycles=2),
        ]
        eka_with_ops = Eka(
            self.lattice,
            blocks=[self.rsc_code_1],
            operations=operations,
        )
        final_step = interpret_eka(eka_with_ops)
        self.assertTrue(isinstance(final_step, InterpretationStep))

        # Test Merge operation
        operations = [
            Merge(
                input_blocks_name=["q1", "q2"],
                output_block_name="merged_q",
                orientation="horizontal",
            ),
        ]
        eka_with_ops = Eka(
            self.lattice,
            blocks=[self.rsc_code_1, self.rsc_code_2],
            operations=operations,
        )
        final_step = interpret_eka(eka_with_ops)
        self.assertTrue(isinstance(final_step, InterpretationStep))

        # Test Split operation
        operations = [
            MeasureBlockSyndromes(input_block_name="q3", n_cycles=2),
            Split(
                input_block_name="q3",
                output_blocks_name=["q1_split", "q2_split"],
                orientation="vertical",
                split_position=3,
            ),
        ]
        eka_with_ops = Eka(
            self.lattice,
            blocks=[self.rsc_code_big],
            operations=operations,
        )
        final_step = interpret_eka(eka_with_ops)
        self.assertTrue(isinstance(final_step, InterpretationStep))

        # Test Shrink operation
        operations = [
            MeasureBlockSyndromes(input_block_name="q3", n_cycles=2),
            Shrink(
                input_block_name="q3",
                direction="left",
                length=2,
            ),
        ]
        eka_with_ops = Eka(
            self.lattice,
            blocks=[self.rsc_code_big],
            operations=operations,
        )
        final_step = interpret_eka(eka_with_ops)
        self.assertTrue(isinstance(final_step, InterpretationStep))

        # Test AuxCNOT operation
        # pylint: disable=duplicate-code
        block_t = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            position=(4, 4),
            lattice=self.lattice,
            unique_label="t",
        )
        aux_cnot = AuxCNOT(["q1", "t"])
        auxcnot_eka = Eka(
            self.lattice,
            blocks=[self.rsc_code_1, block_t],
            operations=[aux_cnot],
        )
        final_step = interpret_eka(auxcnot_eka)
        self.assertTrue(isinstance(final_step, InterpretationStep))


if __name__ == "__main__":
    unittest.main()
