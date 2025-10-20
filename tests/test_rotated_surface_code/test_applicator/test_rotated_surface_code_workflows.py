"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=duplicate-code
import unittest

from loom.eka import Eka, Lattice
from loom.eka.utilities import Direction
from loom.eka.operations import (
    Grow,
    Shrink,
    MeasureLogicalX,
    MeasureLogicalZ,
    ResetAllDataQubits,
    MeasureBlockSyndromes,
)
from loom.interpreter import interpret_eka, InterpretationStep, Syndrome, Detector
from loom.interpreter.applicator import measureblocksyndromes, reset_all_data_qubits


from loom_rotated_surface_code.code_factory import RotatedSurfaceCode
from loom_rotated_surface_code.operations import AuxCNOT
from loom_rotated_surface_code.applicator.move_corners import move_corners


# pylint: disable=duplicate-code
class TestRotatedSurfaceCodeWorkflows(unittest.TestCase):
    """
    Class for Tests of the Rotated Surface Code workflows.
    """

    def setUp(self):
        self.lattice = Lattice.square_2d((10, 10))

    def test_measure_syndrome_and_logical(self):
        """Test that the measurement of syndromes and logical operators is performed
        without warnings or errors.
        """
        block = RotatedSurfaceCode.create(
            dx=3, dz=3, lattice=self.lattice, unique_label="q1", position=(5, 0)
        )

        operations = [
            ResetAllDataQubits(block.unique_label),
            MeasureBlockSyndromes(block.unique_label, n_cycles=2),
            MeasureLogicalX(block.unique_label),
        ]

        final_step = interpret_eka(
            eka=Eka(lattice=self.lattice, blocks=[block], operations=operations),
            debug_mode=True,
        )
        final_block = final_step.get_block(block.unique_label)

        # Obtain the observable cbits
        cbits = final_step.logical_observables[0].measurements

        # Expected cbits
        data_indices = final_block.logical_x_operators[0].data_qubits
        expected_cbits = [(f"c_{d}", 0) for d in data_indices]

        # Check that the cbits are as expected
        # (sort the lists since the order of the cbits is not guaranteed)
        self.assertEqual(sorted(cbits), sorted(expected_cbits))

    def test_grow_shrink_measure_caterpillar(self):
        """Tests that the accumulation of cbits is correct when growing, shrinking and
        measuring a rotated surface code. This is the so-called caterpillar experiment.
        We also check that the output stabilizers are as expected.
        """
        block = RotatedSurfaceCode.create(
            dx=3, dz=3, position=(1, 2), lattice=self.lattice, unique_label="q1"
        )

        direction = Direction.RIGHT
        length = 3

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
            # - Bulk ZZZZ
            ("c_(2, 3, 1)", 0),
            ("c_(3, 4, 1)", 0),
            ("c_(4, 3, 1)", 0),
            # - Bottom stabs
            ("c_(2, 5, 1)", 0),
            ("c_(4, 5, 1)", 0),
            # - Top stab
            ("c_(3, 2, 1)", 0),
            # And the data qubit(s) of the logical operator
            ("c_(4, 2, 0)", 0),
            ("c_(4, 3, 0)", 0),
            ("c_(4, 4, 0)", 0),
        ]
        # NOTE: the data qubits measured during the shrink are not used for the
        # logical Z

        # Check that the cbits are as expected
        # (sort the lists since the order of the cbits is not guaranteed)
        self.assertEqual(sorted(cbits), sorted(expected_cbits))

        # Verify that the output stabilizers of the caterpillar experiment is as
        # expected
        expected_final_block = RotatedSurfaceCode.create(
            dx=3,
            dz=3,
            position=(4, 2),
            lattice=self.lattice,
            unique_label="q1",
            weight_2_stab_is_first_row=False,
        )
        self.assertEqual(final_block, expected_final_block)

    def test_auxcnot_z_basis_initialisation(self):
        """
        Tests that the AuxCNOT can be interpreted after initialisation of the blocks
        in a specific state. This covers a previously reported issue #432
        """
        control = RotatedSurfaceCode.create(
            dx=3, dz=3, position=(1, 2), lattice=self.lattice, unique_label="C"
        )
        target = RotatedSurfaceCode.create(
            dx=3, dz=3, position=(5, 6), lattice=self.lattice, unique_label="T"
        )

        operations = [
            (  # Intialise all blocks
                ResetAllDataQubits(control.unique_label, "0"),  # control in 0 state
                ResetAllDataQubits(target.unique_label, "0"),  # target in 0 state
            ),
            (
                MeasureBlockSyndromes(control.unique_label),
                MeasureBlockSyndromes(target.unique_label),
            ),
            (
                AuxCNOT(
                    input_blocks_name=[
                        control.unique_label,
                        target.unique_label,
                    ]
                ),
            ),
            (
                MeasureBlockSyndromes(control.unique_label),
                MeasureBlockSyndromes(target.unique_label),
            ),
            (
                MeasureLogicalZ(control.unique_label),
                MeasureLogicalZ(target.unique_label),
            ),
        ]

        final_step = interpret_eka(
            Eka(self.lattice, [control, target], operations), True
        )
        final_blocks = [
            final_step.get_block(block.unique_label) for block in [control, target]
        ]
        # Check that the final blocks are the same as the initial ones
        self.assertEqual(
            sorted(final_blocks, key=lambda p: p.unique_label),
            sorted([control, target], key=lambda p: p.unique_label),
        )
        logical_measurements = [
            obs.measurements for obs in final_step.logical_observables
        ]
        expected_control_cbits = (
            # Measurement of the logical Z of the control block (top-left)
            ("c_(1, 2, 0)", 0),
            ("c_(1, 3, 0)", 0),
            ("c_(1, 4, 0)", 0),
        )
        expected_target_cbits = (
            # Measurement of the logical Z of the target block (bottom-right)
            ("c_(5, 6, 0)", 0),
            ("c_(5, 7, 0)", 0),
            ("c_(5, 8, 0)", 0),
            # Split of the merged aux_target block
            ("c_(2, 3, 1)", 3),
            ("c_(3, 4, 1)", 3),
            ("c_(3, 2, 1)", 3),
            ("c_(2, 5, 1)", 3),
            ("c_(5, 2, 1)", 2),
            ("c_(4, 5, 1)", 2),
            ("c_(4, 3, 1)", 2),
            ("c_(5, 4, 1)", 2),
            # Shrink of the merged aux_target block
            ("c_(5, 2, 0)", 0),
            ("c_(5, 3, 0)", 0),
            ("c_(5, 4, 0)", 0),
            ("c_(5, 5, 0)", 0),
        )
        for measurements, expected_measurements in zip(
            logical_measurements,
            (
                expected_control_cbits,
                expected_target_cbits,
            ),
            strict=True,
        ):
            self.assertEqual(sorted(measurements), sorted(expected_measurements))

    def test_auxcnot_x_basis_initialisation(self):
        """
        Tests that the AuxCNOT can be interpreted after initialisation of the blocks
        in a specific state. This covers a previously reported issue #432
        """
        control = RotatedSurfaceCode.create(
            dx=3, dz=3, position=(1, 2), lattice=self.lattice, unique_label="C"
        )
        target = RotatedSurfaceCode.create(
            dx=3, dz=3, position=(5, 6), lattice=self.lattice, unique_label="T"
        )

        operations = [
            (  # Intialise all blocks
                ResetAllDataQubits(control.unique_label, "+"),  # control in 0 state
                ResetAllDataQubits(target.unique_label, "+"),  # target in 0 state
            ),
            (
                MeasureBlockSyndromes(control.unique_label),
                MeasureBlockSyndromes(target.unique_label),
            ),
            (
                AuxCNOT(
                    input_blocks_name=[
                        control.unique_label,
                        target.unique_label,
                    ]
                ),
            ),
            (
                MeasureBlockSyndromes(control.unique_label),
                MeasureBlockSyndromes(target.unique_label),
            ),
            (
                MeasureLogicalX(control.unique_label),
                MeasureLogicalX(target.unique_label),
            ),
        ]

        final_step = interpret_eka(
            Eka(self.lattice, [control, target], operations), True
        )

        final_blocks = [
            final_step.get_block(block.unique_label) for block in [control, target]
        ]
        # Check that the final blocks are the same as the initial ones
        self.assertEqual(
            sorted(final_blocks, key=lambda p: p.unique_label),
            sorted([control, target], key=lambda p: p.unique_label),
        )
        logical_measurements = [
            obs.measurements for obs in final_step.logical_observables
        ]
        expected_control_cbits = (
            # Measurement of the logical X of the control block (top-left)
            ("c_(1, 2, 0)", 0),
            ("c_(2, 2, 0)", 0),
            ("c_(3, 2, 0)", 0),
            # Split of the grown control block
            ("c_(4, 2, 0)", 0),
            # x_operator_update corresponding to ConditionalLogicalZ conditioned on
            # joint observable from merge
            ("c_(8, 4, 1)", 4),
            ("c_(6, 4, 1)", 4),
            ("c_(7, 3, 1)", 4),
            ("c_(5, 3, 1)", 4),
            ("c_(6, 6, 1)", 0),
            ("c_(7, 5, 1)", 0),
            ("c_(5, 5, 1)", 0),
            ("c_(8, 6, 1)", 0),
        )
        expected_target_cbits = (
            # Measurement of the logical X of the target block (bottom-right)
            ("c_(5, 6, 0)", 0),
            ("c_(6, 6, 0)", 0),
            ("c_(7, 6, 0)", 0),
            # x_operator_update corresponding to ConditionalLogicalZ conditioned on
            # joint observable from merge
            ("c_(8, 4, 1)", 6),
            ("c_(6, 4, 1)", 6),
            ("c_(7, 3, 1)", 6),
            ("c_(5, 3, 1)", 6),
            ("c_(6, 6, 1)", 2),
            ("c_(7, 5, 1)", 2),
            ("c_(5, 5, 1)", 2),
            ("c_(8, 6, 1)", 2),
            # syndrome cbits from Shrink operation
            ("c_(8, 4, 1)", 4),
            ("c_(6, 4, 1)", 4),
            ("c_(7, 3, 1)", 4),
            ("c_(5, 3, 1)", 4),
            ("c_(6, 6, 1)", 0),
            ("c_(7, 5, 1)", 0),
            ("c_(5, 5, 1)", 0),
            ("c_(8, 6, 1)", 0),
        )
        for measurements, expected_measurements in zip(
            logical_measurements,
            (
                expected_control_cbits,
                expected_target_cbits,
            ),
            strict=True,
        ):
            self.assertEqual(sorted(measurements), sorted(expected_measurements))

    def test_multiple_move_corners(self):
        """Tests that the move_corners function can be used multiple times in a row
        before finally measuring its syndromes an the detectors are still generated
        correctly.
        """
        block = RotatedSurfaceCode.create(
            dx=5, dz=7, position=(1, 2), lattice=self.lattice, unique_label="q1"
        )
        base_step = InterpretationStep(block_history=((block,),))
        step0 = reset_all_data_qubits(
            interpretation_step=base_step,
            operation=ResetAllDataQubits(block.unique_label, "0"),
            same_timeslice=False,
            debug_mode=False,
        )
        block_after_reset = step0.get_block(block.unique_label)

        step1 = measureblocksyndromes(
            interpretation_step=step0,
            operation=MeasureBlockSyndromes(block.unique_label, 2),
            same_timeslice=False,
            debug_mode=False,
        )
        block_after_mps = step1.get_block(block.unique_label)

        step2 = move_corners(
            interpretation_step=step1,
            block=block_after_mps,
            corner_args=(
                ((1, 2, 0), Direction.RIGHT, 2),
                ((5, 2, 0), Direction.BOTTOM, 2),
                ((5, 8, 0), Direction.TOP, 3),
            ),
            same_timeslice=False,
            debug_mode=True,
        )
        block_after_move_corners = step2.get_block(block.unique_label)

        new_step = measureblocksyndromes(
            interpretation_step=step2,
            operation=MeasureBlockSyndromes(block.unique_label, 3),
            same_timeslice=False,
            debug_mode=False,
        )

        conserved_stab_uuid = [
            stab.uuid
            for stab in block_after_mps.stabilizers
            if stab.ancilla_qubits[0] == (2, 3, 1)
        ][0]

        conserved_stab_syndromes = [
            Syndrome(
                stabilizer=conserved_stab_uuid,
                measurements=(),
                block=block_after_reset.uuid,
                round=0,
            ),
            Syndrome(
                stabilizer=conserved_stab_uuid,
                measurements=(("c_(2, 3, 1)", 0),),
                block=block_after_mps.uuid,
                round=1,
            ),
            Syndrome(
                stabilizer=conserved_stab_uuid,
                measurements=(("c_(2, 3, 1)", 1),),
                block=block_after_mps.uuid,
                round=2,
            ),
            Syndrome(
                stabilizer=conserved_stab_uuid,
                measurements=(("c_(2, 3, 1)", 2),),
                block=block_after_move_corners.uuid,
                round=0,
            ),
            Syndrome(
                stabilizer=conserved_stab_uuid,
                measurements=(("c_(2, 3, 1)", 3),),
                block=block_after_move_corners.uuid,
                round=1,
            ),
            Syndrome(
                stabilizer=conserved_stab_uuid,
                measurements=(("c_(2, 3, 1)", 4),),
                block=block_after_move_corners.uuid,
                round=2,
            ),
            Syndrome(
                stabilizer=conserved_stab_uuid,
                measurements=(("c_(2, 3, 1)", 5),),
                block=block_after_move_corners.uuid,
                round=3,
            ),
        ]

        modified_stab_syndromes = [
            Syndrome(
                stabilizer=(
                    init_id := [
                        stab.uuid
                        for stab in block_after_reset.stabilizers
                        if stab.ancilla_qubits[0] == (5, 8, 1)
                    ][0]
                ),
                measurements=(),
                block=block_after_reset.uuid,
                round=0,
            ),
            Syndrome(
                stabilizer=init_id,
                measurements=(("c_(5, 8, 1)", 0),),
                block=block_after_mps.uuid,
                round=1,
            ),
            Syndrome(
                stabilizer=init_id,
                measurements=(("c_(5, 8, 1)", 1),),
                block=block_after_mps.uuid,
                round=2,
            ),
            Syndrome(
                stabilizer=(
                    moved_id := [
                        stab.uuid
                        for stab in block_after_move_corners.stabilizers
                        if stab.ancilla_qubits[0] == (5, 8, 1)
                    ][0]
                ),
                measurements=(("c_(5, 8, 1)", 2),),
                block=block_after_move_corners.uuid,
                corrections=(("c_(5, 8, 0)", 0),),
                round=0,
            ),
            Syndrome(
                stabilizer=moved_id,
                measurements=(("c_(5, 8, 1)", 3),),
                block=block_after_move_corners.uuid,
                round=1,
            ),
            Syndrome(
                stabilizer=moved_id,
                measurements=(("c_(5, 8, 1)", 4),),
                block=block_after_move_corners.uuid,
                round=2,
            ),
            Syndrome(
                stabilizer=moved_id,
                measurements=(("c_(5, 8, 1)", 5),),
                block=block_after_move_corners.uuid,
                round=3,
            ),
        ]

        expected_conserved_detectors = [
            Detector(
                syndromes=tuple(
                    [conserved_stab_syndromes[i], conserved_stab_syndromes[i + 1]]
                )
            )
            for i in range(0, len(conserved_stab_syndromes) - 1)
        ]
        expected_modified_detectors = [
            Detector(
                syndromes=tuple(
                    [modified_stab_syndromes[i], modified_stab_syndromes[i + 1]]
                )
            )
            for i in range(0, len(modified_stab_syndromes) - 1)
        ]
        expected_detectors = expected_conserved_detectors + expected_modified_detectors

        for detector in expected_detectors:
            self.assertIn(detector, new_step.detectors)


if __name__ == "__main__":
    unittest.main()
