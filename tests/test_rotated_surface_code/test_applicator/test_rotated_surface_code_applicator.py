"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest
import unittest.mock
import itertools
from copy import deepcopy

from loom.eka import Circuit, Channel, Lattice, Eka, PauliOperator
from loom.eka.utilities import SingleQubitPauliEigenstate
from loom.interpreter import (
    InterpretationStep,
    interpret_eka,
    Syndrome,
    LogicalObservable,
)
from loom.interpreter.applicator import (
    logical_pauli,
    reset_all_data_qubits,
    reset_all_ancilla_qubits,
    measurelogicalpauli,
    BaseApplicator,
    CodeApplicator,
)
from loom.eka.operations import (
    MeasureBlockSyndromes,
    MeasureLogicalX,
    MeasureLogicalY,
    MeasureLogicalZ,
    LogicalX,
    LogicalZ,
    LogicalY,
    ResetAllDataQubits,
)

# from loom_rotated_surface_code.operations import TransversalHadamard
from loom_rotated_surface_code.code_factory import RotatedSurfaceCode
from loom_rotated_surface_code.applicator import RotatedSurfaceCodeApplicator


class TestRotatedSurfaceCodeApplicator(unittest.TestCase):
    """
    Test the functionalities of the RotatedSurfaceCodeApplicator class.
    """

    def setUp(self):
        self.base_step = InterpretationStep()

        self.square_2d_lattice = Lattice.square_2d((10, 20))
        self.eka_no_blocks = Eka(self.square_2d_lattice)
        self.rot_surf_code_1 = RotatedSurfaceCode.create(
            dx=3, dz=3, lattice=self.square_2d_lattice, unique_label="q1"
        )
        self.meas_block_op = MeasureBlockSyndromes(self.rot_surf_code_1.unique_label)

    def test_applicator_return_method(self):
        """
        Test that a BaseApplicator raises the right error when giving an unsupported
        operation.
        """
        # Should raise the right type of error.
        applicator = BaseApplicator(self.eka_no_blocks)

        with self.assertRaises(NotImplementedError) as cm:
            applicator.apply(
                deepcopy(self.base_step),
                self.meas_block_op,
                same_timeslice=False,
                debug_mode=True,
            )
        err_str = "Operation MeasureBlockSyndromes is not supported by BaseApplicator"

        self.assertIn(err_str, str(cm.exception))

    def test_applicator_not_mapped(self):
        """
        Test that the applicator raises a NotImplementedError if the operation is not
        included in the supported operations.
        """
        # Mock ReadObservable Operation
        rand_op = unittest.mock.Mock(name="RandomOperation")
        rand_op.__class__.__name__ = "RandomOperation"

        # Using BaseApplicator should raise a NotImplementedError.
        base_applicator = BaseApplicator(self.eka_no_blocks)
        with self.assertRaises(NotImplementedError) as cm:
            base_applicator.apply(
                deepcopy(self.base_step), rand_op, same_timeslice=False, debug_mode=True
            )
        err_str = "Operation RandomOperation is not supported by BaseApplicator"
        self.assertIn(err_str, str(cm.exception))

        # Using CodeApplicator should raise a NotImplementedError.
        code_applicator = CodeApplicator(self.eka_no_blocks)
        with self.assertRaises(NotImplementedError) as cm:
            code_applicator.apply(
                deepcopy(self.base_step), rand_op, same_timeslice=False, debug_mode=True
            )
        err_str = "Operation RandomOperation is not supported by CodeApplicator"
        self.assertIn(err_str, str(cm.exception))

    def test_applicator_measurelogical_xyz(self):  # pylint: disable=too-many-locals
        """Test that the applicator creates the correct circuit, syndromes and
        logical observable for a MeasureLogicalX or MeasureLogicalZ operation.
        MeasureLogicalY is currently not supported. This is done for a standard Rotated
        Surface Code Block and for another one with displaced logical operators."""

        logical_operators = {
            "q1": {
                "X": (
                    self.rot_surf_code_1.logical_x_operators[0],
                    self.rot_surf_code_1.logical_z_operators[0],
                ),
                "Z": (
                    self.rot_surf_code_1.logical_x_operators[0],
                    self.rot_surf_code_1.logical_z_operators[0],
                ),
            },
            "q2": {
                "X": (
                    PauliOperator("X" * 3, [(0, 2, 0), (1, 2, 0), (2, 2, 0)]),
                    self.rot_surf_code_1.logical_z_operators[0],
                ),
                "Z": (
                    self.rot_surf_code_1.logical_x_operators[0],
                    PauliOperator("Z" * 3, [(2, 0, 0), (2, 1, 0), (2, 2, 0)]),
                ),
            },
        }

        measurement_circuit = [
            Circuit(
                "Measurement",
                channels=[
                    Channel(label=f"{q}", type="quantum"),
                    Channel(label=f"c_{q}_0", type="classical"),
                ],
            )
            for q in self.rot_surf_code_1.data_qubits
        ]

        hadamard_layer = [
            Circuit("H", channels=Channel(label=f"{q}", type="quantum"))
            for q in self.rot_surf_code_1.data_qubits
        ]

        circuit_seq_x = [hadamard_layer] + [measurement_circuit]
        circuit_seq_z = [measurement_circuit]

        properties = {
            name: {
                "X": (
                    MeasureLogicalX(name),
                    circuit_seq_x,
                    logical_operators[name]["X"][0],
                ),
                "Z": (
                    MeasureLogicalZ(name),
                    circuit_seq_z,
                    logical_operators[name]["Z"][1],
                ),
            }
            for name in ["q1", "q2"]
        }

        for basis, name in itertools.product(["X", "Z"], ["q1", "q2"]):

            measurement_op, circuit_seq, measured_log = properties[name][basis]

            rsc_block = RotatedSurfaceCode.create(
                dx=3,
                dz=3,
                lattice=self.square_2d_lattice,
                logical_x_operator=logical_operators[name][basis][0],
                logical_z_operator=logical_operators[name][basis][1],
                unique_label=name,
            )

            base_step = InterpretationStep(
                block_history=((rsc_block,),),
            )
            output_step = measurelogicalpauli(
                base_step, measurement_op, same_timeslice=False, debug_mode=False
            )

            expected_circuit = Circuit(
                f"Measure logical {basis} of {rsc_block.unique_label}",
                circuit=circuit_seq,
            )

            # The Circuit has the right number of timesteps.
            self.assertEqual(len(output_step.intermediate_circuit_sequence), 1)
            self.assertEqual(
                output_step.intermediate_circuit_sequence[0][0].circuit,
                expected_circuit.circuit,
            )

            # The output step has the right syndromes
            expected_stab_cbits = [
                tuple((f"c_{qubit}", 0) for qubit in stab.data_qubits)
                for stab in rsc_block.stabilizers
            ]
            expected_syndromes = tuple(
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=stab_cbits,  # Already a list of cbits
                    block=rsc_block.uuid,
                    round=0,
                    corrections=(),
                )
                for stab, stab_cbits in zip(
                    rsc_block.stabilizers, expected_stab_cbits, strict=True
                )
                if set(stab.pauli) == {basis}
            )
            self.assertEqual(output_step.syndromes, expected_syndromes)

            expected_observable = LogicalObservable(
                label=f"{name}_{basis}_0",
                measurements=[(f"c_{qubit}", 0) for qubit in measured_log.data_qubits],
            )
            self.assertEqual(output_step.logical_observables[0], expected_observable)

            ## Check for wrong input
            y_op = MeasureLogicalY(rsc_block.unique_label)
            err_msg = "Logical measurement in Y basis is not supported"
            with self.assertRaises(ValueError) as cm:
                _ = measurelogicalpauli(base_step, y_op, False, False)
            self.assertEqual(str(cm.exception), err_msg)

            wrong_op = LogicalZ(rsc_block.unique_label)
            err_msg = f"Operation {wrong_op.__class__.__name__} not supported"
            with self.assertRaises(ValueError) as cm:
                _ = measurelogicalpauli(base_step, wrong_op, False, False)
            self.assertEqual(str(cm.exception), err_msg)

    def test_logical_xyz(self):
        """
        Test that the applicator correctly applies the logical X, Z, Y operation.
        """
        lattice = self.square_2d_lattice
        rsc_block = self.rot_surf_code_1
        rsc_qubit_channels = {
            qub: Channel(label=str(qub)) for qub in rsc_block.data_qubits
        }

        for log_operation_class in [LogicalX, LogicalZ, LogicalY]:
            # Create the Eka object with only the logical operation
            logical_op = log_operation_class(rsc_block.unique_label)
            input_eka = Eka(lattice, blocks=[rsc_block], operations=[logical_op])
            # Create the base step with the block history and then interpret the
            # operation
            base_step = InterpretationStep(
                block_history=((rsc_block,),),
            )
            output_step = RotatedSurfaceCodeApplicator(input_eka).apply(
                base_step, logical_op, same_timeslice=False, debug_mode=True
            )
            # Obtain the output circuit
            output_circ = output_step.intermediate_circuit_sequence[0][0]

            # Create the expected circuit depending on the logical operation
            match log_operation_class.__name__:
                case "LogicalX":
                    logical_operators = [rsc_block.logical_x_operators[0]]
                case "LogicalZ":
                    logical_operators = [rsc_block.logical_z_operators[0]]
                case "LogicalY":
                    logical_operators = [
                        rsc_block.logical_x_operators[0],
                        rsc_block.logical_z_operators[0],
                    ]
            # Create the expected circuit
            expected_circ = Circuit(
                "expected_logical_circuit",
                circuit=[
                    [
                        Circuit(pauli, channels=rsc_qubit_channels[qb])
                        for qb, pauli in zip(
                            logical_op.data_qubits, logical_op.pauli, strict=True
                        )
                    ]
                    for logical_op in logical_operators
                ],
            )

            # Check that the circuits are the same
            self.assertEqual(output_circ, expected_circ)

        # Test ValueError for the function being called with an invalid operation
        invalid_operation = MeasureLogicalZ("q1")
        with self.assertRaises(ValueError) as cm:
            logical_pauli(
                base_step, invalid_operation, same_timeslice=False, debug_mode=True
            )
        self.assertIn(
            f"Operation {invalid_operation.__class__.__name__} not supported",
            str(cm.exception),
        )

        # Test ValueError for qubit not being in the block
        log_x_op_qubit_1 = LogicalX(input_block_name="q1", logical_qubit=1)
        with self.assertRaises(ValueError) as cm:
            logical_pauli(
                base_step, log_x_op_qubit_1, same_timeslice=False, debug_mode=True
            )
        self.assertIn(
            f"Logical qubit {1} does not exist in block q1", str(cm.exception)
        )

    def test_logical_reset(self):
        """Test that the applicator correctly applies the logical reset operation."""
        rsc_block = self.rot_surf_code_1
        rsc_qubit_channels = {
            qub: Channel(label=str(qub)) for qub in rsc_block.data_qubits
        }

        # Check the reset for all possible states
        for state in SingleQubitPauliEigenstate:
            # Create the Eka object with only the logical operation
            logical_op = ResetAllDataQubits(rsc_block.unique_label, state=state)
            # Create the base step with the block history and then interpret the
            # operation
            base_step = InterpretationStep(
                block_history=((rsc_block,),),
            )
            output_step = reset_all_data_qubits(
                base_step, logical_op, same_timeslice=False, debug_mode=True
            )
            # Obtain the output circuit
            output_circ = output_step.intermediate_circuit_sequence[0][0]

            # Check that the block's uuid is changed but the block is equivalent
            new_block = output_step.get_block(self.rot_surf_code_1.unique_label)
            self.assertEqual(new_block, self.rot_surf_code_1)
            self.assertNotEqual(new_block.uuid, self.rot_surf_code_1.uuid)

            # Create the expected circuit
            expected_circ = Circuit(
                "expected_logical_circuit",
                circuit=[
                    [
                        Circuit("reset_" + state, channels=rsc_qubit_channels[qb])
                        for qb in rsc_block.data_qubits
                    ]
                ],
            )

            # Check that the circuits are the same
            self.assertEqual(output_circ, expected_circ)
            # Check that all the reset operations are done in the same timestep
            self.assertEqual(output_circ.duration, 1)

            # Check that the Syndromes are correctly created
            expected_syndromes = tuple(
                Syndrome(
                    stabilizer=stab.uuid,
                    measurements=(),
                    block=new_block.uuid,  # IMPORTANT, use the new uuid
                    round=0,
                    corrections=(),
                )
                for stab in new_block.stabilizers
                if set(stab.pauli) == {state.pauli_basis}
            )
            self.assertEqual(output_step.syndromes, expected_syndromes)

    def test_ancilla_reset(self):
        """Test that the applicator correctly applies the ancilla reset operation."""
        rsc_block = self.rot_surf_code_1
        rsc_ancilla_channels = {
            qub: Channel("quantum", str(qub)) for qub in rsc_block.ancilla_qubits
        }
        # Create the Eka object with only the logical operation
        ancilla_reset = ResetAllDataQubits(rsc_block.unique_label)
        # Create the base step with the block history and then interpret the
        # operation
        base_step = InterpretationStep(
            block_history=((rsc_block,),),
        )
        output_step = reset_all_ancilla_qubits(
            base_step, ancilla_reset, same_timeslice=False, debug_mode=True
        )
        # Obtain the output circuit
        output_circ = output_step.intermediate_circuit_sequence[0][0]

        # Create the expected circuit
        expected_circ = Circuit(
            "expected_ancilla_reset_circuit",
            circuit=[
                [
                    Circuit("reset_0", channels=rsc_ancilla_channels[qb])
                    for qb in rsc_block.ancilla_qubits
                ]
            ],
        )

        # Check that the circuits are the same
        self.assertEqual(output_circ, expected_circ)
        # Check that all the reset operations are done in the same timestep
        self.assertEqual(output_circ.duration, 1)

    def test_classical_channel_naming(self):
        """Verify that the classical channels created from syndrome measurement field
        are consistent for all operations
        """
        lattice = Lattice.square_2d((10, 20))
        rsc_block = RotatedSurfaceCode.create(
            dx=3, dz=3, lattice=lattice, unique_label="q1"
        )

        # Measure for 2 cycles
        n_cycles = 3
        meas_block_syn = MeasureBlockSyndromes(
            rsc_block.unique_label, n_cycles=n_cycles
        )
        meas_block_log = MeasureLogicalZ(rsc_block.unique_label)
        input_eka = Eka(
            lattice, blocks=[rsc_block], operations=[meas_block_syn, meas_block_log]
        )

        output_step = interpret_eka(input_eka)
        classical_channel_labels = [
            channel.label
            for channel in output_step.final_circuit.channels
            if channel.type == "classical"
        ]

        # Check that the classical channels are named correctly
        for syndrome in output_step.syndromes:
            syndrome_meas_labels = [
                f"{meas[0]}_{meas[1]}" for meas in syndrome.measurements
            ]
            for syndrome_meas_label in syndrome_meas_labels:
                self.assertIn(syndrome_meas_label, classical_channel_labels)


if __name__ == "__main__":
    unittest.main()
