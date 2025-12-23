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

import itertools

import pytest

from loom.eka import (
    Eka,
    Lattice,
    PauliOperator,
    Circuit,
    Channel,
    Block,
    Stabilizer,
)
from loom.eka.utilities import SingleQubitPauliEigenstate
from loom.eka.operations import (
    MeasureBlockSyndromes,
    MeasureLogicalX,
    MeasureLogicalY,
    MeasureLogicalZ,
    LogicalZ,
    ResetAllDataQubits,
)
from loom.interpreter.applicator import (
    reset_all_data_qubits,
    reset_all_ancilla_qubits,
    measurelogicalpauli,
)
from loom.interpreter import (
    InterpretationStep,
    Syndrome,
    LogicalObservable,
    interpret_eka,
)
from loom.interpreter.applicator import BaseApplicator, CodeApplicator


def return_custom_block(
    selected_block: Block,
    new_stabilizers: tuple[Stabilizer, ...] = None,
    new_logical_x_operators: tuple[PauliOperator, ...] = None,
    new_logical_z_operators: tuple[PauliOperator, ...] = None,
    new_unique_label: str = None,
) -> Block:
    """
    Returns a custom block with the specified stabilizers and/or logical operators.
    """
    input_block = Block(
        stabilizers=(
            new_stabilizers if new_stabilizers else selected_block.stabilizers
        ),
        logical_x_operators=(
            new_logical_x_operators
            if new_logical_x_operators
            else selected_block.logical_x_operators
        ),
        logical_z_operators=(
            new_logical_z_operators
            if new_logical_z_operators
            else selected_block.logical_z_operators
        ),
        unique_label=(
            new_unique_label if new_unique_label else selected_block.unique_label
        ),
        skip_validation=selected_block.skip_validation,
    )
    return input_block


class TestApplicator:
    """Tests for the Applicator classes."""

    def test_applicator_return_method(self, n_rsc_block_factory, empty_eka):
        """
        Test that a BaseApplicator raises the right error when giving an unsupported
        operation.
        """
        # Should raise the right type of error.
        applicator = BaseApplicator(empty_eka)
        rsc_block = n_rsc_block_factory(1)[0]

        with pytest.raises(NotImplementedError) as cm:
            applicator.apply(
                InterpretationStep.create(()),
                MeasureBlockSyndromes(rsc_block.unique_label),
                same_timeslice=False,
                debug_mode=True,
            )
        err_str = "Operation MeasureBlockSyndromes is not supported by BaseApplicator"

        assert err_str in str(cm.value)

    def test_applicator_not_mapped(self, mocker, empty_eka):
        """
        Test that the applicator raises a NotImplementedError if the operation is not
        included in the supported operations.
        """
        # Mock ReadObservable Operation
        rand_op = mocker.Mock(name="RandomOperation")
        rand_op.__class__.__name__ = "RandomOperation"

        # Using BaseApplicator should raise a NotImplementedError.
        base_applicator = BaseApplicator(empty_eka)
        with pytest.raises(NotImplementedError) as cm:
            base_applicator.apply(
                InterpretationStep.create(()),
                rand_op,
                same_timeslice=False,
                debug_mode=True,
            )
        err_str = "Operation RandomOperation is not supported by BaseApplicator"
        assert err_str in str(cm.value)

        # Using CodeApplicator should raise a NotImplementedError.
        code_applicator = CodeApplicator(empty_eka)
        with pytest.raises(NotImplementedError) as cm:
            code_applicator.apply(
                InterpretationStep.create(()),
                rand_op,
                same_timeslice=False,
                debug_mode=True,
            )
        err_str = "Operation RandomOperation is not supported by CodeApplicator"
        assert err_str in str(cm.value)

    # pylint: disable=too-many-locals
    def test_applicator_measurelogical_xyz(self, n_rsc_block_factory):
        """Test that the applicator creates the correct circuit, syndromes and
        logical observable for a MeasureLogicalX or MeasureLogicalZ operation.
        MeasureLogicalY is currently not supported. This is done for a standard Rotated
        Surface Code Block and for another one with displaced logical operators."""
        rsc_block = n_rsc_block_factory(1)[0].rename("q1")
        logical_operators = {
            "q1": {
                "X": (
                    rsc_block.logical_x_operators[0],
                    rsc_block.logical_z_operators[0],
                ),
                "Z": (
                    rsc_block.logical_x_operators[0],
                    rsc_block.logical_z_operators[0],
                ),
            },
            "q2": {
                "X": (
                    PauliOperator("X" * 3, [(0, 2, 0), (1, 2, 0), (2, 2, 0)]),
                    rsc_block.logical_z_operators[0],
                ),
                "Z": (
                    rsc_block.logical_x_operators[0],
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
            for q in rsc_block.data_qubits
        ]

        hadamard_layer = [
            Circuit("H", channels=Channel(label=f"{q}", type="quantum"))
            for q in rsc_block.data_qubits
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
            rsc_block = n_rsc_block_factory(1)[0].rename(name)
            # bypass immutability to simply create different blocks for testing
            object.__setattr__(
                rsc_block, "logical_x_operators", [logical_operators[name][basis][0]]
            )
            object.__setattr__(
                rsc_block, "logical_z_operators", [logical_operators[name][basis][1]]
            )

            base_step = InterpretationStep.create(
                [rsc_block],
            )
            output_step = measurelogicalpauli(
                base_step, measurement_op, same_timeslice=False, debug_mode=False
            )

            expected_circuit = Circuit(
                f"Measure logical {basis} of {rsc_block.unique_label}",
                circuit=circuit_seq,
            )

            # The Circuit has the right number of timesteps.
            assert len(output_step.intermediate_circuit_sequence[0]) == 1
            assert (
                output_step.intermediate_circuit_sequence[0][0].circuit
                == expected_circuit.circuit
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
            assert output_step.syndromes == expected_syndromes

            expected_observable = LogicalObservable(
                label=f"{name}_{basis}_0",
                measurements=[(f"c_{qubit}", 0) for qubit in measured_log.data_qubits],
            )
            assert output_step.logical_observables[0] == expected_observable

            # Check that `measured_single_qubit_stabilizers` are updated correctly
            new_block = output_step.get_block(rsc_block.unique_label)
            expected_measudred_single_qubit_stabs = {
                new_block.uuid: {
                    Stabilizer(
                        pauli=basis,
                        data_qubits=(q,),
                    )
                    for q in rsc_block.data_qubits
                }
            }
            assert (
                output_step.measured_single_qubit_stabilizers
                == expected_measudred_single_qubit_stabs
            )

            ## Check for wrong input
            y_op = MeasureLogicalY(rsc_block.unique_label)
            err_msg = "Logical measurement in Y basis is not supported"
            with pytest.raises(ValueError) as cm:
                _ = measurelogicalpauli(base_step, y_op, False, False)
            assert str(cm.value) == err_msg

            wrong_op = LogicalZ(rsc_block.unique_label)
            err_msg = f"Operation {wrong_op.__class__.__name__} not supported"
            with pytest.raises(ValueError) as cm:
                _ = measurelogicalpauli(base_step, wrong_op, False, False)
            assert str(cm.value) == err_msg

    def test_logical_reset(self, n_rsc_block_factory):
        """Test that the applicator correctly applies the logical reset operation."""
        rsc_block = n_rsc_block_factory(1)[0]
        rsc_qubit_channels = {
            qub: Channel(label=str(qub)) for qub in rsc_block.data_qubits
        }

        # Check the reset for all possible states
        for state in SingleQubitPauliEigenstate:
            # Create the Eka object with only the logical operation
            logical_op = ResetAllDataQubits(rsc_block.unique_label, state=state)
            # Create the base step with the block history and then interpret the
            # operation
            base_step = InterpretationStep.create(
                [rsc_block],
            )
            output_step = reset_all_data_qubits(
                base_step, logical_op, same_timeslice=False, debug_mode=True
            )
            # Obtain the output circuit
            output_circ = output_step.intermediate_circuit_sequence[0][0]

            # Check that the block's uuid is changed but the block is equivalent
            new_block = output_step.get_block(rsc_block.unique_label)
            assert new_block == rsc_block
            assert new_block.uuid != rsc_block.uuid

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
            assert output_circ == expected_circ
            # Check that all the reset operations are done in the same timestep
            assert output_circ.duration == 1

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
            assert output_step.syndromes == expected_syndromes

            # Check that `reset_single_qubit_stabilizers` are updated correctly
            expected_reset_single_qubit_stabs = {
                new_block.uuid: {
                    Stabilizer(
                        pauli=state.pauli_basis,
                        data_qubits=(q,),
                    )
                    for q in rsc_block.data_qubits
                }
            }
            assert (
                output_step.reset_single_qubit_stabilizers
                == expected_reset_single_qubit_stabs
            )

    def test_ancilla_reset(self, n_rsc_block_factory):
        """Test that the applicator correctly applies the ancilla reset operation."""
        rsc_block = n_rsc_block_factory(1)[0]
        rsc_ancilla_channels = {
            qub: Channel("quantum", str(qub)) for qub in rsc_block.ancilla_qubits
        }
        # Create the Eka object with only the logical operation
        ancilla_reset = ResetAllDataQubits(rsc_block.unique_label)
        # Create the base step with the block history and then interpret the
        # operation
        base_step = InterpretationStep.create(
            [rsc_block],
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
        assert output_circ == expected_circ
        # Check that all the reset operations are done in the same timestep
        assert output_circ.duration == 1

    def test_classical_channel_naming(self, n_rsc_block_factory):
        """Verify that the classical channels created from syndrome measurement field
        are consistent for all operations
        """
        lattice = Lattice.square_2d((10, 20))
        rsc_block = n_rsc_block_factory(1)[0]

        meas_block_log = MeasureLogicalZ(rsc_block.unique_label)
        input_eka = Eka(lattice, blocks=[rsc_block], operations=[meas_block_log])

        output_step = interpret_eka(input_eka)
        classical_channel_labels = [
            channel.label
            for channel in output_step.final_circuit.channels
            if channel.is_classical()
        ]

        # Check that the classical channels are named correctly
        for syndrome in output_step.syndromes:
            syndrome_meas_labels = [
                f"{meas[0]}_{meas[1]}" for meas in syndrome.measurements
            ]
            for syndrome_meas_label in syndrome_meas_labels:
                assert syndrome_meas_label in classical_channel_labels
