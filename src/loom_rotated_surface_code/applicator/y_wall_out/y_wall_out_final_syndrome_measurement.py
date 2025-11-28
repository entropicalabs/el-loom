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
from loom.eka import Circuit
from loom.interpreter import InterpretationStep

from .y_wall_out_utilities import (
    get_final_block_syndrome_measurement_cnots_circuit,
)
from ..move_block import (
    generate_syndrome_measurement_circuit_and_cbits,
    generate_and_append_block_syndromes_and_detectors,
)
from ...code_factory import RotatedSurfaceCode


def y_wall_out_final_qec_rounds(
    interpretation_step: InterpretationStep,
    block: RotatedSurfaceCode,
    same_timeslice: bool,
    debug_mode: bool,  # pylint: disable=unused-argument
) -> InterpretationStep:
    """
    Perform the final syndrome measurement rounds for the Y wall out operation.
    This operation performs d-2 rounds of syndrome measurement on the block in such
    a way that the final circuit is fault-tolerant.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        The interpretation step.
    block : RotatedSurfaceCode
        The rotated surface code block to perform the final syndrome measurement rounds
        on.
    same_timeslice : bool
        Whether to append the final circuit in the same timeslice as the previous one.
    debug_mode : bool
        Whether to enable debug mode. (Unused since no new Blocks are created here)

    Returns
    -------
    InterpretationStep
        The updated interpretation step.
    """

    # Begin composite operation / multiple rounds of measurement
    init_circ_len = len(interpretation_step.intermediate_circuit_sequence)

    distance = min(block.size)
    n_total_rounds = distance - 2

    # Run d-2 rounds of syndrome measurement
    for _ in range(n_total_rounds):
        # A) CIRCUIT GENERATION
        # A.1) Generate ancilla initialization circuit
        final_block_syndrome_measurement_reset_circuit = Circuit(
            name="Initialization of syndrome measurement ancilla",
            circuit=[
                [
                    Circuit(
                        f"reset_{'0' if stab.pauli_type == 'Z' else '+'}",
                        channels=[
                            interpretation_step.get_channel_MUT(stab.ancilla_qubits[0])
                        ],
                    )
                    for stab in block.stabilizers
                ]
            ],
        )

        # A.2) Generate CNOT circuit
        final_block_syndrome_measurement_cnot_circuit = (
            get_final_block_syndrome_measurement_cnots_circuit(
                block, interpretation_step
            )
        )

        # A.3) Generate measurement circuit and classical bits
        final_block_syndrome_measurement_measure_circuit, final_block_meas_cbits = (
            generate_syndrome_measurement_circuit_and_cbits(
                interpretation_step,
                block,
            )
        )

        # A.4) Assemble and append circuit
        interpretation_step.append_circuit_MUT(
            Circuit(
                name="one round of final syndrome measurement",
                circuit=Circuit.construct_padded_circuit_time_sequence(
                    (
                        (final_block_syndrome_measurement_reset_circuit,),
                        (final_block_syndrome_measurement_cnot_circuit,),
                        (final_block_syndrome_measurement_measure_circuit,),
                    )
                ),
            ),
            same_timeslice=False,
        )

        # B) SYNDROME AND DETECTOR GENERATION
        # B.1) Generate and append syndromes and detectors
        generate_and_append_block_syndromes_and_detectors(
            interpretation_step, block, final_block_meas_cbits
        )

    # C) WRAP MULTI-ROUND CIRCUIT AND APPEND
    new_len = len(interpretation_step.intermediate_circuit_sequence)
    len_op = new_len - init_circ_len
    circuit_seq = interpretation_step.pop_intermediate_circuit_MUT(len_op)
    wrapped_circuit_seq = ()
    for timeslice in circuit_seq:
        timespan = max(composite_circuit.duration for composite_circuit in timeslice)
        # Create a circuit with empty timeslices and align circuits
        template_circ = (
            tuple(composite_circuit for composite_circuit in timeslice),
        ) + ((),) * (timespan - 1)
        wrapped_circuit_seq += template_circ
    wrapped_circuit = Circuit(
        name=(
            f"measure syndromes of block {block.unique_label} {n_total_rounds} times"
        ),
        circuit=wrapped_circuit_seq,
    )
    interpretation_step.append_circuit_MUT(wrapped_circuit, same_timeslice)

    return interpretation_step
