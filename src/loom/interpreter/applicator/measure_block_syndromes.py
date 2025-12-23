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

import logging

from loom.eka import Circuit
from loom.eka.operations import MeasureBlockSyndromes

from .generate_syndromes import generate_syndromes
from .generate_detectors import generate_detectors
from ..interpretation_step import InterpretationStep


logging.basicConfig(format="%(name)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def measureblocksyndromes(
    interpretation_step: InterpretationStep,
    operation: MeasureBlockSyndromes,
    same_timeslice: bool,
    debug_mode: bool,  # pylint: disable=unused-argument
) -> InterpretationStep:
    """
    Measure the syndromes of all stabilizers in a block.

    The algorithm is the following:
    - A.) Begin MeasureBlockSyndromes composite operation session

    - B.) Get stabilizers and syndrome circuit templates from the block
        - B.1) Get the stabilizers from the block
        - B.2) Get the stabilizer circuit templates from the block 
        
    - C.) Resolve circuits with actual channels (quantum and classical)
        - C.1) Find the channels for the qubits (create them if they don't exist) and \
        keep track of these in the right order
        - C.2) Get the classical bit labels from the syndrome circuit

    - D.) For each cycle:
        - D.1) Create classical channels and measurement records
        - D.2) Clone syndrome circuits and remap channels
        - D.3) Weave circuits together into a single time slice
            - NOTE: We currently assume that the circuits are constructed in order, 
              this is the responsibility of the user
        - D.4) Append woven circuit to interpretation step
        - D.5) Generate new syndromes for the stabilizers
        - D.6) Create new detectors for the syndromes
        - D.7) Update interpretation step with syndromes and detectors

    - E.) End MeasureBlockSyndromes composite operation session and append circuit

    Parameters
    ----------
    interpretation_step : InterpretationStep
        Interpretation step containing the block from which the 
        syndrome should be measured.
    operation : MeasureBlockSyndromes
        Syndrome measurement operation description.
    same_timeslice : bool
        Flag indicating whether the operation is part of the same timestep as the
        previous operation.
    debug_mode : bool
        Flag indicating whether the interpretation should be done in debug mode.
        Activating debug mode will enable commutation validation for Block.

    Returns
    -------
    InterpretationStep
        Interpretation step after the syndrome measurement operation.
    """

    # A.) Begin MeasureBlockSyndromes composite operation session
    interpretation_step.begin_composite_operation_session_MUT(
        same_timeslice=same_timeslice,
        circuit_name=(
            f"measure {operation.input_block_name} syndromes "
            f"{operation.n_cycles} time(s)"
        ),
    )

    # B.) Get stabilizers and syndrome circuit templates from the block
    # B.1) Get the stabilizers from the block
    block = interpretation_step.get_block(operation.input_block_name)
    stabilizers = block.stabilizers

    # B.2) Get the stabilizer circuit templates from the block
    syndrome_circuit_uuids = [
        block.stabilizer_to_circuit[stabilizer.uuid] for stabilizer in stabilizers
    ]
    syndrome_circuits_templates = [
        syndrome_circuit
        for id in syndrome_circuit_uuids
        for syndrome_circuit in block.syndrome_circuits
        if syndrome_circuit.uuid == id
    ]

    # C.) Resolve circuits with actual channels (quantum and classical)
    # C.1) Find the channels for the qubits (create them if they don't exist)
    #       and keep track of these in the right order
    data_channels = [
        [
            interpretation_step.get_channel_MUT(q, "quantum")
            for q in map(str, stab.data_qubits)
        ]
        for stab in stabilizers
    ]
    ancilla_channels = [
        [
            interpretation_step.get_channel_MUT(q, "quantum")
            for q in map(str, stab.ancilla_qubits)
        ]
        for stab in stabilizers
    ]
    # C.2) Get the classical bit labels from the syndrome circuit
    cbit_labels = [
        [str(q) for q in stabilizer.ancilla_qubits] for stabilizer in stabilizers
    ]

    # D.) For each cycle
    for idx in range(operation.n_cycles):
        # D.1) Create classical channels and measurement records
        cbit_channels, measurements = [], []
        for each_cbit_label in cbit_labels:
            cbit = interpretation_step.get_new_cbit_MUT("c_" + each_cbit_label[0])
            cbit_channels.append(
                interpretation_step.get_channel_MUT(
                    f"{cbit[0]}_{str(cbit[1])}", "classical"
                )
            )
            measurements.append(cbit)
        measurements = tuple((m,) for m in measurements)

        # D.2) Clone syndrome circuits and remap channels
        mapped_syndrome_circuits = [
            syndrome_circuit.circuit.clone(
                data_channels[i] + ancilla_channels[i] + [cbit_channels[i]]
            )
            for i, syndrome_circuit in enumerate(syndrome_circuits_templates)
        ]

        # D.3) Weave circuits together into a single time slice
        if not all(
            len(each_syndrome_circuit.circuit)
            == len(mapped_syndrome_circuits[0].circuit)
            for each_syndrome_circuit in mapped_syndrome_circuits
        ):
            raise ValueError("All syndrome circuits must be of the same length.")

        woven_circuit_seq = []
        for i in range(len(mapped_syndrome_circuits[0].circuit)):
            time_slice = [
                gate
                for circuit in mapped_syndrome_circuits
                for gate in circuit.circuit[i]
            ]
            woven_circuit_seq.append(time_slice)

        # D.4) Append woven circuit to interpretation step
        interpretation_step.append_circuit_MUT(
            Circuit(
                name=f"measure {block.unique_label} syndromes - cycle {idx}",
                circuit=woven_circuit_seq,
            ),
            same_timeslice=False,
        )

        # D.5) Generate new syndromes for the stabilizers
        new_syndromes = generate_syndromes(
            interpretation_step,
            stabilizers,
            block,
            measurements,
        )

        # D.6) Create new detectors for the syndromes
        new_detectors = generate_detectors(interpretation_step, new_syndromes)

        # D.7) Update interpretation step with syndromes and detectors
        interpretation_step.append_syndromes_MUT(new_syndromes)
        interpretation_step.append_detectors_MUT(new_detectors)

    # E.) End MeasureBlockSyndromes composite operation session and append circuit
    woven_circuit = interpretation_step.end_composite_operation_session_MUT()
    interpretation_step.append_circuit_MUT(woven_circuit, same_timeslice)

    # For debugging purposes, unroll and log the final circuit
    unrolled_woven_circuit = Circuit(
        name=woven_circuit.name, circuit=Circuit.unroll(woven_circuit)
    )
    log.debug(unrolled_woven_circuit.detailed_str())

    return interpretation_step
