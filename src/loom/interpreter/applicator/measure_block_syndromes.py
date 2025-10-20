"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import logging

from loom.eka import Circuit
from loom.eka.operations import MeasureBlockSyndromes

from .generate_syndromes import generate_syndromes
from .generate_detectors import generate_detectors
from ..interpretation_step import InterpretationStep


logging.basicConfig(format="%(name)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def measureblocksyndromes(  # pylint: disable=too-many-locals
    interpretation_step: InterpretationStep,
    operation: MeasureBlockSyndromes,
    same_timeslice: bool,
    debug_mode: bool,  # pylint: disable=unused-argument
) -> InterpretationStep:
    """
    Measure the syndromes of all stabilizers in a block.

    The algorithm is the following:
    
    - A.) For each stabilizer in the block, get the corresponding circuit (template)
    
        - A.1) Get the stabilizers from the block
        - A.2) Get the stabilizer circuit templates from the block
        
    - B.) Resolve the circuit with actual channels:
    
        - B.1) Get the qubit labels from the stabilizer
        - B.2) Get the classical bit labels from the syndrome circuit
        - B.3) Repeat for n_cycle:
        
            - B.3.1) Find the classical channels (create them if they don't exist) and \
            create cbits
            - B.3.2) Clone the syndrome circuit and remap the channels
        
    - C.) Weave circuits together: NOTE we currently assume that the circuits are \
    constructed in order, this is the responsibility of the user.
    
        - C.1) Check the assumption that all circuits are the same length
        - C.2) Add gates one by one to the intermediate circuit
        
    - D.) Generate new syndromes for the stabilizers involved
    
    - E.) Create new detectors for the new syndromes
    
    - F.) Update the interpretation_step with the new syndromes and detectors
    
    - G.) Update the interpretation_step with the new circuit

    Parameters
    ----------
    interpretation_step : InterpretationStep
        Interpretation step containing the block from which the syndrome should be measured.
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

    new_syndromes = tuple()
    woven_circuit_seq = tuple()

    # A) - For each stabilizer, get the corresponding circuit (template)
    #   A.1) - Get the syndrome_circuits uuids from the block
    block = interpretation_step.get_block(operation.input_block_name)
    stabilizers = block.stabilizers
    syndrome_circuit_uuids = [
        block.stabilizer_to_circuit[stabilizer.uuid] for stabilizer in stabilizers
    ]
    #   A.2) - Get the syndrome circuits templates
    syndrome_circuits_templates = [
        syndrome_circuit
        for id in syndrome_circuit_uuids
        for syndrome_circuit in block.syndrome_circuits
        if syndrome_circuit.uuid == id
    ]

    # B) - Resolve the circuit with actual channels:
    #   B.1) - Find the channels for the qubits (create them if they don't exist)
    #        keep track of these in the right order
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

    #   B.2) - Get the classical bit labels from the syndrome circuit
    cbit_labels = [
        [str(q) for q in stabilizer.ancilla_qubits] for stabilizer in stabilizers
    ]  # Maybe test for empty lists ???

    # Repeat for n_cycle:
    # TODO: Use cycle to create the cbits? pylint: disable=fixme
    for _ in range(operation.n_cycles):
        #   B.3) - Find the classical channels (create them if they don't exist) and create cbits
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

        #   B.4) - Clone the syndrome circuit and remap the channels
        mapped_syndrome_circuits = [
            syndrome_circuit.circuit.clone(
                data_channels[i] + ancilla_channels[i] + [cbit_channels[i]]
            )
            for i, syndrome_circuit in enumerate(syndrome_circuits_templates)
        ]  # Needs to be updated in case of several cbit channels per round

        # C) - Weave circuits together: NOTE we currently assume that the circuits are
        # constructed in order, this is the responsibility of the user.
        #    C.1) - Check the assumption that all circuits are the same length
        if not all(
            len(each_syndrome_circuit.circuit)
            == len(mapped_syndrome_circuits[0].circuit)
            for each_syndrome_circuit in mapped_syndrome_circuits
        ):
            raise ValueError("All syndrome circuits must be of the same length.")

        #  C.2) - Add gates one by one to the intermediate circuit
        for i in range(len(mapped_syndrome_circuits[0].circuit)):
            time_slice = tuple(
                gate
                for circuit in mapped_syndrome_circuits
                for gate in circuit.circuit[i]
            )
            woven_circuit_seq += (time_slice,)

        # D) - Generate new syndromes for the stabilizers involved
        new_syndromes = generate_syndromes(
            interpretation_step,
            stabilizers,
            block,
            measurements,
        )
        # E) - Create new detectors for the new syndromes
        new_detectors = generate_detectors(interpretation_step, new_syndromes)

        # F) - Update the interpretation_step with the new syndromes and detectors
        interpretation_step.append_syndromes_MUT(new_syndromes)
        interpretation_step.append_detectors_MUT(new_detectors)

    # Create the woven circuit for all cycles
    woven_circuit = Circuit(
        name=f"measure {block.unique_label} syndromes {operation.n_cycles} time(s)",
        circuit=woven_circuit_seq,
    )

    log.debug(woven_circuit.detailed_str())

    # G) - Update the interpretation_step with the new circuit
    interpretation_step.append_circuit_MUT(woven_circuit, same_timeslice)

    return interpretation_step
