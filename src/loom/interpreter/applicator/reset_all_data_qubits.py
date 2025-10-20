"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from loom.eka import Circuit
from loom.eka.operations import ResetAllDataQubits

from .generate_syndromes import generate_syndromes
from ..interpretation_step import InterpretationStep


def reset_all_data_qubits(
    interpretation_step: InterpretationStep,
    operation: ResetAllDataQubits,
    same_timeslice: bool,
    debug_mode: bool,  # pylint: disable=unused-argument
) -> InterpretationStep:
    """
    Resets all data qubits of a block to a specific SingleQubitPauliEigenstate.
    It also adds empty Syndrome objects for the stabilizers will be deterministic in
    the first round of syndrome measurement cycles dependent on the
    initialization state. This helps to put Detectors on these deterministic
    measurements when the block is measured.

    NOTE: Initializing a Y state may come with some caveats, as the implementation of
    the initialization may not be fault-tolerant for some codes. For example, in the
    case of the Rotated Surface Code, initializing a Y state may require distillation
    for it to be fault-tolerant.

    TODO: This function may need to reset the tracking of Pauli faults on the data
    qubits.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        Interpretation step containing the blocks whose data qubits need to be reset.
    operation : ResetAllDataQubits
        Reset data operation description.
    same_timeslice : bool
        Flag indicating whether the operation is part of the same timestep as the
        previous operation.
    debug_mode : bool
        Flag indicating whether the interpretation should be done in debug mode.
        Activating debug mode will enable commutation validation for Block

    Returns
    -------
    InterpretationStep
        Interpretation step after the reset data operation.
    """

    # Get the block
    block_before_reset = interpretation_step.get_block(operation.input_block_name)
    block = block_before_reset.rename(operation.input_block_name)  # Create a new uuid

    # Create a circuit that resets the data qubits to the given state
    reset_circuit = Circuit(
        name=f"reset all data qubits of block {block.unique_label} to "
        f"|{operation.state.value}>",
        circuit=[
            # Reset the data qubits on the same time step
            [
                Circuit(
                    f"reset_{operation.state.value}",
                    channels=interpretation_step.get_channel_MUT(q),
                )
                for q in block.data_qubits
            ]
        ],
    )
    relevant_stabs = [
        stab
        for stab in block.stabilizers
        if set(stab.pauli) == {operation.state.pauli_basis}
    ]

    initialization_values = [
        (
            (1,)
            if (
                # The parity of the first syndrome is 1 only if:
                # 1. The state is a -1 eigenstate of the corresponding pauli
                # 2. The number of qubits is odd
                operation.state.basis_expectation_value == -1
                and len(stab.pauli) % 2 == 1
            )
            else ()
        )
        for stab in relevant_stabs
    ]

    new_syndromes = generate_syndromes(
        interpretation_step=interpretation_step,
        stabilizers=relevant_stabs,
        block=block,
        stab_measurements=list(initialization_values),
    )

    interpretation_step.append_syndromes_MUT(new_syndromes)
    # Add the reset circuit to the interpretation step
    interpretation_step.append_circuit_MUT(reset_circuit, same_timeslice)

    # Change the block history
    interpretation_step.update_block_history_and_evolution_MUT(
        old_blocks=(block_before_reset,),
        new_blocks=(block,),
        update_evolution=False,
    )

    # Return the new interpretation step
    return interpretation_step
