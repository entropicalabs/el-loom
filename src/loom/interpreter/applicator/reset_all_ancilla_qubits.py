"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from loom.eka import Circuit
from loom.eka.operations import ResetAllAncillaQubits

from ..interpretation_step import InterpretationStep


def reset_all_ancilla_qubits(
    interpretation_step: InterpretationStep,
    operation: ResetAllAncillaQubits,
    same_timeslice: bool,
    debug_mode: bool,  # pylint: disable=unused-argument
) -> InterpretationStep:
    """
    Resets all ancilla qubits of a block to a specific SingleQubitPauliEigenstate.

    NOTE: Initializing a Y state may come with some caveats, as the implementation of
    the initialization may not be fault-tolerant for some codes. For example, in the
    case of the Rotated Surface Code, initializing a Y state may require distillation
    for it to be fault-tolerant.

    TODO: This function may need to reset the tracking of Pauli faults on the data
    qubits.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        Interpretation step containing the blocks whose ancilla qubits need to be reset.
    operation : ResetAllAncillaQubits
        Reset ancilla operation description.
    same_timeslice : bool
        Flag indicating whether the operation is part of the same timestep as the
        previous operation.
    debug_mode : bool
        Flag indicating whether the interpretation should be done in debug mode.
        Activating debug mode will enable commutation validation for Block.

    Returns
    -------
    InterpretationStep
        Interpretation step after the reset ancilla operation.
    """

    # Get the block
    block = interpretation_step.get_block(operation.input_block_name)

    # Create a circuit that resets the data qubits to the given state
    reset_circuit = Circuit(
        name=f"reset all ancilla qubits of block {block.unique_label} to "
        f"|{operation.state.value}>",
        circuit=[
            # Reset the data qubits on the same time step
            [
                Circuit(
                    f"reset_{operation.state.value}",
                    channels=interpretation_step.get_channel_MUT(q, "quantum"),
                )
                for q in block.ancilla_qubits
            ]
        ],
    )
    # Add the reset circuit to the interpretation step
    interpretation_step.append_circuit_MUT(reset_circuit, same_timeslice)

    # Return the new interpretation step
    return interpretation_step
