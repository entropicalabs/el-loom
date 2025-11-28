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
