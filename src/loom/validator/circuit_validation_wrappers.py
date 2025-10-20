"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from ..eka import Circuit, Block, Stabilizer

from .circuit_validation import is_circuit_valid
from .debug_dataclass import DebugData
from .utilities import logical_states_to_check, logical_state_transformations_to_check


def is_syndrome_extraction_circuit_valid(
    circuit: Circuit,
    input_block: Block | tuple[Block, ...],
    measurement_to_input_stabilizer_map: dict[str, Stabilizer],
) -> DebugData:
    """Tests if the syndrome extraction circuit is a valid one.
    The function checks that the circuit:
    - does not alter the check operators
    - acts on the logical level as identity
    - measures indeed the specified input check operators with specific measurement
    operations

    Parameters
    ----------
    circuit : Circuit
        The syndrome extraction circuit.
    input_block : Block | tuple[Block, ...]
        The input Block object(s).
    measurement_to_input_stabilizer_map : dict[str, Stabilizer]
        Dictionary matching the classical channel name of a measurement operation with a
        stabilizer in the input code.

    Returns
    -------
    DebugData
        The result of the checks.
    """

    logical_state_transformations = [
        (state, (state,))
        for state in logical_states_to_check(input_block.n_logical_qubits)
    ]
    # Check if the circuit is valid
    return is_circuit_valid(
        circuit=circuit,
        input_block=input_block,
        output_block=input_block,
        output_stabilizers_parity={},
        output_stabilizers_with_any_value=[],
        logical_state_transformations_with_parity={},
        logical_state_transformations=logical_state_transformations,
        measurement_to_input_stabilizer_map=measurement_to_input_stabilizer_map,
    )


def is_logical_operation_circuit_valid(
    circuit: Circuit,
    input_block: Block | tuple[Block, ...],
    x_operators_sparse_pauli_map: list[str],
    z_operators_sparse_pauli_map: list[str],
) -> DebugData:
    """Checks if the logical operation circuit is a valid one.
    The function checks that the circuit:
    - does not alter the check operators
    - acts on the logical level in a way that is consistent with the logical operation
    defined by the x and z maps

    For example, for a CNOT gate from qubit 0 to qubit 1 we know that:
    - X0 -> X0X1
    - X1 -> X1
    - Z0 -> Z0
    - Z1 -> Z0Z1

    and thus the input pauli map arguments will be:
    - `x_operators_sparse_pauli_map = ["X0X1", "X1"]`
    - `z_operators_sparse_pauli_map = ["Z0", "Z0Z1"]`

    Parameters
    ----------
    circuit : Circuit
        The logical operation circuit.
    input_block : Block | tuple[Block, ...]
        The input Block object(s).
    x_operators_sparse_pauli_map : list[str]
        The list of sparse Pauli strings describing how each X operator is transformed.
        No sign is needed and the order matches the transformation of the logical
        operators X0, X1, ...
    z_operators_sparse_pauli_map : list[str]
        The list of sparse Pauli strings describing how each Z operator is transformed.
        No sign is needed and the order matches the transformation of the logical
        operators Z0, Z1, ...

    Returns
    -------
    DebugData
        The result of the checks.
    """
    # Get the logical state transformations to check
    logical_state_transformations = logical_state_transformations_to_check(
        x_operators_sparse_pauli_map, z_operators_sparse_pauli_map
    )
    return is_circuit_valid(
        circuit=circuit,
        input_block=input_block,
        output_block=input_block,
        output_stabilizers_parity={},
        output_stabilizers_with_any_value=[],
        logical_state_transformations_with_parity={},
        logical_state_transformations=logical_state_transformations,
        measurement_to_input_stabilizer_map={},
    )
