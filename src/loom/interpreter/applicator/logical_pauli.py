"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from loom.eka import Circuit
from loom.eka.operations.code_operation import (
    LogicalX,
    LogicalZ,
    LogicalY,
)

from ..interpretation_step import InterpretationStep


def logical_pauli(
    interpretation_step: InterpretationStep,
    operation: LogicalX | LogicalY | LogicalZ,
    same_timeslice: bool,
    debug_mode: bool,  # pylint: disable=unused-argument
) -> InterpretationStep:
    """
    Apply a logical X, Y or Z operator.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        The interpretation step to which the operation should be applied.
    operation : LogicalX | LogicalY | LogicalZ
        The operation to be applied, can either be a logical X, Y or Z operation.
    same_timeslice : bool
        Flag indicating whether the operation is part of the same timestep as the
        previous operation.
    debug_mode : bool
        Flag indicating whether the interpretation should be done in debug mode.
        Activating debug mode will enable commutation validation for Block.

    Returns
    -------
    InterpretationStep
        New InterpretationStep containing all modifications due to the logical pauli
        gate.
    """
    # Note: This method is used for LogicalX, LogicalY and LogicalZ operations.
    # The operation is applied to the interpretation step by appending the
    # corresponding circuit to the interpretation step.
    # No other changes are made to the interpretation step.

    block = interpretation_step.get_block(operation.input_block_name)
    logical_qubit = operation.logical_qubit

    # Check if the logical qubit exists
    if logical_qubit >= block.n_logical_qubits:
        raise ValueError(
            f"Logical qubit {logical_qubit} does not exist in block "
            f"{block.unique_label}"
        )

    # Check which logical operator to apply and get the corresponding Pauli
    # operator(s)
    match operation.__class__.__name__:
        case "LogicalX":
            logical_operators = [block.logical_x_operators[logical_qubit]]
        case "LogicalZ":
            logical_operators = [block.logical_z_operators[logical_qubit]]
        case "LogicalY":
            logical_operators = [
                block.logical_x_operators[logical_qubit],
                block.logical_z_operators[logical_qubit],
            ]
        case _:
            raise ValueError(f"Operation {operation.__class__.__name__} not supported")

    # Create the circuit
    logical_operation_circuit = Circuit(
        name=(
            f"{operation.__class__.__name__} on block {block.unique_label}, "
            f"logical qubit {logical_qubit}"
        ),
        circuit=[
            [
                Circuit(pauli, channels=[interpretation_step.get_channel_MUT(qb)])
                for qb, pauli in zip(
                    logical_operator.data_qubits, logical_operator.pauli, strict=True
                )
            ]
            for logical_operator in logical_operators
        ],
    )

    # Append the circuit
    interpretation_step.append_circuit_MUT(logical_operation_circuit, same_timeslice)

    return interpretation_step
