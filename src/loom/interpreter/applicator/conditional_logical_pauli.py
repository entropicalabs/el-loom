"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from loom.eka.operations.code_operation import (
    ConditionalLogicalX,
    ConditionalLogicalY,
    ConditionalLogicalZ,
)
from ..interpretation_step import InterpretationStep


# pylint: disable=unused-argument, duplicate-code
def conditional_logical_pauli(
    interpretation_step: InterpretationStep,
    operation: ConditionalLogicalX | ConditionalLogicalY | ConditionalLogicalZ,
    same_timeslice: bool,
    debug_mode: bool,
) -> InterpretationStep:
    """
    Apply a conditional logical X, Y or Z operator.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        The interpretation step to which the operation should be applied.
    operation : ConditionalLogicalX | ConditionalLogicalY | ConditionalLogicalZ
        The operation to be applied, can either be a conditional logical X, Y or Z
        operation.
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

    block = interpretation_step.get_block(operation.input_block_name)
    logical_qubit = operation.logical_qubit

    # Check if the logical qubit exists
    if logical_qubit >= block.n_logical_qubits:
        raise ValueError(
            f"Logical qubit {logical_qubit} does not exist in block "
            f"{block.unique_label}"
        )

    # Conditionally applying LogicalZ (LogicalX) is the same as adding that condition to
    # logical_x_operator_updates (logical_z_operator_updates).
    # Conditionally applying LogicalY is the same as adding that condition to both
    # logical_x_operator_updates and logical_z_operator_updates.

    logical_update = interpretation_step.logical_measurements.get(operation.condition)
    if logical_update is None:
        raise ValueError(
            f"Logical Measurement {operation.condition} has not yet been made"
        )
    match operation.__class__.__name__:
        case "ConditionalLogicalX":
            interpretation_step.update_logical_operator_updates_MUT(
                "Z",
                block.logical_z_operators[logical_qubit].uuid,
                logical_update,
                inherit_updates=False,
            )
            return interpretation_step
        case "ConditionalLogicalZ":
            interpretation_step.update_logical_operator_updates_MUT(
                "X",
                block.logical_x_operators[logical_qubit].uuid,
                logical_update,
                inherit_updates=False,
            )
        case "ConditionalLogicalY":
            interpretation_step.update_logical_operator_updates_MUT(
                "X",
                block.logical_x_operators[logical_qubit].uuid,
                logical_update,
                inherit_updates=False,
            )
            interpretation_step.update_logical_operator_updates_MUT(
                "Z",
                block.logical_z_operators[logical_qubit].uuid,
                logical_update,
                inherit_updates=False,
            )
        case _:
            raise ValueError(f"Operation {operation.__class__.__name__} not supported")

    return interpretation_step
