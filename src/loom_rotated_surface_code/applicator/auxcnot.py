"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from uuid import uuid4

from loom.eka import Circuit
from loom.eka.operations import (
    Merge,
    Split,
    Grow,
    Shrink,
    MeasureBlockSyndromes,
    LogicalMeasurement,
    ConditionalLogicalZ,
)
from loom.eka.utilities import Direction, Orientation
from loom.interpreter.applicator import measureblocksyndromes, conditional_logical_pauli
from loom.interpreter.interpretation_step import InterpretationStep

from .merge import merge
from .split import split
from .grow import grow
from .shrink import shrink
from ..code_factory import RotatedSurfaceCode
from ..operations import AuxCNOT

# pylint: disable=duplicate-code


def auxcnot(
    interpretation_step: InterpretationStep,
    operation: AuxCNOT,
    same_timeslice: bool,
    debug_mode: bool,
) -> InterpretationStep:
    """
    Apply an auxiliary CNOT operation using a Grow - Shrink - Merge - Split approach.

    The algorithm is the following:
    - A) GROW CONTROL BLOCK
    - B) MEASURE SYNDROMES OF GROWN_CONTROL AND TARGET BLOCKS
    - C) SPLIT GROWN_CONTROL INTO CONTROL AND AUXILIARY BLOCKS
    - D) MEASURE SYNDROMES OF CONTROL, AUXILIARY AND TARGET BLOCKS
    - E) MERGE AUXILIARY AND TARGET BLOCKS
    - F) MEASURE SYNDROMES OF CONTROL AND MERGED_TARGET BLOCKS
    - G) APPLY CONDITIONALLOGICALZ CONDITIONED ON JOINT MEASUREMENT
    - H) SHRINK THE NEW TARGET BLOCK
    - I) WRAP THE AUXCNOT CIRCUIT IN A SINGLE CIRCUIT

    Parameters
    ----------
    interpretation_step : InterpretationStep
        Input interpretation step.
    operation : AuxCNOT
        The CNOT operation to apply.
    same_timeslice : bool
        Flag indicating whether the operation is part of the same timestep as the input.
    debug_mode : bool
        Flag indicating whether the interpretation should be done in debug mode.
        Activating debug mode will enable commutation validation for Block.

    Returns
    -------
    InterpretationStep
        The updated interpretation step after applying the auxiliary CNOT operation.
    """
    init_circ_len = len(interpretation_step.intermediate_circuit_sequence)

    c_block = interpretation_step.get_block(operation.input_blocks_name[0])
    t_block = interpretation_step.get_block(operation.input_blocks_name[1])

    auxcnot_consistency_check(c_block, t_block)

    grow_direction, shrink_direction = get_grow_shrink_directions(
        control=c_block,
        target=t_block,
    )

    # A) GROW CONTROL BLOCK
    interpretation_step = auxcnot_grow_control(
        interpretation_step=interpretation_step,
        control=c_block,
        target=t_block,
        grow_direction=grow_direction,
        same_timeslice=False,  # Prevent the same timeslice flag from being set
        debug_mode=debug_mode,
    )

    # Get the grown control block
    grown_control = interpretation_step.get_block(c_block.unique_label)
    aux_unique_label = f"{grown_control.unique_label}_aux"
    # Ensure the auxiliary block has a unique label
    if aux_unique_label in [
        block.unique_label for block in interpretation_step.block_history[-1]
    ]:
        aux_unique_label = str(uuid4())

    # B) MEASURE SYNDROMES OF GROWN_CONTROL AND TARGET BLOCKS
    for block, internal_same_timeslice in ((grown_control, False), (t_block, True)):
        interpretation_step = measureblocksyndromes(
            interpretation_step=interpretation_step,
            operation=MeasureBlockSyndromes(block.unique_label, min(block.size)),
            same_timeslice=internal_same_timeslice,
            debug_mode=debug_mode,
        )

    # C) SPLIT GROWN_CONTROL INTO CONTROL AND AUXILIARY BLOCKS
    interpretation_step = auxcnot_split_control(
        interpretation_step=interpretation_step,
        aux_unique_label=aux_unique_label,
        initial_control=c_block,
        initial_target=t_block,
        grown_control=grown_control,
        same_timeslice=False,
        debug_mode=debug_mode,
    )

    # Get the new control and auxiliary blocks
    new_control = interpretation_step.get_block(c_block.unique_label)
    aux_block = interpretation_step.get_block(aux_unique_label)

    # D) MEASURE SYNDROMES OF CONTROL, AUXILIARY AND TARGET BLOCKS
    for block, internal_same_timeslice in (
        (new_control, False),
        (aux_block, True),
        (t_block, True),
    ):
        interpretation_step = measureblocksyndromes(
            interpretation_step=interpretation_step,
            operation=MeasureBlockSyndromes(block.unique_label, 1),
            same_timeslice=internal_same_timeslice,
            debug_mode=debug_mode,
        )

    # E) MERGE AUXILIARY AND TARGET BLOCKS
    interpretation_step = auxcnot_merge_aux_target(
        interpretation_step=interpretation_step,
        aux=aux_block,
        target=t_block,
        same_timeslice=False,  # Prevent the same timeslice flag from being set
        debug_mode=debug_mode,
    )

    # Get the merged target block
    merged_target = interpretation_step.get_block(t_block.unique_label)

    # F) MEASURE SYNDROMES OF CONTROL AND MERGED_TARGET BLOCKS
    for block, internal_same_timeslice, n_cycles in (
        (new_control, True, 1),
        (new_control, False, min(new_control.size) - 1),
        (merged_target, True, min(merged_target.size) - 1),
    ):
        interpretation_step = measureblocksyndromes(
            interpretation_step=interpretation_step,
            operation=MeasureBlockSyndromes(block.unique_label, n_cycles),
            same_timeslice=internal_same_timeslice,
            debug_mode=debug_mode,
        )

    # G) APPLY CONDITIONALLOGICALZ CONDITIONED ON JOINT MEASUREMENT
    interpretation_step = auxcnot_conditional_logical_z(
        interpretation_step=interpretation_step,
        new_control=new_control,
        initial_target=t_block,
        aux_block=aux_block,
        new_target=merged_target,
        shrink_direction=shrink_direction,
        same_timeslice=True,
        debug_mode=debug_mode,
    )

    # H) SHRINK THE NEW TARGET BLOCK
    interpretation_step = auxcnot_shrink_target(
        interpretation_step=interpretation_step,
        initial_target=t_block,
        merged_target=merged_target,
        shrink_direction=shrink_direction,
        same_timeslice=False,  # Prevent the same timeslice flag from being set
        debug_mode=debug_mode,
    )

    # I) WRAP THE AUXCNOT CIRCUIT IN A SINGLE CIRCUIT
    new_len = len(interpretation_step.intermediate_circuit_sequence)
    len_auxcnot = new_len - init_circ_len
    circuit_seq = interpretation_step.pop_intermediate_circuit_MUT(len_auxcnot)

    wrapped_circuit_seq = ()
    for timeslice in circuit_seq:
        timespan = max(composite_circuit.duration for composite_circuit in timeslice)
        # Create a circuit with empty timeslices and align circuits
        template_circ = (
            tuple(composite_circuit for composite_circuit in timeslice),
        ) + ((),) * (timespan - 1)
        wrapped_circuit_seq += template_circ

    wrapped_circuit = Circuit(
        name=(f"auxcnot between {c_block.unique_label} and {t_block.unique_label}"),
        circuit=wrapped_circuit_seq,
    )
    interpretation_step.append_circuit_MUT(wrapped_circuit, same_timeslice)

    # Return the final step
    return interpretation_step


def auxcnot_consistency_check(
    c_block: RotatedSurfaceCode,
    t_block: RotatedSurfaceCode,
):
    """
    Perform multiple checks to ensure that the auxiliary CNOT operation can be applied.
    The blocks must be of the correct type, have the same size, the same boundary
    orientations and be in a specific configuration.

    The configuration is the following, the upper left corners of the two blocks must
    satisfy the following relations:

    * |t_block.upper_left_qubit[0] - c_block.upper_left_qubit[0]| = c_block.size[0]
    * |t_block.upper_left_qubit[1] - c_block.upper_left_qubit[1]| = c_block.size[1]

    Parameters
    ----------
    c_block : RotatedSurfaceCode
        Control block for the auxiliary CNOT operation.
    t_block : RotatedSurfaceCode
        Target block for the auxiliary CNOT operation.
    """

    if not isinstance(c_block, RotatedSurfaceCode) or not isinstance(
        t_block, RotatedSurfaceCode
    ):
        raise TypeError(
            "The auxcnot operation is only supported for RotatedSurfaceCode blocks. "
            f"{set(type(block) for block in [c_block, t_block])} types "
            "were given."
        )

    # NOTE: this is a sub-case, can be extended later
    if c_block.size != t_block.size:
        raise ValueError(
            f"The blocks must have the same size to perform the auxiliary CNOT "
            f"operation. The sizes of the blocks are {c_block.size}, {t_block.size}."
        )

    # Check that the blocks have the correct boundary orientations
    if c_block.x_boundary != t_block.x_boundary:
        raise ValueError(
            "The blocks must have the same boundary orientations to perform the "
            "auxiliary CNOT operation. The X boundary orientations are "
            f"{c_block.x_boundary} and {t_block.x_boundary}."
        )

    # Check that the blocks are in the correct configuration
    if (
        abs(t_block.upper_left_qubit[0] - c_block.upper_left_qubit[0])
        != c_block.size[0] + 1  # +1 because the auxiliary qubit is one row/column away
        or abs(t_block.upper_left_qubit[1] - c_block.upper_left_qubit[1])
        != c_block.size[1] + 1  # +1 because the auxiliary qubit is one row/column away
    ):
        raise ValueError(
            "The blocks are not in the correct configuration for the auxiliary CNOT "
            "operation. The upper left corners of the blocks must satisfy the "
            "following relations: \n"
            "|t_block.upper_left_qubit[0] - c_block.upper_left_qubit[0]| = "
            "c_block.size[0] + 1, |t_block.upper_left_qubit[1] - "
            "c_block.upper_left_qubit[1]| = c_block.size[1] + 1"
            f"\nGot |{t_block.upper_left_qubit[0]} - {c_block.upper_left_qubit[0]}| = "
            f"{c_block.size[0] + 1}, |{t_block.upper_left_qubit[1]} - "
            f"{c_block.upper_left_qubit[1]}| = {c_block.size[1] + 1} instead."
        )


def get_grow_shrink_directions(
    control: RotatedSurfaceCode,
    target: RotatedSurfaceCode,
) -> tuple[Direction, Direction]:
    """
    Get the grow and shrink directions for the auxiliary CNOT operation.
    The grow direction is the direction of the X operator of the control block, and
    the shrink direction is the direction normal to the X operator of the target block.

    Parameters
    ----------
    control : RotatedSurfaceCode
        Control block for the auxiliary CNOT operation.
    target : RotatedSurfaceCode
        Target block for the auxiliary CNOT operation.

    Returns
    -------
    tuple[Direction, Direction]
        The grow and shrink directions.
    """
    upper_left_qubits_vector = (
        control.upper_left_qubit[0] - target.upper_left_qubit[0],
        control.upper_left_qubit[1] - target.upper_left_qubit[1],
    )
    match upper_left_qubits_vector:
        case (x, y) if x < 0 and y < 0:
            # The control block is to the left and above the target block
            grow_direction = (
                Direction.RIGHT
                if control.x_boundary == Orientation.HORIZONTAL
                else Direction.BOTTOM
            )
            shrink_direction = (
                Direction.TOP
                if control.x_boundary == Orientation.HORIZONTAL
                else Direction.LEFT
            )
        case (x, y) if x < 0 and y > 0:  # pylint: disable=chained-comparison
            # The control block is to the left and below the target block
            grow_direction = (
                Direction.RIGHT
                if control.x_boundary == Orientation.HORIZONTAL
                else Direction.TOP
            )
            shrink_direction = (
                Direction.BOTTOM
                if control.x_boundary == Orientation.HORIZONTAL
                else Direction.LEFT
            )
        case (x, y) if x > 0 and y < 0:  # pylint: disable=chained-comparison
            # The control block is to the right and above the target block
            grow_direction = (
                Direction.LEFT
                if control.x_boundary == Orientation.HORIZONTAL
                else Direction.BOTTOM
            )
            shrink_direction = (
                Direction.TOP
                if control.x_boundary == Orientation.HORIZONTAL
                else Direction.RIGHT
            )
        case (x, y) if x > 0 and y > 0:
            # The control block is to the right and below the target block
            grow_direction = (
                Direction.LEFT
                if control.x_boundary == Orientation.HORIZONTAL
                else Direction.TOP
            )
            shrink_direction = (
                Direction.BOTTOM
                if control.x_boundary == Orientation.HORIZONTAL
                else Direction.RIGHT
            )
        case _:
            raise ValueError(
                "The control block is not in the correct position relative to the "
                "target block."
            )
    return grow_direction, shrink_direction


def auxcnot_grow_control(
    interpretation_step: InterpretationStep,
    control: RotatedSurfaceCode,
    target: RotatedSurfaceCode,
    grow_direction: Direction,
    same_timeslice: bool,
    debug_mode: bool,
) -> InterpretationStep:
    """
    Grow the control block for the auxiliary CNOT operation.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        Input interpretation step.
    control : RotatedSurfaceCode
        Control block for the auxiliary CNOT operation.
    target : RotatedSurfaceCode
        Target block for the auxiliary CNOT operation.
    grow_direction : Direction
        Direction in which to grow the control block.
    same_timeslice : bool
        Flag indicating whether the operation is part of the same timestep as the input.
    debug_mode : bool
        Flag indicating whether the interpretation should be done in debug mode.

    Returns
    -------
    InterpretationStep
        The updated interpretation step after growing the control block.
    """
    # Grow the control block in the orientation of its X operator
    grow_length = (
        target.size[0] + 1
        if control.x_boundary == Orientation.HORIZONTAL
        else target.size[1] + 1
    )

    grow_op = Grow(
        input_block_name=control.unique_label,
        direction=grow_direction,
        length=grow_length,
    )
    interpretation_step = grow(
        interpretation_step=interpretation_step,
        operation=grow_op,
        same_timeslice=same_timeslice,
        debug_mode=debug_mode,
    )
    return interpretation_step


def auxcnot_split_control(
    interpretation_step: InterpretationStep,
    aux_unique_label: str,
    initial_control: RotatedSurfaceCode,
    initial_target: RotatedSurfaceCode,
    grown_control: RotatedSurfaceCode,
    same_timeslice: bool,
    debug_mode: bool,
) -> InterpretationStep:
    """
    Split the grown control block into the control and auxiliary blocks.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        Input interpretation step.
    aux_unique_label : str
        Unique label for the auxiliary block that will be created after the split.
    initial_control : RotatedSurfaceCode
        Initial control block before it was grown.
    initial_target : RotatedSurfaceCode
        Initial target block before the auxiliary CNOT operation.
    grown_control : RotatedSurfaceCode
        Control block for the auxiliary CNOT operation that was just grown.
    same_timeslice : bool
        Flag indicating whether the operation is part of the same timestep as the input.
    debug_mode : bool
        Flag indicating whether the interpretation should be done in debug mode.

    Returns
    -------
    InterpretationStep
        The updated interpretation step after splitting the grown control block.
    """

    # Split the grown control block into the control and auxiliary blocks
    split_orientation = grown_control.x_boundary.perpendicular()

    # Ensure the order of naming is right, if the initial upper-left qubit of the
    # control block is the same as the grown control block, then the split control will
    # also inherit that qubit and the first name in the tuple.
    # The position of the split also depends on the size of the block that is located on
    # the top-left
    if initial_control.upper_left_qubit == grown_control.upper_left_qubit:
        split_blocks_name = (grown_control.unique_label, aux_unique_label)
        split_position = (
            initial_control.size[0]
            if split_orientation == Orientation.VERTICAL
            else initial_control.size[1]
        )
    else:
        split_blocks_name = (aux_unique_label, grown_control.unique_label)
        split_position = (
            initial_target.size[0]
            if split_orientation == Orientation.VERTICAL
            else initial_target.size[1]
        )

    split_op = Split(
        input_block_name=grown_control.unique_label,
        output_blocks_name=split_blocks_name,
        orientation=split_orientation,
        split_position=split_position,
    )

    interpretation_step = split(
        interpretation_step=interpretation_step,
        operation=split_op,
        same_timeslice=same_timeslice,
        debug_mode=debug_mode,
    )

    return interpretation_step


def auxcnot_merge_aux_target(
    interpretation_step: InterpretationStep,
    aux: RotatedSurfaceCode,
    target: RotatedSurfaceCode,
    same_timeslice: bool,
    debug_mode: bool,
) -> InterpretationStep:
    """
    Merge the auxiliary block with the target block.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        Input interpretation step.
    aux : RotatedSurfaceCode
        Auxiliary block for the auxiliary CNOT operation.
    target : RotatedSurfaceCode
        Target block for the auxiliary CNOT operation.
    same_timeslice : bool
        Flag indicating whether the operation is part of the same timestep as the input.
    debug_mode : bool
        Flag indicating whether the interpretation should be done in debug mode.

    Returns
    -------
    InterpretationStep
        The updated interpretation step after merging the auxiliary and target blocks.
    """
    # Merge the auxiliary block with the target block
    merge_op = Merge(
        input_blocks_name=(aux.unique_label, target.unique_label),
        output_block_name=target.unique_label,
    )

    interpretation_step = merge(
        interpretation_step=interpretation_step,
        operation=merge_op,
        same_timeslice=same_timeslice,
        debug_mode=debug_mode,
    )

    return interpretation_step


def auxcnot_shrink_target(
    interpretation_step: InterpretationStep,
    initial_target: RotatedSurfaceCode,
    merged_target: RotatedSurfaceCode,
    shrink_direction: Direction,
    same_timeslice: bool,
    debug_mode: bool,
) -> InterpretationStep:
    """
    Shrink the target block after merging with the auxiliary block.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        Input interpretation step.
    initial_target : RotatedSurfaceCode
        Initial target block for the auxiliary CNOT operation.
    merged_target : RotatedSurfaceCode
        Merged target block after merging with the auxiliary block.
    same_timeslice : bool
        Flag indicating whether the operation is part of the same timestep as the input.
    debug_mode : bool
        Flag indicating whether the interpretation should be done in debug mode.

    Returns
    -------
    InterpretationStep
        The updated interpretation step after shrinking the target block.
    """
    # Shrink the target block in the direction normal to its X operator
    shrink_length = (
        merged_target.size[0] - initial_target.size[0]
        if merged_target.x_boundary == Orientation.VERTICAL
        else merged_target.size[1] - initial_target.size[1]
    )

    shrink_op = Shrink(
        input_block_name=merged_target.unique_label,
        direction=shrink_direction,
        length=shrink_length,
    )

    interpretation_step = shrink(
        interpretation_step=interpretation_step,
        operation=shrink_op,
        same_timeslice=same_timeslice,
        debug_mode=debug_mode,
    )

    return interpretation_step


def auxcnot_conditional_logical_z(
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    interpretation_step: InterpretationStep,
    new_control: RotatedSurfaceCode,
    initial_target: RotatedSurfaceCode,
    aux_block: RotatedSurfaceCode,
    new_target: RotatedSurfaceCode,
    shrink_direction: Direction,
    same_timeslice: bool,
    debug_mode: bool,
) -> InterpretationStep:
    """
    Applies ConditionalLogicalZ to the blocks conditioned on the value of the joint
    measurement obtained from Merge.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        Input interpretation step.
    new_control : RotatedSurfaceCode
        New control block after the AuxCNOT operations.
    initial_target : RotatedSurfaceCode
        Target block before the the AuxCNOT operations.
    aux_block : RotatedSurfaceCode
        Auxiliary block for the AuxCNOT operation.
    new_target : RotatedSurfaceCode
        New target block after the AuxCNOT operations.
    shrink_direction : Direction
        Direction in which the target block was shrunk.
    same_timeslice : bool
        Flag indicating whether the operation is part of the same timestep as the input.
    debug_mode : bool
        Flag indicating whether the interpretation should be done in debug mode.

    Returns
    -------
    InterpretationStep
        The updated interpretation step after applying the LogicalZ
    """

    log_meas = LogicalMeasurement(
        (initial_target.unique_label, aux_block.unique_label), "XX"
    )

    log_op = ConditionalLogicalZ(new_control.unique_label, condition=log_meas)

    interpretation_step = conditional_logical_pauli(
        interpretation_step=interpretation_step,
        operation=log_op,
        same_timeslice=same_timeslice,
        debug_mode=debug_mode,
    )

    # Additional correction needed if merged_target inherits the auxiliary block's
    # logical X operator, which happens when shrink_direction is "TOP" or "LEFT"

    if shrink_direction in {Direction.TOP, Direction.LEFT}:
        log_op = ConditionalLogicalZ(new_target.unique_label, condition=log_meas)

        interpretation_step = conditional_logical_pauli(
            interpretation_step=interpretation_step,
            operation=log_op,
            same_timeslice=same_timeslice,
            debug_mode=debug_mode,
        )

    return interpretation_step
