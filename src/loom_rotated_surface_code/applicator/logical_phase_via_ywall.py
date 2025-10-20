"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from loom.eka import Circuit
from loom.interpreter.applicator.code_applicator import measureblocksyndromes
from loom.eka.operations import Grow, Shrink, MeasureBlockSyndromes
from loom.interpreter.interpretation_step import InterpretationStep
from loom.eka.utilities import Direction, Orientation, SyndromeMissingError

from .grow import grow
from .shrink import shrink
from .move_corners import move_corners
from .y_wall_out import y_wall_out
from ..operations import LogicalPhaseViaYwall
from ..code_factory import RotatedSurfaceCode


# pylint: disable=too-many-statements
def logical_phase_via_ywall(  # pylint: disable=too-many-locals
    interpretation_step: InterpretationStep,
    operation: LogicalPhaseViaYwall,
    same_timeslice: bool,
    debug_mode: bool,
) -> InterpretationStep:
    """
    Apply the logical phase via y-wall operation.
    The algorithm is the following:

    - A.) Run consistency checks
    - B.) Relocate the x logical operator to the appropriate position
    - C.) Grow the block
    - D.) Move a corner to prepare for the y-wall operation
    - E.) Measure the syndromes
    - F.) Apply the y_wall_out operation
    - G.) Move all topological corners back to their geometric positions and \
    potentially grow the block towards the initial position
    - H.) Shrink the block
    - I.) Relocate the x logical operator to the initial position
    - J.) Wrap the circuit in a single circuit with the correct duration and empty \
    timeslices

    Note that whenever the logical operator is moved, there may be some extra cycles of
    measurement of the block syndromes if there are missing syndromes.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        The current state of the interpretation step.
    operation : LogicalPhaseViaYwall
        The operation to be applied.
    same_timeslice : bool
        Whether to apply the operation in the same timeslice or not.
    debug_mode : bool
        Whether to run in debug mode or not.

    Returns
    -------
    InterpretationStep
        The updated interpretation step after applying the operation.
    """
    ## A) Run consistency checks
    # Check that the RotatedSurfaceCode block is square and has odd dimensions
    # Check that the topological corners coincide with the geometric corners
    # Store the initial block and the current block, initial_block is a reference that
    # will never be modified while current_block will be re-assigned during the
    # operation.
    init_block = current_block = check_consistency(interpretation_step, operation)

    growth_direction = operation.growth_direction
    init_circ_len = len(interpretation_step.intermediate_circuit_sequence)
    init_x_log_operator = init_block.logical_x_operators[0]

    # Extract operation parameters after the checks
    distance = init_block.size[0]
    is_growth_towards_negative = growth_direction in [Direction.LEFT, Direction.TOP]
    is_init_top_left_bulk_stab_x = (
        init_block.upper_left_4body_stabilizer.pauli[0] == "X"
    )
    is_top_left_bulk_stab_x = is_growth_towards_negative ^ is_init_top_left_bulk_stab_x
    is_x_boundary_horizontal = init_block.x_boundary == Orientation.HORIZONTAL

    ## B) Relocate the x logical operator to the appropriate position
    # Depending on the top-left bulk stabilizer and the x boundary orientation,
    # the top-left qubit of the x logical operator needs to be moved to a
    # different position such that after the block is grown, the x logical operator
    # is in the correct position for the y-wall operation.
    match (is_top_left_bulk_stab_x, is_x_boundary_horizontal):
        case (False, False):
            # Top-left qubit
            new_x_log_op_top_left_qubit = max(
                current_block.data_qubits, key=lambda x: -x[0] - x[1]
            )
        case (True, False):
            # Top-right qubit
            new_x_log_op_top_left_qubit = max(
                current_block.data_qubits, key=lambda x: +x[0] - x[1]
            )
        case (True, True):
            # Bottom-left qubit
            new_x_log_op_top_left_qubit = max(
                current_block.data_qubits, key=lambda x: -x[0] + x[1]
            )
        case (False, True):
            # Top-left qubit
            new_x_log_op_top_left_qubit = max(
                current_block.data_qubits, key=lambda x: -x[0] - x[1]
            )
    # Move the x logical operator to the new position
    try:
        interpretation_step, current_block = move_logical(
            interpretation_step, current_block, new_x_log_op_top_left_qubit, "X"
        )
    except SyndromeMissingError:
        # Measure block syndromes to get missing syndromes
        interpretation_step = measureblocksyndromes(
            interpretation_step,
            MeasureBlockSyndromes(init_block.unique_label, distance),
            same_timeslice=False,
            debug_mode=debug_mode,
        )
        # Move the x logical operator to the new position
        interpretation_step, current_block = move_logical(
            interpretation_step, current_block, new_x_log_op_top_left_qubit, "X"
        )

    ## C) Grow the block
    # The block is grown to the right if the x boundary is horizontal and down if
    # the x boundary is vertical
    growth_length = distance

    interpretation_step = grow(
        interpretation_step,
        Grow(init_block.unique_label, growth_direction, growth_length),
        same_timeslice=False,
        debug_mode=debug_mode,
    )

    # If the growth direction is towards the negative direction, the Z logical operator
    # needs to be moved to the new top-left qubit of the block.
    if is_growth_towards_negative:

        current_block = interpretation_step.get_block(init_block.unique_label)
        # Measure block syndromes to get missing syndromes
        interpretation_step = measureblocksyndromes(
            interpretation_step,
            MeasureBlockSyndromes(init_block.unique_label, distance),
            same_timeslice=False,
            debug_mode=debug_mode,
        )

        # Move the Z logical operator to the new position
        interpretation_step, current_block = move_logical(
            interpretation_step,
            current_block,
            current_block.upper_left_qubit,
            pauli="Z",
        )

    ## D) Move the corner to prepare for the y-wall operation
    # Depending on the top-left bulk stabilizer and the x boundary orientation,
    # a different corner qubit needs to be moved to prepare for the y-wall operation.
    current_block: RotatedSurfaceCode = interpretation_step.get_block(
        init_block.unique_label
    )
    match (is_top_left_bulk_stab_x, is_x_boundary_horizontal):
        case (False, False):
            corner_0_location_directions = [Direction.BOTTOM, Direction.LEFT]
            corner_0_move_direction = Direction.TOP
        case (True, False):
            corner_0_location_directions = [Direction.BOTTOM, Direction.RIGHT]
            corner_0_move_direction = Direction.TOP
        case (True, True):
            corner_0_location_directions = [Direction.BOTTOM, Direction.RIGHT]
            corner_0_move_direction = Direction.LEFT
        case (False, True):
            corner_0_location_directions = [Direction.TOP, Direction.RIGHT]
            corner_0_move_direction = Direction.LEFT
    corner_0_location = (
        set(current_block.boundary_qubits(corner_0_location_directions[0]))
        .intersection(current_block.boundary_qubits(corner_0_location_directions[1]))
        .pop()
    )
    # Find the direction to move the corner qubit
    corner_0_how_far = distance - 1
    # Move the corner
    corner_args = ((corner_0_location, corner_0_move_direction, corner_0_how_far),)
    interpretation_step = move_corners(
        interpretation_step,
        current_block,
        corner_args,
        same_timeslice=False,
        debug_mode=debug_mode,
    )

    ## E) Measure the syndromes
    # Measure the syndromes of the block such that the block is projected onto
    # the new stabilizers before the y-wall operation is applied.
    # Measure for distance - 1 since the move_corners operation has already measured
    # the stabilizers once.
    interpretation_step = measureblocksyndromes(
        interpretation_step,
        MeasureBlockSyndromes(init_block.unique_label, distance - 1),
        same_timeslice=False,
        debug_mode=debug_mode,
    )

    ## F) Apply the y_wall_out operation
    # The y_wall_out operation is applied to implement the main part of the
    # logical phase via y-wall operation.
    wall_position = distance
    wall_orientation = init_block.x_boundary.perpendicular()
    current_block = interpretation_step.get_block(init_block.unique_label)

    interpretation_step = y_wall_out(
        interpretation_step, current_block, wall_position, wall_orientation, debug_mode
    )

    ## G) Move all topological corners back to their geometric positions and
    ## potentially grow the block towards the initial position
    # Move the topological corners back to their geometric positions
    current_block = interpretation_step.get_block(init_block.unique_label)

    # This is a geometric corner but at the incorrect position.
    # Depending on the top-left bulk stabilizer and the x boundary orientation,
    # a different corner qubit needs to be moved to the appropriate direction.
    match (is_top_left_bulk_stab_x, is_x_boundary_horizontal):
        case (False, False):
            corner_1_location_directions = [Direction.BOTTOM, Direction.RIGHT]
            corner_1_move_direction = Direction.LEFT
        case (True, False):
            corner_1_location_directions = [Direction.BOTTOM, Direction.LEFT]
            corner_1_move_direction = Direction.RIGHT
        case (True, True):
            corner_1_location_directions = [Direction.TOP, Direction.RIGHT]
            corner_1_move_direction = Direction.BOTTOM
        case (False, True):
            corner_1_location_directions = [Direction.BOTTOM, Direction.RIGHT]
            corner_1_move_direction = Direction.TOP
    corner_1_location = (
        set(current_block.boundary_qubits(corner_1_location_directions[0]))
        .intersection(current_block.boundary_qubits(corner_1_location_directions[1]))
        .pop()
    )
    corner_1_move_how_far = distance - 1
    # Corner 1 arguments
    corner_1_args = (corner_1_location, corner_1_move_direction, corner_1_move_how_far)

    # Move corner 2
    # This is a topological corner but at the incorrect position.
    current_block = interpretation_step.get_block(init_block.unique_label)
    # Find the topological corner that is not located at the geometric corner
    corner_2_location = (
        set(current_block.topological_corners) - set(current_block.geometric_corners)
    ).pop()
    # It needs to be moved in the opposite direction of corner 0
    corner_2_move_direction = corner_0_move_direction.opposite()
    corner_2_how_far = distance - 1
    # Corner 2 arguments
    corner_2_args = (corner_2_location, corner_2_move_direction, corner_2_how_far)

    # Move the corners
    interpretation_step = move_corners(
        interpretation_step=interpretation_step,
        block=current_block,
        corner_args=(
            corner_1_args,
            corner_2_args,
        ),
        same_timeslice=False,
        debug_mode=debug_mode,
    )

    # If the growth direction is towards the negative direction, the block needs to
    # be grown by one unit towards the opposite direction of the growth direction
    # for the final shrink to leave it at the correct position.
    # NOTE: This is because y_wall_out has been implemented such that the block is
    # shrunk from the right or from the bottom.
    if is_growth_towards_negative:
        # Grow by one towards the opposite direction of the growth direction
        interpretation_step = grow(
            interpretation_step,
            Grow(init_block.unique_label, growth_direction.opposite(), 1),
            same_timeslice=False,
            debug_mode=debug_mode,
        )

    ## H) Shrink the block
    # The block is shrunk to the original size. Since the y_wall_out operation
    # shrunk the block by 1, the block needs to be shrunk by 1 unit less than the
    # growth operation.
    shrink_direction = growth_direction
    shrink_length = growth_length - (1 if not is_growth_towards_negative else 0)
    # Shrink the block
    interpretation_step = shrink(
        interpretation_step,
        Shrink(init_block.unique_label, shrink_direction, shrink_length),
        same_timeslice=False,
        debug_mode=debug_mode,
    )

    ## I) Relocate the x logical operator to the initial position
    # The x logical operator is moved back to its original position.
    current_block = interpretation_step.get_block(init_block.unique_label)
    # Find the top-left corner of the initial x logical operator
    final_x_log_op_top_left_qubit = max(
        init_x_log_operator.data_qubits, key=lambda x: -x[0] - x[1]
    )

    # Move the x logical operator back to the original position
    try:
        interpretation_step, current_block = move_logical(
            interpretation_step, current_block, final_x_log_op_top_left_qubit, "X"
        )
    except SyndromeMissingError:
        # Measure block syndromes to get missing syndromes
        interpretation_step = measureblocksyndromes(
            interpretation_step,
            MeasureBlockSyndromes(init_block.unique_label, distance),
            same_timeslice=False,
            debug_mode=debug_mode,
        )
        # Move the x logical operator back to the original position
        interpretation_step, current_block = move_logical(
            interpretation_step, current_block, final_x_log_op_top_left_qubit, "X"
        )

    ## J) Wrap the circuit in a single circuit with the correct duration and empty
    # timeslices
    new_len = len(interpretation_step.intermediate_circuit_sequence)
    len_ywall_phase_circ = new_len - init_circ_len
    circuit_seq = interpretation_step.pop_intermediate_circuit_MUT(len_ywall_phase_circ)

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
            f"LogicalPhaseViaYwall on block {init_block.unique_label} "
            f"towards the {growth_direction.name}."
        ),
        circuit=wrapped_circuit_seq,
    )
    interpretation_step.append_circuit_MUT(wrapped_circuit, same_timeslice)

    return interpretation_step


def check_consistency(
    interpretation_step: InterpretationStep, operation: LogicalPhaseViaYwall
) -> RotatedSurfaceCode:
    """
    Check that the block is consistent with the logical phase via y-wall operation.

    Parameters
    ----------
    interpretation_step: InterpretationStep
        The InterpretationStep that contains the block to which the operation is
        applied.
    operation: LogicalPhaseViaYwall
        The operation to be applied.

    Returns
    -------
    RotatedSurfaceCode
        The block to which the operation is applied.

    Raises
    ------
    ValueError
        If the block is not square and has does not have odd dimensions.
        If the topological corners do not coincide with the geometric corners.
    """
    # Obtain the info from interpretation step
    block = interpretation_step.get_block(operation.input_block_name)
    if not isinstance(block, RotatedSurfaceCode):
        raise ValueError(
            f"The LogicalPhaseViaYwall operation can only be applied to "
            f"RotatedSurfaceCode blocks. The block {block.unique_label} "
            f"is a {block.__class__.__name__} block."
        )

    dim_x, dim_y = block.size
    if dim_x != dim_y or dim_x % 2 != 1:
        raise ValueError(
            "Block must be square and have odd dimensions for the logical "
            "phase via y-wall operation."
        )

    if set(block.topological_corners) != set(block.geometric_corners):
        raise ValueError(
            "Topological corners must coincide with geometric corners for "
            "the logical phase via y-wall operation."
        )

    growth_direction = operation.growth_direction
    if growth_direction.to_orientation() != block.x_boundary:
        raise ValueError(
            f"The growth direction ({growth_direction.name}) must be parallel to the x "
            f"boundary ({block.x_boundary.name}) of the block for the logical phase "
            "via y-wall operation."
        )

    return block


def move_logical(
    interpretation_step: InterpretationStep,
    current_block: RotatedSurfaceCode,
    new_up_left_qubit: tuple[int, int, int],
    pauli: str,
) -> tuple[InterpretationStep, RotatedSurfaceCode]:
    """
    Move the logical operator to a new position. If the logical operator is
    already in the new position, no action is taken. If the logical operator
    is not in the new position, it is moved to the new position and the stabilizers are
    redefined. We assume that the block is square and its logical operator is a
    straight line. Can be used for both X and Z logical operators.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        The current state of the interpretation step.
    current_block : RotatedSurfaceCode
        The block to which the operation is applied.
    new_up_left_qubit : tuple[int, int, int]
        The new position of the top-left qubit of the logical x operator.
    pauli : str
        The logical operator to be moved. Can be either "X" or "Z".

    Returns
    -------
    interpretation_step : InterpretationStep
        The updated interpretation step after applying the operation.
    current_block : RotatedSurfaceCode
        The updated block after applying the operation.

    Raises
    ------
    SyndromeMissingError
        If syndromes required to redefine the logical x operator are missing.
    """
    if pauli == "X":
        current_logical_op = current_block.logical_x_operators[0]
    elif pauli == "Z":
        current_logical_op = current_block.logical_z_operators[0]
    else:
        raise ValueError(
            f"Logical operator {pauli} is not supported. Only 'X' and 'Z' are "
            "supported."
        )

    # Use method to obtain the new logical operator and the stabilizers
    # that need to be redefined
    new_logical, stabilizers_to_redefine = (
        current_block.get_shifted_equivalent_logical_operator(
            current_logical_op, new_up_left_qubit
        )
    )

    # If the logical x operator is not in the new position, we need to redefine
    # the logical x operator and declare its evolution
    if stabilizers_to_redefine:
        # Check that the stabilizers to redefine have syndromes associated
        # with the input block. This is to ensure that the logical phase
        # NOTE: At this point a `SyndromeMissingError` is raised if the syndromes are
        # not associated with the input block.
        redefinition_stab_cbits = interpretation_step.retrieve_cbits_from_stabilizers(
            stabilizers_to_redefine, current_block
        )

        # Find the evolution of the logical operators
        if pauli == "X":
            interpretation_step.logical_x_evolution[new_logical.uuid] = (
                current_block.logical_x_operators[0].uuid,
            ) + tuple(stab.uuid for stab in stabilizers_to_redefine)

            # Define the new logical x/z operator
            # (only the logical x operator is moved)
            new_logical_x = new_logical
            new_logical_z = current_block.logical_z_operators[0]
        else:
            interpretation_step.logical_z_evolution[new_logical.uuid] = (
                current_block.logical_z_operators[0].uuid,
            ) + tuple(stab.uuid for stab in stabilizers_to_redefine)

            # Define the new logical z/x operator
            # (only the logical z operator is moved)
            new_logical_x = current_block.logical_x_operators[0]
            new_logical_z = new_logical

        # Pass the logical updates onto the new logical operator
        # along with the Cbits from the stabilizers that are need to redefine it
        interpretation_step.update_logical_operator_updates_MUT(
            pauli, new_logical.uuid, redefinition_stab_cbits, inherit_updates=True
        )

        new_block = RotatedSurfaceCode(
            stabilizers=current_block.stabilizers,
            logical_x_operators=(new_logical_x,),
            logical_z_operators=(new_logical_z,),
            syndrome_circuits=current_block.syndrome_circuits,
            stabilizer_to_circuit=current_block.stabilizer_to_circuit,
            unique_label=current_block.unique_label,
        )
        interpretation_step.update_block_history_and_evolution_MUT(
            (new_block,), (current_block,)
        )
        current_block = new_block

    return interpretation_step, current_block
