"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from loom.interpreter import InterpretationStep
from loom.interpreter.applicator import measureblocksyndromes
from loom.eka import Circuit
from loom.eka.utilities import Direction, Orientation, DiagonalDirection
from loom.eka.operations import MeasureBlockSyndromes, Grow, Shrink
from .utilities import add_vector
from .grow import grow
from .shrink import shrink
from .move_corners import move_corners
from .move_block import move_block
from .logical_phase_via_ywall import move_logical
from ..code_factory import RotatedSurfaceCode
from ..operations import MoveBlock, RotateBlock


def rotate_block(
    interpretation_step: InterpretationStep,
    operation: RotateBlock,
    same_timeslice: bool,
    debug_mode: bool,
) -> InterpretationStep:
    """
    Rotate the boundary of a rotated surface code block by growing it in the specified
    direction, moving the corners in a fault-tolerant manner and shrinking back to the
    original size.

    The rotation is performed by:
    - A) VALIDITY CHECKS
        - A.1) Check that the block is a valid RotatedSurfaceCode block for
                RotateBlock
    - B) GROW THE BLOCK IN THE SPECIFIED DIRECTION
        - B.1) Grow the block to double its size minus one in the specified direction
        - B.2) Measure syndromes to complete the grow operation
        - B.3) Move the logical operators so that they are correctly located on the
                top-left qubit of the grown block
    - C) MOVE THE CORNERS APPROPRIATELY
        - C.1) Move each corner individually
        - C.2) Measure syndromes to complete the corner move in a fault-tolerant manner
    - D) SHRINK THE BLOCK
        - D.1) Shrink the block from the grown side by distance - 2
        - D.2) Shrink the block from the opposite side by 1 to return to original size
    - E) MOVE BLOCK
        - E.1) Move the block so that it's occupying the same data qubits as initially
        - E.2) Move the logical operators so that they are correctly located on the
                top-left qubit of the shrunk block
    - F) WRAP THE CIRCUIT
        - F.1) Isolate the circuit corresponding to the RotateBlock operation and append
                it as a single Circuit object to the intermediate circuit sequence


    Parameters:
    ------------------
    interpretation_step : InterpretationStep
        The current interpretation step containing the block to be rotated.
    operation : RotateBlock
        The RotateBlock operation specifying the block and direction.
    same_timeslice : bool
        Flag to apply the operation in the same timeslice as the previous operation(s).
    debug_mode: bool
        Flag to apply validation of the new block or not.

    Returns:
    ------------------
    InterpretationStep
        The updated interpretation step after applying the RotateBlock operation.
    """
    # Extract operation parameters
    rsc_block = interpretation_step.get_block(operation.input_block_name)
    grow_direction = operation.grow_direction

    # A) VALIDITY CHECKS
    # A.1) Check that the block is a valid RotatedSurfaceCode block for RotateBlock
    rotate_validity_check(rsc_block)

    # Extract useful parameters
    # - distance of the block
    # - initial length of the intermediate circuit sequence so that we can later
    #   isolate the rotate boundary circuit
    distance = min(rsc_block.size)
    init_circ_len = len(interpretation_step.intermediate_circuit_sequence)
    current_block = interpretation_step.get_block(rsc_block.unique_label)

    # B) GROW THE BLOCK IN THE SPECIFIED DIRECTION
    # B.1) Grow the block to double its size minus one in the specified direction
    interpretation_step = grow(
        interpretation_step,
        Grow(rsc_block.unique_label, grow_direction, length=distance - 1),
        same_timeslice=False,
        debug_mode=debug_mode,
    )
    current_block = interpretation_step.get_block(rsc_block.unique_label)

    # B.2) Measure syndromes to complete the grow operation
    interpretation_step = measureblocksyndromes(
        interpretation_step,
        MeasureBlockSyndromes(rsc_block.unique_label, n_cycles=distance),
        same_timeslice=False,
        debug_mode=debug_mode,
    )

    # B.3) Move the logical operators so that they are correctly located on the top-left
    #     qubit of the grown block
    top_left_qubit = current_block.upper_left_qubit
    for logical_pauli in ["X", "Z"]:
        interpretation_step, current_block = move_logical(
            interpretation_step,
            current_block,
            new_up_left_qubit=top_left_qubit,
            pauli=logical_pauli,
        )

    # C) MOVE THE CORNERS APPROPRIATELY
    for corner_args in get_boundary_rotation_corner_args_5_1(current_block):
        # C.1) Move each corner individually
        # pylint: disable=duplicate-code
        interpretation_step = move_corners(
            interpretation_step,
            current_block,
            corner_args,
            same_timeslice=False,
            debug_mode=debug_mode,
        )
        current_block = interpretation_step.get_block(rsc_block.unique_label)
        # C.2) Measure syndromes to complete the corner move in a fault-tolerant manner
        interpretation_step = measureblocksyndromes(
            interpretation_step,
            MeasureBlockSyndromes(rsc_block.unique_label, n_cycles=distance - 1),
            same_timeslice=False,
            debug_mode=debug_mode,
        )

    # D) SHRINK THE BLOCK
    # D.1) Shrink the block from the grown side by distance - 2
    interpretation_step = shrink(
        interpretation_step,
        Shrink(current_block.unique_label, grow_direction, length=distance - 2),
        same_timeslice=False,
        debug_mode=debug_mode,
    )
    current_block = interpretation_step.get_block(rsc_block.unique_label)

    # D.2) Shrink the block from the opposite side by 1 to return to original size
    interpretation_step = shrink(
        interpretation_step,
        Shrink(current_block.unique_label, grow_direction.opposite(), length=1),
        same_timeslice=False,
        debug_mode=debug_mode,
    )
    current_block = interpretation_step.get_block(rsc_block.unique_label)

    # E) MOVE BLOCK
    # E.1) Move the block so that it's occupying the same data qubits as initially
    interpretation_step = move_block(
        interpretation_step,
        MoveBlock(current_block.unique_label, direction=grow_direction.opposite()),
        same_timeslice=False,
        debug_mode=debug_mode,
    )
    current_block = interpretation_step.get_block(rsc_block.unique_label)

    # E.2) Move the logical operators so that they are correctly located on the top-left
    #     qubit of the shrunk block
    top_left_qubit = current_block.upper_left_qubit
    for logical_pauli in ["X", "Z"]:
        interpretation_step, current_block = move_logical(
            interpretation_step,
            current_block,
            new_up_left_qubit=top_left_qubit,
            pauli=logical_pauli,
        )

    # F) WRAP THE CIRCUIT
    # F.1) Isolate the circuit corresponding to the RotateBlock operation and append
    #      it as a single Circuit object to the intermediate circuit sequence
    new_len = len(interpretation_step.intermediate_circuit_sequence)
    len_rotate_block = new_len - init_circ_len
    circuit_seq = interpretation_step.pop_intermediate_circuit_MUT(len_rotate_block)
    # pylint: disable=duplicate-code
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
            f"rotate boundary {rsc_block.unique_label} by growing towards "
            f"{grow_direction.name}"
        ),
        circuit=wrapped_circuit_seq,
    )
    interpretation_step.append_circuit_MUT(wrapped_circuit, same_timeslice)
    return interpretation_step


def rotate_validity_check(rsc_block: RotatedSurfaceCode):
    """
    Validity checks for applying RotateBlock to a RotatedSurfaceCode block.
    We check that:
    - The block is indeed a RotatedSurfaceCode block
    - The block is square and has odd distance
    - The topological and geometric corners match
    """
    if not isinstance(rsc_block, RotatedSurfaceCode):
        raise ValueError("Cannot apply RotateBlock to non-RotatedSurfaceCode blocks.")
    if not (rsc_block.size[0] == rsc_block.size[1] and rsc_block.size[0] % 2 == 1):
        raise ValueError(
            "Cannot apply RotateBlock to non-square blocks or blocks with even "
            "distance."
        )
    if not set(rsc_block.topological_corners) == set(rsc_block.geometric_corners):
        raise ValueError(
            "Cannot apply RotateBlock to blocks when the topological and geometric "
            "corners do not match."
        )


def get_boundary_rotation_corner_args_5_1(
    rsc_block: RotatedSurfaceCode,
) -> tuple[tuple[tuple[int, ...], Direction, int], ...]:
    """
    Rotate the boundaries by using move_corners 5 times, moving 1 corner at
    a time. The corners are moved in a way such that the distance of the block for all
    logical operators is at least the original distance, ensuring fault-tolerance.
    For this to be possible, the block must be of size (d, 2 * d - 1) or (2 * d - 1, d)
    with odd d.

    Parameters:
    ------------------
    rsc_block : RotatedSurfaceCode
        The grown rectangular rotated surface code block.

    Returns:
    ------------------
    tuple[tuple[tuple[int, ...], Direction, int], ...]
        A tuple containing 5 tuples, each specifying the arguments for a call to
        move_corners. Each inner tuple contains:
        - The coordinates of the corner to move (as a tuple of integers).
        - The direction in which to move the corner (as a Direction enum).
        - The distance to move the corner (as an integer).
    """
    long_orientation = (
        Orientation.HORIZONTAL
        if rsc_block.size[0] > rsc_block.size[1]
        else Orientation.VERTICAL
    )

    # Find the corner qubits for ease of access
    tl = rsc_block.get_corner_from_direction(
        DiagonalDirection.from_directions((Direction.TOP, Direction.LEFT))
    )
    tr = rsc_block.get_corner_from_direction(
        DiagonalDirection.from_directions((Direction.TOP, Direction.RIGHT))
    )
    br = rsc_block.get_corner_from_direction(
        DiagonalDirection.from_directions((Direction.BOTTOM, Direction.RIGHT))
    )
    bl = rsc_block.get_corner_from_direction(
        DiagonalDirection.from_directions((Direction.BOTTOM, Direction.LEFT))
    )

    # Get the distances
    short_distance = min(rsc_block.size)
    long_distance = max(rsc_block.size)

    if long_distance != 2 * short_distance - 1 or short_distance % 2 != 1:
        raise ValueError(
            "Cannot obtain corner moves for blocks that are not of size (d, 2 * d -1)"
            f" or (2 * d - 1, d) with odd d. Given size: {rsc_block.size}"
        )

    # The lengths to move each corner
    short_length = min(rsc_block.size) - 1
    long_length = max(rsc_block.size) - 1

    # Return the appropriate sequence of corner moves
    match (rsc_block.weight_2_stab_is_first_row, long_orientation):
        case (False, Orientation.VERTICAL):
            return (
                ((tl, Direction.BOTTOM, short_length),),  # Move TL halfway down
                ((tr, Direction.LEFT, short_length),),  # Move TR left
                ((br, Direction.TOP, long_length),),  # Move BR up
                ((bl, Direction.RIGHT, short_length),),  # Move BL right
                (
                    (
                        add_vector(tl, Direction.BOTTOM, short_length),
                        Direction.BOTTOM,
                        short_length,
                    ),
                ),  # Move initial corner all the way down
            )
        case (True, Orientation.VERTICAL):
            return (
                ((tr, Direction.BOTTOM, short_length),),  # Move TR halfway down
                ((tl, Direction.RIGHT, short_length),),  # Move TL right
                ((bl, Direction.TOP, long_length),),  # Move BL up
                ((br, Direction.LEFT, short_length),),  # Move BR left
                (
                    (
                        add_vector(tr, Direction.BOTTOM, short_length),
                        Direction.BOTTOM,
                        short_length,
                    ),
                ),  # Move initial corner all the way down
            )
        case (False, Orientation.HORIZONTAL):
            return (
                ((bl, Direction.RIGHT, short_length),),  # Move BL halfway right
                ((tl, Direction.BOTTOM, short_length),),  # Move TL down
                ((tr, Direction.LEFT, long_length),),  # Move TR left
                ((br, Direction.TOP, short_length),),  # Move BR up
                (
                    (
                        add_vector(bl, Direction.RIGHT, short_length),
                        Direction.RIGHT,
                        short_length,
                    ),
                ),  # Move initial corner all the way right
            )
        case (True, Orientation.HORIZONTAL):
            return (
                ((tl, Direction.RIGHT, short_length),),  # Move TL halfway right
                ((bl, Direction.TOP, short_length),),  # Move BL up
                ((br, Direction.LEFT, long_length),),  # Move BR left
                ((tr, Direction.BOTTOM, short_length),),  # Move TR down
                (
                    (
                        add_vector(tl, Direction.RIGHT, short_length),
                        Direction.RIGHT,
                        short_length,
                    ),
                ),  # Move initial corner all the way right
            )
        case _:
            raise ValueError(
                "Invalid combination of weight_2_stab_is_first_row and "
                "long_orientation"
            )
