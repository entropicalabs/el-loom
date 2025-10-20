"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=too-many-lines
from loom.eka import Circuit, ChannelType, SyndromeCircuit, Stabilizer, PauliOperator
from loom.eka.utilities import Direction, Orientation, DiagonalDirection
from loom.eka.operations import MeasureBlockSyndromes
from loom.interpreter.applicator import measureblocksyndromes
from loom.interpreter import InterpretationStep

from loom_rotated_surface_code.code_factory import RotatedSurfaceCode
from .utilities import generate_syndrome_extraction_circuits


def move_corners(
    interpretation_step: InterpretationStep,
    block: RotatedSurfaceCode,
    corner_args: tuple[tuple[tuple[int, int, int], Direction, int], ...],
    same_timeslice: bool,
    debug_mode: bool,
) -> InterpretationStep:
    """
    Move the selected topological corners of the block in the specified direction. We
     assume that the initial state is a RotatedSurfaceCode with a single logical qubit
     and the topological corners are located at the geometric corners.
     This function can move he topological corners mutlitple times in a row, as long as
     the qubits selected are topological corners of the block.

    NOTE: This function automatically measures the syndromes of the final block once.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        The interpretation step before moving the corners
    block : RotatedSurfaceCode
        The initial block
    corner_args : tuple[tuple[tuple[int, int, int], Direction, int], ...]
        List of tuples containing the corner qubit to move, the direction in which to
        move it and the distance by which to move it
    same_timeslice : bool
        Flag to apply move_corners in the same timeslice as the previous operation(s).
    debug_mode: bool
        Flag to apply validation of the new block or not.

    Returns
    -------
    InterpretationStep
        The interpretation step after moving the corners
    """
    initial_block_name = block.unique_label

    # Collect the stabilizers that haven't been measured yet but are required to
    # update the logical operators
    stabilizers_required_for_future_update = {"X": (), "Z": ()}

    # Iterate over the corners to move
    for corner_qubit, move_direction, how_far in corner_args:
        # Store the previous logical_operators to check if we need to inherit updates
        previous_logical_operators = (
            block.logical_x_operators[0],
            block.logical_z_operators[0],
        )

        # Rotate corners
        interpretation_step, past_stabilizers, future_stabilizers = move_corner(
            interpretation_step,
            block,
            corner_qubit,
            move_direction,
            how_far,
        )
        stabilizers_required_for_future_update["X"] += future_stabilizers[0]
        stabilizers_required_for_future_update["Z"] += future_stabilizers[1]
        # Change the reference to the block
        block = interpretation_step.get_block(initial_block_name)

        # Add updates from the past stabilizers
        for pauli_type, past_stabs, logical_operator in zip(
            ("X", "Z"),
            past_stabilizers,
            (block.logical_x_operators[0], block.logical_z_operators[0]),
            strict=True,
        ):
            cbits = tuple(
                cbit
                for stab in past_stabs
                for cbit in interpretation_step.get_prev_syndrome(
                    stab.uuid, block.uuid
                )[0].measurements
            )
            # We only want to inherit updates once every time the logical operator is
            # changed
            inherit_updates = not logical_operator in previous_logical_operators
            interpretation_step.update_logical_operator_updates_MUT(
                pauli_type,
                logical_operator.uuid,
                cbits,
                inherit_updates=inherit_updates,
            )

    # Measure the syndromes of the new block
    interpretation_step = measureblocksyndromes(
        interpretation_step=interpretation_step,
        operation=MeasureBlockSyndromes(initial_block_name),
        same_timeslice=same_timeslice,
        debug_mode=debug_mode,
    )
    # Add the logical updates from the remaining stabilizers
    # Find the previous Cbit associated with the stabilizers required for the updates
    for pauli_type, future_stabs in stabilizers_required_for_future_update.items():
        logical_operator = (
            interpretation_step.get_block(initial_block_name).logical_x_operators[0]
            if pauli_type == "X"
            else interpretation_step.get_block(initial_block_name).logical_z_operators[
                0
            ]
        )
        cbits = tuple(
            cbit
            for stab in future_stabs
            for cbit in interpretation_step.get_prev_syndrome(stab.uuid, block.uuid)[
                0
            ].measurements
        )
        interpretation_step.update_logical_operator_updates_MUT(
            pauli_type, logical_operator.uuid, cbits, inherit_updates=False
        )

    return interpretation_step


def move_corner(  # pylint: disable=too-many-locals, too-many-statements
    interpretation_step: InterpretationStep,
    block: RotatedSurfaceCode,
    corner_qubit: tuple[int, int, int],
    move_direction: Direction,
    how_far: int,
) -> tuple[
    InterpretationStep,
    tuple[tuple[Stabilizer, ...], tuple[Stabilizer, ...]],
    tuple[tuple[Stabilizer, ...], tuple[Stabilizer, ...]],
]:
    """
    Move the selected topological corner of the block in the specified direction. We
     assume that the initial state is a RotatedSurfaceCode with a single logical qubit
     and the topological corners are located at the geometric corners.

    The algorithm is the following:
    - A) STABILIZERS
        - A.1) Cut the corner stabilizer if a 2-body stabilizer is involved
        - A.2) Generate the two-body stabilizers to be added
        - A.3) Find which stabilizers are untouched on the modified boundary
        - A.4) Compute the evolution of the stabilizers
        - A.5) Construct the set of kept bulk stabilizers
        - A.6) Construct the new set of stabilizers
    - B) LOGICAL OPERATORS
        - B.1) Move the logical operators so they stay on the right boundary or
        extend/contract them
        - B.2) Update the logical operators
        - B.3) Collect the necessary stabilizers for logical operator updates
    - C) SYNDROME CIRCUITS
        - C.1) Find corner configuration of new block
        - C.2) Generate new syndrome circuits and stabilizer schedules based on corner
        config
        - C.3) Update stabilizer evolution
    - D) CIRCUIT
        - D.1) Measure the corner if needed
        - D.2) Create stabilizer and logical operator updates
    - E) BUILD BLOCK

    Parameters
    ----------
    interpretation_step : InterpretationStep
        The interpretation step before moving the corner
    block : RotatedSurfaceCode
        The initial block
    corner_qubit : tuple[int, int, int],
        Corner to move
    move_direction : Direction
        Direction in which to move the corner
    how_far : int
        Distance by which to move the corner

    Returns
    -------
    tuple[
        InterpretationStep,
        tuple[tuple[Stabilizer, ...], tuple[Stabilizer, ...]],
        tuple[tuple[Stabilizer, ...], tuple[Stabilizer, ...]],
    ]
        The interpretation step after moving the corner, the set of X and Z stabilizers
         removed during the move that are required for logical updates and the set of X
         and Z stabilizers added during the move that are required for logical updates.
    """
    # We first compute some important values:
    # The boundary associated with the corner
    corner_directions = get_associated_boundaries(block, corner_qubit)
    # The boundary modified depends on the corner and the direction of the movement
    modified_boundary_direction = find_new_boundary_direction(
        corner_directions, move_direction
    )
    # The boundary that is extended, e.g. if the top left corner is moved to the right,
    # the left boundary is extended
    unit_vector = move_direction.to_vector()

    two_body_is_included = is_2_body_included(
        block, corner_qubit, modified_boundary_direction
    )
    # A) STABILIZERS
    #   A.1) Cut the corner stabilizer if a 2-body stabilizer is involved
    #   A.2) Generate the two-body stabilizers to be added
    #   A.3) Find which stabilizers are untouched on the modified boundary
    #   A.4) Compute the evolution of the stabilizers
    (
        kept_boundary_stabilizers,
        new_boundary_stabilizers,
        cut_stabilizer,
        stabilizer_evolution,
    ) = find_new_boundary_stabilizers(
        block,
        corner_qubit,
        modified_boundary_direction,
        unit_vector,
        how_far,
        two_body_is_included,
    )
    old_stabs_to_be_removed = [
        stab
        for stab in block.boundary_stabilizers(modified_boundary_direction)
        if stab not in kept_boundary_stabilizers
    ]
    interpretation_step.stabilizer_evolution.update(stabilizer_evolution)
    #   A.5) Construct the set of kept bulk stabilizers
    kept_bulk_stabilizers = [
        stab
        for stab in block.bulk_stabilizers
        if not (corner_qubit in stab.data_qubits and two_body_is_included)
    ]
    #   A.6) Construct the new set of stabilizers
    new_stabilizers = (
        kept_boundary_stabilizers + new_boundary_stabilizers + kept_bulk_stabilizers
    )
    if cut_stabilizer is not None:
        new_stabilizers.append(cut_stabilizer)

    # B) LOGICAL OPERATORS
    #   B.1) Move the logical operators so they stay on the right boundary or
    #        extend/contract them
    new_x_op, new_z_op, log_x_evolution_dict, log_z_evolution_dict = (
        move_corner_logical_operators(
            block=block,
            corner_qubit=corner_qubit,
            unit_vector=unit_vector,
            how_far=how_far,
            new_stabs_to_be_added=new_boundary_stabilizers,
            old_stabs_to_be_removed=old_stabs_to_be_removed,
        )
    )
    #   B.2) Update the logical operator evolution
    interpretation_step.logical_x_evolution.update(log_x_evolution_dict)
    interpretation_step.logical_z_evolution.update(log_z_evolution_dict)

    #   B.3) Collect the necessary stabilizers for logical operator updates
    # Collect the removed stabilizers that are required for logical updates
    # (their syndromes should have been measured before)
    past_x_stabilizers = tuple(
        stab
        for stab in old_stabs_to_be_removed
        for pauli_op_id in interpretation_step.logical_x_evolution.get(
            new_x_op.uuid, ()
        )
        if stab.uuid == pauli_op_id
    )
    past_z_stabilizers = tuple(
        stab
        for stab in old_stabs_to_be_removed
        for pauli_op_id in interpretation_step.logical_z_evolution.get(
            new_z_op.uuid, ()
        )
        if stab.uuid == pauli_op_id
    )
    past_stabilizers = (past_x_stabilizers, past_z_stabilizers)

    # Collect the new stabilizers that are required for logical updates
    # (their syndromes have not been measured yet)
    future_x_stabilizers = tuple(
        stab
        for stab in new_boundary_stabilizers
        for evolution in interpretation_step.logical_x_evolution.values()
        for pauli_op_id in evolution
        if stab.uuid == pauli_op_id
    )
    future_z_stabilizers = tuple(
        stab
        for stab in new_boundary_stabilizers
        for evolution in interpretation_step.logical_z_evolution.values()
        for pauli_op_id in evolution
        if stab.uuid == pauli_op_id
    )
    future_stabilizers = (future_x_stabilizers, future_z_stabilizers)

    # C) SYNDROME CIRCUITS
    # C.1) Create the mock block that describes the final geometry
    mock_block = RotatedSurfaceCode(
        unique_label=block.unique_label,
        stabilizers=new_stabilizers,
        logical_x_operators=[new_x_op],
        logical_z_operators=[new_z_op],
    )

    # C.2) Select the starting qubit diagonal direction
    # By default use TOP-RIGHT direction.
    # For type 2 and 3, the starting qubit should be away from the new boundary and
    # towards the move boundary in the next move corner to ensure fault-tolerance
    config, pivot_corners = mock_block.config_and_pivot_corners
    starting_diag_direction = DiagonalDirection.TOP_RIGHT
    if config in (2, 3):
        # Type 2 and 3 (U and L)
        # Extract geometry information
        is_move_direction_horizontal = (
            move_direction.to_orientation() == Orientation.HORIZONTAL
        )
        is_long_edge_horizontal = mock_block.size[0] > mock_block.size[1]
        is_move_direction_long_edge = (
            is_move_direction_horizontal == is_long_edge_horizontal
        )
        long_end_corner = pivot_corners[0]
        short_end_corner = pivot_corners[3]

        # Comparison of these two corners will determine the starting diagonal direction
        top_left_corner = mock_block.upper_left_qubit
        compare_corner = (
            long_end_corner if is_move_direction_long_edge else short_end_corner
        )

        hor_direction = (
            Direction.RIGHT
            if compare_corner[0] > top_left_corner[0]
            else Direction.LEFT
        )
        vert_direction = (
            Direction.BOTTOM
            if compare_corner[1] > top_left_corner[1]
            else Direction.TOP
        )
        starting_diag_direction = DiagonalDirection.from_directions(
            (hor_direction, vert_direction)
        )

    #   C.2) Find syndrome circuits using the mock block
    new_syndrome_circuits, new_stab_to_circuit = generate_syndrome_extraction_circuits(
        mock_block, starting_diag_direction
    )

    #   C.3) Get the cut corner syndrome circuit (not calculated by the function above)
    if isinstance(cut_stabilizer, Stabilizer):
        cut_syndrome_circuit = cut_corner_syndrome_circuit(
            block, cut_stabilizer, corner_qubit, corner_directions
        )
        new_syndrome_circuits += (cut_syndrome_circuit,)
        new_stab_to_circuit.update({cut_stabilizer.uuid: cut_syndrome_circuit.uuid})

    # D) CIRCUIT
    #   D.1) Measure the corner if needed
    circuit_output = move_corner_circuit(
        interpretation_step=interpretation_step,
        block=block,
        corner_qubit=corner_qubit,
        cut_stabilizer=cut_stabilizer,
        move_direction=move_direction,
        how_far=how_far,
    )
    if circuit_output is not None:
        corner_circuit, new_cbit = circuit_output
        interpretation_step.append_circuit_MUT(corner_circuit)
        #   D.2) Create stabilizer and logical operator updates
        interpretation_step.stabilizer_updates[cut_stabilizer.uuid] = (new_cbit,)
        if corner_qubit in block.logical_x_operators[0].data_qubits:
            interpretation_step.logical_x_operator_updates[new_x_op.uuid] = (new_cbit,)
        if corner_qubit in block.logical_z_operators[0].data_qubits:
            interpretation_step.logical_z_operator_updates[new_z_op.uuid] = (new_cbit,)

    syndrome_circuits_in_used = tuple(new_stab_to_circuit.values())
    new_syndrome_circuits = tuple(
        syndrome_circuit
        for syndrome_circuit in new_syndrome_circuits
        if syndrome_circuit.uuid in syndrome_circuits_in_used
    )

    # E) BUILD BLOCK
    new_block = RotatedSurfaceCode(
        unique_label=block.unique_label,
        stabilizers=new_stabilizers,
        logical_x_operators=[new_x_op],
        logical_z_operators=[new_z_op],
        syndrome_circuits=new_syndrome_circuits,
        stabilizer_to_circuit=new_stab_to_circuit,
    )
    interpretation_step.update_block_history_and_evolution_MUT(
        new_blocks=(new_block,), old_blocks=(block,)
    )

    return interpretation_step, past_stabilizers, future_stabilizers


def get_associated_boundaries(
    block: RotatedSurfaceCode,
    corner_qubit: tuple[int, int, int],
) -> tuple[Direction, ...]:
    """Get the boundary direction(s) associated with the corner qubit

    Parameters
    ----------
    block : RotatedSurfaceCode
        Block of rotated surface code
    corner_qubit : tuple[int, int, int]
        Corner qubit

    Returns
    -------
    tuple[Direction, ...]
        Boundary direction(s) associated with the corner qubit

    Raises
    ------
    ValueError
        If the corner given is not a topological corner of the rotated surface code
        block, i.e. the product of stabilizers at this point is not Y.
    """

    # Check if the corner is a topological corner
    if corner_qubit not in block.topological_corners:
        raise ValueError(
            f"The selected corner qubit {corner_qubit} is not a topological corner of"
            + f" the block `{block.unique_label}`"
        )

    # Find the boundaries associated with the corner
    corner_directions = tuple(
        direction
        for direction in Direction
        if corner_qubit in block.boundary_qubits(direction)
    )
    # Should not happen unless there are defects in the block, our assumptions break
    # down here
    if len(corner_directions) == 0:
        raise ValueError(
            f"The corner qubit {corner_qubit} is not associated with any boundary"
        )

    return corner_directions


def find_new_boundary_direction(
    corner_boundaries: tuple[Direction, ...],
    move_direction: Direction,
) -> Direction:
    """Find the modified boundary direction that corresponds to moving the corner in the
    specified direction. E.g. moving the top left corner towards the right modifies the
    top boundary.

    Parameters
    ----------
    corner_boundaries : tuple[Direction, ...]
        Description of the corner in terms of the geometric directions. E.g. (TOP, LEFT)
        describes the top left corner. A topological corner can be included in a single
        boundary or two boundaries.
    move_direction : Direction
        Direction in which to move the corner

    Returns
    -------
    Direction
        Direction of the boundary that will be modified
    """
    if (n_boundaries := len(corner_boundaries)) not in (1, 2):
        raise ValueError(
            f"Invalid number of corner boundaries: {n_boundaries}, "
            "must be either 1 or 2"
        )
    if set(corner_boundaries) in (
        {Direction.TOP, Direction.BOTTOM},
        {Direction.LEFT, Direction.RIGHT},
    ):
        raise ValueError(
            f"Invalid corner boundaries: {corner_boundaries}, they must be "
            + "orthogonal, e.g. (TOP, LEFT) or a single direction."
        )
    if move_direction in corner_boundaries:
        raise ValueError(
            f"Cannot move the corner {corner_boundaries} towards the "
            f"{move_direction.value}"
        )

    return next(
        direction
        for direction in corner_boundaries
        if direction != move_direction.opposite()
    )


def is_2_body_included(
    block: RotatedSurfaceCode,
    corner_qubit: tuple[int, int, int],
    boundary: Direction,
) -> bool:
    """Check if the corner qubit is included in a 2-body stabilizer that is part of the
    chosen boundary.

    Parameters
    ----------
    block : RotatedSurfaceCode
        Initial block
    corner_qubit : tuple[int, int, int]
        Corner qubit to check
    boundary : Direction
        Boundary to check

    Returns
    -------
    bool
        True if the corner qubit is included in a 2-body stabilizer that is part of the
        chosen boundary. False otherwise.
    """
    if corner_qubit not in (boundary_qubits := block.boundary_qubits(boundary)):
        raise ValueError(
            f"The corner qubit {corner_qubit} is not part of the {boundary.value} "
            f"boundary"
        )
    return any(
        stab
        for stab in block.stabilizers
        if corner_qubit in stab.data_qubits
        and all(q in boundary_qubits for q in stab.data_qubits)
    )


def cut_corner_stabilizer(
    block: RotatedSurfaceCode,
    corner_qubit: tuple[int, int, int],
) -> Stabilizer:
    """Create a new 3-body stabilizer by cutting the corner qubit from initial 4-body
    stabilizer.

    Parameters
    ----------
    block : RotatedSurfaceCode
        Initial block
    corner_qubit : tuple[int, int, int]
        Corner qubit that is cut

    Returns
    -------
    Stabilizer
        New stabilizer with the corner qubit cut
    """
    # We assume that the corner is indeed a corner
    stab_to_cut = next(
        stab
        for stab in block.stabilizers
        if corner_qubit in stab.data_qubits and len(stab.data_qubits) == 4
    )
    # Keep the order of the data qubits consistent with the original stabilizer
    new_pauli, new_qubits = zip(
        *(
            (pauli, qubit)
            for pauli, qubit in zip(
                stab_to_cut.pauli, stab_to_cut.data_qubits, strict=True
            )
            if qubit != corner_qubit
        ),
        strict=True,
    )
    corner_stabilizer = Stabilizer(
        pauli="".join(new_pauli),
        data_qubits=new_qubits,
        ancilla_qubits=stab_to_cut.ancilla_qubits,
    )
    return corner_stabilizer


def cut_corner_syndrome_circuit(
    block: RotatedSurfaceCode,
    cut_stabilizer: Stabilizer,
    corner_qubit: tuple[int, int, int],
    which_corner: tuple[Direction, Direction],
) -> SyndromeCircuit:
    """Cut the corner qubit from the syndrome circuit that measures the stabilizer to
    cut. Note that the name of the SyndromeCircuit is always
    vertical_dir-horizontal_dir-pauli.

    Parameters
    ----------
    block: RotatedSurfaceCode
        Initial block
    cut_stabilizer: Stabilizer
        Stabilizer with the corner qubit cut out
    corner_qubit : tuple[int, int, int]
        Corner qubit to cut
    which_corner : tuple[Direction, Direction]
        Position of the corner qubit to cut

    Returns
    -------
    SyndromeCircuit
        Syndrome circuit with the corner qubit cut
    """
    stab_to_cut = next(
        stab for stab in block.bulk_stabilizers if corner_qubit in stab.data_qubits
    )

    # Order the directions
    vertical_direction = next(
        dir for dir in which_corner if dir in {Direction.TOP, Direction.BOTTOM}
    )
    horizontal_direction = next(
        dir for dir in which_corner if dir in {Direction.LEFT, Direction.RIGHT}
    )

    # Create a syndrome circuit with the corner qubit cut
    cut_syndrome_circuit = RotatedSurfaceCode.generate_syndrome_circuit(
        pauli=cut_stabilizer.pauli,
        padding=[stab_to_cut.data_qubits.index(corner_qubit)],
        name=f"{vertical_direction}-{horizontal_direction}-{cut_stabilizer.pauli}",
    )

    return cut_syndrome_circuit


def generate_updated_2_body_stabilizers(
    old_corner_qubit: tuple[int, int, int],
    new_boundary_direction: Direction,
    unit_vector: tuple[int, int],
    how_far: int,
    pauli_type: str,
) -> list[Stabilizer]:
    """Generate the new boundary stabilizers that are created when moving the corner.

    Parameters
    ----------
    old_corner_qubit : tuple[int, int, int]
        Old corner qubit
    new_boundary_direction : Direction
        Direction of the modified boundary
    unit_vector : tuple[int, int]
        Unit vector of the movement
    how_far : int
        Distance by which the corner is moved
    pauli_type : str
        Pauli type of the new stabilizers

    Returns
    -------
    list[Stabilizer]
        List of new stabilizers
    """
    if (abs(unit_vector[0]), abs(unit_vector[1])) not in ((1, 0), (0, 1)):
        raise ValueError("Invalid unit vector")
    orientation = Orientation.from_vector(unit_vector)

    is_bottom_or_right = new_boundary_direction in (Direction.RIGHT, Direction.BOTTOM)

    adjusted_old_corner_qubit = (
        old_corner_qubit[0] + unit_vector[0] * how_far % 2,  # If odd, the corner is cut
        old_corner_qubit[1] + unit_vector[1] * how_far % 2,  # If odd, the corner is cut
        0,
    )
    final_stab_position = (
        old_corner_qubit[0] + unit_vector[0] * (how_far - 1),
        old_corner_qubit[1] + unit_vector[1] * (how_far - 1),
        0,
    )
    # We create the stabilizer from top to bottom or left to right
    initial_position = min(
        (adjusted_old_corner_qubit, final_stab_position), key=lambda x: x[0] + x[1]
    )

    new_stabs = RotatedSurfaceCode.generate_weight2_stabs(
        pauli=pauli_type * 2,
        initial_position=initial_position,
        num_stabs=how_far // 2,
        orientation=orientation,
        is_bottom_or_right=is_bottom_or_right,
    )
    return new_stabs


def find_new_boundary_stabilizers(
    block: RotatedSurfaceCode,
    corner_qubit: tuple[int, int, int],
    modified_boundary_direction: Direction,
    unit_vector: tuple[int, int],
    how_far: int,
    two_body_is_included: bool,
) -> tuple[list[Stabilizer], list[Stabilizer], Stabilizer | None, dict[str, str]]:
    """Finds the stabilizers of the new modified boundary after moving the corner. Some
    stabilizers of the old boundary are not modified and returned as well. If a 2-body
    stabilizer is involved in the selected corner and the modified boundary, the corner
    stabilizer is cut and the stabilizer that is moved is returned.
    It returns, the kept boundaries stabilizers, the newly created ones, the cut bulk
    stabilizer and the stabilizer evolution.

    Parameters
    ----------
    block : RotatedSurfaceCode
        Initial block
    corner_qubit : tuple[int, int, int]
        Corner qubit to move
    modified_boundary_direction : Direction
        Direction of the modified boundary
    unit_vector : tuple[int, int]
        Unit vector describing the movement of the corner
    how_far : int
        Distance by which to move the corner
    two_body_is_included : bool
        Is there a two-body stabilizer involved in the corner and the modified boundary
        in the direction of movement.

    Returns
    -------
    tuple[list[Stabilizer], list[Stabilizer], Stabilizer | None, dict[str, str]]
        Kept boundaries stabilizers, new boundary stabilizers, cut stabilizer,
        stabilizer evolution.

    Raises
    ------
    ValueError
        If the distance is not odd when a 2-body stabilizer is involved or if the
        distance is not even when there is no 2-body stabilizer involved.
    """
    qubits_to_update = [
        (
            corner_qubit[0] + i * unit_vector[0],
            corner_qubit[1] + i * unit_vector[1],
            0,
        )
        for i in range(how_far + 1)  # Old corner and new corners are included
    ]
    kept_boundary_stabilizers = [
        stab
        for stab in block.all_boundary_stabilizers
        if not all(q in qubits_to_update for q in stab.data_qubits)
    ]
    # Check if a 2-body stabilizer is involved in the selected corner and the modified
    # boundary
    # If it is, we need to cut the geometric corner to move the topological corner
    if two_body_is_included:
        # Only odd distances are allowed
        if how_far % 2 == 0:
            raise ValueError(
                "Only odd distances are allowed for moving the corner in the direction"
                " of the 2-body stabilizer"
            )
        stab_to_cut = next(
            stab for stab in block.bulk_stabilizers if corner_qubit in stab.data_qubits
        )
        # Cut the corner
        cut_stabilizer = cut_corner_stabilizer(block, corner_qubit)

        stabilizer_evolution = {cut_stabilizer.uuid: (stab_to_cut.uuid,)}

    else:
        # Only even distances are allowed
        if how_far % 2 == 1:
            raise ValueError(
                "Only even distances are allowed for moving the corner in the direction"
                " where there is no 2-body stabilizer"
            )

        cut_stabilizer = None
        stabilizer_evolution = {}
        # Just change the 2-body stabilizers

    try:
        new_boundary_stabilizers = generate_updated_2_body_stabilizers(
            corner_qubit,
            modified_boundary_direction,
            unit_vector,
            how_far,
            block.boundary_type(modified_boundary_direction),
        )
    except RuntimeError:
        # If there is more than 1 boundary type
        # Do this to get the exact boundary type.
        boundary_pauli_charges = list(
            set(block.pauli_charges[each_qubit] for each_qubit in qubits_to_update)
            - set("Y")
        )
        new_boundary_stabilizers = generate_updated_2_body_stabilizers(
            corner_qubit,
            modified_boundary_direction,
            unit_vector,
            how_far,
            boundary_pauli_charges[0],
        )

    return (
        kept_boundary_stabilizers,
        new_boundary_stabilizers,
        cut_stabilizer,
        stabilizer_evolution,
    )


def move_corner_logical_operators(  # pylint: disable=too-many-locals
    block: RotatedSurfaceCode,
    corner_qubit: tuple[int, int, int],
    unit_vector: tuple[int, int],
    how_far: int,
    old_stabs_to_be_removed: list[Stabilizer],
    new_stabs_to_be_added: list[Stabilizer],
) -> tuple[PauliOperator, PauliOperator, dict[str, str], dict[str, str]]:
    """Modify the logical operators of the block after moving the corner. There is no
    guarantee that the distance of the block is preserved after moving the corners,
    therefore no guarantee on the length of the operators.
    Note that the logical operator is not modified if the movement does not affect the
    commutation properties, it may not be of minimal weight anymore. The distance of
    the code may not be preserved.

    Parameters
    ----------
    block : RotatedSurfaceCode
        Initial block
    corner_qubit : tuple[int, int, int]
        Corner qubit to move
    unit_vector : tuple[int, int]
        Unit vector describing the movement of the corner
    how_far : int
        Distance by which to move the corner
    old_stabs_to_be_removed : list[Stabilizer]
        Stabilizers that are removed by the corner movement.
    new_stabs_to_be_added : list[Stabilizer]
        Stabilizers that are added by the corner movement.

    Returns
    -------
    tuple[PauliOperator, PauliOperator, dict[str, str], dict[str, str]]
        New logical X operator, new logical Z operator, logical X evolution dictionary,
        logical Z evolution dictionary.
    """
    qubits_to_update = [
        (
            corner_qubit[0] + i * unit_vector[0],
            corner_qubit[1] + i * unit_vector[1],
            0,
        )
        for i in range(how_far + 1)  # Both new and old corners are included
    ]

    # Represent a product of pauli P1*P2 = pauli_product[P1][P2] (up to phase)
    pauli_product = {
        "X": {"X": "I", "Z": "Y", "Y": "Z", "I": "X"},
        "Z": {"X": "Y", "Z": "I", "Y": "X", "I": "Z"},
        "Y": {"X": "Z", "Z": "X", "Y": "I", "I": "Y"},
        "I": {"X": "X", "Z": "Z", "Y": "Y", "I": "I"},
    }

    # Make sure we are working with a single logical qubit
    if len(block.logical_x_operators) > 1 or len(block.logical_z_operators) > 1:
        raise ValueError("Only a single logical qubit is supported")
    # NOTE we do not support Y paulis

    # Update the logical operators
    old_x_op = block.logical_x_operators[0]
    old_z_op = block.logical_z_operators[0]

    # Case 1 - The logical operator is not affected if the qubits are not included in
    # the movement.
    # Case 2 : All qubits to be modified are included in the operator: we shorten
    # the operator but make sure the new corner (qubits_to_update[-1]) is included.
    # Case 3 - The operator is on the boundary and needs to be modified. We use the
    # anchor (see inline documentation below)on the boundary to determine where to
    # modify the operator and drag the operator to the new topological corner.
    # Refer to the tests for examples of each case.

    new_ops, log_evolution_dicts = [], []
    for old_op in (old_x_op, old_z_op):
        logical_op_dict = {
            q: p for p, q in zip(old_op.pauli, old_op.data_qubits, strict=True)
        }
        qubits_included_op = set(qubits_to_update).intersection(set(old_op.data_qubits))
        # Case 1: The logical operator is not affected
        if len(qubits_included_op) == 0:
            new_op = old_op
            log_evolution_dict = {}
        # Case 2 : All qubits to be modified are included in the operator: we shorten
        # the operator but make sure the new corner (qubits_to_update[-1]) is included
        elif set(qubits_to_update) == qubits_included_op:
            new_op = PauliOperator(
                pauli="".join(
                    p
                    for (q, p) in logical_op_dict.items()
                    if q not in qubits_to_update[:-1]
                ),
                data_qubits=tuple(
                    q
                    for (q, p) in logical_op_dict.items()
                    if q not in qubits_to_update[:-1]
                ),
            )
            log_evolution_dict = {
                new_op.uuid: (old_op.uuid,)
                + tuple(stab.uuid for stab in new_stabs_to_be_added)
            }
        # Case 3: The operator is on the boundary and needs to be modified, we find
        # the anchor and "drag" the operator to the new corner
        else:
            # Find the anchor on the boundary
            if corner_qubit in old_op.data_qubits:
                # The anchor is either the old corner
                anchor = corner_qubit
            else:
                # Or the first qubit that breaks the commutation relation with the new
                # stabilizers, please refer to tests for an example
                old_op_data_qubits_set = set(old_op.data_qubits)
                intersection_with_stabs = [
                    set(stab.data_qubits).intersection(old_op_data_qubits_set)
                    for stab in new_stabs_to_be_added
                ]
                anchor = next(
                    next(iter(intersection))  # We reasonably assume a single anchor
                    for intersection in intersection_with_stabs
                    if len(intersection) == 1
                )
            old_stab_dict = {
                q: p
                for stab in old_stabs_to_be_removed
                for p, q in zip(stab.pauli, stab.data_qubits, strict=True)
            }
            # The operator is not modified until we touch its anchor
            start_modify_index = qubits_to_update.index(anchor)
            # The operator is only modified by the old stabilizers located after the
            # anchor
            included_old_stabs = [
                stab
                for stab in old_stabs_to_be_removed
                if all(
                    q in qubits_to_update[start_modify_index:] for q in stab.data_qubits
                )
            ]
            # Modify the operator for each qubit that is updated
            for qubit in qubits_to_update[start_modify_index:]:
                # Create a dictionary of the logical operator including new qubits
                if qubit not in logical_op_dict.keys():
                    logical_op_dict[qubit] = "I"
                # Multiply the operator by the old stabilizers that are removed
                if qubit in old_stab_dict.keys():
                    logical_op_dict[qubit] = pauli_product[logical_op_dict[qubit]][
                        old_stab_dict[qubit]
                    ]
            new_op = PauliOperator(
                pauli="".join(p for p in logical_op_dict.values() if p != "I"),
                data_qubits=tuple(q for (q, p) in logical_op_dict.items() if p != "I"),
            )
            log_evolution_dict = {
                new_op.uuid: (old_op.uuid,)
                + tuple(stab.uuid for stab in included_old_stabs)
            }
        new_ops.append(new_op)
        log_evolution_dicts.append(log_evolution_dict)

    new_x_op, new_z_op = new_ops  # pylint: disable=unbalanced-tuple-unpacking
    # pylint: disable=unbalanced-tuple-unpacking
    log_x_evolution_dict, log_z_evolution_dict = log_evolution_dicts

    return new_x_op, new_z_op, log_x_evolution_dict, log_z_evolution_dict


def move_corner_circuit(
    interpretation_step: InterpretationStep,
    block: RotatedSurfaceCode,
    corner_qubit: tuple[int, int, int],
    cut_stabilizer: Stabilizer | None,
    move_direction: Direction,
    how_far: int,
):
    """Create the circuit that measures out the corner if needed. Returns the
    measurement circuit and the cbit that contains the measurement of the corner or
    None if the corner isn't cut.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        Interpretation step before moving the corner
    block : RotatedSurfaceCode
        The initial block
    corner_qubit : tuple[int, int, int],
        Corner to move
    cut_stabilizer : Stabilizer | None
        Stabilizer after being cut (only if a 2-body is involved in the movement, else
        None)
    move_direction : Direction
        Direction in which to move the corner
    how_far : int
        Distance by which to move the corner

    Returns
    -------
    tuple[Circuit, Cbit] | None
        Circuit measuring the corner qubit and Cbit that contains the corner measurement
        or None if the corner isn't cut.
    """
    if isinstance(cut_stabilizer, Stabilizer):
        stab_to_cut = next(
            stab for stab in block.bulk_stabilizers if corner_qubit in stab.data_qubits
        )
        pauli = next(
            p
            for p, q in zip(stab_to_cut.pauli, stab_to_cut.data_qubits, strict=True)
            if q == corner_qubit
        )
        measure_basis = "" if pauli == "Z" else f"_{pauli}"
        q_chan = interpretation_step.get_channel_MUT(str(corner_qubit))
        c_chan = interpretation_step.get_channel_MUT(
            f"c_{corner_qubit}", ChannelType.CLASSICAL
        )
        cbit = interpretation_step.get_new_cbit_MUT(f"c_{corner_qubit}")
        circuit = Circuit(
            name=f"moving corner {corner_qubit} to {move_direction} by {how_far}",
            circuit=((Circuit(f"measure{measure_basis}", channels=[q_chan, c_chan]),),),
        )

        return (circuit, cbit)

    return None
