"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from loom.eka import Circuit, Stabilizer, PauliOperator, SyndromeCircuit
from loom.eka.utilities import (
    Direction,
    Orientation,
)
from loom.interpreter import InterpretationStep

from ..code_factory import RotatedSurfaceCode
from .y_wall_out_circuit import get_y_wall_out_circ_sequence_and_updates


def y_wall_out(
    interpretation_step: InterpretationStep,
    block: RotatedSurfaceCode,
    wall_position: int,
    wall_orientation: Orientation,
    debug_mode: bool,
) -> InterpretationStep:
    """Implement the y_wall_out operation.

    The algorithm is the following:
    - A) CONSISTENCY CHECK
    - B) EXTRACT GEOMETRIC INFORMATION
        - B.1) Extract some information about the geometry of the block and the wall
        - B.2) Find the qubits to measure
        - B.3) Find the qubits to idle
        - B.4) Find the qubits to Hadamard
    - C) LOGICAL OPERATORS
        - C.1) Find where the X logical operator should be depending on the geometry
        - C.2) Create the new X logical operator
        - C.3) Update the logical operator evolution using appropriate stabilizers
    - D) STABILIZERS
        - D.1) Find the idling stabilizers
        - D.2) Find the stabilizers that will be Hadamard-ed and moved
        - D.3) Find the new hadamard stabilizers
        - D.4) Update the stabilizer evolution
        - D.5) Put the stabilizers together
    - E) SYNDROME CIRCUITS
        - E.1) Find the idling part syndrome circuits
        - E.2) Find the new syndrome circuits for the hadamard-ed stabilizers
    - F) NEW BLOCK
    - G) CIRCUIT AND LOGICAL/STABILIZER UPDATES

    Example: the block on the left is transformed into the block on the right:

                   X                                    X
           *(0,0) --- (1,0) --- (2,0)*          *(0,0) --- (1,0) --- (2,0)*
              |         |         |                |         |         |
              |    Z    |    X    |  Z             |    Z    |    X    |  Z
              |         |         |                |         |         |
            (0,1) --- (1,1) --- (2,1)            (0,1) --- (1,1) --- (2,1)
              |         |         |                |         |         |
           Z  |    X    |    Z    |             Z  |    X    |    Z    |
              |         |         |                |         |         |
            (0,2) --- (1,2) --- (2,2)     ->     (0,2) --- (1,2) --- (2,2)*
              |         |         |                |         |         |
              |    Z    |    X    |  Z             |    Z    |    X    |
              |         |         |                |         |         |
           *(0,3) --- (1,3) --- (2,3)            (0,3) --- (1,3) --- (2,3)
              |         |         |                |         |         |
              |    X    |    Z    |             Z  |    X    |    Z    |  X
              |         |         |                |         |         |
            (0,4) --- (1,4) --- (2,4)            (0,4) --- (1,4) --- (2,4)*
              |         |         |                     Z
           X  |    Z    |    X    |  Z
              |         |         |
            (0,5) --- (1,5) --- (2,5)*
                   X

    Other allowed blocks are reflection along a vertical axis and rotation by 90
    degrees. More examples in tests.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        The interpretation step containing the blocks involved in the operation.
    block : RotatedSurfaceCode
        The block to which the operation will be applied.
    wall_position : int
        The position of the wall.
    wall_orientation : Orientation
        The orientation of the wall.
    debug_mode : bool
        Flag indicating whether the interpretation should be done in debug mode.
        Activating debug mode will enable commutation validation for Block.

    Returns
    -------
    InterpretationStep
        The interpretation step after applying the y_wall_out operation.
    """

    # A) CONSISTENCY CHECK
    y_wall_out_consistency_check(block, wall_position, wall_orientation)

    # B) EXTRACT GEOMETRIC INFORMATION
    # B.1) Extract some information about the geometry of the block and the wall
    is_wall_hor = wall_orientation == Orientation.HORIZONTAL
    is_block_horizontal = wall_orientation.perpendicular() == Orientation.HORIZONTAL
    is_top_left_bulk_stab_x = block.upper_left_4body_stabilizer.pauli[0] == "X"

    # B.2) Find the qubits to measure
    # B.3) Find the qubits to idle
    # B.4) Find the qubits to Hadamard
    qubits_to_measure, qubits_to_idle, qubits_to_hadamard = find_qubit_sets(
        block, wall_position, is_wall_hor
    )

    # C) LOGICAL OPERATORS
    # C.1) Find where the X logical operator should be depending on the geometry
    # C.2) Create the new X logical operator
    # C.3) Find appropriate stabilizers for the logical operator evolution
    new_x_logical_operator, stabilizers_for_x_operator_jump = (
        find_new_x_logical_operator(
            block, is_block_horizontal, is_top_left_bulk_stab_x, qubits_to_idle
        )
    )
    # C.4) Update the logical operator evolution
    interpretation_step.logical_x_evolution[new_x_logical_operator.uuid] = (
        block.logical_x_operators[0].uuid,
    ) + tuple(stab.uuid for stab in stabilizers_for_x_operator_jump)

    # D) STABILIZERS
    # D.1) Find the idling stabilizers
    # D.2) Find the stabilizers that will be Hadamard-ed and moved
    # D.3) Find the new hadamard stabilizers
    # D.4) Find the stabilizer evolution
    idle_stabilizers, new_hadamard_stabilizers, y_wall_out_stabilizer_evolution = (
        find_stabilizer_evolution(
            block,
            is_block_horizontal,
            qubits_to_measure,
            qubits_to_idle,
            qubits_to_hadamard,
        )
    )

    # D.5) Put the stabilizers together
    new_stabilizers = idle_stabilizers + new_hadamard_stabilizers

    # D.6) Update the stabilizer evolution
    interpretation_step.stabilizer_evolution.update(y_wall_out_stabilizer_evolution)

    # E) SYNDROME CIRCUITS
    # E.1) Find the idling part syndrome circuits
    idle_circ_tuple = ()
    idle_stabilizer_to_circ = {}
    for stab in idle_stabilizers:
        synd_circ = next(
            circ
            for circ in block.syndrome_circuits
            if block.stabilizer_to_circuit[stab.uuid] == circ.uuid
        )
        idle_stabilizer_to_circ[stab.uuid] = synd_circ.uuid
        if synd_circ not in idle_circ_tuple:
            idle_circ_tuple += (synd_circ,)

    # E.2) Find the new syndrome circuits for the hadamard-ed stabilizers
    # New syndrome circuit tuple contains the new circuits along with previous ones
    # associated with idling stabilizers
    new_synd_circ_tuple, new_stabilizer_to_circuit = hadamard_side_syndrome_circuits(
        block,
        new_hadamard_stabilizers,
        idle_stabilizer_to_circ,
        idle_circ_tuple,
    )

    # F) NEW BLOCK
    new_block = RotatedSurfaceCode(
        stabilizers=new_stabilizers,
        logical_x_operators=(new_x_logical_operator,),
        logical_z_operators=block.logical_z_operators,
        syndrome_circuits=new_synd_circ_tuple,
        stabilizer_to_circuit=new_stabilizer_to_circuit,
        unique_label=block.unique_label,
        skip_validation=not debug_mode,
    )

    interpretation_step.update_block_history_and_evolution_MUT((new_block,), (block,))

    # G) CIRCUIT AND LOGICAL/STABILIZER UPDATES
    circuit_seq = get_y_wall_out_circ_sequence_and_updates(
        block,
        new_block,
        is_top_left_bulk_stab_x,
        qubits_to_hadamard,
        qubits_to_idle,
        qubits_to_measure,
        stabilizers_for_x_operator_jump,
        y_wall_out_stabilizer_evolution,
        interpretation_step,
    )

    # Make the circuit sequence into a Circuit object and append it to the
    # interpretation step
    interpretation_step.append_circuit_MUT(
        Circuit(
            name=f"y_wall_out operation on block {block.unique_label}",
            circuit=Circuit.construct_padded_circuit_time_sequence(
                circuit_seq,
            ),
        )
    )

    return interpretation_step


def y_wall_out_consistency_check(
    block: RotatedSurfaceCode,
    wall_pos: int,
    wall_orientation: Orientation,
) -> None:
    """
    Check if the y_wall_out operation can be applied to a given block. Note that there
    are 4 different cases defined by:
    - the orientation of the block
    - whether the top-left bulk stabilizer is X or Z

    Parameters
    ----------
    block: RotatedSurfaceCode
        The block to which the operation will be applied.
    wall_pos: int
        The position of the wall.
    wall_orientation: Orientation
        The orientation of the wall.

    Raises
    ------
    ValueError
        If the block dimensions are not valid.
        If the wall position is not valid.
        If the block and the wall do not have perpendicular orientations.
        If the left boundary of a horizontal block is not a Z-type boundary.
        If the top boundary of a vertical block is not a Z-type boundary.
        If the block does not have 3 topological corners located at the geometric
        corners.
        If the missing geometric corner is not the expected one.
        If the non-geometric topological corner is not the expected one.
        If the Z logical operator is not located at the expected position.
        If the X logical operator is not located at the expected position.
    """

    # Extract some information about the geometry of the block and the wall
    is_wall_hor = wall_orientation == Orientation.HORIZONTAL
    topological_corners = block.topological_corners
    geometric_corners = block.geometric_corners
    u_l_qub = block.upper_left_qubit
    is_u_l_stab_x = block.upper_left_4body_stabilizer.pauli[0] == "X"
    dim_x, dim_z = block.size
    larger_dim = max(dim_x, dim_z)
    smaller_dim = min(dim_x, dim_z)

    # Check the block dimensions
    if smaller_dim % 2 == 0 or larger_dim % 2 != 0:
        raise ValueError(
            "The smaller dimension of the block must be odd and the larger dimension "
            "must be even."
        )

    # Check block orientation and wall orientation
    is_block_hor = dim_x == larger_dim
    if is_block_hor == is_wall_hor:
        raise ValueError("The block and the wall must have perpendicular orientations.")

    if is_block_hor:
        if block.boundary_type("left") != "Z":
            raise ValueError(
                "The left boundary of a horizontal block must be a Z-type boundary."
            )
    else:
        if block.boundary_type("top") != "Z":
            raise ValueError(
                "The top boundary of a vertical block must be a Z-type boundary."
            )

    # Check the wall position
    if wall_pos != smaller_dim:
        raise ValueError(
            "The wall position must be such that the block on the bottom/right side "
            "of the wall is a square."
        )

    # Check the corners
    common_corners = set(topological_corners) & set(geometric_corners)
    # There should be 3 topological corners located at the geometric corners
    if len(common_corners) != 3:
        raise ValueError(
            "The block must have 3 topological corners located at the geometric "
            "corners."
        )

    # Find which topological corner is not geometric and which geometric corner is not
    # topological
    geom_corner_not_topol = next(iter(set(geometric_corners) - common_corners))
    topol_corner_not_geom = next(iter(set(topological_corners) - common_corners))

    # Find the expected non-geometric topological corner:
    # s: start of the block (left/top), e: end of the block(right/bot), w: wall_position
    # If the block is horizontal:
    #   If the block has the top left stabilizer as X:
    #      (e, w)
    #   else:
    #      (s, w)
    # else:
    #   If the block has the top left stabilizer as X:
    #      (w, e)
    #   else:
    #      (w, s)
    exp_topol_corner_not_geom = (
        u_l_qub[0] + (wall_pos if is_block_hor else is_u_l_stab_x * (smaller_dim - 1)),
        u_l_qub[1] + (is_u_l_stab_x * (smaller_dim - 1) if is_block_hor else wall_pos),
        0,
    )

    if exp_topol_corner_not_geom != topol_corner_not_geom:
        raise ValueError(
            f"The non-geometric topological corner should have been "
            f"{exp_topol_corner_not_geom} but it is {topol_corner_not_geom}."
        )

    # Find the expected non-topological geometric corner:
    # If the top left stabilizer is X:
    #   it's the botton right corner i.e. maximize(x[0] + x[1])
    # else:
    #   If the block is horizontal:
    #      it's the top right corner i.e. maximize(x[0] - x[1])
    #   else:
    #      it's the bottom left corner i.e. maximize(-x[0] + x[1])
    exp_geom_corner_not_topol = max(
        geometric_corners,
        key=lambda x: (
            x[0] + x[1] if is_u_l_stab_x else (-x[0] + x[1]) * (-1) ** (is_block_hor)
        ),
    )

    if exp_geom_corner_not_topol != geom_corner_not_topol:
        raise ValueError(
            f"The non-topological geometric corner should have been "
            f"{exp_geom_corner_not_topol} but it is {geom_corner_not_topol}."
        )

    # Check that the logical operators are located in the correct positions
    expected_z_logical = PauliOperator(
        "Z" * smaller_dim,
        block.boundary_qubits("left") if is_block_hor else block.boundary_qubits("top"),
    )
    if block.logical_z_operators[0] != expected_z_logical:
        raise ValueError(
            "The Z logical operator is not located at the expected position. It needs "
            "to be on the left for a horizontal block and on the top for a vertical "
            "block."
        )

    # The X logical operator suffices to be straight and with the right distance
    # given the geometry of the block
    x_log_data_qubits = block.logical_x_operators[0].data_qubits
    is_x_logical_vert = all(q[0] == x_log_data_qubits[0][0] for q in x_log_data_qubits)
    is_x_logical_hor = all(q[1] == x_log_data_qubits[0][1] for q in x_log_data_qubits)
    is_x_straight = is_x_logical_vert or is_x_logical_hor
    is_x_correct_distance = len(x_log_data_qubits) == smaller_dim + 1
    if not (is_x_straight and is_x_correct_distance):
        raise ValueError(
            "The X logical operator is not located at the expected position. It needs "
            "to be straight and with the smallest distance possible given the geometry "
            "of the block."
        )


def find_qubit_sets(
    block: RotatedSurfaceCode, wall_position: int, is_wall_hor: bool
) -> tuple[list[tuple[int, int, int]], ...]:
    """
    Find the qubits to measure, to idle, and to Hadamard.

    Parameters
    ----------
    block : RotatedSurfaceCode
        The block to which the operation will be applied.
    wall_position : int
        The position of the wall.
    is_wall_hor : bool
        Whether the wall is horizontal or vertical.

    Returns
    -------
    tuple[list[tuple[int, int, int]], ...]
        The qubits to measure, to idle, and to Hadamard.
    """

    had_side_unit_vector = (0, 1) if is_wall_hor else (1, 0)
    idle_side_unit_vector = (0, -1) if is_wall_hor else (-1, 0)

    # B.2) Find the qubits to measure
    qubits_to_measure = (
        [(q[0], q[1] + wall_position, 0) for q in block.boundary_qubits("top")]
        if is_wall_hor
        else [(q[0] + wall_position, q[1], 0) for q in block.boundary_qubits("left")]
    )
    # B.3) Find the qubits to idle (top or left qubits depending on the orientation)
    qubits_to_idle = [
        (q[0] + idle_side_unit_vector[0] * d, q[1] + idle_side_unit_vector[1] * d, q[2])
        for q in qubits_to_measure
        for d in range(1, wall_position + 1)
    ]
    # B.4) Find the qubits to Hadamard
    # (bottom or right qubits depending on the orientation)
    qubits_to_hadamard = [
        (
            q[0] + had_side_unit_vector[0] * d,
            q[1] + had_side_unit_vector[1] * d,
            q[2],
        )
        for q in qubits_to_measure
        for d in range(1, max(block.size) - wall_position)
    ]

    return qubits_to_measure, qubits_to_idle, qubits_to_hadamard


def find_new_x_logical_operator(
    block: RotatedSurfaceCode,
    is_block_horizontal: bool,
    is_top_left_bulk_stab_x: bool,
    qubits_to_idle: list[tuple[int, int, int]],
) -> tuple[PauliOperator, tuple[Stabilizer, ...]]:
    """
    Find the new X logical operator and the stabilizers for the logical operator to jump
    across.

    Parameters
    ----------
    block : RotatedSurfaceCode
        The block to which the operation will be applied.
    is_block_horizontal : bool
        Whether the block is horizontal or vertical.
    is_top_left_bulk_stab_x : bool
        Whether the top left bulk stabilizer is X or Z.
    qubits_to_idle : list[tuple[int, int, int]]
        The qubits to idle.

    Returns
    -------
    tuple[PauliOperator, tuple[Stabilizer, ...]]
        The new X logical operator and the stabilizers for the logical operator to jump
        across.
    """
    # C.1) Find where the X logical operator should be depending on the geometry
    # Note that the transversal hadamard operation makes the topological corner jump
    # across the block.
    # So even though for a vertical block with the top left stabilizer as Z, the
    # topological corner is mid-left of the block, after the transversal hadamard
    # operation, it will be mid-right of the block.
    match (is_block_horizontal, is_top_left_bulk_stab_x):
        case (False, False):
            x_log_op_side = Direction.RIGHT
        case (False, True):
            x_log_op_side = Direction.LEFT
        case (True, False):
            x_log_op_side = Direction.BOTTOM
        case (True, True):
            x_log_op_side = Direction.TOP

    # C.2) Create the new X logical operator
    # It should contain all boundary idling qubits on the side where the logical
    # operator has to be placed.
    x_log_qubits = [
        qub for qub in block.boundary_qubits(x_log_op_side) if qub in qubits_to_idle
    ]
    new_x_logical_operator = PauliOperator(
        pauli="X" * len(x_log_qubits),
        data_qubits=x_log_qubits,
    )

    # C.3) Find appropriate stabilizers for the logical operator evolution
    # The stabilizers for the logical operator to jump across requires ALL the
    # stabilizers of the block that contain the qubits to idle.
    stabilizers_for_x_operator_jump = tuple(
        stab
        for stab in block.stabilizers
        if set(stab.data_qubits).intersection(qubits_to_idle)
    )
    return new_x_logical_operator, stabilizers_for_x_operator_jump


def find_stabilizer_evolution(
    initial_block: RotatedSurfaceCode,
    is_block_horizontal: bool,
    qubits_to_measure: list[tuple[int, int, int]],
    qubits_to_idle: list[tuple[int, int, int]],
    qubits_to_hadamard: list[tuple[int, int, int]],
) -> tuple[list[Stabilizer], list[Stabilizer], dict[str, tuple[str, ...]]]:
    """
    Find the stabilizers that will be idled, Hadamard-ed and moved, and the stabilizer
    evolution.

    Parameters
    ----------
    initial_block : RotatedSurfaceCode
        The block to which the operation will be applied.
    is_block_horizontal : bool
        Whether the block is horizontal or vertical.
    qubits_to_measure : list[tuple[int, int, int]]
        The qubits to measure.
    qubits_to_idle : list[tuple[int, int, int]]
        The qubits to idle.
    qubits_to_hadamard : list[tuple[int, int, int]]
        The qubits to Hadamard.

    Returns
    -------
    tuple[list[Stabilizer], list[Stabilizer], dict[str, tuple[str, ...]]]
        The idling stabilizers, the new Hadamard stabilizers, and the stabilizer
        evolution.
    """
    # D.1) Find the idling stabilizers
    idle_stabilizers = [
        stab
        for stab in initial_block.stabilizers
        if set(stab.data_qubits).issubset(qubits_to_idle)
    ]

    # D.2) Find the stabilizers that will be Hadamard-ed and moved
    # Note that these stabilizers also include the qubits to measure since these qubits
    # are the boundary
    hadamard_stabilizers = [
        stab
        for stab in initial_block.stabilizers
        if set(stab.data_qubits).issubset(qubits_to_hadamard + qubits_to_measure)
    ]

    # D.3) Find the new hadamard stabilizers
    # If the block is horizontal, the hadamard stabilizers are moved to the left by 1
    # else they are moved to the top by 1
    had_trans = {"X": "Z", "Z": "X"}
    new_hadamard_stabilizers_with_unordered_data_qubits = [
        Stabilizer(
            data_qubits=[
                (
                    (q[0] - 1, q[1], q[2])
                    if is_block_horizontal
                    else (q[0], q[1] - 1, q[2])
                )
                for q in stab.data_qubits
            ],
            ancilla_qubits=[
                (
                    (a[0] - 1, a[1], a[2])
                    if is_block_horizontal
                    else (a[0], a[1] - 1, a[2])
                )
                for a in stab.ancilla_qubits
            ],
            pauli="".join(had_trans[each_pauli] for each_pauli in stab.pauli),
        )
        for stab in hadamard_stabilizers
    ]

    # Take all BULK stabilizers and put their data qubits in the same order as the
    # original stabilizers that were in the position same of the initial block.
    # This is needed because this affects the syndrome extraction circuit.
    new_hadamard_stabilizers = [
        Stabilizer(
            data_qubits=(
                next(
                    init_stab.data_qubits
                    for init_stab in initial_block.stabilizers
                    if set(init_stab.data_qubits) == set(stab.data_qubits)
                )
                if len(stab.data_qubits) == 4
                else stab.data_qubits
            ),
            ancilla_qubits=stab.ancilla_qubits,
            pauli=stab.pauli,
        )
        for stab in new_hadamard_stabilizers_with_unordered_data_qubits
    ]

    # D.4) Find the stabilizer evolution
    y_wall_out_stabilizer_evolution = {}
    for old_stab, new_stab in zip(hadamard_stabilizers, new_hadamard_stabilizers):
        y_wall_out_stabilizer_evolution[new_stab.uuid] = (old_stab.uuid,)

        # If it's a bulk stabilizer touching the wall, then we also need to put the
        # stabilizer on the other side of the wall in the evolution
        qubits_to_idle_overlap = set(new_stab.data_qubits).intersection(qubits_to_idle)
        bulk_stabilizer_touches_wall = (
            len(new_stab.pauli) == 4 and len(qubits_to_idle_overlap) > 0
        )

        if bulk_stabilizer_touches_wall:
            # Find the stabilizer on the other side of the wall
            wall_stabilizer = next(
                stab
                for stab in initial_block.stabilizers
                if set(stab.data_qubits) == set(new_stab.data_qubits)
            )
            # and add it to the evolution
            y_wall_out_stabilizer_evolution[new_stab.uuid] += (wall_stabilizer.uuid,)

    return (
        idle_stabilizers,
        new_hadamard_stabilizers,
        y_wall_out_stabilizer_evolution,
    )


def hadamard_side_syndrome_circuits(
    initial_block: RotatedSurfaceCode,
    new_hadamard_stabilizers: list[Stabilizer],
    idle_stabilizer_to_circ: dict[str, SyndromeCircuit],
    idle_circ_tuple: tuple[SyndromeCircuit],
) -> tuple[tuple[SyndromeCircuit], dict[str, str]]:
    """
    Associate all new stabilizers on the hadamard side with the new SyndromeCircuits.
    If a syndrome circuit does not already exist in the idling side SyndromeCircuits,
    then create it.

    Parameters
    ----------
    initial_block : RotatedSurfaceCode
        The block to which the operation will be applied.
    new_hadamard_stabilizers : list[Stabilizer]
        The new Hadamard stabilizers.
    idle_stabilizer_to_circ : dict[str, SyndromeCircuit]
        The mapping of idle stabilizers to their syndrome circuits.
    idle_circ_tuple : tuple[SyndromeCircuit]
        The tuple of idle syndrome circuits.

    Returns
    -------
    tuple[tuple[SyndromeCircuit], dict[str, str]]
        The new syndrome circuit tuple and the mapping of stabilizers to their
        corresponding circuits.
    """
    final_synd_circ_tuple = idle_circ_tuple
    final_stabilizer_to_circuit = idle_stabilizer_to_circ

    # Separate the bulk and boundary Hadamard stabilizers
    new_bulk_hadamard_stabilizers = [
        stab for stab in new_hadamard_stabilizers if len(stab.data_qubits) == 4
    ]
    new_boundary_hadamard_stabilizers = [
        stab
        for stab in new_hadamard_stabilizers
        if stab not in new_bulk_hadamard_stabilizers
    ]

    # Associate the new bulk Hadamard stabilizers with their corresponding syndrome
    # circuits
    for bulk_stab in new_bulk_hadamard_stabilizers:
        # Find the syndrome circuit for the bulk stabilizer
        synd_circ = next(
            circ for circ in idle_circ_tuple if circ.pauli == bulk_stab.pauli
        )
        # Add the new syndrome circuit to the interpretation step
        final_stabilizer_to_circuit[bulk_stab.uuid] = synd_circ.uuid

    # Find all new data qubits in the new stabilizers
    all_new_data_qubs = set(
        qubit for stab in new_bulk_hadamard_stabilizers for qubit in stab.data_qubits
    )
    for boundary_stab in new_boundary_hadamard_stabilizers:
        # Find the direction of the boundary stabilizer. We need to do it the following
        # way because we do not have access to the final_block yet.

        # Condition for the stabilizer to be on the RIGHT boundary
        if all(
            q_s[0] >= dq[0]
            for q_s in boundary_stab.data_qubits
            for dq in all_new_data_qubs
        ):
            boundary_direction = Direction.RIGHT
        # Condition for the stabilizer to be on the LEFT boundary
        elif all(
            q_s[0] <= dq[0]
            for q_s in boundary_stab.data_qubits
            for dq in all_new_data_qubs
        ):
            boundary_direction = Direction.LEFT
        # Condition for the stabilizer to be on the BOTTOM boundary
        elif all(
            q_s[1] >= dq[1]
            for q_s in boundary_stab.data_qubits
            for dq in all_new_data_qubs
        ):
            boundary_direction = Direction.BOTTOM
        # Condition for the stabilizer to be on the TOP boundary
        elif all(
            q_s[1] <= dq[1]
            for q_s in boundary_stab.data_qubits
            for dq in all_new_data_qubs
        ):
            boundary_direction = Direction.TOP
        else:
            raise ValueError(
                "The boundary stabilizer and the bulk stabilizers do not have the "
                "same orientation."
            )

        # Find the syndrome circuit name for the boundary stabilizer
        syndrome_circuit_name = (
            f"{boundary_direction.value}-{boundary_stab.pauli.lower()}"
        )
        match_syndrome_circuit = next(
            (
                syndrome_circuit
                for syndrome_circuit in final_synd_circ_tuple
                if syndrome_circuit.name == syndrome_circuit_name
            ),
            None,
        )

        # If it exists, we need to associate the stabilizer with the SyndromeCircuit
        if match_syndrome_circuit is not None:
            final_stabilizer_to_circuit[boundary_stab.uuid] = (
                match_syndrome_circuit.uuid
            )
        # If it does not exist, we need to create a new one
        else:
            padding = RotatedSurfaceCode.find_padding(
                boundary=boundary_direction,
                schedule=(
                    initial_block.weight_4_x_schedule
                    if boundary_stab.pauli == "XX"
                    else initial_block.weight_4_z_schedule
                ),
            )

            new_syndrome_circuit = RotatedSurfaceCode.generate_syndrome_circuit(
                pauli=boundary_stab.pauli,
                padding=padding,
                name=syndrome_circuit_name,
            )

            # Add the new syndrome circuit to the tuple
            final_synd_circ_tuple += (new_syndrome_circuit,)
            # Associate
            final_stabilizer_to_circuit[boundary_stab.uuid] = new_syndrome_circuit.uuid

    return final_synd_circ_tuple, final_stabilizer_to_circuit
