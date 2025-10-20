"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from __future__ import annotations
from enum import Enum

from loom.eka import Circuit, Stabilizer, SyndromeCircuit, PauliOperator
from loom.eka.utilities import Direction, DiagonalDirection
from ..utilities.enums import FourBodySchedule
from ..code_factory.rotated_surface_code import RotatedSurfaceCode


def add_vector(
    qubit: tuple[int, int, int], direction: Direction, length: int
) -> tuple[int, int, int]:
    """Add a direction vector to a qubit coordinate scaled by length. Used to
    calculate the new position of a corner after moving it."""
    direction_vect = direction.to_vector()
    qubit = list(qubit)
    for i, coord in enumerate(direction_vect):
        qubit[i] += coord * length
    return tuple(qubit)


def generate_syndrome_extraction_circuits(
    rsc_block: RotatedSurfaceCode, starting_qubit_diagonal_direction: DiagonalDirection
) -> tuple[tuple[SyndromeCircuit, ...], dict[str, str]]:
    """
    Given a rotated surface code block, generate syndrome extraction circuits for
    each stabilizer based on the starting diagonal direction. The generated syndrome
    circuits should lead to a fault-tolerant behavior if used appropriately (with an
    appropriate `starting_qubit_diagonal_direction`).

    Parameters
    ----------
    rsc_block: RotatedSurfaceCode
        A rotated surface code block.
    starting_qubit_diagonal_direction: DiagonalDirection
        The diagonal direction of the starting data qubit for each syndrome extraction
        circuit.

    Returns
    -------
    tuple[tuple[SyndromeCircuit, ...], dict[str, str]]:
        A tuple containing:
        - A tuple of generated syndrome extraction circuits.
        - A dictionary mapping stabilizer UUIDs to their corresponding syndrome
        circuit UUIDs.
    """
    # find stabilizer schedules
    (
        non_triangle_x_schedule,
        non_triangle_z_schedule,
        triangle_x_schedule,
        triangle_z_schedule,
    ) = find_schedules(rsc_block)

    # create syndrome circuits and new stabilizers with correct schedules
    (
        new_syndrome_circuits,
        new_stabilizer_to_circuit,
    ) = create_new_syndrome_circuits_with_known_schedules(
        rsc_block,
        non_triangle_x_schedule,
        non_triangle_z_schedule,
        triangle_x_schedule,
        triangle_z_schedule,
        starting_qubit_diagonal_direction,
    )

    return (
        new_syndrome_circuits,
        new_stabilizer_to_circuit,
    )


def find_schedules(rsc_block: RotatedSurfaceCode) -> tuple[
    FourBodySchedule,
    FourBodySchedule,
    FourBodySchedule | None,
    FourBodySchedule | None,
]:
    """
    Determine schedule for stabilizers of the block.
    For type 1 and 2 corner configuration (rectangle and U-shape), schedules can be
    inferred from boundary type of the short edge connecting two
    topological corners.
    Example::

        (type 1)    1-------4           (type 2)    1-------|
                    |       |                       |       |
                    |       |                       |       2
                    |       |                       |       |
                    2-------3                       3-------4

    For type 3 corner configuration (L-shape), draw a diagonal line through the
    middle-edge topological corner and the geometric corner that is not occupied by
    a topological corner. This line divides the block into two parts, one triangle
    part and one non-triangle part. Return schedules for X and Z-type stabilizers in
    these parts.
    Example: the line is drawn through topological corner 2 and geometric corner x::

        (type 3)    1-------x
                    |       |
                    2       |
                    |       |
                    3-------4

    The algorithm is the following:
    - A) Find the boundary type of the short boundary connecting two topological
        corners. For type 1, pick any short edge.
    - B) Find schedules
        - B.1) Boundary type is of opposite Pauli type to the logical operator
        running along the boundary.
        (e.g. boundary_type = "X" means logical Z running along the boundary)
        - B.2) For type 1 (rectangle) and type 2 (U shape), the opposite short edge
            has the same Pauli type. The schedule of stabilizers with the same Pauli
            type as the short edge is determined so that the propagation of the
            opposite Pauli type is perpendicular to the short edge. The schedule of
            the-opposite-type stabilizers is the opposite schedule of the above
            schedule.
        - B.3) For type 3 (L shape), the opposite short edge has the opposite Pauli
        type. Inside the triangle, the schedule for stabilizers with the same type
        as the short edge is determined. Then the schedule for the other type
        stabilizers inside the triangle is the opposite schedule. Outside the
        triangle, the schedules are the opposite schedules of the same stabilizer
        type.

    Returns
    -------
    FourBodySchedule:
        The schedule type of X-stabilizer in the non-triangle part (for 1,2,3, config).
    FourBodySchedule:
        The schedule type of Z-stabilizer in the non-triangle part (for 1,2,3, config).
    FourBodySchedule | None:
        The schedule type of X-stabilizer in the triangle part (for type 3 config).
        None for other type of config.
    FourBodySchedule | None:
        The schedule type of Z-stabilizer in the triangle part (for type 3 config).
        None for other type of config.
    """
    config, pivot_corners = rsc_block.config_and_pivot_corners
    is_long_edge_horizontal = rsc_block.size[0] > rsc_block.size[1]
    stabilizers = rsc_block.stabilizers
    if config == 0:
        triangle_x_schedule = triangle_z_schedule = None
        non_triangle_x_schedule = rsc_block.weight_4_x_schedule
        non_triangle_z_schedule = non_triangle_x_schedule.opposite_schedule()
        return (
            non_triangle_x_schedule,
            non_triangle_z_schedule,
            triangle_x_schedule,
            triangle_z_schedule,
        )

    short_end_corner = pivot_corners[3]  # short end corner
    long_edge_idx = (
        0 if is_long_edge_horizontal else 1
    )  # coordinate index along the long side

    # A) Find boundary type for the short-edge boundary passing through short_end_corner
    weight2_stabs_at_boundary = [
        stab
        for stab in stabilizers
        if len(stab.pauli) == 2
        and all(
            dqubit[long_edge_idx] == short_end_corner[long_edge_idx]
            for dqubit in stab.data_qubits
        )
    ]
    boundary_paulis = [stab.pauli for stab in weight2_stabs_at_boundary]
    if len(set(boundary_paulis)) != 1:
        raise ValueError(
            f"Accept only one type of stabilizers along short-edge boundary, "
            f"found {boundary_paulis}."
        )
    boundary_type = boundary_paulis[0][0]

    # B) Find schedules
    match config:
        case 1 | 2:
            if boundary_type == "Z":
                # logical X runs along the short edge
                non_triangle_x_schedule = (
                    FourBodySchedule.Z
                    if is_long_edge_horizontal
                    else FourBodySchedule.N
                )
            else:
                # logical Z runs along the short edge
                non_triangle_x_schedule = (
                    FourBodySchedule.N
                    if is_long_edge_horizontal
                    else FourBodySchedule.Z
                )
            non_triangle_z_schedule = non_triangle_x_schedule.opposite_schedule()
            triangle_x_schedule, triangle_z_schedule = None, None
        case 3:
            if boundary_type == "Z":
                # logical Z runs along the short edge of the triangle part
                triangle_x_schedule = (
                    FourBodySchedule.N
                    if is_long_edge_horizontal
                    else FourBodySchedule.Z
                )
            else:
                # logical X runs along the short edge of the triangle part
                triangle_x_schedule = (
                    FourBodySchedule.Z
                    if is_long_edge_horizontal
                    else FourBodySchedule.N
                )
            triangle_z_schedule = triangle_x_schedule.opposite_schedule()
            non_triangle_x_schedule = triangle_x_schedule.opposite_schedule()
            non_triangle_z_schedule = triangle_z_schedule.opposite_schedule()
        case _:
            raise ValueError(f"Unknown corner configuration {config}.")

    return (
        non_triangle_x_schedule,
        non_triangle_z_schedule,
        triangle_x_schedule,
        triangle_z_schedule,
    )


def create_new_syndrome_circuits_with_known_schedules(
    # pylint: disable=too-many-statements, too-many-branches, too-many-locals
    rsc_block: RotatedSurfaceCode,
    non_triangle_x_schedule: FourBodySchedule,
    non_triangle_z_schedule: FourBodySchedule,
    triangle_x_schedule: FourBodySchedule | None,
    triangle_z_schedule: FourBodySchedule | None,
    starting_qubit_diag_directions: DiagonalDirection,
) -> tuple[tuple[SyndromeCircuit, ...], dict[str, str]]:
    """
    Create the syndrome extraction circuits for each stabilizer in the input block
    based on the schedules provided in the input. The data qubits of each stabilizer
    are ordered according to the schedule and the starting qubit direction.

    Parameters
    ----------
    rsc_block: RotatedSurfaceCode
        The rotated surface code block.
    non_triangle_x_schedule: FourBodySchedule
        Schedule for X-type stabilizers in the non-triangle part.
    non_triangle_z_schedule: FourBodySchedule
        Schedule for Z-type stabilizers in the non-triangle part.
    triangle_x_schedule: FourBodySchedule | None
        Schedule for X-type stabilizers in the triangle part. None if there is no
        triangle part.
    triangle_z_schedule: FourBodySchedule | None
        Schedule for Z-type stabilizers in the triangle part. None if there is no
        triangle part.
    starting_qubit_diag_directions: DiagonalDirection
        The diagonal direction of the starting data qubit for each syndrome extraction
        circuit.

    Returns
    -------
    tuple[tuple[SyndromeCircuit, ...], dict[str, str]]:
        A tuple containing:
        - A tuple of generated syndrome extraction circuits.
        - A dictionary mapping stabilizer UUIDs to their corresponding syndrome
        circuit UUIDs.
    """

    # Define 2 auxiliary functions
    def find_stabilizer_position(
        rsc_block: RotatedSurfaceCode,
        config: int,
        pivot_corners: tuple[
            tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
        ],
        stab: Stabilizer,
        is_long_edge_horizontal: bool,
    ) -> tuple[bool, Direction | None]:
        """
        Find stabilizer position relative to the triangle partition
        (in type 3 configuration)

        Parameters
        ----------
        block: RotatedSurfaceCode
            The initial block
        config: int
            Corner configuration
        pivot_corners:
            tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]
            Four topological corners
        stab: Stabilizer
            The input stabilizer
        is_long_edge_horizontal: bool
            True if the long edge of the block is horizontal

        Returns
        -------
        tuple[bool, Direction | None]
            True if the stabilizer is inside triangular part (if any)
            Direction of the boundary if the input stabilizer is a weight-2 stabilizer

        """

        if config == 0:
            is_in_triangle = False
        else:
            long_end_corner, middle_corner, _, short_end_corner = pivot_corners
            long_edge_idx = (
                0 if is_long_edge_horizontal else 1
            )  # coordinate index along the long side
            short_edge_idx = (
                1 if is_long_edge_horizontal else 0
            )  # coordinate index along the short side

            def is_triangular_qubit(q) -> bool:
                """Determine if the input qubit is inside the triangular part
                (including the boundaries of the triangle). This is done by checking
                whether the distance along the long side of the block between the input
                qubit and the middle-edge corner is at least the distance along the
                short side between the same two qubits.
                """
                match config:
                    case 1 | 2:
                        return False
                    case 3:
                        dshort = (
                            middle_corner[short_edge_idx] - q[short_edge_idx]
                            if middle_corner[short_edge_idx]
                            > short_end_corner[short_edge_idx]
                            else q[short_edge_idx] - middle_corner[short_edge_idx]
                        )
                        dlong = (
                            middle_corner[long_edge_idx] - q[long_edge_idx]
                            if middle_corner[long_edge_idx]
                            > long_end_corner[long_edge_idx]
                            else q[long_edge_idx] - middle_corner[long_edge_idx]
                        )
                        return 0 <= dshort <= dlong
                    case _:
                        raise ValueError(f"Unknown corner configuration {config}.")

            is_in_triangle = all(
                is_triangular_qubit(dqubit) for dqubit in stab.data_qubits
            )

        # find boundary direction for weight-2 stabilizer
        boundary = None
        weight = len(stab.pauli)
        if weight == 2:
            qubit_directions = [
                set(
                    direction
                    for direction in Direction
                    if dqubit in rsc_block.boundary_qubits(direction)
                )
                for dqubit in stab.data_qubits
            ]
            boundary = list(qubit_directions[0].intersection(qubit_directions[1]))[0]

        return is_in_triangle, boundary

    def find_data_qubits_order(
        stab: Stabilizer,
        starting_qubit_diag_direction: DiagonalDirection,
        schedule: FourBodySchedule,
        boundary_direction: Direction | None,
    ) -> tuple[tuple[int, ...], ...]:
        """
        Find correct data qubit order with respect to the input schedule and starting
        qubit direction.

        Parameters
        ----------
        stab: Stabilizer
            The input stabilizer
        starting_qubit_directions: tuple[Direction, Direction]
            Direction of the starting qubit
        schedule: FourBodySchedule
            Schedule
        boundary_direction: Direction | None
            Boundary direction if the input stabilizer is a weight-2 stabilizer

        Returns
        -------
        tuple[tuple[int, ...], ...]
            Ordered data qubits
        """
        weight = len(stab.pauli)
        is_n_schedule = schedule == FourBodySchedule.N
        lambda_expressions = {
            Direction.TOP: lambda x: -x[1],
            Direction.BOTTOM: lambda x: x[1],
            Direction.LEFT: lambda x: -x[0],
            Direction.RIGHT: lambda x: x[0],
        }

        qubits = list(stab.data_qubits)
        if weight == 2:
            # add padding qubits
            boundary_vect = boundary_direction.to_vector()
            qubits += [
                (dq[0] + boundary_vect[0], dq[1] + boundary_vect[1], dq[2])
                for dq in qubits
            ]

        # determine the starting qubit coordinate
        for direction in starting_qubit_diag_direction.components:
            max_value = max(
                lambda_expressions[direction](dq) for dq in stab.data_qubits
            )
            qubits = [
                dq for dq in qubits if lambda_expressions[direction](dq) == max_value
            ]
        starting_qubit = qubits[0]

        # determine the other qubits
        diag_vect = starting_qubit_diag_direction.to_vector()
        data_qubits = [
            starting_qubit,
            (
                starting_qubit[0] - diag_vect[0] * (not is_n_schedule),
                starting_qubit[1] - diag_vect[1] * is_n_schedule,
                starting_qubit[2],
            ),
            (
                starting_qubit[0] - diag_vect[0] * is_n_schedule,
                starting_qubit[1] - diag_vect[1] * (not is_n_schedule),
                starting_qubit[2],
            ),
            (
                starting_qubit[0] - diag_vect[0],
                starting_qubit[1] - diag_vect[1],
                starting_qubit[2],
            ),
        ]
        data_qubits = tuple(dq for dq in data_qubits if dq in stab.data_qubits)

        return data_qubits

    is_long_edge_horizontal = rsc_block.size[0] > rsc_block.size[1]
    config, pivot_corners = rsc_block.config_and_pivot_corners

    if triangle_x_schedule is None or triangle_z_schedule is None:
        if config == 3:
            raise ValueError(
                "Triangle schedules must be provided for type-3 corner configuration."
            )

    syndrome_circuits = tuple()
    stabilizer_to_circuit = {}
    for stab in rsc_block.stabilizers:
        pauli_type = stab.pauli_type
        is_in_triangular, boundary_direction = find_stabilizer_position(
            rsc_block, config, pivot_corners, stab, is_long_edge_horizontal
        )
        if is_in_triangular:
            schedule = triangle_x_schedule if pauli_type == "X" else triangle_z_schedule
        else:
            schedule = (
                non_triangle_x_schedule
                if pauli_type == "X"
                else non_triangle_z_schedule
            )

        # Generate syndrome circuit based on triangular and boundary
        weight = stab.weight
        match config:
            case 0 | 1:
                # for standard block, name it as convention
                if weight == 4:
                    syndrome_circuit_name = stab.pauli.lower()
                elif weight == 2:
                    syndrome_circuit_name = f"{boundary_direction}-{stab.pauli.lower()}"
                else:
                    # Unsupported weight
                    continue
            case 2 | 3:
                # for non-standard block, name it randomly
                syndrome_circuit_name = (
                    f"{'triangular' if is_in_triangular else 'non_triangular'}-"
                    f"{boundary_direction if boundary_direction else 'bulk'}"
                    f"-{str(pauli_type)*len(stab.pauli)}"
                )
            case _:
                raise ValueError(f"Unknown corner configuration {config}.")

        padding = None
        if weight == 2:
            # find padding for weight-2 stabilizer
            padding_boundary = boundary_direction
            for direction in starting_qubit_diag_directions.components:
                # Padding function in RotatedSurfaceCode is written only for
                # TOP-RIGHT starting direction .
                # Infer padding for other starting directions.
                if direction not in [Direction.TOP, Direction.RIGHT]:
                    padding_boundary = padding_boundary.mirror_across_orientation(
                        direction.to_orientation().perpendicular()
                    )
            padding = RotatedSurfaceCode.find_padding(padding_boundary, schedule)

        syndrome_circuit = RotatedSurfaceCode.generate_syndrome_circuit(
            stab.pauli, padding, syndrome_circuit_name
        )
        # This syndrome circuit doesn't take into consideration that the order of data
        # qubits may not match the order with which they need to be entangled.
        # As a result, we are going to rearrange the channel order so that the data
        # qubits interact with the ancilla in the appropriate order

        # Rearrange the the quantum channels in the Circuit part
        ordered_data_qubits = find_data_qubits_order(
            stab, starting_qubit_diag_directions, schedule, boundary_direction
        )
        ordered_data_qubit_indices = [
            ordered_data_qubits.index(dq) for dq in stab.data_qubits
        ]
        old_circuit = syndrome_circuit.circuit
        # Order the channels by first ordering the data qubits, and then append the rest
        # of the channels as they are
        channels_ordered = [
            old_circuit.channels[idx] for idx in ordered_data_qubit_indices
        ] + list(old_circuit.channels[len(ordered_data_qubit_indices) :])
        new_circuit = Circuit(old_circuit.name, old_circuit.circuit, channels_ordered)
        # Reconstruct the syndrome circuit using the new circuit with correct data qubit
        # order
        syndrome_circuit = SyndromeCircuit(
            syndrome_circuit.pauli, syndrome_circuit.name, new_circuit
        )

        # Avoid duplicate syndrome circuits
        same_syndrome_circuit = next(
            (circ for circ in syndrome_circuits if circ == syndrome_circuit), None
        )
        if same_syndrome_circuit is not None:
            syndrome_circuit = same_syndrome_circuit
        else:
            syndrome_circuits += (syndrome_circuit,)
        # Associate the stabilizer to the syndrome circuit
        stabilizer_to_circuit.update({stab.uuid: syndrome_circuit.uuid})

    syndrome_circuits = tuple(set(syndrome_circuits))

    return (
        syndrome_circuits,
        stabilizer_to_circuit,
    )


def direction_to_coord(  # pylint: disable=inconsistent-return-statements
    direction: Direction | DiagonalDirection, sublattice_index: int = None
) -> tuple[int, int, int]:
    """
    Converts a direction to a coordinate vector. This assumes the coordinates are in
    3-dimensions with the last coordinate being a sub-lattice index. The sub-lattice
    index of the current coordinate is also required for diagonal movements if the qubit
    is moved to a different sub-lattice.

    Parameters
    ----------
    direction: Direction | DiagonalDirection
        The direction in which the block is to be moved.
    sublattice_index: Optional[int]
        The sub-lattice index of the current coordinate. This should be defined for
        diagonal movements where the qubit is moved to a different sub-lattice.
        If it's a diagonal movement but on the same sub-lattice, this should be None.

    Returns
    -------
    tuple[int, int, int]
        The coordinate vector representing the direction in which the block is to be
        moved.

    Raises
    ------
    ValueError
        If the direction is not of type Direction or set of Directions of size 2.
        If the sub-lattice index is not an integer or None for diagonal movements.
        If the direction is not a valid direction.
    """
    # Direction vector for the 4 directions in the same sub-lattice.
    direction_vector = {
        Direction.TOP: (0, -1, 0),
        Direction.RIGHT: (1, 0, 0),
        Direction.LEFT: (-1, 0, 0),
        Direction.BOTTOM: (0, 1, 0),
    }
    # Sets are not hashable. Frozensets are.
    # The diagonal vector for moving within the same sub-lattice.
    diagonal_vector = {
        DiagonalDirection.TOP_RIGHT: (1, -1, 0),
        DiagonalDirection.TOP_LEFT: (-1, -1, 0),
        DiagonalDirection.BOTTOM_RIGHT: (1, 1, 0),
        DiagonalDirection.BOTTOM_LEFT: (-1, 1, 0),
    }
    # The diagonal vector for moving from sub-lattice 0 to sub-lattice 1.
    diagonal_vector_0 = {
        DiagonalDirection.TOP_RIGHT: (1, 0, 1),
        DiagonalDirection.TOP_LEFT: (0, 0, 1),
        DiagonalDirection.BOTTOM_RIGHT: (1, 1, 1),
        DiagonalDirection.BOTTOM_LEFT: (0, 1, 1),
    }
    # The diagonal vector for moving from sub-lattice 1 to sub-lattice 0.
    diagonal_vector_1 = {
        DiagonalDirection.TOP_RIGHT: (0, -1, -1),
        DiagonalDirection.TOP_LEFT: (-1, -1, -1),
        DiagonalDirection.BOTTOM_RIGHT: (0, 0, -1),
        DiagonalDirection.BOTTOM_LEFT: (-1, 0, -1),
    }

    if isinstance(direction, Direction):
        return direction_vector[direction]
    if isinstance(direction, DiagonalDirection):
        if not (isinstance(sublattice_index, int) or sublattice_index is None):
            raise ValueError(
                "Sub-lattice index must be an integer or None for diagonal movements."
            )
        if sublattice_index == 0:
            return diagonal_vector_0[direction]
        if sublattice_index == 1:
            return diagonal_vector_1[direction]
        if sublattice_index is None:
            return diagonal_vector[direction]
    else:
        raise ValueError(
            "Invalid direction. direction must be type Direction or set of Direction"
        )


def update_qubit_coords(
    input_qubit_coords: tuple[tuple[int, int, int]],
    direction: Direction | set[Direction],
) -> tuple[int, int, int]:
    """
    This function updates the coordinates of the qubits based on the direction of
    movement.
    """
    updated_qubit_coords = tuple()
    for each_qubit_coord in input_qubit_coords:
        updated_qubit_coords += (
            tuple(
                coord + dir
                for coord, dir in zip(
                    each_qubit_coord,
                    direction_to_coord(direction, each_qubit_coord[-1]),
                    strict=True,
                )
            ),
        )

    return updated_qubit_coords


def shift_block_towards_direction(
    rsc_block: RotatedSurfaceCode,
    direction: Direction | DiagonalDirection,
    debug_mode,
) -> tuple[
    RotatedSurfaceCode,
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
]:
    """
    Move the input block by one unit towards the input direction.

    Parameters
    ----------
    rsc_block: RotatedSurfaceCode
        The rotated surface code block to be moved.
    direction: Direction | DiagonalDirection
        The direction towards which the block is moved.
    debug_mode: bool
        If True, skip validation when creating the new block.

    Returns
    -------
    RotatedSurfaceCode:
        The moved rotated surface code block.
    dict[str, tuple[str, ...]]:
        A dictionary mapping old stabilizer UUIDs to new stabilizer UUIDs.
    dict[str, tuple[str, ...]]:
        A dictionary mapping old logical X operator UUIDs to new logical X operator
        UUIDs.
    dict[str, tuple[str, ...]]:
        A dictionary mapping old logical Z operator UUIDs to new logical Z operator
        UUIDs.
    """
    new_stabilizers = [
        Stabilizer(
            pauli=stab.pauli,
            data_qubits=update_qubit_coords(stab.data_qubits, direction),
            ancilla_qubits=update_qubit_coords(stab.ancilla_qubits, direction),
        )
        for stab in rsc_block.stabilizers
    ]
    new_logical_x_operators = [
        PauliOperator(
            log_x.pauli,
            data_qubits=update_qubit_coords(log_x.data_qubits, direction),
        )
        for log_x in rsc_block.logical_x_operators
    ]
    new_logical_z_operators = [
        PauliOperator(
            log_z.pauli,
            data_qubits=update_qubit_coords(log_z.data_qubits, direction),
        )
        for log_z in rsc_block.logical_z_operators
    ]

    # Construct the new stabilizer to circuit map by replacing old stabilizer uuids
    # with the new ones
    stabs_map_old_to_new_uuid = {
        stab.uuid: new_stabilizers[i].uuid
        for i, stab in enumerate(rsc_block.stabilizers)
    }
    new_stabilizer_to_circuit = {
        stabs_map_old_to_new_uuid[old_stab_uuid]: circ_uuid
        for old_stab_uuid, circ_uuid in rsc_block.stabilizer_to_circuit.items()
    }

    new_block = RotatedSurfaceCode(
        unique_label=rsc_block.unique_label,
        stabilizers=new_stabilizers,
        logical_x_operators=new_logical_x_operators,
        logical_z_operators=new_logical_z_operators,
        syndrome_circuits=rsc_block.syndrome_circuits,
        stabilizer_to_circuit=new_stabilizer_to_circuit,
        skip_validation=not debug_mode,
    )

    # Create evolution maps
    stab_evolution = {
        new_stab.uuid: (stab.uuid,)
        for stab, new_stab in zip(rsc_block.stabilizers, new_stabilizers)
    }
    log_x_evolution = {
        new_log_x.uuid: (log_x.uuid,)
        for log_x, new_log_x in zip(
            rsc_block.logical_x_operators, new_logical_x_operators
        )
    }
    log_z_evolution = {
        new_log_z.uuid: (log_z.uuid,)
        for log_z, new_log_z in zip(
            rsc_block.logical_z_operators, new_logical_z_operators
        )
    }

    return new_block, stab_evolution, log_x_evolution, log_z_evolution


class DetailedSchedule(Enum):
    """
    Assumptions made (The 4-body stabilizer looks like this):
    1 -- 2
    |    |
    3 -- 4

    For N-type schedule, if the first qubit is 1. (i.e. N1)
    The schedule will be: 1 -> 3 -> 2 -> 4

    For Z-type schedule, if the first qubit is 4. (i.e. Z4)
    The schedule will be: 4 -> 3 -> 2 -> 1

    The key of the Enum is the schedule type and first qubit in the schedule.
    (i.e. N1, N2, ...)
    The value is a tuple of qubit indices in the order of the schedule.
    (i.e. [1, 3, 2, 4]: tuple[int, int, int, int])

    NOTE: If the Block was created by the RotatedSurfaceCode.create() method, the
    schedules will be N2 or Z2, for the X(Z) or Z(X) Stabilizer.

    Code example:
    ```python
    schedule = DetailedSchedule.N1
    type(schedule)  # <enum 'DetailedSchedule'>
    type(schedule.value)  # <class 'tuple'>
    ```
    """

    # N-type schedules
    N1 = (1, 3, 2, 4)  # 1 -> 3 -> 2 -> 4
    N2 = (2, 4, 1, 3)  # 2 -> 4 -> 1 -> 3
    N3 = (3, 1, 4, 2)  # 3 -> 1 -> 4 -> 2
    N4 = (4, 2, 3, 1)  # 4 -> 2 -> 3 -> 1

    # Z-type schedules
    Z1 = (1, 2, 3, 4)  # 1 -> 2 -> 3 -> 4
    Z2 = (2, 1, 4, 3)  # 2 -> 1 -> 4 -> 3
    Z3 = (3, 4, 1, 2)  # 3 -> 4 -> 1 -> 2
    Z4 = (4, 3, 2, 1)  # 4 -> 3 -> 2 -> 1

    @classmethod
    def from_schedule_and_direction(  # pylint: disable=too-many-return-statements
        cls, schedule: FourBodySchedule, starting_qubit_direction: DiagonalDirection
    ) -> DetailedSchedule:
        """
        Create a DetailedSchedule from a FourBodySchedule and a starting qubit
        direction.
        """
        match (schedule, starting_qubit_direction):
            case (FourBodySchedule.N, DiagonalDirection.TOP_LEFT):
                return cls.N1
            case (FourBodySchedule.N, DiagonalDirection.TOP_RIGHT):
                return cls.N2
            case (FourBodySchedule.N, DiagonalDirection.BOTTOM_LEFT):
                return cls.N3
            case (FourBodySchedule.N, DiagonalDirection.BOTTOM_RIGHT):
                return cls.N4
            case (FourBodySchedule.Z, DiagonalDirection.TOP_LEFT):
                return cls.Z1
            case (FourBodySchedule.Z, DiagonalDirection.TOP_RIGHT):
                return cls.Z2
            case (FourBodySchedule.Z, DiagonalDirection.BOTTOM_LEFT):
                return cls.Z3
            case (FourBodySchedule.Z, DiagonalDirection.BOTTOM_RIGHT):
                return cls.Z4
            case _:
                raise ValueError(
                    f"Invalid combination of schedule {schedule} and "
                    f"starting qubit direction {starting_qubit_direction}."
                )

    def is_N(self) -> bool:  # pylint: disable=invalid-name
        """
        Checks if the schedule is an N-type schedule.

        Returns
        -------
        bool
            True if the schedule is an N-type schedule, False otherwise.
        """
        return self.name[0] == "N"

    def is_Z(self) -> bool:  # pylint: disable=invalid-name
        """
        Checks if the schedule is a Z-type schedule.

        Returns
        -------
        bool
            True if the schedule is a Z-type schedule, False otherwise.
        """
        return self.name[0] == "Z"

    def invert_vertically(self) -> DetailedSchedule:
        """
        Inverts the schedule vertically. For example, N1 = (1, 3, 2, 4) is transformed
        into N2 = (2, 4, 1, 3).

        Returns
        -------
        DetailedSchedule
            The inverted schedule.
        """
        v_inversion_dict = {
            DetailedSchedule.N1: DetailedSchedule.N2,
            DetailedSchedule.N2: DetailedSchedule.N1,
            DetailedSchedule.N3: DetailedSchedule.N4,
            DetailedSchedule.N4: DetailedSchedule.N3,
            DetailedSchedule.Z1: DetailedSchedule.Z2,
            DetailedSchedule.Z2: DetailedSchedule.Z1,
            DetailedSchedule.Z3: DetailedSchedule.Z4,
            DetailedSchedule.Z4: DetailedSchedule.Z3,
        }
        return v_inversion_dict[self]

    def rotate_ccw_90(self) -> DetailedSchedule:
        """
        Rotates the schedule counter-clockwise by 90 degrees.
        For example, N1 = (1, 3, 2, 4) is transformed into Z3 = (3, 4, 1, 2).

        Returns
        -------
        DetailedSchedule
            The rotated schedule.
        """
        ccw_rotation_dict = {
            # First length 4 cycle
            DetailedSchedule.N1: DetailedSchedule.Z3,
            DetailedSchedule.Z3: DetailedSchedule.N4,
            DetailedSchedule.N4: DetailedSchedule.Z2,
            DetailedSchedule.Z2: DetailedSchedule.N1,
            # Second length 4 cycle
            DetailedSchedule.Z1: DetailedSchedule.N3,
            DetailedSchedule.N3: DetailedSchedule.Z4,
            DetailedSchedule.Z4: DetailedSchedule.N2,
            DetailedSchedule.N2: DetailedSchedule.Z1,
        }
        return ccw_rotation_dict[self]

    def get_stabilizer_qubits(
        self, stab: Stabilizer, boundary_direction: Direction = None
    ) -> list[tuple[int, int, int]]:
        """For a specific stabilizer and a specific schedule, return the qubits in the
        order they should be entangled. The qubits are returned in the order of the
        schedule. The qubits that are not in the schedule are set to None.

        Parameters
        ----------
        stab: Stabilizer
            The stabilizer for which the qubits are to be determined.
        boundary_direction: Direction
            The direction of the boundary. This is required for weight-2 stabilizers.

        Returns
        -------
        list[tuple[int, int, int]]
            The qubits in the order they should be entangled. The qubits that are not
            in the schedule are set to None.
        """

        # Find the qubits that correspond to the indices in the schedule
        index_to_qubit = {
            1: max(stab.data_qubits, key=lambda x: -x[0] - x[1]),
            2: max(stab.data_qubits, key=lambda x: x[0] - x[1]),
            3: max(stab.data_qubits, key=lambda x: -x[0] + x[1]),
            4: max(stab.data_qubits, key=lambda x: x[0] + x[1]),
        }

        # If the stabilizer is weight-2, we need to set the qubits that are not in the
        # schedule to None
        qubit_idxs_to_set_none = []
        if len(stab.data_qubits) == 2:
            match boundary_direction:
                case Direction.TOP:
                    qubit_idxs_to_set_none = [1, 2]
                case Direction.RIGHT:
                    qubit_idxs_to_set_none = [2, 4]
                case Direction.LEFT:
                    qubit_idxs_to_set_none = [1, 3]
                case Direction.BOTTOM:
                    qubit_idxs_to_set_none = [3, 4]
                case _:
                    raise ValueError("The boundary direction must be specified.")

        # Evaluate the qubits and pad with None if necessary
        qubits = [
            index_to_qubit[idx] if idx not in qubit_idxs_to_set_none else None
            for idx in self.value
        ]

        return qubits


def find_detailed_schedules(
    rsc_block: RotatedSurfaceCode, starting_direction: DiagonalDirection
) -> dict[Stabilizer, DetailedSchedule]:
    """
    Find the detailed schedules for each stabilizer in the input block. The detailed
    schedule is determined by the schedule type (N or Z) and the starting qubit
    direction.
    TODO: Currently, this function only supports blocks without triangle schedules.
    We can expand it by refactoring the code to distinguish between triangle and
    non-triangle stabilizers with a function.

    Parameters
    ----------
    rsc_block: RotatedSurfaceCode
        The rotated surface code block.
    starting_direction: DiagonalDirection
        The diagonal direction of the starting data qubit for each syndrome extraction
        circuit.
    Returns
    -------
    dict[Stabilizer, DetailedSchedule]:
        A dictionary mapping each stabilizer to its corresponding detailed schedule.
    """
    (
        non_triangle_x_schedule,
        non_triangle_z_schedule,
        triangle_x_schedule,
        triangle_z_schedule,
    ) = find_schedules(rsc_block)

    if triangle_x_schedule is not None or triangle_z_schedule is not None:
        raise ValueError("Current implementation requires no triangle schedules.")

    x_stab_detailed_schedule = DetailedSchedule.from_schedule_and_direction(
        non_triangle_x_schedule, starting_direction
    )
    z_stab_detailed_schedule = DetailedSchedule.from_schedule_and_direction(
        non_triangle_z_schedule, starting_direction
    )

    return {
        stab: (
            x_stab_detailed_schedule
            if stab.pauli_type == "X"
            else z_stab_detailed_schedule
        )
        for stab in rsc_block.stabilizers
    }
