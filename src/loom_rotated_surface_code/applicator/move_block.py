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

from loom.eka import Stabilizer, Circuit
from loom.eka.utilities import Direction, DiagonalDirection, Orientation
from loom.interpreter import InterpretationStep, Cbit
from loom.interpreter.applicator import generate_syndromes, generate_detectors

from loom_rotated_surface_code.code_factory import RotatedSurfaceCode
from loom_rotated_surface_code.operations import MoveBlock
from .utilities import (
    shift_block_towards_direction,
    find_detailed_schedules,
    direction_to_coord,
    DetailedSchedule,
    update_qubit_coords,
)


def move_block(
    interpretation_step: InterpretationStep,
    operation: MoveBlock,
    same_timeslice: bool,
    debug_mode: bool,
) -> InterpretationStep:
    """
    Applicator to move a block in a fault tolerant manner based on the MoveBlock
    operation. The Block will be moved 1-unit in the specified direction by moving
    twice diagonally. By default, the diagonal moves are chosen using a secondary
    direction TOP or LEFT.

    For example, if the direction is RIGHT, the block will move diagonally upwards to
    the right in the first syndrome extraction round and diagonally downwards to the
    right in the second round.

    The algorithm is as follows:

    - A.) Begin MoveBlock composite operation session

    - B.) Valid move check
        - B.1) Check if the qubits required for the block to be moved are available.

    - C.) Shift the block
        - C.1) Shift the block in the specified direction
        - C.2) Update all the evolutions
        - C.3) Propagate all the updates

    - D.) Circuit generation
        - D.1) Find the qubit initializations required for the swap-then-qec operation \
              and create the reset circuit
        - D.2) Find the stabilizer schedules for the new block and generate the cnot \
              circuit
        - D.3) Generate the teleportation finalization circuit with necessary updates \
              to stabilizers and logical operators
        - D.4) Final measurement of the stabilizers and generation of syndromes
        - D.5) Combine all the circuits into one diagonal move circuit
    
    - E.) Append necessary information to the interpretation step
        - E.1) Append the circuit to the interpretation step
        - E.2) Update the block history and evolution
        - E.3) Create and append the new syndromes and detectors

    - Repeat steps B, C, D, E for the second diagonal direction.

    - F.) Final Circuit
        - F.1) End the composite operation session and append the full move block \
        circuit


    If the block is moved to the top:

    The syndrome extraction circuits assign the qubits as follows:

    1. The first set of syndrome extraction rounds assigns the qubits as follows::

                    a
            x --- x --- x
         a  |  a  |  a  |
            x --- x --- x
            |  a  |  a  |  a
            x --- x --- x
                a


        where x are data channels and a are ancilla channels.

    2. The second set of syndrome extraction rounds creates a new set of channels
    BASED on how the channels are first created, in most cases as ancilla qubits of
    either the teleport circuits or that of the syndrome extraction rounds during the
    move.

    As such the new set of qubits are assigned as follows::

            a     a     a
               x --- x --- x
            a  |  a  |  a  |
               x --- x --- x
            a  |  a  |  a  |  a
               x --- x --- x
                  a


    2b. Following step::

               x     x     x
            a --- a --- a
         x  |  x  |  x  |  x
            a --- a --- a
            |  x  |  x  |  x
            a --- a --- a     a
               x     x     x
                  a


    2c. Final Expected Configuration::

                        a
               x --- x --- x
            a  |  a  |  a  |
         a     x --- x --- x
            a  |  a  |  a  |  a
               x --- x --- x
            a     a     a     a
               x     x     x
                  a

    Parameters
    ----------
    interpretation_step: InterpretationStep
        The current interpretation step.
    operation: MoveBlock
        MoveBlock operation description.

    Returns
    -------
    InterpretationStep
        Interpretation Step after the MoveBlock operation has been applied.

    Raises
    ------
    ValueError
        If the block is not 2D or if the block cannot be moved in the specified
        direction.
    """
    block = interpretation_step.get_block(operation.input_block_name)

    # A) Begin MoveBlock composite operation session
    interpretation_step.begin_composite_operation_session_MUT(
        same_timeslice=same_timeslice,
        circuit_name=(
            f"Move block {block.unique_label} towards {operation.direction.value}"
        ),
    )

    # Find occupied qubits from other blocks in the latest timeslice
    other_blocks = [
        each_block
        for each_block in interpretation_step.get_blocks_at_index(-1)
        if each_block != block
    ]
    occupied_qubits = [
        each_qubit for each_block in other_blocks for each_qubit in each_block.qubits
    ]

    # Decompose the direction into 2 diagonal directions
    decomposed_directions = composite_direction(operation.direction)

    current_block = block
    for diag_direction in decomposed_directions:
        # B) - Valid move check
        # B.1) Check if the qubits required for the block to be moved are available.
        check_valid_move(occupied_qubits, current_block.qubits, diag_direction)

        # C, D, E) - Move the block diagonally via swap-QEC
        interpretation_step = move_block_diagonally_via_swap_qec(
            interpretation_step, current_block, diag_direction, debug_mode
        )

        # Prepare for the next iteration
        current_block = interpretation_step.get_block(operation.input_block_name)

    # F) Final Circuit
    # F.1) End the composite operation session and append the full move block circuit
    move_block_circuit = interpretation_step.end_composite_operation_session_MUT()
    interpretation_step.append_circuit_MUT(move_block_circuit, same_timeslice)

    return interpretation_step


def move_block_diagonally_via_swap_qec(
    interpretation_step: InterpretationStep,
    current_block: RotatedSurfaceCode,
    diag_direction: Direction,
    debug_mode: bool,
) -> InterpretationStep:
    """
    Moves the block diagonally in the specified direction using a swap-then-qec
    procedure.

    Parameters
    ----------
    interpretation_step: InterpretationStep
        The current interpretation step.
    current_block: RotatedSurfaceCode
        The current block to be moved.
    diag_direction: Direction
        The diagonal direction in which the block is to be moved.
    debug_mode: bool
        If True, skip validation when creating the new block.
    Returns
    -------
    tuple[RotatedSurfaceCode, Circuit]
        The new block after the move and the circuit that performs the move.
    """

    # C) - Shift the block
    # C.1) Shift the block in the specified direction
    new_block, stab_ev, logx_ev, logz_ev = shift_block_towards_direction(
        current_block, diag_direction, debug_mode=debug_mode
    )

    # C.2) Update all the evolutions
    # Stabilizer evolution
    interpretation_step.stabilizer_evolution.update(stab_ev)
    # Logical operator evolution
    interpretation_step.logical_x_evolution.update(logx_ev)
    interpretation_step.logical_z_evolution.update(logz_ev)

    # C.3) Propagate all the updates
    # Stabilizer updates
    for new_stab_uuid, old_stab_uuids in stab_ev.items():
        propagated_updates = ()
        for old_stab_uuid in old_stab_uuids:
            propagated_updates += interpretation_step.stabilizer_updates.get(
                old_stab_uuid, ()
            )
        if propagated_updates:
            interpretation_step.stabilizer_updates[new_stab_uuid] = propagated_updates
    # Logical operator updates
    interpretation_step.update_logical_operator_updates_MUT(
        "X", new_block.logical_x_operators[0].uuid, (), True
    )
    interpretation_step.update_logical_operator_updates_MUT(
        "Z", new_block.logical_z_operators[0].uuid, (), True
    )

    # D) Circuit generation
    # D.1) Find the qubit initializations required for the swap-then-qec operation
    # and create the reset circuit
    (
        anc_qubits_to_init,
        data_qubits_to_init,
        teleportation_qubit_pairs,
    ) = find_swap_then_qec_qubit_initializations(
        current_block.stabilizers, diag_direction
    )
    reset_basis_circuit = Circuit(
        name=("Initialization of qubits for first swap-then-qec"),
        circuit=[
            [
                Circuit(
                    f"reset_{'0' if pauli == 'Z' else '+'}",
                    channels=[interpretation_step.get_channel_MUT(q)],
                )
                for pauli in ["X", "Z"]
                for q in anc_qubits_to_init[pauli] + data_qubits_to_init[pauli]
            ]
        ],
    )

    # D.2) Find the stabilizer schedules for the new block and generate the cnot
    # circuit
    stab_schedule_dict = find_detailed_schedules(new_block, diag_direction)
    cnots_circuit = get_swap_qec_cnots(
        interpretation_step,
        new_block,
        diag_direction,
        stab_schedule_dict,
        anc_qubits_to_init,
        data_qubits_to_init,
    )

    # D.3) Generate the teleportation finalization circuit with necessary updates
    # to stabilizers and logical operators
    tp_finalization = generate_teleportation_measurement_circuit_with_updates(
        interpretation_step,
        new_block,
        anc_qubits_to_init,
        teleportation_qubit_pairs,
    )

    # D.4) Final measurement of the stabilizers and generation of syndromes
    syndrome_meas_circ, stab_measurements = (
        generate_syndrome_measurement_circuit_and_cbits(
            interpretation_step,
            new_block,
        )
    )

    # D.5) Combine all the circuits into one diagonal move circuit
    circ = Circuit(
        name=f"move block {new_block.unique_label} towards {diag_direction.value}",
        circuit=Circuit.construct_padded_circuit_time_sequence(
            [
                (reset_basis_circuit,),
                (cnots_circuit,),
                (tp_finalization, syndrome_meas_circ),
            ]
        ),
    )

    # E) - Append necessary information to the interpretation step
    # E.1) Append the circuit to the interpretation step
    interpretation_step.append_circuit_MUT(circ, same_timeslice=False)

    # E.2) Update the block history and evolution
    interpretation_step.update_block_history_and_evolution_MUT(
        (new_block,), (current_block,)
    )

    # E.3) Create and append the new syndromes and detectors
    # Create all new syndromes
    generate_and_append_block_syndromes_and_detectors(
        interpretation_step=interpretation_step,
        block=new_block,
        syndrome_measurement_cbits=stab_measurements,
    )

    return interpretation_step


def check_valid_move(
    occupied_qubits: tuple[tuple[int, int, int], ...],
    moving_qubits: tuple[tuple[int, int, int], ...],
    direction: Direction | DiagonalDirection,
):
    """
    This function checks if the move operation is valid by ensuring that none of the
    moving qubits will be moved onto an occupied qubit. If any of the moving qubits
    will be moved onto an occupied qubit, a ValueError is raised.

    Parameters
    ----------
    occupied_qubits: tuple[tuple[int, int, int], ...]
        A tuple of qubit coordinates representing "occupied" qubits. Qubits taking part
        in the move, cannot be moved onto these set of qubits as they are "occupied".
    moving_qubits: tuple[tuple[int, int, int], ...]
        A tuple of qubit coordinates representing qubits that will be moved. The last
        value in each qubit coordinate tuple represents the sub-lattice index of the
        qubit.
    direction: Direction | DiagonalDirection
        The direction in which the qubits will be moving.

    Raises
    ------
    ValueError
        If the move is invalid, i.e., if any of the moving qubits will be moved onto an
        occupied qubit.
    """
    # Do the diagonal move.
    for each_qubit in moving_qubits:
        new_qubit_coords = tuple(
            q + dir
            for q, dir in zip(
                each_qubit,
                direction_to_coord(direction, each_qubit[-1]),
                strict=True,
            )
        )
        if new_qubit_coords in occupied_qubits:
            raise ValueError(
                f"The move operation is invalid. The following qubit, {each_qubit}, is "
                f"moving to an occupied qubit, {new_qubit_coords}."
            )


def composite_direction(
    direction: Direction,
) -> tuple[DiagonalDirection, DiagonalDirection]:
    """
    For the move operation, the user defines only "right", "left", "top" or "bottom".
    However, the actual movement of the block is done via 2 diagonal directions. This
    function returns a tuple of 2 sets of directions based on the user-defined
    direction.

    For e.g.
    If the user specifies "right", the actual movement involves qubits moving to the
    "top right" then "bottom right".
    It would return a tuple with a set of directions "right" and "bottom", and a tuple
    with the directions "top" and "right", representing "bottom right" and "top right"
    respectively.

    composite_direction(Direction.RIGHT)
    -> (DiagonalDirection.TOP_RIGHT, DiagonalDirection.BOTTOM_RIGHT)

    Parameters
    ----------
    direction: Direction
        The direction in which the block is to be moved.

    Returns
    -------
    tuple[DiagonalDirection, DiagonalDirection]
        A tuple of 2 sets of diagonal directions which together represent the composite
        direction of movement.
    """

    composite_directions = {
        Direction.TOP: (
            DiagonalDirection.TOP_LEFT,
            DiagonalDirection.TOP_RIGHT,
        ),
        Direction.RIGHT: (
            DiagonalDirection.TOP_RIGHT,
            DiagonalDirection.BOTTOM_RIGHT,
        ),
        Direction.LEFT: (
            DiagonalDirection.TOP_LEFT,
            DiagonalDirection.BOTTOM_LEFT,
        ),
        Direction.BOTTOM: (
            DiagonalDirection.BOTTOM_LEFT,
            DiagonalDirection.BOTTOM_RIGHT,
        ),
    }

    return composite_directions[direction]


def find_swap_then_qec_qubit_initializations(  # pylint: disable=too-many-locals
    stabilizers: list[Stabilizer],
    diag_direction: DiagonalDirection,
    relocation_diag_direction: DiagonalDirection = None,
) -> tuple[
    dict[str, list[tuple[int, ...]]],
    dict[str, list[tuple[int, ...]]],
    list[tuple[tuple[int, ...], tuple[int, ...]]],
]:
    """
    Find the qubits to initialize for the swap_then_qec operation in the y_wall_out
    operation context. This is for a subset of stabilizers of the block that are
    associated with the boundary qubits.

    The recipe is as follows for moving towards TOP-RIGHT:

    - 1) FIND (non-teleporting) ANCILLA QUBIT INITIALIZATIONS:

        For each stabilizer that is NOT a the TOP-RIGHT boundary stabilizer,
        find the ancilla qubit on the TOP-RIGHT of the initial ancilla qubit
        and initialize it in the basis corresponding to the stabilizer.

    - 2) FIND TELEPORTATION QUBIT PAIRS:

        For each stabilizer that is NOT a BOTTOM-LEFT boundary stabilizer,
        check if the ancilla qubit of the stabilizer is already initialized. If not,
        initialize it in the basis corresponding to the stabilizer and form a
        teleportation qubit pair with the data qubit on the BOTTOM-LEFT of the ancilla.

    - 3) FIND DATA QUBIT INITIALIZATIONS:

        For each stabilizer that is a TOP-RIGHT boundary stabilizer, find the data qubit
        on the TOP-RIGHT of the ancilla qubit and initialize it in the basis
        corresponding to the stabilizer.

    Say we have the following block annotated by its stabilizers::

                     Z
            o --- o --- o
         X  |  Z  |  X  |
            o --- o --- o
            |  X  |  Z  |  X
            o --- o --- o
               Z

    To perform SWAP-THEN-QEC to move the block diagonally towards the TOP-RIGHT, we
    need to initialize the qubits  in the basis as shown below::

                    --- z
               x  |  z  |
            o --- o --- o
            |  z  |  x  |  z
            o --- o --- o --- x
            |  x  |  z  |  x
            o --- o --- o

    Of these, the teleportation qubit pairs can be seen with the numberings below::

                    --- z
               x  |  z  |
            o --- o --- o
            |  1  |  x  |  z
            1 --- o --- o --- x
            |  2  |  z  |  3
            2 --- o --- 3


    Parameters
    ----------
    stabilizers: list[Stabilizer]
        The stabilizers of the block.
    diag_direction: DiagonalDirection
        The diagonal direction that the block is moving.
    relocation_diag_direction: DiagonalDirection
        The directions to relocate the qubits. If None, the qubits are not relocated.

    Returns
    -------
    dict[str, list[tuple[int, ...]]]
        The ancilla qubits to initialize in the X and Z basis.
    dict[str, list[tuple[int, ...]]]
        The data qubits to initialize in the X and Z basis.
    list[tuple[tuple[int, ...], tuple[int, ...]]]
        The teleportation qubit pairs. The first qubit is the ancilla qubit and the
        second is the data qubit.
    """

    # Extract the component directions from the diagonal direction
    vert_direction = diag_direction.direction_along_orientation(Orientation.VERTICAL)
    hor_direction = diag_direction.direction_along_orientation(Orientation.HORIZONTAL)

    # Find the data qubits
    data_qubits = list({q for stab in stabilizers for q in stab.data_qubits})

    data_qubit_sublattice_index = set(q[2] for q in data_qubits)
    if len(data_qubit_sublattice_index) != 1:
        raise ValueError(
            "The stabilizers should all belong to the same sublattice of data qubits."
        )
    data_qubit_sublattice_index = data_qubit_sublattice_index.pop()
    anc_qubit_sublattice_index = 1 - data_qubit_sublattice_index

    # Find the boundary qubits
    bound_coords = {
        Direction.LEFT: min(q[0] for q in data_qubits),
        Direction.RIGHT: max(q[0] for q in data_qubits),
        Direction.TOP: min(q[1] for q in data_qubits),
        Direction.BOTTOM: max(q[1] for q in data_qubits),
    }
    # Find the boundary qubits towards the mixed direction and the opposite direction
    same_direction_boundary_qubits = [
        q
        for q in data_qubits
        if q[0] == bound_coords[hor_direction] or q[1] == bound_coords[vert_direction]
    ]
    opposite_direction_boundary_qubits = [
        q
        for q in data_qubits
        if q[0] == bound_coords[hor_direction.opposite()]
        or q[1] == bound_coords[vert_direction.opposite()]
    ]

    # Find the stabilizers that are only associated with the boundary qubits
    mixed_direction_bound_stabs = [
        stab
        for stab in stabilizers
        if set(stab.data_qubits).issubset(same_direction_boundary_qubits)
    ]
    opposite_mixed_direction_bound_stabs = [
        stab
        for stab in stabilizers
        if set(stab.data_qubits).issubset(opposite_direction_boundary_qubits)
    ]

    # Find the mixed direction vector
    mixed_direction_vector = direction_to_coord(diag_direction)

    anc_qubits_to_init = {"X": [], "Z": []}

    # Step 1
    # Find the first set of ancilla qubits to initialize
    for stab in stabilizers:
        if stab in mixed_direction_bound_stabs:
            # Skip same boundary stabilizers
            continue

        stab_anc = stab.ancilla_qubits[0]
        stab_type = stab.pauli_type

        other_anc = tuple(
            coord1 + coord2
            for coord1, coord2 in zip(stab_anc, mixed_direction_vector, strict=True)
        )

        anc_qubits_to_init[stab_type].append(other_anc)

    # Step 2
    # Find the teleportation qubits
    teleportation_qubit_pairs = []
    data_qubit_vector = direction_to_coord(
        diag_direction.opposite(), anc_qubit_sublattice_index
    )

    for stab in stabilizers:
        if stab in opposite_mixed_direction_bound_stabs:
            # Skip opposite boundary stabilizers
            continue
        stab_anc = stab.ancilla_qubits[0]
        pauli = stab.pauli_type

        # Check if it's a teleportation qubit
        if stab_anc not in anc_qubits_to_init[pauli]:
            anc_qubits_to_init[pauli].append(stab_anc)

            # Find its data qubit pair
            data_qubit = tuple(map(sum, zip(stab_anc, data_qubit_vector, strict=True)))
            teleportation_qubit_pairs.append((stab_anc, data_qubit))

    # Step 3
    # Find the data qubits to initialize
    data_qubits_to_init = {"X": [], "Z": []}
    # We need the ancilla to data qubit vector for that
    anc_to_data_vector = direction_to_coord(diag_direction, anc_qubit_sublattice_index)
    for stab in mixed_direction_bound_stabs:
        anc_qubit = stab.ancilla_qubits[0]
        diag_data_qubit = tuple(
            coord1 + coord2
            for coord1, coord2 in zip(anc_qubit, anc_to_data_vector, strict=True)
        )
        data_qubits_to_init[stab.pauli_type].append(diag_data_qubit)

    # Finally, relocate the qubits if needed
    if relocation_diag_direction is not None:
        anc_relocation_vector = direction_to_coord(
            relocation_diag_direction, anc_qubit_sublattice_index
        )
        data_relocation_vector = direction_to_coord(
            relocation_diag_direction, data_qubit_sublattice_index
        )

        def sum_vecs(vec1, vec2):
            return tuple(map(sum, zip(vec1, vec2, strict=True)))

        anc_qubits_to_init = {
            key: [sum_vecs(q, anc_relocation_vector) for q in val]
            for key, val in anc_qubits_to_init.items()
        }
        data_qubits_to_init = {
            key: [sum_vecs(q, data_relocation_vector) for q in val]
            for key, val in data_qubits_to_init.items()
        }
        teleportation_qubit_pairs = [
            tuple(
                (
                    sum_vecs(q, anc_relocation_vector)
                    if q[2] == 1
                    else sum_vecs(q, data_relocation_vector)
                )
                for q in q_pair
            )
            for q_pair in teleportation_qubit_pairs
        ]

    return anc_qubits_to_init, data_qubits_to_init, teleportation_qubit_pairs


def get_swap_qec_cnots(
    interpretation_step: InterpretationStep,
    new_block: RotatedSurfaceCode,
    move_diag_direction: DiagonalDirection,
    stab_schedule_dict: dict[Stabilizer, DetailedSchedule],
    anc_qubit_initialization: dict[str, list[tuple[int, ...]]],
    data_qubit_initialization: dict[str, list[tuple[int, ...]]],
) -> Circuit:
    """Generates the circuit for the final step of the swap-then-qec operation. This
    step moves the data qubits to their final positions where the new_block definition
    is.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        The interpretation step. Note that it may be mutated by generating new
        channels.
    new_block : RotatedSurfaceCode
        The new block after the diagonal move.
    move_diag_direction : DiagonalDirection
        The diagonal direction in which the block is moving.
    stab_schedule_dict : dict[Stabilizer, DetailedSchedule]
        A dictionary mapping each stabilizer to its detailed schedule.
    anc_qubit_initialization : dict[str, list[tuple[int, ...]]]
        A dictionary mapping each Pauli type to its list of ancilla qubit
        initializations.
    data_qubit_initialization : dict[str, list[tuple[int, ...]]]
        A dictionary mapping each Pauli type to its list of data qubit initializations.

    Returns
    -------
    Circuit
        The generated circuit for the final step of the swap-then-qec operation.
    """

    # Find the vectors to and from the final positions
    op_move_diag_direction = move_diag_direction.opposite()

    # Initialize the cnots and find the first layer of them from
    cnots = [[] for _ in range(4)]
    for q in new_block.data_qubits:
        q_prev = update_qubit_coords([q], op_move_diag_direction)[0]
        if q in anc_qubit_initialization["Z"] + data_qubit_initialization["Z"]:
            cnot_pair = (q_prev, q)
        elif q in anc_qubit_initialization["X"] + data_qubit_initialization["X"]:
            cnot_pair = (q, q_prev)
        else:
            raise ValueError(
                "Data qubit not found in either ancilla or data qubit initialization "
                "lists."
            )

        cnots[0].append(cnot_pair)

    # Find the rest of the cnots
    for stab in new_block.stabilizers:
        if len(stab.data_qubits) == 4:
            data_qubits = stab_schedule_dict[stab].get_stabilizer_qubits(stab)
        else:
            boundary_direction = next(
                dir for dir in Direction if stab in new_block.boundary_stabilizers(dir)
            )
            data_qubits = stab_schedule_dict[stab].get_stabilizer_qubits(
                stab, boundary_direction
            )
        for i, data_qubit in enumerate(data_qubits):
            if data_qubit is None or i == 0:
                # Skip if the data qubit is:
                # - None  (2-qubit stabilizer)
                # - the first qubit (already handled due to swap-qec)
                continue
            if stab.pauli_type == "Z":
                # If the stabilizer is Z, cnot from data qubit to ancilla
                cnot_pair = (data_qubit, stab.ancilla_qubits[0])
            elif stab.pauli_type == "X":
                # If the stabilizer is X, cnot from ancilla to data qubit
                cnot_pair = (stab.ancilla_qubits[0], data_qubit)
            else:
                raise ValueError("Unknown stabilizer type.")
            cnots[i].append(cnot_pair)

    # Generate the circuit containing the cnots
    swap_then_qec_cnots = Circuit(
        name=(
            f"Swap-then-QEC CNOTs for moving block {new_block.unique_label} towards "
            f"{move_diag_direction.value}"
        ),
        circuit=[
            [
                Circuit(
                    "cx",
                    channels=[
                        interpretation_step.get_channel_MUT(q) for q in qubit_pair
                    ],
                )
                for qubit_pair in cnot_slice
            ]
            for cnot_slice in cnots
        ],
    )

    return swap_then_qec_cnots


def generate_syndrome_measurement_circuit_and_cbits(
    interpretation_step: InterpretationStep,
    block: RotatedSurfaceCode,
    actual_anc_qubit_relocation_vector: tuple[int, ...] = (0, 0, 0),
) -> tuple[Circuit, list[tuple[Cbit, ...]]]:
    """
    Generate and return the circuit that measures the stabilizers of the block and
    the list of stabilizer measurements as tuples of Cbits.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        The interpretation step. Note that it may be mutated by generating new
        channels.
    block : RotatedSurfaceCode
        The block whose stabilizers are to be measured.
    actual_anc_qubit_relocation_vector : tuple[int, ...], optional
        The vector to the actual ancilla qubit, by default (0, 0, 0)

    Returns
    -------
    Circuit
        The circuit that performs the measurement of the stabilizers of the block.
        Note that it's just the final measurement operations and not the full
        stabilizer measurement circuit.
    list[tuple[Cbit, ...]]
        The list of stabilizer measurements as tuples of Cbits. The order of the list
        corresponds to the order of the stabilizers in block.stabilizers.
    """

    # Initialize the list of circuits
    stab_measurements = []
    meas_circ_seq = [[]]
    # Find the vector to the actual ancilla qubit
    for stab in block.stabilizers:
        # Find the data qubit containing the syndrome
        actual_anc_qubit = tuple(
            map(
                sum,
                zip(
                    stab.ancilla_qubits[0],
                    actual_anc_qubit_relocation_vector,
                    strict=True,
                ),
            )
        )

        actual_anc_channel = interpretation_step.get_channel_MUT(actual_anc_qubit)

        cbit = interpretation_step.get_new_cbit_MUT(f"c_{actual_anc_qubit}")
        stab_measurements.append((cbit,))

        cbit_channel = interpretation_step.get_channel_MUT(
            f"{cbit[0]}_{cbit[1]}", channel_type="classical"
        )

        m_circ = Circuit(
            f"measure_{stab.pauli_type}", channels=[actual_anc_channel, cbit_channel]
        )

        # Append the circuit to the list
        meas_circ_seq[0].append(m_circ)

    # Compile the stabilizer measurement circuit
    stab_meas_circ = Circuit(
        "measure stabilizer ancillas",
        circuit=meas_circ_seq,
    )

    return stab_meas_circ, stab_measurements


def generate_and_append_block_syndromes_and_detectors(
    interpretation_step: InterpretationStep,
    block: RotatedSurfaceCode,
    syndrome_measurement_cbits: list[tuple[Cbit, ...]],
) -> None:
    """Generate and append syndromes and detectors to the interpretation step.

    NOTE: This should probably be made into a method of InterpretationStep.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        The interpretation step. Note that it may be mutated by generating new
        channels.
    block : RotatedSurfaceCode
        The block to which the operation will be applied.
    syndrome_measurement_cbits : list[tuple[Cbit, ...]]
        The list of syndrome measurement classical bits. It has to correspond to the
        stabilizers of the block.
    """
    new_syndromes = generate_syndromes(
        interpretation_step=interpretation_step,
        stabilizers=block.stabilizers,
        block=block,
        stab_measurements=syndrome_measurement_cbits,
    )
    # Generate the new detectors for the new syndromes
    new_detectors = generate_detectors(interpretation_step, new_syndromes)
    # Append the syndromes and detectors to the interpretation step
    interpretation_step.append_syndromes_MUT(new_syndromes)
    interpretation_step.append_detectors_MUT(new_detectors)


def generate_teleportation_measurement_circuit_with_updates(
    interpretation_step: InterpretationStep,
    new_block: RotatedSurfaceCode,
    anc_qubits_to_init: dict[str, list[tuple[int, ...]]],
    teleportation_qubit_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]],
) -> Circuit:
    """Generate the circuit that finalizes the teleportation operation. This includes
    the measurement of the data qubits and the updating of the necessary stabilizers
    based on the measurement outcomes.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        The interpretation step. Note that it may be mutated by generating new
        channels and updating stabilizers and logical operators.
    new_block : RotatedSurfaceCode
        The new block after the diagonal move.
    anc_qubits_to_init : dict[str, list[tuple[int, ...]]]
        A dictionary mapping each Pauli type to its list of ancilla qubit
        initializations.
    teleportation_qubit_pairs : list[tuple[tuple[int, ...], tuple[int, ...]]]
        A list of tuples, each containing a pair of qubits involved in the
        teleportation operation. The first qubit in each tuple is the ancilla qubit to
        be corrected, and the second qubit is the data qubit to be measured.

    Returns
    -------
    Circuit
        The circuit that finalizes the teleportation operation
    """
    teleportation_circ_seq = [[]]
    for corrected_qubit, measured_qubit in teleportation_qubit_pairs:
        # Obtain the necessary channels and cbit
        cbit = interpretation_step.get_new_cbit_MUT(f"c_{measured_qubit}")
        cbit_channel = interpretation_step.get_channel_MUT(
            f"{cbit[0]}_{cbit[1]}", channel_type="classical"
        )
        measured_qubit_channel = interpretation_step.get_channel_MUT(measured_qubit)

        # Determine the operations based on the initialization basis of the
        # ancilla qubit
        if corrected_qubit in anc_qubits_to_init["X"]:
            measure_op_name = "measure_z"
            update_pauli = "Z"
        elif corrected_qubit in anc_qubits_to_init["Z"]:
            measure_op_name = "measure_x"
            update_pauli = "X"
        else:
            raise ValueError("The ancilla qubit was not found in the initialization.")

        # Define the circuits that measure the data qubit and operate on the ancilla
        # qubit based on the measurement
        meas_circ = Circuit(
            measure_op_name,
            channels=[measured_qubit_channel, cbit_channel],
        )

        # Append the circuits to the list
        teleportation_circ_seq[0].append(meas_circ)

        # Do the same for the logical operator (it's going to be only one operator)
        logical_ops = (
            new_block.logical_z_operators
            if update_pauli == "Z"
            else new_block.logical_x_operators
        )
        logical_ops_involved = [
            op for op in logical_ops if corrected_qubit in op.data_qubits
        ]
        for op in logical_ops_involved:

            interpretation_step.update_logical_operator_updates_MUT(
                update_pauli, op.uuid, (cbit,), False
            )

        # Find all appropriate stabilizers
        stabs_to_update = [
            stab
            for stab in new_block.stabilizers
            if stab.pauli_type == update_pauli  # correct flavor
            and corrected_qubit in stab.data_qubits  # qubit part of stabilizer
        ]

        # Append the Cbit on the updates of the stabilizers
        for stab in stabs_to_update:
            current_upd = interpretation_step.stabilizer_updates.get(stab.uuid, ())
            interpretation_step.stabilizer_updates[stab.uuid] = current_upd + (cbit,)

    # Compile the teleportation circuit finalization
    teleportation_circuit_finalization = Circuit(
        f"teleportation measurements of block {new_block.unique_label}",
        circuit=teleportation_circ_seq,
    )

    return teleportation_circuit_finalization
