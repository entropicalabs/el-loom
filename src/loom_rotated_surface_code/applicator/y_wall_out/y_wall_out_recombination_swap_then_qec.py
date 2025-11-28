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

# pylint: disable=duplicate-code
from loom.eka import Circuit, Stabilizer
from loom.eka.utilities import DiagonalDirection, Orientation
from loom.interpreter import InterpretationStep

from .y_wall_out_utilities import (
    generate_teleportation_finalization_circuit_with_updates,
    find_stabilizer_position_in_final_block,
    qubit_mapper,
    move_logicals_and_append_evolution_and_updates,
)
from ..move_block import (
    DetailedSchedule,
    direction_to_coord,
    find_swap_then_qec_qubit_initializations,
    generate_syndrome_measurement_circuit_and_cbits,
    generate_and_append_block_syndromes_and_detectors,
)
from ..utilities import FourBodySchedule, find_relative_diagonal_direction
from ...code_factory import RotatedSurfaceCode


def y_wall_out_recombination_swap_then_qec(
    # pylint: disable=too-many-locals, too-many-arguments, too-many-positional-arguments
    interpretation_step: InterpretationStep,
    block: RotatedSurfaceCode,
    initial_block: RotatedSurfaceCode,
    qubits_to_idle: list[tuple[int, ...]],
    qubits_measured: list[tuple[int, ...]],
    qubits_to_had: list[tuple[int, ...]],
    same_timeslice: bool,
    debug_mode: bool,
) -> InterpretationStep:
    """
    Apply the y_wall_out recombination using the swap-then-qec method. The initial
    block that is missing the y_wall qubits is recombined into a new, compact block
    on the dual lattice.

    - A) Geometry

      - A.1) Find idle and hadamard side stabilizers and their directions along with \
            wall stabilizers

    - B) New stabilizers with evolution and updates

    - C) Logical operators with evolution and updates

    - D) Define new Block

    - E) Circuit generation

      - E.1) Find wall, idle and hadamard side qubits to initialize (ancillas, data, \
        and teleportation pairs)
      - E.2) Generate initialization circuit
      - E.3) Generate CNOT circuit
      - E.4) Generate teleportation finalization circuit and syndrome measurement \
        circuit
      - E.5) Assemble and append the full circuit to the interpretation step

    - F) Update Block history and evolution
    
    - G) Generate and append Block syndromes and detectors

    Parameters
    ----------
    interpretation_step: InterpretationStep
        The interpretation step.
    block: RotatedSurfaceCode
        The block to which the operation will be applied.
    initial_block: RotatedSurfaceCode
        The initial block before the y_wall_out operation. This is needed to find the
        wall data qubits to initialize.
    qubits_to_idle: list[tuple[int, ...]]
        The list of qubits to idle.
    qubits_measured: list[tuple[int, ...]]
        The list of qubits to measure.
    qubits_to_had: list[tuple[int, ...]]
        The list of qubits to Hadamard.
    same_timeslice: bool
        Whether to append the generated circuit to the same timeslice as the last
        circuit in the interpretation step.
    debug_mode: bool
        Whether to enable debug mode.
    """

    # A.) Geometry
    # A.1) Find idle and hadamard side stabilizers and their directions along with wall
    # stabilizers
    (
        idle_side_stabilizers,
        had_side_stabilizers,
        wall_stabilizers,
        idle_side_directions,
        had_side_directions,
    ) = get_idle_hadamard_info(
        block,
        qubits_to_had,
        qubits_to_idle,
    )

    # B.) New stabilizers with evolution and updates
    # Find new stabilizers after recombination and append their evolution and updates
    new_stabilizers = find_new_stabilizers_and_append_evolution_and_updates(
        interpretation_step,
        qubits_to_idle,
        idle_side_stabilizers,
        had_side_stabilizers,
        wall_stabilizers,
        idle_side_directions,
        had_side_directions,
    )

    # C.) Logical operators with evolution and updates
    # Move logical operators and append their evolution and updates
    new_logical_x, new_logical_z = move_logicals_and_append_evolution_and_updates(
        interpretation_step, block, idle_side_directions
    )

    # D.) Define new Block
    recombined_block = RotatedSurfaceCode(
        stabilizers=new_stabilizers,
        logical_x_operators=(new_logical_x,),
        logical_z_operators=(new_logical_z,),
        unique_label=block.unique_label,
        skip_validation=not debug_mode,
    )

    # E.) Circuit generation
    # E.1) Find wall, idle and hadamard side qubits to initialize (ancillas, data, and
    # teleportation pairs)
    # Wall qubits to initialize
    wall_data_qubits_to_init = get_wall_data_qubit_init(initial_block, qubits_measured)
    # Idle side qubits to initialize
    (
        anc_qubits_to_init_idle,
        data_qubits_to_init_idle,
        teleportation_qubit_pairs_idle,
    ) = find_swap_then_qec_qubit_initializations(
        idle_side_stabilizers,
        idle_side_directions,
    )
    # Hadamard side qubits to initialize
    anc_qubits_to_init_had, data_qubits_to_init_had, teleportation_qubit_pairs_had = (
        find_swap_then_qec_qubit_initializations(
            had_side_stabilizers,
            had_side_directions,
        )
    )

    # E.2) Generate initialization circuit
    first_swap_then_qec_reset_circuit = Circuit(
        name="Initialization of qubits for first swap-then-qec",
        circuit=[
            [
                Circuit(
                    f"reset_{'0' if pauli == 'Z' else '+'}",
                    channels=[interpretation_step.get_channel_MUT(q)],
                )
                for pauli in ["X", "Z"]
                for q in wall_data_qubits_to_init[pauli]
                + data_qubits_to_init_idle[pauli]
                + data_qubits_to_init_had[pauli]
                + anc_qubits_to_init_had[pauli]
                + anc_qubits_to_init_idle[pauli]
            ]
        ],
    )

    # E.3) Generate CNOT circuit
    first_swap_then_qec_cnots_circuit = (
        get_final_block_syndrome_measurement_first_swap_then_qec_cnots_circuit(
            interpretation_step,
            recombined_block,
            qubits_to_idle,
            qubits_to_had,
            idle_side_directions,
            had_side_directions,
            anc_qubits_to_init_idle,
            anc_qubits_to_init_had,
        )
    )
    # E.4) Generate teleportation finalization circuit and syndrome measurement circuit
    # Teleportation finalization circuit
    all_ancillas_to_init = {
        "X": anc_qubits_to_init_idle["X"] + anc_qubits_to_init_had["X"],
        "Z": anc_qubits_to_init_idle["Z"] + anc_qubits_to_init_had["Z"],
    }
    first_swap_then_qec_teleportation_finalization_circuit = (
        generate_teleportation_finalization_circuit_with_updates(
            interpretation_step,
            recombined_block,
            all_ancillas_to_init,
            teleportation_qubit_pairs_idle + teleportation_qubit_pairs_had,
        )
    )
    # Syndrome measurement circuit and classical bits
    first_swap_then_qec_measurement_circuit, first_swap_then_qec_cbits = (
        generate_syndrome_measurement_circuit_and_cbits(
            interpretation_step,
            recombined_block,
        )
    )

    # E.5) Assemble and append the full circuit to the interpretation step
    interpretation_step.append_circuit_MUT(
        Circuit(
            name="swap-then-qec to recombine block",
            circuit=Circuit.construct_padded_circuit_time_sequence(
                (
                    (first_swap_then_qec_reset_circuit,),
                    (first_swap_then_qec_cnots_circuit,),
                    (
                        first_swap_then_qec_measurement_circuit,
                        first_swap_then_qec_teleportation_finalization_circuit,
                    ),
                ),
            ),
        ),
        same_timeslice=same_timeslice,
    )

    # F.) Update Block history and evolution
    interpretation_step.update_block_history_and_evolution_MUT(
        (recombined_block,), (block,)
    )

    # G.) Generate and append Block syndromes and detectors
    generate_and_append_block_syndromes_and_detectors(
        interpretation_step, recombined_block, first_swap_then_qec_cbits
    )

    return interpretation_step


def get_idle_hadamard_info(
    block: RotatedSurfaceCode,
    qubits_to_had: list[tuple[int, ...]],
    qubits_to_idle: list[tuple[int, ...]],
) -> tuple[
    list[Stabilizer],
    list[Stabilizer],
    list[Stabilizer],
    DiagonalDirection,
    DiagonalDirection,
]:
    """
    Find the stabilizers associated with the idling and hadamard side of the block
    along with their directions.

    Parameters
    ----------
    block: RotatedSurfaceCode
        The block to which the operation will be applied.
    qubits_to_had: list[tuple[int, ...]]
        The list of qubits to Hadamard.
    qubits_to_idle: list[tuple[int, ...]]
        The list of qubits to idle.

    Returns
    -------
    list[Stabilizer]
        The stabilizers associated with the idling side.
    list[Stabilizer]
        The stabilizers associated with the hadamard side AFTER the transversal
        Hadamard.
    DiagonalDirection
        The directions of movement of the idling side.
    DiagonalDirection
        The directions of movement the hadamard side.
    """

    # Deduce geometric parameters
    block_orientation = block.orientation
    is_top_left_bulk_stab_x = block.upper_left_4body_stabilizer.pauli_type == "X"

    # Find idle side stabilizers
    idle_side_stabilizers = [
        stab
        for stab in block.stabilizers
        if set(stab.data_qubits).issubset(qubits_to_idle)
    ]

    # Find had side stabilizers along with their status after the Hadamard operation.
    # We need to find the stabilizers and then change the pauli type of the stabilizer.
    had_side_stabilizers = [
        stab
        for stab in block.stabilizers
        if set(stab.data_qubits).issubset(qubits_to_had)
    ]

    wall_stabilizers = [
        stab
        for stab in block.stabilizers
        if stab not in idle_side_stabilizers + had_side_stabilizers
    ]

    # Find idle and hadamard side directions
    match (block_orientation, is_top_left_bulk_stab_x):
        case (Orientation.VERTICAL, False):
            idle_side_directions = DiagonalDirection.BOTTOM_RIGHT
            had_side_directions = DiagonalDirection.TOP_RIGHT
        case (Orientation.VERTICAL, True):
            idle_side_directions = DiagonalDirection.BOTTOM_LEFT
            had_side_directions = DiagonalDirection.TOP_LEFT
        case (Orientation.HORIZONTAL, True):
            idle_side_directions = DiagonalDirection.TOP_RIGHT
            had_side_directions = DiagonalDirection.TOP_LEFT
        case (Orientation.HORIZONTAL, False):
            idle_side_directions = DiagonalDirection.BOTTOM_RIGHT
            had_side_directions = DiagonalDirection.BOTTOM_LEFT

    return (
        idle_side_stabilizers,
        had_side_stabilizers,
        wall_stabilizers,
        idle_side_directions,
        had_side_directions,
    )


def find_new_stabilizers_and_append_evolution_and_updates(
    interpretation_step: InterpretationStep,
    qubits_to_idle: list[tuple[int, ...]],
    idle_side_stabilizers: list[Stabilizer],
    had_side_stabilizers: list[Stabilizer],
    wall_stabilizers: list[Stabilizer],
    idle_side_directions: DiagonalDirection,
    had_side_directions: DiagonalDirection,
) -> list[Stabilizer]:
    """
    Find the new stabilizers after the recombination of the block and
    append their evolution and updates to the interpretation step.

    Parameters
    ----------
    interpretation_step: InterpretationStep
        The interpretation step. Note that it may be mutated by generating new
        channels.
    qubits_to_idle: list[tuple[int, ...]]
        The list of qubits to idle.
    idle_side_stabilizers: list[Stabilizer]
        The stabilizers associated with the idling side.
    had_side_stabilizers: list[Stabilizer]
        The stabilizers associated with the hadamard side AFTER the transversal
        Hadamard.
    wall_stabilizers: list[Stabilizer]
        The stabilizers associated with the wall.
    idle_side_directions: DiagonalDirection
        The directions of movement of the idling side.
    had_side_directions: DiagonalDirection
        The directions of movement the hadamard side.

    Returns
    -------
    list[Stabilizer]
        The new stabilizers after the recombination of the block.
    """

    new_stabilizers = []
    stab_evolution = {}
    # Find new stabilizers for the idle side
    for stab in idle_side_stabilizers:
        new_stab = Stabilizer(
            stab.pauli,
            [qubit_mapper(dq, idle_side_directions) for dq in stab.data_qubits],
            ancilla_qubits=[
                qubit_mapper(aq, idle_side_directions) for aq in stab.ancilla_qubits
            ],
        )
        new_stabilizers.append(new_stab)
        stab_evolution[new_stab.uuid] = (stab.uuid,)
    # Find new stabilizers for the hadamard side
    for stab in had_side_stabilizers:
        new_stab = Stabilizer(
            stab.pauli,
            [qubit_mapper(dq, had_side_directions) for dq in stab.data_qubits],
            ancilla_qubits=[
                qubit_mapper(aq, had_side_directions) for aq in stab.ancilla_qubits
            ],
        )
        new_stabilizers.append(new_stab)
        stab_evolution[new_stab.uuid] = (stab.uuid,)
    # Find new stabilizers for the wall
    for stab in wall_stabilizers:
        # Some part of the wall stabilizer is on the idling side and some on the
        # hadamard side.
        # The ancilla is on the idling side
        new_stab = Stabilizer(
            stab.pauli,
            [
                (
                    qubit_mapper(q, idle_side_directions)
                    if q in qubits_to_idle
                    else qubit_mapper(q, had_side_directions)
                )
                for q in stab.data_qubits
            ],
            ancilla_qubits=[
                qubit_mapper(aq, idle_side_directions) for aq in stab.ancilla_qubits
            ],
        )
        new_stabilizers.append(new_stab)
        stab_evolution[new_stab.uuid] = (stab.uuid,)

    # Update stabilizer updates
    for new_stab_uuid, old_stabs_uuid in stab_evolution.items():
        # Only one parent per new stabilizer
        old_stab_uuid = old_stabs_uuid[0]
        if old_stab_uuid in interpretation_step.stabilizer_updates:
            interpretation_step.stabilizer_updates[new_stab_uuid] = (
                interpretation_step.stabilizer_updates.pop(old_stab_uuid)
            )

    interpretation_step.stabilizer_evolution.update(stab_evolution)
    return new_stabilizers


def get_wall_data_qubit_init(
    y_wall_initial_block: RotatedSurfaceCode,
    qubits_to_measure: list[tuple[int, ...]],
) -> dict[str, list[tuple[int, ...]]]:
    """
    Find the data qubits to initialize for the wall in the y_wall_out operation context
    and the corresponding pauli type. There is a wall qubit that is excluded from the
    initialization and the rest are initialized according to the idle side stabilizer
    that contains the qubit. Which stabilizer is chosen is determined by the qubit being
    on the right corner of a weight 4 stabilizer.

    Parameters
    ----------
    y_wall_initial_block: RotatedSurfaceCode
        The block to which the operation will be applied.
    qubits_to_measure: list[tuple[int, ...]]
        The list of qubits to measure.

    Returns
    -------
    dict[str, list[tuple[int, ...]]]
        The data qubits to initialize in the X and Z basis.
    """
    # Deduce geometric parameters
    block_orientation = y_wall_initial_block.orientation
    is_top_left_bulk_stab_x = (
        y_wall_initial_block.upper_left_4body_stabilizer.pauli_type == "X"
    )

    wall_data_qubits_to_init = {"X": [], "Z": []}
    match (block_orientation, is_top_left_bulk_stab_x):
        # pylint: disable=unnecessary-lambda-assignment
        case (Orientation.VERTICAL, False):
            # Exclude left qubit
            qubit_to_exclude = min(qubits_to_measure, key=lambda x: x[0])
            # Get the type of stab from qubit being the bottom right of some stabilizer
            lambda_to_max = lambda x: x[0] + x[1]

        case (Orientation.VERTICAL, True):
            # Exclude right qubit
            qubit_to_exclude = max(qubits_to_measure, key=lambda x: x[0])
            # Get the type of stab from qubit being the bottom left of some stabilizer
            lambda_to_max = lambda x: -x[0] + x[1]

        case (Orientation.HORIZONTAL, True):
            # Exclude bottom qubit
            qubit_to_exclude = max(qubits_to_measure, key=lambda x: x[1])
            # Get the type of stab from qubit being the top right of some stabilizer
            lambda_to_max = lambda x: x[0] - x[1]

        case (Orientation.HORIZONTAL, False):
            # Exclude top qubit
            qubit_to_exclude = min(qubits_to_measure, key=lambda x: x[1])
            # Get the type of stab from qubit being the bottom right of some stabilizer
            lambda_to_max = lambda x: +x[0] + x[1]

    for q in qubits_to_measure:
        # Skip the qubit to exclude
        if q == qubit_to_exclude:
            continue
        # Find pauli type by seeing for which stabilizer the qubit is on the appropriate
        # corner of a weight 4 stabilizer.
        # EXAMPLE:
        # For (block_orientation, is_top_left_bulk_stab_x) = (VERTICAL, False),
        # the initialization basis is the same as the pauli flavor of the stabilizer
        # whose bottom right corner is the qubit q.
        # "bottom right" is defined by the lambda function lambda_to_max.
        stab = next(
            s
            for s in y_wall_initial_block.stabilizers
            if max(s.data_qubits, key=lambda_to_max) == q and len(s.data_qubits) == 4
        )
        # Append the qubit to the corresponding list
        wall_data_qubits_to_init[stab.pauli[0]].append(q)

    return wall_data_qubits_to_init


def get_final_block_syndrome_measurement_first_swap_then_qec_cnots_circuit(
    interpretation_step: InterpretationStep,
    recombined_block: RotatedSurfaceCode,
    qubits_to_idle: list[tuple[int, ...]],
    qubits_to_had: list[tuple[int, ...]],
    idle_side_directions: DiagonalDirection,
    had_side_directions: DiagonalDirection,
    anc_qubits_to_init_idle: dict[str, list[tuple[int, ...]]],
    anc_qubits_to_init_had: dict[str, list[tuple[int, ...]]],
) -> Circuit:
    # pylint: disable=too-many-locals, too-many-arguments, too-many-positional-arguments
    # pylint: disable=too-many-branches, too-many-statements
    """
    Generate the CNOT schedule of the first SWAP-then-QEC syndrome measurement round
    of the final block of y_wall_out operation.

    Parameters
    ----------
    interpretation_step: InterpretationStep
        The interpretation step. Note that it may be mutated by generating new
        channels.
    recombined_block: RotatedSurfaceCode
        The recombined block after the y_wall_out recombination.
    qubits_to_idle: list[tuple[int, ...]]
        The list of qubits to idle.
    qubits_to_had: list[tuple[int, ...]]
        The list of qubits to Hadamard.
    idle_side_directions: DiagonalDirection
        The directions that the idling side half of the block will move.
    had_side_directions: DiagonalDirection
        The directions that the hadamard side half of the block will move.
    anc_qubits_to_init_idle: dict[str, list[tuple[int, ...]]]
        The ancilla qubits to initialize on the idling side.
    anc_qubits_to_init_had: dict[str, list[tuple[int, ...]]]
        The ancilla qubits to initialize on the hadamard side.

    Returns
    -------
    Circuit
        The CNOT circuit of the first SWAP-then-QEC syndrome measurement round
        of the final block of y_wall_out operation.
    """
    config, pivot_corners = recombined_block.config_and_pivot_corners
    if config != 3:
        raise ValueError(
            "The final block of Y-wall-out must have type-3 corner configuration."
        )

    long_end_corner, angle_corner = pivot_corners[0], pivot_corners[2]
    top_left_corner = recombined_block.upper_left_qubit
    is_horizontal = recombined_block.is_horizontal

    # Schedules
    # Z schedule is determined based on the orientation of the short edge
    region_124_z_schedule = FourBodySchedule.Z if is_horizontal else FourBodySchedule.N
    region_3_z_schedule = region_124_z_schedule.opposite_schedule()
    region_12_x_schedule = region_124_z_schedule.opposite_schedule()
    region_34_x_schedule = region_12_x_schedule.opposite_schedule()

    # Initialize the cnots
    cnots = [[] for _ in range(5)]

    # TIMESLICE 0
    # The idling part and the hadamard sides are going to meet at the wall so they need
    # to be moved in different directions

    # Find CNOTs for the hadamard side

    had_side_data_to_anc_vec = direction_to_coord(had_side_directions, 0)
    for qub in qubits_to_had:
        anc_qub = tuple(map(sum, zip(qub, had_side_data_to_anc_vec, strict=True)))
        if anc_qub in anc_qubits_to_init_had["Z"]:
            cnot_pair = (qub, anc_qub)
        elif anc_qub in anc_qubits_to_init_had["X"]:
            cnot_pair = (anc_qub, qub)
        else:
            raise ValueError("The ancilla qubit was not found.")
        cnots[0].append(cnot_pair)

    # Find CNOTs for the idling side
    idle_side_data_to_anc_vec = direction_to_coord(idle_side_directions, 0)
    for qub in qubits_to_idle:
        anc_qub = tuple(map(sum, zip(qub, idle_side_data_to_anc_vec, strict=True)))
        if anc_qub in anc_qubits_to_init_idle["Z"]:
            cnot_pair = (qub, anc_qub)
        elif anc_qub in anc_qubits_to_init_idle["X"]:
            cnot_pair = (anc_qub, qub)
        else:
            raise ValueError("The ancilla qubit was not found.")
        cnots[0].append(cnot_pair)

    # TIMESLICE 1-4
    for stab in recombined_block.stabilizers:
        # Find stabilizer's position
        pauli_type = stab.pauli_type
        (
            is_in_region_12,
            is_in_region_3,
            _,
            is_on_wall,
            is_on_hadamard_side,
            boundary_direction,
        ) = find_stabilizer_position_in_final_block(
            recombined_block, stab, pivot_corners
        )
        is_on_idle_side = not (is_on_wall or is_on_hadamard_side)

        # Find stabilizer's schedule
        if pauli_type == "Z":
            schedule = region_3_z_schedule if is_in_region_3 else region_124_z_schedule
        else:
            schedule = region_12_x_schedule if is_in_region_12 else region_34_x_schedule

        # Find the starting qubit direction
        compare_corner = angle_corner if is_on_hadamard_side else long_end_corner
        starting_diag_direction = find_relative_diagonal_direction(
            top_left_corner, compare_corner
        )
        detailed_schedule = DetailedSchedule.from_schedule_and_direction(
            schedule, starting_diag_direction
        )

        if len(stab.data_qubits) == 4:
            data_qubits = detailed_schedule.get_stabilizer_qubits(stab)
        else:
            data_qubits = detailed_schedule.get_stabilizer_qubits(
                stab, boundary_direction
            )

        for i, data_qubit in enumerate(data_qubits):
            if (not is_on_wall and i == 0) or data_qubit is None:
                continue
            anc_qubit = stab.ancilla_qubits[0]

            if pauli_type == "Z":
                # If the stabilizer is Z, cnot from data qubit to ancilla
                cnot_pair = (data_qubit, anc_qubit)
            elif pauli_type == "X":
                # If the stabilizer is X, cnot from ancilla to data qubit
                cnot_pair = (anc_qubit, data_qubit)
            else:
                raise ValueError("Unknown stabilizer type.")

            # Find correct CNOT layer
            layer = (
                i + 1
                if (
                    is_on_wall  # on wall
                    or is_on_idle_side
                    and not is_in_region_12
                    and (
                        pauli_type == "Z"
                        and i in [2, 3]
                        or pauli_type == "X"
                        and i in [1, 2, 3]
                    )  # idle side
                    or is_in_region_3
                    and (
                        pauli_type == "Z"
                        and i in [1, 2, 3]
                        or pauli_type == "X"
                        and i == 3
                    )  # region 3
                )
                else i
            )
            cnots[layer].append(cnot_pair)

    cnots_circuit = Circuit(
        name="First SWAP-then-QEC final block syndrome measurement CNOT circuit",
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

    return cnots_circuit
