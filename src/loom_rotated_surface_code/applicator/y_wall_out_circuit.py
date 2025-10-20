"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# Here we do not define an operation but rather the circuit for the y_wall_out
# operation due to the complexity of the operation.
from __future__ import annotations

from loom.eka import Circuit, Stabilizer
from loom.eka.utilities import Direction, Orientation, DiagonalDirection
from loom.interpreter import InterpretationStep, Cbit

from .move_block import (
    DetailedSchedule,
    direction_to_coord,
    find_swap_then_qec_qubit_initializations,
    generate_syndrome_measurement_operations_and_syndromes,
)
from ..code_factory import RotatedSurfaceCode


# pylint: disable=too-many-lines, unnecessary-lambda-assignment


def get_y_wall_out_circ_sequence_and_updates(
    # pylint: disable=too-many-locals, too-many-arguments, too-many-positional-arguments
    block: RotatedSurfaceCode,
    new_block: RotatedSurfaceCode,
    is_top_left_bulk_stab_x: bool,
    qubits_to_had: list[tuple[int, ...]],
    qubits_to_idle: list[tuple[int, ...]],
    qubits_to_measure: list[tuple[int, ...]],
    stabilizers_for_x_operator_jump: list[Stabilizer],
    y_wall_out_stabilizer_evolution: dict[str, tuple[str, ...]],
    interpretation_step: InterpretationStep,
) -> tuple[tuple[Circuit, ...], ...]:
    """
    Generate the circuit for the y_wall_out operation.

    Parameters
    ----------
    block: RotatedSurfaceCode
        The block to which the operation will be applied.
    new_block: RotatedSurfaceCode
        The new block after the operation.
    is_top_left_bulk_stab_x: bool
        Whether the top left stabilizer is X (or Z).
    qubits_to_had: list[tuple[int, ...]]
        The list of qubits to apply the Hadamard gate.
    qubits_to_idle: list[tuple[int, ...]]
        The list of qubits to idle.
    qubits_to_measure: list[tuple[int, ...]]
        The list of qubits to measure.
    stabilizers_for_x_operator_jump: list[Stabilizer]
        The stabilizers for the X operator jump.
    y_wall_out_stabilizer_evolution: dict[str, tuple[str, ...]]
        The stabilizer evolution during the y_wall_out operation.
    interpretation_step: InterpretationStep
        The interpretation step. Note that it may be mutated by generating new
        channels.

    Returns
    -------
    tuple[tuple[Circuit, ...], ...]
        The sequence of circuits to apply.
    """

    # Obtain some information about the block
    distance = min(block.size)
    block_orientation = (
        Orientation.VERTICAL
        if block.size[0] < block.size[1]
        else Orientation.HORIZONTAL
    )
    wall_orientation = block_orientation.perpendicular()
    is_wall_hor = wall_orientation == Orientation.HORIZONTAL
    wall_side_to_hadamard = Direction.BOTTOM if is_wall_hor else Direction.RIGHT

    hadamard_circuit = Circuit(
        name=(
            f"Transversal Hadamard on the data qubits on the "
            f"{wall_side_to_hadamard.value} of the {wall_orientation.value}"
            f"wall of block {block.unique_label}"
        ),
        circuit=[
            [
                Circuit("h", channels=[interpretation_step.get_channel_MUT(q)])
                for q in qubits_to_had
            ]
        ],
    )

    y_wall_measurement_circuit, y_meas_cbits = measure_wall_qubits_in_y_circuit(
        block, qubits_to_measure, interpretation_step
    )

    # After the hadamard operations, find the stabilizers that are associated with the
    # idling and hadamard side, along with their directions.
    (
        idle_side_stabilizers,
        post_had_side_stabilizers,
        idle_side_directions,
        had_side_directions,
    ) = get_idle_hadamard_info(
        block,
        is_top_left_bulk_stab_x,
        qubits_to_had,
        qubits_to_idle,
        block_orientation,
    )

    # STABILIZER UPDATES
    # Find the stabilizers on the idling side that contain the qubits to measure
    stabilizers_to_update = [
        stab
        for stab in new_block.stabilizers
        if set(stab.data_qubits).intersection(qubits_to_measure)
        and set(stab.data_qubits).intersection(qubits_to_idle)
    ]
    for stab in stabilizers_to_update:
        # Find cbits from y measurements associated with the stabilizer
        cbits = tuple(
            cbit
            for cbit, qubit_measured in zip(y_meas_cbits, qubits_to_measure)
            if qubit_measured in stab.data_qubits
        )
        current_updates = interpretation_step.stabilizer_updates.get(stab.uuid, ())
        # Add the cbits to the stabilizer updates and add 1 that stems from the
        # Y measurement that joins the 2 stabilizers on each side of the wall
        interpretation_step.stabilizer_updates[stab.uuid] = (
            current_updates + cbits + (1,)
        )

    # LOGICAL OPERATOR UPDATES
    interpretation_step.update_logical_operator_updates_MUT(
        "X",
        new_block.logical_x_operators[0].uuid,
        # - Add the z operator updates (because it's a phase gate)
        interpretation_step.logical_z_operator_updates.get(
            block.logical_z_operators[0].uuid, ()
        )
        +
        # - Add the stabilizer cbits needed to redefine the logical operator
        interpretation_step.retrieve_cbits_from_stabilizers(
            stabilizers_for_x_operator_jump, block
        )
        +
        # - Add the y measurement cbits
        tuple(y_meas_cbits) +
        # - Change the parity if the number of measured qubits is 5, 9, 13, ...
        # That is because of the accumulation of the imaginary units from the Y
        # measurements
        (1 * (len(y_meas_cbits) % 4 == 1),),
        inherit_updates=True,
    )

    # Find the basis on which the data qubits of the wall should be initialized
    wall_data_qubits_to_init = get_wall_data_qubit_init(
        block, is_top_left_bulk_stab_x, qubits_to_measure, block_orientation
    )

    # Find the qubits to initialize on the side of the idling stabilizers
    (
        anc_qubits_to_init_idle,
        data_qubits_to_init_idle,
        teleportation_qubit_pairs_idle,
    ) = find_swap_then_qec_qubit_initializations(
        idle_side_stabilizers,
        idle_side_directions,
    )

    # Find the qubits to initialize on the side of the hadamard stabilizers
    anc_qubits_to_init_had, data_qubits_to_init_had, teleportation_qubit_pairs_had = (
        find_swap_then_qec_qubit_initializations(
            post_had_side_stabilizers,
            had_side_directions,
        )
    )

    # INITIALIZATION CIRCUIT
    change_basis_circuit = Circuit(
        name=("Initialization of qubits for first swap-then-qec"),
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

    # SWAP-THEN-QEC CNOTs
    swap_then_qec_cnots_circuit = get_ywall_out_swap_then_qec_cnot_circ(
        interpretation_step,
        block,
        new_block,
        is_top_left_bulk_stab_x,
        qubits_to_had,
        qubits_to_idle,
        distance,
        block_orientation == Orientation.HORIZONTAL,
        idle_side_directions,
        had_side_directions,
        anc_qubits_to_init_idle,
        anc_qubits_to_init_had,
    )

    # MEASURE THE STABILIZERS FROM SWAP-THEN-QEC
    # Also, append the necessary Syndrome objects
    actual_anc_qubit_relocation_vec = direction_to_coord(idle_side_directions, 1)

    # Complete the teleportation circuit
    all_ancillas_to_init = {
        "X": anc_qubits_to_init_idle["X"] + anc_qubits_to_init_had["X"],
        "Z": anc_qubits_to_init_idle["Z"] + anc_qubits_to_init_had["Z"],
    }
    teleportation_finalization_circuit = (
        generate_teleportation_finalization_circuit_with_updates(
            interpretation_step,
            block,
            new_block,
            y_wall_out_stabilizer_evolution,
            qubits_to_idle,
            all_ancillas_to_init,
            teleportation_qubit_pairs_idle + teleportation_qubit_pairs_had,
        )
    )

    # Generate the syndrome measurement operations and syndromes
    stab_measurement_circuit = generate_syndrome_measurement_operations_and_syndromes(
        interpretation_step,
        new_block,
        actual_anc_qubit_relocation_vector=actual_anc_qubit_relocation_vec,
    )

    # Perform the final move of the block in its defined position
    # The moving directions are the opposite of the idle side directions
    final_moving_directions = idle_side_directions.opposite()

    # Find the final initialization of the stabilizers
    (
        anc_qubits_to_init_final,
        data_qubits_to_init_final,
        teleportation_qubit_pairs_final,
    ) = find_swap_then_qec_qubit_initializations(
        new_block.stabilizers,
        final_moving_directions,
        relocation_diag_direction=idle_side_directions,
    )

    # FINAL INITIALIZATION CIRCUIT
    initialization_circuit_final = Circuit(
        name=("Initialization of qubits for second swap-then-qec"),
        circuit=[
            [
                Circuit(
                    f"reset_{'0' if pauli == 'Z' else '+'}",
                    channels=[interpretation_step.get_channel_MUT(q)],
                )
                for pauli in ["X", "Z"]
                for q in data_qubits_to_init_final[pauli]
                + anc_qubits_to_init_final[pauli]
            ]
        ],
    )

    # Find the final swap-then-qec cnots circuit
    swap_then_qec_cnots_circuit_final = get_swap_qec_cnots_final_circuit(
        block,
        new_block,
        is_top_left_bulk_stab_x,
        qubits_to_idle,
        interpretation_step,
        distance,
        block_orientation,
        idle_side_directions,
        anc_qubits_to_init_final,
    )

    # Complete the final teleportation circuit
    teleportation_finalization_circuit_final = (
        generate_teleportation_finalization_circuit_with_updates(
            interpretation_step,
            block,
            new_block,
            y_wall_out_stabilizer_evolution,
            qubits_to_idle,
            anc_qubits_to_init_final,
            teleportation_qubit_pairs_final,
        )
    )

    # Find the final stabilizer measurements
    stab_meas_circ_final = generate_syndrome_measurement_operations_and_syndromes(
        interpretation_step,
        new_block,
    )

    circuit_sequence = (
        # Measure wall in Y and apply Hadamard
        (y_wall_measurement_circuit, hadamard_circuit),
        # Perform the recombination of the circuit
        # # Change basis of qubits to initialize in the X basis
        (change_basis_circuit,),
        # QEC-then-SWAP back to realign top left
        (swap_then_qec_cnots_circuit,),
        # Measure stabilizers with ancillas on lattice 0
        # and perform teleportation
        (
            stab_measurement_circuit,
            teleportation_finalization_circuit,
        ),
        # Change basis of qubits to initialize in the X basis
        (initialization_circuit_final,),
        # QEC-then-SWAP back to realign to the final position
        (swap_then_qec_cnots_circuit_final,),
        # Perform teleportation
        (stab_meas_circ_final, teleportation_finalization_circuit_final),
    )

    return circuit_sequence


def measure_wall_qubits_in_y_circuit(
    block: RotatedSurfaceCode,
    qubits_to_measure: list[tuple[int, ...]],
    interpretation_step: InterpretationStep,
) -> tuple[Circuit, list[Cbit]]:
    """Generates a circuit that measures all the data qubits of the wall in the Y basis.

    Parameters
    ----------
    block : RotatedSurfaceCode
        The block to which the operation will be applied.
    qubits_to_measure : list[tuple[int, ...]]
        The list of qubits to measure.
    interpretation_step : InterpretationStep
        The interpretation step. Note that it may be mutated by generating new
        channels.

    Returns
    -------
    Circuit
        The circuit to create the Y wall.
    """
    # Get classical channels corresponding to the qubits_to_measure
    cbits = [interpretation_step.get_new_cbit_MUT(f"c_{q}") for q in qubits_to_measure]
    cbit_channels = [
        interpretation_step.get_channel_MUT(
            f"{cbit[0]}_{cbit[1]}", channel_type="classical"
        )
        for cbit in cbits
    ]

    # Get the circuit that measures all the data qubits in the Y basis
    create_y_wall_circuit = Circuit(
        name=(f"Measure wall of qubits in the Y basis for block {block.unique_label}"),
        circuit=[
            [
                Circuit(
                    "measure_y",
                    channels=[interpretation_step.get_channel_MUT(qubit), cbit_channel],
                )
                for qubit, cbit_channel in zip(
                    qubits_to_measure, cbit_channels, strict=True
                )
            ]
        ],
    )

    return create_y_wall_circuit, cbits


def get_idle_hadamard_info(
    block: RotatedSurfaceCode,
    is_top_left_bulk_stab_x: bool,
    qubits_to_had: list[tuple[int, ...]],
    qubits_to_idle: list[tuple[int, ...]],
    block_orientation: Orientation,
) -> tuple[
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
    is_top_left_bulk_stab_x: bool
        Whether the top left stabilizer is X (or Z).
    qubits_to_had: list[tuple[int, ...]]
        The list of qubits to Hadamard.
    qubits_to_idle: list[tuple[int, ...]]
        The list of qubits to idle.
    block_orientation: Orientation
        The orientation of the block.

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
    # Find idle side stabilizers
    idle_side_stabilizers = [
        stab
        for stab in block.stabilizers
        if set(stab.data_qubits).issubset(qubits_to_idle)
    ]

    # Find had side stabilizers along with their status after the Hadamard operation.
    # We need to find the stabilizers and then change the pauli type of the stabilizer.
    pre_had_side_stabilizers = [
        stab
        for stab in block.stabilizers
        if set(stab.data_qubits).issubset(qubits_to_had)
    ]
    had_pauli = lambda pauli: "".join({"X": "Z", "Z": "X"}[p] for p in pauli)
    post_had_side_stabilizers = [
        Stabilizer(
            had_pauli(stab.pauli), stab.data_qubits, ancilla_qubits=stab.ancilla_qubits
        )
        for stab in pre_had_side_stabilizers
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
        post_had_side_stabilizers,
        idle_side_directions,
        had_side_directions,
    )


def get_wall_data_qubit_init(
    block, is_top_left_bulk_stab_x, qubits_to_measure, block_orientation
) -> dict[str, list[tuple[int, ...]]]:
    """
    Find the data qubits to initialize for the wall in the y_wall_out operation context
    and the corresponding pauli type. There is a wall qubit that is excluded from the
    initialization and the rest are initialized according to the an idle side stabilizer
    that contains the qubit. Which stabilizer is chosen is determined by the qubit being
    on the right corner of a weight 4 stabilizer.

    Parameters
    ----------
    block: RotatedSurfaceCode
        The block to which the operation will be applied.
    is_top_left_bulk_stab_x: bool
        Whether the top left stabilizer is X (or Z).
    qubits_to_measure: list[tuple[int, ...]]
        The list of qubits to measure.
    block_orientation: Orientation
        The orientation of the block.

    Returns
    -------
    dict[str, list[tuple[int, ...]]]
        The data qubits to initialize in the X and Z basis.
    """
    wall_data_qubits_to_init = {"X": [], "Z": []}
    match (block_orientation, is_top_left_bulk_stab_x):
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
            for s in block.stabilizers
            if max(s.data_qubits, key=lambda_to_max) == q and len(s.data_qubits) == 4
        )
        # Append the qubit to the corresponding list
        wall_data_qubits_to_init[stab.pauli[0]].append(q)

    return wall_data_qubits_to_init


def get_ywall_out_swap_then_qec_cnot_circ(
    # pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-positional-arguments
    interpretation_step: InterpretationStep,
    block: RotatedSurfaceCode,
    new_block: RotatedSurfaceCode,
    is_top_left_bulk_stab_x: bool,
    qubits_to_had: list[tuple[int, ...]],
    qubits_to_idle: list[tuple[int, ...]],
    distance: int,
    is_block_horizontal: bool,
    idle_side_directions: DiagonalDirection,
    had_side_directions: DiagonalDirection,
    anc_qubits_to_init_idle: dict[str, list[tuple[int, ...]]],
    anc_qubits_to_init_had: dict[str, list[tuple[int, ...]]],
) -> Circuit:
    """
    Generate the schedule of the phase swap then qec operation for the y_wall_out
    operation context.

    Parameters
    ----------
    interpretation_step: InterpretationStep
        The interpretation step. Note that it may be mutated by generating new
        channels.
    block: RotatedSurfaceCode
        The block to which the operation will be applied.
    new_block: RotatedSurfaceCode
        The new block after the operation.
    is_top_left_bulk_stab_x: bool
        Whether the top left stabilizer is X (or Z).
    qubits_to_had: list[tuple[int, ...]]
        The list of qubits to Hadamard.
    qubits_to_idle: list[tuple[int, ...]]
        The list of qubits to idle.
    qubits_to_measure: list[tuple[int, ...]]
        The list of qubits to measure.
    distance: int
        The distance of the shortest dimension of the block.
    is_block_horizontal: bool
        Whether the block is horizontal.
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
        The circuit of cnots for the phase swap then qec operation.
    """

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

    # TIMESLICES 1-4
    # To find the CNOTs for the rest of the timeslices, we need to associate each
    # stabilizer with
    # a) the schedule of the stabilizer
    # b) the timeslices in which the stabilizer is going to be applied

    # Initialize the stabilizer schedule dictionary. This dictionary is the one
    # corresponding to the case where:
    # (is_block_hor, is_top_left_bulk_stab_x) = (False, False)
    stab_schedule_dict_default, timeslices_stab_dict = classify_stabilizers(
        block,
        new_block,
        is_block_horizontal,
        qubits_to_idle,
        distance,
    )

    # MAP SCHEDULE DEPENDING ON THE ORIENTATION OF THE BLOCK AND THE TOP LEFT STABILIZER
    stab_schedule_dict = map_stabilizer_schedule(
        is_top_left_bulk_stab_x, is_block_horizontal, stab_schedule_dict_default
    )

    # Find the ancilla to data qubit vector and the data qubit to ancilla qubit
    # vector
    anc_data_qub_vec = direction_to_coord(idle_side_directions, 1)
    data_anc_qub_vec = direction_to_coord(idle_side_directions, 0)
    for timeslices, stabs in timeslices_stab_dict.items():
        for stab in stabs:
            # Find data qubits
            if len(stab.data_qubits) == 4:
                data_qubits = stab_schedule_dict[stab].get_stabilizer_qubits(stab)
            else:

                boundary_direction = next(
                    dir
                    for dir in Direction
                    if stab in new_block.boundary_stabilizers(dir)
                )
                data_qubits = stab_schedule_dict[stab].get_stabilizer_qubits(
                    stab, boundary_direction
                )

            # Find the CNOT pairs for the stabilizer
            new_anc_qub = tuple(
                map(sum, zip(stab.ancilla_qubits[0], anc_data_qub_vec, strict=True))
            )
            for data_qubit, t in zip(data_qubits, timeslices, strict=True):
                if t == 0 or data_qubit is None:
                    # Skip if it's the first timeslice (already appended) or
                    # if the qubit is None (weight-2 stabilizer)
                    continue

                # We shift the data qubit by the ancilla to data qubit vector
                # since the block is located on the sub-lattice with index 1
                new_data_qub = tuple(
                    map(sum, zip(data_qubit, data_anc_qub_vec, strict=True))
                )

                if stab.pauli[0] == "Z":
                    cnot_pair = (new_data_qub, new_anc_qub)
                else:
                    cnot_pair = (new_anc_qub, new_data_qub)
                cnots[t].append(cnot_pair)

    swap_then_qec_cnots = Circuit(
        name=(
            f"Swap-then-QEC CNOTs for the y_wall_out operation of block "
            f"{block.unique_label}"
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


def classify_stabilizers(
    block: RotatedSurfaceCode,
    new_block: RotatedSurfaceCode,
    is_block_horizontal: bool,
    qubits_to_idle: list[tuple[int, ...]],
    distance: int,
) -> tuple[dict[Stabilizer, DetailedSchedule], dict[tuple[int, ...], list[Stabilizer]]]:
    """
    Classify the stabilizers of the block into different categories. The classification
    entails assigning each stabilizer to a schedule and timeslices based on some
    geometric properties of the stabilizer and its type.
    The schedule shows in which order the data qubits will interact with the ancilla
    qubit during the syndrome measurement.
    The timeslices show in which timeslice these interactions will occur.

    Parameters
    ----------
    block: RotatedSurfaceCode
        The block to which the operation will be applied.
    new_block: RotatedSurfaceCode
        The new block after the operation.
    is_block_horizontal: bool
        Whether the block is horizontal.
    qubits_to_idle: list[tuple[int, ...]]
        The list of qubits to idle.
    distance: int
        The distance of the shortest dimension of the block.

    Returns
    -------
    dict[Stabilizer, DetailedSchedule]
        The dictionary of the stabilizer schedules.
    dict[tuple[int, ...], list[Stabilizer]]
        The dictionary showing the timeslices in which the stabilizers are going to be
        measured.
    """
    # Initialize the schedule dictionary
    stab_schedule_dict_default: dict[Stabilizer, DetailedSchedule] = {}
    # Initialize the order_stab_dict
    timeslices_stab_dict: dict[Stabilizer, tuple[int, ...]] = {
        (black_order := (0, 1, 2, 3)): [],
        (red_order := (0, 2, 3, 4)): [],
        (blue_order := (0, 1, 3, 4)): [],
        (cyan_order := (1, 2, 3, 4)): [],
    }

    # Get  the topological corner that is not geometric from the previous block
    prev_block_top_corn_not_geom = next(
        iter(set(block.topological_corners) - set(block.geometric_corners))
    )

    # Get the last line of idling qubits
    coord_to_max_idle_qub = 0 if is_block_horizontal else 1
    max_idle_qub = max(q[coord_to_max_idle_qub] for q in qubits_to_idle)
    last_line_idling_qubits = [
        q for q in qubits_to_idle if q[coord_to_max_idle_qub] == max_idle_qub
    ]

    # Categorize each stabilizer
    for stab in new_block.stabilizers:
        data_qubit_set = set(stab.data_qubits)
        pauli = stab.pauli[0]

        # Case 1:
        # Stabilizers on the idling side (black or blue)
        if data_qubit_set.issubset(qubits_to_idle):
            # Order is blue if the stabilizer has a Z-pauli and intersects with the last
            # line of idling qubits.
            # Otherwise, the order is black.
            order = (
                blue_order
                if (
                    data_qubit_set.intersection(last_line_idling_qubits)
                    and pauli == "Z"
                )
                else black_order
            )
            timeslices_stab_dict[order] += [stab]
            if pauli == "Z":
                stab_schedule_dict_default[stab] = DetailedSchedule.N4
            else:
                stab_schedule_dict_default[stab] = DetailedSchedule.Z4

        # Case 2:
        # Stabilizers on wall (cyan)
        elif data_qubit_set.intersection(qubits_to_idle):
            timeslices_stab_dict[cyan_order] += [stab]
            stab_schedule_dict_default[stab] = DetailedSchedule.N4

        # Case 3:
        # The rest of the stabilizers will either be red or black depending on the
        # distance of the data qubits to the topological corner that is not geometric
        # from the previous block
        else:
            # Find the manhattan distance of each data qubit to the topological corner
            # that is not geometric
            data_qub_top_corn_distance = [
                abs(q[0] - prev_block_top_corn_not_geom[0])
                + abs(q[1] - prev_block_top_corn_not_geom[1])
                for q in stab.data_qubits
            ]

            # Find the number of data qubits that are near the topological corner
            # (proximity is defined as being at a distance less than distance - 2)
            data_qubs_near_topological_corner = len(
                [d for d in data_qub_top_corn_distance if d < distance - 2]
            )

            # Based on the number of data qubits near the topological corner, assign the
            # stabilizer to the red or black order
            if len(data_qubit_set) == 4:
                # weight 4 stabilizers
                if data_qubs_near_topological_corner >= 3:
                    timeslices_stab_dict[black_order] += [stab]
                    stab_schedule_dict_default[stab] = DetailedSchedule.N2
                else:
                    timeslices_stab_dict[red_order] += [stab]
                    stab_schedule_dict_default[stab] = DetailedSchedule.Z2
            else:
                # weight 2 stabilizers
                if data_qubs_near_topological_corner >= 1:
                    timeslices_stab_dict[black_order] += [stab]
                    stab_schedule_dict_default[stab] = DetailedSchedule.N2
                else:
                    timeslices_stab_dict[red_order] += [stab]
                    stab_schedule_dict_default[stab] = DetailedSchedule.Z2

    return stab_schedule_dict_default, timeslices_stab_dict


def map_stabilizer_schedule(
    is_top_left_bulk_stab_x: bool,
    is_block_horizontal: bool,
    stab_schedule_dict_default: dict[Stabilizer, DetailedSchedule],
) -> dict[Stabilizer, DetailedSchedule]:
    """Map the dictionary of the default stabilizer schedules to the new stabilizer
    schedules based on the given parameters.
    The default stabilizer schedules are assumed to be for the top left bulk stabilizer
    being Z and the block being vertical.

    Parameters
    ----------
    is_top_left_bulk_stab_x : bool
        Whether the top left bulk stabilizer is X.
    is_block_horizontal : bool
        Whether the block is horizontal.
    stab_schedule_dict_default : dict[Stabilizer, DetailedSchedule]
        The dictionary of the default stabilizer schedules.

    Returns
    -------
    dict[Stabilizer, DetailedSchedule]
        The dictionary of the new stabilizer schedules.
    """
    # Get all the schedules appearing in the stabilizer schedule dictionary
    all_schedules = set(stab_schedule_dict_default.values())

    match (is_block_horizontal, is_top_left_bulk_stab_x):
        case (False, False):
            # identity map
            schedule_map = {s: s for s in all_schedules}
        case (False, True):
            # This is equivalent to inverting the schedule along the vertical axis
            schedule_map = {s: s.invert_vertically() for s in all_schedules}
        case (True, True):
            # This is equivalent to rotating the schedule 90 degrees counter-clockwise
            schedule_map = {s: s.rotate_ccw_90() for s in all_schedules}
        case (True, False):
            # This is equivalent to inverting the schedule along the vertical axis and
            # then rotating 90 degrees counter-clockwise
            schedule_map = {
                s: s.invert_vertically().rotate_ccw_90() for s in all_schedules
            }

    # Update the schedules of the stabilizers
    stab_schedule_dict = {
        stab: schedule_map[schedule]
        for stab, schedule in stab_schedule_dict_default.items()
    }

    return stab_schedule_dict


def generate_teleportation_finalization_circuit_with_updates(
    # pylint: disable=too-many-branches, too-many-locals
    interpretation_step: InterpretationStep,
    block: RotatedSurfaceCode,
    new_block: RotatedSurfaceCode,
    y_wall_out_stabilizer_evolution: dict[str, tuple[str, ...]],
    qubits_to_idle: list[tuple[int, ...]],
    anc_qubits_to_init: dict[str, list[tuple[int, ...]]],
    teleportation_qubit_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]],
) -> Circuit:
    """Generate the circuit that finalizes the teleportation operation. This includes
    the measurement of the data qubits and the updating of the necessary stabilizers
    based on the measurement outcomes.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        The interpretation step.
    block : RotatedSurfaceCode
        The block to which the operation will be applied.
    new_block : RotatedSurfaceCode
        The new block after the operation.
    y_wall_out_stabilizer_evolution : dict[str, tuple[str, ...]]
        The evolution of the stabilizers. Used to determine the necessary updates.
    qubits_to_idle: list[tuple[int, ...]]
        A list of qubits that idle during the first step of the y_wall_out process.
    anc_qubits_to_init : dict[str, list[tuple[int, ...]]]
        A dictionary containing the ancilla qubits to initialize in the X and Z basis.
    teleportation_qubit_pairs : list[tuple[tuple[int, ...], tuple[int, ...]]]
        The list of teleportation qubit pairs. The first qubit is the ancilla qubit
        and the second is the data qubit.

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
            stabs_to_update_pauli = "Z"
        elif corrected_qubit in anc_qubits_to_init["Z"]:
            measure_op_name = "measure_x"
            stabs_to_update_pauli = "X"
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

        # Find which qubit of the two is in the block and will be associated with
        # some stabilizers
        if measured_qubit in block.data_qubits:
            qubit_in_block = measured_qubit
            first_swap_then_qec = True
        elif corrected_qubit in block.data_qubits:
            qubit_in_block = corrected_qubit
            first_swap_then_qec = False
        else:
            raise ValueError("Neither qubit is in the block.")

        # Do the same for the logical operator (it's going to be only one operator)
        logical_ops = (
            new_block.logical_z_operators
            if stabs_to_update_pauli == "Z"
            else new_block.logical_x_operators
        )
        logical_ops_involved = [
            op for op in logical_ops if qubit_in_block in op.data_qubits
        ]
        for op in logical_ops_involved:
            interpretation_step.update_logical_operator_updates_MUT(
                stabs_to_update_pauli, op.uuid, (cbit,), True
            )

        # Append stabilizer updates to the necessary stabilizers
        if first_swap_then_qec:
            # CORRESPONDS TO THE FIRST SWAP-THEN-QEC
            if qubit_in_block in qubits_to_idle:
                # Updated stabilizer will end up at the same spot eventually
                stabs_to_update = [
                    stab
                    for stab in new_block.stabilizers
                    if stab.pauli[0] == stabs_to_update_pauli
                    and qubit_in_block in stab.data_qubits
                ]
            else:
                # Updated stabilizer will be relocated by the operation
                stabs_id_pre_evolution = [
                    stab.uuid
                    for stab in block.stabilizers
                    if qubit_in_block in stab.data_qubits
                ]
                # That will evolve into the stabs_post_evolution
                stabs_post_evolution = [
                    interpretation_step.stabilizers_dict[final_id]
                    for final_id, init_id in y_wall_out_stabilizer_evolution.items()
                    if set(init_id).intersection(stabs_id_pre_evolution)
                ]
                # Of those, we need to find the ones that correspond to the correct
                # Pauli flavor
                stabs_to_update = [
                    stab
                    for stab in stabs_post_evolution
                    if stab.pauli[0] == stabs_to_update_pauli
                ]
        else:
            # CORRESPONDS TO THE SECOND SWAP-THEN-QEC
            # Find all appropriate stabilizers
            stabs_to_update = [
                stab
                for stab in new_block.stabilizers
                if stab.pauli[0] == stabs_to_update_pauli  # correct flavor
                and qubit_in_block in stab.data_qubits  # qubit part of stabilizer
            ]

        # Append the Cbit on the updates of the stabilizers
        for stab in stabs_to_update:
            current_upd = interpretation_step.stabilizer_updates.get(stab.uuid, ())
            interpretation_step.stabilizer_updates[stab.uuid] = current_upd + (cbit,)

    # Compile the teleportation circuit finalization
    teleportation_circuit_finalization = Circuit(
        f"teleportation finalization from sub-lattice "
        f"{0 if first_swap_then_qec else 1} to "
        f"sub-lattice {1 if first_swap_then_qec else 0}",
        circuit=teleportation_circ_seq,
    )

    return teleportation_circuit_finalization


def get_swap_qec_cnots_final_circuit(
    # pylint: disable=too-many-positional-arguments, too-many-arguments
    block: RotatedSurfaceCode,
    new_block: RotatedSurfaceCode,
    is_top_left_bulk_stab_x: bool,
    qubits_to_idle: list[tuple[int, ...]],
    interpretation_step: InterpretationStep,
    distance: int,
    block_orientation: Orientation,
    idle_side_directions: DiagonalDirection,
    anc_qubits_to_init_final: dict[str, list[tuple[int, ...]]],
) -> Circuit:
    """Generates the circuit for the final step of the swap-then-qec operation. This
    step moves the data qubits to their final positions where the new_block definition
    is.

    Parameters
    ----------
    block : RotatedSurfaceCode
        The block to which the operation will be applied.
    new_block : RotatedSurfaceCode
        The new block after the operation.
    is_top_left_bulk_stab_x : bool
        Indicates if the top-left bulk stabilizer is an X stabilizer.
    qubits_to_idle : list[tuple[int, ...]]
        List of qubits that should be idled.
    qubits_to_measure : list[tuple[int, ...]]
        List of qubits that should be measured.
    interpretation_step : InterpretationStep
        The interpretation step for the circuit generation.
    distance : int
        The distance of the shortest dimension of the block.
    block_orientation : Orientation
        The orientation of the block (HORIZONTAL or VERTICAL).
    idle_side_directions : DiagonalDirection
        Directions of the sides where qubits are idled.
    anc_qubits_to_init_final : dict[str, list[tuple[int, ...]]]
        Dictionary containing ancilla qubits to initialize for final
        X and Z stabilizers.

    Returns
    -------
    Circuit
        The generated circuit for the final step of the swap-then-qec operation.
    """

    # Reuse the classify_stabilizers function to get the stabilizer schedules
    # and then modify them appropriately so that they fit in 4 timeslices
    stab_schedule_dict_default, _ = classify_stabilizers(
        block,
        new_block,
        block_orientation == Orientation.HORIZONTAL,
        qubits_to_idle,
        distance,
    )
    # Modify the schedules by casting all of them to Z1 or N1
    # NOTE: In the default case where:
    # (block_orientation, is_top_left_bulk_stab_x) = (VERTICAL, False), the moving is
    # done towards up and left and this is why they have to be cast to Z1 or N1.
    # All X stabilizers are cast to N1 while some Z stabilizers are cast to Z1.
    # preserve fault-tolerance.
    stab_schedule_dict_standard_swap_qec_default = {
        stab: (
            DetailedSchedule.N1
            if (schedule.is_N() and stab.pauli[0] == "Z")
            else DetailedSchedule.Z1
        )
        for stab, schedule in stab_schedule_dict_default.items()
    }
    # Then proceed to rotate/invert the stabilizer schedules as needed
    stab_schedule_dict = map_stabilizer_schedule(
        is_top_left_bulk_stab_x,
        block_orientation == Orientation.HORIZONTAL,
        stab_schedule_dict_standard_swap_qec_default,
    )

    # Find the vectors to and from the final positions
    vec_from_final_pos = direction_to_coord(idle_side_directions, 0)

    # Initialize the cnots and find the first layer of them from
    cnots = [[] for _ in range(4)]
    for q in new_block.data_qubits:
        # q (final position) is in sublattice 0, while its current position is in
        # sublattice 1
        q_current_pos = tuple(map(sum, zip(q, vec_from_final_pos, strict=True)))

        if q in anc_qubits_to_init_final["X"]:
            cnot_pair = (q, q_current_pos)
        elif q in anc_qubits_to_init_final["Z"]:
            cnot_pair = (q_current_pos, q)
        else:
            raise ValueError("The ancilla qubit was not found.")
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
                continue
            if stab.pauli[0] == "Z":
                # If the stabilizer is Z, cnot from data qubit to ancilla
                cnot_pair = (data_qubit, stab.ancilla_qubits[0])
            elif stab.pauli[0] == "X":
                # If the stabilizer is X, cnot from ancilla to data qubit
                cnot_pair = (stab.ancilla_qubits[0], data_qubit)
            else:
                raise ValueError("Unknown stabilizer type.")
            cnots[i].append(cnot_pair)

    # Generate the circuit containing the cnots
    swap_then_qec_cnots_standard_swap_qec = Circuit(
        name=(
            f"Swap-then-QEC CNOTs for the standard swap-then-qec operation of block "
            f"{block.unique_label}"
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

    return swap_then_qec_cnots_standard_swap_qec
