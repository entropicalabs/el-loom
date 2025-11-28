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

from loom.eka import Circuit, Stabilizer, PauliOperator
from loom.eka.utilities import Direction, DiagonalDirection
from loom.interpreter import InterpretationStep

from ..move_block import DetailedSchedule, direction_to_coord
from ..utilities import FourBodySchedule, find_relative_diagonal_direction
from ...code_factory import RotatedSurfaceCode

# pylint: disable=duplicate-code


def qubit_mapper(
    q: tuple[int, ...], move_direction: DiagonalDirection
) -> tuple[int, ...]:
    """
    Map qubit to appropriate diagonal qubit after movement.

    Parameters
    ----------
    q : tuple[int, ...]
        The original qubit coordinates.
    move_direction : DiagonalDirection
        The direction of movement.

    Returns
    -------
    tuple[int, ...]
        The new qubit coordinates after movement.
    """
    return tuple(
        map(
            sum,
            zip(
                q,
                direction_to_coord(move_direction, q[2]),
                strict=True,
            ),
        )
    )


def move_logicals_and_append_evolution_and_updates(
    interpretation_step: InterpretationStep,
    block: RotatedSurfaceCode,
    move_direction: DiagonalDirection,
) -> tuple[PauliOperator, PauliOperator]:
    """
    Move logical operators diagonally and append their evolution and updates
    to the interpretation step.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        The interpretation step.
    block : RotatedSurfaceCode
        The rotated surface code block whose logicals are to be moved.
    move_direction : DiagonalDirection
        The direction of movement.

    Returns
    -------
    tuple[PauliOperator, PauliOperator]
        The new logical X and Z operators.
    """
    # Find new logicals
    old_logical_x = block.logical_x_operators[0]
    new_logical_x = PauliOperator(
        pauli=old_logical_x.pauli,
        data_qubits=[
            qubit_mapper(dq, move_direction) for dq in old_logical_x.data_qubits
        ],
    )
    old_logical_z = block.logical_z_operators[0]
    new_logical_z = PauliOperator(
        pauli=old_logical_z.pauli,
        data_qubits=[
            qubit_mapper(dq, move_direction) for dq in old_logical_z.data_qubits
        ],
    )

    # Update evolution
    interpretation_step.logical_x_evolution[new_logical_x.uuid] = (old_logical_x.uuid,)
    interpretation_step.logical_z_evolution[new_logical_z.uuid] = (old_logical_z.uuid,)

    # Update logical updates
    interpretation_step.update_logical_operator_updates_MUT(
        "X",
        new_logical_x.uuid,
        (),
        inherit_updates=True,
    )
    interpretation_step.update_logical_operator_updates_MUT(
        "Z", new_logical_z.uuid, (), inherit_updates=True
    )

    return new_logical_x, new_logical_z


def generate_teleportation_finalization_circuit_with_updates(
    interpretation_step: InterpretationStep,
    new_block: RotatedSurfaceCode,
    anc_qubits_to_init: dict[str, list[tuple[int, ...]]],
    teleportation_qubit_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]],
) -> Circuit:
    """
    Generate the circuit that finalizes the teleportation operation. This includes
    the measurement of the data qubits and the updating of the necessary stabilizers
    and logical operators based on the measurement outcomes to complete the
    teleportation without needing to apply any gates.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        The interpretation step.
    new_block : RotatedSurfaceCode
        The new block after the operation.
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

        # Do the same for the logical operator (it's going to be only one operator)
        logical_ops = (
            new_block.logical_z_operators
            if stabs_to_update_pauli == "Z"
            else new_block.logical_x_operators
        )
        logical_ops_involved = [
            op for op in logical_ops if corrected_qubit in op.data_qubits
        ]
        for op in logical_ops_involved:
            interpretation_step.update_logical_operator_updates_MUT(
                stabs_to_update_pauli, op.uuid, (cbit,), False
            )

        # Find all appropriate stabilizers
        stabs_to_update = [
            stab
            for stab in new_block.stabilizers
            if stab.pauli[0] == stabs_to_update_pauli  # correct flavor
            and corrected_qubit in stab.data_qubits  # qubit part of stabilizer
        ]

        # Append the Cbit on the updates of the stabilizers
        for stab in stabs_to_update:
            current_upd = interpretation_step.stabilizer_updates.get(stab.uuid, ())
            interpretation_step.stabilizer_updates[stab.uuid] = current_upd + (cbit,)

    # Compile the teleportation circuit finalization
    teleportation_circuit_finalization = Circuit(
        "teleportation finalization",
        circuit=teleportation_circ_seq,
    )

    return teleportation_circuit_finalization


def get_final_block_syndrome_measurement_cnots_circuit(
    block: RotatedSurfaceCode,
    interpretation_step: InterpretationStep,
    is_swap_then_qec: bool = False,
    idle_side_directions: DiagonalDirection | None = None,
    anc_qubits_to_init: dict[str, list[tuple[int, ...]]] | None = None,
) -> Circuit:
    # pylint: disable=too-many-locals, too-many-branches
    """
    Generates fault-tolerant CNOT scheduling for syndrome measurement of
    the final block. The block is divided into regions as explained below.

    Examples: a 7x13 block with three lines corresponding to three ways
    for dividing the block::

                3 * * * * * 4
                *         . *
                * .     .   *
                *   . .     *
                *   . .     *
                * .     .   *
                2         . *
                * .         o
                *   .       *
                *     .     *
                *       .   *
                *         . *
                1 * * * * * *

    - Region 1: draw a diagonal line from qubit o towards corner 3, note that the \
    line doesn't pass through corner 3. Qubit o is the qubit shifted by 1 from the \
    opposite of corner 2 away from corner 4. Region 2 is defined as the region \
    containing corner 4.
    - Region 2: draw a diagonal line from corner 2 (middle-edge corner) to corner 4 \
    (short end corner). Region 2 is defined as the region containing corner 3 \
    (angle corner).
    - Region 3: draw a diagonal line from corner 2 to the non-topological geometric \
    corner. Region 3 is defined as the region containing corner 1.
    - Region 4: qubits not contained in region 1, 2, or 3.

    The scheduling algorithm is the following.

    - Z stabilizers outside region 3 are scheduled so that Z errors propagate \
    perpendicular to the edge connecting corners 3 and 4.
    - Z stabilizers inside region 3 are scheduled opposite to Z stabilizers \
    outside the region.
    - X stabilizers inside regions 1 and 2 are scheduled opposite to the schedule of Z \
    stabilizers outside region 3.
    - X stabilizers outside regions 1 and 2 are scheduled opposite to the schedule of \
    X stabilizers inside regions 1 and 2.

    Parameters
    ----------
    block : RotatedSurfaceCode
        The final block after the y_wall_out operation.
    interpretation_step : InterpretationStep
        The interpretation step for the circuit generation.
    is_swap_then_qec: bool
        True if the syndrome measurement round is a SWAP-then-QEC round
    idle_side_directions: DiagonalDirection
        The directions that the idling side half of the block will move.
    anc_qubits_to_init: dict[str, list[tuple[int, ...]]]
        The ancilla qubits to initialize in the second SWAP-then-QEC round.

    Returns
    -------
    Circuit
        The generated CNOT circuit for the syndrome measurement of the final block.

    """
    config, pivot_corners = block.config_and_pivot_corners
    if config != 3:
        raise ValueError(
            "The final block of Y-wall-out must have type-3 corner configuration."
        )

    short_end_corner = pivot_corners[3]
    top_left_corner = block.upper_left_qubit
    is_horizontal = block.is_horizontal

    # The starting qubit direction is the direction of the short-end corner
    starting_diag_direction = find_relative_diagonal_direction(
        top_left_corner, short_end_corner
    )

    # Schedules
    # Z schedule is determined based on the orientation of the short edge
    region_124_z_schedule = FourBodySchedule.Z if is_horizontal else FourBodySchedule.N
    region_3_z_schedule = region_124_z_schedule.opposite_schedule()
    region_12_x_schedule = region_124_z_schedule.opposite_schedule()
    region_34_x_schedule = region_12_x_schedule.opposite_schedule()

    # Determine cnot sequence, require 5 time steps
    cnots = [[] for _ in range(5)]

    if is_swap_then_qec:
        vec_from_final_pos = direction_to_coord(idle_side_directions, 0)
        for q in block.data_qubits:
            # q (final position) is in sublattice 0, while its current position is in
            # sublattice 1
            q_current_pos = tuple(map(sum, zip(q, vec_from_final_pos, strict=True)))

            if q in anc_qubits_to_init["X"]:
                cnot_pair = (q, q_current_pos)
            elif q in anc_qubits_to_init["Z"]:
                cnot_pair = (q_current_pos, q)
            else:
                raise ValueError("The ancilla qubit was not found.")
            cnots[0].append(cnot_pair)

    for stab in block.stabilizers:
        pauli_type = stab.pauli_type
        (
            is_in_region_12,
            is_in_region_3,
            is_on_boundary_12,
            _,
            _,
            boundary_direction,
        ) = find_stabilizer_position_in_final_block(block, stab, pivot_corners)
        if pauli_type == "Z":
            schedule = region_3_z_schedule if is_in_region_3 else region_124_z_schedule
        else:
            schedule = region_12_x_schedule if is_in_region_12 else region_34_x_schedule

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
            if (is_swap_then_qec and i == 0) or data_qubit is None:
                continue
            if pauli_type == "Z":
                # If the stabilizer is Z, cnot from data qubit to ancilla
                cnot_pair = (data_qubit, stab.ancilla_qubits[0])
            elif pauli_type == "X":
                # If the stabilizer is X, cnot from ancilla to data qubit
                cnot_pair = (stab.ancilla_qubits[0], data_qubit)
            else:
                raise ValueError("Unknown stabilizer type.")

            # Determine the cnot layer
            layer = (
                i + 1
                if (
                    i > 0
                    and (
                        (
                            not is_in_region_12 and not is_on_boundary_12
                        )  # bulk of region 3 and 4
                        or (
                            not is_in_region_12 and is_on_boundary_12 and i == 3
                        )  # boundary of region 4
                    )
                )
                else i
            )
            cnots[layer].append(cnot_pair)

    # Generate the circuit containing the cnots
    cnots_circuit = Circuit(
        name="Final block syndrome measurement CNOT circuit",
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


def find_stabilizer_position_in_final_block(
    block: RotatedSurfaceCode,
    stab: Stabilizer,
    pivot_corners: tuple[
        tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
    ],
) -> tuple[bool, bool, bool, bool, bool, Direction | None]:
    # pylint: disable=too-many-locals
    """
    Find stabilizer position with respect to the regions of the final block

    Parameters
    ----------
    block: RotatedSurfaceCode
        The final block after the y_wall_out operation.
    stab: Stabilizer
        The input stabilizer.
    pivot_corners:
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]
        Four topological corners.

    Returns
    -------
    tuple[bool, bool, bool, Direction | None]
        True if the stabilizer is inside region 1 or 2
        True if the stabilizer is inside region 3
        True if the stabilizer is inside region 4 and next to region 1 or 2
        True if the stabilizer in on the wall
        True if the stabilizer in the Hadamard side
        Direction of the boundary if the input stabilizer is a weight-2 stabilizer

    """

    long_end_corner, middle_corner, _, short_end_corner = pivot_corners
    is_horizontal = block.is_horizontal
    long_edge_idx = 0 if is_horizontal else 1  # coordinate index along the long side
    short_edge_idx = 1 if is_horizontal else 0  # coordinate index along the short side

    def is_in_region_1(q) -> bool:
        """Determine if the input qubit is inside region 1
        (including the boundaries of the region). This is done by checking
        whether the distance along the long side of the block between the input
        qubit and the qubit o is at least the distance along the
        short side between the same two qubits.
        """
        # Compare middle_corner and short_end_corner along the short edge to determine
        # if the qubit is on the left or right of middle_corner.
        # Distance is calculated with respect to short_end_corner because it has the
        # same (short-edge) coordinate as qubit o.
        dshort = (
            q[short_edge_idx] - short_end_corner[short_edge_idx]
            if middle_corner[short_edge_idx] > short_end_corner[short_edge_idx]
            else short_end_corner[short_edge_idx] - q[short_edge_idx]
        )
        # Compare middle_corner and long_end_corner along the long edge to determine
        # if the qubit is on the top or bottom of middle_corner.
        # Distance is calculated with respect to middle_corner because the (long-edge)
        # coordinate of qubit o can be deduced from it.
        dlong = (
            middle_corner[long_edge_idx] - q[long_edge_idx] + 1
            if middle_corner[long_edge_idx] < long_end_corner[long_edge_idx]
            else q[long_edge_idx] - middle_corner[long_edge_idx] + 1
        )
        return 0 <= dshort <= dlong and dlong >= 0

    def is_in_region_2(q) -> bool:
        """Determine if the input qubit is inside region 2
        (including the boundaries of the region). This is done by checking
        whether the distance along the long side of the block between the input
        qubit and the middle_corner is at least the distance along the
        short side between the same two qubits.
        """
        # Compare middle_corner and short_end_corner along the short edge to determine
        # if the qubit is on the left or right of middle_corner.
        # Distance is calculated with respect to middle_corner
        dshort = (
            q[short_edge_idx] - middle_corner[short_edge_idx]
            if middle_corner[short_edge_idx] < short_end_corner[short_edge_idx]
            else middle_corner[short_edge_idx] - q[short_edge_idx]
        )
        # Compare middle_corner and long_end_corner along the long edge to determine
        # if the qubit is on the top or bottom of middle_corner.
        # Distance is calculated with respect to middle_corner
        dlong = (
            middle_corner[long_edge_idx] - q[long_edge_idx]
            if middle_corner[long_edge_idx] < long_end_corner[long_edge_idx]
            else q[long_edge_idx] - middle_corner[long_edge_idx]
        )
        return 0 <= dshort <= dlong and dlong >= 0

    def is_in_region_3(q) -> bool:
        """Determine if the input qubit is inside region 3
        (including the boundaries of the region). This is done by checking
        whether the distance along the long side of the block between the input
        qubit and the qubit o is at least the distance along the
        short side between the same two qubits.
        """
        # Compare middle_corner and short_end_corner along the short edge to determine
        # if the qubit is on the left or right of middle_corner.
        # Distance is calculated with respect to middle_corner
        dshort = (
            q[short_edge_idx] - middle_corner[short_edge_idx]
            if middle_corner[short_edge_idx] < short_end_corner[short_edge_idx]
            else middle_corner[short_edge_idx] - q[short_edge_idx]
        )
        # Compare middle_corner and long_end_corner along the long edge to determine
        # if the qubit is on the top or bottom of middle_corner.
        # Distance is calculated with respect to middle_corner
        # Note that region 3 is on the opposite side (along the long edge) of region 2
        # with respect to middle_corner.
        dlong = (
            q[long_edge_idx] - middle_corner[long_edge_idx]
            if middle_corner[long_edge_idx] < long_end_corner[long_edge_idx]
            else middle_corner[long_edge_idx] - q[long_edge_idx]
        )
        return 0 <= dshort <= dlong and dlong >= 0

    dq_in_region_1 = [dqubit for dqubit in stab.data_qubits if is_in_region_1(dqubit)]
    dq_in_region_2 = [dqubit for dqubit in stab.data_qubits if is_in_region_2(dqubit)]
    dq_in_region_3 = [dqubit for dqubit in stab.data_qubits if is_in_region_3(dqubit)]

    is_stab_in_region_1 = len(dq_in_region_1) == len(stab.data_qubits)
    is_stab_in_region_2 = len(dq_in_region_2) == len(stab.data_qubits)
    is_stab_in_region_3 = len(dq_in_region_3) == len(stab.data_qubits)
    is_stab_in_region_4 = not (
        is_stab_in_region_1 or is_stab_in_region_2 or is_stab_in_region_3
    )

    is_stab_in_region_12 = is_stab_in_region_1 or is_stab_in_region_2

    is_stab_on_boundary_region_1 = (len(stab.data_qubits) - len(dq_in_region_1)) == 1
    is_stab_on_boundary_region_2 = (len(stab.data_qubits) - len(dq_in_region_2)) == 1
    is_stab_in_region_4_and_on_boundary_region_12 = is_stab_in_region_4 and (
        is_stab_on_boundary_region_1 or is_stab_on_boundary_region_2
    )

    # Find the stabilizer's position relative to the wall
    top_left_qubit = max(stab.data_qubits, key=lambda x: -x[0] - x[1])
    bottom_right_qubit = max(stab.data_qubits, key=lambda x: x[0] + x[1])
    compare_qubit = (
        top_left_qubit
        if middle_corner[long_edge_idx] < long_end_corner[long_edge_idx]
        else bottom_right_qubit
    )
    is_on_wall = middle_corner[long_edge_idx] == compare_qubit[long_edge_idx]
    is_on_hadamard_side = (
        compare_qubit[long_edge_idx] > middle_corner[long_edge_idx]
        if long_end_corner[long_edge_idx] > middle_corner[long_edge_idx]
        else compare_qubit[long_edge_idx] < middle_corner[long_edge_idx]
    )

    # Find the boundary direction of weight-2 stabilizer
    boundary = None
    weight = len(stab.pauli)
    if weight == 2:
        qubit_directions = [
            set(
                direction
                for direction in Direction
                if dqubit in block.boundary_qubits(direction)
            )
            for dqubit in stab.data_qubits
        ]
        boundary = list(qubit_directions[0].intersection(qubit_directions[1]))[0]

    return (
        is_stab_in_region_12,
        is_stab_in_region_3,
        is_stab_in_region_4_and_on_boundary_region_12,
        is_on_wall,
        is_on_hadamard_side,
        boundary,
    )
