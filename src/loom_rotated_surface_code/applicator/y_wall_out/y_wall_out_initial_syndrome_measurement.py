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

from loom.eka import Circuit, Stabilizer
from loom.eka.utilities import Direction
from loom.interpreter import InterpretationStep

from .y_wall_out_utilities import find_relative_diagonal_direction
from ..move_block import (
    DetailedSchedule,
    generate_syndrome_measurement_circuit_and_cbits,
    generate_and_append_block_syndromes_and_detectors,
)
from ..utilities import FourBodySchedule
from ...code_factory import RotatedSurfaceCode


def y_wall_out_initial_syndrome_measurement(
    interpretation_step: InterpretationStep,
    block: RotatedSurfaceCode,
    same_timeslice: bool,
    debug_mode: bool,  # pylint: disable=unused-argument
) -> InterpretationStep:
    """
    Generates and appends the initial syndrome measurement circuit for the
    Y-wall-out applicator. This syndrome measurement is performed on the
    initial block before the y-wall-out operation and is crafted such that the whole
    operation remains fault-tolerant.


    Parameters
    ----------
    interpretation_step : InterpretationStep
        The interpretation step.
    block : RotatedSurfaceCode
        The block to which the operation will be applied.
    same_timeslice : bool
        Flag indicating whether the operation is part of the same timestep as the
        previous operation.
    debug_mode : bool
        Flag indicating whether the interpretation should be done in debug mode.
        Activating debug mode will enable commutation validation for Block.
        (Currently unused since no new Block is created here.)

    Returns
    -------
    InterpretationStep
        The updated interpretation step with the initial syndrome measurement
    """

    # Initialization circuit
    init_block_syndrome_measurement_reset_circuit = Circuit(
        name="Initialization of syndrome measurement ancilla",
        circuit=[
            [
                Circuit(
                    f"reset_{'0' if stab.pauli_type == 'Z' else '+'}",
                    channels=[
                        interpretation_step.get_channel_MUT(stab.ancilla_qubits[0])
                    ],
                )
                for stab in block.stabilizers
            ]
        ],
    )

    # CNOTs circuit
    init_block_syndrome_measurement_cnot_circuit = (
        get_initial_block_syndrome_measurement_cnots_circuit(block, interpretation_step)
    )

    # Measurements and generation of cbits
    # (their order matches the stabilizer order in the block)
    init_block_syndrome_measurement_measure_circuit, init_cbits = (
        generate_syndrome_measurement_circuit_and_cbits(
            interpretation_step,
            block,
        )
    )

    # Assemble and append circuit
    interpretation_step.append_circuit_MUT(
        Circuit(
            name="Y wall out - Initial block syndrome measurement",
            circuit=(
                init_block_syndrome_measurement_reset_circuit,
                init_block_syndrome_measurement_cnot_circuit,
                init_block_syndrome_measurement_measure_circuit,
            ),
        ),
        same_timeslice=same_timeslice,
    )

    # Generate and append syndromes and detectors
    generate_and_append_block_syndromes_and_detectors(
        interpretation_step, block, init_cbits
    )

    return interpretation_step


def get_initial_block_syndrome_measurement_cnots_circuit(
    # pylint: disable=too-many-locals
    block: RotatedSurfaceCode,
    interpretation_step: InterpretationStep,
) -> Circuit:
    """
    Generates fault-tolerant CNOT scheduling for syndrome measurement of
    the initial block.
    The block is divided into regions as explained below.

    Examples: a 7x14 block with three lines corresponding to three ways
    for dividing the block::

                3 * * * * * 4
                *         . *
                * .     .   *
                *   . .     *
                *   . .     *
                * .     .   *
                o         . *
                * .         2
                *   .       *
                *     .     *
                *       .   *
                *         . *
                *           *
                1 * * * * * *

    - Region 1: draw a diagonal line from corner 2 (middle-edge corner) \
    towards corner 3 (angle corner), note that the line doesn't pass through \
    corner 3). The line split the data qubits into two regions. \
    Region 1 is defined as the region containing corner 4 (short-end corner).
    - Region 2: draw a diagonal line from qubit o to corner 4. Qubit o is the qubit \
    shifted by 1 from the opposite of corner 2 towards corner 3. Region 2 is defined \
     as the region containing corner 3.
    - Region 3: draw a diagonal line from qubit o towards the non-topological \
    geometric corner. Region 3 is defined as the region containing corner 1.
    - Region 4: qubits not contained in region 1, 2, or 3.

    The scheduling algorithm is the following.

    - Z stabilizers have the same schedule across the block. The schedule is such that \
    Z errors propagating perpendicular to the edge connecting corners 3 and 4.
    - X stabilizers in region 1, 2, and 3 are scheduled opposite to the schedule of \
    Z stabilizers.
    - X stabilizers in region 4 are scheduled opposite to the schedule of X \
    stabilizers in region 1, 2, and 3.

    Parameters
    ----------
    block : RotatedSurfaceCode
        The block to which the operation will be applied.
    interpretation_step : InterpretationStep
        The interpretation step for the circuit generation.

    Returns
    -------
    Circuit
        The generated CNOT circuit for the syndrome measurement of the initial block.

    """
    config, pivot_corners = block.config_and_pivot_corners
    if config != 4:
        raise ValueError(
            "The initial block of Y-wall-out must have type-4 corner configuration."
        )

    long_end_corner = pivot_corners[0]
    top_left_corner = block.upper_left_qubit
    is_horizontal = block.is_horizontal

    # The starting qubit direction is the direction of the long-end corner
    starting_diag_direction = find_relative_diagonal_direction(
        top_left_corner, long_end_corner
    )

    # Schedules
    # Z schedule is determined based on the orientation of the short edge
    z_schedule = FourBodySchedule.Z if is_horizontal else FourBodySchedule.N
    region_123_x_schedule = z_schedule.opposite_schedule()
    region_4_x_schedule = region_123_x_schedule.opposite_schedule()

    # Determine cnot sequence, require 5 time steps
    cnots = [[] for _ in range(5)]
    for stab in block.stabilizers:
        pauli_type = stab.pauli_type
        (
            is_in_region_4,
            is_on_boundary_12_and_in_region_4,
            is_closer_long_end_corner,
            boundary_direction,
        ) = find_stabilizer_position_in_initial_block(block, stab, pivot_corners)
        if is_in_region_4:
            schedule = region_4_x_schedule if pauli_type == "X" else z_schedule
        else:
            schedule = region_123_x_schedule if pauli_type == "X" else z_schedule

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
            if data_qubit is None:
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
                    # long end corner half
                    is_closer_long_end_corner
                    # bulk of region 4
                    or (is_in_region_4 and not is_on_boundary_12_and_in_region_4)
                    # boundary of region 4
                    or (
                        is_in_region_4
                        and is_on_boundary_12_and_in_region_4
                        and i in [2, 3]
                    )
                )
                else i
            )
            cnots[layer].append(cnot_pair)

    # Generate the circuit containing the cnots
    cnots_circuit = Circuit(
        name="Initial block syndrome measurement CNOT circuit",
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


def find_stabilizer_position_in_initial_block(
    block: RotatedSurfaceCode,
    stab: Stabilizer,
    pivot_corners: tuple[
        tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
    ],
) -> tuple[bool, bool, bool, Direction | None]:
    # pylint: disable=too-many-locals
    """
    Find stabilizer position with respect to the regions of the initial block

    Parameters
    ----------
    block: RotatedSurfaceCode
        The initial block
    stab: Stabilizer
        The input stabilizer
    pivot_corners:
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]
        Four topological corners

    Returns
    -------
    tuple[bool, bool, bool, Direction | None]
        True if the stabilizer is inside region 4
        True if the stabilizer is inside region 4 and next to region 1 or 2
        True if the stabilizer is in the half of the block containing corner 1 and 2
        Direction of the boundary if the input stabilizer is a weight-2 stabilizer

    """

    long_end_corner, middle_corner = pivot_corners[0], pivot_corners[1]
    is_horizontal = block.is_horizontal
    long_edge_idx = 0 if is_horizontal else 1  # coordinate index along the long side
    short_edge_idx = 1 if is_horizontal else 0  # coordinate index along the short side

    def is_in_region_1(q) -> bool:
        """Determine if the input qubit is inside region 1
        (including the boundaries of the region). This is done by checking
        whether the distance along the long side of the block between the input
        qubit and the middle-edge corner is at least the distance along the
        short side between the same two qubits.
        """
        # Compare middle_corner and long_end_corner along the short edge to determine
        # if the qubit is on the left or right of middle_corner
        # Distance is calculated with respect to middle_corner
        dshort = (
            middle_corner[short_edge_idx] - q[short_edge_idx]
            if middle_corner[short_edge_idx] > long_end_corner[short_edge_idx]
            else q[short_edge_idx] - middle_corner[short_edge_idx]
        )
        # Compare middle_corner and long_end_corner along the long edge to determine
        # if the qubit is on the top or bottom of middle_corner
        # Distance is calculated with respect to middle_corner
        dlong = (
            middle_corner[long_edge_idx] - q[long_edge_idx]
            if middle_corner[long_edge_idx] < long_end_corner[long_edge_idx]
            else q[long_edge_idx] - middle_corner[long_edge_idx]
        )
        return 0 <= dshort <= dlong and dlong >= 0

    def is_in_region_2(q) -> bool:
        """Determine if the input qubit is inside region 2
        (including the boundaries of the region). This is done by checking
        whether the distance along the long side of the block between the input
        qubit and the qubit o is at least the distance along the
        short side between the same two qubits.
        """
        # Compare middle_corner and long_end_corner along the short edge to determine
        # if the qubit is on the left or right of middle_corner.
        # Distance is calculated with respect to long_end_corner because it has the
        # same (short-edge) coordinate as qubit o.
        dshort = (
            q[short_edge_idx] - long_end_corner[short_edge_idx]
            if middle_corner[short_edge_idx] > long_end_corner[short_edge_idx]
            else long_end_corner[short_edge_idx] - q[short_edge_idx]
        )
        # Compare middle_corner and long_end_corner along the long edge to determine
        # if the qubit is on the top or bottom of middle_corner.
        # Distance is calculated with respect to middle_corner because the (long-edge)
        # coordinate of qubit o can be deduced from it.
        dlong = (
            middle_corner[long_edge_idx] - q[long_edge_idx] - 1
            if middle_corner[long_edge_idx] < long_end_corner[long_edge_idx]
            else q[long_edge_idx] - middle_corner[long_edge_idx] - 1
        )
        return 0 <= dshort <= dlong and dlong >= 0

    def is_in_region_3(q) -> bool:
        """Determine if the input qubit is inside region 2
        (including the boundaries of the region). This is done by checking
        whether the distance along the long side of the block between the input
        qubit and the qubit o is at least the distance along the
        short side between the same two qubits.
        """
        # Compare middle_corner and long_end_corner along the short edge to determine
        # if the qubit is on the left or right of middle_corner.
        # Distance is calculated with respect to long_end_corner because it has the
        # same (short-edge) coordinate as qubit o.
        dshort = (
            q[short_edge_idx] - long_end_corner[short_edge_idx]
            if middle_corner[short_edge_idx] > long_end_corner[short_edge_idx]
            else long_end_corner[short_edge_idx] - q[short_edge_idx]
        )
        # Compare middle_corner and long_end_corner along the long edge to determine
        # if the qubit is on the top or bottom of middle_corner.
        # Distance is calculated with respect to middle_corner because the (long-edge)
        # coordinate of qubit o can be deduced from it.
        # Note that region 3 is on the opposite side (along the long edge) of region 2
        # with respect to qubit o.
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

    is_stab_on_boundary_region_1 = (len(stab.data_qubits) - len(dq_in_region_1)) == 1
    is_stab_on_boundary_region_2 = (len(stab.data_qubits) - len(dq_in_region_2)) == 1
    is_stab_in_region_4_and_on_boundary_region_12 = is_stab_in_region_4 and (
        is_stab_on_boundary_region_1 or is_stab_on_boundary_region_2
    )

    # Find if the stabilizer is near the long-end corner
    top_left_qubit = max(stab.data_qubits, key=lambda x: -x[0] - x[1])
    bottom_right_qubit = max(stab.data_qubits, key=lambda x: x[0] + x[1])
    compare_qubit = (
        top_left_qubit
        if middle_corner[long_edge_idx] < long_end_corner[long_edge_idx]
        else bottom_right_qubit
    )
    is_closer_long_end_corner = (
        middle_corner[long_edge_idx]
        <= compare_qubit[long_edge_idx]
        <= long_end_corner[long_edge_idx]
        if long_end_corner[long_edge_idx] > middle_corner[long_edge_idx]
        else long_end_corner[long_edge_idx]
        <= compare_qubit[long_edge_idx]
        <= middle_corner[long_edge_idx]
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
        is_stab_in_region_4,
        is_stab_in_region_4_and_on_boundary_region_12,
        is_closer_long_end_corner,
        boundary,
    )
