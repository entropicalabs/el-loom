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
from loom.eka.utilities import (
    Direction,
    Orientation,
)
from loom.interpreter import InterpretationStep, Cbit

from loom_rotated_surface_code.code_factory import RotatedSurfaceCode

from .y_wall_out_utilities import qubit_mapper


def y_wall_out_measurement_and_hadamard(
    interpretation_step: InterpretationStep,
    block: RotatedSurfaceCode,
    wall_position: int,
    wall_orientation: Orientation,
    same_timeslice: bool,
    debug_mode: bool,
) -> InterpretationStep:
    """
    Measure a wall of qubits in the Y basis and apply Hadamard gates to the
    appropriate qubits to implement the y_wall_out operation. This is the core of
    a logical phase operation in the rotated surface code.

    - A.) Geometry

        - A.1) Find geometry-dependent variables
        - A.2) Find the qubits to measure, to idle, and to Hadamard

    - B.) Circuit Generation

        - B.1) Generate the circuit of the operation
        - B.2) Append circuit

    - C.) Stabilizer evolution and updates

        - C.1) Identify idle stabilizers that remain unchanged
        - C.2) Identify Hadamard stabilizers that change and create new ones
        - C.3) Reorder data qubits of new Hadamard stabilizers to match original \
            stabilizers order of data qubits per pauli type
        - C.4) Identify wall stabilizers that get merged and create new ones
        - C.5) Define stabilizer evolution and append to interpretation step
        - C.6) Define stabilizer updates and append to interpretation step

    - D.) Logical operator evolution and updates

        - D.1) Find new X logical operator and stabilizers for logical operator jump
        - D.2) Define logical X evolution
        - D.3) Define logical X updates
        
    - E.) Define new Block and update Block history and evolution


    Parameters
    ----------
    interpretation_step : InterpretationStep
        The interpretation step.
    block : RotatedSurfaceCode
        The block to which the operation will be applied.
    wall_position : int
        The position of the wall.
    wall_orientation : Orientation
        The orientation of the wall (horizontal or vertical).
    same_timeslice : bool
        Flag indicating whether the operation is part of the same timestep as the
        previous operation.
    debug_mode : bool
        Flag indicating whether the interpretation should be done in debug mode.
        Activating debug mode will enable commutation validation for Block.

    Returns
    -------
    InterpretationStep
        The updated interpretation step.
    """

    # A.) Geometry
    # A.1) Find geometry-dependent variables
    is_wall_hor = wall_orientation == Orientation.HORIZONTAL
    is_top_left_bulk_stab_x = block.upper_left_4body_stabilizer.pauli[0] == "X"
    # A.2) Find the qubits to measure, to idle, and to Hadamard
    qubits_to_measure, qubits_to_idle, qubits_to_hadamard = find_qubit_sets(
        block, wall_position, is_wall_hor
    )

    # B.) Circuit Generation
    # B.1) Generate the circuit of the operation
    circuit, y_meas_cbits = measure_y_and_hadamard_circuit(
        block, qubits_to_measure, qubits_to_hadamard, interpretation_step
    )
    # B.2) Append circuit
    interpretation_step.append_circuit_MUT(
        circuit,
        same_timeslice=same_timeslice,
    )

    # C.) Stabilizer evolution and updates
    new_stabilizers = find_new_stabilizers_and_define_evolution_and_updates(
        interpretation_step,
        block,
        qubits_to_measure,
        qubits_to_idle,
        qubits_to_hadamard,
        y_meas_cbits,
    )

    # D.) Logical operator evolution and updates
    # D.1) Find new X logical operator and stabilizers for logical operator jump
    new_x_log_op, stabilizers_for_x_operator_jump = find_new_x_logical_operator(
        block,
        is_top_left_bulk_stab_x,
        qubits_to_idle,
    )
    # D.2) Define logical X evolution
    interpretation_step.logical_x_evolution.update(
        {
            new_x_log_op.uuid: (block.logical_x_operators[0].uuid,),
        }
    )
    # D.3) Define logical X updates
    interpretation_step.update_logical_operator_updates_MUT(
        "X",
        new_x_log_op.uuid,
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

    # E.) Define new Block and update Block history and evolution
    block_without_y_wall_qubits = RotatedSurfaceCode(
        stabilizers=new_stabilizers,
        logical_x_operators=(new_x_log_op,),
        logical_z_operators=block.logical_z_operators,
        unique_label=block.unique_label,
        skip_validation=not debug_mode,
    )
    interpretation_step.update_block_history_and_evolution_MUT(
        (block_without_y_wall_qubits,), (block,)
    )

    return interpretation_step


def find_new_stabilizers_and_define_evolution_and_updates(
    # pylint: disable=too-many-locals
    interpretation_step: InterpretationStep,
    block: RotatedSurfaceCode,
    qubits_to_measure: list[tuple[int, ...]],
    qubits_to_idle: list[tuple[int, ...]],
    qubits_to_hadamard: list[tuple[int, ...]],
    y_meas_cbits: list[Cbit],
) -> list[Stabilizer]:
    """
    Find the new stabilizers after the Y wall measurement and Hadamard operation.
    Also define the stabilizer evolution and updates in the interpretation step.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        The interpretation step. Note that it may be mutated by generating new
        channels.
    block : RotatedSurfaceCode
        The block to which the operation will be applied.
    qubits_to_measure : list[tuple[int, ...]]
        The list of qubits to measure.
    qubits_to_idle : list[tuple[int, ...]]
        The list of qubits to idle.
    qubits_to_hadamard : list[tuple[int, ...]]
        The list of qubits to apply Hadamard gates to.
    y_meas_cbits : list[Cbit]
        The classical bits corresponding to the Y measurements.

    Returns
    -------
    list[Stabilizer]
        The new stabilizers after the Y wall measurement and Hadamard operation.
    """

    # Find vector from idle to hadamard side
    upper_left_idle_qubit = min(qubits_to_idle, key=lambda q: q[0] + q[1])
    upper_left_hadamard_qubit = min(qubits_to_hadamard, key=lambda q: q[0] + q[1])
    idle_to_had_dir = Direction.from_vector(
        (
            upper_left_hadamard_qubit[0] - upper_left_idle_qubit[0],
            upper_left_hadamard_qubit[1] - upper_left_idle_qubit[1],
            0,
        )
    )

    # C.1) Identify idle stabilizers that remain unchanged
    old_idle_stabilizers = [
        stab
        for stab in block.stabilizers
        if set(stab.data_qubits).issubset(qubits_to_idle)
    ]
    new_idle_stabilizers = old_idle_stabilizers

    # C.2) Identify Hadamard stabilizers that change and create new ones
    old_had_stabilizers = [
        stab
        for stab in block.stabilizers
        if set(stab.data_qubits).issubset(qubits_to_hadamard)
    ]
    new_had_stabilizers_with_unordered_data_qubits = [
        Stabilizer(
            pauli="".join({"X": "Z", "Z": "X"}[p] for p in stab.pauli),
            data_qubits=stab.data_qubits,
            ancilla_qubits=stab.ancilla_qubits,
        )
        for stab in old_had_stabilizers
    ]
    # C.3) Reorder data qubits of new Hadamard stabilizers to match original stabilizers
    # order of data qubits per pauli type
    # Take all BULK stabilizers and put their data qubits in the same order as the
    # original stabilizers that were in the position same of the initial block.
    # This is needed because this affects the syndrome extraction circuit.
    new_had_stabilizers = [
        Stabilizer(
            data_qubits=(
                next(
                    [qubit_mapper(q, idle_to_had_dir) for q in init_stab.data_qubits]
                    for init_stab in block.stabilizers
                    if set(
                        (
                            qubit_mapper(q, idle_to_had_dir)
                            for q in init_stab.data_qubits
                        )
                    )
                    == set(stab.data_qubits)
                )
                if len(stab.data_qubits) == 4
                else stab.data_qubits
            ),
            ancilla_qubits=stab.ancilla_qubits,
            pauli=stab.pauli,
        )
        for stab in new_had_stabilizers_with_unordered_data_qubits
    ]

    # C.4) Identify wall stabilizers that get merged and create new ones
    # Wall stabilizers get merged
    old_wall_stabilizers_idle = [
        stab
        for stab in block.stabilizers
        if set(stab.data_qubits).intersection(qubits_to_measure)
        and set(stab.data_qubits).intersection(qubits_to_idle)
        and len(stab.data_qubits) == 4
    ]
    old_wall_stabilizers_had = [
        stab
        for stab in block.stabilizers
        if set(stab.data_qubits).intersection(qubits_to_measure)
        and set(stab.data_qubits).intersection(qubits_to_hadamard)
        and len(stab.data_qubits) == 4
    ]
    # At this point, we need to order the hadamard wall stabilizers so that they
    # correspond to the idle wall stabilizers
    old_wall_stabilizers_had_ordered = [
        old_h_stab
        for old_i_stab in old_wall_stabilizers_idle
        for old_h_stab in old_wall_stabilizers_had
        if len(set(old_h_stab.data_qubits).intersection(old_i_stab.data_qubits)) == 2
    ]

    # Create new wall stabilizers by merging the idle and hadamard wall stabilizers
    new_wall_stabilizers = []
    for old_i_stab in old_wall_stabilizers_idle:
        new_stab_qubits = [
            (
                q
                if q not in qubits_to_measure
                else (
                    q[0] + idle_to_had_dir.to_vector()[0],
                    q[1] + idle_to_had_dir.to_vector()[1],
                    q[2],
                )
            )
            for q in old_i_stab.data_qubits
        ]
        new_stab = Stabilizer(
            old_i_stab.pauli,
            new_stab_qubits,
            ancilla_qubits=old_i_stab.ancilla_qubits,
        )
        new_wall_stabilizers.append(new_stab)

    # C.5) Define stabilizer evolution and append to interpretation step
    stab_evolution = {}
    stab_evolution |= {
        new_stab.uuid: (old_stab.uuid,)
        for old_stab, new_stab in zip(
            old_had_stabilizers + old_idle_stabilizers,
            new_had_stabilizers + new_idle_stabilizers,
            strict=True,
        )
    }
    stab_evolution |= {
        new_wall_stab.uuid: (old_wall_i_stab.uuid, old_wall_h_stab.uuid)
        for old_wall_i_stab, old_wall_h_stab, new_wall_stab in zip(
            old_wall_stabilizers_idle,
            old_wall_stabilizers_had_ordered,
            new_wall_stabilizers,
            strict=True,
        )
    }
    interpretation_step.stabilizer_evolution.update(stab_evolution)

    # C.6) Define stabilizer updates and append to interpretation step
    for stab_new, stab_old_idle in zip(
        new_wall_stabilizers,
        old_wall_stabilizers_idle,
        strict=True,
    ):
        qubits_measured = set(stab_old_idle.data_qubits).intersection(qubits_to_measure)
        # Find cbits from y measurements associated with the stabilizer
        cbits = tuple(
            cbit
            for cbit, qubit_measured in zip(y_meas_cbits, qubits_to_measure)
            if qubit_measured in qubits_measured
        )
        current_updates = interpretation_step.stabilizer_updates.get(stab_new.uuid, ())
        # Add the cbits to the stabilizer updates and add 1 that stems from the
        # Y measurement that joins the 2 stabilizers on each side of the wall
        interpretation_step.stabilizer_updates[stab_new.uuid] = (
            current_updates + cbits + (1,)
        )

    return new_had_stabilizers + new_idle_stabilizers + new_wall_stabilizers


def measure_y_and_hadamard_circuit(
    block: RotatedSurfaceCode,
    qubits_to_measure: list[tuple[int, ...]],
    qubits_to_hadamard: list[tuple[int, ...]],
    interpretation_step: InterpretationStep,
) -> tuple[Circuit, list[Cbit]]:
    """
    Generates a circuit that measures all the data qubits of the wall in the Y basis
    and applies Hadamard gates to the appropriate qubits. It also returns the classical
    bits corresponding to the Y measurements.

    Parameters
    ----------
    block : RotatedSurfaceCode
        The block to which the operation will be applied.
    qubits_to_measure : list[tuple[int, ...]]
        The list of qubits to measure.
    qubits_to_hadamard : list[tuple[int, ...]]
        The list of qubits to apply Hadamard gates to.
    interpretation_step : InterpretationStep
        The interpretation step. Note that it may be mutated by generating new
        channels.

    Returns
    -------
    Circuit
        The circuit to create the Y wall.
    list[Cbit]
        The classical bits corresponding to the Y measurements.
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
    y_wall_measurement_circuit = Circuit(
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

    hadamard_circuit = Circuit(
        name=(
            f"transversal hadamard on the data qubits for the y_wall_out operation "
            f"of block {block.unique_label}"
        ),
        circuit=[
            [
                Circuit("h", channels=[interpretation_step.get_channel_MUT(q)])
                for q in qubits_to_hadamard
            ]
        ],
    )

    circuit = Circuit(
        name="y_wall_out - y wall measurement and hadamard",
        circuit=((y_wall_measurement_circuit, hadamard_circuit),),
    )

    return circuit, cbits


def find_new_x_logical_operator(
    block: RotatedSurfaceCode,
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
    match (block.is_horizontal, is_top_left_bulk_stab_x):
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

    # Find the qubits to measure
    qubits_to_measure = (
        [(q[0], q[1] + wall_position, 0) for q in block.boundary_qubits("top")]
        if is_wall_hor
        else [(q[0] + wall_position, q[1], 0) for q in block.boundary_qubits("left")]
    )
    # Find the qubits to idle (top or left qubits depending on the orientation)
    qubits_to_idle = [
        (q[0] + idle_side_unit_vector[0] * d, q[1] + idle_side_unit_vector[1] * d, q[2])
        for q in qubits_to_measure
        for d in range(1, wall_position + 1)
    ]
    # Find the qubits to Hadamard
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
