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
from loom.eka.utilities import DiagonalDirection
from loom.interpreter import InterpretationStep

from .y_wall_out_utilities import (
    generate_teleportation_finalization_circuit_with_updates,
    qubit_mapper,
    get_final_block_syndrome_measurement_cnots_circuit,
    move_logicals_and_append_evolution_and_updates,
)
from ..move_block import (
    find_swap_then_qec_qubit_initializations,
    generate_syndrome_measurement_circuit_and_cbits,
    generate_and_append_block_syndromes_and_detectors,
)

from ..utilities import (
    generate_syndrome_extraction_circuits,
    find_relative_diagonal_direction,
)
from ...code_factory import RotatedSurfaceCode


def y_wall_out_final_swap_then_qec(
    interpretation_step: InterpretationStep,
    block: RotatedSurfaceCode,
    idle_side_directions: DiagonalDirection,
    same_timeslice: bool,
    debug_mode: bool,
) -> InterpretationStep:
    """
    Apply the final swap-then-QEC operation for Y-wall-out recombination. By the end
    of this operation, the block has been aligned with its original position before the
    y_wall_out operation in a fault-tolerant manner.

    The operation consists of the following steps:

    A.) Stabilizer evolution and updates

        - A.1) Move stabilizers and define evolution in the interpretation step
        - A.2) Update stabilizer updates in interpretation step

    B.) Logical operators evolution and updates

    C.) Generate final Block and appropriate SyndromeCircuits

        C.1) Create a mock block to find the new syndrome circuits
        C.2) Find the starting qubit direction
        C.3) Find syndrome circuits using the mock block
        C.4) Create the new block with the new syndrome circuits

    D.) Circuit Generation

        D.1) Find the qubits to initialize (data and ancilla) and teleportation pairs
        D.2) Generate the initialization circuit
        D.3) Generate the CNOTs circuit for syndrome measurement
        D.4) Generate the teleportation finalization circuit
        D.5) Generate the syndrome measurement circuit and classical bits
        D.6) Append the full circuit to the interpretation step

    E.) Update Block history and evolution

    F.) Generate and append Block syndromes and detectors


    Parameters
    ----------
    interpretation_step: InterpretationStep
        The interpretation step.
    block: RotatedSurfaceCode
        The block to which the operation will be applied.
    idle_side_directions: DiagonalDirection
        The direction towards the idle side of the block was moved during the
        first swap-then-QEC operation.
    same_timeslice: bool
        Whether to append the generated circuit to the same timeslice as the last
        circuit in the interpretation step.
    debug_mode: bool
        Whether to enable debug mode.
    """

    final_moving_directions = idle_side_directions.opposite()

    # A.) Stabilizer evolution and updates
    new_stabilizers = move_stabilizers_and_append_evolution_and_updates(
        interpretation_step,
        block.stabilizers,
        final_moving_directions,
    )

    # B.) Logical operators evolution and updates
    new_logical_x, new_logical_z = move_logicals_and_append_evolution_and_updates(
        interpretation_step, block, final_moving_directions
    )

    # C.) Generate final Block and appropriate SyndromeCircuits
    final_block = generate_final_block_with_syndrome_circuits(
        block.unique_label, new_stabilizers, new_logical_x, new_logical_z, debug_mode
    )

    # D.) Circuit Generation
    # D.1) Find the qubits to initialize (data and ancilla) and teleportation pairs
    (
        anc_qubits_to_init_final,
        data_qubits_to_init_final,
        teleportation_qubit_pairs_final,
    ) = find_swap_then_qec_qubit_initializations(
        block.stabilizers, final_moving_directions
    )
    # D.2) Generate the initialization circuit
    second_swap_then_qec_reset_circuit = Circuit(
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

    # D.3) Generate the CNOTs circuit for syndrome measurement
    second_swap_then_qec_cnots_circuit = (
        get_final_block_syndrome_measurement_cnots_circuit(
            final_block,
            interpretation_step,
            True,
            idle_side_directions,
            anc_qubits_to_init_final,
        )
    )

    # D.4) Generate the teleportation finalization circuit
    second_swap_then_qec_teleportation_finalization_circuit = (
        generate_teleportation_finalization_circuit_with_updates(
            interpretation_step,
            final_block,
            anc_qubits_to_init_final,
            teleportation_qubit_pairs_final,
        )
    )
    # D.5) Generate the syndrome measurement circuit and classical bits
    second_swap_then_qec_measurement_circuit, second_swap_then_qec_cbits = (
        generate_syndrome_measurement_circuit_and_cbits(
            interpretation_step,
            final_block,
        )
    )

    # D.6) Append the full circuit to the interpretation step
    interpretation_step.append_circuit_MUT(
        Circuit(
            name="Second swap-then-QEC for Y-wall-out recombination",
            circuit=Circuit.construct_padded_circuit_time_sequence(
                (
                    (second_swap_then_qec_reset_circuit,),
                    (second_swap_then_qec_cnots_circuit,),
                    (
                        second_swap_then_qec_measurement_circuit,
                        second_swap_then_qec_teleportation_finalization_circuit,
                    ),
                )
            ),
        ),
        same_timeslice=same_timeslice,
    )

    # E.) Update Block history and evolution
    interpretation_step.update_block_history_and_evolution_MUT((final_block,), (block,))

    # F.) Generate and append Block syndromes and detectors
    generate_and_append_block_syndromes_and_detectors(
        interpretation_step, final_block, second_swap_then_qec_cbits
    )

    return interpretation_step


def generate_final_block_with_syndrome_circuits(
    block_unique_label: str,
    new_stabilizers: list[Stabilizer],
    new_logical_x: PauliOperator,
    new_logical_z: PauliOperator,
    debug_mode: bool,
) -> RotatedSurfaceCode:
    """
    Generate the final block after relocating the recombined block to its original
    position, with the appropriate syndrome extraction circuits.

    Parameters
    ----------
    block_unique_label: str
        The unique label of the block.
    new_stabilizers: list[Stabilizer]
        The stabilizers of the new block.
    new_logical_x: PauliOperator
        The logical X operator of the new block.
    new_logical_z: PauliOperator
        The logical Z operator of the new block.
    debug_mode: bool
        Whether to enable debug mode.

    Returns
    -------
    RotatedSurfaceCode
        The new block with the appropriate syndrome extraction circuits.
    """

    # C.1) Create a mock block to find the new syndrome circuits
    mock_block = RotatedSurfaceCode(
        unique_label=block_unique_label,
        stabilizers=new_stabilizers,
        logical_x_operators=(new_logical_x,),
        logical_z_operators=(new_logical_z,),
    )
    config, pivot_corners = mock_block.config_and_pivot_corners
    if config != 3:
        raise ValueError(
            f"New block must have type-3 corner configuration, found type-{config}."
        )

    # C.2) Find the starting qubit direction
    # It should be the opposite direction of the short_end_corner
    short_end_corner = pivot_corners[3]
    top_left_corner = mock_block.upper_left_qubit
    starting_diag_direction = find_relative_diagonal_direction(
        top_left_corner, short_end_corner
    )

    # C.3) Find syndrome circuits using the mock block
    new_synd_circ_tuple, new_stabilizer_to_circuit = (
        generate_syndrome_extraction_circuits(mock_block, starting_diag_direction)
    )

    # C.4) Create the new block with the new syndrome circuits
    new_block = RotatedSurfaceCode(
        stabilizers=mock_block.stabilizers,
        logical_x_operators=mock_block.logical_x_operators,
        logical_z_operators=mock_block.logical_z_operators,
        syndrome_circuits=new_synd_circ_tuple,
        stabilizer_to_circuit=new_stabilizer_to_circuit,
        unique_label=block_unique_label,
        skip_validation=not debug_mode,
    )

    return new_block


def move_stabilizers_and_append_evolution_and_updates(
    interpretation_step: InterpretationStep,
    stabilizers: list[Stabilizer],
    move_direction: DiagonalDirection,
) -> list[Stabilizer]:
    """
    Move stabilizers in the given direction and append their evolution and updates
    to the interpretation step.

    Parameters
    ----------
    interpretation_step: InterpretationStep
        The interpretation step to update.
    stabilizers: list[Stabilizer]
        The stabilizers to move.
    move_direction: DiagonalDirection
        The direction to move the stabilizers.

    Returns
    -------
    list[Stabilizer]
        The new stabilizers after the move.
    """
    # A.1) Move stabilizers and define evolution in the interpretation step
    new_stabilizers = []
    stab_evolution = {}
    for stab in stabilizers:
        new_stab = Stabilizer(
            stab.pauli,
            [qubit_mapper(dq, move_direction) for dq in stab.data_qubits],
            ancilla_qubits=[
                qubit_mapper(aq, move_direction) for aq in stab.ancilla_qubits
            ],
        )
        new_stabilizers.append(new_stab)
        stab_evolution[new_stab.uuid] = (stab.uuid,)

    # A.2) Update stabilizer updates in interpretation step
    for new_stab_uuid, old_stabs_uuid in stab_evolution.items():
        # Only one parent per new stabilizer
        old_stab_uuid = old_stabs_uuid[0]
        if old_stab_uuid in interpretation_step.stabilizer_updates:
            interpretation_step.stabilizer_updates[new_stab_uuid] = (
                interpretation_step.stabilizer_updates.pop(old_stab_uuid)
            )

    interpretation_step.stabilizer_evolution.update(stab_evolution)
    return new_stabilizers
