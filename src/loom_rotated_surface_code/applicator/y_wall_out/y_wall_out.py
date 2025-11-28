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

from loom.eka import Circuit, PauliOperator
from loom.eka.utilities import Orientation
from loom.interpreter import InterpretationStep

from .y_wall_out_initial_syndrome_measurement import (
    y_wall_out_initial_syndrome_measurement,
)
from .y_wall_measurement_hadamard import (
    y_wall_out_measurement_and_hadamard,
    find_qubit_sets,
)
from .y_wall_out_recombination_swap_then_qec import (
    y_wall_out_recombination_swap_then_qec,
    get_idle_hadamard_info,
)
from .y_wall_out_final_swap_then_qec import y_wall_out_final_swap_then_qec
from .y_wall_out_final_syndrome_measurement import y_wall_out_final_qec_rounds
from ...code_factory import RotatedSurfaceCode


def y_wall_out(
    interpretation_step: InterpretationStep,
    block: RotatedSurfaceCode,
    wall_position: int,
    wall_orientation: Orientation,
    same_timeslice: bool,
    debug_mode: bool,
) -> InterpretationStep:
    """
    Implement the y_wall_out operation. The operation consists of three steps:

    - A) Consistency check

    - B) Extract geometric information

    - C) Call the sub-operations in sequence:

        - C.1) One round of syndrome measurement of the initial block, \
            with CNOT scheduling specifically designed for fault tolerance
        - C.2) Y wall measurement and Hadamard
        - C.3) Recombination of the Block to get rid of the wall via swap-then-QEC
        - C.4) Swap-then-QEC to move the recombined block back to its original position
        - C.5) d-2 rounds of syndrome measurement of the final block, \
            with CNOT scheduling specifically designed for fault tolerance
    
    - D) Wrap the y_wall_out circuit in a single circuit and append it


    Example: the block on the left is transformed into the block on the right::

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
    same_timeslice : bool
        Flag indicating whether the operation is part of the same timestep as the
        previous operation.
    debug_mode : bool
        Flag indicating whether the interpretation should be done in debug mode.
        Activating debug mode will enable commutation validation for Block.

    Returns
    -------
    InterpretationStep
        The interpretation step after applying the y_wall_out operation.
    """

    # Begin composite operation / y_wall_out
    init_circ_len = len(interpretation_step.intermediate_circuit_sequence)

    # A) Consistency check
    y_wall_out_consistency_check(block, wall_position, wall_orientation)

    # B) Extract geometric information
    is_wall_hor = wall_orientation == Orientation.HORIZONTAL
    qubits_to_measure, qubits_to_idle, qubits_to_hadamard = find_qubit_sets(
        block, wall_position, is_wall_hor
    )

    # C) Call the sub-operations in sequence
    # C.1) One round of syndrome measurement of the initial block
    interpretation_step = y_wall_out_initial_syndrome_measurement(
        interpretation_step,
        block,
        same_timeslice=False,
        debug_mode=debug_mode,
    )

    # C.2) Y wall measurement and Hadamard
    interpretation_step = y_wall_out_measurement_and_hadamard(
        interpretation_step,
        block,
        wall_position,
        wall_orientation,
        same_timeslice=False,
        debug_mode=debug_mode,
    )
    current_block: RotatedSurfaceCode = interpretation_step.get_block(
        block.unique_label
    )
    # Extract idle side directions for later use
    _, _, _, idle_side_directions, _ = get_idle_hadamard_info(
        current_block,
        qubits_to_hadamard,
        qubits_to_idle,
    )

    # C.3) Recombination of the Block to get rid of the wall via swap-then-QEC
    interpretation_step = y_wall_out_recombination_swap_then_qec(
        interpretation_step,
        current_block,
        block,
        qubits_to_idle,
        qubits_to_measure,
        qubits_to_hadamard,
        same_timeslice=False,
        debug_mode=debug_mode,
    )
    current_block = interpretation_step.get_block(block.unique_label)

    # C.4) Swap-then-QEC to move the recombined block back to its original position
    interpretation_step = y_wall_out_final_swap_then_qec(
        interpretation_step,
        current_block,
        idle_side_directions,
        same_timeslice=False,
        debug_mode=debug_mode,
    )
    current_block = interpretation_step.get_block(block.unique_label)

    # C.5) d-2 rounds of syndrome measurement of the final block
    interpretation_step = y_wall_out_final_qec_rounds(
        interpretation_step,
        current_block,
        same_timeslice=False,
        debug_mode=debug_mode,
    )

    # D) Wrap the y_wall_out circuit in a single circuit and append it
    # pylint: disable=duplicate-code
    new_len = len(interpretation_step.intermediate_circuit_sequence)
    len_y_wall_out = new_len - init_circ_len
    circuit_seq = interpretation_step.pop_intermediate_circuit_MUT(len_y_wall_out)
    wrapped_circuit_seq = ()
    for timeslice in circuit_seq:
        timespan = max(composite_circuit.duration for composite_circuit in timeslice)
        # Create a circuit with empty timeslices and align circuits
        template_circ = (
            tuple(composite_circuit for composite_circuit in timeslice),
        ) + ((),) * (timespan - 1)
        wrapped_circuit_seq += template_circ
    wrapped_circuit = Circuit(
        name=f"y_wall_out operation on block {block.unique_label}",
        circuit=wrapped_circuit_seq,
    )
    interpretation_step.append_circuit_MUT(wrapped_circuit, same_timeslice)

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
