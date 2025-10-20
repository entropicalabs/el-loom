"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=duplicate-code
from loom.eka import Circuit, PauliOperator
from loom.eka.operations import Grow
from loom.eka.utilities import Direction, Orientation
from loom.interpreter import InterpretationStep
from loom.interpreter.applicator.generate_syndromes import generate_syndromes

from ..code_factory import RotatedSurfaceCode


def grow(  # pylint: disable=too-many-branches, too-many-statements, too-many-locals
    interpretation_step: InterpretationStep,
    operation: Grow,
    same_timeslice: bool,
    debug_mode: bool,
) -> InterpretationStep:
    """
    Grow a Block in the specified direction.

    The algorithm is the following:

    - A.) Create the new piece of surface code (the "extra" part of the grown block)

        - A.1) Find the corner of the additional piece
        - A.2) Find the Pauli flavour of the upper left 4-body for the new piece
        - A.3) Create the new 4-body stabilizers
        - A.4) Create the new 2-body stabilizers
        - A.5) Identify the stabilizers to remove and the ones to replace them with
        - A.6) Add the new stabilizers to the stabilizer_evolution

    - B.) Create the reset circuit

        - B.1) Reset all data qubits involved in the growth in the right basis
        - B.2) Create empty syndromes for the relevant stabilizers

    - C.) Associate the new stabilizers to their respective syndrome circuits

        - C.1) Recover the syndrome circuits by name
        - C.2) Update the stabilizer to circuit mapping

    - D.) Create the new logical operators and keep track of the evolution

    Parameters
    ----------
    interpretation_step : InterpretationStep
        Interpretation step containing the block to grow.
    operation : Grow
        Grow operation description.
    same_timeslice : bool
        Flag indicating whether the operation is part of the same timestep as the
        previous operation.
    debug_mode : bool
        Flag indicating whether the interpretation should be done in debug mode.
        Activating debug mode will enable commutation validation for Block

    Returns
    -------
    InterpretationStep
        Interpretation step after the grow operation.
    """

    # Get some useful information from the block
    block = interpretation_step.get_block(operation.input_block_name)
    if not isinstance(block, RotatedSurfaceCode):
        raise TypeError(
            f"The grow operation is not supported for {type(block)} blocks."
        )
    stabilizers = block.stabilizers
    direction = operation.direction
    length = operation.length
    old_x_logicals = block.logical_x_operators
    old_z_logicals = block.logical_z_operators

    is_horizontal = direction in (Direction.LEFT, Direction.RIGHT)
    dx = length + 1 if is_horizontal else block.size[0]
    dz = length + 1 if not is_horizontal else block.size[1]
    weight_4_x_schedule = block.weight_4_x_schedule
    weight_4_z_schedule = block.weight_4_z_schedule

    # A) - Create the new piece of surface code
    #   A.1) - Find the new corner of the block
    new_upper_left_position = list(block.upper_left_qubit[:2])
    match direction:
        case Direction.LEFT:
            new_upper_left_position[0] = new_upper_left_position[0] - length
        case Direction.RIGHT:
            new_upper_left_position[0] = new_upper_left_position[0] + (
                block.size[0] - 1
            )
        case Direction.TOP:
            new_upper_left_position[1] = new_upper_left_position[1] - length
        case Direction.BOTTOM:
            new_upper_left_position[1] = new_upper_left_position[1] + (
                block.size[1] - 1
            )

    #   A.2) - For the newly created stabilizers, find the Pauli flavour of the upper
    #   leftmost 4-body stabilizer.
    #   The pauli string of the upper left corner is conserved if the sum of the
    #   size of the block in the direction of growth and the length of growth is
    #   even. Otherwise, the Pauli string is flipped.
    if (
        length * (direction in (Direction.TOP, Direction.LEFT))
        + (block.size[0] - 1) * (direction == Direction.RIGHT)
        + (block.size[1] - 1) * (direction == Direction.BOTTOM)
    ) % 2 == 0:
        new_upper_left_4body_pauli = block.upper_left_4body_stabilizer.pauli
    else:
        new_upper_left_4body_pauli = (
            "ZZZZ" if block.upper_left_4body_stabilizer.pauli == "XXXX" else "XXXX"
        )

    #   A.3) - Create the new 4-body stabilizers
    new_upleft_4_body_stabs = RotatedSurfaceCode.generate_weight4_stabs(
        pauli=new_upper_left_4body_pauli,
        schedule=(
            weight_4_x_schedule
            if new_upper_left_4body_pauli == "XXXX"
            else weight_4_z_schedule
        ),
        start_in_top_left_corner=True,
        dx=dx,
        dz=dz,
        initial_position=new_upper_left_position,
    )
    new_rest_4_body_stabs = RotatedSurfaceCode.generate_weight4_stabs(
        pauli="XXXX" if new_upper_left_4body_pauli == "ZZZZ" else "ZZZZ",
        schedule=(
            weight_4_x_schedule
            if new_upper_left_4body_pauli != "XXXX"
            else weight_4_z_schedule
        ),
        start_in_top_left_corner=False,
        dx=dx,
        dz=dz,
        initial_position=new_upper_left_position,
    )
    new_4_body_stabs = new_upleft_4_body_stabs + new_rest_4_body_stabs

    #   A.4) - Create the new 2-body stabilizers
    new_top_left_is_xxxx = new_upper_left_4body_pauli == "XXXX"
    stab_left_right_is_x = block.x_boundary == Orientation.HORIZONTAL
    # Does the new left boundary start with a 2-body stabilizer on the first row?
    new_weight_2_stab_is_first_row = new_top_left_is_xxxx != stab_left_right_is_x
    stab_left_right = "XX" if stab_left_right_is_x else "ZZ"
    stab_top_bottom = "ZZ" if stab_left_right_is_x else "XX"
    # Left boundary
    if direction != Direction.RIGHT:
        if dz % 2 == 1:
            num_weight2_stabs = (dz - 1) / 2
        else:
            num_weight2_stabs = dz / 2 - (not new_weight_2_stab_is_first_row)
        stabs_left = RotatedSurfaceCode.generate_weight2_stabs(
            pauli=stab_left_right,
            initial_position=(
                new_upper_left_position[0],
                new_upper_left_position[1] + (not new_weight_2_stab_is_first_row),
            ),
            num_stabs=num_weight2_stabs,
            orientation=Orientation.VERTICAL,
            is_bottom_or_right=False,
        )
    else:
        stabs_left = []
    # Right boundary
    if direction != Direction.LEFT:
        # Does the new right boundary start with a 2-body stabilizer on the first row?
        new_right_first_row = new_weight_2_stab_is_first_row != (dx % 2)
        if dz % 2 == 1:
            num_weight2_stabs = (dz - 1) / 2
        else:
            num_weight2_stabs = dz / 2 - (not new_right_first_row)
        stabs_right = RotatedSurfaceCode.generate_weight2_stabs(
            pauli=stab_left_right,
            initial_position=(
                new_upper_left_position[0] + dx - 1,
                new_upper_left_position[1] + (not new_right_first_row),
            ),
            num_stabs=num_weight2_stabs,
            orientation=Orientation.VERTICAL,
            is_bottom_or_right=True,
        )
    else:
        stabs_right = []
    # Top boundary
    if direction != Direction.BOTTOM:
        # Does the new top boundary start with a 2-body stabilizer on the first column?
        num_weight2_stabs = dx // 2
        if dx % 2 == 1:
            num_weight2_stabs = (dx - 1) / 2
        else:
            num_weight2_stabs = dx / 2 - new_weight_2_stab_is_first_row
        stabs_top = RotatedSurfaceCode.generate_weight2_stabs(
            pauli=stab_top_bottom,
            initial_position=(
                new_upper_left_position[0] + new_weight_2_stab_is_first_row,
                new_upper_left_position[1],
            ),
            num_stabs=num_weight2_stabs,
            orientation=Orientation.HORIZONTAL,
            is_bottom_or_right=False,
        )
    else:
        stabs_top = []
    # Bottom boundary
    if direction != Direction.TOP:
        # Does the new bottom boundary start with a 2-body stabilizer on the first
        # column?
        bottom_first_col = new_weight_2_stab_is_first_row == (dz % 2)
        if dx % 2 == 1:
            num_weight2_stabs = (dx - 1) / 2
        else:
            num_weight2_stabs = dx / 2 - (not bottom_first_col)
        stabs_bottom = RotatedSurfaceCode.generate_weight2_stabs(
            pauli=stab_top_bottom,
            initial_position=(
                new_upper_left_position[0] + (not bottom_first_col),
                new_upper_left_position[1] + dz - 1,
            ),
            num_stabs=num_weight2_stabs,
            orientation=Orientation.HORIZONTAL,
            is_bottom_or_right=True,
        )
    else:
        stabs_bottom = []

    #   A.5) - Identify the stabilizers to remove and the ones to replace them with
    stabs_to_remove = [
        stab
        for stab in stabilizers
        if set(stab.data_qubits).issubset(set(block.boundary_qubits(direction)))
        and len(stab.data_qubits) == 2
    ]
    stabs_uuid_to_remove = [stab.uuid for stab in stabs_to_remove]
    # There is a 1-to-1 mapping between the stabilizers to remove and the new 4-body
    # stabilizers in stabs_uuid_replace. This mapping is used to update the
    # stabilizer_evolution dictionary
    stabs_uuid_replace = [
        stab.uuid
        for old_stab in stabs_to_remove
        for stab in new_4_body_stabs
        if set(old_stab.data_qubits).issubset(set(stab.data_qubits))
        and old_stab.pauli in stab.pauli
    ]
    final_to_initial_stab_map = {
        stabs_uuid_replace[i]: (uuid,) for i, uuid in enumerate(stabs_uuid_to_remove)
    }
    #   A.6) - Add the new stabilizers to the stabilizer_evolution
    interpretation_step.stabilizer_evolution.update(final_to_initial_stab_map)

    new_2_body_stabs = stabs_left + stabs_right + stabs_top + stabs_bottom
    new_stabilizers = (
        [stab for stab in stabilizers if stab not in stabs_to_remove]
        + new_2_body_stabs
        + new_4_body_stabs
    )

    # B) - Create the reset circuit
    #   B.1) - Reset all data qubits involved in the growth in the right basis
    if block.boundary_type(direction) == "X":
        # X boundary means Z stabilizers
        reset_state = "0"
    else:
        # Z boundary means X stabilizers
        reset_state = "+"
    new_data_qubits = tuple(
        set(
            q
            for stab in new_4_body_stabs
            for q in stab.data_qubits
            if q not in block.boundary_qubits(direction)
        )
    )
    existing_ancilla_qubits = tuple(
        q for stab in stabs_to_remove for q in stab.ancilla_qubits
    )
    reset_sequence = [
        [
            Circuit(
                name=f"reset_{reset_state}",
                channels=interpretation_step.get_channel_MUT(q, "quantum"),
            )
            for q in new_data_qubits
        ]
    ]
    grow_circuit = Circuit(
        name=f"grow {block.unique_label} by {length} to the {direction.value}",
        circuit=reset_sequence,
    )
    interpretation_step.append_circuit_MUT(grow_circuit, same_timeslice)

    #   B.2) - Create empty syndromes for the relevant stabilizers
    deterministic_stab_pauli = "X" if reset_state == "+" else "Z"
    # The new syndromes are created for the new 4-body stabilizers that are not morphed
    # from a 2-body. If they are, no need to account for corrections (it would be 0).
    relevant_stabs = [
        stab
        for stab in new_2_body_stabs + new_4_body_stabs
        if set(stab.pauli) == {deterministic_stab_pauli}
        and not all(a in existing_ancilla_qubits for a in stab.ancilla_qubits)
    ]
    new_syndromes = generate_syndromes(
        interpretation_step=interpretation_step,
        stabilizers=relevant_stabs,
        block=block,
        stab_measurements=[() for _ in relevant_stabs],
    )
    interpretation_step.append_syndromes_MUT(new_syndromes)

    # C) - Associate the new stabilizers to their respective syndrome circuits
    #   C.1) - Recover the syndrome circuits by name
    xxxx_syndrome_circuit = next(
        syndrome_circuit
        for syndrome_circuit in block.syndrome_circuits
        if syndrome_circuit.name == "xxxx"
    )
    zzzz_syndrome_circuit = next(
        syndrome_circuit
        for syndrome_circuit in block.syndrome_circuits
        if syndrome_circuit.name == "zzzz"
    )
    top_syndrome_circuit = next(
        syndrome_circuit
        for syndrome_circuit in block.syndrome_circuits
        if syndrome_circuit.name == f"top-{stab_top_bottom.lower()}"
    )
    bottom_syndrome_circuit = next(
        syndrome_circuit
        for syndrome_circuit in block.syndrome_circuits
        if syndrome_circuit.name == f"bottom-{stab_top_bottom.lower()}"
    )
    left_syndrome_circuit = next(
        syndrome_circuit
        for syndrome_circuit in block.syndrome_circuits
        if syndrome_circuit.name == f"left-{stab_left_right.lower()}"
    )
    right_syndrome_circuit = next(
        syndrome_circuit
        for syndrome_circuit in block.syndrome_circuits
        if syndrome_circuit.name == f"right-{stab_left_right.lower()}"
    )
    # The syndrome circuits can be reused, they are not modified
    new_syndrome_circuits = block.syndrome_circuits

    #   C.2) - Update the stabilizer to circuit mapping
    new_stab_to_circ = (
        {
            stab.uuid: (
                xxxx_syndrome_circuit.uuid
                if new_upleft_4_body_stabs[0].pauli == "XXXX"
                else zzzz_syndrome_circuit.uuid
            )
            for stab in new_upleft_4_body_stabs
        }
        | {
            stab.uuid: (
                zzzz_syndrome_circuit.uuid
                if new_rest_4_body_stabs[0].pauli == "ZZZZ"
                else xxxx_syndrome_circuit.uuid
            )
            for stab in new_rest_4_body_stabs
        }
        | {stab.uuid: top_syndrome_circuit.uuid for stab in stabs_top}
        | {stab.uuid: bottom_syndrome_circuit.uuid for stab in stabs_bottom}
        | {stab.uuid: left_syndrome_circuit.uuid for stab in stabs_left}
        | {stab.uuid: right_syndrome_circuit.uuid for stab in stabs_right}
    )
    new_stab_to_circ.update(
        {
            k: v
            for (k, v) in block.stabilizer_to_circuit.items()
            if k not in stabs_uuid_to_remove
        }
    )

    if block.boundary_type(direction) == "X":
        update_type = "Z"
        log_ops_untouched = old_x_logicals
        log_ops_to_modify = old_z_logicals
    else:
        update_type = "X"
        log_ops_untouched = old_z_logicals
        log_ops_to_modify = old_x_logicals

    # Detect if qubits involved are not part of the boundary anymore
    qubits_involved = list(log_ops_to_modify[0].data_qubits)
    qubit_to_extend = next(
        q for q in block.boundary_qubits(direction) if q in qubits_involved
    )
    match direction:
        case Direction.LEFT:
            new_qubits = qubits_involved + [
                (qubit_to_extend[0] - i, qubit_to_extend[1], qubit_to_extend[2])
                for i in range(1, length + 1)
            ]
        case Direction.RIGHT:
            new_qubits = qubits_involved + [
                (qubit_to_extend[0] + i, qubit_to_extend[1], qubit_to_extend[2])
                for i in range(1, length + 1)
            ]
        case Direction.TOP:
            new_qubits = qubits_involved + [
                (qubit_to_extend[0], qubit_to_extend[1] - i, qubit_to_extend[2])
                for i in range(1, length + 1)
            ]
        case Direction.BOTTOM:
            new_qubits = qubits_involved + [
                (qubit_to_extend[0], qubit_to_extend[1] + i, qubit_to_extend[2])
                for i in range(1, length + 1)
            ]
    new_logical = PauliOperator(
        pauli=update_type * len(new_qubits),
        data_qubits=new_qubits,
    )
    # D) - Create the new logical operators and keep track of the evolution
    if update_type == "X":
        new_log_x = (new_logical,)
        new_log_z = log_ops_untouched
        interpretation_step.logical_x_evolution.update(
            {new_logical.uuid: (log_ops_to_modify[0].uuid,)}
        )
        # Inherit the updates from the previous operator
        # (new_logical is always a new operator)
        interpretation_step.update_logical_operator_updates_MUT(
            "X", new_logical.uuid, (), True
        )
    else:
        new_log_x = log_ops_untouched
        new_log_z = (new_logical,)
        interpretation_step.logical_z_evolution.update(
            {new_logical.uuid: (log_ops_to_modify[0].uuid,)}
        )
        # Inherit the updates from the previous operator
        # (new_logical is always a new operator)
        interpretation_step.update_logical_operator_updates_MUT(
            "Z", new_logical.uuid, (), True
        )

    new_block = RotatedSurfaceCode(
        unique_label=block.unique_label,
        stabilizers=new_stabilizers,
        logical_x_operators=tuple(new_log_x),
        logical_z_operators=tuple(new_log_z),
        syndrome_circuits=new_syndrome_circuits,
        stabilizer_to_circuit=new_stab_to_circ,
        skip_validation=not debug_mode,
    )
    # Update the block history
    interpretation_step.update_block_history_and_evolution_MUT((new_block,), (block,))

    return interpretation_step
