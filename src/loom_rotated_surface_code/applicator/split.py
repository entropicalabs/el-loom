"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from loom.eka import Circuit, ChannelType, Stabilizer, PauliOperator
from loom.eka.operations import Split
from loom.eka.utilities import Direction, Orientation
from loom.interpreter import InterpretationStep
from loom.interpreter.utilities import Cbit

from loom_rotated_surface_code.code_factory import RotatedSurfaceCode


# pylint: disable=duplicate-code
def split_consistency_check(
    interpretation_step: InterpretationStep, operation: Split
) -> RotatedSurfaceCode:
    """Check that the split operation can be performed on the given block.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        Interpretation step containing the block to split.
    operation : Split
        Split operation to perform.

    Returns
    -------
    RotatedSurfaceCode
        The block to split.

    Raises
    ------
    ValueError
        If the split position is larger than the width or height of the block.
    """
    block = interpretation_step.get_block(operation.input_block_name)

    if not isinstance(block, RotatedSurfaceCode):
        raise TypeError(
            f"The split operation is not supported for {type(block)} blocks."
        )
    position = operation.split_position
    if len(block.logical_x_operators) >= 2 or len(block.logical_z_operators) >= 2:
        raise NotImplementedError(
            "Splitting blocks with more than one logical operator is not supported."
        )
    if operation.orientation == Orientation.HORIZONTAL:
        if position >= block.size[1]:
            raise ValueError(
                f"Split position {position} is larger than the width of the block "
                f"({block.size[1]})."
            )
        if position in (0, block.size[1] - 1):
            raise ValueError("Split position cannot be at the edge of the block.")
    else:
        if position >= block.size[0]:
            raise ValueError(
                f"Split position {position} is larger than the height of the block "
                f"({block.size[0]})."
            )
        if position in (0, block.size[0] - 1):
            raise ValueError("Split position cannot be at the edge of the block.")

    return block


def create_split_circuit(
    interpretation_step: InterpretationStep,
    block: RotatedSurfaceCode,
    operation: Split,
    qubits_to_measure: tuple[tuple[int, ...], ...],
    boundary_type: str,
) -> tuple[Circuit, list[Cbit]]:
    """
    Create the circuit for the split operation.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        Interpretation step containing the block to split.
    block : RotatedSurfaceCode
        Block to split.
    operation : Split
        Split operation to perform.
    qubits_to_measure : tuple[tuple[int, ...], ...]
        Qubits to be measured in the split operation.
    boundary_type : str
        Type of the boundary of the block.

    Returns
    -------
    Circuit
        Circuit for the split operation.
    """
    # B) CIRCUIT
    #    B.1) Create classical channels for all data qubit measurements
    cbits = [interpretation_step.get_new_cbit_MUT(f"c_{q}") for q in qubits_to_measure]
    cbit_channels = [
        interpretation_step.get_channel_MUT(
            f"{cbit[0]}_{cbit[1]}", channel_type=ChannelType.CLASSICAL
        )
        for cbit in cbits
    ]
    #    B.2) Create a measurement circuit for every measured data qubit
    measure_circuit_seq = [
        [
            Circuit(
                "Measurement",
                channels=[interpretation_step.get_channel_MUT(q), cbit_channels[i]],
            )
            for i, q in enumerate(qubits_to_measure)
        ]
    ]

    #  If needed, apply a basis change
    if boundary_type == "X":
        # If the boundary type is X, the data qubit can directly be read out in the
        # Z basis
        split_circuit_list = measure_circuit_seq
    else:
        # If the boundary type is Z, the 2-bodies stabilizers are X, the data qubits
        # have to be read out in the X basis. Apply Hadamard gates before the
        # Z measurement to effectively measure in the X basis
        basis_change_circuit_seq = [
            [
                Circuit("H", channels=[interpretation_step.get_channel_MUT(q)])
                for q in qubits_to_measure
            ]
        ]
        # We add two sequences of gates to the circuit, i.e. 2 timesteps
        split_circuit_list = basis_change_circuit_seq + measure_circuit_seq

    row_column_str = (
        "row" if operation.orientation == Orientation.HORIZONTAL else "column"
    )
    split_circuit = Circuit(
        name=f"Split {block.unique_label} at {row_column_str} "
        f"{operation.split_position}",
        circuit=split_circuit_list,
    )
    return split_circuit, cbits


def split_stabilizers(
    block: RotatedSurfaceCode,
    operation: Split,
    qubits_to_measure: tuple[tuple[int, ...], ...],
) -> tuple[list[Stabilizer], list[Stabilizer], list[Stabilizer], list[Stabilizer]]:
    """
    Split the initial block's stabilizers into two lists of stabilizers, once for
    each block, and the list of old stabilizers that are gonna be replaced and the new
    stabilizers that will replace them.

    Parameters
    ----------
    block : RotatedSurfaceCode
        Initial block to be split
    operation : Split
        Split operation description
    qubits_to_measure : tuple[tuple[int, ...], ...]
        Qubits to be measured in the split operation.

    Returns
    -------
    tuple[list[Stabilizer], list[Stabilizer], list[Stabilizer], list[Stabilizer]]
        A tuple of four lists of stabilizers, one for each block, the old
        stabilizers that are gonna be replaced and the new stabilizers that will replace
        them (in the same order).
    """

    # C) - STABILIZERS
    #    C.1) Find stabilizers which are completely removed
    #    C.2) Find stabilizers which have to be reduced in weight
    #    C.3) Create new stabilizers with reduced weight
    #    C.4) Create two sets of stabilizers, one for each new block
    split_boundary = (
        Direction.LEFT
        if operation.orientation == Orientation.VERTICAL
        else Direction.TOP
    )
    boundary_type = block.boundary_type(split_boundary)
    split_is_vertical = operation.orientation == Orientation.VERTICAL
    split_position = operation.split_position

    #    C.1) Find stabilizers which are completely removed
    stabs_to_remove = [
        stab
        for stab in block.stabilizers
        if any(q in qubits_to_measure for q in stab.data_qubits)
    ]
    #    C.2) Find stabilizers which have to be reduced in weight
    old_stabs_to_reduce_weight = [
        stab
        for stab in stabs_to_remove
        if stab.pauli[0] != boundary_type
        and len([q for q in stab.data_qubits if q in qubits_to_measure]) == 2
        and len(stab.data_qubits) == 4
    ]
    #    C.3) Create new stabilizers with reduced weight
    new_stabs_reduced_weight = [
        Stabilizer(
            pauli="".join(
                stab.pauli[i]
                for i, q in enumerate(stab.data_qubits)
                if q not in qubits_to_measure
            ),
            data_qubits=[q for q in stab.data_qubits if q not in qubits_to_measure],
            ancilla_qubits=stab.ancilla_qubits,
        )
        for stab in old_stabs_to_reduce_weight
    ]

    new_stabilizers = [
        stab for stab in block.stabilizers if stab not in stabs_to_remove
    ] + new_stabs_reduced_weight

    data_qubits_block_1 = set(
        q
        for q in block.data_qubits
        if q[not split_is_vertical]
        < split_position + block.upper_left_qubit[not split_is_vertical]
    )
    data_qubits_block_2 = set(
        q
        for q in block.data_qubits
        if q[not split_is_vertical]
        > split_position + block.upper_left_qubit[not split_is_vertical]
    )
    #    C.4) Create two sets of stabilizers, one for each new block
    new_stabs_block_1 = [
        stab
        for stab in new_stabilizers
        if set(stab.data_qubits).issubset(data_qubits_block_1)
    ]
    new_stabs_block_2 = [
        stab
        for stab in new_stabilizers
        if set(stab.data_qubits).issubset(data_qubits_block_2)
    ]

    remaining_stabs = (
        set(new_stabilizers) - set(new_stabs_block_1) - set(new_stabs_block_2)
    )
    # Consistency check that all new stabilizers are assigned to a block
    if len(remaining_stabs) != 0:
        raise ValueError(
            f"Stabilizers {remaining_stabs} are not assigned to any block."
        )

    return (
        new_stabs_block_1,
        new_stabs_block_2,
        old_stabs_to_reduce_weight,
        new_stabs_reduced_weight,
    )


def find_split_stabilizer_to_circuit_mappings(
    block: RotatedSurfaceCode,
    new_block_stabilizers: list[Stabilizer],
    new_boundary_direction: Direction,
) -> dict[str, tuple[str, ...]]:
    """
    Finds the mapping between the stabilizers in the new block and the associated
    syndrome circuits.

    Parameters
    ----------
    block : RotatedSurfaceCode
        Initial block to be split
    new_block_stabilizers : list[Stabilizer]
        List of stabilizers in the one of the new blocks
    new_boundary_direction : Direction
        Direction of the the new boundary in the new block

    Returns
    -------
    dict[str, tuple[str, ...]]
        New mapping of stabilizers to syndrome circuits for the block.
    """
    stabilizer_to_circuit = block.stabilizer_to_circuit
    new_stabs_id = [stab.uuid for stab in new_block_stabilizers]
    # Copy the old mapping for stabilizers that are conserved
    new_stab_to_circ = {
        stab_id: circ_id
        for (stab_id, circ_id) in stabilizer_to_circuit.items()
        if stab_id in new_stabs_id
    }
    # Find the syndrome circuits that should be associated with the new boundary
    # stabilizers
    new_boundary_stabs = [
        stab
        for stab in new_block_stabilizers
        if stab.uuid not in new_stab_to_circ.keys()
    ]
    pauli_str = new_boundary_stabs[0].pauli
    new_boundary_circuit = next(
        synd_circ
        for synd_circ in block.syndrome_circuits
        if synd_circ.pauli == pauli_str
        if synd_circ.name
        == f"{new_boundary_direction.value.lower()}-{pauli_str.lower()}"
    )
    new_stab_to_circ.update(
        {stab.uuid: new_boundary_circuit.uuid for stab in new_boundary_stabs}
    )

    return new_stab_to_circ


def split_logical_operators(
    # pylint: disable=too-many-locals
    interpretation_step: InterpretationStep,
    block: RotatedSurfaceCode,
    qubits_to_measure: tuple[tuple[int, ...], ...],
    operation: Split,
    upleft_qubit_block_1: tuple[int, ...],
    upleft_qubit_block_2: tuple[int, ...],
) -> tuple[
    InterpretationStep,
    tuple[PauliOperator, PauliOperator],
    tuple[PauliOperator, PauliOperator],
    dict[str, tuple[Cbit, ...]],
    dict[str, tuple[Cbit, ...]],
]:
    """
    Finds the logical operators for the new blocks after the split operation. If an
    operator is partially measured, it will be split into two operators. If it is fully
    measured, it is shifted to the new block, adding stabilizers to the operator
    evolution.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        Interpretation step containing the block to split.
    block : RotatedSurfaceCode
        Block to split.
    qubits_to_measure : tuple[tuple[int, ...], ...]
        Qubits involved in the split operation.
    operation : Split
        Split operation description.
    upleft_qubit_block_1 : tuple[int, ...]
        Upper left qubit of the first new block.
    upleft_qubit_block_2 : tuple[int, ...]
        Upper left qubit of the second new block.

    Returns
    -------
    tuple[
        InterpretationStep,
        tuple[PauliOperator, PauliOperator],
        tuple[PauliOperator, PauliOperator]
        dict[str, tuple[Cbit, ...]],
        dict[str, tuple[Cbit, ...]],
    ]
        The interpretation step updated with the logical operator evolutions, logical X
        and logical Z operators for the first block, logical X and logical Z
        operators for the second block, updates for the logical X and Z operators.

    Raises
    ------
    NotImplementedError
        If the initial block encodes more than one logical qubit.
    """
    # D) LOGICAL OPERATORS

    # Find the stabilizer required to update the logical operator
    split_is_vertical = operation.orientation == Orientation.VERTICAL
    split_position = operation.split_position

    #    D.1) Find logical operators which are not partially measured
    # The logical operator is "preserved" if it is not partially measured
    # If it was located at the same position as the split, we move it to the new blocks
    preserved_log_x_ops = [
        op
        for op in block.logical_x_operators
        if not any(q in op.data_qubits for q in qubits_to_measure)
        or all(q in op.data_qubits for q in qubits_to_measure)
    ]
    #    D.2) Find logical operators which have to be split
    # The logical operator is split if it is partially measured
    split_log_x_ops = [
        op
        for op in block.logical_x_operators
        if any(q in op.data_qubits for q in qubits_to_measure)
        and not all(q in op.data_qubits for q in qubits_to_measure)
    ]
    if len(preserved_log_x_ops) != 0 and len(split_log_x_ops) != 0:
        raise NotImplementedError(
            "Splitting blocks with more than one logical X operator is not supported."
        )
    #    D.3) Create new logical operators
    # If the operator is split, construct two operators from it
    if len(split_log_x_ops) != 0:
        qubits_in_log_x_1 = [
            q
            for q in split_log_x_ops[0].data_qubits
            if q[not split_is_vertical]
            < split_position + block.upper_left_qubit[not split_is_vertical]
        ]
        new_log_x_op_1 = PauliOperator(
            "".join(split_log_x_ops[0].pauli[i] for i in range(len(qubits_in_log_x_1))),
            qubits_in_log_x_1,
        )

        qubits_in_log_x_2 = [
            q
            for q in split_log_x_ops[0].data_qubits
            if q[not split_is_vertical]
            > split_position + block.upper_left_qubit[not split_is_vertical]
        ]
        new_log_x_op_2 = PauliOperator(
            "".join(split_log_x_ops[0].pauli[i] for i in range(len(qubits_in_log_x_2))),
            qubits_in_log_x_2,
        )
        stabs_required_x_1 = []
        stabs_required_x_2 = []
        new_x_op_updates = {}
    # The operator is not partially measured, it may need to be displaced
    else:
        new_log_x_op_1, stabs_required_x_1 = (
            block.get_shifted_equivalent_logical_operator(
                initial_operator=preserved_log_x_ops[0],
                new_upleft_qubit=upleft_qubit_block_1,
            )
        )
        new_log_x_op_2, stabs_required_x_2 = (
            block.get_shifted_equivalent_logical_operator(
                initial_operator=preserved_log_x_ops[0],
                new_upleft_qubit=upleft_qubit_block_2,
            )
        )
        new_x_op_updates = {
            new_log_x_op_1.uuid: interpretation_step.retrieve_cbits_from_stabilizers(
                stabs_required_x_1, block
            ),
            new_log_x_op_2.uuid: interpretation_step.retrieve_cbits_from_stabilizers(
                stabs_required_x_2, block
            ),
        }

    # The logical operator is "preserved" if it is not partially measured
    # If it was located at the same position as the split, we move it to the new blocks
    preserved_log_z_ops = [
        op
        for op in block.logical_z_operators
        if not any(q in op.data_qubits for q in qubits_to_measure)
        or all(q in op.data_qubits for q in qubits_to_measure)
    ]
    # The logical operator is split if it is partially measured
    split_log_z_ops = [
        op
        for op in block.logical_z_operators
        if any(q in op.data_qubits for q in qubits_to_measure)
        and not all(q in op.data_qubits for q in qubits_to_measure)
    ]
    if len(preserved_log_z_ops) != 0 and len(split_log_z_ops) != 0:
        raise NotImplementedError(
            "Splitting blocks with more than one logical Z operator is not supported."
        )
    if len(split_log_z_ops) != 0:
        qubits_in_log_z_1 = [
            q
            for q in split_log_z_ops[0].data_qubits
            if q[not split_is_vertical]
            < split_position + block.upper_left_qubit[not split_is_vertical]
        ]
        new_log_z_op_1 = PauliOperator(
            "".join(split_log_z_ops[0].pauli[i] for i in range(len(qubits_in_log_z_1))),
            qubits_in_log_z_1,
        )
        qubits_in_log_z_2 = [
            q
            for q in split_log_z_ops[0].data_qubits
            if q[not split_is_vertical]
            > split_position + block.upper_left_qubit[not split_is_vertical]
        ]
        new_log_z_op_2 = PauliOperator(
            "".join(split_log_z_ops[0].pauli[i] for i in range(len(qubits_in_log_z_2))),
            qubits_in_log_z_2,
        )
        stabs_required_z_1 = []
        stabs_required_z_2 = []
        new_z_op_updates = {}
    else:
        new_log_z_op_1, stabs_required_z_1 = (
            block.get_shifted_equivalent_logical_operator(
                initial_operator=preserved_log_z_ops[0],
                new_upleft_qubit=upleft_qubit_block_1,
            )
        )
        new_log_z_op_2, stabs_required_z_2 = (
            block.get_shifted_equivalent_logical_operator(
                initial_operator=preserved_log_z_ops[0],
                new_upleft_qubit=upleft_qubit_block_2,
            )
        )
        new_z_op_updates = {
            new_log_z_op_1.uuid: interpretation_step.retrieve_cbits_from_stabilizers(
                stabs_required_z_1, block
            ),
            new_log_z_op_2.uuid: interpretation_step.retrieve_cbits_from_stabilizers(
                stabs_required_z_2, block
            ),
        }

    #    D.4) Update `logical_x/z_evolution`
    # Hardcoded index for the logical operators !!!
    interpretation_step.logical_x_evolution[new_log_x_op_1.uuid] = tuple(
        [block.logical_x_operators[0].uuid] + [stab.uuid for stab in stabs_required_x_1]
    )
    interpretation_step.logical_x_evolution[new_log_x_op_2.uuid] = tuple(
        [block.logical_x_operators[0].uuid] + [stab.uuid for stab in stabs_required_x_2]
    )
    interpretation_step.logical_z_evolution[new_log_z_op_1.uuid] = tuple(
        [block.logical_z_operators[0].uuid] + [stab.uuid for stab in stabs_required_z_1]
    )
    interpretation_step.logical_z_evolution[new_log_z_op_2.uuid] = tuple(
        [block.logical_z_operators[0].uuid] + [stab.uuid for stab in stabs_required_z_2]
    )

    return (
        interpretation_step,
        (new_log_x_op_1, new_log_z_op_1),
        (new_log_x_op_2, new_log_z_op_2),
        new_x_op_updates,
        new_z_op_updates,
    )


def split(  # pylint: disable=line-too-long, too-many-locals
    interpretation_step: InterpretationStep,
    operation: Split,
    same_timeslice: bool,
    debug_mode: bool,
) -> InterpretationStep:
    """
    Applicator to split a block using the Split description.

    The algorithm is the following:

    - A.) DATA QUBITS
    
        - Find data qubits to be measured, the initial block will be split into two

    - B.) CIRCUIT
    
        - B.1) Create classical channels for all data qubit measurements
        - B.2) Create a measurement circuit for every measured data qubit
        - B.3) Append the measurement circuits to the InterpretationStep circuit. \
        If needed, apply a basis change

    - C.) STABILIZERS
    
        - C.1) Find stabilizers which are completely removed
        - C.2) Find stabilizers which have to be reduced in weight
        - C.3) Create new stabilizers with reduced weight
        - C.4) Create two sets of stabilizers, one for each new block
        - C.5) Update ``stabilizer_evolution`` and ``stabilizer_updates`` for the \
        stabilizers which have been reduced in weight
        - C.6) Create the new ``stabilizer_to_circuit`` mapping

    - D.) LOGICAL OPERATORS
    
        - D.1) Find logical operators which are not partially measured
        - D.2) Find logical operators which have to be split
        - D.3) Create new logical operators and final to initial mapping
        - D.4) Update ``logical_x/z_evolution``
        - D.5) Update ``logical_x/z_updates``

    - E.) NEW BLOCK AND NEW INTERPRETATION STEP
    
        - E.1) Create the new blocks
        - E.2) Update the block history

    Parameters
    ----------
    interpretation_step : InterpretationStep
        Interpretation step containing the block to split.
    operation : Split
        Split operation description.
    same_timeslice : bool
        Flag indicating whether the operation is part of the same timestep as the
        previous operation.
    debug_mode : bool
        Flag indicating whether the interpretation should be done in debug mode.
        Activating debug mode will enable commutation validation for Block

    Returns
    -------
    InterpretationStep
        Interpretation step after the split operation.
    """

    # Check that the operation can be performed
    block = split_consistency_check(interpretation_step, operation)
    split_is_vertical = operation.orientation == Orientation.VERTICAL
    split_position = operation.split_position

    # A) - DATA QUBITS
    #    Find data qubits to be measured, the initial block will be split into two
    split_boundary = (
        Direction.LEFT
        if operation.orientation == Orientation.VERTICAL
        else Direction.TOP
    )
    boundary_qubits = block.boundary_qubits(split_boundary)
    boundary_type = block.boundary_type(split_boundary)
    shift_vector = (
        split_is_vertical * split_position,
        (not split_is_vertical) * split_position,
        0,
    )
    qubits_to_measure = tuple(
        tuple(
            coord1 + coord2 for coord1, coord2 in zip(qubit, shift_vector, strict=True)
        )
        for qubit in boundary_qubits
    )

    # block 1 is either the left or top block
    qubits_block_1 = [
        q
        for q in block.data_qubits
        if q[not split_is_vertical]
        < split_position + block.upper_left_qubit[not split_is_vertical]
    ]
    upleft_qubit_block_1 = min(qubits_block_1, key=lambda x: x[0] + x[1])
    # block 2 is either the right or bottom block
    qubits_block_2 = [
        q
        for q in block.data_qubits
        if q[not split_is_vertical]
        > split_position + block.upper_left_qubit[not split_is_vertical]
    ]
    upleft_qubit_block_2 = min(qubits_block_2, key=lambda x: x[0] + x[1])

    # B) - CIRCUIT
    #    B.1) Create classical channels for all data qubit measurements
    #    B.2) Create a measurement circuit for every measured data qubit
    split_circuit, cbits = create_split_circuit(
        interpretation_step, block, operation, qubits_to_measure, boundary_type
    )
    #    B.3) Append the measurement circuits to the InterpretationStep circuit
    interpretation_step.append_circuit_MUT(split_circuit, same_timeslice)

    # C) - STABILIZERS
    #    C.1) Find stabilizers which are completely removed
    #    C.2) Find stabilizers which have to be reduced in weight
    #    C.3) Create new stabilizers with reduced weight
    #    C.4) Create two sets of stabilizers, one for each new block
    (
        new_stabs_block_1,
        new_stabs_block_2,
        old_stabs_to_reduce_weight,
        new_stabs_reduced_weight,
    ) = split_stabilizers(block, operation, qubits_to_measure)

    #    C.5) Update `stabilizer_evolution` and `stabilizer_updates` for the
    #         stabilizers which have been reduced in weight
    for i, stab in enumerate(new_stabs_reduced_weight):
        previous_updates = (
            interpretation_step.stabilizer_updates[stab.uuid]
            if stab.uuid in interpretation_step.stabilizer_updates.keys()
            else ()
        )
        old_stab = old_stabs_to_reduce_weight[i]
        # Find the data qubits of this stabilizer which are measured and whose
        # corresponding cbit has to be included in the stabilizer update
        qubits_measured = [q for q in old_stab.data_qubits if q in qubits_to_measure]
        # Find the cbits and include them in `stabilizer_updates`
        cbit_indices = [qubits_to_measure.index(q) for q in qubits_measured]
        new_stab_updates = tuple(cbits[cbit_idx] for cbit_idx in cbit_indices)
        if updates := previous_updates + new_stab_updates:
            interpretation_step.stabilizer_updates[stab.uuid] = updates

    stab_map_weight4_to_weight2 = {
        new_stabs_reduced_weight[i].uuid: (stab.uuid,)
        for i, stab in enumerate(old_stabs_to_reduce_weight)
    }
    interpretation_step.stabilizer_evolution.update(stab_map_weight4_to_weight2)

    #   C.6) Create the new `stabilizer_to_circuit` mapping
    # block 1 is the left block -> new boundary is right if split is vertical
    # block 1 is the top block -> new boundary is bottom if split is horizontal
    stab_to_circ_block_1 = find_split_stabilizer_to_circuit_mappings(
        block,
        new_stabs_block_1,
        Direction.RIGHT if split_is_vertical else Direction.BOTTOM,
    )
    # block 2 is the right block -> new boundary is left if split is vertical
    # block 2 is the bottom block -> new boundary is top if split is horizontal
    stab_to_circ_block_2 = find_split_stabilizer_to_circuit_mappings(
        block,
        new_stabs_block_2,
        Direction.LEFT if split_is_vertical else Direction.TOP,
    )

    # D) LOGICAL OPERATORS
    #    D.1) Find logical operators which are not partially measured
    #    D.2) Find logical operators which have to be split
    #    D.3) Create new logical operators and final to initial mapping
    #    D.4) Update `logical_x/z_evolution`
    (
        interpretation_step,
        (new_log_x_op_1, new_log_z_op_1),
        (new_log_x_op_2, new_log_z_op_2),
        new_x_updates,
        new_z_updates,
    ) = split_logical_operators(
        interpretation_step,
        block,
        qubits_to_measure,
        operation,
        upleft_qubit_block_1,
        upleft_qubit_block_2,
    )
    #    D.5) Create the new `logical_x/z_operator_updates`
    # Find measurements that split the relevant operator, if the operator is aligned
    # with the split, it's either fully measured or not measured at all.
    # It's not contributing to the operator updates in both cases.
    x_and_split_aligned = block.x_boundary == operation.orientation
    x_op_measurements = (
        tuple(
            c
            for c in cbits
            if c[0].split("_")[1] in map(str, block.logical_x_operators[0].data_qubits)
        )
        if not x_and_split_aligned
        else ()
    )
    z_op_measurements = (
        tuple(
            c
            for c in cbits
            if c[0].split("_")[1] in map(str, block.logical_z_operators[0].data_qubits)
        )
        if x_and_split_aligned
        else ()
    )
    # The new operator is mapped to the measurement that split the initial logical operator
    # The first block always inherits the updates from the initial block if the operator is modified
    interpretation_step.update_logical_operator_updates_MUT(
        operator_type="X",
        logical_operator_id=new_log_x_op_1.uuid,
        new_updates=new_x_updates.get(new_log_x_op_1.uuid, ()) + x_op_measurements,
        inherit_updates=(new_log_x_op_1.uuid != block.logical_x_operators[0].uuid),
    )
    interpretation_step.update_logical_operator_updates_MUT(
        operator_type="Z",
        logical_operator_id=new_log_z_op_1.uuid,
        new_updates=new_z_updates.get(new_log_z_op_1.uuid, ()) + z_op_measurements,
        inherit_updates=(new_log_z_op_1.uuid != block.logical_z_operators[0].uuid),
    )
    interpretation_step.update_logical_operator_updates_MUT(
        operator_type="X",
        logical_operator_id=new_log_x_op_2.uuid,
        new_updates=new_x_updates.get(new_log_x_op_2.uuid, ()),
        inherit_updates=x_and_split_aligned
        and (new_log_x_op_2.uuid != block.logical_x_operators[0].uuid),
    )
    interpretation_step.update_logical_operator_updates_MUT(
        operator_type="Z",
        logical_operator_id=new_log_z_op_2.uuid,
        new_updates=new_z_updates.get(new_log_z_op_2.uuid, ()),
        inherit_updates=not x_and_split_aligned
        and (new_log_z_op_2.uuid != block.logical_z_operators[0].uuid),
    )

    # E) NEW BLOCK AND NEW INTERPRETATION STEP
    #    E.1) Create the new blocks
    new_block_1 = RotatedSurfaceCode(
        stabilizers=new_stabs_block_1,
        logical_x_operators=[new_log_x_op_1],
        logical_z_operators=[new_log_z_op_1],
        syndrome_circuits=block.syndrome_circuits,
        stabilizer_to_circuit=stab_to_circ_block_1,
        unique_label=operation.output_blocks_name[0],
        skip_validation=not debug_mode,
    )
    new_block_2 = RotatedSurfaceCode(
        stabilizers=new_stabs_block_2,
        logical_x_operators=[new_log_x_op_2],
        logical_z_operators=[new_log_z_op_2],
        syndrome_circuits=block.syndrome_circuits,
        stabilizer_to_circuit=stab_to_circ_block_2,
        unique_label=operation.output_blocks_name[1],
        skip_validation=not debug_mode,
    )
    #    E.2) Update the block history
    interpretation_step.update_block_history_and_evolution_MUT(
        new_blocks=(new_block_1, new_block_2), old_blocks=(block,)
    )

    return interpretation_step
