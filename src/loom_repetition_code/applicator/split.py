"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=duplicate-code
from loom.eka import ChannelType, Circuit, PauliOperator, Stabilizer
from loom.eka.operations import Split
from loom.eka.utilities import Direction, Orientation
from loom.interpreter import InterpretationStep
from loom.interpreter.utilities import Cbit

from ..code_factory import RepetitionCode


# pylint: disable=too-many-locals
def split_consistency_check(
    interpretation_step: InterpretationStep, operation: Split
) -> RepetitionCode:
    """
    Check that the split operation can be performed on the given chain.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        InterpretationStep containing the block to split.
    operation : Split
        Split operation to perform.

    Returns
    -------
    RepetitionCode
        Block to split.

    Raises
    ------
    ValueError
        If the split position is larger than the length of the chain.
        If the split position is at the edge of the chain.
        If one of the chains has a length of one.
    """
    # Extract Block
    block = interpretation_step.get_block(operation.input_block_name)

    # Check valid Block type
    if not isinstance(block, RepetitionCode):
        raise ValueError(
            f"This split operation is not supported for {type(block)} blocks."
        )

    relative_split_position = operation.split_position
    left_boundary = block.boundary_qubits(Direction.LEFT)[0]
    right_boundary = block.boundary_qubits(Direction.RIGHT)[0]

    split_position = left_boundary + relative_split_position

    # Check that the split position is under correct parameters
    if split_position > right_boundary or split_position < left_boundary:
        raise ValueError(f"Split position {split_position} is outside the chain.")

    if split_position in [left_boundary, right_boundary]:
        raise ValueError("Split position cannot be at the edge of the chain.")

    if split_position in [left_boundary + 1, right_boundary - 1]:
        raise ValueError(
            "Split has to partition chain in units of at least two qubits."
        )

    # Check that the split operation can be performed
    if operation.orientation != Orientation.VERTICAL:
        raise ValueError("Only vertical splits are supported.")

    return block


def find_qubit_to_measure(
    block: RepetitionCode, split_position: int
) -> tuple[int, int]:
    """Find the qubit to measure in the split operation.

    Parameters
    ----------
    block : RepetitionCode
        Block to split.
    split_position : int
        Position where the chain will be split.

    Returns
    -------
    tuple[int, int]
        Qubit to be measured in the split operation.
    """
    boundary_qubit_position = block.boundary_qubits(Direction.LEFT)[0]
    qubit_to_measure = (boundary_qubit_position + split_position, 0)
    return qubit_to_measure


def create_split_circuit(
    interpretation_step: InterpretationStep,
    check_type: str,
    qubit_to_measure: tuple[int, int],
    circuit_name: str,
) -> tuple[Circuit, Cbit]:
    """Create the circuit for the split operation.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        Interpretation step containing the chain to split.
    check_type : str
        Type of stabilizer to measure.
    qubits_to_measure : tuple[int, int]
        Qubit to be measured in the split operation.
    circuit_name : str
        Name of the circuit for the split operation.

    Returns
    -------
    tuple[Circuit, Cbit]
        Circuit for the split operation and the classical bit used in the circuit.
    """

    # Create classical channels for all data qubit measurements
    cbit = interpretation_step.get_new_cbit_MUT(f"c_{qubit_to_measure}")
    cbit_channel = interpretation_step.get_channel_MUT(
        f"{cbit[0]}_{cbit[1]}", channel_type=ChannelType.CLASSICAL
    )

    # Create a measurement circuit for every measured data qubit
    measure_circuit_seq = [
        Circuit(
            "Measurement",
            channels=[
                interpretation_step.get_channel_MUT(qubit_to_measure),
                cbit_channel,
            ],
        )
    ]

    # If needed, apply a basis change
    if check_type == "X":
        # If phaseflip code qubits can already be measured
        split_circuit_list = measure_circuit_seq
    else:
        # If bitflip code qubits need to be Hadamard transformed before measurement
        basis_change_circuit_seq = [
            Circuit(
                "H", channels=[interpretation_step.get_channel_MUT(qubit_to_measure)]
            )
        ]
        # Add basis change before measurement
        split_circuit_list = basis_change_circuit_seq + measure_circuit_seq

    # Build circuit
    split_circuit = Circuit(
        name=circuit_name,
        circuit=split_circuit_list,
    )
    return split_circuit, cbit


def find_new_stabilizers(
    block: RepetitionCode,
    qubit_to_measure: tuple[int, int],
) -> tuple[list[Stabilizer], list[Stabilizer]]:
    """Split the initial Block stabilizers into two lists of stabilizers, one for
    each Block.

    Parameters
    ----------
    block : RepetitionCode
        Initial block to be split
    qubits_to_measure : tuple[int,int]
        Qubit to be measured in the split operation.

    Returns
    -------
    tuple[list[Stabilizer], list[Stabilizer]]
        A tuple of two lists of stabilizers, one for each block.
    """

    # Find stabilizers which are completely removed
    stabs_to_remove = [
        stab
        for stab in block.stabilizers
        if any(q == qubit_to_measure for q in stab.data_qubits)
    ]

    # Create two sets of stabilizers, one for each new block
    new_stabilizers = [
        stab for stab in block.stabilizers if stab not in stabs_to_remove
    ]

    data_qubits_block_1 = set(
        q for q in block.data_qubits if q[0] < qubit_to_measure[0]
    )
    data_qubits_block_2 = set(
        q for q in block.data_qubits if q[0] > qubit_to_measure[0]
    )

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

    # Compute total set of remaining stabilizers
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
    )


def get_logical_operator_and_updates(
    int_step: InterpretationStep,
    block: RepetitionCode,
    check_type: str,
    qubit_to_measure: tuple[int, int],
    cbit: Cbit,
) -> tuple[
    list[list[PauliOperator]],
    list[list[PauliOperator]],
    list[dict[str, tuple[str, ...]]],
    list[dict[str, tuple[str, ...]]],
    list[dict[str, tuple[Cbit, ...]]],
    list[dict[str, tuple[Cbit, ...]]],
]:
    """Finds the logical operators for the new blocks after the split operation. The
    logical spanning the chain will be split in two, the logical at the left end remains
    there for the first chain and a new one is created for the second chain, on its left
    end.

    Parameters
    ----------
    int_step : InterpretationStep
        Interpretation step containing the block to split.
    block : RepetitionCode
        Block to split.
    check_type : str
        Type of stabilizer to measure.
    qubit_to_measure : tuple[int,int]
        Qubit to be measured in the split operation.
    cbit : Cbit
        Classical bit used to measure the qubit that splits the chain.

    Returns
    -------
    tuple[
        list[list[PauliOperator]],list[list[PauliOperator]],
        list[dict[str, tuple[str, ...]]],
        list[dict[str, tuple[str, ...]]],
        list[dict[str, tuple[Cbit, ...]]],
        list[dict[str, tuple[Cbit, ...]]],
    ]
        Tuple containing the new logical operators for the blocks, the evolution,
        and the updates.
    """

    other_check_type = "X" if check_type == "Z" else "Z"

    # block 1 is the left one
    qubits_block_1 = [q for q in block.data_qubits if q[0] < qubit_to_measure[0]]
    left_boundary_qubit_1 = min(qubits_block_1, key=lambda x: x[0])

    # block 2 is the right one
    qubits_block_2 = [q for q in block.data_qubits if q[0] > qubit_to_measure[0]]
    left_boundary_qubit_2 = min(qubits_block_2, key=lambda x: x[0])

    # Extract old long and short operators
    old_short_log, old_long_log = sorted(
        (block.logical_x_operators[0], block.logical_z_operators[0]),
        key=lambda x: len(x.data_qubits),
    )

    # Define new logical operators
    new_long_log_1 = PauliOperator(
        pauli=other_check_type * len(qubits_block_1), data_qubits=qubits_block_1
    )
    long_log_evolution_1 = {new_long_log_1.uuid: (old_long_log.uuid,)}
    long_log_updates_1 = {
        new_long_log_1.uuid: (cbit,)
    }  # Give the correction to the first operator

    new_long_log_2 = PauliOperator(
        pauli=other_check_type * len(qubits_block_2), data_qubits=qubits_block_2
    )
    long_log_evolution_2 = {new_long_log_2.uuid: (old_long_log.uuid,)}
    long_log_updates_2 = {}

    # Short logicals get placed in the left boundary in the new blocks
    new_short_log_1, required_stabs_1 = block.get_shifted_equivalent_logical_operator(
        left_boundary_qubit_1
    )
    id_required_stabs_1 = [stab.uuid for stab in required_stabs_1]

    new_short_log_2, required_stabs_2 = block.get_shifted_equivalent_logical_operator(
        left_boundary_qubit_2
    )
    id_required_stabs_2 = [stab.uuid for stab in required_stabs_2]

    # If logical was already in the default position, no evolution
    if id_required_stabs_1 == []:
        new_short_log_1 = old_short_log
        short_log_evolution_1 = {}
        short_log_update_1 = {}
    # If logical in left Block was shifted, add evolution
    else:
        short_log_evolution_1 = {
            new_short_log_1.uuid: tuple([old_short_log.uuid] + id_required_stabs_1)
        }
        short_log_cbits_1 = int_step.retrieve_cbits_from_stabilizers(
            required_stabs_1, block
        )
        short_log_update_1 = {new_short_log_1.uuid: short_log_cbits_1}

    # If logical was already in the default position, no evolution
    if id_required_stabs_2 == []:
        new_short_log_2 = old_short_log
        short_log_evolution_2 = {}
        short_log_update_2 = {}
    # If logical in right Block was shifted, add evolution
    else:
        short_log_evolution_2 = {
            new_short_log_2.uuid: tuple([old_short_log.uuid] + id_required_stabs_2)
        }
        short_log_cbits_2 = int_step.retrieve_cbits_from_stabilizers(
            required_stabs_2, block
        )
        short_log_update_2 = {new_short_log_2.uuid: short_log_cbits_2}

    # New logicals
    new_logs_1 = [[new_long_log_1], [new_short_log_1]]
    new_logs_2 = [[new_long_log_2], [new_short_log_2]]

    # New operator evolution and updates
    # Find cbits for the logical updates
    log_updates_1 = [long_log_updates_1, short_log_update_1]
    log_evolution_1 = [long_log_evolution_1, short_log_evolution_1]

    log_updates_2 = [long_log_updates_2, short_log_update_2]
    log_evolution_2 = [long_log_evolution_2, short_log_evolution_2]

    return (
        new_logs_1,
        new_logs_2,
        log_evolution_1,
        log_evolution_2,
        log_updates_1,
        log_updates_2,
    )


def split(
    interpretation_step: InterpretationStep,
    operation: Split,
    same_timeslice: bool,
    debug_mode: bool,
) -> InterpretationStep:
    """
    Applicator to split a Repetition Code chain using the Split description.
    Since the Repetition code is a 1D chain, the split operation is always vertical.

    The algorithm is the following:

    - A.) DATA QUBITS

        - A.1) Find data qubit to be measured

    - B.) - CIRCUIT

        - B.1) Create a measurement circuit for the measured data qubit
        - B.2) Append the measurement circuits to the InterpretationStep circuit

    - C.) - STABILIZERS

        - C.1) Create two sets of stabilizers, one for each new block, after the split

    - D.) LOGICAL OPERATORS

        - D.1) Find logical operators which are not partially measured
        - D.2) Find logical operators which have to be split
        - D.3) Create new logical operators and final to initial mapping
        - D.4) Update `logical_x/z_evolution`
        - D.5) Update `logical_x/z_updates`

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
        Activating debug mode will enable commutation validation for Block.

    Returns
    -------
    InterpretationStep
        Interpretation step after the split operation.
    """

    # Check that the operation can be performed
    block = split_consistency_check(interpretation_step, operation)

    # Extract split position
    split_position = operation.split_position

    # Extract check type
    check_type = block.check_type

    # A) - DATA QUBITS
    #   A.1) Find data qubit to be measured
    qubit_to_measure = find_qubit_to_measure(block, split_position)

    # B) - CIRCUIT
    #    B.1) Create a measurement circuit for the measured data qubit
    circuit_name = f"split {block.unique_label} at {split_position}"
    split_circuit, cbit = create_split_circuit(
        interpretation_step, check_type, qubit_to_measure, circuit_name
    )

    #    B.2) Append the measurement circuits to the InterpretationStep circuit
    interpretation_step.append_circuit_MUT(split_circuit, same_timeslice)

    # C) - STABILIZERS
    #    C.1) Create two sets of stabilizers, one for each new block, after the split
    (
        new_stabs_block_1,
        new_stabs_block_2,
    ) = find_new_stabilizers(block, qubit_to_measure)

    # D) LOGICAL OPERATORS
    #    D.1) Extract logical operators and updates
    (
        new_logs_1,
        new_logs_2,
        log_evolution_1,
        log_evolution_2,
        log_updates_1,
        log_updates_2,
    ) = get_logical_operator_and_updates(
        interpretation_step, block, check_type, qubit_to_measure, cbit
    )

    # D.2) Update the logical operator history
    # Evolution and updates are returned in the order [long_log,short_log]
    # Depending on the check_type we invert the order of the evolution and
    # updates output, to match the order of variable assignment below
    ordering = 1 if check_type == "Z" else -1
    x_is_long_operator = check_type == "Z"

    new_log_x_ops_1, new_log_z_ops_1 = new_logs_1[::ordering]
    new_log_x_ops_2, new_log_z_ops_2 = new_logs_2[::ordering]
    logical_x_evolution_1, logical_z_evolution_1 = log_evolution_1[::ordering]
    logical_x_evolution_2, logical_z_evolution_2 = log_evolution_2[::ordering]
    logical_x_updates_1, logical_z_updates_1 = log_updates_1[::ordering]
    logical_x_updates_2, logical_z_updates_2 = log_updates_2[::ordering]

    interpretation_step.logical_x_evolution.update(
        logical_x_evolution_1 | logical_x_evolution_2
    )
    # Update the logical X operator updates:
    # The first block's operator always inherits updates, only the second block's short
    # operator also inherits updates
    for op_id, cbits in logical_x_updates_1.items():
        interpretation_step.update_logical_operator_updates_MUT("X", op_id, cbits, True)
    for op_id, cbits in logical_x_updates_2.items():
        interpretation_step.update_logical_operator_updates_MUT(
            "X", op_id, cbits, not x_is_long_operator
        )

    interpretation_step.logical_z_evolution.update(
        logical_z_evolution_1 | logical_z_evolution_2
    )
    # Update the logical Z operator updates:
    # The first block's operator always inherits updates, only the second block's short
    # operator also inherits updates
    for op_id, cbits in logical_z_updates_1.items():
        interpretation_step.update_logical_operator_updates_MUT("Z", op_id, cbits, True)
    for op_id, cbits in logical_z_updates_2.items():
        interpretation_step.update_logical_operator_updates_MUT(
            "Z", op_id, cbits, x_is_long_operator
        )

    # E) NEW BLOCK AND NEW INTERPRETATION STEP
    #    E.1) Create the new blocks
    new_block_1 = RepetitionCode(
        stabilizers=new_stabs_block_1,
        logical_x_operators=new_log_x_ops_1,
        logical_z_operators=new_log_z_ops_1,
        unique_label=operation.output_blocks_name[0],
        skip_validation=not debug_mode,
    )
    new_block_2 = RepetitionCode(
        stabilizers=new_stabs_block_2,
        logical_x_operators=new_log_x_ops_2,
        logical_z_operators=new_log_z_ops_2,
        unique_label=operation.output_blocks_name[1],
        skip_validation=not debug_mode,
    )
    #    E.2) Update the block history
    interpretation_step.update_block_history_and_evolution_MUT(
        new_blocks=(new_block_1, new_block_2), old_blocks=(block,)
    )

    return interpretation_step
