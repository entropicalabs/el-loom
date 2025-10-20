"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from loom.eka import Circuit, PauliOperator, Stabilizer
from loom.eka.operations import Merge
from loom.eka.utilities import Direction, Orientation
from loom.interpreter import InterpretationStep

from ..code_factory import RepetitionCode


# pylint: disable=too-many-locals
def merge_consistency_check(
    interpretation_step: InterpretationStep, operation: Merge
) -> tuple[RepetitionCode, RepetitionCode]:
    """
    Check the consistency of the merge operation.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        InterpretationStep containing the blocks to merge.
    operation : Merge
        Merge operation description.

    Returns
    -------
    tuple[RepetitionCode,RepetitionCode]
        Blocks to merge.
    """

    name_1, name_2 = operation.input_blocks_name
    block_1 = interpretation_step.get_block(name_1)
    block_2 = interpretation_step.get_block(name_2)

    # Check valid Block type
    if not isinstance(block_1, RepetitionCode) or not isinstance(
        block_2, RepetitionCode
    ):
        wrong_types = tuple(
            set((type(block_1), type(block_2))).difference({RepetitionCode})
        )
        raise TypeError(
            f"This merge operation is not supported for {tuple(wrong_types)} blocks."
        )

    # Ensure block_1 is left and block_2 is right
    if (
        block_1.boundary_qubits(Direction.LEFT)[0]
        > block_2.boundary_qubits(Direction.LEFT)[0]
    ):
        block_1, block_2 = block_2, block_1

    # Check that Blocks have same type of stabilizers
    if block_1.check_type != block_2.check_type:
        raise TypeError(
            "Cannot merge blocks with different check types: "
            f"{block_1.check_type} and {block_2.check_type}"
        )

    # Check that Blocks do not overlap
    left_boundary_block_1 = block_1.boundary_qubits(Direction.LEFT)[0]
    distance_block_1 = len(block_1.data_qubits)
    left_boundary_block_2 = block_2.boundary_qubits(Direction.LEFT)[0]

    if left_boundary_block_2 < left_boundary_block_1 + distance_block_1:
        raise ValueError("Cannot merge blocks that overlap.")

    # Check that orientation is horizontal
    if operation.orientation != Orientation.HORIZONTAL:
        raise ValueError(
            "Repetition code does not support merging in the vertical orientation."
        )

    return block_1, block_2


def get_new_data_qubits_info(
    blocks: list[RepetitionCode],
) -> list[tuple[int, int]]:
    """Get the new data qubits to be added in the merged block.

    Parameters
    ----------
    blocks : list[RepetitionCode]
        List of blocks (capped at two) to be merged.

    Returns
    -------
    list[tuple[int, int]]
        List of new data qubits to be added.
    """

    # Extract blocks
    block_1, block_2 = blocks

    # Extract boundaries
    right_boundary_block_1 = block_1.boundary_qubits(Direction.RIGHT)[0]
    left_boundary_block_2 = block_2.boundary_qubits(Direction.LEFT)[0]

    # Define new data qubits
    new_data_qubits = [
        (i, 0) for i in range(right_boundary_block_1 + 1, left_boundary_block_2)
    ]

    return new_data_qubits


def find_merge_circuit(
    interpretation_step: InterpretationStep,
    check_type: str,
    new_data_qubits: list[tuple[int, int]],
    circuit_name: str,
) -> Circuit:
    """
    Generates the circuit required to reset the new data that get added during the
    merge operation.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        Interpretation step containing the blocks to be merged.
    check_type : str
        Type of stabilizers in the codes.
    new_data_qubits : list[tuple[int,int]]
        List of new data qubits to reset.
    circuit_name : str
        Name of the circuit.

    Returns
    -------
    Circuit
        Circuit to reset the new data and ancilla qubits.
    """

    # Select reset state based on stabilizer type
    if check_type == "X":
        reset_state = "0"
    else:
        reset_state = "+"

    # Reset sequence
    reset_sequence = [
        [
            Circuit(
                name=f"reset_{reset_state}",
                channels=interpretation_step.get_channel_MUT(q, "quantum"),
            )
            for q in new_data_qubits
        ]
    ]

    # Circuit for the merge operation
    merge_circuit = Circuit(
        name=circuit_name,
        circuit=reset_sequence,
    )
    return merge_circuit


def find_new_stabilizers(
    blocks: list[RepetitionCode], check_type: str
) -> list[Stabilizer]:
    """Find the new stabilizers for the merged block.

    Parameters
    ----------
    blocks : list[RepetitionCode]
        List of blocks (capped at two) to be merged.
    check_type : str
        Type of stabilizers in the codes.

    Returns
    -------
    list[Stabilizer]
        New stabilizers for the merged block.
    """

    # Extract blocks
    block_1, block_2 = blocks

    # Extract boundaries
    right_boundary_block_1 = block_1.boundary_qubits(Direction.RIGHT)[0]
    left_boundary_block_2 = block_2.boundary_qubits(Direction.LEFT)[0]

    # Generate added stabilizers
    add_stabilizers = [
        Stabilizer(
            pauli=check_type * 2,
            data_qubits=[(i, 0), (i + 1, 0)],
            ancilla_qubits=[(i, 1)],
        )
        for i in range(right_boundary_block_1, left_boundary_block_2)
    ]

    # Generate new stabilizers
    new_stabilizers = (
        list(block_1.stabilizers) + add_stabilizers + list(block_2.stabilizers)
    )

    return new_stabilizers


def get_logical_operator_and_evolution(  # pylint: disable=too-many-locals
    blocks: list[RepetitionCode],
    check_type: str,
    new_data_qubits: list[tuple[int, int]],
) -> tuple[list[list[PauliOperator]], list[dict]]:
    """
    Get the new logical operators and evolution for the merged block.

    Parameters
    ----------
    blocks : list[RepetitionCode]
        List of blocks (capped at two) to be merged.
    check_type : str
        Type of stabilizers in the codes.
    new_data_qubits : list[tuple[int,int]]
        List of new data qubits to be added.

    Returns
    -------
    tuple[list[list[PauliOperator]], list[dict]]
        New logical operators and evolution for the merged block.
    """

    block_1, block_2 = blocks

    other_check_type = "X" if check_type == "Z" else "Z"

    # Extract old long and short operators
    old_logs_1 = [block_1.logical_x_operators[0], block_1.logical_z_operators[0]]
    old_short_log_1, old_long_log_1 = sorted(
        old_logs_1, key=lambda x: len(x.data_qubits)
    )
    old_logs_2 = [block_2.logical_x_operators[0], block_2.logical_z_operators[0]]
    _, old_long_log_2 = sorted(old_logs_2, key=lambda x: len(x.data_qubits))

    new_long_data_qubits = (
        list(block_1.data_qubits) + new_data_qubits + list(block_2.data_qubits)
    )
    new_long_log = PauliOperator(
        pauli=other_check_type * len(new_long_data_qubits),
        data_qubits=new_long_data_qubits,
    )

    # We impose Block 1 to be the left one in the main merge function
    left_boundary_block_1 = block_1.boundary_qubits(Direction.LEFT)
    new_short_log, stabs_required = block_1.get_shifted_equivalent_logical_operator(
        left_boundary_block_1
    )

    # New logicals
    new_logs = [[new_long_log], [new_short_log]]

    long_log_evolution = {new_long_log.uuid: (old_long_log_1.uuid, old_long_log_2.uuid)}
    if stabs_required:
        id_stabs_required = [stab.uuid for stab in stabs_required]
        short_log_evolution = {
            new_short_log.uuid: tuple([old_short_log_1.uuid] + id_stabs_required)
        }
    else:
        short_log_evolution = {}

    # Evolution
    log_evolution = [long_log_evolution, short_log_evolution]

    return new_logs, log_evolution


def merge(  # pylint: disable=too-many-locals
    interpretation_step: InterpretationStep,
    operation: Merge,
    same_timeslice: bool,
    debug_mode: bool,
):
    """
    Merge two Repetition Code chains.

    The algorithm is the following:

    - A.) DATA QUBITS

        - A.1) Find the new data qubits to be added in the merged block

    - B.) CIRCUIT

        - B.1) Find the circuit for the merge operation
        - B.2) Add the circuit to the interpretation step

    - C.) - STABILIZERS
        - C.1) Find the new stabilizers for the merged block

    - D.) LOGICAL OPERATORS

        - D.1) Find new logical operator and evolution
        - D.2) Update the logical operator history

    - E.) NEW BLOCK AND NEW INTERPRETATION STEP

        - E.1) Create the new block
        - E.2) Update the block history

    Parameters
    ----------
    interpretation_step : InterpretationStep
        Interpretation step containing the blocks to merge.
    operation : Merge
        Merge operation description.
    same_timeslice : bool
        Flag indicating whether the operation is part of the same timestep as the
        previous operation.
    debug_mode : bool
        Flag indicating whether the interpretation should be done in debug mode.
        Activating debug mode will enable commutation validation for Block.

    Returns
    -------
    InterpretationStep
        Interpretation step after the merge operation.
    """

    # Merge consistency check
    block_1, block_2 = merge_consistency_check(interpretation_step, operation)

    # Extract check type
    check_type = block_1.check_type

    # A) DATA QUBITS
    # A.1) Find the new data qubits to be added in the merged block
    new_data_qubits = get_new_data_qubits_info([block_1, block_2])

    # B) CIRCUIT
    # B.1) Find the circuit for the merge operation
    circuit_name = (
        f"merge {block_1.unique_label} and "
        f"{block_2.unique_label} into {operation.output_block_name}"
    )
    merge_circuit = find_merge_circuit(
        interpretation_step, check_type, new_data_qubits, circuit_name
    )

    # B.2) Add the circuit to the interpretation step
    interpretation_step.append_circuit_MUT(merge_circuit, same_timeslice)

    # C) STABILIZERS
    # C.1) Find the new stabilizers for the merged block
    new_stabilizers = find_new_stabilizers([block_1, block_2], check_type)

    # D) LOGICAL OPERATORS
    # D.1) Find new logical operator and evolution
    new_logs, log_evolution = get_logical_operator_and_evolution(
        [block_1, block_2], check_type, new_data_qubits
    )

    # D.2) Update the logical operator history
    ordering = 1 if check_type == "Z" else -1
    new_log_x_ops, new_log_z_ops = new_logs[::ordering]
    logical_x_evolution, logical_z_evolution = log_evolution[::ordering]
    interpretation_step.logical_x_evolution.update(logical_x_evolution)
    interpretation_step.logical_z_evolution.update(logical_z_evolution)
    # Inherit updates for the new operators
    for op in new_log_x_ops:
        interpretation_step.update_logical_operator_updates_MUT("X", op.uuid, (), True)
    for op in new_log_z_ops:
        interpretation_step.update_logical_operator_updates_MUT("Z", op.uuid, (), True)

    # E) NEW BLOCK
    # E.1) Create the new block
    new_block = RepetitionCode(
        stabilizers=new_stabilizers,
        logical_x_operators=new_log_x_ops,
        logical_z_operators=new_log_z_ops,
        unique_label=operation.output_block_name,
        skip_validation=not debug_mode,
    )

    # E.2) Update the block history
    interpretation_step.update_block_history_and_evolution_MUT(
        new_blocks=(new_block,),
        old_blocks=(block_1, block_2),
    )

    return interpretation_step
