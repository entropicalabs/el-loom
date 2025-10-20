"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from loom.eka import Circuit, PauliOperator, Stabilizer
from loom.eka.operations import Grow
from loom.eka.utilities import Direction
from loom.interpreter import InterpretationStep

from ..code_factory import RepetitionCode


# pylint: disable=duplicate-code, disable=too-many-locals
def grow_consistency_check(
    interpretation_step: InterpretationStep, operation: Grow
) -> RepetitionCode:
    """
    Check the consistency of the grow operation.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        InterpretationStep containing the block to grow.
    operation : Grow
        Grow operation description.

    Returns
    -------
        The RepetitionCode block to grow.

    Raises
    ------
    ValueError
        If the Block is not RepetitionCode.
        If the grown operation direction is not RIGHT or LEFT.
        If growing in the left direction goes beyond the lattice boundary.
    """
    # Extract Block
    block = interpretation_step.get_block(operation.input_block_name)

    # Check valid Block type
    if not isinstance(block, RepetitionCode):
        raise ValueError(
            f"This grow operation is not supported for {type(block)} blocks."
        )

    # Check grow is applied in the correct direction
    if operation.direction not in [Direction.RIGHT, Direction.LEFT]:
        raise ValueError(
            "Repetition code does not support growing in the "
            f"{operation.direction} direction."
        )
    # Check that we are not growing beyond the boundaries of the lattice
    if operation.direction == Direction.LEFT:
        left_boundary_position = block.boundary_qubits(Direction.LEFT)[0]
        if left_boundary_position - operation.length < 0:
            raise ValueError("Cannot grow beyond the boundary of the lattice.")

    return block


def get_new_data_qubits_info(
    block: RepetitionCode, direction: Direction, length: int
) -> list[tuple[int, int]]:
    """Find the qubits to add during the grow operation.

    Parameters
    ----------
    block : RepetitionCode
        Block to grow.
    direction : Direction
        Direction of the grow operation.
    length : int
        Length of the grow operation.

    Returns
    -------
    list[tuple[int, int]]
        List containing the new qubits to add.
    """

    left_boundary_qubit = block.boundary_qubits(Direction.LEFT)

    # Find the new data qubits that need to be added
    match direction:
        case Direction.LEFT:
            new_data_qubits = [
                (-i + left_boundary_qubit[0] - 1, 0) for i in range(length)
            ]
        case Direction.RIGHT:
            right_boundary_qubit = block.boundary_qubits(Direction.RIGHT)
            new_data_qubits = [
                (i + right_boundary_qubit[0] + 1, 0) for i in range(length)
            ]

    return new_data_qubits


def create_grow_circuit(
    interpretation_step: InterpretationStep,
    check_type: str,
    new_data_qubits: list[tuple[int, int]],
    circuit_name: str,
) -> Circuit:
    """
    Generates the circuit required to reset the new data that get added during the
    grow operation.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        Interpretation step containing the block to grow.
    check_type : str
        Type of stabilizers in the code.
    new_data_qubits : list[tuple[int,int]]
        List of new data qubits to reset.
    circuit_name : str
        Name of the circuit.

    Returns
    -------
    Circuit
        Circuit to reset the new data qubits.
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

    # Circuit for the grow operation
    grow_circuit = Circuit(
        name=circuit_name,
        circuit=reset_sequence,
    )

    return grow_circuit


def find_new_stabilizers(
    block: RepetitionCode,
    check_type: str,
    is_left: bool,
    new_data_qubits: list[tuple[int, int]],
) -> list[Stabilizer]:
    """Find the new stabilizers that will form the grown block.

    Parameters
    ----------
    block : RepetitionCode
        Block to grow.
    check_type : str
        Type of stabilizer to add.
    is_left: bool
        Whether the grow operation is to the left or right.
    new_data_qubits : list[tuple[int,int]]
        New data qubits to add.

    Returns
    -------
    list[Stabilizer]
        List containing the new stabilizers forming the grown block.
    """

    # Define the stabilizers to add
    stabs_to_add = [
        Stabilizer(
            pauli=check_type * 2,
            data_qubits=sorted(
                [(coord - (1 - 2 * is_left), 0), (coord, 0)], key=lambda x: x[0]
            ),
            ancilla_qubits=[(coord - (1 - is_left), 1)],
        )
        for coord, _ in new_data_qubits
    ]
    # Combine the stabilizers to get the new set of stabilizers
    new_stabilizers = list(block.stabilizers) + stabs_to_add

    return new_stabilizers


def get_logical_operator_and_evolution(
    block: RepetitionCode,
    check_type: str,
    new_data_qubits: list[tuple[int, int]],
) -> tuple[list[list[PauliOperator]], list[dict[str, tuple[str, ...]]]]:
    """
    Extract the logical operators and the evolution after the grow operation.

    Parameters
    ----------
    block : RepetitionCode
        Block to grow.
    check_type : str
        Type of stabilizer defining the code.
    new_data_qubits : list[tuple[int,int]]
        New data qubits to add.

    Returns
    -------
    tuple[list[list[PauliOperator]], list[dict[str, tuple[str, ...]]]]
        Tuple containing the new logical operators, the evolution of the logical X
        operators, and the evolution of the logical Z operators.
    """

    complementary_check_type = "X" if check_type == "Z" else "Z"

    # Combine old and new data qubits
    all_qubits = list(block.data_qubits) + new_data_qubits

    # Extract old long and short operators
    old_logs = [block.logical_x_operators[0], block.logical_z_operators[0]]
    old_short_log, old_long_log = sorted(old_logs, key=lambda x: len(x.data_qubits))

    # Define new logical operators
    new_long_log = PauliOperator(
        pauli=complementary_check_type * len(all_qubits), data_qubits=all_qubits
    )

    new_short_log = old_short_log

    # New logicals
    new_logs = [[new_long_log], [new_short_log]]

    # Log evolution
    long_log_evolution = {new_long_log.uuid: (old_long_log.uuid,)}
    short_log_evolution = {}
    log_evolution = [long_log_evolution, short_log_evolution]

    return new_logs, log_evolution


def grow(  # pylint: disable=too-many-locals
    interpretation_step: InterpretationStep,
    operation: Grow,
    same_timeslice: bool,
    debug_mode: bool,
) -> InterpretationStep:
    """
    Grow a Repetition Code chain in the specified direction (left or right).

    The algorithm is the following:

    - A.) DATA QUBITS

        - A.1) Extract new data qubits

    - B.) CIRCUIT

        - B.1) Generate the circuit
        - B.2) Add the circuit to the interpretation step

    - C.) STABILIZERS

        - C.1) Find the new stabilizers that get added
        - C.2) Add to current set of stabilizers

    - D.) LOGICAL OPERATORS

        - D.1) Extract logical operators and updates
        - D.2) Update the logical operator history

    - E.) NEW BLOCK

        - E.1) Create the new block
        - E.2) Update the block history

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
        Activating debug mode will enable commutation validation for Block.

    Returns
    -------
    InterpretationStep
        Interpretation step after the grow operation.
    """

    # Check consistency of the grow operation
    block = grow_consistency_check(interpretation_step, operation)

    # Extract stabilizer type
    check_type = block.check_type
    is_left = operation.direction == Direction.LEFT

    # A) DATA QUBITS
    # A.1) Extract new data qubits
    new_data_qubits = get_new_data_qubits_info(
        block, direction=operation.direction, length=operation.length
    )

    # B) CIRCUIT
    # B.1) Generate the circuit
    circuit_name = (
        f"grow {block.unique_label} by {operation.length} to the {operation.direction}"
    )
    grow_circuit = create_grow_circuit(
        interpretation_step, check_type, new_data_qubits, circuit_name
    )

    # B.2) Add the circuit to the interpretation step
    interpretation_step.append_circuit_MUT(grow_circuit, same_timeslice)

    # C) STABILIZERS
    # C.1) Find the new set of stabilizers
    new_stabilizers = find_new_stabilizers(block, check_type, is_left, new_data_qubits)

    # D) LOGICAL OPERATORS
    # D.1) Extract logical operators and updates
    new_logs, log_evolution = get_logical_operator_and_evolution(
        block,
        check_type,
        new_data_qubits,
    )

    # D.2) Update the logical operator history
    ordering = 1 if check_type == "Z" else -1
    new_log_x_ops, new_log_z_ops = new_logs[::ordering]
    logical_x_evolution, logical_z_evolution = log_evolution[::ordering]

    # Update the logical operator evolution
    interpretation_step.logical_x_evolution.update(logical_x_evolution)
    interpretation_step.logical_z_evolution.update(logical_z_evolution)
    # Update the logical operator updates
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
        unique_label=block.unique_label,
        skip_validation=not debug_mode,
    )

    # E.2) Update the block history
    interpretation_step.update_block_history_and_evolution_MUT(
        new_blocks=(new_block,),
        old_blocks=(block,),
    )

    return interpretation_step
