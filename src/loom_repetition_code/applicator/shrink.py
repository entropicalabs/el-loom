"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=duplicate-code
from loom.eka import ChannelType, Circuit, PauliOperator, Stabilizer
from loom.eka.operations import Shrink
from loom.eka.utilities import Direction
from loom.interpreter import InterpretationStep
from loom.interpreter.utilities import Cbit

from ..code_factory import RepetitionCode


# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
def shrink_consistency_check(
    interpretation_step: InterpretationStep, operation: Shrink
) -> RepetitionCode:
    """
    Check the consistency of the shrink operation.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        InterpretationStep containing the block to shrink.
    operation : Shrink
        Shrink operation description.

    Returns
    -------
    RepetitionCode
        Block to shrink.
    """
    # Extract Block
    block = interpretation_step.get_block(operation.input_block_name)

    # Check valid Block type
    if not isinstance(block, RepetitionCode):
        raise ValueError(
            f"This shrink operation is not supported for {type(block)} blocks."
        )

    # Check shrink size will leave more than one data qubit in the block
    if operation.length >= (len(block.data_qubits) - 1):
        raise ValueError("Shrink size is too large.")

    # Check shrink is applied in the correct direction
    if operation.direction in [Direction.BOTTOM, Direction.TOP]:
        raise ValueError(
            "Repetition code does not support "
            f"shrinking in the {operation.direction} direction."
        )

    return block


def get_qubits_to_measure(
    block: RepetitionCode, direction: Direction, length: int
) -> list[tuple[int, int]]:
    """Find the qubits to measure during the shrink operation.

    Parameters
    ----------
    block : RepetitionCode
        Block to shrink.
    direction : Direction
        Direction of the shrink operation.
    length : int
        Length of the shrink operation.

    Returns
    -------
    list[tuple[int, int]]
        List containing the qubits to measure.
    """

    # Extract qubit in the boundary to be shrunk
    boundary_qubit = block.boundary_qubits(direction=direction)

    # Generate a list of vectors shifting the new left boundary qubit after shrink
    match direction:
        case Direction.LEFT:
            shift_vectors = [(i, 0) for i in range(length)]
        case Direction.RIGHT:
            shift_vectors = [(-i, 0) for i in range(length)]

    # Extract qubits to measure, using the boundary as a reference
    qubits_to_measure = [
        tuple(coord1 + coord2 for coord1, coord2 in zip(boundary_qubit, shift_vect))
        for shift_vect in shift_vectors
    ]

    return qubits_to_measure  # type: ignore


def find_shrink_circuit(
    interpretation_step: InterpretationStep,
    check_type: str,
    qubits_to_measure: list[tuple[int, int]],
    circuit_name: str,
) -> tuple[Circuit, list[Cbit]]:
    """Generate a circuit to measure the qubits to be shrunk. Qubits are measured in
    the Z basis. If the check type is Z, the qubits are measured directly, else they
    are applied a Hadamard gate before measurement.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        Interpretation step containing the block to shrink.
    check_type : str
        Type of the stabilizer operators.
    qubits_to_measure : list[tuple[int,int]]
        List of qubits to measure.
    circuit_name : str
        Name of the circuit.

    Returns
    -------
    tuple[Circuit, list[Cbit]]
        Tuple containing the shrink circuit and the classical bits associated with the
        measured qubits.
    """

    # Generate fresh classical bits and extract the channels
    cbits = [interpretation_step.get_new_cbit_MUT(f"c_{q}") for q in qubits_to_measure]

    cbit_channels = [
        interpretation_step.get_channel_MUT(
            f"{cbit[0]}_{cbit[1]}", channel_type=ChannelType.CLASSICAL
        )
        for cbit in cbits
    ]

    # Create sequence of measurements for every measured data qubit
    measure_circuit_seq = [
        [
            Circuit(
                "Measurement",
                channels=[interpretation_step.get_channel_MUT(q), cbc],
            )
            for q, cbc in zip(qubits_to_measure, cbit_channels, strict=True)
        ]
    ]

    if check_type == "X":
        # If phaseflip code qubits can already be measured
        shrink_circuit_list = measure_circuit_seq
    else:
        # If bitflip code qubits need to be Hadamard transformed before measurement
        basis_change_circuit_seq = [
            [
                Circuit("H", channels=[interpretation_step.get_channel_MUT(qb)])
                for qb in qubits_to_measure
            ]
        ]

        # Add basis change before measurement
        shrink_circuit_list = basis_change_circuit_seq + measure_circuit_seq

    # Construct shrink circuit
    shrink_circuit = Circuit(
        name=circuit_name,
        circuit=shrink_circuit_list,
    )

    return shrink_circuit, cbits


def find_new_stabilizers(
    block: RepetitionCode, qubits_to_measure: list[tuple[int, int]]
) -> list[Stabilizer]:
    """Find the new set of stabilizers after the shrink operation.
    We also compute the uuids of the stabilizers that need to be removed, if the block
    shrunk from the left, as they will be used to update one of the logical operators.

    Parameters
    ----------
    block : RepetitionCode
        Block to shrink.
    qubits_to_measure : list[tuple[int,int]]
        List of qubits to measure.

    Returns
    -------
    new_stabs : list[Stabilizer]
        New set of stabilizers after the shrink operation.
    """

    stabs_to_remove = [
        stab
        for stab in block.stabilizers
        if any(qb in qubits_to_measure for qb in stab.data_qubits)
    ]

    # Combine the stabilizers to get the new set of stabilizers
    new_stabs = list(set(block.stabilizers) - set(stabs_to_remove))

    return new_stabs


def get_logical_operator_and_updates(
    interpretation_step: InterpretationStep,
    block: RepetitionCode,
    check_type: str,
    is_left: bool,
    qubits_to_measure: list[tuple[int, int]],
    cbits: list[Cbit],
) -> tuple[
    list[list[PauliOperator]],
    list[dict[str, tuple[str, ...]]],
    list[dict[str, tuple[Cbit, ...]]],
]:
    """
    Generate new logical operators and their respective evolution and updates to be
    added to the interpretation step. We treat the two logicals separately in as "long"
    and "short". The long corresponds to the one covering the entire chain, with type
    opposite to the check type. The short corresponds to the one covering the left
    boundary qubit by default, with type equal to the check type.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        The interpretation step.
    block : RepetitionCode
        Block to shrink.
    check_type : str
        Type of the stabilizer operators.
    is_left : bool
        Boolean defining whether shrink operations is performed from right or left.
    qubits_to_measure : list[tuple[int,int]]
        List of qubits to measure.
    cbits : list[Cbit]
        List of classical bits associated with measured qubits.

    Returns
    -------
    tuple[
        list[list[PauliOperator]],
        list[dict[str,tuple[str,...]]],
        list[dict[str,tuple[Cbit,...]]]
    ]
        Tuple containing the new logical operators, their evolution and updates.
    """

    other_check_type = "X" if check_type == "Z" else "Z"

    # Find remaining qubits
    remaining_qubits = list(set(block.data_qubits) - set(qubits_to_measure))

    # Extract old long and short operators
    old_logs = [block.logical_x_operators[0], block.logical_z_operators[0]]
    old_short_log, old_long_log = sorted(old_logs, key=lambda x: len(x.data_qubits))

    # Check if the short logical qubit inside the qubits to measure
    to_shift = old_short_log.data_qubits[0] in qubits_to_measure

    # Define new long logical operator and updates
    new_long_log = PauliOperator(
        pauli=other_check_type * len(remaining_qubits), data_qubits=remaining_qubits
    )
    long_log_updates = {new_long_log.uuid: tuple(cbits)}
    long_log_evolution = {new_long_log.uuid: (old_long_log.uuid,)}

    # Define new short logical operator and updates
    # If the short logical operator is inside the qubits to measure, shift it
    if to_shift:
        selector_function = min if is_left else max
        new_qubit = selector_function(remaining_qubits, key=lambda x: x[0])
        new_short_log, stabs_required = block.get_shifted_equivalent_logical_operator(
            new_qubit
        )
        id_stabs_required = [stab.uuid for stab in stabs_required]
        short_log_evolution = {
            new_short_log.uuid: tuple([old_short_log.uuid] + id_stabs_required)
        }

        # Add to the updates the cbits associated with the measured qubits and the
        # stabilizers required for the shift.
        # These cbits are the last syndromes measured for those stabilizers.
        cbits = interpretation_step.retrieve_cbits_from_stabilizers(
            stabs_required, block
        )

        short_log_updates = {new_short_log.uuid: cbits}

    else:
        new_short_log = old_short_log
        short_log_evolution = {}
        short_log_updates = {}

    # New logicals
    new_logs = [[new_long_log], [new_short_log]]

    # Updates and evolution
    log_evolution = [long_log_evolution, short_log_evolution]
    log_updates = [long_log_updates, short_log_updates]

    return new_logs, log_evolution, log_updates


def shrink(
    interpretation_step: InterpretationStep,
    operation: Shrink,
    same_timeslice: bool,
    debug_mode: bool,
) -> InterpretationStep:
    """
    Shrink a Repetition Code chain in the left or right specified direction.

    The algorithm is the following:

    - A.) DATA QUBITS

        - A.1) Find measured data qubits during shrink

    - B.) CIRCUIT

        - B.1) Generate shrink circuit and classical bits
        - B.2) Add circuit to the interpretation step

    - C.) - STABILIZERS

        - C.1) Find new set of stabilizers

    - D.) LOGICAL OPERATORS

        - D.1) Extract logical operators and updates
        - D.2) Update the logical operator history

    - E.) NEW BLOCK AND NEW INTERPRETATION STEP
        - E.1) Create the new block
        - E.2) Update the block history

    Parameters
    ----------
    interpretation_step : InterpretationStep
        Interpretation step containing the block to shrink.
    operation : Shrink
        Shrink operation description.
    same_timeslice : bool
        Flag indicating whether the operation is part of the same timestep as the
        previous operation.
    debug_mode : bool
        Flag indicating whether the interpretation should be done in debug mode.
        Activating debug mode will enable commutation validation for Block.

    Returns
    -------
    InterpretationStep
        Interpretation step after the shrink operation.
    """

    # Check consistency of the shrink operation
    block = shrink_consistency_check(interpretation_step, operation)

    # Get shrink direction
    is_left = operation.direction == Direction.LEFT

    # Extract stabilizer type
    check_type = block.check_type

    # A) DATA QUBITS
    # A.1) Find measured data qubits during shrink
    qubits_to_measure = get_qubits_to_measure(
        block, direction=operation.direction, length=operation.length
    )

    # B) CIRCUIT
    # B.1) Generate shrink circuit and classical bits
    circuit_name = (
        f"shrink {block.unique_label} by {operation.length} from {operation.direction}"
    )
    shrink_circuit, cbits = find_shrink_circuit(
        interpretation_step, check_type, qubits_to_measure, circuit_name
    )

    # B.2) Add circuit to the interpretation step
    interpretation_step.append_circuit_MUT(shrink_circuit, same_timeslice)

    # C) STABILIZERS
    # C.1) Find new set of stabilizers
    new_stabilizers = find_new_stabilizers(block, qubits_to_measure)

    # D) LOGICAL OPERATORS
    # D.1) Extract logical operators and updates
    new_logs, log_evolution, log_updates = get_logical_operator_and_updates(
        interpretation_step,
        block,
        check_type,
        is_left,
        qubits_to_measure,
        cbits,
    )

    # D.2) Update the logical operator history
    ordering = 1 if check_type == "Z" else -1
    new_log_x_ops, new_log_z_ops = new_logs[::ordering]
    logical_x_evolution, logical_z_evolution = log_evolution[::ordering]
    logical_x_updates, logical_z_updates = log_updates[::ordering]

    # Update the logical operator evolution
    interpretation_step.logical_x_evolution.update(logical_x_evolution)
    interpretation_step.logical_z_evolution.update(logical_z_evolution)

    # Update the logical operator updates
    for op_id, measurements in logical_x_updates.items():
        interpretation_step.update_logical_operator_updates_MUT(
            "X",
            op_id,
            measurements,
            True,
        )
    for op_id, measurements in logical_z_updates.items():
        interpretation_step.update_logical_operator_updates_MUT(
            "Z",
            op_id,
            measurements,
            True,
        )

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
