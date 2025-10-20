"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from loom.eka import Circuit, ChannelType, PauliOperator, Stabilizer
from loom.interpreter import InterpretationStep, Syndrome
from loom.interpreter.applicator import measureblocksyndromes, generate_syndromes
from loom.eka.utilities import (
    Direction,
    SingleQubitPauliEigenstate,
    ResourceState,
    Orientation,
)
from loom.eka.operations import MeasureBlockSyndromes, StateInjection

from ..code_factory import RotatedSurfaceCode


# pylint: disable=duplicate-code
def state_injection(  # pylint: disable=too-many-locals
    interpretation_step: InterpretationStep,
    operation: StateInjection,
    same_timeslice: bool,
    debug_mode: bool,
):
    """
    Inject a resource state into the specified block. This operation resets the central
    qubit of the block into the T state and resets the rest of the data qubits into four
    quadrants, such that two quadrants are in the $|0⟩$ state and two quadrants are in 
    the $|+⟩$ state. This ensures that the Z stabilizer measurements are deterministic 
    in the $|0⟩$ quadrants and the X stabilizer measurements are deterministic in the 
    $|+⟩$ quadrants.

    Note: the syndromes are measured once at the end of the operation in order to
    displace the logical operators to the top-left corner of the block.

    The algorithm is the following:

    - A.) CIRCUIT
    
        - A.1) Resets the central qubit in a physical T state.
        - A.2) Resets the rest of the data qubits into four quadrants, such that two \
        quadrants are in the $|0⟩$ state and two quadrants are in the $|+⟩$ state.
            
    - B.) LOGICAL OPERATORS
    
        - B.1) Displace the logical operators to the center of the block.
        
    - C.) SYNDROMES
    
        - C.1) Creates syndromes for the block after the T state injection. Only \
        the stabilizers that are deterministic generate syndromes.
            
    - D.) MEASURE BLOCK SYNDROMES
    
        - D.1) Measure the block syndromes.
        
    - E.) DISPLACE LOGICAL OPERATORS
    
        - E.1) Create the new logical operators on the top-left corner of the block.
        
    - F.) WRAP THE CIRCUIT
    
        - F.1) Wrap the state injection circuit in a single circuit with the correct \
        duration and empty timeslices.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        The interpretation step containing the input block.
    operation : StateInjection
        The operation containing the name of the input block to inject the T state into.
    same_timeslice : bool
        Whether to append the circuit to the same timeslice as the current
        interpretation step.
    debug_mode : bool
        Flag indicating whether the interpretation should be done in debug mode.
        Currently, the effects of debug mode are:
        - Disabling the commutation validation of Block
    """
    input_block = interpretation_step.get_block(operation.input_block_name)
    init_circ_len = len(interpretation_step.intermediate_circuit_sequence)

    if not isinstance(input_block, RotatedSurfaceCode):
        raise TypeError(
            f"Expected input_block to be of type RotatedSurfaceCode, "
            f"but got {type(input_block)}."
        )
    if not all(s % 2 == 1 for s in input_block.size):
        raise ValueError(
            f"Expected input_block.size to be all odd, but got {input_block.size}."
        )

    # A) CIRCUIT
    #   A.1) Resets the central qubit in a physical resource state.
    qubit_to_reset = (
        input_block.upper_left_qubit[0] + input_block.size[0] // 2,
        input_block.upper_left_qubit[1] + input_block.size[1] // 2,
        input_block.upper_left_qubit[2],
    )
    resource_state_circuit = get_physical_state_reset(
        interpretation_step, input_block, qubit_to_reset, operation.resource_state
    )

    #   A.2) Resets the rest of the data qubits into four quadrants
    reset_circuit, deterministic_stabs = reset_into_four_quadrants(
        interpretation_step,
        input_block,
    )

    state_injection_circuit = Circuit(
        name=(
            f"Inject {operation.resource_state.value} into block "
            f"{input_block.unique_label}"
        ),
        circuit=((resource_state_circuit, reset_circuit),),
    )
    interpretation_step.append_circuit_MUT(
        circuit=state_injection_circuit,
        same_timeslice=False,  # Prevent the same timeslice flag from being set
    )

    # B) LOGICAL OPERATORS
    # Displace the logical operators to the center of the block.
    centered_x_op, centered_z_op = find_centered_logical_operators(
        input_block=input_block, center_qubit=qubit_to_reset
    )

    centered_block = RotatedSurfaceCode(
        stabilizers=input_block.stabilizers,
        logical_x_operators=[centered_x_op],
        logical_z_operators=[centered_z_op],
        syndrome_circuits=input_block.syndrome_circuits,
        stabilizer_to_circuit=input_block.stabilizer_to_circuit,
        unique_label=input_block.unique_label,
        skip_validation=not debug_mode,
    )
    interpretation_step.update_block_history_and_evolution_MUT(
        new_blocks=(centered_block,), old_blocks=(input_block,)
    )

    # C) SYNDROMES
    # Create syndromes for the block after the state injection.
    deterministic_syndromes = create_deterministic_syndromes(
        interpretation_step=interpretation_step,
        block=centered_block,
        deterministic_stabs=deterministic_stabs,
    )
    interpretation_step.append_syndromes_MUT(syndromes=deterministic_syndromes)

    # D) MEASURE BLOCK SYNDROMES
    interpretation_step = measureblocksyndromes(
        interpretation_step=interpretation_step,
        operation=MeasureBlockSyndromes(
            input_block_name=centered_block.unique_label, n_cycles=1
        ),
        same_timeslice=False,  # Prevent the same timeslice flag from being set
        debug_mode=debug_mode,
    )

    # E) DISPLACE LOGICAL OPERATORS
    # Create the new logical operators
    new_x_op, required_x_stabs = centered_block.get_shifted_equivalent_logical_operator(
        initial_operator=centered_block.logical_x_operators[0],
        new_upleft_qubit=centered_block.upper_left_qubit,
    )
    new_z_op, required_z_stabs = centered_block.get_shifted_equivalent_logical_operator(
        initial_operator=centered_block.logical_z_operators[0],
        new_upleft_qubit=centered_block.upper_left_qubit,
    )
    # Update the block with the new logical operators
    new_block = RotatedSurfaceCode(
        stabilizers=centered_block.stabilizers,
        logical_x_operators=[new_x_op],
        logical_z_operators=[new_z_op],
        syndrome_circuits=centered_block.syndrome_circuits,
        stabilizer_to_circuit=centered_block.stabilizer_to_circuit,
        unique_label=centered_block.unique_label,
        skip_validation=not debug_mode,
    )
    interpretation_step.update_block_history_and_evolution_MUT(
        new_blocks=(new_block,), old_blocks=(centered_block,)
    )
    # Add the required syndromes
    required_x_cbits = interpretation_step.retrieve_cbits_from_stabilizers(
        stabs_required=required_x_stabs, current_block=new_block
    )
    required_z_cbits = interpretation_step.retrieve_cbits_from_stabilizers(
        stabs_required=required_z_stabs, current_block=new_block
    )
    interpretation_step.update_logical_operator_updates_MUT(
        operator_type="X",
        logical_operator_id=new_x_op.uuid,
        new_updates=required_x_cbits,
        inherit_updates=True,
    )
    interpretation_step.update_logical_operator_updates_MUT(
        operator_type="Z",
        logical_operator_id=new_z_op.uuid,
        new_updates=required_z_cbits,
        inherit_updates=True,
    )

    # F) WRAP THE CIRCUIT
    new_len = len(interpretation_step.intermediate_circuit_sequence)
    len_auxcnot = new_len - init_circ_len
    circuit_seq = interpretation_step.pop_intermediate_circuit_MUT(len_auxcnot)

    # Wrap the state injection circuit in a single circuit with the correct duration
    # and empty timeslices
    wrapped_circuit_seq = ()
    for timeslice in circuit_seq:
        timespan = max(composite_circuit.duration for composite_circuit in timeslice)
        # Create a circuit with empty timeslices and align circuits
        template_circ = (
            tuple(composite_circuit for composite_circuit in timeslice),
        ) + ((),) * (timespan - 1)
        wrapped_circuit_seq += template_circ

    wrapped_circuit = Circuit(
        name=(
            f"Inject {operation.resource_state.value} into block "
            f"{input_block.unique_label} and measure syndromes"
        ),
        circuit=wrapped_circuit_seq,
    )
    # Append the wrapped circuit to the interpretation step with same_timeslice
    interpretation_step.append_circuit_MUT(wrapped_circuit, same_timeslice)

    return interpretation_step


def get_physical_state_reset(
    interpretation_step: InterpretationStep,
    input_block: RotatedSurfaceCode,
    qubit_to_reset: tuple[int, int, int],
    resource_state: ResourceState,
):
    """
    Resets a physical qubit in the given resource state. The given qubit must be a data
    qubit in the input block.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        The interpretation step containing the input block.
    input_block : RotatedSurfaceCode
        The input block containing the data qubits.
    qubit_to_reset : tuple[int, int, int]
        The qubit to reset into the given state. It must be a data qubit in the input
        block.
    resource_state: ResourceState
        The resource state to reset the qubit into. Currently, only T and S states are
        supported.

    Returns
    -------
    Circuit
        A circuit that resets the specified qubit into the given resource state, which
        consists of a reset operation in $|+⟩$ followed by a physical gate that depends
        on the resource state.
    """
    if qubit_to_reset not in input_block.data_qubits:
        raise ValueError(
            f"Qubit {qubit_to_reset} is not a data qubit in the given block"
            f" {input_block.unique_label}."
        )

    qubit_channel = interpretation_step.get_channel_MUT(
        qubit_to_reset, channel_type=ChannelType.QUANTUM
    )
    match resource_state:
        case ResourceState.T:
            gate_sequence = (
                Circuit(name="Reset_+", channels=[qubit_channel]),
                Circuit(name="T", channels=[qubit_channel]),
            )
        case ResourceState.S:
            gate_sequence = (
                Circuit(name="Reset_+", channels=[qubit_channel]),
                Circuit(name="phase", channels=[qubit_channel]),
            )
        case _:
            raise ValueError(
                f"Resource state {resource_state} is not supported for physical reset."
            )

    return Circuit(
        name="Resource state reset",
        circuit=gate_sequence,
        channels=[qubit_channel],
    )


def find_qubits_quadrant(
    block: RotatedSurfaceCode,
    quadrant: Direction,
) -> tuple[tuple[int, int, int], ...]:
    """
    Finds the qubits in a specific quadrant of a rotated surface code block.

    Parameters
    ----------
    block : RotatedSurfaceCode
        The rotated surface code block containing the stabilizers.
    quadrant : Direction
        The direction of the quadrant to search for qubits.

    Returns
    -------
    tuple[tuple[int, int, int], ...]
        The qubits found in the specified quadrant.

    Raises
    ------
    ValueError
        If there is more than one pauli type in the boundary stabilizers.
    """
    # Order the stabilizers by the x coordinate for LEFT/RIGHT quadrants
    # and by the y coordinate for TOP/BOTTOM quadrants.
    # The reverse order is used for the RIGHT and BOTTOM quadrants.
    ordered_stabilizers = sorted(
        block.stabilizers,
        key=lambda stab: (
            stab.data_qubits[0][0]
            if quadrant in (Direction.LEFT, Direction.RIGHT)
            else stab.data_qubits[0][1]
        ),
        reverse=(quadrant in (Direction.RIGHT, Direction.BOTTOM)),
    )

    boundary_stabs = block.boundary_stabilizers(quadrant)
    pauli_quadrant = set(p for stab in boundary_stabs for p in stab.pauli)
    if len(pauli_quadrant) != 1:
        raise ValueError(
            f"Expected a single Pauli type for quadrant {quadrant},"
            f" found: {pauli_quadrant}"
        )

    # First find the qubits included in the quadrant because of the boundary stabilizers
    quadrant_qubit_set = set(
        q for stab in block.boundary_stabilizers(quadrant) for q in stab.data_qubits
    )
    # Iterate through the ordered stabilizers and populate the quadrant_qubit_set with
    # qubits that are part of the stabilizers with the same Pauli type
    # and two data qubits already added to the quadrant (e.g. the property of
    # stabilizers being deterministic propagates to the closest stabilizers from the
    # boundary).
    for stab in ordered_stabilizers:
        if (
            set(stab.pauli) == set(pauli_quadrant)
            and len(quadrant_qubit_set.intersection(stab.data_qubits)) == 2
        ):
            quadrant_qubit_set |= set(stab.data_qubits)

    return tuple(quadrant_qubit_set)


def reset_into_four_quadrants(
    interpretation_step: InterpretationStep,
    block: RotatedSurfaceCode,
) -> tuple[Circuit, tuple[Stabilizer, ...]]:
    """
    Reset the rest of the data qubits of the block into four quadrants, such that two
    quadrants are in the $|0⟩$ state, making the Z stabilizer measurements deterministic
    and two quadrants are in the $|+⟩$ state, making the X stabilizer measurements
    deterministic.

    E.g. for a distance 5 block, the reset will look like this:

    .. code-block::

                 d           d
        + --- 0 --- 0 --- 0 --- 0
      d |     |     |  d  |     |
        + --- + --- 0 --- 0 --- +
        |  d  |     |     |     | d
        + --- + --- 𝛙 --- + --- +
      d |     |     |     |  d  |
        + --- 0 --- 0 --- + --- +
        |     |  d  |     |     | d
        0 --- 0 --- 0 --- 0 --- +
           d           d

    "d" denotes a deterministic stabilizer measurement in the above figure.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        The interpretation step containing the input block.
    input_block : RotatedSurfaceCode
        The input block to reset.

    Returns
    -------
    tuple[Circuit, tuple[Stabilizer, ...]]
        The circuit resetting all qubits aside from the central qubit and the
        stabilizers that are initialized in a deterministic state because of these
        resets.
    """

    # Find the qubits in each quadrant and associate them with their reset state.
    qubits_to_state_map: dict[
        tuple[tuple[int, int, int], ...], SingleQubitPauliEigenstate
    ] = {}
    for quadrant in Direction:
        pauli_type = set(
            p for stab in block.boundary_stabilizers(quadrant) for p in stab.pauli
        )
        if len(pauli_type) != 1:
            raise ValueError(
                f"Expected a single Pauli type for quadrant {quadrant},"
                f" found: {pauli_type}"
            )
        quadrant_qubits = find_qubits_quadrant(block, quadrant)
        qubits_to_state_map |= {
            qubit: (
                SingleQubitPauliEigenstate.ZERO
                if pauli_type == {"Z"}
                else SingleQubitPauliEigenstate.PLUS
            )
            for qubit in quadrant_qubits
        }

    # Create the reset circuits:
    reset_circuit = Circuit(
        name="reset four quadrants",
        circuit=(
            (
                Circuit(
                    name=f"reset_{state.value}",
                    channels=[
                        interpretation_step.get_channel_MUT(
                            qubit, channel_type=ChannelType.QUANTUM
                        )
                    ],
                )
                for qubit, state in qubits_to_state_map.items()
            ),
        ),
    )

    deterministic_stabs = tuple(
        stab
        for stab in block.stabilizers
        if all(
            qubits_to_state_map.get(qubit, None) == ("+" if pauli == "X" else "0")
            for qubit, pauli in zip(stab.data_qubits, stab.pauli, strict=True)
        )
    )

    return reset_circuit, deterministic_stabs


def find_centered_logical_operators(
    input_block: RotatedSurfaceCode,
    center_qubit: tuple[int, int, int],
) -> tuple[PauliOperator, PauliOperator]:
    """
    Create new logical operators that go through the center of the block such that the
    state injection can be measured directly by the given logical operators.

    Parameters
    ----------
    input_block : RotatedSurfaceCode
        The input block that may or may not have centered logical operators.
    center_qubit : tuple[int, int, int]
        The qubit onto which the state is injected.

    Returns
    -------
    tuple[PauliOperator, PauliOperator]
        The centered logical operators.
    """
    x_log_is_horizontal = input_block.x_boundary == Orientation.HORIZONTAL
    x_op_len, z_op_len = (
        input_block.size if x_log_is_horizontal else input_block.size[::-1]
    )
    centered_logical_operators = (
        PauliOperator(
            pauli="X" * x_op_len,
            data_qubits=tuple(
                (
                    center_qubit[0] + i * x_log_is_horizontal,
                    center_qubit[1] + i * (not x_log_is_horizontal),
                    center_qubit[2],
                )
                for i in range(-(x_op_len // 2), x_op_len // 2 + 1)
            ),
        ),
        PauliOperator(
            pauli="Z" * z_op_len,
            data_qubits=tuple(
                (
                    center_qubit[0] + i * (not x_log_is_horizontal),
                    center_qubit[1] + i * (x_log_is_horizontal),
                    center_qubit[2],
                )
                for i in range(-(z_op_len // 2), z_op_len // 2 + 1)
            ),
        ),
    )

    return centered_logical_operators


def create_deterministic_syndromes(
    interpretation_step: InterpretationStep,
    block: RotatedSurfaceCode,
    deterministic_stabs: tuple[Stabilizer, ...],
) -> tuple[Syndrome, ...]:
    """
    Create syndromes that are deterministic for the given state.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        The interpretation step containing the input block.
    block : RotatedSurfaceCode
        The input block to create syndromes for.
    deterministic_stabs: tuple[Stabilizer, ...]
        The stabilizers that have a deterministic value after the state injection.

    Returns
    -------
    tuple[Syndrome, ...]
        The tuple of deterministic syndromes.
    """

    new_syndromes = generate_syndromes(
        interpretation_step=interpretation_step,
        stabilizers=deterministic_stabs,
        block=block,
        stab_measurements=tuple(() for _ in deterministic_stabs),
    )
    return new_syndromes
