"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from ..eka import Circuit, Channel, Block, Stabilizer, LogicalState
from ..executor import convert_circuit_to_cliffordsim
from ..cliffordsim.operations import DeleteQubit, AddQubit, SWAP, Operation

from .debug_dataclass import DebugData, AllChecks

from .check_stabilizer_measurement import check_input_stabilizer_measurement
from .check_logical_ops import check_logical_operators_transformation
from .check_code_stabilizers import check_code_stabilizers_output


# pylint: disable=too-many-arguments, too-many-positional-arguments
def is_circuit_valid(
    circuit: Circuit,
    input_block: Block | tuple[Block, ...],
    output_block: Block | tuple[Block, ...],
    output_stabilizers_parity: dict[Stabilizer, tuple[str | int, ...]],
    output_stabilizers_with_any_value: list[Stabilizer],
    logical_state_transformations_with_parity: dict[
        LogicalState,
        tuple[LogicalState, dict[int, tuple[str | int, ...]]],
    ],
    logical_state_transformations: list[tuple[LogicalState, tuple[LogicalState, ...]]],
    measurement_to_input_stabilizer_map: dict[str, Stabilizer],
    seed: int | None = None,
) -> DebugData:
    """
    Tests if a QEC circuit is valid, ie if it meets the criteria:
    
    - The input code is transformed into the output code
    
        - If any generators are allowed to have any value, positive or negative, the \
        ``output_stabilizers_with_any_value`` list should contain them
        - If the generators are not allowed to have any value, but their parity is \
        specified, the ``output_stabilizers_parity`` list should contain the classical \
        channels and integer constant flips(0 or 1) that are applied to them.

    - The ``logical_state_transformations_with_parity`` dictionary should contain:
    
        - The input logical state as the key
        - A tuple as the value where:
        
            - The first element is the output logical state
            - The second is a dictionary where:
            
                - The keys are the logical operator indices
                - The values are lists of strings or integers (0 or 1) that represent \
                the classical channels and parity flips applied to the logical operators.
                
    - The input code stabilizers are measured correctly

    Parameters
    ----------
    circuit : Circuit
        The QEC circuit.
    input_block : Block | tuple[Block, ...]
        The input Block object(s) corresponding to the input code.
    output_block : Block | tuple[Block, ...]
        The output Block object(s) corresponding to the output code.
    output_stabilizers_parity : dict[Stabilizer, tuple[str | int, ...]]
        Dictionary where the keys are output stabilizers and the value is the expected
        parity. The parity is represented as a tuple of strings and integers (0 or 1),
        where the strings are the labels of the classical bits where a result is stored
        at runtime, and the integers are the constant parity changes. The final parity
        is calculated by XORing the values of all of these bits.
    output_stabilizers_with_any_value : list[Stabilizer]
        List of output code generators that are allowed to have any value in the end of
        the circuit.
    logical_state_transformations_with_parity : dict[
        LogicalState,
        tuple[LogicalState, dict[int, tuple[str | int, ...]]],
    ]
        Dictionary where the keys are the input logical states and the values are
        tuples containing the output logical state and a dictionary of parity flips
        that correspond to each logical operator. The keys of the dictionary are the
        logical operator indices, and the values are lists of strings or integers (0 or
        1) that represent the classical channels and parity flips applied to the logical
        operators.
    logical_state_transformations :
    list[tuple[LogicalState, tuple[LogicalState, ...]]]
        List of tuples where each tuple contains the transformation of the logical
        operators from the input state to the output state(s) that will be checked.
    measurement_to_input_stabilizer_map : dict[str, Stabilizer]
        Dictionary matching the classical channel name of a measurement operation with a
        stabilizer in the input code.
    seed : int, optional
        The seed for the cliffordsim engine, by default None. None means that the
        cliffordsim engine will not be seeded and will use a random seed.

    Returns
    -------
    DebugData
        The result of the checks.
    """
    # If input or output block is a tuple, convert it to a single Block object
    if not isinstance(input_block, Block):
        input_block = Block.from_blocks(input_block)
    if not isinstance(output_block, Block):
        output_block = Block.from_blocks(output_block)

    ### VALIDATION OF INPUTS ###

    # check that  measurement_to_input_stabilizer_map indeed has Stabilizer as values
    validate_measurement_to_input_stabilizer_map(
        circuit, input_block, measurement_to_input_stabilizer_map
    )
    # check that output_stabilizers_with_any_value indeed has Stabilizer objects that
    # are part of the output block(s)
    validate_output_stabilizers_with_any_value(
        output_block, output_stabilizers_with_any_value
    )
    # Check that logical_state_transformations_with_parity has the correct type
    validate_logical_state_transformations_with_parity(
        circuit,
        input_block,
        output_block,
        logical_state_transformations_with_parity,
    )
    # Check Logical State Transformations input
    validate_logical_state_transformations(
        input_block, output_block, logical_state_transformations
    )
    # Check that output_stabilizers_parity has the correct type
    validate_output_stabilizers_parity(circuit, output_block, output_stabilizers_parity)

    ### CLIFFORDSIM OPERATIONS ###

    # Obtain the cliffordsim operations for the circuit to run validator
    base_cliffordsim_operations = get_validator_cliffordsim_operations(
        circuit, input_block, output_block
    )

    ### RUN THE VALIDATOR CHECKS ###

    # Test if the code stabilizers were correctly transformed.
    code_stab_check = check_code_stabilizers_output(
        base_cliffordsim_operations,
        input_block,
        output_block,
        output_stabilizers_parity,
        output_stabilizers_with_any_value,
        seed,
    )
    # Test if the logical operators were altered
    log_op_check = check_logical_operators_transformation(
        base_cliffordsim_operations,
        input_block,
        output_block,
        logical_state_transformations_with_parity,
        logical_state_transformations,
        seed,
    )
    # Test if the syndrome was correctly measured
    stab_meas_check = check_input_stabilizer_measurement(
        base_cliffordsim_operations,
        input_block,
        measurement_to_input_stabilizer_map,
        seed,
    )

    return DebugData(AllChecks(code_stab_check, log_op_check, stab_meas_check))


### VALIDATION FUNCTIONS ###


def validate_measurement_to_input_stabilizer_map(
    circuit: Circuit,
    input_block: Block,
    measurement_to_input_stabilizer_map: dict[str, Stabilizer],
) -> None:
    """
    Validate the measurement_to_input_stabilizer_map input and raise an appropriate
    error if invalid.

    Parameters
    ----------
    circuit : Circuit
        The QEC circuit.
    input_block : Block
        The input Block object corresponding to the input code.
    measurement_to_input_stabilizer_map : dict[str, Stabilizer]
        Dictionary matching the classical channel name of a measurement operation with a
        stabilizer in the input code.
    """

    if not isinstance(measurement_to_input_stabilizer_map, dict):
        raise TypeError(
            "The measurement_to_input_stabilizer_map should be of type dict."
            f"Value {measurement_to_input_stabilizer_map} is of type "
            f"{type(measurement_to_input_stabilizer_map)}."
        )

    classical_channel_labels = [
        chan.label for chan in circuit.channels if chan.is_classical()
    ]

    for cchan_label, stab in measurement_to_input_stabilizer_map.items():
        # check that measurement_to_input_stabilizer_map indeed has Stabilizer as values
        if not isinstance(stab, Stabilizer):
            raise TypeError(
                "The values of measurement_to_input_stabilizer_map should be of type "
                f"Stabilizer. Value {stab} is of type {type(stab)}."
            )
        if stab not in input_block.stabilizers:
            raise ValueError(f"Stabilizer {stab} was not found in the input block.)")
        if not isinstance(cchan_label, str):
            raise TypeError(
                "The keys of measurement_to_input_stabilizer_map should be of type "
                f"str. Key {cchan_label} is of type {type(cchan_label)}."
            )
        if cchan_label not in classical_channel_labels:
            raise ValueError(
                f"Classical channel label {cchan_label} was not found in the circuit's "
                "classical channels."
            )


def validate_output_stabilizers_with_any_value(
    output_block: Block,
    output_stabilizers_with_any_value: list[Stabilizer],
) -> None:
    """
    Validate the output_stabilizers_with_any_value input and raise an appropriate
    error if invalid.

    Parameters
    ----------
    output_block : Block
        The output Block object corresponding to the output code.
    output_stabilizers_with_any_value : list[Stabilizer]
        List of output code generators that are allowed to have any value in the end of
        the circuit.
    """
    if any(
        stab not in output_block.stabilizers
        for stab in output_stabilizers_with_any_value
    ):
        raise ValueError(
            "The stabilizers in output_stabilizers_with_any_value should be in the "
            "stabilizers of the output block(s)."
        )


def validate_output_stabilizers_parity(
    circuit: Circuit,
    output_block: Block,
    output_stabilizers_parity: dict[Stabilizer, tuple[str | int, ...]],
) -> None:
    """Validate the output_stabilizers_parity input and raise an appropriate error if
    invalid.

    Parameters
    ----------
    circuit : Circuit
        The QEC circuit.
    output_block : Block
        The output Block object corresponding to the output code.
    output_stabilizers_parity : dict[Stabilizer, tuple[str | int, ...]]
        Dictionary where the keys are output stabilizers and the value is the expected
        parity. The parity is represented as a tuple of strings and integers (0 or 1),
        where the strings are the labels of the classical bits where a result is stored
        at runtime, and the integers are the constant parity changes. The final parity
        is calculated by XORing the values of all of these bits.
    """
    if not isinstance(output_stabilizers_parity, dict):
        raise TypeError("The output_stabilizers_parity should be a dictionary.")

    for key_stab, value_parity in output_stabilizers_parity.items():
        if not isinstance(key_stab, Stabilizer):
            raise TypeError(
                "The keys of output_stabilizers_parity should be of type Stabilizer."
                f" Key {key_stab} is of type {type(key_stab)}."
            )

        if not key_stab in output_block.stabilizers:
            raise ValueError(
                "Every stabilizer in output_stabilizers_parity should be in the "
                f"stabilizers of the output block. {key_stab} was not found in the"
                f" output block."
            )
        if not isinstance(value_parity, (tuple, list)):
            raise TypeError(
                "The values of the output_stabilizers_parity should be a tuple or "
                f"a list. Value {value_parity} is of type {type(value_parity)}."
            )

        for cbit in value_parity:
            if not isinstance(cbit, (str, int)):
                raise TypeError(
                    "Every element of the output_stabilizers_parity value should be "
                    "either a string (classical channel label) or an integer (0 or 1). "
                    f"Element {cbit} is of type {type(cbit)}."
                )
            if isinstance(cbit, str):
                is_channel_in_output = any(
                    chan.label == cbit and chan.is_classical()
                    for chan in circuit.channels
                )
                if not is_channel_in_output:
                    raise ValueError(
                        f"Channel {cbit} which is referenced in the "
                        f"output_stabilizers_parity is not a classical channel in the "
                        f"circuit."
                    )
            if isinstance(cbit, int):
                if cbit not in (0, 1):
                    raise ValueError(
                        f"Only 0 and 1 are allowed as parity changes. {cbit} was found."
                    )


# pylint: disable=too-many-branches
def validate_logical_state_transformations_with_parity(
    circuit: Circuit,
    input_block: Block,
    output_block: Block,
    logical_state_transformations_with_parity: dict[
        LogicalState,
        tuple[LogicalState, dict[int, tuple[str | int, ...] | list[str | int]]],
    ],
) -> None:
    """
    Validate the logical_state_transformations_with_parity input and raise an
    appropriate error if invalid.

    Parameters
    ----------
    circuit : Circuit
        The QEC circuit.
    input_block : Block
        The input Block object corresponding to the input code.
    output_block : Block
        The output Block object corresponding to the output code.
    logical_state_transformations_with_parity : dict[
        LogicalState,
        tuple[LogicalState, dict[int, tuple[str | int, ...] | list[str | int]]],
    ]
        Dictionary where the keys are the input logical states and the values are
        tuples containing the output logical state and a dictionary of parity flips
        that correspond to each logical operator. The keys of the dictionary are the
        logical operator indices, and the values are lists of strings or integers (0 or
        1) that represent the classical channels and parity flips applied to the logical
        operators.
    """

    for logical_state_in, (
        logical_state_out,
        parity_flips,
    ) in logical_state_transformations_with_parity.items():

        ## Check Key
        if not isinstance(logical_state_in, LogicalState):
            raise TypeError(
                "The input of logical_state_transformations_with_parity should be of "
                f"type LogicalState. Element {logical_state_in} is of type "
                f"{type(logical_state_in)}."
            )
        if not isinstance(logical_state_out, LogicalState):
            raise TypeError(
                "The output of logical_state_transformations_with_parity should be of "
                f"type LogicalState. Element {logical_state_out} is of type "
                f"{type(logical_state_out)}."
            )

        # Check dimensional compatibility
        if logical_state_in.n_logical_qubits != input_block.n_logical_qubits:
            raise ValueError(
                f"Input code has {input_block.n_logical_qubits} logical qubits, "
                f"but a logical state has {logical_state_in.n_logical_qubits}."
            )
        if logical_state_out.n_logical_qubits != output_block.n_logical_qubits:
            raise ValueError(
                f"Output code has {output_block.n_logical_qubits} logical qubits, "
                f"but a logical state has {logical_state_out.n_logical_qubits}."
            )

        ## Check Value
        if not isinstance(parity_flips, dict):
            raise TypeError(
                "The second element of each tuple in "
                "logical_state_transformations_with_parity should be a dictionary."
            )
        if not all(isinstance(key, int) for key in parity_flips.keys()):
            raise TypeError(
                "The keys of the second element of each tuple in "
                "logical_state_transformations_with_parity should be of type int."
            )

        for logical_idx, cbits_list in parity_flips.items():
            if logical_idx < 0 or logical_idx >= logical_state_out.n_logical_qubits:
                raise ValueError(
                    "The keys of the second element of each tuple in "
                    "logical_state_transformations_with_parity should be in the range "
                    f"of 0 to {logical_state_out.n_logical_qubits - 1}."
                    f" Found {logical_idx}."
                )
            if not isinstance(cbits_list, (tuple, list)):
                raise TypeError(
                    "The second element of each tuple in "
                    "logical_state_transformations_with_parity should be a list or "
                    f"tuple. Element {logical_idx} is of type {type(cbits_list)}."
                )
            for cbit in cbits_list:
                if not isinstance(cbit, (str, int)):
                    raise TypeError(
                        "The value of the second element of each tuple in "
                        "logical_state_transformations_with_parity should be an iterable"
                        f" of strings or ints. Found {cbit} of type {type(cbit)}."
                    )
                if isinstance(cbit, str):
                    is_channel_in_output = any(
                        chan.label == cbit and chan.is_classical()
                        for chan in circuit.channels
                    )
                    if not is_channel_in_output:
                        raise ValueError(
                            f"Channel {cbit} which is referenced in the "
                            "logical_state_transformations_with_parity is not a "
                            "classical channel in the output block."
                        )
                if isinstance(cbit, int):
                    if cbit not in (0, 1):
                        raise ValueError(
                            f"Only 0 and 1 are allowed as parity changes in"
                            f"logical_state_transformations_with_parity. {cbit} "
                            "was found."
                        )


def validate_logical_state_transformations(
    input_block: Block,
    output_block: Block,
    logical_state_transformations: list[tuple[LogicalState, tuple[LogicalState, ...]]],
) -> None:
    """
    Validate the logical_state_transformations input and raise an appropriate error
    if invalid.

    Parameters
    ----------
    input_block : Block
        The input Block object corresponding to the input code.
    output_block : Block
        The output Block object corresponding to the output code.
    logical_state_transformations : list[tuple[LogicalState, tuple[LogicalState, ...]]]
        List of tuples where each tuple contains the transformation of the logical
        operators from the input state to the output state(s) that will be checked.
    """
    for logical_state_in, logical_states_out in logical_state_transformations:
        # Check Type
        if not isinstance(logical_state_in, LogicalState):
            raise TypeError(
                "The input of logical_state_transformations should be of type "
                "LogicalState."
            )

        if not all(
            isinstance(logical_state, LogicalState)
            for logical_state in logical_states_out
        ):
            raise TypeError(
                "The output of logical_state_transformations should be a tuple of "
                "LogicalState objects."
            )

        # Check dimensional compatibility
        if logical_state_in.n_logical_qubits != input_block.n_logical_qubits:
            raise ValueError(
                f"Input code has {input_block.n_logical_qubits} logical qubits, "
                f"but a logical state has {logical_state_in.n_logical_qubits}."
            )
        for logical_state in logical_states_out:
            if logical_state.n_logical_qubits != output_block.n_logical_qubits:
                raise ValueError(
                    f"Output code has {output_block.n_logical_qubits} logical "
                    f"qubits, but a logical state has {logical_state.n_logical_qubits}."
                )


### CLIFFORDSIM INSTRUCTIONS UTILITY FUNCTIONS ###


def find_final_swaps(
    input_all_qubit_to_channel_map: dict[int, Channel],
    output_data_qubit_to_channel_map: dict[int, Channel],
) -> list[SWAP]:
    """Find the SWAP operations to bring the final qubits into the correct position.

    Parameters
    ----------
    input_all_qubit_to_channel_map : dict[int, Channel]
        A dictionary matching all qubit indices with their Channel at the beginning
        of the circuit
    output_data_qubit_to_channel_map : dict[int, Channel]
        A dictionary matching data qubit indices with their Channel at the end of
        the circuit.

    Returns
    -------
    list[SWAP]
        A list of SWAP operations to bring the final qubits into the correct position.
    """
    # find channels that don't have the correct data qubit already
    channels_with_wrong_dq = {
        chan: idx
        for idx, chan in output_data_qubit_to_channel_map.items()
        if input_all_qubit_to_channel_map[idx] != chan
    }

    # initialize the map with the data qubits
    channel_to_idx_map = {
        chan: idx for idx, chan in input_all_qubit_to_channel_map.items()
    }

    # make an inverse map for channel to index
    idx_to_chan_map = {idx: chan for chan, idx in channel_to_idx_map.items()}

    # SWAP the final qubits into the correct position
    swap_operations = []
    while channels_with_wrong_dq:
        # pop one by one the items from the dictionary
        chan, dqubit_idx = channels_with_wrong_dq.popitem()

        chan_qubit_idx = channel_to_idx_map[chan]
        dqubit_chan = idx_to_chan_map[dqubit_idx]

        if chan_qubit_idx != dqubit_idx:
            # if the qubits are not in the correct position, swap them
            swap_operations.append(SWAP(dqubit_idx, chan_qubit_idx))

            # update maps accordingly (both ways)
            channel_to_idx_map[chan] = dqubit_idx
            channel_to_idx_map[dqubit_chan] = chan_qubit_idx

            idx_to_chan_map[dqubit_idx] = chan
            idx_to_chan_map[chan_qubit_idx] = dqubit_chan

    return swap_operations


# pylint: disable=too-many-locals, anomalous-backslash-in-string
def get_validator_cliffordsim_operations(
    circuit: Circuit,
    input_block: Block,
    output_block: Block,
) -> tuple[Operation, ...]:
    """Generate a list of cliffordsim operations from a circuit and some input and
    output data qubit to channel maps.
    The qubit channels of the circuit that are not in the input map are considered to be
    auxiliary qubits and initialized in state :math:`\ket{0}`.
    The qubit channels of the output are swapped into the correct position as dictated
    by the output map.
    The qubit channels that are not in the output map are deleted at the end of the
    circuit.

    Parameters
    ----------
    circuit : Circuit
        The base circuit to be converted to cliffordsim instructions.
    input_block : Block
        The Block object containing the input stabilizers and logical operators.
    output_block : Block
        The Block object containing the output stabilizers and logical operators.

    Returns
    -------
    tuple[:class:`loom.cliffordsim.operations.base_operation.Operation`, ...]
        A tuple of cliffordsim operations to be executed.
    """

    # Get all qubit channels
    circuit_qubit_channels = [inp for inp in circuit.channels if inp.is_quantum()]

    # Find the data qubit labels
    input_block_data_qubits_str = [str(qubit) for qubit in input_block.data_qubits]
    # Find all qubit channel labels
    circuit_qubit_channels_labels = [chan.label for chan in circuit_qubit_channels]

    # Find which input data qubits are not indexed
    input_qubits_not_indexed = [
        qub
        for qub in input_block_data_qubits_str
        if qub not in circuit_qubit_channels_labels
    ]

    n_qubits = len(circuit_qubit_channels) + len(input_qubits_not_indexed)
    # Count data qubits and auxiliary qubits for input and output
    n_dqubits_in = input_block.n_data_qubits
    n_auxqubits_in = n_qubits - n_dqubits_in
    n_dqubits_out = output_block.n_data_qubits
    n_auxqubits_out = n_qubits - n_dqubits_out

    # Define the input data qubit channel map
    input_data_qubit_to_channel_map = {
        input_block_data_qubits_str.index(chan.label): chan
        for chan in circuit_qubit_channels
        if chan.label in input_block_data_qubits_str
    }

    # Define ancilla qubit channel map
    input_aux_qubit_channels = [
        chan
        for chan in circuit_qubit_channels
        if chan.label not in input_block_data_qubits_str
    ]
    input_aux_qubit_to_channel_map = {
        idx + n_dqubits_in: chan for idx, chan in enumerate(input_aux_qubit_channels)
    }

    # Get all qubit to channel map
    input_all_qubit_to_channel_map = {
        **input_data_qubit_to_channel_map,
        **input_aux_qubit_to_channel_map,
        # Add qubit channels that are not indexed
        **{
            input_block_data_qubits_str.index(qubit): Channel(label=qubit)
            for qubit in input_qubits_not_indexed
        },
    }

    # Define the output data qubit channel map
    output_data_qubits_str = [str(qubit) for qubit in output_block.data_qubits]
    output_data_qubit_to_channel_map = {
        output_data_qubits_str.index(chan.label): chan
        for chan in list(input_all_qubit_to_channel_map.values())
        if chan.label in output_data_qubits_str
    }

    ### CONSTRUCT THE CLIFFORDSIM INSTRUCTION LIST ###

    # Start with the initialization of auxiliary qubits
    cliffordsim_instructions = [AddQubit(n_dqubits_in) for _ in range(n_auxqubits_in)]

    # Append the circuit instructions
    cliffordsim_instructions += convert_circuit_to_cliffordsim(
        circuit, input_all_qubit_to_channel_map
    )

    # Add the SWAP operations to bring the final qubits into the correct position
    cliffordsim_instructions += find_final_swaps(
        input_all_qubit_to_channel_map,
        output_data_qubit_to_channel_map,
    )

    # Append the deletion of auxiliary qubits
    cliffordsim_instructions += [
        DeleteQubit(n_dqubits_out) for _ in range(n_auxqubits_out)
    ]

    return tuple(cliffordsim_instructions)
