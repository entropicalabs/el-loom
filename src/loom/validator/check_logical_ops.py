"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from pydantic.dataclasses import dataclass

from ..eka import Block, LogicalState
from ..eka.utilities import (
    is_subset_of_stabarray,
    StabArray,
    SignedPauliOp,
    reduce_stabarray,
)
from ..cliffordsim import Engine
from ..cliffordsim.operations import UpdateTableau, Operation

from .check_abstract import AbstractValidityCheck
from .utilities import get_parity_from_cbits, all_possible_pauli_strings


@dataclass(frozen=True)
class LogicalOperatorCheckOutput:
    """
    Dataclass to store the output of the Logical Operator check.

    Parameters
    ----------
    input_vs_expected_vs_actual_logicals_multi: tuple[
        tuple[LogicalState, tuple[LogicalState, ...], LogicalState | None], ...
    ]
        A tuple containing the input logical states, the expected output logical states,
        and the actual output logical state. If the actual output state is None, it
        means that no valid logical state was found in the output.
    input_vs_expected_vs_actual_logicals_with_parity: tuple[
        tuple[LogicalState, tuple[LogicalState, tuple[int, ...]], LogicalState | None],
        ...
    ]
        A tuple containing the input logical states, the expected output logical states
        with parity flips, and the actual output logical state. If the actual output
        state is None, it means that no valid logical state was found in the output.

    """

    input_vs_expected_vs_actual_logicals_multi: tuple[
        tuple[LogicalState, tuple[LogicalState, ...], LogicalState | None], ...
    ]
    input_vs_expected_vs_actual_logicals_with_parity: tuple[
        tuple[LogicalState, tuple[LogicalState, tuple[int, ...]], LogicalState | None],
        ...,
    ]

    def __len__(self) -> int:
        """Returns the number of failed logical state transformations."""
        return len(self.input_vs_expected_vs_actual_logicals_multi) + len(
            self.input_vs_expected_vs_actual_logicals_with_parity
        )

    def __str__(self):
        out = ""
        if self.input_vs_expected_vs_actual_logicals_multi:
            out += "- Failed Logical State Transformations:\n"
            for (
                initial,
                allowed_final,
                actual,
            ) in self.input_vs_expected_vs_actual_logicals_multi:
                out += (
                    f"Initial: {initial}, Allowed Finals: {allowed_final},"
                    f" Actual: {actual}\n"
                )
        if self.input_vs_expected_vs_actual_logicals_with_parity:
            out += "- Expected vs Actual Logical Transformations:\n"
            for (
                initial,
                (expected, flips),
                actual,
            ) in self.input_vs_expected_vs_actual_logicals_with_parity:
                out += (
                    f"Initial: {initial}, Expected: {expected}, Flips: {flips},"
                    f" Actual: {actual}\n"
                )
        return out.rstrip("\n")  # Remove trailing newline for cleaner output


@dataclass(frozen=True)
class LogicalOperatorCheck(AbstractValidityCheck):
    """Dataclass to store the results of the Logical Operator check.

    Parameters
    ----------
    output: LogicalOperatorCheckOutput
        An object containing the output of the check, including any failed logical state
        transformations and the expected vs actual logical transformations with parity
        flips for each logical state.

    Properties
    ----------
    message: str
        A message indicating the result of the check. It will be empty if the check is
        valid, otherwise it will contain a message describing the issue.
    valid: bool
        True if the check is valid (i.e. all logical states were transformed
        correctly), False otherwise.
    """

    # Define allowed messages
    output: LogicalOperatorCheckOutput

    @property
    def message(self) -> str:
        if not self.valid:
            return "Some logical states were not transformed correctly."
        return ""


# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
def check_logical_operators_transformation(
    base_cliffordsim_operations: tuple[Operation, ...],
    input_block: Block,
    output_block: Block,
    logical_state_transformations_with_parity: dict[
        LogicalState,
        tuple[LogicalState, dict[int, tuple[str | int, ...]]],
    ],
    logical_state_transformations: list[tuple[LogicalState, tuple[LogicalState, ...]]],
    seed: int | None,
) -> LogicalOperatorCheck:
    """Checks whether the logical operators are transformed correctly.

    Parameters
    ----------
    base_cliffordsim_operations : tuple[ \
        :class:`loom.cliffordsim.operations.base_operation.Operation`, ...]
        A tuple of base cliffordsim operations that will be used to run the circuit.
    input_block : Block
        The Block object that represents the input code.
    output_block : Block
        The Block object that represents the output code.
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
    logical_state_transformations : list[tuple[LogicalState, tuple[LogicalState, ...]]]
        A list of tuples where each tuple contains an input logical state and a tuple
        of expected output logical states. When the circuit is run for a given input
        logical state, at least one of the expected output logical states should be
        found in the output.
    seed : int | None
        The seed for the cliffordsim engine.

    Returns
    -------
    LogicalOperatorCheck
        The result of the Logical Operator check.
    """

    input_vs_expected_vs_actual_logicals_multi: list[
        tuple[LogicalState, tuple[LogicalState, ...], LogicalState | None]
    ] = []

    for input_state, output_states in logical_state_transformations:
        tableau = input_state.get_tableau(input_block)
        # Run the circuit from initial state
        operations = (UpdateTableau(tableau), *base_cliffordsim_operations)
        cliffordsim_engine = Engine(operations, input_block.n_data_qubits, seed=seed)
        cliffordsim_engine.run()

        # Retrieve the output stabilizer array
        out_stabarray = StabArray(cliffordsim_engine.tableau_w_scratch.stabilizer_array)

        # Get all possible valid output logical stabarrays
        expected_output_log_op_stabarray = [
            LogicalState.convert_logical_to_base_representation(
                output_block, output_state.stabarray
            )
            for output_state in output_states
        ]
        # Check that at least one of the expected logical operators is a subset of the
        # output StabArray
        if not any(
            is_subset_of_stabarray(exp_log_op_stabarr, out_stabarray)
            for exp_log_op_stabarr in expected_output_log_op_stabarray
        ):
            actual_output_state = find_logical_state(output_block, out_stabarray)

            input_vs_expected_vs_actual_logicals_multi.append(
                (
                    input_state,
                    output_states,
                    actual_output_state,
                )
            )

    input_vs_expected_vs_actual_logicals_with_parity: list[
        tuple[LogicalState, tuple[LogicalState, tuple[int, ...]], LogicalState | None]
    ] = []
    for input_state, (
        expected_output_state,
        parity_dict,
    ) in logical_state_transformations_with_parity.items():
        tableau = input_state.get_tableau(input_block)
        # Run the circuit from initial state
        operations = (UpdateTableau(tableau), *base_cliffordsim_operations)
        cliffordsim_engine = Engine(operations, input_block.n_data_qubits, seed=seed)
        cliffordsim_engine.run()

        # Retrieve the output stabilizer array
        out_stabarray = StabArray(cliffordsim_engine.tableau_w_scratch.stabilizer_array)

        # Get all possible valid output logical stabarrays
        exp_output_log_op_stabarray_without_parity = (
            LogicalState.convert_logical_to_base_representation(
                output_block, expected_output_state.stabarray
            )
        )

        # Create a list of lists of cbits for each logical operator
        # If the parity for a logical operator is not specified, it defaults to an
        # empty list
        cbits_list = [
            parity_dict.get(i, []) for i in range(output_block.n_logical_qubits)
        ]
        runtime_parity_list = [
            get_parity_from_cbits(cliffordsim_engine, cbits) for cbits in cbits_list
        ]

        # Find the expected output logical operator with the correct parities
        expected_output_log_op_stabarray = StabArray.from_signed_pauli_ops(
            [
                (op if runtime_parity == 0 else op.with_flipped_sign())
                for op, runtime_parity in zip(
                    exp_output_log_op_stabarray_without_parity,
                    runtime_parity_list,
                    strict=True,
                )
            ]
        )

        # Check that at least one of the expected logical operators is a subset of the
        # output StabArray
        if not is_subset_of_stabarray(expected_output_log_op_stabarray, out_stabarray):
            actual_output_state = find_logical_state(output_block, out_stabarray)

            input_vs_expected_vs_actual_logicals_with_parity.append(
                (
                    input_state,
                    (
                        expected_output_state,
                        tuple(runtime_parity_list),
                    ),
                    actual_output_state,
                )
            )

    return LogicalOperatorCheck(
        LogicalOperatorCheckOutput(
            input_vs_expected_vs_actual_logicals_multi=tuple(
                input_vs_expected_vs_actual_logicals_multi
            ),
            input_vs_expected_vs_actual_logicals_with_parity=tuple(
                input_vs_expected_vs_actual_logicals_with_parity
            ),
        )
    )


def find_logical_state(
    block: Block, output_stabarray: StabArray
) -> LogicalState | None:
    """
    Finds the logical state from the output stabilizer array. If the output stabilizer
    array does not contain a full set of logical operators, the logical state is not
    well-defined and None is returned.

    Parameters
    ----------
    block : Block
        The Block object that represents the code.
    output_stabarray : StabArray
        The stabilizer array from the output of the circuit.

    Returns
    -------
    LogicalState | None
        The logical state if it is well-defined, otherwise None.
    """

    log_stabarr_reduced = StabArray.trivial()
    for p_string in all_possible_pauli_strings(block.n_logical_qubits):
        # Convert the logical pauli string to a SignedPauliOp
        p_op = SignedPauliOp.from_string(p_string)

        # Convert the logical Pauli operator to its base representation
        # i.e. the logical Pauli operator is expressed in terms of block data qubits
        base_repr_p_op = LogicalState.convert_logical_pauli_op_to_base_representation(
            p_op, block.x_log_stabarray, block.z_log_stabarray
        )

        # Check if the base representation of the logical Pauli operator is a subset of
        # the output stabilizer array
        if is_subset_of_stabarray(base_repr_p_op, output_stabarray):
            # Append the operator to the reduced stabarray and reduce it again
            log_stabarr_reduced = reduce_stabarray(
                StabArray.from_signed_pauli_ops(list(log_stabarr_reduced) + [p_op])
            )

            # If the reduced logical stabarray has the same number of stabilizers as
            # the logical qubits of the block, we have found the logical state.
            # This means that the logical state is well-defined.
            if log_stabarr_reduced.nstabs == block.n_logical_qubits:
                return LogicalState.from_stabarray(log_stabarr_reduced)

    # If no well-defined logical state was found, return None.
    return None
