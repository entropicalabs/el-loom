"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from pydantic.dataclasses import dataclass

from ..eka import Stabilizer, Block, LogicalState
from ..eka.utilities import is_subset_of_stabarray, StabArray
from ..cliffordsim.engine import Engine
from ..cliffordsim.operations import UpdateTableau, Operation

from .utilities import get_parity_from_cbits
from .check_abstract import AbstractValidityCheck


@dataclass(frozen=True)
class CodeStabilizerCheckOutput:
    """
    Dataclass to store the output of the Code Stabilizer check.

    Parameters
    ----------
    missing_stabilizers: tuple[Stabilizer, ...]
        A tuple containing the code stabilizers that were not found in the output.
    stabilizers_with_incorrect_parity: tuple[Stabilizer, ...]
        A tuple containing the code stabilizers that were found in the output but have
        incorrect parity.
    """

    missing_stabilizers: tuple[Stabilizer, ...]
    stabilizers_with_incorrect_parity: tuple[Stabilizer, ...]

    def __len__(self) -> int:
        """
        Returns the total number of issues found in the check.
        """
        return len(self.missing_stabilizers) + len(
            self.stabilizers_with_incorrect_parity
        )

    def __str__(self) -> str:
        """String representation of the CodeStabilizerCheckOutput."""
        out = ""
        if self.missing_stabilizers:
            out += "- Missing Stabilizers:\n"
            out += "\n".join(str(stab) for stab in self.missing_stabilizers) + "\n"
        if self.stabilizers_with_incorrect_parity:
            out += "- Stabilizers with Incorrect Parity:\n"
            out += (
                "\n".join(str(stab) for stab in self.stabilizers_with_incorrect_parity)
                + "\n"
            )
        return out.rstrip("\n")


@dataclass(frozen=True)
class CodeStabilizerCheck(AbstractValidityCheck):
    """Dataclass to store the results of the Code Stabilizer check.

    Parameters
    ----------
    output: CodeStabilizerCheckOutput
        An object containing the output of the check.

    Properties
    ----------
    message: str
        A message indicating the result of the check. It will be empty if the check is
        valid, otherwise it will contain a message describing the issue.
    valid: bool
        True if the check is valid (i.e., no missing stabilizers and no stabilizers
        with incorrect parity), False otherwise.
    """

    output: CodeStabilizerCheckOutput

    @property
    def message(self) -> str:
        match len(self.output):
            case 0:
                return ""
            case _:
                return "Some code stabilizer(s) were not found in the output."


# pylint: disable=too-many-arguments, too-many-locals, too-many-positional-arguments
def check_code_stabilizers_output(
    base_cliffordsim_operations: tuple[Operation, ...],
    input_block: Block,
    output_block: Block,
    output_stabilizers_parity: dict[Stabilizer, tuple[str | int, ...]],
    output_stabilizers_with_any_value: list[Stabilizer],
    seed: int | None,
) -> CodeStabilizerCheck:
    """Checks whether the correct code stabilizers are found in the output.

    Parameters
    ----------
    base_cliffordsim_operations : tuple[ \
        :class:`loom.cliffordsim.operations.base_operation.Operation`, ...]
        The cliffordsim operations that represent the circuit to be checked.
    input_block : Block
        The Block object that represents the input code.
    output_block : Block
        The Block object that represents the output code.
    output_stabilizers_parity : list[tuple[Stabilizer, tuple[str | int, ...]]]
        A list of stabilizers that are expected to be in the output with a specific
        parity. The parity is represented as a tuple of strings and integers, where the
        strings are the labels of the classical bits where the result is stored at
        runtime, and the integers are the constant parity changes. The final parity is
        calculated by XORing the values of all of these bits.
    output_stabilizers_with_any_value : list[Stabilizer]
        The list of stabilizers that are expected to be in the output, but with
        any value.
    seed : int | None
        The seed for the cliffordsim engine.

    Returns
    -------
    CodeStabilizerCheck
        The result of the Code Stabilizer check.
    """
    all_zeros_state = LogicalState(
        [f"+Z{i}" for i in range(input_block.n_logical_qubits)]
    )

    all_zeros_tableau = all_zeros_state.get_tableau(input_block)
    # Test conditions CodeStabilizersAltered
    # by running from initial logical state |00...0>
    # Any other logical state can be used as well, but |00...0> is the simplest choice.
    operations = (UpdateTableau(all_zeros_tableau), *base_cliffordsim_operations)
    cliffordsim_engine = Engine(operations, input_block.n_data_qubits, seed=seed)
    cliffordsim_engine.run()
    # Retrieve the output stabilizer array
    out_stab_array_np = cliffordsim_engine.tableau_w_scratch.stabilizer_array
    out_stab_array = StabArray(out_stab_array_np)

    # Create dictionary to flip the parity of the stabilizers using the
    # output_stabilizer_updates
    # Initialize it with all parities being 0 for all stabilizers
    stab_parity = {stab: 0 for stab in output_block.stabilizers}
    # Update the parity of the stabilizers using the output_stabilizer_updates
    stab_parity.update(
        {
            stab: get_parity_from_cbits(cliffordsim_engine, cbits)
            for stab, cbits in output_stabilizers_parity.items()
        }
    )

    # Find the stabilizers that are to be found in the output with exact values (+/-)
    output_stabilizers_with_exact_values = [
        stab
        for stab in output_block.stabilizers
        if stab not in output_stabilizers_with_any_value
    ]

    # Find the stabilizers with exact values that are missing in the output
    existing_stabilizers_with_incorrect_parity = []
    missing_stabilizers = []
    for stab in output_stabilizers_with_exact_values:
        # Get the stabilizer as a signed Pauli operator
        # and its flipped version based on the parity
        stab_as_pauli_op = stab.as_signed_pauli_op(output_block.data_qubits)
        stab_as_pauli_op_with_flipped_parity = stab_as_pauli_op.with_flipped_sign()

        # If parity is 1, swap the stabilizer with its flipped version.
        if stab_parity[stab] == 1:
            stab_as_pauli_op, stab_as_pauli_op_with_flipped_parity = (
                stab_as_pauli_op_with_flipped_parity,
                stab_as_pauli_op,
            )

        if not is_subset_of_stabarray(stab_as_pauli_op, out_stab_array):
            # The stabilizer is not in the output
            if is_subset_of_stabarray(
                stab_as_pauli_op_with_flipped_parity, out_stab_array
            ):
                # If the stabilizer is in the output, but with incorrect parity,
                # add it to the list of stabilizers with incorrect parity
                existing_stabilizers_with_incorrect_parity += [stab]
            else:
                # If neither the stabilizer nor its flipped version is in the output,
                # it is missing
                missing_stabilizers += [stab]

    # Find the stabilizers with any (+/-) values that are missing in the output
    missing_stabilizers += [
        stab
        for stab in output_stabilizers_with_any_value
        if not is_subset_of_stabarray(
            stab.as_signed_pauli_op(output_block.data_qubits),
            out_stab_array,
        )
        and not is_subset_of_stabarray(
            stab.as_signed_pauli_op(output_block.data_qubits).with_flipped_sign(),
            out_stab_array,
        )
    ]

    return CodeStabilizerCheck(
        output=CodeStabilizerCheckOutput(
            missing_stabilizers=missing_stabilizers,
            stabilizers_with_incorrect_parity=existing_stabilizers_with_incorrect_parity,
        )
    )
