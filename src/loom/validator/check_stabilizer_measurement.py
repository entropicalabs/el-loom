"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from functools import reduce
from pydantic.dataclasses import dataclass
import numpy as np

from ..eka import Stabilizer, Block
from ..eka.utilities import SignedPauliOp
from ..cliffordsim import Engine
from ..cliffordsim.operations import UpdateTableau, Operation


from .check_abstract import AbstractValidityCheck
from .utilities import get_all_cliffordsim_registers_with_random_flags


@dataclass(frozen=True)
class StabilizerMeasurementCheckOutput:
    """
    Dataclass to store the output of the Stabilizer Measurement check.

    Parameters
    ----------
    expected_vs_measured_stabs: tuple[tuple[str,Stabilizer,tuple[Stabilizer,...]],...]
        A tuple containing the stabilizers that were not measured correctly. Each tuple
        contains the measurement index, the stabilizer that was supposed to be measured
        by it, and the stabilizers that were measured instead.
    probabilistic_measurements: tuple[str, ...]
        A tuple containing all the measurements that are probabilistic
    """

    expected_vs_measured_stabs: tuple[
        tuple[str, Stabilizer, tuple[Stabilizer, ...]], ...
    ]
    probabilistic_measurements: tuple[str, ...]

    def __len__(self):
        return len(self.expected_vs_measured_stabs) + len(
            self.probabilistic_measurements
        )

    def __str__(self):
        out = ""
        if any(self.expected_vs_measured_stabs):
            out = "- Expected vs Measured Stabilizers:\n"
            for idx, stab, measured_stabs in self.expected_vs_measured_stabs:
                measured_stabs_str = (
                    ", ".join(str(stab) for stab in measured_stabs)
                    if measured_stabs
                    else "None"
                )
                out += (
                    f"Measurement {idx}: Expected {stab}, "
                    f"Measured: {measured_stabs_str}\n"
                )
        if any(self.probabilistic_measurements):
            out += "- Probabilistic Measurements: "
            out += ", ".join(str(idx) for idx in self.probabilistic_measurements)
            out += "\n"
        return out.rstrip("\n")  # Remove the last newline character for cleaner output


@dataclass(frozen=True)
class StabilizerMeasurementCheck(AbstractValidityCheck):
    """Dataclass to store the results of the Stabilizer Measurement check.

    Parameters
    ----------
    output: StabilizerMeasurementCheckOutput
        An object containing the output of the check.

    Properties
    -----------
    message: str
        A message indicating the result of the check. It will be empty if the check is
        valid, otherwise it will contain a message describing the issue.
    valid: bool
        True if the check is valid (i.e., all measurements were deterministic and
        measured the assigned stabilizer), False otherwise.
    """

    output: StabilizerMeasurementCheckOutput

    @property
    def message(self) -> str:
        match (
            len(self.output.expected_vs_measured_stabs),
            len(self.output.probabilistic_measurements),
        ):
            case (0, 0):
                return ""
            case (0, _):
                return "Some measurement(s) were not deterministic."
            case (_, 0):
                return "Some measurement(s) did not measure the assigned stabilizer."
            case (_, _):
                return (
                    "Some measurement(s) were not deterministic and some did not "
                    "measure the assigned stabilizer."
                )


# pylint: disable=too-many-locals, too-many-branches, line-too-long, anomalous-backslash-in-string
def check_input_stabilizer_measurement(
    base_cliffordsim_operations: tuple[Operation, ...],
    input_block: Block,
    measurement_to_stabilizer_map: dict[str, Stabilizer],
    seed: int | None,
) -> StabilizerMeasurementCheck:
    """Checks whether the measurement of the stabilizers happened and whether the
    result was registered at the correct place. This is done by running the circuit with
    an initial state of :math:`\ket{00...0}` and recording the measurement results.
    Then, for each stabilizer, the stabilizer is flipped and the circuit is run again.
    The measurement results are compared to the reference results to check if the
    measurement was correct. The stabilizers that were not measured correctly are stored
    in the output.

    Parameters
    ----------
    base_cliffordsim_operations: tuple[:class:`loom.cliffordsim.operations.base_operation.Operation`, ...]
        The cliffordsim operations that represent the circuit to be checked.
    input_block: Block
        The Block object that represents the input code.
    measurement_to_input_stabilizer_map: dict[str, Stabilizer]
        Dictionary matching the classical channel name of a measurement operation with a
        stabilizer in the input code.
    seed: int | None
        The seed for the cliffordsim engine.

    Returns
    -------
    StabilizerMeasurementCheck
        The result of the Stabilizer Measurement check.
    """

    # Get the measurement results from the reference run
    mresults_reference = get_mresults_with_flipped_reduced_stabilizers(
        [], base_cliffordsim_operations, input_block, seed
    )
    # Check if the measurement indices given by the user are valid
    for chan_label in measurement_to_stabilizer_map.keys():
        if chan_label not in mresults_reference:
            raise ValueError(
                f"Measurement channel label {chan_label} not found in cliffordsim data store."
            )

    # Get the deterministic and probabilistic measurements
    probabilistic_measurements: set[str] = set()
    measurement_reference: dict[str, int] = {}
    for chan_label, (meas_result, meas_is_random) in mresults_reference.items():
        if chan_label in measurement_to_stabilizer_map:
            if meas_is_random:
                probabilistic_measurements.add(chan_label)
            else:
                measurement_reference[chan_label] = meas_result

    # For every measurement find the stabilizers that contributed to the measurement
    contributions_present: dict[str, set[Stabilizer]] = {
        i: set() for i in measurement_reference
    }
    # Check what happens by flipping one by one the stabilizers
    for stab_flipped_idx, stab_flipped in enumerate(input_block.stabilizers):
        # Use bookkeeping to find which reduced stabilizers to flip.
        # The rows that need to be flipped are the ones that have the value True in the
        # column stab_flipped_idx
        reduced_stab_idx_to_flip = input_block.reduced_bookkeeping[:, stab_flipped_idx]

        if not np.any(reduced_stab_idx_to_flip):
            # If the stabilizer does not appear in the reduced stabilizer array, skip
            # as it will be the same as running the reference we did previously
            # The stabilizer is not present in the reduced stabilizer array because in
            # overdefined sets of stabilizers, not all stabilizers are needed to
            # create the reduced stabilizer array.
            continue

        # Get the measurement results with the flipped stabilizers
        mresults_flipped = get_mresults_with_flipped_reduced_stabilizers(
            reduced_stab_idx_to_flip, base_cliffordsim_operations, input_block, seed
        )

        # Go through the measurement results to check for consistency and the stabilizer
        # contributions
        for chan_label, (meas_result, meas_is_random) in mresults_flipped.items():

            if chan_label not in measurement_to_stabilizer_map:
                # Skip if the measurement is not in the map
                continue

            # Check for any inconsistencies in the measurements being deterministic
            measurement_should_be_random = chan_label in probabilistic_measurements
            if measurement_should_be_random != meas_is_random:
                raise RuntimeError(
                    f"Measurement with index {chan_label} is sometimes random and "
                    "sometimes deterministic."
                )

            if meas_is_random:
                # Skip if the measurement is random
                continue

            # Check that the measurement result matches the sign of the stabilizer
            # in the initialized array
            if meas_result != measurement_reference[chan_label]:
                contributions_present[chan_label].add(stab_flipped)

    n_dqubits = input_block.n_data_qubits
    # Gather the stabilizers that were not measured correctly
    expected_vs_measured_stabs = tuple(
        (idx, stab, contributions_present[idx])
        for idx, stab in measurement_to_stabilizer_map.items()
        # Do not include the deterministic measurements
        if idx not in probabilistic_measurements
        # Do not include the measurements that indeed measured the stabilizer or the
        # product of the stabilizers that it measured is equivalent to the correct
        # stabilizer
        and not reduce(
            lambda x, y: x * y,
            (
                stab.as_signed_pauli_op(input_block.data_qubits)
                for stab in contributions_present[idx]
            ),
            SignedPauliOp.identity(n_dqubits),
        )  # The product of the measured stabilizers
        == stab.as_signed_pauli_op(input_block.data_qubits)  # The correct stabilizer
    )

    # Sort based on the first element of the tuple
    expected_vs_measured_stabs = tuple(
        sorted(expected_vs_measured_stabs, key=lambda x: x[0])
    )

    # Initialize the output dataclass
    stab_meas_check_output = StabilizerMeasurementCheckOutput(
        expected_vs_measured_stabs, sorted(tuple(probabilistic_measurements))
    )

    return StabilizerMeasurementCheck(stab_meas_check_output)


# pylint: disable=anomalous-backslash-in-string
def get_mresults_with_flipped_reduced_stabilizers(
    reduced_stab_idx_to_flip: np.ndarray | list[int],
    base_cliffordsim_operations: tuple[Operation, ...],
    input_block: Block,
    seed: int | None,
) -> dict[str, tuple[int, bool | None]]:
    """
    This function runs the circuit with some stabilizers flipped and returns the
    measurement results. The logical state that is used is the :math:`\ket{00...0}` state.

    Parameters
    ----------
    reduced_stab_idx_to_flip : np.ndarray | list[int]
        The indices of the reduced stabilizers to flip.
    base_cliffordsim_operations : tuple[ \
        :class:`loom.cliffordsim.operations.base_operation.Operation`, ...]
        The base cliffordsim operations that will be used to run the circuit.
    input_block : Block
        The Block object that represents the input code.
    seed : int | None
        The seed for the cliffordsim engine.

    Returns
    -------
    dict[str, tuple[int, bool | None]]
        A dictionary containing the names of the classical registers and their values,
        along with a boolean indicating whether the value is a result of a random
        measurement. If the value is not a result of a measurement, the boolean is None.
    """
    stab_flipped_base_array = input_block.reduced_stabarray.array.copy()
    # Flip the signs of the stabilizers that need to be flipped
    stab_flipped_base_array[reduced_stab_idx_to_flip, -1] ^= 1
    # Construct tableau corresponding to the array with flipped stabilizer
    # while using logical state |00...0>
    stab_flipped_tableau = np.vstack(
        (
            input_block.x_log_stabarray.array,
            input_block.destabarray.array,
            input_block.z_log_stabarray.array,
            stab_flipped_base_array,
        )
    )

    # Run the circuit from the initial state corresponding to the tableau
    # with the flipped stabilizers
    operations = (UpdateTableau(stab_flipped_tableau), *base_cliffordsim_operations)
    cliffordsim_engine = Engine(operations, input_block.n_data_qubits, seed=seed)
    cliffordsim_engine.run()

    return get_all_cliffordsim_registers_with_random_flags(cliffordsim_engine)
