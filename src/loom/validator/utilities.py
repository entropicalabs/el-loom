"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from functools import reduce
from itertools import product
import numpy as np

from ..eka import LogicalState
from ..eka.utilities import SignedPauliOp, is_tableau_valid, StabArray
from ..cliffordsim import Engine


def get_all_cliffordsim_registers_with_random_flags(
    cat_engine: Engine,
) -> dict[str, tuple[int, bool | None]]:
    """
    Get all the classical registers in the given cliffordsim engine.

    Parameters
    ----------
    cat_engine : Engine
        The cliffordsim engine that contains the classical registers.

    Returns
    -------
    dict[str, tuple[int, bool | None]]
        A dictionary where the keys are the classical register names and the values are
        tuples containing the value of the register and a flag indicating whether the
        register is a result of a random measurement (True) or not (False). If the
        register is not a result of a measurement, the flag is set to None.
    """
    # Get all the classical registers in the cat_engine
    # Set their random flags to None
    res_is_random_dict = {
        bit_id: (bit_value, None)
        for reg in cat_engine.registry.values()
        for bit_id, bit_value in reg.id_bit_reg.items()
    }

    # Look up whether the bit comes as a result of a random measurement
    # Flatten the measurement results to a dict of tuples (m_result, is_random)
    cat_mres_dict = cat_engine.data_store.measurements
    measurement_results = {}
    for time_step in cat_mres_dict["time_step"]:
        bit_id = list(cat_mres_dict[str(time_step)].keys())[0]
        meas_result = cat_mres_dict[str(time_step)][bit_id]["measurement_result"]
        meas_is_random = cat_mres_dict[str(time_step)][bit_id]["is_random"]
        measurement_results[bit_id] = (meas_result, meas_is_random)

    # Update the res_is_random_dict with the measurement results and return it
    res_is_random_dict.update(measurement_results)
    return res_is_random_dict


def get_parity_from_cbits(cat_engine: Engine, cbits: tuple[str | int, ...]) -> int:
    """
    Get the parity of the cbits in the given list.

    Parameters
    ----------
    cat_engine : Engine
        The cliffordsim engine that contains the runtime results.
    cbits : tuple[str | int, ...]
        The list of cbits to check. The cbits can be either strings or integers.
        The integers are treated as constant values (0 or 1) that can flip
        the expected parity. The strings are the labels of the classical channels
        whose values are to be retrieved at runtime. The register name is the first part
        of the label, e.g. c_(0_0)_0 -> c.

    Returns
    -------
    int
        The parity of the cbits. The parity is calculated by XORing the values
        of all of the cbits.
    """
    # Get all the classical registers in the cat_engine and their values
    cliffordsim_classical_registers = get_all_cliffordsim_registers_with_random_flags(
        cat_engine
    )

    # Separate the int cbits from the c_reg values
    # cbit_int_values : Constant values (0 or 1) that can flip the expected parity
    # cbit_labels : Labels of the classical channels whose values are to be
    # retrieved at runtime
    cbit_int_values = [cbit for cbit in cbits if isinstance(cbit, int)]
    cbit_labels = [cbit for cbit in cbits if isinstance(cbit, str)]

    cbit_runtime_values = []
    for cbit_label in cbit_labels:
        # Append the value of the cbit to the list
        cbit_runtime_values += [cliffordsim_classical_registers[cbit_label][0]]

    # Evaluate change in parity
    parity = reduce(
        lambda x, y: x ^ y,
        cbit_int_values + cbit_runtime_values,
        0,
    )

    if parity not in (0, 1):
        raise ValueError(
            "The parity of the cbits is not valid. The parity should be either 0 or 1."
        )

    return parity


def logical_states_to_check(n_logical_qubits: int) -> list[LogicalState]:
    """
    Returns the logical states to check for a given number of logical qubits. These
    logical states are constructed in such a way that if these are transformed
    correctly, then all the logical states should be transformed correctly for a
    Clifford gate operation. The selection is done such that the effect of the gate
    on each input logical operator is isolated.

    For example, to verify the effect on the input logical operators Z1, X1
    (logical qubit 1) for a 3-logical-qubit system, the effect should be captured by
    checking the output of the logical states:
    {+Z0, +Z1, +Z2}, {+Z0, +X1, +Z2}, {+X0, +X1, +X2} and {+X0, +Z1, +X2}.
    We can see that the action of the gate on the logical qubit 1 is isolated by keeping
    the logical qubits 0 and 2 in the same state for the first two and the last two
    states while changing the state of the logical qubit 1.

    NOTE: The above has not been mathematically proven, but it is a reasonable
    assumption that hasn't been disproven yet.

    Parameters
    ----------
    n_logical_qubits : int
        The number of logical qubits.

    Returns
    -------
    list[LogicalState]
        The list of logical states to check.
    """

    if n_logical_qubits == 1:
        # Return the |0> and |+> states
        return [LogicalState(["+Z0"]), LogicalState(["+X0"])]
    if n_logical_qubits == 2:
        # Return the |00>, |++>, |0+>, and |+0> states
        return [
            LogicalState(["+Z0", "+Z1"]),
            LogicalState(["+X0", "+X1"]),
            LogicalState(["+Z0", "+X1"]),
            LogicalState(["+X0", "+Z1"]),
        ]
    return (
        # |00...0> state
        [LogicalState([f"+Z{j}" for j in range(n_logical_qubits)])]
        # |++...+> state
        + [LogicalState([f"+X{j}" for j in range(n_logical_qubits)])]
        # |+0...0>, |0+0...0>, ..., |00...0+> states
        + [
            LogicalState(
                [f"+Z{j}" if i != j else f"+X{j}" for j in range(n_logical_qubits)]
            )
            for i in range(n_logical_qubits)
        ]
        # |0+...+>, |+0+...+>, ..., |++...0> states
        + [
            LogicalState(
                [f"+X{j}" if i != j else f"+Z{j}" for j in range(n_logical_qubits)]
            )
            for i in range(n_logical_qubits)
        ]
    )


def all_possible_pauli_strings(n_qubits: int) -> list[str]:
    """
    Returns all possible sparse Pauli strings for a given number of qubits.
    The sparse Pauli strings are generated by iterating over all possible combinations
    of X and Z operators for each qubit. The list size scales exponentially with the
    number of qubits (as 2*4**n_qubits), so it is important to consider the performance
    implications when working with large number of (logical) qubits.

    Parameters
    ----------
    n_qubits : int
        The number of qubits.
    Returns
    -------
    list[str]
        The list of all possible sparse Pauli strings for the given number of qubits.
    """
    if n_qubits < 1 or not isinstance(n_qubits, int):
        raise ValueError("The number of qubits must be at least 1.")

    paulis = ["_", "X", "Z", "Y"]
    signs = ["+", "-"]
    return [
        f"{sign}{''.join(p)}"
        for sign, p in product(signs, product(paulis, repeat=n_qubits))
        if any(c != "_" for c in p)  # skip all-identity
    ]  # Exclude the empty string (all I's)


# pylint:disable=anomalous-backslash-in-string, line-too-long
def logical_state_transformations_to_check(
    x_operators_sparse_pauli_map: list[str], z_operators_sparse_pauli_map: list[str]
) -> list[tuple[LogicalState, tuple[LogicalState]]]:
    """Returns the logical state transformations to check for a logical operation. The
    input is the sparse Pauli strings for the individual X and Z operators. The
    output is a list of tuples where each tuple contains the input and expected output
    logical states for the logical operation.

    Example:
    For a CNOT gate from qubit 0 to qubit 1 we know that:

    - X0 -> X0X1
    - X1 -> X1
    - Z0 -> Z0
    - Z1 -> Z1Z0

    Thus, we can find the logical state transformations for the CNOT gate by calling
    this function with the following inputs:

    - x_operators_sparse_pauli_map = ["X0X1", "X1"]
    - z_operators_sparse_pauli_map = ["Z0", "Z1Z0"]

    This will return the list of logical state transformations to check for the CNOT
    gate which will be a list containing the following tuples:

    - (LogicalState(('+Z0', '+Z1')), (LogicalState(('+Z0', '+Z0Z1')),)) :math:`\ket{00} -> \ket{00}`
    - (LogicalState(('+X0', '+X1')), (LogicalState(('+X0X1', '+X1')),)) :math:`\ket{++}-> \ket{++}`
    - (LogicalState(('+Z0', '+X1')), (LogicalState(('+Z0', '+X1')),)) :math:`\ket{0+} -> \ket{0+}`
    - (LogicalState(('+X0', '+Z1')), (LogicalState(('+X0X1', '+Z0Z1')),)) :math:`\ket{+0} ->` Bell pair

    The format of the tuples is such that they can be used as input for validator logical
    state transformation checks.

    Parameters
    ----------
    x_operators_sparse_pauli_map : list[str]
        The list of sparse Pauli strings describing how each X operator is transformed.
        No sign is needed and the order matches the transformation of the logical
        operators X0, X1, ...
    z_operators_sparse_pauli_map : list[str]
        The list of sparse Pauli strings describing how each Z operator is transformed.
        No sign is needed and the order matches the transformation of the logical
        operators Z0, Z1, ...

    Returns
    -------
    list[tuple[LogicalState, tuple[LogicalState]]]
        The list of logical state transformations to check.
    """

    if len(z_operators_sparse_pauli_map) != len(x_operators_sparse_pauli_map):
        raise ValueError(
            "The number of X and Z operators should be the same for the logical state "
            "transformations."
        )

    # Get the number of logical qubits
    n_logical_qubits = len(x_operators_sparse_pauli_map)

    # Convert the sparse Pauli strings to SignedPauliOp objects and store them in an
    # np.array for easy access
    x_signed_pauli_op_map = [
        SignedPauliOp.from_sparse_string("+" + x, n_logical_qubits)
        for x in x_operators_sparse_pauli_map
    ]

    z_signed_pauliop_map = [
        SignedPauliOp.from_sparse_string("+" + z, n_logical_qubits)
        for z in z_operators_sparse_pauli_map
    ]

    # For the LogicalStates to be valid, the transformed operators should generate
    # a valid tableau when stacked, similarly to how the X_i and Z_i operators generate
    # a valid tableau when stacked.
    tableau_to_check = np.vstack(
        [x.array for x in x_signed_pauli_op_map]
        + [z.array for z in z_signed_pauliop_map]
    )
    if not is_tableau_valid(tableau_to_check):
        raise ValueError(
            "The transformed X and Z operators do not generate a valid tableau when "
            "stacked. Check the input sparse Pauli string maps."
        )

    x_transform_stabarray = StabArray.from_signed_pauli_ops(x_signed_pauli_op_map)
    z_transform_stabarray = StabArray.from_signed_pauli_ops(z_signed_pauliop_map)

    # Get the input logical states to check
    input_states = logical_states_to_check(n_logical_qubits)

    def _multiply_signed_pauli_ops(
        signed_pauli_ops: list[SignedPauliOp],
        n_logical_qubits: int = n_logical_qubits,
        negative: bool = False,
    ) -> SignedPauliOp:
        """Multiply all of the Pauli operators together and return the result as a
        SignedPauliOp object. Need to specify the number of logical qubits and whether
        the starting sign is negative.
        """
        return reduce(
            lambda x, y: x * y,
            signed_pauli_ops,
            SignedPauliOp.identity(n_logical_qubits, negative=negative),
        )

    # Get the output logical states to check
    output_states = [
        LogicalState(
            # The logical state needs to be defined as a list of sparse Pauli
            # strings which will be found by taking the product of the transformed
            # X and Z operators for each input operator
            [
                # Take the product of the transformed X and Z operators that
                # are present in the input state
                # For example, for the CNOT gate and the input state |00>:
                # we have the input operators +Z0, +Z1
                # which are transformed to +Z0, +Z0Z1 respectively
                _multiply_signed_pauli_ops(
                    # Include all transformed X, Z operators that are present
                    # in the input state (.x==1 or .z==1)
                    x_transform_stabarray[input_pauli_op.x == 1]
                    + z_transform_stabarray[input_pauli_op.z == 1],
                    negative=input_pauli_op.sign == 1,
                ).as_sparse_string()
                # cast to sparse string such that it can be used to create a
                # LogicalState object
                for input_pauli_op in input_state.stabarray
            ]
        )
        for input_state in input_states
    ]

    # Because the output states need to be tuples of LogicalState objects, we need to
    # convert the output states to a tuple of LogicalState objects
    output_states = [(output_state,) for output_state in output_states]

    # Return the input and output logical states as a list of tuples for validator checks
    return list(zip(input_states, output_states, strict=True))
