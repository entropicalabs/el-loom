"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import numpy as np

from .stab_array import StabArray, reindex_stabarray, stabarray_standard_form


def find_logical_operator_set(stabarr: StabArray) -> tuple[StabArray, StabArray]:
    """Given a stabilizer array of a code, find a set of X and Z logical operators.
    The Z logical operators shall contain only Z operators while the X logical operators
    may contain both X and Z operators. They may not be in their simplified form.

    This algorithm follows the notation and algorithm described in:
    Nielsen, M. A., & Chuang, I. L. (2011). (p.470-471) Quantum Computation and Quantum
    Information: 10th Anniversary Edition. Cambridge University Press,

    How it works (ignoring signs):

        1. The stabilizer array is first put into standard form. The standard form is of
        the form:

            (I, A1, A2 | B, 0, C)
            (0,  0,  0 | D, I, E)

        Note that the stabilizer array has to be reindexed to the standard form.

        2. The Z logical operators are then given by:

            (0, 0, 0 | A2^T, 0, I)

        and the X logical operators are given by:

            (0, E^T, I | C^T , 0, 0)

        3. Reindex the logical operators to the original qubit order.

    Parameters
    ----------
    stabarr : StabArray

    Returns
    -------
    tuple[StabArray, StabArray]
        A tuple containing the X and Z logical operator StabArray objects.
    """
    stabarray_std_form, r, inv_qbit_map = stabarray_standard_form(stabarr)

    n = stabarr.nqubits  # Number of qubits
    k = n - stabarray_std_form.nstabs  # Number of logical qubits

    # Find the auxiliary arrays A2, E and C
    a2_array = stabarray_std_form.array[:r, n - k : n]
    e_array = stabarray_std_form.array[r:, -k - 1 : -1]  # -1 to exclude the sign column
    c_array = stabarray_std_form.array[:r, -k - 1 : -1]  # -1 to exclude the sign column

    # Construct the logical operator arrays and cast the data to the correct type
    z_logop_arr = np.hstack(
        [
            np.zeros((k, stabarr.nqubits), dtype=StabArray.DTYPE),
            a2_array.transpose(),
            np.zeros((k, n - k - r), dtype=StabArray.DTYPE),
            np.eye(k, dtype=StabArray.DTYPE),
            np.zeros((k, 1), dtype=StabArray.DTYPE),  # the sign column
        ],
        dtype=StabArray.DTYPE,
    )

    x_logop_arr = np.hstack(
        [
            np.zeros((k, r), dtype=StabArray.DTYPE),
            e_array.transpose(),
            np.eye(k, dtype=StabArray.DTYPE),
            c_array.transpose(),
            np.zeros((k, n - r), dtype=StabArray.DTYPE),
            np.zeros((k, 1), dtype=StabArray.DTYPE),  # the sign column
        ],
        dtype=StabArray.DTYPE,
    )

    # Cast the logical operator arrays to StabArray objects and reindex them to the
    # original
    # NOTE: We bypass the validation of the StabArray objects since we know that the
    # logical operator arrays are valid
    z_log_ops_stabarr = reindex_stabarray(
        StabArray(z_logop_arr, validated=True), inv_qbit_map
    )

    x_log_ops_stabarr = reindex_stabarray(
        StabArray(x_logop_arr, validated=True), inv_qbit_map
    )

    return x_log_ops_stabarr, z_log_ops_stabarr
