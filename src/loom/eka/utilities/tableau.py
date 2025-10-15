"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import numpy as np

from .graph_matrix_utils import binary_gaussian_elimination
from .pauli_binary_vector_rep import SignedPauliOp, pauliops_anti_commute


def tableau_generates_pauli_group(tableau: np.ndarray) -> bool:
    """
    Given a tableau, check if all of its operators, (stabilizers & destabilizers) can
    generate the pauli group. This is done by checking if the tableau is equivalent to
    the identity matrix after performing gaussian elimination.

    Parameters
    ----------
    tableau : np.ndarray
        The tableau to be checked.

    Returns
    -------
    bool
        Whether the tableau generates the Pauli group.
    """

    _, ncols = tableau.shape
    nqubits = ncols // 2

    # remove signs (last column) from the tableau, if present
    if ncols % 2 == 1:
        tableau = tableau[:, : ncols - 1]

    # perform gaussian elimination
    post_gauss_tab = binary_gaussian_elimination(tableau)

    # initialize identity matrix to check pauli group generation
    id_matr = np.eye(2 * nqubits, dtype=post_gauss_tab.dtype)

    return np.array_equal(post_gauss_tab, id_matr)


def is_tableau_valid(tableau: np.ndarray) -> bool:
    """
    Checks if a tableau is valid. This is done by checking if it generates the pauli
    group and whether the correct commutation relations hold.

    For the commutation relations, the tableau is split into two sets of
    stabilizers of equal size calling them S (stabilizers) and D (destabilizers). The
    following is then checked:

    * All stabilizers of S commute with each other and all stabilizers of D
        commute with each other.
    * Stabilizer i from list S has to anti-commute with stabilizer j from list
        D if i=j. If i != j, they have to commute.

    Parameters
    ----------
    tableau : np.ndarray
        The tableau to be checked.

    Returns
    -------
    bool
        Whether the tableau is valid.
    """

    nrows, _ = tableau.shape
    nstabs = nrows // 2

    # check if it generates the pauli group
    # NOTE: It might be the case that this test is not needed, i.e. if the other
    # conditions are fulfilled, this may be certain.
    generates_pauli_group = tableau_generates_pauli_group(tableau)

    if not generates_pauli_group:
        return False

    # Check if commutation relations are violated
    # We are comparing pairs of operators by calculating
    # the 2d array that shows the anti-commutation of
    # pairs stabilizers/destabilizers.
    destab_array = tableau[:nstabs, :]
    stab_array = tableau[nstabs:, :]

    # all commutations of stabilizers should be zero
    # [s_i, s_j] = 0
    if any(
        pauliops_anti_commute(
            SignedPauliOp(stab_array[s1_idx, :]), SignedPauliOp(stab_array[s2_idx, :])
        )
        for s1_idx in range(nstabs)
        for s2_idx in range(s1_idx + 1)
    ):
        return False

    # all commutations of destabilizers should be zero
    # [d_i, d_j] = 0
    if any(
        pauliops_anti_commute(
            SignedPauliOp(destab_array[d1_idx, :]),
            SignedPauliOp(destab_array[d2_idx, :]),
        )
        for d1_idx in range(nstabs)
        for d2_idx in range(d1_idx, nstabs)
    ):
        return False

    # Commutation relations for stabilizers-destabilizers:
    # {s_i, d_j} = 0 iff i=j
    if any(
        pauliops_anti_commute(
            SignedPauliOp(stab_array[s_idx, :]), SignedPauliOp(destab_array[d_idx, :])
        )
        ^ (s_idx == d_idx)
        for s_idx in range(nstabs)
        for d_idx in range(s_idx, nstabs)
    ):  # iff i=j, then flip the condition
        return False

    return True
