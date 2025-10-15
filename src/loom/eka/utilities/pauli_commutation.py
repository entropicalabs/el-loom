"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import numpy as np


def paulis_anti_commute(x1: int, z1: int, x2: int, z2: int) -> int:
    """
    Calculates the anti-commutation value of two pauli operators.

    A value of 0 means that the paulis commute while a value of 1 means that
    they anti-commute.
    """
    return (x1 & z2) ^ (z1 & x2)


def anti_commutes_npfunc(
    x1: np.ndarray, z1: np.ndarray, x2: np.ndarray, z2: np.ndarray
) -> np.ndarray:
    """
    Vectorized anti-commutation function.

    Parameters
    ----------
    x1 : np.ndarray
        The x bits of the first pauli string.
    z1 : np.ndarray
        The z bits of the first pauli string.
    x2 : np.ndarray
        The x bits of the second pauli string.
    z2 : np.ndarray
        The z bits of the second pauli string.

    Returns
    -------
    np.ndarray
        The anti-commutation values.
    """
    return np.frompyfunc(paulis_anti_commute, 4, 1)(x1, z1, x2, z2)
