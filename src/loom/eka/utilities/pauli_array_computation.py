"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from __future__ import annotations
from typing import TypeVar

import numpy as np

from .pauli_array import PauliArray
from .pauli_computation import g_npfunc


# Dynamically bound type variable for PauliArray and its subclasses.
T = TypeVar("T", bound=PauliArray)


def ndarray_rowsum(array: np.ndarray, h: int, i: int) -> np.ndarray:
    """
    The rowsum function as described in Aaronson's paper for np.ndarray.
    Reference: https://arxiv.org/abs/quant-ph/0406196

    Parameters
    ----------
    array : np.ndarray
        The array representation of the PauliArray to be modified.
    h : int
        The row-index of the pauli string that will be modified.
    i : int
        The row-index of the pauli string that will be used.

    Returns
    -------
    np.ndarray
        The rowsum'ed array.

    Raises
    ------
    ValueError
        If the rowsum value is odd.
    """
    nqubits = array.shape[1] // 2

    g_array = g_npfunc(
        array[i, :nqubits],
        array[i, nqubits : 2 * nqubits],
        array[h, :nqubits],
        array[h, nqubits : 2 * nqubits],
    )

    sum_g = np.sum(g_array)

    lc_rowsum = 2 * array[h, 2 * nqubits] + 2 * array[i, 2 * nqubits] + sum_g

    if lc_rowsum % 4 == 2:
        array[h, 2 * nqubits] = 1
    elif lc_rowsum % 4 == 0:
        array[h, 2 * nqubits] = 0
    else:
        raise ValueError("rowsum cannot be odd!")

    array[h, :-1] = array[i, :-1] ^ array[h, :-1]

    return array


def rowsum(pauli_array: T, h: int, i: int) -> T:
    """
    The rowsum function as described in Aaronson's paper.
    Reference: https://arxiv.org/abs/quant-ph/0406196

    Parameters
    ----------
    pauli_array : PauliArray
        The PauliArray object to be modified.
    h : int
        The row-index of the pauli string that will be modified.
    i : int
        The row-index of the pauli string that will be used.

    Returns
    -------
    PauliArray
        The rowsum'ed PauliArray.
    """
    pauli_array.array = ndarray_rowsum(pauli_array.array, h, i)
    return pauli_array
