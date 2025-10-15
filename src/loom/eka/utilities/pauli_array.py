"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from __future__ import annotations
from abc import ABC

import numpy as np


class PauliArray(ABC):
    """
    Abstract class for PauliArray, parent of StabArray and Tableau.

    Parameters
    ----------
    array : np.ndarray
        The array representation of the PauliArray.
    """

    array: np.ndarray

    @property
    def nqubits(self) -> int:
        """
        The number of qubits that the PauliArray operators act on.
        """
        return self.array.shape[1] // 2

    @property
    def x(self) -> np.ndarray:
        """
        The array representing the X-component of the PauliArray in binary representation.
        """
        return self.array[:, : self.nqubits]

    @property
    def z(self) -> np.ndarray:
        """
        The array representing the Z-component of the PauliArray in binary representation.
        """
        return self.array[:, self.nqubits : 2 * self.nqubits]
