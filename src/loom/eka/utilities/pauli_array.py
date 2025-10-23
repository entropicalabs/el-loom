"""
Copyright 2024 Entropica Labs Pte Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

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
