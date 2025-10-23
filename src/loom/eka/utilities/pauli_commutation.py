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
