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


def g(x1: int, z1: int, x2: int, z2: int) -> int:
    """
    The g function as described in Aaronson's paper but written with
    bitwise operations. The result is the exponent of the imaginary
    unit accompanying the multiplication result of the two paulis P_2 * P_1.
    Example:
    X * Y = X * i Z X = -i Z
    So input (1,1,1,0) gives -1.

    Reference: https://arxiv.org/abs/quant-ph/0406196
    """
    return (
        (x1 | z1)
        * (1 - 2 * (x1 & z1))
        * (x1 * z2 * (2 * x2 - 1) + z1 * x2 * (1 - 2 * z2))
    )  # makes product zero for x1=z1=0  # flips sign for x1=z1=1


def g_npfunc(
    x1: np.ndarray, z1: np.ndarray, x2: np.ndarray, z2: np.ndarray
) -> np.ndarray:
    """
    Vectorized g function.

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
        The g values.
    """
    return np.frompyfunc(g, 4, 1)(x1, z1, x2, z2)
