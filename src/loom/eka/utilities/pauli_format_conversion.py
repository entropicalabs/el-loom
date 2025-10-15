"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import numpy as np


def paulichar_to_xz(p: str) -> tuple[int, int]:
    """
    Function that turns a Pauli into a pair of x,z bits.

    Parameters
    ----------
    p : str
        The Pauli character.

    Returns
    -------
    tuple[int, int]
        x, z bits

    Raises
    ------
    ValueError
        If the Pauli is not I, Z, X or Y.
    """

    match p:
        case "_" | "I" | "i":
            return (0, 0)
        case "Z" | "z":
            return (0, 1)
        case "X" | "x":
            return (1, 0)
        case "Y" | "y":
            return (1, 1)
        case _:
            raise ValueError(
                "The Pauli character should be I, Z, X or Y or their "
                "lower case versions. _ is also accepted as I."
            )


def paulichar_to_xz_npfunc(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized version of paulichar_to_xz.

    Parameters
    ----------
    p : np.ndarray
        An array of Pauli characters.

    Returns
    -------
    tuple[np.ndarray]
        Two arrays of x and z bits.
    """

    return np.frompyfunc(paulichar_to_xz, 1, 2)(p)


def paulixz_to_char(
    x: int,
    z: int,
) -> str:
    """
    Function that turns a pair of x, z bits into their Pauli as a str.

    Parameters
    ----------
    x : int
        The x bit.
    z : int
        The z bit.

    Returns
    -------
    str
        The Pauli character as a string.

    Raises
    ------
    ValueError
        If x or z are not 0 or 1.
    """

    match (x, z):
        case (0, 0):
            return "_"
        case (0, 1):
            return "Z"
        case (1, 0):
            return "X"
        case (1, 1):
            return "Y"
        case _:
            raise ValueError("The x and z values should be 0 or 1.")


def paulixz_to_char_npfunc(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Vectorized version of paulixz_to_char.

    Parameters
    ----------
    x : np.ndarray
        An array of x bits.
    z : np.ndarray
        An array of z bits.

    Returns
    -------
    np.ndarray
        An array of Pauli characters.
    """

    return np.frompyfunc(paulixz_to_char, 2, 1)(x, z)
