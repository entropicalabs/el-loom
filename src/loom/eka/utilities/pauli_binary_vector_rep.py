"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from __future__ import annotations

from abc import ABC
from typing import Sequence
from copy import deepcopy
import re

import numpy as np

from .pauli_format_conversion import (
    paulichar_to_xz_npfunc,
    paulixz_to_char_npfunc,
)
from .pauli_commutation import anti_commutes_npfunc
from .pauli_computation import g_npfunc


class PauliOp(ABC):
    """
    Abstract PauliOp class, parent of SignedPauliOp and UnsignedPauliOp.

    Parameters
    ---------
    DTYPE : np.dtype
        The type of numpy arrays used as bit arrays.
    array : np.ndarray
        The array representation of the Pauli operator.
    """

    DTYPE = np.int8
    array: np.ndarray

    @property
    def nqubits(self) -> int:
        """
        The number of qubits that the Pauli operator acts on.
        """
        return len(self.array) // 2

    @property
    def x(self) -> np.array:
        """
        The array representing the X-component of the Pauli operator in binary
        representation.
        """
        return self.array[0 : self.nqubits]

    @property
    def z(self) -> np.array:
        """
        The array representing the Z-component of the Pauli operator in binary
        representation.
        """
        return self.array[self.nqubits : 2 * self.nqubits]

    @property
    def is_trivial(self) -> bool:
        """
        Checks if the Pauli operator is identity.
        """
        return not np.any(self.array)

    def __eq__(self, o_obj: PauliOp) -> bool:
        if not isinstance(o_obj, self.__class__):
            raise TypeError(
                f"Cannot compare {self.__class__} with object of type {type(o_obj)}"
            )
        return np.array_equal(self.array, o_obj.array)


class SignedPauliOp(PauliOp):
    """
    A class describing a SignedPauliOp, a PauliOp operator with a sign that is + or -.

    Parameters
    ----------
    array : np.ndarray | Sequence
        The array representation of the SignedPauliOp.
    """

    # Constructors
    def __init__(self, array: np.ndarray | Sequence, validated: bool = False) -> None:
        """
        Initialization of the SignedPauliOp via a numpy array.

        Parameters
        ----------
        array : np.ndarray | Sequence
            The array representation of the SignedPauliOp
        """
        if not validated:
            if not isinstance(array, (np.ndarray, Sequence)):
                raise TypeError("Input argument should be a NumPy array or a sequence.")

            if not isinstance(array, np.ndarray):
                # make it into an array
                array = np.array(array)

            if array.ndim != 1:
                raise ValueError("NumPy array should be 1-D.")

            if not len(array) % 2 == 1:
                raise ValueError("Numpy array has to have an odd number of bits")

            # check if all elements are 0 or 1
            if not np.all(array * (array - 1) == 0):
                raise ValueError("The input array has to consist of 0 and 1.")

            if not array.dtype == self.DTYPE:
                # cast it into the correct data type
                array = array.astype(self.DTYPE)

        self.array = array

    @classmethod
    def from_string(cls, pauli_str: str) -> SignedPauliOp:
        """
        Create a SignedPauliOp from a Pauli string, like "+IXZZY"

        Parameters
        ----------
        pauli_str : str
            The Pauli string to create the SignedPauliOp from.

        Returns
        -------
        SignedPauliOp
            The SignedPauliOp created from the Pauli string.

        Raises
        ------
        ValueError
            If the first character of the Pauli string is not '+' or '-'.
        """

        sign = pauli_str[0]
        pauli_chars = pauli_str[1:]

        nqubits = len(pauli_chars)
        array = np.zeros(2 * nqubits + 1, dtype=cls.DTYPE)

        if sign == "+":
            pass
        elif sign == "-":
            array[-1] = 1
        else:
            raise ValueError(
                "The first character of the a SignedPauliOp string should be '+' or  '-"
            )

        # set the array values from
        x, z = paulichar_to_xz_npfunc(np.array(list(pauli_chars)))
        array[0:nqubits] = x
        array[nqubits : 2 * nqubits] = z

        return cls(array, validated=True)

    @classmethod
    def from_sparse_string(
        cls, pauli_str: str, nqubits: int | None = None
    ) -> SignedPauliOp:
        """
        Create a SignedPauliOp from a sparse Pauli string, like "+X2Z5Y7"

        Parameters
        ----------
        pauli_str : str
            The sparse Pauli string to create the SignedPauliOp from. It is not case
            sensitive.
        nqubits : int | None
            The total number of qubits the Pauli operator acts on. Since SignedPauliOp
            is a dense representation, all qubits with index starting from 0 until
            nqubits-1 will be described by the array. The number of qubits should be
            greater than the maximum index in the Pauli string. If None, the number of
            qubits is inferred from the maximum index in the Pauli string.

        Returns
        -------
        SignedPauliOp
            The SignedPauliOp created from the Pauli string.

        Raises
        ------
        ValueError
            If the first character of the Pauli string is not '+' or '-'.
            If there are invalid elements in the Pauli string.
            If the indices in the Pauli string are not unique.
            If the number of qubits is not greater than the maximum index in the Pauli
            string.
        """
        # Get the the Pauli index pairs
        pauli_index_pairs = pauli_str[1:]

        # Parse the sign
        match pauli_str[0]:
            case "+":
                sign = 0
            case "-":
                sign = 1
            case _:
                raise ValueError(
                    "The first character of the a Pauli string should be '+' or '-'."
                )

        # Define the regular expression pattern
        pattern = re.compile(r"([XYZ])(\d+)", re.IGNORECASE)

        # Find all matches in the input string
        matches = pattern.findall(pauli_index_pairs)

        # Check for invalid segments that do not match the pattern
        invalid_segments = pattern.sub("", pauli_index_pairs)
        if invalid_segments:
            raise ValueError(
                f"Invalid elements in the Pauli string: {invalid_segments}."
            )

        # Get the indices and the pauli operators into 2 separate arrays
        indices = np.array([int(idx) for _, idx in matches])
        paulis = np.array([op for op, _ in matches])

        # check that the indices are unique
        if len(indices) != len(set(indices)):
            repeating_indices = {
                int(idx) for idx in indices if list(indices).count(idx) > 1
            }
            raise ValueError(
                f"Qubit indices {repeating_indices} appear more than once in the "
                "Pauli string."
            )

        # Get the maximum index
        max_index = max(indices)

        # Infer the number of qubits if not provided
        if nqubits is None:
            nqubits = max_index + 1
        else:
            if nqubits < max_index + 1:
                raise ValueError(
                    f"Qubit index {max_index} is out of range for {nqubits} qubits."
                )
        # Convert the Pauli operators to x and z bit values
        x_vals, z_vals = paulichar_to_xz_npfunc(paulis)

        # Initialize the array
        array = np.zeros(2 * nqubits + 1, dtype=cls.DTYPE)
        # Set the array values from the sparse string
        array[indices] = x_vals
        array[nqubits + indices] = z_vals
        array[-1] = sign

        return cls(array, validated=True)

    @classmethod
    def identity(cls, nqubits: int, negative=False) -> SignedPauliOp:
        """
        Create an identity operator acting on nqubits. Can be used to create a
        negative identity operator as well.

        Parameters
        ----------
        nqubits : int
            The number of qubits the identity operator should act on.
        negative : bool
            If True, return a negative identity operator.

        Returns
        -------
        SignedPauliOp
            The identity operator acting on nqubits.
        """
        array = np.zeros(2 * nqubits + 1, dtype=cls.DTYPE)
        if negative:
            array[-1] = 1
        return SignedPauliOp(array, validated=True)

    # Magic methods
    def __str__(self) -> str:
        """
        The string representation of the SignedPauliOp
        """
        sign = "+" if self.array[-1] == 0 else "-"
        return sign + "".join(paulixz_to_char_npfunc(self.x, self.z))

    def __repr__(self) -> str:
        """
        The representation of the SignedPauliOp.
        """
        return f"{self.__class__.__name__}({str(self)})"

    def __mul__(self, other: SignedPauliOp) -> SignedPauliOp:
        """
        Multiply two SignedPauliOps.

        Parameters
        ----------
        other : SignedPauliOp
            The other SignedPauliOp to multiply with.

        Returns
        -------
        SignedPauliOp
            The product of the two SignedPauliOps.
        """
        if not isinstance(other, SignedPauliOp):
            raise TypeError("Can only multiply with another SignedPauliOp.")

        if self.nqubits != other.nqubits:
            raise ValueError(
                "The two Pauli operators should act on the same number of qubits."
            )

        if pauliops_anti_commute(self, other):
            raise ValueError(
                "Cannot multiply anti-commuting Pauli operators. That "
                "would give an imaginary unit contribution in the product."
                "To multiply anti-commuting operators use the method "
                "multiply_with_anticommuting_operator instead."
            )

        # Get the product of the 2 Pauli operators without the sign
        x = self.x ^ other.x
        z = self.z ^ other.z

        # If the sum of g modulo 4 is:
        # 0 -> positive sign (0) contribution
        # 2 -> negative sign (1) contribution
        # 3 and 1 are only possible for anti-commuting Pauli operators.
        sign = (
            self.sign
            ^ other.sign
            ^ sum(g_npfunc(self.x, self.z, other.x, other.z)) % 4 // 2
        )

        return SignedPauliOp(
            np.concatenate((x, z, [sign])).astype(self.DTYPE), validated=True
        )

    # Properties
    @property
    def sign(self) -> SignedPauliOp.DTYPE:
        """
        The sign of the Pauli operator, either 0 or 1.
        """
        return self.array[-1]

    @property
    def is_minus_identity(self) -> bool:
        """
        Check if the Pauli operator is a minus identity operator. Returns True if the
        Pauli operator is a minus identity operator, False otherwise.
        """
        return np.all(self.array[:-1] == 0) and self.array[-1] == 1

    # Methods
    def as_sparse_string(self) -> str:
        """
        Return the Pauli operator as a sparse Pauli string.

        Returns
        -------
        str
            The Pauli operator as a sparse Pauli string.
        """
        # Find the indices of the non-zero elements
        x_indices = np.nonzero(self.x)[0]
        z_indices = np.nonzero(self.z)[0]
        indices = np.union1d(x_indices, z_indices)
        # Get the Pauli operators
        paulis = paulixz_to_char_npfunc(self.x[indices], self.z[indices])
        # Get the sign
        sign = "+" if self.sign == 0 else "-"
        # Combine the sign and the Pauli operators
        return sign + "".join(
            f"{pauli}{idx}" for idx, pauli in zip(indices, paulis, strict=True)
        )

    def with_flipped_sign(self) -> SignedPauliOp:
        """
        Return a copy of the operator with the sign flipped.

        Returns
        -------
        SignedPauliOp
            The Pauli operator with the sign flipped.
        """
        return SignedPauliOp(
            np.concatenate((self.array[:-1], [1 - self.sign])).astype(self.DTYPE),
            validated=True,
        )

    def reindexed(
        self, qubit_map: list[int], nqubits: int | None = None
    ) -> SignedPauliOp:
        """
        Return a copy of the operator with the qubits reindexed according to the
        qubit map.

        Parameters
        ----------
        qubit_map : list[int]
            The mapping of the new qubit indices to the old qubit indices.
        nqubits : int | None
            The total number of qubits the Pauli operator acts on. If None, the number
            of qubits is inferred from the maximum index in the qubit map.

        Returns
        -------
        SignedPauliOp
            The Pauli operator with the reindexed qubits.
        """
        if nqubits is None:
            nqubits = max(qubit_map) + 1

        if len(set(qubit_map)) != len(qubit_map):
            raise ValueError("The qubit map should not contain duplicate indices.")

        if len(qubit_map) != self.nqubits:
            raise ValueError(
                f"The qubit map length ({len(qubit_map)}) should be equal "
                f"to the number of qubits in the Pauli operator ({self.nqubits})."
            )

        if nqubits <= max(qubit_map):
            raise ValueError(
                f"The number of qubits ({nqubits}) should be greater than "
                f"the maximum index in the qubit map ({max(qubit_map)})."
            )
        # cast the qubit map to a numpy array
        qubit_map = np.array(qubit_map)
        # make the new array
        new_array = np.zeros(2 * nqubits + 1, dtype=self.DTYPE)
        new_array[-1] = self.sign
        new_array[qubit_map] = self.x
        new_array[nqubits + qubit_map] = self.z

        return SignedPauliOp(new_array, validated=True)

    # Methods
    def multiply_with_anticommuting_operator(
        self, other: SignedPauliOp
    ) -> SignedPauliOp:
        """
        Multiply two Pauli operators that anti-commute by including a factor of i.
        For 2 Pauli operators A and B that anti-commute,
        A.multiply_with_anticommuting_operator(B) returns the SignedPauliOp i * A * B.

        Parameters
        ----------
        other : SignedPauliOp
            The other Pauli operator to multiply with.

        Returns
        -------
        SignedPauliOp
            The product of the two Pauli operators with a factor of i.
        """
        if not isinstance(other, SignedPauliOp):
            raise TypeError("Can only multiply with another SignedPauliOp.")

        if self.nqubits != other.nqubits:
            raise ValueError(
                "The two Pauli operators should act on the same number of qubits."
            )

        if not pauliops_anti_commute(self, other):
            raise ValueError(
                "Can only multiply anti-commuting Pauli operators. "
                "To multiply commuting operator use the operator *"
            )

        # Find the exponent of the imaginary unit
        i_exp = (
            sum(g_npfunc(self.x, self.z, other.x, other.z))  # from the commutation #
            + 2 * self.sign  # from the sign of the first operator #
            + 2 * other.sign  # from the sign of the second operator #
            + 1  # from the imaginary unit #
        ) % 4  # i^4 = 1 #
        # And then deduce the sign
        sign = i_exp // 2

        # Get the product of the 2 Pauli operators without the sign
        x = self.x ^ other.x
        z = self.z ^ other.z

        return SignedPauliOp(
            np.concatenate((x, z, [sign])).astype(self.DTYPE), validated=True
        )

    def copy(self) -> SignedPauliOp:
        """
        Returns a deep copy of the SignedPauliOp object.

        Returns
        -------
        SignedPauliOp
            Deep copy of the SignedPauliOp object
        """
        return deepcopy(self)


def pauliops_anti_commute(op1: PauliOp, op2: PauliOp) -> int:
    """
    Given two pauli strings, find their anti-commutation value.
    A value of `0` means that they commute and a value of `1` means that they
    anti-commute.

    Parameters
    ----------
    op1 : PauliOp
        One of the Pauli operators.
    op2 : PauliOp
        The other Pauli operator.

    Returns
    -------
    int
        The anti-commutation value of the two Pauli operators.
        0 if they commute, 1 if they anti-commute.
    """
    if not (isinstance(op1, PauliOp) and isinstance(op2, PauliOp)):
        raise TypeError("Both inputs should be of type PauliOp.")

    anti_comm_array = anti_commutes_npfunc(op1.x, op1.z, op2.x, op2.z)

    return np.sum(anti_comm_array) % 2


class UnsignedPauliOp(PauliOp):
    """A class describing an UnsignedPauliOp, a Pauli operator without a sign."""

    def __init__(self, array: np.ndarray | Sequence, validated: bool = False) -> None:
        """Initialization of the UnsignedPauliOp via a numpy array.

        Parameters
        ----------
        array : np.ndarray | Sequence
            The array representation of the UnsignedPauliOp
        """
        if not validated:
            if not isinstance(array, (np.ndarray, Sequence)):
                raise TypeError("Input argument should be a NumPy array or a sequence.")

            if not isinstance(array, np.ndarray):
                # make it into an array
                array = np.array(array)

            if array.ndim != 1:
                raise ValueError("NumPy array should be 1-D.")

            if not len(array) % 2 == 0:
                raise ValueError("Numpy array has to have an even number of bits")

            # check if all elements are 0 or 1
            if not np.all(array * (array - 1) == 0):
                raise ValueError("The input array has to consist of 0 and 1.")

            if not array.dtype == self.DTYPE:
                # cast it into the correct data type
                array = array.astype(self.DTYPE)

        self.array = array

    @classmethod
    def from_string(cls, pauli_str: str) -> UnsignedPauliOp:
        """Create an UnsignedPauliOp from a Pauli string, like "IXZZY"

        Parameters
        ----------
        pauli_str : str
            The Pauli string to create the UnsignedPauliOp from.

        Returns
        -------
        UnsignedPauliOp
            The UnsignedPauliOp created from the Pauli string.
        """
        if pauli_str[0] in ["+", "-"]:
            raise ValueError(
                "The first character of the a UnsignedPauliOp cannot be '+' or '-'. "
                "Maybe you want to use SignedPauliOp class instead."
            )
        pauli_chars = pauli_str
        nqubits = len(pauli_chars)
        array = np.zeros(2 * nqubits, dtype=cls.DTYPE)

        # set the array values from
        x, z = paulichar_to_xz_npfunc(np.array(list(pauli_chars)))
        array[0:nqubits] = x
        array[nqubits : 2 * nqubits] = z

        return cls(array, validated=True)

    def __str__(self) -> str:
        """The string representation of the UnSignedPauliOp"""
        return "".join(paulixz_to_char_npfunc(self.x, self.z))

    def copy(self) -> UnsignedPauliOp:
        """
        Returns a deep copy of the UnsignedPauliOp object.
        """
        return deepcopy(self)
