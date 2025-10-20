"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from __future__ import annotations
from uuid import uuid4

from pydantic.dataclasses import dataclass
from pydantic import field_validator, Field
import numpy as np

from .utilities.validation_tools import (
    nr_of_qubits_error,
    distinct_error,
    dataclass_params,
    ensure_tuple,
    coordinate_length_error,
    pauli_error,
)
from .utilities.pauli_format_conversion import paulichar_to_xz, paulixz_to_char_npfunc
from .utilities.pauli_binary_vector_rep import SignedPauliOp


@dataclass(**dataclass_params)
class PauliOperator:
    """
    A PauliOperator is defined by a pauli string, and a set of data qubits.

    Parameters
    ----------
    pauli: str
        The Pauli string that defines this operator.
    data_qubits: tuple[tuple[int, ...], ...]
        Qubits involved in the operator. They are referred to by their coordinates
        in the lattice.
    uuid : str
        Unique identifier of the operator. This is automatically set to a random UUID.
    """

    pauli: str
    data_qubits: tuple[tuple[int, ...], ...]
    uuid: str = Field(default_factory=lambda: str(uuid4()), validate_default=True)

    # Validation functions
    _validate_pauli = field_validator("pauli")(pauli_error)
    _validate_qubits_list = field_validator("data_qubits", mode="before")(ensure_tuple)
    _validate_number_qubits = field_validator("data_qubits")(nr_of_qubits_error)
    _validate_distinct_qubits = field_validator("data_qubits")(distinct_error)
    _validate_coordinate_lengths_qubits = field_validator("data_qubits", mode="before")(
        coordinate_length_error
    )

    # Magic methods
    def __str__(self) -> str:
        pauli_ops = [
            f"{p}_{idx}" for p, idx in zip(self.pauli, self.data_qubits, strict=True)
        ]
        return " ".join(pauli_ops)

    def __repr__(self) -> str:
        # use the __str__ method to represent the PauliOperator but add the class name
        return f"{self.__class__.__name__}({self})"

    def __eq__(self, other: PauliOperator) -> bool:
        """
        Ignore the uuid in the equality check.
        """
        if not isinstance(other, PauliOperator):
            return NotImplemented
        return dict(zip(self.data_qubits, self.pauli, strict=True)) == dict(
            zip(other.data_qubits, other.pauli, strict=True)
        )

    # Properties

    @property
    def weight(self) -> int:
        """Number of qubits involved in the operator."""
        return len(self.data_qubits)

    @property
    def pauli_type(self) -> str:
        """Type of the Pauli operator: 'X', 'Y', or 'Z'."""
        unique_paulis = set(self.pauli)
        if len(unique_paulis) != 1:
            raise ValueError(
                "PauliOperator must consist of a single type of Pauli operator "
                f"to determine its type, got {unique_paulis} instead."
            )
        return unique_paulis.pop()

    # Methods
    def as_signed_pauli_op(
        self, all_qubits: tuple[tuple[int, ...], ...]
    ) -> SignedPauliOp:
        """
        Get the SignedPauliOp representation of the PauliOperator.

        Parameters
        ----------
        all_qubits: tuple[tuple[int, ...], ...]
            All qubits coordinates in the system.

        Returns
        -------
        SignedPauliOp
            The SignedPauliOp representation of the PauliOperator.

        Raises
        ------
        ValueError
            If the number of qubits in the system is less than the number of qubits in
            the operator.
        """
        all_qubits = tuple(map(tuple, all_qubits))
        if len(all_qubits) < len(self.data_qubits):
            raise ValueError(
                f"Number of qubits in the operator {len(self.data_qubits)} exceeds the "
                f"total number of qubits in the system {len(all_qubits)}."
            )

        # Get the x and z values for each qubit in the operator
        x_values, z_values = tuple(
            zip(*[paulichar_to_xz(p) for p in self.pauli], strict=True)
        )
        # Get the sign of the operator
        sign = 0
        # Cast the indexed dqubits to a numpy array for indexing
        all_qubits_map = {q: i for i, q in enumerate(all_qubits)}
        idx_dqubits = np.array(
            [all_qubits_map[q] for q in self.data_qubits if q in all_qubits_map]
        )

        # Initialize the operator row
        op_row = np.zeros(2 * len(all_qubits) + 1, dtype=SignedPauliOp.DTYPE)
        # Fill the operator row with the x, z values and the sign
        op_row[idx_dqubits] = x_values
        op_row[len(all_qubits) + idx_dqubits] = z_values
        op_row[-1] = sign

        # Use the op_row to create a SignedPauliOp object that is already validated
        return SignedPauliOp(op_row, validated=True)

    @staticmethod
    def from_signed_pauli_op(
        signed_pauli_op: SignedPauliOp, index_to_qubit_map: dict[int, tuple[int, ...]]
    ) -> PauliOperator:
        """
        Create a PauliOperator from a SignedPauliOp.

        Parameters
        ----------
        signed_pauli_op : SignedPauliOp
            The SignedPauliOp to convert to a PauliOperator.
        index_to_qubit_map : dict[int, tuple[int, ...]]
            A dictionary mapping the indices of the SignedPauliOp to the qubit
            coordinates in the lattice.

        Returns
        -------
        PauliOperator
            The PauliOperator representation of the SignedPauliOp.
        """
        # Find the indexed_dqubits
        indexed_dqubits = np.union1d(
            np.where(signed_pauli_op.x != 0)[0],
            np.where(signed_pauli_op.z != 0)[0],
        )

        # Check if all the indexed dqubits have a corresponding qubit in the map
        missing_indices = set(indexed_dqubits) - set(index_to_qubit_map.keys())
        if missing_indices:
            # Sort and cast them into a sorted list of integers
            missing_indices = sorted(map(int, missing_indices))
            raise ValueError(
                f"Missing qubit coordinates for indices {missing_indices}."
            )

        # Get the pauli string of the indexed dqubits
        pauli_str = "".join(
            paulixz_to_char_npfunc(
                signed_pauli_op.x[indexed_dqubits], signed_pauli_op.z[indexed_dqubits]
            )
        )

        return PauliOperator(
            pauli=pauli_str,
            data_qubits=tuple(index_to_qubit_map[i] for i in indexed_dqubits),
        )

    def commutes_with(self, other_operator: PauliOperator) -> bool:
        """
        Check if the PauliOperator commutes with another PauliOperator.

        Parameters
        ----------
        other_operator : PauliOperator
            The other PauliOperator to check commutation with.

        Returns
        -------
        bool
            True if the two objects commute, False otherwise.
        """
        if not isinstance(other_operator, PauliOperator):
            raise ValueError(
                "Expected PauliOperator object, got " f"{type(other_operator)} instead."
            )
        # Find common qubits
        common_qubits = set(self.data_qubits).intersection(
            set(other_operator.data_qubits)
        )

        # Find for each of the common qubits whether their paulis anti-commute.
        # They anti-commute if their paulis are different.
        anti_commutation_of_common_qubits = [
            self.pauli[self.data_qubits.index(qubit)]
            # the above is the pauli of the self PauliOperator for the qubit
            !=
            # the one below is the pauli of the other PauliOperator for the qubit
            other_operator.pauli[other_operator.data_qubits.index(qubit)]
            for qubit in common_qubits
        ]

        # Return the total commutation of the common qubits
        return not bool(np.sum(anti_commutation_of_common_qubits) % 2)
