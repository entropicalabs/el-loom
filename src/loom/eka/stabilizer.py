"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from __future__ import annotations

from pydantic.dataclasses import dataclass
from pydantic import field_validator, model_validator, Field

from .pauli_operator import PauliOperator
from .utilities.validation_tools import (
    distinct_error,
    dataclass_params,
    ensure_tuple,
    coordinate_length_error,
)


@dataclass(**dataclass_params)
class Stabilizer(PauliOperator):
    """
    A stabilizer, representing the parity of a set of data qubits in the basis as
    defined in `pauli`.

    Parameters
    ---------
    pauli : str
        The Pauli string of the stabilizer, i.e. in which basis the data qubits are
        included in the stabilizer.

    data_qubits : tuple[tuple[int, ...], ...]
        Data qubits involved in the stabilizer. They are referred to by their coordinates
        in the lattice.

    uuid : str
        Unique identifier of the stabilizer. This is automatically set to a random UUID.

    ancilla_qubits : tuple[tuple[int, ...], ...], optional
        Ancilla qubits involved in the stabilizer. They are referred to by their
        coordinates in the lattice. Default is an empty tuple.
    """

    ancilla_qubits: tuple[tuple[int, ...], ...] = Field(
        default_factory=tuple, validate_default=True
    )

    # Validation functions
    @model_validator(mode="before")
    @classmethod
    def _calculate_ancilla_qubits_from_nr_of_ancillae(cls, data):
        """
        Calculate ancilla qubit indices when the number of ancillae is
        specified. This is useful when deserializing Workbench-generated JSON
        to Eka Block.
        """
        if hasattr(data, "kwargs") and data.kwargs is not None:
            if "nr_of_ancillae" not in data.kwargs or "ancilla_qubits" in data.kwargs:
                return data
        else:
            return data

        nr_of_ancillae = data.kwargs.pop("nr_of_ancillae")
        if nr_of_ancillae == 0:
            return data

        # This is a convention to choose coordinates of ancilla qubits. We
        # take the coordinates to be those of data_qubit[0] and the third
        # index is taken to be the count of the ancilla qubit, starting
        # from 1.
        ancilla_qubits = [
            (
                data.kwargs["data_qubits"][0][0],
                data.kwargs["data_qubits"][0][1],
                ancilla_qubit + 1,
            )
            for ancilla_qubit in range(nr_of_ancillae)
        ]
        data.kwargs["ancilla_qubits"] = tuple(ancilla_qubits)

        return data

    _validate_ancilla_qubits_list = field_validator("ancilla_qubits", mode="before")(
        ensure_tuple
    )
    _validate_distinct_ancilla_qubits = field_validator(
        "ancilla_qubits", mode="before"
    )(distinct_error)
    _validate_coordinate_lengths_ancilla = field_validator(
        "ancilla_qubits", mode="before"
    )(coordinate_length_error)

    # Magic methods
    # def __str__(self) -> str: Method is inherited from PauliOperator

    def __repr__(self) -> str:
        """
        Return a string representation of the stabilizer.
        """
        return f"{self.pauli} {self.data_qubits} {self.ancilla_qubits}"

    def __eq__(self, other: Stabilizer) -> bool:
        """
        Ignore the uuid in the equality check.
        """
        return (
            self.pauli == other.pauli
            and self.data_qubits == other.data_qubits
            and self.ancilla_qubits == other.ancilla_qubits
        )

    def __hash__(self):
        """
        Ignore the uuid in hashing.
        """
        return hash((self.pauli, self.data_qubits, self.ancilla_qubits))
