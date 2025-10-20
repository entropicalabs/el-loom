"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from __future__ import annotations
from uuid import uuid4
import logging

from pydantic import field_validator, Field, ValidationInfo
from pydantic.dataclasses import dataclass

from .circuit import Circuit, Channel
from .utilities.validation_tools import (
    pauli_error,
    dataclass_params,
    retrieve_field,
    no_name_error,
)

logging.basicConfig(format="%(name)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def default_circuit(pauli_str: str) -> Circuit:
    """
    Construct the default circuit corresponding to the input Pauli string.

    Parameters
    ----------
    pauli_str: str
        The input Pauli string of the stabilizer

    Returns
    -------
    Circuit
        The circuit containing instructions measuring the stabilizer
    """
    weight = len(pauli_str)
    ancilla_channel = Channel(type="quantum", label="a0")
    data_channels = [Channel(type="quantum", label=f"d{i}") for i in range(weight)]
    cbit_channel = Channel(type="classical", label="c0")

    reset = [Circuit("Reset_0", channels=[ancilla_channel])]
    hadamard1 = [Circuit("H", channels=[ancilla_channel])]
    hadamard2 = [Circuit("H", channels=[ancilla_channel])]
    entangle_ancilla = [
        [Circuit(f"C{pauli}", channels=[ancilla_channel, data_channels[i]])]
        for i, pauli in enumerate(pauli_str)
    ]
    measurement = [Circuit("Measurement", channels=[ancilla_channel, cbit_channel])]
    ops_list = [reset, hadamard1] + entangle_ancilla + [hadamard2, measurement]

    circuit_name = f"stab_{pauli_str}"
    return Circuit(
        circuit_name,
        channels=data_channels + [ancilla_channel, cbit_channel],
        circuit=ops_list,
    )


@dataclass(**dataclass_params)
class SyndromeCircuit:
    """
    A SyndromeCircuit object specifies one way to measure a pauli string.
    There can be multiple SyndromeCircuit objects for the same pauli string,
    specifying different circuits for how to measure it.
    The qubits in the `circuit` field are labeled 0, ..., n.
    Once the actual syndrome extraction circuit for a stabilizer is generated,
    qubit i of the SyndromeCircuit object is replaced by stabilizer.data_qubits[i].
    Therefore the qubit order inside stabilizer.data_qubits must match the
    order in which they are used in SyndromeCircuit.

    Parameter
    ---------
    pauli : str
        The pauli string which this circuit is supposed to measure. Must be a string
    name: str, optional
        A name for the SyndromeCircuit object consisting of the characters {X, Y, Z}.
    circuit: Circuit | None
        The circuit instructions to measure the syndrome. If `None` is provided, a
        default circuit for the given pauli string is automatically constructed.
    uuid: str
        A uuid associated with the object
    """

    pauli: str = Field(min_length=1)
    name: str = Field(min_length=1, default_factory=lambda: "SyndromeCircuit")
    circuit: Circuit | None = Field(default_factory=lambda: None, validate_default=True)
    uuid: str = Field(default_factory=lambda: str(uuid4()), validate_default=True)

    # Validation functions
    _validate_pauli = field_validator("pauli")(pauli_error)
    _validate_name = field_validator("name")(no_name_error)

    @field_validator("circuit", mode="after")
    @classmethod
    def default_syndrome_circuit(cls, value, info: ValidationInfo):
        """
        In case the field circuit is not provided, the default
        syndrome extraction circuit is automatically assigned
        """
        pauli_str = retrieve_field("pauli", info)
        value = default_circuit(pauli_str) if value is None else value
        return value

    # Magic methods
    def __eq__(self, other: SyndromeCircuit) -> bool:
        """
        Ignore the uuid in the equality check.
        """
        if self.pauli != other.pauli:
            log.info("The two circuits measure different Pauli strings.")
            log.debug("%s != %s\n", self.pauli, other.pauli)
            return False

        if self.name != other.name:
            log.info("The two circuits have different names.")
            log.debug("%s != %s\n", self.name, other.name)
            return False

        if self.circuit != other.circuit:
            log.info("The two circuits have different circuit instructions.")
            return False
            # log.debug(
            #     f"\n{self.circuit}\n  !=\n{other.circuit}"
            # )

        return True

    def __hash__(self):
        """
        Ignore the uuid in hashing.
        """
        return hash((self.pauli, self.name, self.circuit))
