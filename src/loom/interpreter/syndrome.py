"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from __future__ import annotations
from uuid import uuid4

from pydantic.fields import Field
from pydantic.dataclasses import dataclass
from loom.eka.utilities import dataclass_params

from .utilities import Cbit


@dataclass(**dataclass_params)
class Syndrome:
    """
    A syndrome is the measurement result of a stabilizer. This dataclass does not
    contain the actual value of the syndrome but it stores how to calculate the syndrome
    value based on the output one gets from executing the quantum circuit.

    Parameters
    ----------
    stabilizer : str
        Uuid of the stabilizer that this syndrome corresponds to.
    measurements : tuple[Cbit, ...]
        List of classical bits from which the syndrome is calculated. The syndrome is
        calculated by taking the parity of these measurements. This includes the readout
        of ancilla qubits as well as potential updates/corrections based on data qubit
        readouts during e.g. split or shrink operations.
    block : str
        UUID of the block which contains the stabilizer that is measured in this
        syndrome.
    round : int
        Syndrome extraction round of the respective block in which this syndrome was
        measured. Note that the rounds of syndrome extraction are counted for each block
        individually.
    corrections: tuple[Cbit, ...]
        List of classical bits that needed to account for parity computation between
        subsequent rounds of syndrome extraction. This is necessary when the stabilizer
        was modified during a lattice surgery operation
    uuid : str
        Unique identifier of the syndrome, created with the uuid module.
    """

    stabilizer: str
    measurements: tuple[Cbit, ...]
    block: str
    round: int
    corrections: tuple[Cbit, ...] = Field(default_factory=tuple)
    labels: dict[str, str | tuple[int, ...] | int] = Field(default_factory=dict)
    uuid: str = Field(default_factory=lambda: str(uuid4()), validate_default=True)

    def __eq__(self, other: Syndrome) -> bool:
        if not isinstance(other, Syndrome):
            raise NotImplementedError(
                f"Cannot compare Syndrome with {type(other)} object."
            )
        return (
            self.stabilizer == other.stabilizer
            and set(self.measurements) == set(other.measurements)
            and set(self.corrections) == set(other.corrections)
            and self.block == other.block
            and self.round == other.round
        )

    def __repr__(self) -> str:
        return (
            f"Syndrome(Measurements: {self.measurements}, "
            f"Corrections: {self.corrections}, Round: {self.round}, "
            f"Labels: {self.labels})"
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.stabilizer,
                self.measurements,
                self.block,
                self.round,
                self.corrections,
            )
        )
