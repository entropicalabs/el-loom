"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import json

from pydantic.dataclasses import dataclass, Field
from loom.eka.utilities import dataclass_params

from .syndrome import Syndrome
from .utilities import Cbit


@dataclass(**dataclass_params)
class Detector:
    """
    A detector is the parity of multiple syndromes. This dataclass does not store the
    actual value of a detector but instead the required information on how to calculate
    it given the output of a quantum circuit.

    Parameters
    ----------
    syndromes : tuple[Syndrome, ...]
        Syndromes that this detector corresponds to, in temporal order of measurement.
        NOTE: "Please ensure this order for proper functioning of the detector"
    """

    syndromes: tuple[Syndrome, ...]
    labels: dict[str, str | tuple[int, ...] | int] = Field(default_factory=dict)

    @property
    def measurements(self) -> tuple[Cbit, ...]:
        """
        Returns the measurements of the constituent syndromes. The corrections
        must be used in a detector definition only once.

        Returns
        -------
        tuple[Cbit, ...]
            The detector measurements
        """
        measurements_list = []
        for syndrome in self.syndromes:
            measurements_list.extend(syndrome.measurements)

        # only include corrections from the last syndrome
        measurements_list.extend(self.syndromes[-1].corrections)
        return tuple(measurements_list)

    def dumps(self) -> str:
        """
        Return the detector as a JSON string. This string does not contain the
        complete Syndrome objects but only their UUIDs to not store redundant
        information.
        """
        self_dict = {"syndromes": [syndrome.uuid for syndrome in self.syndromes]}
        return json.dumps(self_dict)

    def rounds(self) -> tuple[int]:
        """
        Returns the rounds of the constituent syndromes

        Returns
        -------
        tuple[int]
            The detector rounds
        """
        rounds_list = []
        for syndrome in self.syndromes:
            rounds_list.append(syndrome.round)

        return tuple(set(rounds_list))

    def stabilizer(self) -> tuple[str]:
        """
        Returns the UUIDs of the stabilizer(s) of the constituent syndromes

        Returns
        -------
        tuple[str]
            The UUIDs of the stabilizer(s) of the constituent syndromes.
        """
        stab_ids = tuple(syndrome.stabilizer for syndrome in self.syndromes)
        return stab_ids

    def __eq__(self, other: "Detector") -> bool:
        if not isinstance(other, Detector):
            raise NotImplementedError(
                f"Cannot compare Detector with {type(other)} object."
            )
        return set(self.syndromes) == set(other.syndromes)

    def __repr__(self) -> str:
        syndrome_repr = ", ".join(repr(syndrome) for syndrome in self.syndromes)
        return f"Detector(Syndromes: ({syndrome_repr}), Labels: {self.labels})"

    def __hash__(self):
        return hash(self.syndromes)
