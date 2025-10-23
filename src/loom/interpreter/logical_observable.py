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

from pydantic.dataclasses import dataclass
from loom.eka.utilities import dataclass_params

from .utilities import Cbit


@dataclass(**dataclass_params)
class LogicalObservable:
    """
    Once a logical operator is measured, the details of the measurement are stored in
    an instance of LogicalObservable. This dataclass does not store the actual value of
    the logical observable but instead the required information on how to calculate it
    given the output of a quantum circuit.

    Parameters
    ----------
    label : str
        Label of the logical observable measurement which is later on used to access the
        measurement result
    measurements : tuple[Cbit, ...]
        Tuple of classical bits from which the logical observable is calculated. The
        logical observable is calculated by taking the parity of these measurements.
        This includes the readout of data qubits as well as potential updates/
        corrections based on data qubit readouts during e.g. split or shrink operations.
    """

    label: str
    measurements: tuple[Cbit, ...]

    def __eq__(self, other: LogicalObservable) -> bool:
        return self.label == other.label and set(self.measurements) == set(
            other.measurements
        )
