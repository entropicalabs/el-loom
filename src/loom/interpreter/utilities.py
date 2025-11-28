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

from uuid import uuid4
from typing import Literal
from pydantic.dataclasses import dataclass, Field

#: A Cbit denotes the location of a classical bit by a tuple of a string and an integer.
#: str: Name/id of classical register, int: measurement index inside this register
Cbit = tuple[str, int] | Literal[1, 0]


@dataclass
class CompositeOperationSession:
    """
    Class representing a composite operation session.

    Attributes:
    ------------
    start_timeslice_index : int
        The index of the timeslice where the composite operation session starts.
    same_timeslice : bool
        A flag indicating whether all operations in the composite operation are
        to be executed in the same timeslice as the previous timeslice.
    circuit_name : str
        The name assigned to the composite circuit.
    uuid : str
        A unique identifier for the composite operation session.
    """

    start_timeslice_index: int
    same_timeslice: bool
    circuit_name: str
    uuid: str = Field(
        default_factory=lambda: str(uuid4()), validate_default=True, init=False
    )
