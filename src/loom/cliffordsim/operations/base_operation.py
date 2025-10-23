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

# pylint: disable=duplicate-code
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class Operation(ABC):
    """
    The base class for all operations that can be performed within the Engine during
    runtime.
    """

    name: str = field(init=False)
    operation_type: Enum = field(init=False)


class OpType(Enum):
    """
    The types of operations that can be performed within the Engine during runtime.
    """

    CLASSICAL = "Classical"
    QUANTUMGATE = "QuantumGate"
    RESIZE = "Resize"
    MEASUREMENT = "Measurement"
    DATAMANIPULATION = "DataManipulation"
    CCONTROL = "CControl"
