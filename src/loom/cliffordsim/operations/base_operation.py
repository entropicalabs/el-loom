"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

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
