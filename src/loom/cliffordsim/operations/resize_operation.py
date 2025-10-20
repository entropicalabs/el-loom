"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from dataclasses import dataclass, field

from .base_operation import Operation, OpType
from .controlled_operation import has_ccontrol


@dataclass
@has_ccontrol
class ResizeOperation(Operation):
    """
    Operations of this type resize the number of qubits in the Engine during runtime.
    """

    operation_type: str = field(default=OpType.RESIZE, init=False)
    target_qubit: int


@dataclass
class AddQubit(ResizeOperation):
    """
    An Operation that adds a qubit to the Engine during runtime. The qubit will be
    initialised in the `|0>` state.
    """

    name: str = field(default="AddQubit", init=False)


@dataclass
class DeleteQubit(ResizeOperation):
    """
    An Operation that deletes a qubit from the Engine during runtime.
    """

    name: str = field(default="DeleteQubit", init=False)
