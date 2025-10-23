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
