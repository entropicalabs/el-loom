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

from pydantic.dataclasses import dataclass
from .base_operation import Operation
from ..utilities import SingleQubitPauliEigenstate, dataclass_config


@dataclass(config=dataclass_config)
class LogicalOperation(Operation):
    """
    Parent class for all logical operations in the Eka. All logical operations acts
    on logical qubits. Details of implementation are abstracted away, including how the
    qubits are constructed. E.g. they may be constructed from multiple blocks with a
    single qubit each, or from a single block with multiple qubits.

    Note that these operations only modify the state of the logical qubits, they do not
    perform any code level operations like making sure the structure of the code is
    preserved (e.g. a block ends up being rotated).
    """


@dataclass(config=dataclass_config)
class Reset(LogicalOperation):
    """
    Reset a logical qubit to one of the supported states.

    Parameters
    ----------
    target_qubit : str
        Name of the logical qubit to reset.
    state : SingleQubitPauliEigenstate
        State to reset the qubit to.
    """

    target_qubit: str
    state: SingleQubitPauliEigenstate


@dataclass(config=dataclass_config)
class CNOT(LogicalOperation):
    """
    Describes a CNOT gate between two logical qubits.

    Parameters
    ----------
    control_qubit : str
        Name of the control logical qubit.
    target_qubit : str
        Name of the target logical qubit.
    """

    control_qubit: str
    target_qubit: str


@dataclass(config=dataclass_config)
class Hadamard(LogicalOperation):
    """
    Describes a Hadamard gate on a logical qubit.

    Parameters
    ----------
    target_qubit : str
        Name of the target logical qubit.
    """

    target_qubit: str


@dataclass(config=dataclass_config)
class Phase(LogicalOperation):
    """
    Describes a Phase gate on a logical qubit.

    Parameters
    ----------
    target_qubit : str
        Name of the target logical qubit.
    """

    target_qubit: str


@dataclass(config=dataclass_config)
class PhaseInverse(LogicalOperation):
    """
    Describes an inverse Phase gate on a logical qubit.

    Parameters
    ----------
    target_qubit : str
        Name of the target logical qubit.
    """

    target_qubit: str


@dataclass(config=dataclass_config)
class X(LogicalOperation):
    """
    Describes an X gate on a logical qubit.

    Parameters
    ----------
    target_qubit : str
        Name of the target logical qubit.
    """

    target_qubit: str


@dataclass(config=dataclass_config)
class Y(LogicalOperation):
    """
    Describes a Y gate on a logical qubit.

    Parameters
    ----------
    target_qubit : str
        Name of the target logical qubit.
    """

    target_qubit: str


@dataclass(config=dataclass_config)
class Z(LogicalOperation):
    """
    Describes a Z gate on a logical qubit.

    Parameters
    ----------
    target_qubit : str
        Name of the target logical qubit.
    """

    target_qubit: str


# what about non-Clifford gates?
@dataclass(config=dataclass_config)
class T(LogicalOperation):
    """
    Describes a T gate on a logical qubit.

    Parameters
    ----------
    target_qubit : str
        Name of the target logical qubit.
    """

    target_qubit: str
