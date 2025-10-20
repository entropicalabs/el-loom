"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from pydantic.dataclasses import dataclass
from .base_operation import Operation
from ..utilities import SingleQubitPauliEigenstate, dataclass_params


@dataclass(**dataclass_params)
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


@dataclass(**dataclass_params)
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


@dataclass(**dataclass_params)
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


@dataclass(**dataclass_params)
class Hadamard(LogicalOperation):
    """
    Describes a Hadamard gate on a logical qubit.

    Parameters
    ----------
    target_qubit : str
        Name of the target logical qubit.
    """

    target_qubit: str


@dataclass(**dataclass_params)
class Phase(LogicalOperation):
    """
    Describes a Phase gate on a logical qubit.

    Parameters
    ----------
    target_qubit : str
        Name of the target logical qubit.
    """

    target_qubit: str


@dataclass(**dataclass_params)
class PhaseInverse(LogicalOperation):
    """
    Describes an inverse Phase gate on a logical qubit.

    Parameters
    ----------
    target_qubit : str
        Name of the target logical qubit.
    """

    target_qubit: str


@dataclass(**dataclass_params)
class X(LogicalOperation):
    """
    Describes an X gate on a logical qubit.

    Parameters
    ----------
    target_qubit : str
        Name of the target logical qubit.
    """

    target_qubit: str


@dataclass(**dataclass_params)
class Y(LogicalOperation):
    """
    Describes a Y gate on a logical qubit.

    Parameters
    ----------
    target_qubit : str
        Name of the target logical qubit.
    """

    target_qubit: str


@dataclass(**dataclass_params)
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
@dataclass(**dataclass_params)
class T(LogicalOperation):
    """
    Describes a T gate on a logical qubit.

    Parameters
    ----------
    target_qubit : str
        Name of the target logical qubit.
    """

    target_qubit: str
