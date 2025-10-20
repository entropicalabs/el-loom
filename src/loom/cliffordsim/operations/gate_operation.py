"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from dataclasses import dataclass, field
from abc import abstractmethod

from .base_operation import Operation, OpType
from .controlled_operation import has_ccontrol


@dataclass
@has_ccontrol
class GateOperation(Operation):
    """
    Operations of this type perform quantum gate operations on qubits within the Engine
    during runtime.
    """

    operation_type: str = field(default=OpType.QUANTUMGATE, init=False)

    @property
    @abstractmethod
    def operating_qubit(self):
        """
        Returns a list of qubits that the gate operation acts on.
        """
        raise NotImplementedError


@dataclass
class OneQubitGateOperation(GateOperation):
    """
    A gate operation that acts on a single qubit.
    """

    target_qubit: int

    @property
    def operating_qubit(self):
        return [self.target_qubit]


@dataclass
class Identity(OneQubitGateOperation):
    """
    The Identity gate operation. It does nothing to the qubit.
    """

    name: str = field(default="Identity", init=False)


@dataclass
class Hadamard(OneQubitGateOperation):
    """
    The Hadamard gate operation. Acts on a single qubit.
    """

    name: str = field(default="Hadamard", init=False)


@dataclass
class Phase(OneQubitGateOperation):
    """
    The Phase gate operation. Acts on a single qubit.
    """

    name: str = field(default="Phase", init=False)


@dataclass
class PhaseInv(OneQubitGateOperation):
    """
    The Phase Inverse gate operation. Acts on a single qubit.
    """

    name: str = field(default="PhaseInv", init=False)


@dataclass
class X(OneQubitGateOperation):
    """
    The Pauli-X gate operation. Acts on a single qubit.
    """

    name: str = field(default="X", init=False)


@dataclass
class Z(OneQubitGateOperation):
    """
    The Pauli-Z gate operation. Acts on a single qubit.
    """

    name: str = field(default="Z", init=False)


@dataclass
class Y(OneQubitGateOperation):
    """
    The Pauli-Y gate operation. Acts on a single qubit.
    """

    name: str = field(default="Y", init=False)


@dataclass
class TwoQubitGateOperation(GateOperation):
    """
    A gate operation that acts on two qubits.
    """

    control_qubit: int
    target_qubit: int

    @property
    def operating_qubit(self):
        return [self.control_qubit, self.target_qubit]


@dataclass
class CNOT(TwoQubitGateOperation):
    """
    The CNOT gate operation. Acts on two qubits, the control and target qubits.
    """

    name: str = field(default="CNOT", init=False)


@dataclass
class CZ(TwoQubitGateOperation):
    """
    The CZ gate operation. Acts on two qubits, the control and target qubits.
    """

    name: str = field(default="CZ", init=False)


@dataclass
class CY(TwoQubitGateOperation):
    """
    The CY gate operation. Acts on two qubits, the control and target qubits.
    """

    name: str = field(default="CY", init=False)


@dataclass
class SWAP(TwoQubitGateOperation):
    """
    The SWAP gate operation. Acts on two qubits, swapping their states.
    """

    name: str = field(default="SWAP", init=False)
