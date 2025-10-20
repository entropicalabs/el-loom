"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=too-many-lines
from abc import abstractmethod
from functools import reduce
from enum import Enum
import typing
from typing import Dict, List, Tuple
from copy import deepcopy

import numpy as np

from loom.eka.utilities import ndarray_rowsum, is_tableau_valid
from ..operations.base_operation import Operation
from ..operations.gate_operation import (
    Identity,
    Hadamard,
    Phase,
    PhaseInv,
    X,
    Z,
    Y,
    CNOT,
    CY,
    CZ,
    SWAP,
)
from ..operations.measurement_operation import Measurement, Reset
from ..operations.resize_operation import AddQubit, DeleteQubit
from ..tableau import Tableau
from ..operations.datamanipulation_operation import (
    UpdateTableau,
    CreatePauliFrame,
    RecordPauliFrame,
    CreateClassicalRegister,
    RecordClassicalRegister,
)
from ..operations.classical_operation import (
    ClassicalNOT,
    ClassicalOR,
    ClassicalAND,
)
from ..operations.controlled_operation import ControlledOperation
from ..pauli_frame import PauliFrame
from ..data_store import DataStore
from ..classicalreg import ClassicalRegister
from ..exceptions import (
    InvalidTableauError,
    ClassicalRegisterError,
    ClassicalOperationError,
)


class Instruction:
    """An Instruction contains the implementation logic for a quantum operation.

    Every Instruction requires an `apply` method for Tableau-based simulations,
    and a `apply_pf` and `apply_pf_back` method for PauliFrame-based simulations.

    PauliFrame simulations can be performed by propagating a PauliFrame forward or
    backward through a circuit. The methods `apply_pf` and `apply_pf_back` are used to
    propagate the PauliFrame forward and backward respectively.

    The Instruction class is an abstract class and should not be instantiated directly.

    Raises
    ------
    NotImplementedError
        If the method is not implemented in the subclass.
    """

    @abstractmethod
    def apply(self, input_tableau: Tableau):
        """
        Abstract method that applies the quantum operation to the input Tableau.
        """
        return NotImplementedError

    # pylint: disable=unused-argument
    def apply_pf(self, input_pauliframes: List[PauliFrame]):
        """
        Abstract method that propagates the PauliFrames forward through the quantum
        operation.
        """
        return NotImplementedError

    def apply_pf_back(self, input_pauliframes: List[PauliFrame]):
        """
        Abstract method that propagates the PauliFrames backward through the quantum
        operation.
        """
        return NotImplementedError


class IdentityInstruction(Instruction):
    """
    A simple Instruction that does nothing. This will be the initial Instruction in
    a Moment. Other Instructions within the Moment will decorate this Instruction.
    """

    # pylint: disable=arguments-differ
    def apply(
        self,
        input_tableau: Tableau,
        data_store: DataStore,
        **kwargs,
    ) -> Tuple[Tableau, DataStore, Dict]:
        """
        Applies the quantum operation to the input Tableau.
        """
        # print("Applying I")
        return input_tableau, data_store, kwargs

    def apply_pf(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ) -> Tuple[List[PauliFrame], DataStore, Dict]:
        """
        Propagates the PauliFrames forward through the quantum operation.
        """
        return input_pauliframes, data_store, kwargs

    def apply_pf_back(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ) -> Tuple[List[PauliFrame], DataStore, Dict]:
        """
        Propagates the PauliFrames backward through the quantum operation.
        """
        return input_pauliframes, data_store, kwargs


class InstructionDecorator(Instruction):
    """An InstructionDecorator is a wrapper around an Instruction that adds additional
    functionality to the Instruction. The wrapped Instruction is then used to apply
    quantum operations to a Tableau, or PauliFrame, representation of a quantum state.

    Parameters
    ----------
    instruction: Instruction
        The instruction that is being decorated.
    input_operation: :class:`loom.cliffordsim.operations.base_operation.Operation`
        The Operation that will be applied.

    Attributes
    ----------
    wrapped_instruction : Instruction
        The Instruction that is being decorated.
    input_operation : :class:`loom.cliffordsim.operations.base_operation.Operation`
        The quantum operation that is being applied to the quantum state.
    """

    def __init__(self, instruction: Instruction, input_operation: Operation):
        self.wrapped_instruction = instruction
        if self.__class__.__name__[:-9] != input_operation.__class__.__name__:
            raise TypeError(
                "This InstructionDecorator only accepts                 "
                f" {self.__class__.__name__[:-9]} Operation type."
            )
        self.input_operation = input_operation

    # pylint: disable=arguments-differ
    def apply(
        self,
        input_tableau: Tableau,
        data_store: DataStore,
        **kwargs,
    ) -> Tuple[Tableau, DataStore, Dict]:
        """
        Applies the quantum operation to the input Tableau.
        """
        input_tableau, data_store, kwargs = self.extra(
            input_tableau, data_store, **kwargs
        )
        return self.wrapped_instruction.apply(input_tableau, data_store, **kwargs)

    @abstractmethod
    def extra(
        self,
        input_tableau: Tableau,
        data_store: DataStore,
        **kwargs,
    ) -> Tuple[Tableau, DataStore, Dict]:
        """
        An abstract method that adds additional functionality to the `apply` method.
        """
        return NotImplementedError

    def apply_pf(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ) -> Tuple[List[PauliFrame], DataStore, Dict]:
        """
        Applies the quantum operation to the input PauliFrames.
        """
        input_pauliframes, data_store, kwargs = self.extra_pf(
            input_pauliframes, data_store, **kwargs
        )
        return self.wrapped_instruction.apply_pf(
            input_pauliframes, data_store, **kwargs
        )

    @abstractmethod
    def extra_pf(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ) -> Tuple[List[PauliFrame], DataStore, Dict]:
        """
        An abstract method that adds additional functionality to the `apply_pf` method.
        """
        return NotImplementedError

    def apply_pf_back(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ) -> Tuple[List[PauliFrame], DataStore, Dict]:
        """
        Applies the quantum operation to the input PauliFrames. (For backward
        propagation.)
        """
        input_pauliframes, data_store, kwargs = self.extra_pf_back(
            input_pauliframes, data_store, **kwargs
        )
        return self.wrapped_instruction.apply_pf_back(
            input_pauliframes, data_store, **kwargs
        )

    @abstractmethod
    def extra_pf_back(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ) -> Tuple[List[PauliFrame], DataStore, Dict]:
        """
        An abstract method that adds additional functionality to the `apply_pf_back`
        method.
        """
        return NotImplementedError


class IdentityDecorator(InstructionDecorator):
    """The IdentityDecorator returns the input Tableau or PauliFrames without any
    modifications.

    Parameters
    ----------
    instruction: Instruction
        The instruction that is being decorated.
    input_operation: :class:`loom.cliffordsim.operations.Identity`
        The Identity Operation that will be applied.

    Attributes
    ----------
    qubit : int
        The qubit to apply the Identity gate to.
    """

    def __init__(self, instruction: Instruction, input_operation: Identity):
        super().__init__(instruction, input_operation)

    def extra(
        self,
        input_tableau: Tableau,
        data_store: DataStore,
        **kwargs,
    ):
        return input_tableau, data_store, kwargs

    def extra_pf(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        return input_pauliframes, data_store, kwargs

    def extra_pf_back(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        return input_pauliframes, data_store, kwargs


class HadamardDecorator(InstructionDecorator):
    """The HadamardDecorator applies a Hadamard gate to a qubit in a quantum state.

    Parameters
    ----------
    instruction: Instruction
        The instruction that is being decorated.
    input_operation: :class:`loom.cliffordsim.operations.Hadamard`
        The Hadamard Operation that will be applied.

    Attributes
    ----------
    qubit : int
        The qubit to apply the Hadamard gate to.
    """

    def __init__(self, instruction: Instruction, input_operation: Hadamard):
        super().__init__(instruction, input_operation)
        self.qubit = input_operation.target_qubit

    def extra(
        self,
        input_tableau: Tableau,
        data_store: DataStore,
        **kwargs,
    ):
        self.apply_hadamard(input_tableau, self.qubit)

        return input_tableau, data_store, kwargs

    @staticmethod
    def apply_hadamard(input_tableau: Tableau, qubit: int):
        """Applies a hadamard gate on a specific qubit.

        Parameters
        ----------
        input_tableau : Tableau
            The input representation of the quantum state.
        qubit : int
            The qubit to apply the Hadamard gate to.
        """
        # print("Applying H")
        input_tableau.r ^= input_tableau.x[:, qubit] & input_tableau.z[:, qubit]
        # swap x and z without using temp
        input_tableau.x[:, qubit] ^= input_tableau.z[:, qubit]
        input_tableau.z[:, qubit] ^= input_tableau.x[:, qubit]
        input_tableau.x[:, qubit] ^= input_tableau.z[:, qubit]

    def extra_pf(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        input_pauliframes = self.apply_hadamard_pf(input_pauliframes, self.qubit)
        return input_pauliframes, data_store, kwargs

    def extra_pf_back(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        """
        The Hadamard gate is its own inverse, so applying it again will revert the
        PauliFrame to its original state.
        """
        input_pauliframes = self.apply_hadamard_pf(input_pauliframes, self.qubit)
        return input_pauliframes, data_store, kwargs

    @staticmethod
    def apply_hadamard_pf(
        input_pauliframes: List[PauliFrame], qubit: int
    ) -> List[PauliFrame]:
        """
        Applies a Hadamard gate on a specific qubit in a list of PauliFrames.
        """
        # swap z and x values
        for input_pauliframe in input_pauliframes:
            input_pauliframe.z[qubit], input_pauliframe.x[qubit] = (
                input_pauliframe.x[qubit],
                input_pauliframe.z[qubit],
            )
        return input_pauliframes


class PhaseDecorator(InstructionDecorator):
    """The PhaseDecorator applies a phase gate, Z(+pi/2), to a qubit in a quantum state.

    Parameters
    ----------
    instruction: Instruction
        The instruction that is being decorated.
    input_operation: :class:`loom.cliffordsim.operations.Phase`
        The Phase Operation that will be applied.

    Attributes
    ----------
    qubit : int
        The qubit to apply the phase gate to.
    """

    def __init__(self, instruction: Instruction, input_operation: Phase):
        super().__init__(instruction, input_operation)
        self.qubit = input_operation.target_qubit

    def extra(
        self,
        input_tableau: Tableau,
        data_store: DataStore,
        **kwargs,
    ):
        self.apply_phase(input_tableau, self.qubit)
        return input_tableau, data_store, kwargs

    @staticmethod
    def apply_phase(input_tableau: Tableau, qubit: int):
        """Applies a phase gate on a specific qubit.

        Parameters
        ----------
        input_tableau : Tableau
            The input representation of the quantum state.
        qubit : int
            The qubit to apply the phase gate to.
        """
        input_tableau.r ^= input_tableau.x[:, qubit] & input_tableau.z[:, qubit]
        input_tableau.z[:, qubit] ^= input_tableau.x[:, qubit]

    def extra_pf(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        input_pauliframes = self.apply_phase_pf(input_pauliframes, self.qubit)

        return input_pauliframes, data_store, kwargs

    def extra_pf_back(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        """
        The inverse of the phase gate is the phase inverse gate, which can be applied
        using the same transformation as the phase gate in the PauliFrame
        representation.
        """
        input_pauliframes = self.apply_phase_pf(input_pauliframes, self.qubit)

        return input_pauliframes, data_store, kwargs

    @staticmethod
    def apply_phase_pf(
        input_pauliframes: List[PauliFrame], qubit: int
    ) -> List[PauliFrame]:
        """
        Applies a phase gate on a specific qubit in a list of PauliFrames.
        """
        # Z<->Z, X<->Y
        for input_pauliframe in input_pauliframes:
            input_pauliframe.z[qubit] ^= input_pauliframe.x[qubit]
        return input_pauliframes


class PhaseInvDecorator(InstructionDecorator):
    """The PhaseInvDecorator applies the inverse of a phase gate, Z(-pi/2), to a qubit
    in a quantum state.

    Parameters
    ----------
    instruction: Instruction
        The instruction that is being decorated.
    input_operation: :class:`loom.cliffordsim.operations.PhaseInv`
        The PhaseInv Operation that will be applied.

    Attributes
    ----------
    qubit : int
        The qubit to apply the inverse of the phase gate to.
    """

    def __init__(self, instruction: Instruction, input_operation: PhaseInv):
        super().__init__(instruction, input_operation)
        self.qubit = input_operation.target_qubit

    @staticmethod
    def apply_phase_inv(input_tableau: Tableau, qubit: int):
        """Applies the inverse of a phase gate on a specific qubit.

        Parameters
        ----------
        input_tableau : Tableau
            The input representation of the quantum state.
        qubit : int
            The qubit to apply the inverse of the phase gate to.
        """
        # X-> -Y -> -X -> Y -> X
        input_tableau.r ^= input_tableau.x[:, qubit] & (~input_tableau.z[:, qubit])
        input_tableau.z[:, qubit] ^= input_tableau.x[:, qubit]

    def extra(
        self,
        input_tableau: Tableau,
        data_store: DataStore,
        **kwargs,
    ):
        self.apply_phase_inv(input_tableau, self.qubit)
        return input_tableau, data_store, kwargs

    def extra_pf(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        input_pauliframes = PhaseDecorator.apply_phase_pf(input_pauliframes, self.qubit)

        return input_pauliframes, data_store, kwargs

    def extra_pf_back(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        input_pauliframes = PhaseDecorator.apply_phase_pf(input_pauliframes, self.qubit)

        return input_pauliframes, data_store, kwargs


class XDecorator(InstructionDecorator):
    """The XDecorator applies an X gate to a qubit in a quantum state.

    Parameters
    ----------
    instruction: Instruction
        The instruction that is being decorated.
    input_operation: :class:`loom.cliffordsim.operations.X`
        The X Operation that will be applied.

    Attributes
    ----------
    qubit : int
        The qubit to apply the X gate to.
    """

    def __init__(self, instruction: Instruction, input_operation: X):
        super().__init__(instruction, input_operation)
        self.qubit = input_operation.target_qubit

    def extra(
        self,
        input_tableau: Tableau,
        data_store: DataStore,
        **kwargs,
    ):
        # print("Applying X")
        self.apply_x(input_tableau, self.qubit)
        return input_tableau, data_store, kwargs

    @staticmethod
    def apply_x(input_tableau: Tableau, qubit: int):
        """Applies an X gate on a specific qubit.

        Parameters
        ----------
        input_tableau : Tableau
            The input representation of the quantum state.
        qubit : int
            The qubit to apply the X gate to.
        """
        input_tableau.r ^= input_tableau.z[:, qubit]

    def extra_pf(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        return input_pauliframes, data_store, kwargs

    def extra_pf_back(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        return input_pauliframes, data_store, kwargs


class ZDecorator(InstructionDecorator):
    """The ZDecorator applies a Z gate to a qubit in a quantum state.

    Parameters
    ----------
    instruction: Instruction
        The instruction that is being decorated.
    input_operation: :class:`loom.cliffordsim.operations.Z`
        The Z Operation that will be applied.

    Attributes
    ----------
    qubit : int
        The qubit to apply the Z gate to.
    """

    def __init__(self, instruction: Instruction, input_operation: Z):
        super().__init__(instruction, input_operation)
        self.qubit = input_operation.target_qubit

    def extra(
        self,
        input_tableau: Tableau,
        data_store: DataStore,
        **kwargs,
    ):
        # print("Applying Z")
        input_tableau.r ^= input_tableau.x[:, self.qubit]
        return input_tableau, data_store, kwargs

    def extra_pf(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        return input_pauliframes, data_store, kwargs

    def extra_pf_back(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        return input_pauliframes, data_store, kwargs


class YDecorator(InstructionDecorator):
    """The YDecorator applies a Y gate to a qubit in a quantum state.

    Parameters
    ----------
    instruction: Instruction
        The instruction that is being decorated.
    input_operation: :class:`loom.cliffordsim.operations.Y`
        The Y Operation that will be applied.

    Attributes
    ----------
    qubit : int
        The qubit to apply the Y gate to.
    """

    def __init__(self, instruction: Instruction, input_operation: Y):
        super().__init__(instruction, input_operation)
        self.qubit = input_operation.target_qubit

    def extra(
        self,
        input_tableau: Tableau,
        data_store: DataStore,
        **kwargs,
    ):
        # print("Applying Y")
        input_tableau.r ^= (
            input_tableau.x[:, self.qubit] ^ input_tableau.z[:, self.qubit]
        )
        return input_tableau, data_store, kwargs

    def extra_pf(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        return input_pauliframes, data_store, kwargs

    def extra_pf_back(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        return input_pauliframes, data_store, kwargs


class CNOTDecorator(InstructionDecorator):
    """The CNOTDecorator applies a CNOT, Controlled-NOT, gate across 2 qubits, a
    control qubit and a target qubit.

    Parameters
    ----------
    instruction: Instruction
        The instruction that is being decorated.
    input_operation: :class:`loom.cliffordsim.operations.CNOT`
        The CNOT Operation that will be applied.

    Attributes
    ----------
    c_qubit : int
        The control qubit that controls the CNOT gate.
    t_qubit : int
        The target qubit where the NOT/X gate will be applied if the control qubit is 1.
    """

    def __init__(self, instruction: Instruction, input_operation: CNOT):
        super().__init__(instruction, input_operation)
        self.c_qubit = input_operation.control_qubit
        self.t_qubit = input_operation.target_qubit

    def extra(
        self,
        input_tableau: Tableau,
        data_store: DataStore,
        **kwargs,
    ):
        # print("Applying CNOT")
        input_tableau.r ^= (
            input_tableau.x[:, self.c_qubit]
            & input_tableau.z[:, self.t_qubit]
            & ~(input_tableau.x[:, self.t_qubit] ^ input_tableau.z[:, self.c_qubit])
        )

        input_tableau.x[:, self.t_qubit] ^= input_tableau.x[:, self.c_qubit]
        input_tableau.z[:, self.c_qubit] ^= input_tableau.z[:, self.t_qubit]
        return input_tableau, data_store, kwargs

    def extra_pf(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        """
        Applies a CNOT gate to a list of PauliFrames.
        """
        input_pauliframes = self.apply_cnot_pf(
            input_pauliframes, self.c_qubit, self.t_qubit
        )

        return input_pauliframes, data_store, kwargs

    @staticmethod
    def apply_cnot_pf(
        input_pauliframes: List[PauliFrame], c_qubit: int, t_qubit: int
    ) -> List[PauliFrame]:
        """
        A staticmethod that applies a CNOT gate to a list of PauliFrames.
        """
        for input_pauliframe in input_pauliframes:
            # XI -> XX
            input_pauliframe.x[t_qubit] ^= input_pauliframe.x[c_qubit]
            # IZ -> ZZ
            input_pauliframe.z[c_qubit] ^= input_pauliframe.z[t_qubit]

        return input_pauliframes

    def extra_pf_back(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        input_pauliframes = self.apply_cnot_pf(
            input_pauliframes, self.c_qubit, self.t_qubit
        )

        return input_pauliframes, data_store, kwargs


class CYDecorator(InstructionDecorator):
    """The CYDecorator applies a CY, Controlled-Y, gate across 2 qubits, a control qubit
    and a target qubit.

    Parameters
    ----------
    instruction: Instruction
        The instruction that is being decorated.
    input_operation: :class:`loom.cliffordsim.operations.CY`
        The CY Operation that will be applied.

    Attributes
    ----------
    c_qubit : int
        The control qubit that controls the CY gate.
    t_qubit : int
        The target qubit where the Y gate will be applied if the control qubit is 1.
    """

    def __init__(self, instruction: Instruction, input_operation: CY):
        super().__init__(instruction, input_operation)
        self.c_qubit = input_operation.control_qubit
        self.t_qubit = input_operation.target_qubit

    def extra(
        self,
        input_tableau: Tableau,
        data_store: DataStore,
        **kwargs,
    ):
        input_tableau.r ^= (
            input_tableau.x[:, self.c_qubit]
            & (input_tableau.x[:, self.t_qubit] ^ input_tableau.z[:, self.t_qubit])
            & ~(input_tableau.z[:, self.c_qubit] ^ input_tableau.z[:, self.t_qubit])
        )

        # IX->ZX, IZ->ZZ but IY->IY
        input_tableau.z[:, self.c_qubit] ^= (
            input_tableau.x[:, self.t_qubit] ^ input_tableau.z[:, self.t_qubit]
        )

        # XI->XY
        input_tableau.x[:, self.t_qubit] ^= input_tableau.x[:, self.c_qubit]
        input_tableau.z[:, self.t_qubit] ^= input_tableau.x[:, self.c_qubit]
        return input_tableau, data_store, kwargs

    def extra_pf(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        """
        Applies a CY gate to a list of PauliFrames.
        """
        input_pauliframes = self.apply_cy_pf(
            input_pauliframes, self.c_qubit, self.t_qubit
        )
        return input_pauliframes, data_store, kwargs

    @staticmethod
    def apply_cy_pf(
        input_pauliframes: List[PauliFrame], c_qubit: int, t_qubit: int
    ) -> List[PauliFrame]:
        """
        A staticmethod that applies a CY gate to a list of PauliFrames.
        """
        for input_pauliframe in input_pauliframes:
            input_pauliframe.z[c_qubit] ^= (
                input_pauliframe.x[t_qubit] ^ input_pauliframe.z[t_qubit]
            )
            input_pauliframe.x[t_qubit] ^= input_pauliframe.x[c_qubit]
            input_pauliframe.z[t_qubit] ^= input_pauliframe.x[c_qubit]
        return input_pauliframes

    def extra_pf_back(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        self.apply_cy_pf(input_pauliframes, self.c_qubit, self.t_qubit)

        return input_pauliframes, data_store, kwargs


class CZDecorator(InstructionDecorator):
    """The CZDecorator applies a CZ, Controlled-Z, gate across 2 qubits, a control qubit
    and a target qubit.

    Parameters
    ----------
    instruction: Instruction
        The instruction that is being decorated.
    input_operation: :class:`loom.cliffordsim.operations.CZ`
        The CZ Operation that will be applied.

    Attributes
    ----------
    c_qubit : int
        The control qubit that controls the CZ gate.
    t_qubit : int
        The target qubit where the Z gate will be applied if the control qubit is 1.
    """

    def __init__(self, instruction: Instruction, input_operation: CZ):
        super().__init__(instruction, input_operation)
        self.c_qubit = input_operation.control_qubit
        self.t_qubit = input_operation.target_qubit

    def extra(
        self,
        input_tableau: Tableau,
        data_store: DataStore,
        **kwargs,
    ):
        # print("Applying CZ")
        input_tableau.r ^= (
            input_tableau.x[:, self.c_qubit]
            & input_tableau.x[:, self.t_qubit]
            & ~(input_tableau.z[:, self.t_qubit] ^ input_tableau.z[:, self.c_qubit])
        )

        input_tableau.z[:, self.t_qubit] ^= input_tableau.x[:, self.c_qubit]
        input_tableau.z[:, self.c_qubit] ^= input_tableau.x[:, self.t_qubit]
        return input_tableau, data_store, kwargs

    def extra_pf(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        """
        Applies a CZ gate to a list of PauliFrames.
        """
        input_pauliframes = self.apply_cz_pf(
            input_pauliframes, self.c_qubit, self.t_qubit
        )

        return input_pauliframes, data_store, kwargs

    @staticmethod
    def apply_cz_pf(
        input_pauliframes: List[PauliFrame], c_qubit: int, t_qubit: int
    ) -> List[PauliFrame]:
        """
        A staticmethod that applies a CZ gate to a list of PauliFrames.
        """
        for input_pauliframe in input_pauliframes:
            input_pauliframe.z[t_qubit] ^= input_pauliframe.x[c_qubit]
            input_pauliframe.z[c_qubit] ^= input_pauliframe.x[t_qubit]
        return input_pauliframes

    def extra_pf_back(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        self.apply_cz_pf(input_pauliframes, self.c_qubit, self.t_qubit)

        return input_pauliframes, data_store, kwargs


class SWAPDecorator(InstructionDecorator):
    """The SWAPDecorator applies a SWAP gate across 2 qubits, labelled as control
    qubit and target qubit.

    Parameters
    ----------
    instruction: Instruction
        The instruction that is being decorated.
    input_operation: :class:`loom.cliffordsim.operations.SWAP`
        The SWAP Operation that will be applied.

    Attributes
    ----------
    c_qubit : int
        The qubit that will have its state "swapped" with the target qubit.
    t_qubit : int
        The qubit that will have its state "swapped" with the control qubit.
    """

    def __init__(self, instruction: Instruction, input_operation: SWAP):
        super().__init__(instruction, input_operation)
        self.c_qubit = input_operation.control_qubit
        self.t_qubit = input_operation.target_qubit

    def extra(
        self,
        input_tableau: Tableau,
        data_store: DataStore,
        **kwargs,
    ):
        # print("Applying SWAP")

        # swap the columns of the 2 qubits for x and z arrays
        input_tableau.x[:, [self.t_qubit, self.c_qubit]] = input_tableau.x[
            :, [self.c_qubit, self.t_qubit]
        ]
        input_tableau.z[:, [self.t_qubit, self.c_qubit]] = input_tableau.z[
            :, [self.c_qubit, self.t_qubit]
        ]

        return input_tableau, data_store, kwargs

    def extra_pf(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        """
        Applies a SWAP gate to a list of PauliFrames.
        """
        self.apply_swap_pf(input_pauliframes, self.c_qubit, self.t_qubit)

        return input_pauliframes, data_store, kwargs

    @staticmethod
    def apply_swap_pf(
        input_pauliframes: List[PauliFrame], c_qubit: int, t_qubit: int
    ) -> List[PauliFrame]:
        """
        A staticmethod that applies a SWAP gate to a list of PauliFrames.
        """
        for input_pauliframe in input_pauliframes:
            # swap Z and X
            input_pauliframe.z[c_qubit], input_pauliframe.z[t_qubit] = (
                input_pauliframe.z[t_qubit],
                input_pauliframe.z[c_qubit],
            )

            input_pauliframe.x[c_qubit], input_pauliframe.x[t_qubit] = (
                input_pauliframe.x[t_qubit],
                input_pauliframe.x[c_qubit],
            )

        return input_pauliframes

    def extra_pf_back(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        """
        Applies a SWAP gate to a list of PauliFrames. (For backward propagation.)
        The SWAP gate is its own inverse, so applying it again will revert the
        PauliFrame to its original state.
        """
        input_pauliframes = self.apply_swap_pf(
            input_pauliframes, self.c_qubit, self.t_qubit
        )

        return input_pauliframes, data_store, kwargs


class ClassicalBitDecorator(InstructionDecorator):  # pylint: disable=abstract-method
    """
    The ClassicalBitDecorator contains private methods to get information about a bit
    in a classical register, these methods will be used by its subclasses.

    Any InstructionDecorator that involves classical bits should inherit from this
    class.
    """

    def _get_bit_info(self, classical_register, bit_order, bit_id):
        """
        Get information of a bit in the classical register defined either by the
        ordering within the classical register or its bit ID.
        """
        bit_value = None
        if bit_order is not None:
            bit_value, bit_order, bit_id = self._get_bit_info_w_bit_order(
                classical_register, bit_order
            )
        elif bit_id is not None:
            bit_value, bit_order, bit_id = self._get_bit_info_w_bit_id(
                classical_register, bit_id
            )

        return bit_value, bit_order, bit_id

    def _get_bit_info_w_bit_order(self, classical_register, bit_order):
        """Get information of a bit in the classical register defined by its ordering
        within the classical register.
        """
        try:
            bit_id, bit_value = classical_register.reg[bit_order]

            return bit_value, bit_order, bit_id

        except Exception as exc:
            raise ClassicalOperationError(
                f"The selected bit_order, {bit_order}, does not exist or is not valid "
                f"in {classical_register.name}. There are only "
                f"{len(classical_register.reg)} bits in the register."
            ) from exc

    def _get_bit_info_w_bit_id(self, classical_register, bit_id):
        """Get information of a bit in the classical register defined by its bit ID."""
        try:
            bit_value = classical_register.id_bit_reg[bit_id]
            for each_order, each_pair in enumerate(classical_register.reg):
                if each_pair[0] == bit_id:
                    bit_order = each_order
                    break
            return bit_value, bit_order, bit_id

        except Exception as exc:
            raise ClassicalOperationError(
                f"The selected bit_id, {bit_id}, does not exist or is not valid in "
                f"{classical_register.name}. The only available bit IDs in this "
                f"register are: {classical_register.bit_ids}"
            ) from exc

    def _select_register(self, input_registry: dict, register_name: str):
        """Get a classical register from the input registry with the given name."""
        try:
            select_register = input_registry[register_name]

            return select_register

        except Exception as exc:
            raise ClassicalRegisterError(
                "An error has occured when trying to select the "
                f"register, {register_name}."
            ) from exc


class MeasurementDecorator(ClassicalBitDecorator):
    """The MeasurementDecorator applies a measurement operation to a qubit in a quantum
    state. It also records the measurement result in a classical register if it exists.

    Parameters
    ----------
    instruction: Instruction
        The instruction that is being decorated.
    input_operation: :class:`loom.cliffordsim.operations.MeasurementOperation`
        The measurement operation that will be applied.

    Attributes
    ----------
    qubit : int
        The qubit to be measured.
    measurement_id : str
        The label of the measurement operation.
    bias : float
        The bias of the measurement result.
    reg_name : str
        The name of the classical register where the measurement result will be stored.
    bit_order : int
        The order of the bit in the classical register.
    bit_id : int
        The ID of the bit in the classical register.
    """

    def __init__(self, instruction: Instruction, input_operation: Measurement):
        super().__init__(instruction, input_operation)
        self.qubit = input_operation.target_qubit
        self.measurement_id = input_operation.label
        self.bias = input_operation.bias
        self.basis = input_operation.basis
        self.reg_name = input_operation.reg_name
        self.bit_order = input_operation.bit_order
        self.bit_id = input_operation.bit_id

    def extra(
        self,
        input_tableau: Tableau,
        data_store: DataStore,
        **kwargs,
    ):
        m_qubit = self.qubit

        # Apply any necessary operations to the tableau before measurement to change
        # the basis of the measurement.
        if self.basis == "X":
            HadamardDecorator.apply_hadamard(input_tableau, m_qubit)
        elif self.basis == "Y":
            # To map |+i> to |0> and |-i> to |1>, we need to a PhaseInv and a Hadamard
            # gate.
            PhaseInvDecorator.apply_phase_inv(input_tableau, m_qubit)
            HadamardDecorator.apply_hadamard(input_tableau, m_qubit)
        elif self.basis == "Z":
            pass
        else:
            raise ValueError(
                f"Invalid measurement basis: {self.basis}. Must be 'X', 'Y', or 'Z'."
            )

        # Find pivot index
        is_result_random, measurement_result = self.measure(
            input_tableau, m_qubit, self.bias
        )

        # Add measured results to DataStore
        data_store.record_measurements(
            self.measurement_id, measurement_result, is_result_random
        )

        # Write to a Classical Register if it exists
        if self.reg_name is not None:
            output_reg = self._select_register(kwargs["registry"], self.reg_name)
            bit_value, bit_order, bit_id = self._get_bit_info(
                output_reg, self.bit_order, self.bit_id
            )

            # Write output to register
            bit_value = measurement_result
            kwargs["registry"][self.reg_name].reg[bit_order] = (bit_id, bit_value)

        return input_tableau, data_store, kwargs

    @classmethod
    def measure(
        cls, input_tableau: Tableau, m_qubit: int, bias: float = 0.5
    ) -> tuple[bool, int]:
        """
        Determines the measurement result of a qubit and whether it is random.
        It also updates the input tableau if the measurement result is random.

        Parameters
        ----------
        input_tableau : Tableau
            The input tableau of the quantum state.
        m_qubit : int
            The qubit to be measured.
        bias : float, optional
            The bias of the result, by default 0.5

        Returns
        -------
        tuple[bool, int]
            A tuple containing whether the measurement result is random and what its
            value is.
        """
        p = cls.find_pivot(input_tableau, m_qubit)
        # Find if m result shall be random
        is_result_random = cls.is_result_random(input_tableau, m_qubit, p)
        # Deduce measurement result
        measurement_result = (
            cls.deduce_mresult_probabilistic(input_tableau, m_qubit, p, bias)
            if is_result_random
            else cls.deduce_mresult_deterministic(input_tableau, m_qubit)
        )

        return is_result_random, measurement_result

    @staticmethod
    def find_pivot(input_tableau: Tableau, m_qubit: int) -> int:
        """
        Finds the pivot point for the measurement operation.

        Parameters
        ----------
        input_tableau : Tableau
            The input tableau of the quantum state.
        m_qubit : int
            Qubit to be measured

        Returns
        -------
        int
            Pivot index
        """
        # nqubits is added since it was sliced out of the search
        return (
            np.argmax(input_tableau.x[input_tableau.nqubits :, m_qubit] == 1)
            + input_tableau.nqubits
        )

    @staticmethod
    def is_result_random(input_tableau: Tableau, m_qubit: int, p: int) -> bool:
        """
        Checks if the measurement of a qubit, will result in a
        random outcome. The pivot point has to also be given.

        Parameters
        ----------
        input_tableau : Tableau
            The input tableau of the quantum state.
        m_qubit : int
            Qubit to be measured
        p : int
            Pivot index

        Returns
        -------
        bool
            Whether the result will be random or not.
        """
        return input_tableau.x[p, m_qubit] == 1

    @staticmethod
    def deduce_mresult_deterministic(
        input_tableau: Tableau,
        m_qubit: int,
    ) -> typing.Literal[0, 1]:
        """
        Performs measurement operation and returns the result.

        Parameters
        ----------
        input_tableau : Tableau
            The input tableau of the quantum state.
        m_qubit : int
            Qubit to be measured

        Returns
        -------
        typing.Literal[0, 1]
            Result of the measurement operation.
        """

        nqubits = input_tableau.nqubits
        # Set scratch row to all zeros
        input_tableau.scratch_row[:] = np.zeros_like(input_tableau.scratch_row)
        # Apply rowsum appropriately to get result in scratch row
        input_tableau.tableau_w_scratch[:, :] = reduce(
            lambda tab, idx: (
                ndarray_rowsum(tab, -1, idx + nqubits)
                if input_tableau.x[idx, m_qubit] == 1
                else tab
            ),
            range(nqubits),
            input_tableau.tableau_w_scratch,
        )

        # obtain measurement result
        measurement_result = input_tableau.scratch_row[-1]
        return measurement_result

    @staticmethod
    def deduce_mresult_probabilistic(
        input_tableau: Tableau, m_qubit: int, p: int, bias: float = 0.5
    ) -> typing.Literal[0, 1]:
        """
        Performs measurement operation and returns the result for a random
        measurement.

        Parameters
        ----------
        input_tableau : Tableau
            The input tableau of the quantum state.
        m_qubit : int
            Qubit to be measured
        p : int
            Pivot index
        bias : float, optional
            The bias of the result, by default 0.5
            If this is set to 0 then the result will always be 0.

        Returns
        -------
        typing.Literal[0, 1]
            Result of the measurement operation.
        """
        nqubits = input_tableau.nqubits
        # Firstly, apply rowsum on the appropriate indices
        idcs_wo_p = np.concatenate(
            [
                np.arange(p - nqubits),
                np.arange(p - nqubits + 1, p),
                np.arange(p + 1, 2 * nqubits),
            ]
        )
        # Apply rowsum only if x[idx, m_qubit] == 1
        input_tableau.tableau_w_scratch[:, :] = reduce(
            lambda tab, idx: (
                ndarray_rowsum(tab, idx, p)
                if input_tableau.x[idx, m_qubit] == 1
                else tab
            ),
            idcs_wo_p,
            input_tableau.tableau_w_scratch,
        )

        # Secondly modify line p-n
        input_tableau.tableau[p - input_tableau.nqubits, :] = input_tableau.tableau[
            p, :
        ]

        # Third modify line p, to make it into ±III...Z...I
        input_tableau.tableau[p, :] = np.zeros_like(input_tableau.tableau[p, :])
        input_tableau.z[p, m_qubit] = 1
        # Obtain measurement result
        measurement_result = 0
        if input_tableau.rand_gen.random() < bias:
            # r has be set to 0 already, so flip it
            input_tableau.r[p] = 1
            measurement_result = 1
        return measurement_result

    def extra_pf(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        input_pauliframes, flip_results = self.apply_meas_pf(
            input_pauliframes, self.qubit, self.basis
        )

        data_store.record_measurement_from_pauliframes(
            self.measurement_id, flip_results, input_pauliframes
        )

        return input_pauliframes, data_store, kwargs

    @staticmethod
    def apply_meas_pf(
        input_pauliframes: List[PauliFrame], qubit: int, basis: str
    ) -> tuple[List[PauliFrame], List[int]]:
        """
        Applies a measurement operation to a list of PauliFrames and returns the
        modified PauliFrames along with the measurement results.
        """

        flip_results = []
        if basis == "Z":
            flip_results = [
                1 if input_pauliframe.x[qubit] == 1 else 0
                for input_pauliframe in input_pauliframes
            ]
        elif basis == "X":
            flip_results = [
                1 if input_pauliframe.z[qubit] == 1 else 0
                for input_pauliframe in input_pauliframes
            ]
        elif basis == "Y":
            flip_results = [
                1 if (input_pauliframe.x[qubit] ^ input_pauliframe.z[qubit]) == 1 else 0
                for input_pauliframe in input_pauliframes
            ]

        return input_pauliframes, flip_results

    def extra_pf_back(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        """
        Note that changes that causes the pauliframes to interact with Datastore
        when propagating forward, are not properly implemented here for the backward
        pass. Basically ignore the problem until we have a better solution.
        """
        input_pauliframes, _ = self.apply_meas_pf(
            input_pauliframes, self.qubit, self.basis
        )

        return input_pauliframes, data_store, kwargs


class AddQubitDecorator(InstructionDecorator):
    """The AddQubitDecorator adds a qubit to the quantum state. If the index exists in
    the current quantum state, the new qubit will be inserted at the index while
    the existing qubits with an index larger than the inserted qubits index
    will have their index increased by 1.

    Parameters
    ----------
    instruction: Instruction
        The instruction that is being decorated.
    input_operation: AddQubit
        The AddQubit Operation that will be applied.

    Attributes
    ----------
    index : int
        The index of the qubit to be added.
    """

    def __init__(self, instruction: Instruction, input_operation: AddQubit):
        super().__init__(instruction, input_operation)
        self.index = input_operation.target_qubit

    def extra(
        self,
        input_tableau: Tableau,
        data_store: DataStore,
        **kwargs,
    ):
        prev_nqubits = input_tableau.nqubits
        new_nqubits = prev_nqubits + 1
        # add columns
        zero_column = np.zeros(2 * prev_nqubits + 1, dtype=np.int8).reshape(
            2 * prev_nqubits + 1, 1
        )

        tab = input_tableau.tableau_w_scratch

        input_tableau.tableau_w_scratch = np.concatenate(
            [
                tab[:, : self.index],
                zero_column,
                tab[:, self.index : prev_nqubits + self.index],
                zero_column,
                tab[:, prev_nqubits + self.index :],
            ],
            axis=1,
        )

        # add rows

        zero_row = np.zeros(2 * new_nqubits + 1, dtype=np.int8).reshape(
            1, 2 * new_nqubits + 1
        )

        tab = input_tableau.tableau_w_scratch
        input_tableau.tableau_w_scratch = np.concatenate(
            [
                tab[: self.index, :],
                zero_row,
                tab[self.index : prev_nqubits + self.index, :],
                zero_row,
                tab[prev_nqubits + self.index :, :],
            ],
            axis=0,
        )

        # set the destabilizer of this qubit to X
        input_tableau.tableau_w_scratch[self.index, self.index] = 1
        # set the stabilizer of this qubit to Z
        input_tableau.tableau_w_scratch[
            new_nqubits + self.index, new_nqubits + self.index
        ] = 1

        input_tableau.update()

        return input_tableau, data_store, kwargs

    def extra_pf(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        input_pauliframes = self.apply_add_qubit_pf(input_pauliframes, self.index)
        return input_pauliframes, data_store, kwargs

    @staticmethod
    def apply_add_qubit_pf(
        input_pauliframes: List[PauliFrame], index: int
    ) -> List[PauliFrame]:
        """
        Applies an AddQubit operation to a list of PauliFrames.
        A staticmethod that applies an AddQubit operation to a list of PauliFrames.
        """
        for input_pauliframe in input_pauliframes:
            # add 0 to the new index
            input_pauliframe.z = np.insert(input_pauliframe.z, index, 0)
            input_pauliframe.x = np.insert(input_pauliframe.x, index, 0)
        return input_pauliframes

    def extra_pf_back(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        # remove the qubit at the index since it's backwards Addition
        input_pauliframes = DeleteQubitDecorator.apply_delete_qubit_pf(
            input_pauliframes, self.index
        )

        return input_pauliframes, data_store, kwargs


class DeleteQubitDecorator(InstructionDecorator):
    """The DeleteQubitDecorator deletes a qubit from the quantum state. If the index
    exists in the current quantum state, the qubit will be removed from the state
    while the existing qubits with an index larger than the deleted qubits index
    will have their index decreased by 1.

    Parameters
    ----------
    instruction: Instruction
        The instruction that is being decorated.
    input_operation: DeleteQubit
        The DeleteQubit Operation that will be applied.

    Attributes
    ----------
    qubit : int
        The qubit to be deleted.
    """

    def __init__(self, instruction: Instruction, input_operation: DeleteQubit):
        super().__init__(instruction, input_operation)
        self.qubit = input_operation.target_qubit

    def extra(
        self,
        input_tableau: Tableau,
        data_store: DataStore,
        **kwargs,
    ):
        # find pivot
        p = MeasurementDecorator.find_pivot(input_tableau, self.qubit)

        # find if result shall be random
        is_result_random = MeasurementDecorator.is_result_random(
            input_tableau, self.qubit, p
        )

        if not is_result_random:
            # make it random by applying an H gate
            HadamardDecorator.apply_hadamard(input_tableau, self.qubit)

            # find pivot again
            p = MeasurementDecorator.find_pivot(input_tableau, self.qubit)

            # find if result shall be random
            is_result_random = MeasurementDecorator.is_result_random(
                input_tableau, self.qubit, p
            )

            assert is_result_random, "The measurement result should be random."

        # We bias the measurement to come out positive such that we can safely delete
        # the row containing the +Z_i operator.
        # If we allow -Z_i as a result, any instance of other operator containing the
        # Z_i operator should have its sign flipped when the row is deleted.
        MeasurementDecorator.deduce_mresult_probabilistic(
            input_tableau, self.qubit, p, bias=0
        )

        # find which rowws to delete
        nqubits = input_tableau.nqubits
        stab_row_idx = p
        destab_row_idx = p - nqubits
        # delete the rows
        input_tableau.tableau_w_scratch = np.delete(
            input_tableau.tableau_w_scratch,
            [stab_row_idx, destab_row_idx],
            axis=0,
        )

        # delete the columns of the
        z_col = self.qubit + nqubits
        x_col = self.qubit
        input_tableau.tableau_w_scratch = np.delete(
            input_tableau.tableau_w_scratch,
            [z_col, x_col],
            axis=1,
        )

        input_tableau.update()

        return input_tableau, data_store, kwargs

    def extra_pf(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        """
        Applies a DeleteQubit operation to a list of PauliFrames.
        """
        input_pauliframes = self.apply_delete_qubit_pf(input_pauliframes, self.qubit)

        return input_pauliframes, data_store, kwargs

    @staticmethod
    def apply_delete_qubit_pf(
        input_pauliframes: List[PauliFrame], index: int
    ) -> List[PauliFrame]:
        """
        A staticmethod that applies a DeleteQubit operation to a list of PauliFrames.
        """
        for input_pauliframe in input_pauliframes:
            # Z and X entries are deleted
            input_pauliframe.z = np.delete(input_pauliframe.z, index)
            input_pauliframe.x = np.delete(input_pauliframe.x, index)
        return input_pauliframes

    def extra_pf_back(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        # Backwards deletion is addition
        input_pauliframes = AddQubitDecorator.apply_add_qubit_pf(
            input_pauliframes, self.qubit
        )
        return input_pauliframes, data_store, kwargs


class ResetDecorator(InstructionDecorator):
    """The ResetDecorator applies a reset operation to a qubit in a quantum state. The
    qubit is first measured, if the measurement result of the qubit is 1, an X gate
    will be applied to the qubit to reset it to 0.

    Parameters
    ----------
    instruction: Instruction
        The instruction that is being decorated.
    input_operation: :class:`loom.cliffordsim.operations.Reset`
        The Reset Operation that will be applied.

    Attributes
    ----------
    qubit : int
        The qubit to be reset.
    """

    def __init__(self, instruction: Instruction, input_operation: Reset):
        super().__init__(instruction, input_operation)
        self.qubit = input_operation.target_qubit
        self.state = input_operation.state

    def extra(
        self,
        input_tableau: Tableau,
        data_store: DataStore,
        **kwargs,
    ):
        # First measure the qubit
        _, mres = MeasurementDecorator.measure(input_tableau, self.qubit)

        # Figure out what corrections are needed
        mres_needed = 0
        apply_h = False
        apply_s = False
        match self.state:
            case "0":
                pass
            case "1":
                mres_needed = 1
            case "+":
                apply_h = True
            case "-":
                mres_needed, apply_h = 1, True
            case "+i":
                apply_h, apply_s = True, True
            case "-i":
                mres_needed, apply_h, apply_s = 1, True, True
            case _:
                raise RuntimeError("Incorrect state given for reset operation.")

        # Get to the state that we need to be in
        if mres != mres_needed:
            XDecorator.apply_x(input_tableau, self.qubit)
        if apply_h:
            HadamardDecorator.apply_hadamard(input_tableau, self.qubit)
        if apply_s:
            PhaseDecorator.apply_phase(input_tableau, self.qubit)

        return input_tableau, data_store, kwargs

    def extra_pf(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        """
        Applies a Reset operation to a list of PauliFrames.
        """
        input_pauliframes = self.apply_reset_pf(input_pauliframes, self.qubit)

        return input_pauliframes, data_store, kwargs

    @staticmethod
    def apply_reset_pf(
        input_pauliframes: List[PauliFrame], qubit: int
    ) -> List[PauliFrame]:
        """
        A staticmethod that applies a Reset operation to a list of PauliFrames
        """
        for input_pauliframe in input_pauliframes:
            # Z and X errors disappear
            input_pauliframe.z[qubit] = 0
            input_pauliframe.x[qubit] = 0

        return input_pauliframes

    def extra_pf_back(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        input_pauliframes = self.apply_reset_pf(input_pauliframes, self.qubit)

        return input_pauliframes, data_store, kwargs


class UpdateTableauDecorator(InstructionDecorator):
    """
    The UpdateTableauDecorator updates the tableau of the quantum state with a new
    tableau. The new tableau must be a numpy ndarray. The decorator can also validate
    the tableau before updating.

    The new tableau must be of the same shape as the current tableau. (Ignoring the
    scratch row)

    Parameters
    ----------
    instruction: Instruction
        The instruction that is being decorated.
    input_operation: UpdateTableau
        The UpdateTableau Operation that will be applied.

    Attributes
    ----------
    tableau : np.ndarray
        The new tableau to be updated.
    validate : bool
        A flag to validate the tableau before updating.
    """

    def __init__(self, instruction: Instruction, input_operation: UpdateTableau):
        super().__init__(instruction, input_operation)
        self.tableau = input_operation.tableau
        self.validate = input_operation.validate

    def extra(
        self,
        input_tableau: Tableau,
        data_store: DataStore,
        **kwargs,
    ):
        if not isinstance(self.tableau, np.ndarray):
            raise InvalidTableauError("The given tableau must be of type numpy ndarray")

        if self.validate:
            if not is_tableau_valid(self.tableau):
                raise InvalidTableauError("The given tableau is not valid.")

        # Update Tableau here
        input_tableau = input_tableau.rewrite_tableau(self.tableau)

        return input_tableau, data_store, kwargs

    def extra_pf(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        return input_pauliframes, data_store, kwargs

    def extra_pf_back(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        return input_pauliframes, data_store, kwargs


class CreatePauliFrameDecorator(InstructionDecorator):
    """
    The CreatePauliFrameDecorator creates a new PauliFrame object and adds it to the
    list of propagating PauliFrames at runtime.

    Parameters
    ----------
    instruction: Instruction
        The instruction that is being decorated.
    input_operation: CreatePauliFrame
        The CreatePauliFrame Operation that will be applied.

    Attributes
    ----------
    pauli_frame : PauliFrame
        The PauliFrame object to be created.
    """

    def __init__(self, instruction: Instruction, input_operation: CreatePauliFrame):
        super().__init__(instruction, input_operation)
        self.pauli_frame = deepcopy(input_operation.pauli_frame)

    def extra(
        self,
        input_tableau: Tableau,
        data_store: DataStore,
        **kwargs,
    ):
        return input_tableau, data_store, kwargs

    def extra_pf(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        # Add the PauliFrame to the list of propagating frames
        # only if the direction of propagation is forward
        if self.pauli_frame.direction == "forward":
            input_pauliframes.append(self.pauli_frame)
        return input_pauliframes, data_store, kwargs

    def extra_pf_back(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        # Add the PauliFrame to the list of propagating frames
        # only if the direction of propagation is backward
        if self.pauli_frame.direction == "backward":
            input_pauliframes.append(self.pauli_frame)
        return input_pauliframes, data_store, kwargs


class RecordPauliFrameDecorator(InstructionDecorator):
    """
    The RecordPauliFrameDecorator records the state of a PauliFrame object in the
    DataStore. The PauliFrame object must be in the list of propagating PauliFrames at
    runtime.

    Parameters
    ----------
    instruction: Instruction
        The instruction that is being decorated.
    input_operation: RecordPauliFrame
        The RecordPauliFrame Operation that will be applied.

    Attributes
    ----------
    pauli_frame : PauliFrame
        The PauliFrame object to be recorded.
    """

    def __init__(self, instruction: Instruction, input_operation: RecordPauliFrame):
        super().__init__(instruction, input_operation)
        self.pauli_frame = deepcopy(input_operation.pauli_frame)

    def extra(
        self,
        input_tableau: Tableau,
        data_store: DataStore,
        **kwargs,
    ):
        return input_tableau, data_store, kwargs

    def extra_pf(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        if self.pauli_frame.direction == "forward":
            for input_pauliframe in input_pauliframes:
                if input_pauliframe.id == self.pauli_frame.id:
                    # Record the state of pauli_frame in DataStore
                    data_store.record_pauli_frame(self.pauli_frame, input_pauliframe)
        return input_pauliframes, data_store, kwargs

    def extra_pf_back(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        if self.pauli_frame.direction == "backward":
            for input_pauliframe in input_pauliframes:
                if input_pauliframe.id == self.pauli_frame.id:
                    # Record the state of pauli_frame in DataStore
                    data_store.record_pauli_frame(self.pauli_frame, input_pauliframe)
        return input_pauliframes, data_store, kwargs


class CreateClassicalRegisterDecorator(InstructionDecorator):
    """
    The CreateClassicalRegisterDecorator creates a new classical register and adds it
    to the registry in the Engine at runtime.

    Parameters
    ----------
    instruction: Instruction
        The instruction that is being decorated.
    input_operation: CreateClassicalRegister
        The CreateClassicalRegister Operation that will be applied.

    Attributes
    ----------
    reg_name : str
        The name of the classical register.
    no_of_bits : int
        The number of bits in the classical register.
    bit_ids : List[int]
        The IDs of the bits in the classical register.
    """

    def __init__(
        self, instruction: Instruction, input_operation: CreateClassicalRegister
    ):
        super().__init__(instruction, input_operation)
        self.reg_name = input_operation.reg_name
        self.no_of_bits = input_operation.no_of_bits
        self.bit_ids = input_operation.bit_ids

    def extra(
        self,
        input_tableau: Tableau,
        data_store: DataStore,
        **kwargs,
    ):
        resolve_creg_names = list(kwargs["registry"].keys())
        if self.reg_name in resolve_creg_names:
            raise ClassicalRegisterError(
                f"The Classical Register of the same name, {self.reg_name}, already "
                "exists."
            )
        kwargs["registry"].update(
            {
                self.reg_name: ClassicalRegister(
                    self.reg_name, self.no_of_bits, self.bit_ids
                )
            }
        )

        return input_tableau, data_store, kwargs

    def extra_pf(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        return input_pauliframes, data_store, kwargs

    def extra_pf_back(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        return input_pauliframes, data_store, kwargs


class RecordClassicalRegisterDecorator(InstructionDecorator):
    """
    The RecordClassicalRegisterDecorator records the state of a classical register in
    the DataStore. The classical register must exist in the registry in the Engine at
    runtime.

    Parameters
    ----------
    instruction: Instruction
        The instruction that is being decorated.
    input_operation: RecordClassicalRegister
        The RecordClassicalRegister Operation that will be applied.

    Attributes
    ----------
    reg_name : str
        The name of the classical register.
    """

    def __init__(
        self, instruction: Instruction, input_operation: RecordClassicalRegister
    ):
        super().__init__(instruction, input_operation)
        self.reg_name = input_operation.reg_name

    def extra(
        self,
        input_tableau: Tableau,
        data_store: DataStore,
        **kwargs,
    ):
        # Record the current state of classical register named reg_name into the DataStore.
        data_store.record_classical_register(kwargs["registry"][self.reg_name])

        return input_tableau, data_store, kwargs

    def extra_pf(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        return input_pauliframes, data_store, kwargs

    def extra_pf_back(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        return input_pauliframes, data_store, kwargs


class ClassicalNOTDecorator(ClassicalBitDecorator):
    """
    The ClassicalNOTDecorator applies a NOT operation to a bit in a classical
    register.

    Parameters
    ----------
    instruction: Instruction
        The instruction that is being decorated.
    input_operation: ClassicalNOT
        The ClassicalNOT Operation that will be applied.

    Attributes
    ----------
    reg_name : str
        The name of the classical register.
    bit_order : int
        The order of the bit in the classical register.
    bit_id : int
        The ID of the bit in the classical register.
    """

    def __init__(self, instruction: Instruction, input_operation: ClassicalNOT):
        super().__init__(instruction, input_operation)
        self.reg_name = input_operation.reg_name
        self.bit_order = input_operation.bit_order
        self.bit_id = input_operation.bit_id

    def extra(
        self,
        input_tableau: Tableau,
        data_store: DataStore,
        **kwargs,
    ):
        # Select by Order or ID. Get info on the bit.
        input_reg = self._select_register(kwargs["registry"], self.reg_name)

        bit_value, bit_order, bit_id = self._get_bit_info(
            input_reg, self.bit_order, self.bit_id
        )

        # Flip the bit
        bit_value ^= 1
        kwargs["registry"][self.reg_name].reg[bit_order] = (bit_id, bit_value)

        return input_tableau, data_store, kwargs

    def extra_pf(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        return input_pauliframes, data_store, kwargs

    def extra_pf_back(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        return input_pauliframes, data_store, kwargs


class ClassicalORDecorator(ClassicalBitDecorator):
    """
    The ClassicalORDecorator applies a bitwise OR operation to two bits in a classical
    register and stores the result in another bit in the same register. (If the output
    bit exists, it will be overwritten)

    Parameters
    ----------
    instruction: Instruction
        The instruction that is being decorated.
    input_operation: ClassicalOR
        The ClassicalOR Operation that will be applied.

    Attributes
    ----------
    reg_name : str
        The name of the classical register.
    input_bit_order : List[int]
        The order of the input bits in the classical register.
    input_bit_ids : List[int]
        The IDs of the input bits in the classical register.
    output_reg_name : str
        The name of the classical register where the output bit is stored.
    write_bit_order : int
        The order of the output bit in the classical register.
    write_bit_id : int
        The ID of the output bit in the classical register.
    """

    def __init__(self, instruction: Instruction, input_operation: ClassicalOR):
        super().__init__(instruction, input_operation)
        self.reg_name = input_operation.reg_name
        self.input_bit_order = input_operation.input_bit_order
        self.input_bit_ids = input_operation.input_bit_ids
        self.output_reg_name = input_operation.output_reg_name
        self.write_bit_order = input_operation.write_bit_order
        self.write_bit_id = input_operation.write_bit_id

    def extra(
        self,
        input_tableau: Tableau,
        data_store: DataStore,
        **kwargs,
    ):
        # Get Value of Input Bits by Order or ID.
        input_bit_1, input_bit_2 = None, None
        input_reg = self._select_register(kwargs["registry"], self.reg_name)
        if self.input_bit_order != []:
            input_bit_1, _, _ = self._get_bit_info_w_bit_order(
                input_reg, self.input_bit_order[0]
            )
            input_bit_2, _, _ = self._get_bit_info_w_bit_order(
                input_reg, self.input_bit_order[1]
            )
        elif self.input_bit_ids != []:
            input_bit_1, _, _ = self._get_bit_info_w_bit_id(
                input_reg, self.input_bit_ids[0]
            )
            input_bit_2, _, _ = self._get_bit_info_w_bit_id(
                input_reg, self.input_bit_ids[1]
            )

        # Get Information on the Output Bit
        output_reg = self._select_register(kwargs["registry"], self.output_reg_name)
        _, write_bit_order, write_bit_id = self._get_bit_info(
            output_reg, self.write_bit_order, self.write_bit_id
        )

        # Evaluate Operation and Write to Output Bit
        kwargs["registry"][self.output_reg_name].reg[write_bit_order] = (
            write_bit_id,
            input_bit_1 | input_bit_2,
        )

        return input_tableau, data_store, kwargs

    def extra_pf(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        return input_pauliframes, data_store, kwargs

    def extra_pf_back(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        return input_pauliframes, data_store, kwargs


class ClassicalANDDecorator(ClassicalBitDecorator):
    """
    The ClassicalANDDecorator applies a bitwise AND operation to two bits in a
    classical register and stores the result in another bit in the same register. (If
    the output bit exists, it will be overwritten)

    Parameters
    ----------
    instruction: Instruction
        The instruction that is being decorated.
    input_operation: ClassicalAND
        The ClassicalAND Operation that will be applied.

    Attributes
    ----------
    reg_name : str
        The name of the classical register.
    input_bit_order : List[int]
        The order of the input bits in the classical register.
    input_bit_ids : List[int]
        The IDs of the input bits in the classical register.
    output_reg_name : str
        The name of the classical register where the output bit is stored.
    write_bit_order : int
        The order of the output bit in the classical register.
    write_bit_id : int
        The ID of the output bit in the classical register.
    """

    def __init__(self, instruction: Instruction, input_operation: ClassicalAND):
        super().__init__(instruction, input_operation)
        self.reg_name = input_operation.reg_name
        self.input_bit_order = input_operation.input_bit_order
        self.input_bit_ids = input_operation.input_bit_ids
        self.output_reg_name = input_operation.output_reg_name
        self.write_bit_order = input_operation.write_bit_order
        self.write_bit_id = input_operation.write_bit_id

    def extra(
        self,
        input_tableau: Tableau,
        data_store: DataStore,
        **kwargs,
    ):
        # Get Value of Input Bits by Order or ID.
        input_bit_1, input_bit_2 = None, None
        input_reg = self._select_register(kwargs["registry"], self.reg_name)
        if self.input_bit_order != []:
            input_bit_1, _, _ = self._get_bit_info_w_bit_order(
                input_reg, self.input_bit_order[0]
            )
            input_bit_2, _, _ = self._get_bit_info_w_bit_order(
                input_reg, self.input_bit_order[1]
            )
        elif self.input_bit_ids != []:
            input_bit_1, _, _ = self._get_bit_info_w_bit_id(
                input_reg, self.input_bit_ids[0]
            )
            input_bit_2, _, _ = self._get_bit_info_w_bit_id(
                input_reg, self.input_bit_ids[1]
            )

        # Get Information on the Output Bit
        output_reg = self._select_register(kwargs["registry"], self.output_reg_name)
        _, write_bit_order, write_bit_id = self._get_bit_info(
            output_reg, self.write_bit_order, self.write_bit_id
        )

        # Evaluate Operation and Write to Output Bit
        kwargs["registry"][self.output_reg_name].reg[write_bit_order] = (
            write_bit_id,
            input_bit_1 & input_bit_2,
        )

        return input_tableau, data_store, kwargs

    def extra_pf(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        return input_pauliframes, data_store, kwargs

    def extra_pf_back(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        return input_pauliframes, data_store, kwargs


class ControlledOperationDecorator(InstructionDecorator):
    """
    The ControlledOperationDecorator allows the user to transform an Operation into
    a Controlled Operation conditioned on a bit in the classical register. The operation
    is applied if the control bit is 1.

    Parameters
    ----------
    instruction: Instruction
        The instruction that is being decorated.
    input_operation: ControlledOperation
        The ControlledOperation Operation that will be applied. The information about
        the conditioned classical bit and the operation to be applied is stored in the
        `app_operation` of the ControlledOperation.

    Attributes
    ----------
    app_operation : :class:`loom.cliffordsim.operations.base_operation.Operation`
        The operation to be applied.
    reg_name : str
        The name of the classical register.
    bit_order : int
        The order of the bit in the classical register.
    bit_id : int
        The ID of the bit in the classical register.
    """

    def __init__(self, instruction: Instruction, input_operation: ControlledOperation):
        super().__init__(instruction, input_operation)
        self.app_operation = input_operation.app_operation
        self.reg_name = input_operation.reg_name
        self.bit_order = input_operation.bit_order
        self.bit_id = input_operation.bit_id

    @property
    def _app_instruction(self) -> Instruction:
        """Creates the controlled Instruction from the Operation."""
        return getattr(DecoratorSelector, self.app_operation.name).value(
            IdentityInstruction(), self.app_operation
        )

    def extra(
        self,
        input_tableau: Tableau,
        data_store: DataStore,
        **kwargs,
    ):
        # Select by Order or ID. Get info on the bit.
        bit_value = None
        if self.bit_order is not None:
            _, bit_value = kwargs["registry"][self.reg_name].reg[self.bit_order]
        elif self.bit_id is not None:
            for _, each_bit_info in enumerate(kwargs["registry"][self.reg_name].reg):
                if each_bit_info[0] == self.bit_id:
                    bit_value = each_bit_info[1]

        # Apply the Instruction if the bit_value is 1.
        if bit_value == 1:
            input_tableau, data_store, kwargs = self._app_instruction.apply(
                input_tableau, data_store, **kwargs
            )

        return input_tableau, data_store, kwargs

    def extra_pf(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        return input_pauliframes, data_store, kwargs

    def extra_pf_back(
        self,
        input_pauliframes: List[PauliFrame],
        data_store: DataStore,
        **kwargs,
    ):
        return input_pauliframes, data_store, kwargs


# pylint: disable=invalid-name
class DecoratorSelector(Enum):
    """
    The Decorator Selector chooses which Decorator to call based on the name
    of the input Operation. Decorators should not be selected directly other
    than through this selector.
    """

    DeleteQubit = DeleteQubitDecorator
    AddQubit = AddQubitDecorator
    SWAP = SWAPDecorator
    Identity = IdentityDecorator
    Hadamard = HadamardDecorator
    Phase = PhaseDecorator
    PhaseInv = PhaseInvDecorator
    CNOT = CNOTDecorator
    CY = CYDecorator
    CZ = CZDecorator
    X = XDecorator
    Z = ZDecorator
    Y = YDecorator
    Measurement = MeasurementDecorator
    Reset = ResetDecorator
    UpdateTableau = UpdateTableauDecorator
    CreatePauliFrame = CreatePauliFrameDecorator
    RecordPauliFrame = RecordPauliFrameDecorator
    CreateClassicalRegister = CreateClassicalRegisterDecorator
    RecordClassicalRegister = RecordClassicalRegisterDecorator
    ClassicalNOT = ClassicalNOTDecorator
    ClassicalOR = ClassicalORDecorator
    ClassicalAND = ClassicalANDDecorator
    ControlledOperation = ControlledOperationDecorator
