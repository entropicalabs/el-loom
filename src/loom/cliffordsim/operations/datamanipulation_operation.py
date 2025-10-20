"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from dataclasses import dataclass, field

import numpy as np

from .base_operation import Operation, OpType
from ..pauli_frame import PauliFrame
from .controlled_operation import has_ccontrol


@dataclass
@has_ccontrol
class DataManipulationOperation(Operation):
    """
    Operations of this type manipulate data within the Engine during runtime.
    """

    operation_type: str = field(default=OpType.DATAMANIPULATION, init=False)


@dataclass
class UpdateTableau(DataManipulationOperation):
    """
    An Operation that updates the state of the Tableau in the Engine during runtime. Note
    that the tableau must be a numpy array with bits, 0s and 1s, for every element.

    Parameters
    ----------
    tableau: numpy.ndarray
        The state of the Tableau to be updated to.
    validate: bool
        The tableau will be validated at runtime. The tableau provided must be a valid
        stabilizer tableau. The default is True.
    """

    name: str = field(default="UpdateTableau", init=False)
    tableau: np.ndarray
    validate: bool = field(default=True)
    # Maybe add tableau valdiation here instead? Currently done in Instructions if
    # validate = True.


@dataclass
class CreatePauliFrame(DataManipulationOperation):
    """
    An Operation that creates a Pauli Frame within the Engine during runtime. The
    Pauli Frame created will be associated with the name provided by the user, and can
    be accessed at runtime by other operations. Since there can be multiple Pauli Frames,
    the frames will be associated by their names.
    """

    name: str = field(default="CreatePauliFrame", init=False)
    pauli_frame: PauliFrame

    def __post_init__(self):
        # Check validity of pauli_frame
        if not isinstance(self.pauli_frame, PauliFrame):
            raise TypeError(
                f"Invalid PauliFrame '{self.pauli_frame}'. Must be of type 'PauliFrame'."
            )


@dataclass
class RecordPauliFrame(DataManipulationOperation):
    """
    Records the current state of the Pauli Frame into the DataStore. The state can then
    be accessed at the end of the run from the DataStore. The Pauli Frame to be recorded
    will be identified by its name.
    """

    name: str = field(default="RecordPauliFrame", init=False)
    pauli_frame: PauliFrame

    def __post_init__(self):
        # Check validity of pauli_frame
        if not isinstance(self.pauli_frame, PauliFrame):
            raise TypeError(
                f"Invalid PauliFrame '{self.pauli_frame}'. Must be of type 'PauliFrame'."
            )


@dataclass
class CreateClassicalRegister(DataManipulationOperation):
    """
    An operation that creates a Classical Register within the Engine that can be
    accessed at runtime by classical operations. Since there can be multiple classical
    registers, the registers will be associated by their reg_name, and their bits by
    their respective bit IDs or bit ordering as provided by the user.

    If trying to initialize a classical register with bit IDs, the number of bit IDs
    must be equal to the number of bits in the register.

    Example of valid Classical Register:
    CreateClassicalRegister(reg_name="testreg", no_of_bits=3, bit_ids=["bit_id_1",
    "bit_id_2", "bit_id_3"])

    Parameters
    ----------
    reg_name: str
        The name of the classical register. This name will be used by cliffordsim to
        identify the classical register being referred to by the user.
    no_of_bits: int
        The number of bits to initialize the register with. Default is 1.
    bit_ids: list[str]
        The bit IDs of all the bits in the register. Example input: ["bit_id_1",
        "bit_id_2", "bit_id_3"].
        The number of bit IDs provided must be equal to the number of bits in the
        classical register.
        If no bit IDs are provided, the classical register will randomly generate
        the IDs upon initialization.
    """

    name: str = field(default="CreateClassicalRegister", init=False)
    reg_name: str
    no_of_bits: int = field(default=1)
    bit_ids: list[str] = field(default=None)

    def __post_init__(self):
        # Check validity of bit IDs
        if self.bit_ids is not None:
            for each_id in self.bit_ids:
                if not isinstance(each_id, str):
                    raise TypeError("The bit ID must be a string.")
            if len(self.bit_ids) != self.no_of_bits:
                raise ValueError(
                    "The number of bit IDs must be equal to the number of bits in the register."
                )


@dataclass
class RecordClassicalRegister(DataManipulationOperation):
    """
    Records the current state of the Classical Register into the DataStore. The state
    can then be accessed at the end of the run from the DataStore.

    Parameters
    ----------
    reg_name: str
        The name of the Classical Register to record.
    """

    name: str = field(default="RecordClassicalRegister", init=False)
    reg_name: str
