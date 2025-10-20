"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from ..operations.base_operation import Operation
from ..tableau import Tableau
from ..pauli_frame import PauliFrame
from ..data_store import DataStore
from .instruction import Instruction, IdentityInstruction, DecoratorSelector


class MomentInterface(ABC):  # pylint: disable=too-few-public-methods
    """
    An abstract base class for Moments.
    """

    @abstractmethod
    def transform_tab(self, input_tableau: np.ndarray) -> np.ndarray:
        """
        Transform the input tableau and return the transformed tableau.
        """


class Moment(MomentInterface):
    """
    A Moment contains a set of instructions that are directly applicable
    on the Tableau.
    """

    def __init__(self, input_operations: tuple[Operation], time_step: int) -> None:
        self.root_operations = input_operations
        self.time_step = time_step
        self.instruction = self._create_instruction()

    def _create_instruction(self) -> Instruction:
        """
        Maps Operations to Instructions and builds the Moment's
        Instruction as an attribute, instruction.
        """
        # Converts the first Operation in root_operations into an Instruction
        # and wraps that Instruction around IdentityInstruction
        main_inst = IdentityInstruction()
        wrapped_inst_dec = getattr(
            DecoratorSelector, self.root_operations[0].name
        ).value
        wrapped_inst = wrapped_inst_dec(main_inst, self.root_operations[0])

        # Converts subsequent Operation(s) into Instruction(s) and wrap those
        # around the previous Instruction
        for each_operation in self.root_operations[1:]:
            inst_decorator = getattr(DecoratorSelector, each_operation.name).value
            wrapped_inst = inst_decorator(wrapped_inst, each_operation)
        return wrapped_inst

    # pylint: disable=arguments-differ
    def transform_tab(
        self, input_tableau: Tableau, data_store: DataStore, **kwargs
    ) -> Tableau:
        """
        Applies tranformation of Instructions on incoming Tableau.
        Also requires a DataStore that manages information keeping throughout
        the instruction application process and kwargs to transparently support
        future extensions.
        """
        return self.instruction.apply(input_tableau, data_store, **kwargs)

    def transform_pf(
        self, input_pauliframes: List[PauliFrame], data_store: DataStore, **kwargs
    ) -> List[PauliFrame]:
        """
        Applies tranformation of Instructions on incoming PauliFrame.
        Also requires a DataStore that manages information keeping throughout
        the instruction application process and kwargs to transparently support
        future extensions.
        """
        return self.instruction.apply_pf(input_pauliframes, data_store, **kwargs)

    def transform_pf_back(
        self, input_pauliframes: List[PauliFrame], data_store: DataStore, **kwargs
    ) -> List[PauliFrame]:
        """
        Applies backwards tranformation of Instructions on incoming PauliFrame.
        Also requires a DataStore that manages information keeping throughout
        the instruction application process and kwargs to transparently support
        future extensions.
        """
        return self.instruction.apply_pf_back(input_pauliframes, data_store, **kwargs)
