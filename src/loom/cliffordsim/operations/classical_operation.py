"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=duplicate-code
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .base_operation import Operation, OpType
from .controlled_operation import has_ccontrol


@dataclass
@has_ccontrol
class ClassicalOperation(Operation):
    """
    Operations of this type manipulate classical bits within the Engine during runtime.
    """

    operation_type: Enum = field(default=OpType.CLASSICAL, init=False)


@dataclass
class ClassicalNOT(ClassicalOperation):
    """Apply a classical NOT operation on a bit.

    Parameters
    ----------
    reg_name: str
        The name of the classical register whose bit to apply the Operation to.
    bit_order: int
        The numerical order of the bit within the classical register.
    bit_id: str
        The ID of the bit within the classical register.
    """

    name: str = field(default="ClassicalNOT", init=False)
    reg_name: str
    bit_order: int = field(default=None)
    bit_id: str = field(default=None)

    def __post_init__(self):
        if self.bit_order is None and self.bit_id is None:
            raise ValueError(
                "Either the bit order or the bit id of the classical bit has to be "
                "specified."
            )
        if self.bit_order is not None and self.bit_id is not None:
            raise ValueError(
                "Both bit_order and bit_id cannot be specified together. Only input 1 "
                "parameter, not both."
            )


@dataclass
class ClassicalTwoBitOperation(ClassicalOperation):
    """Classical 2 Bit Operations that require 2 bits as input and 1 bit as output.

    Parameters
    ----------
    reg_name: str
        The name of the classical register whose bits is used as input to evaluate the
        2-bit operation.
    input_bit_order: list[int]
        The order of the bits within the classical register that will be used as inputs
        for the 2-bit operation.
    input_bit_ids: list[str]
        The bit IDs of the bits within the classical reigster that will be used as
        inputs for the 2-bit operation.
    output_reg_name: Optional[str]
        The name of the classical register where the output of the 2-bit operation will
        be written to. If not provided, this is equal to the register referred to by
        reg_name.
    write_bit_order: int
        The order of the bit within the classical register, specified by
        output_reg_name, where the output of the 2-bit operation will be written to.
    write_bit_ids: str
        The bit ID of the bit within the classical register, specified by
        output_reg_name, where the output of the 2-bit operation will be written to.
    """

    reg_name: str
    input_bit_order: list[int] = field(default_factory=list)
    input_bit_ids: list[str] = field(default_factory=list)
    output_reg_name: Optional[str] = field(default=None)
    write_bit_order: int = field(default=None)
    write_bit_id: str = field(default=None)

    def __post_init__(self):
        if (bool(self.input_bit_order)) ^ (bool(self.input_bit_ids)):
            if not (len(self.input_bit_order) == 2) ^ (len(self.input_bit_ids) == 2):
                raise ValueError(
                    "There must be at least 2 bits referenced using their bit ordering "
                    "or their bit IDs for this operation."
                )
        else:
            raise ValueError(
                "Both input_bit_order and input_bit_ids cannot be specified together."
                "Only input 1 parameter, not both."
            )

        if not (self.write_bit_order is not None) ^ (self.write_bit_id is not None):
            raise ValueError(
                "Both input_bit_order and input_bit_ids cannot be specified together."
                "Only input 1 parameter, not both."
            )

        if self.output_reg_name is None:
            self.output_reg_name = self.reg_name


@dataclass
class ClassicalOR(ClassicalTwoBitOperation):
    """Evaluates a classical OR operation from 2 input bits and writes the output onto
    a referenced bit, known as the write bit.
    """

    name: str = field(default="ClassicalOR", init=False)


@dataclass
class ClassicalAND(ClassicalTwoBitOperation):
    """
    Evaluates a classical AND operation from 2 input bits and writes the output onto
    a referenced bit, known as the write bit.
    """

    name: str = field(default="ClassicalAND", init=False)
