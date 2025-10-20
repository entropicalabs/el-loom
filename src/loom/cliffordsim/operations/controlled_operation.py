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


@dataclass
class ControlledOperation(Operation):
    """
    A decorator of the Operation class that turns any operation into a classically
    controlled operation.
    The Operation will be conditioned on the value of a single bit.

    NOTE: The Operation classes decorated by the ControlledOperation class can only
    be controlled by one bit.

    Parameters
    ----------
    app_operation: :class:`loom.cliffordsim.operations.base_operation.Operation`
        The Operation that will be Classically Controlled.
    reg_name: str
        The name of the Classical Register where the bit that classically controls
        app_operation will be located.
    bit_order: Optional[int]
        The ordering of the bit within the classical register that app_operation will
        be conditioned on. Either bit_order or bit_id needs to be specified. If both
        or none are specified, an error is raised.
    bit_id: Optional[str]
        The bit ID of the bit within the classical register that app_operation will be
        conditioned on. Either bit_order or bit_id needs to be specified. If both or
        none are specified, an error is raised.
    """

    operation_type: Enum = field(default=OpType.CCONTROL, init=False)
    name: str = field(default="ControlledOperation", init=False)
    app_operation: Operation
    reg_name: str
    bit_order: Optional[int] = field(default=None)
    bit_id: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.bit_order is None and self.bit_id is None:
            raise ValueError(
                "Either the bit order or the bit id of the classical bit has to be"
                "specified."
            )
        if self.bit_order is not None and self.bit_id is not None:
            raise ValueError(
                "Both bit_order and bit_id cannot be specified together. Only input 1 "
                "parameter, not both."
            )


def has_ccontrol(cls: Operation):
    """
    This function acts as a decorator that adds the with_ccontrol method to the input
    Operation class.
    """

    def with_ccontrol(self, reg_name, bit_order=None, bit_id=None):
        """
        This method returns a ControlledOperation wrapped version of the Operation
        class. The wrapped Operation class is classically controlled by the bit from
        the classical register specified by reg_name in bit_order or by bit_id.

        NOTE: ControlledOperation(s) can only be conditioned on one classical bit.
        """
        return ControlledOperation(
            app_operation=self, reg_name=reg_name, bit_order=bit_order, bit_id=bit_id
        )

    setattr(cls, "with_ccontrol", with_ccontrol)
    return cls
