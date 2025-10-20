"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from __future__ import annotations
from enum import Enum

from loom.eka.utilities.enums import enum_missing


class FourBodySchedule(str, Enum):
    """
    Enum for the four-body schedule used in the rotated surface code.
    """

    N = "N"
    Z = "Z"

    def opposite_schedule(self) -> FourBodySchedule:
        """Get the opposite schedule."""
        if self == FourBodySchedule.N:
            return FourBodySchedule.Z
        if self == FourBodySchedule.Z:
            return FourBodySchedule.N
        raise ValueError("Invalid schedule. Cannot determine opposite schedule.")

    @classmethod
    def _missing_(cls, value):
        """
        Allow inputs with upper-case characters. For more details, see the
        documentation of ``enum_missing`` at the beginning of the file.
        """
        return enum_missing(cls, value)
