"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from __future__ import annotations
from math import cos, pi

from enum import Enum
import numpy as np


def enum_missing(cls, value):
    """
    This method is called when `value` is not found in the Enum. This could be the
    case when the input is not lower case but its lower case version is in the Enum.
    Therefore `value` is first converted to lower-case and then compared to the values
    in the Enum.

    If no match is found, an exception is raised. Also note that the exception message
    is more informative than the default one since we here include the allowed values.
    """
    # Check the input in a case insensitive way
    value = value.lower()
    for member in cls:
        if member.value == value:
            return member

    # If no case insensitive match is found, raise an exception
    allowed_values = [member.value for member in cls]
    raise ValueError(
        f"`{value}` is an invalid input for enum `{cls.__name__}`. Only the following "
        f"values are allowed as an input: {', '.join(allowed_values)}."
    )


class SingleQubitPauliEigenstate(str, Enum):
    """
    Supported states in reset operations.
    """

    ZERO = "0"
    ONE = "1"
    PLUS = "+"
    MINUS = "-"
    PLUS_I = "+i"
    MINUS_I = "-i"

    @property
    def pauli_basis(self) -> str:
        """
        Get the Pauli basis of the state.
        """
        if self in (SingleQubitPauliEigenstate.ZERO, SingleQubitPauliEigenstate.ONE):
            return "Z"
        if self in (
            SingleQubitPauliEigenstate.PLUS,
            SingleQubitPauliEigenstate.MINUS,
        ):
            return "X"
        if self in (
            SingleQubitPauliEigenstate.PLUS_I,
            SingleQubitPauliEigenstate.MINUS_I,
        ):
            return "Y"

        raise ValueError(f"Invalid state: {self}")

    @property
    def basis_expectation_value(self) -> int:
        """
        Get the expectation value of the state in the Pauli basis of the state.
        """
        if self in (
            SingleQubitPauliEigenstate.ZERO,
            SingleQubitPauliEigenstate.PLUS,
            SingleQubitPauliEigenstate.PLUS_I,
        ):
            return 1
        if self in (
            SingleQubitPauliEigenstate.ONE,
            SingleQubitPauliEigenstate.MINUS,
            SingleQubitPauliEigenstate.MINUS_I,
        ):
            return -1

        raise ValueError(f"Invalid state: {self}")


class Direction(str, Enum):
    """
    Direction indicator for Operations (e.g. Grow and Shrink).
    """

    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"

    @classmethod
    def _missing_(cls, value):
        """
        Allow inputs with upper-case characters. For more details, see the
        documentation of `enum_missing` at the beginning of the file.
        """
        return enum_missing(cls, value)

    def __str__(self):
        return str(self.value)

    def to_vector(self) -> tuple[int, int]:
        """
        Convert the direction to a 2D vector.
        """
        if self == Direction.LEFT:
            return (-1, 0)
        if self == Direction.RIGHT:
            return (1, 0)
        if self == Direction.TOP:
            return (0, -1)
        if self == Direction.BOTTOM:
            return (0, 1)

        raise ValueError(f"Direction has no vector representation: {self}")

    @classmethod
    def from_vector(cls, vector: tuple[int, ...], bottom_is_plus=True) -> Direction:
        """
        Get the direction from a 2D vector. The vector should have only
        one non-zero component in the first or the second dimension. The
        direction is determined by the non-zero component. The direction
        is returned as a Direction enum.
        """
        np_vector = np.array(vector)
        if len(np_vector) < 2:
            raise ValueError("Cannot get direction from a 1D vector.")

        if any(np.nonzero(np_vector)[0] > 1):
            raise ValueError(
                "Only the first and the second components of the vector may be non zero"
            )

        # Get the sign of the components in an array
        sign_array = np.sign(np_vector)
        if not bottom_is_plus:
            sign_array[1] *= -1

        # Determine the direction
        match (sign_array[0], sign_array[1]):
            case (1, 0):
                return Direction.RIGHT
            case (-1, 0):
                return Direction.LEFT
            case (0, 1):
                return Direction.BOTTOM
            case (0, -1):
                return Direction.TOP
            case _:
                raise ValueError(
                    "Direction cannot be found from the given vector. The vector should"
                    " have exactly one non-zero component"
                )

    def opposite(self) -> Direction:
        """
        Get the opposite direction.
        """
        if self == Direction.LEFT:
            return Direction.RIGHT
        if self == Direction.RIGHT:
            return Direction.LEFT
        if self == Direction.TOP:
            return Direction.BOTTOM
        if self == Direction.BOTTOM:
            return Direction.TOP

        raise ValueError(f"Direction has no opposite: {self}")

    def to_orientation(self) -> Orientation:
        """Convert the direction to an orientation."""
        return Orientation.from_direction(self)


class Orientation(str, Enum):
    """
    Orientation indicator for Operations (e.g. Split) and for Block initialization.
    """

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"

    @classmethod
    def _missing_(cls, value):
        """
        Allow inputs with upper-case characters. For more details, see the
        documentation of `enum_missing` at the beginning of the file.
        """
        return enum_missing(cls, value)

    @classmethod
    def from_direction(cls, direction: Direction) -> Orientation:
        """
        Get the orientation from a direction.
        """
        match direction:
            case Direction.LEFT | Direction.RIGHT:
                return Orientation.HORIZONTAL
            case Direction.TOP | Direction.BOTTOM:
                return Orientation.VERTICAL
            case _:
                raise ValueError("Invalid direction. Cannot determine orientation.")

    @classmethod
    def from_vector(cls, vector: tuple[int, ...]) -> Orientation:
        """
        Get the orientation from a 2D vector. The orientation is determined
        by the direction of the vector. The orientation is returned as an
        Orientation enum.
        """
        return cls.from_direction(Direction.from_vector(vector))

    def perpendicular(self) -> Orientation:
        """
        Get the perpendicular orientation.
        """
        if self == Orientation.HORIZONTAL:
            return Orientation.VERTICAL
        if self == Orientation.VERTICAL:
            return Orientation.HORIZONTAL

        raise ValueError(f"Orientation has no perpendicular: {self}")


class ResourceState(str, Enum):
    """Supported states in state injection operations."""

    T = "t"
    S = "s"

    def get_expectation_value(self, basis: str) -> float:
        """Get the expectation value of the state in the Pauli basis of the state."""
        if basis not in ("X", "Y", "Z"):
            raise ValueError(
                f"Invalid basis: {basis}. Allowed values are 'X', 'Y', or 'Z'."
            )
        match self:
            case ResourceState.T:
                expectation = {
                    "Z": 0,  # 50% chance of measuring +1/-1 in Z basis
                    "X": cos(pi / 4),  # ~85% chance of measuring +1 in X basis
                    "Y": cos(pi / 4),  # ~85% chance of measuring +1 in Y basis
                }
            case ResourceState.S:
                expectation = {
                    "Z": 0,  # 50% chance of measuring +1/-1 in Z basis
                    "X": 0,  # 50% chance of measuring +1/-1 in X basis
                    "Y": 1,  # 100% chance of measuring +1 in Y basis
                }
            case _:
                raise ValueError(
                    f"Invalid resource state: {self}, allowed values are"
                    f"{list(ResourceState)}"
                )
        return float(expectation[basis])

    @classmethod
    def _missing_(cls, value):
        """Allow inputs with upper-case characters. For more details, see the
        documentation of `enum_missing` at the beginning of the file."""
        return enum_missing(cls, value)
