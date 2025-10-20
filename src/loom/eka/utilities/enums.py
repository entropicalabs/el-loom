"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from __future__ import annotations
from collections.abc import Iterable
from math import cos, pi
from functools import reduce

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

    def mirror_across_orientation(self, orientation: Orientation) -> Direction:
        """Get the mirrored direction across the given orientation."""
        return self if self.to_orientation() == orientation else self.opposite()


class DiagonalDirection(str, Enum):
    """Diagonal direction indicator for Operations (e.g. Grow and Shrink)."""

    TOP_LEFT = "top-left"
    TOP_RIGHT = "top-right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_RIGHT = "bottom-right"

    # Map diagonals to their component directions

    @property
    def components(self):
        """Return the two cardinal directions composing this diagonal."""
        match self.value:
            case "top-left":
                return (Direction.TOP, Direction.LEFT)
            case "top-right":
                return (Direction.TOP, Direction.RIGHT)
            case "bottom-left":
                return (Direction.BOTTOM, Direction.LEFT)
            case "bottom-right":
                return (Direction.BOTTOM, Direction.RIGHT)

    @classmethod
    def _missing_(cls, value):
        """Allow inputs with upper-case characters. For more details, see the
        documentation of `enum_missing` at the beginning of the file."""
        return enum_missing(cls, value)

    def __str__(self):
        return str(self.value)

    @classmethod
    def from_directions(
        cls, directions: tuple[Direction, Direction]
    ) -> DiagonalDirection:
        """Get the diagonal direction from two cardinal directions. The two
        directions must be perpendicular."""

        if len(directions) != 2:
            raise ValueError("Exactly two directions are required.")
        if directions[0].to_orientation() == directions[1].to_orientation():
            raise ValueError(
                "The two directions must be perpendicular."
                f" Got {directions[0]} and {directions[1]}."
            )
        # Order should not matter
        directions_pair = frozenset(directions)
        for diag in cls:
            if frozenset(diag.components) == directions_pair:
                return diag
        raise ValueError(f"Invalid direction pair: {directions}.")

    @classmethod
    def from_vector(cls, vector: tuple[int, ...]) -> DiagonalDirection:
        """Get the diagonal direction from a 2D vector. The vector should have only
        two non-zero components in the first and the second dimension. The
        direction is determined by the sign of the components. The direction
        is returned as a DiagonalDirection enum.
        """
        if len(vector) != 2:
            raise ValueError("Vector must be 2D.")

        hor_coord, vert_coord = vector
        if hor_coord == 0 or vert_coord == 0:
            raise ValueError("Vector cannot have zero components.")

        hor_direction = Direction.from_vector((hor_coord, 0))
        vert_direction = Direction.from_vector((0, vert_coord))
        return DiagonalDirection.from_directions((hor_direction, vert_direction))

    def to_vector(self) -> tuple[int, int]:
        """Convert the diagonal direction to a 2D vector."""
        vectors = [direction.to_vector() for direction in self.components]
        # Sum the vectors to get the resulting vector
        return reduce(
            lambda acc, vec: (acc[0] + vec[0], acc[1] + vec[1]), vectors, (0, 0)
        )

    def opposite(self) -> DiagonalDirection:
        """Get the opposite diagonal direction."""
        opposite_directions = [direction.opposite() for direction in self.components]
        return DiagonalDirection.from_directions(opposite_directions)

    def direction_along_orientation(self, orientation: Orientation) -> Direction:
        """Get the cardinal direction along the given orientation."""
        return next(
            direction
            for direction in self.components
            if direction.to_orientation() == orientation
        )

    def mirror_across_orientation(self, orientation: Orientation) -> DiagonalDirection:
        """Get the mirrored diagonal direction across the given orientation."""
        return DiagonalDirection.from_directions(
            [
                direction.mirror_across_orientation(orientation)
                for direction in self.components
            ]
        )

    def __iter__(self) -> Iterable[Direction]:
        """Allow iteration over the component directions."""
        return iter(self.components)


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
