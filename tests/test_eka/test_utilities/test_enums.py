"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest
from loom.eka.utilities.enums import (
    Direction,
    Orientation,
    ResourceState,
    DiagonalDirection,
)


class TestEnums(unittest.TestCase):
    """Test cases for enums in eka.utilities.enums module."""

    def test_validation_of_direction(self):
        """Test whether the direction enum is validated correctly. Check that only the
        values defined in the enum and case insensitive matches of them are accepted.
        """
        allowed_values = [
            "left",
            "right",
            "top",
            "bottom",
            "Left",
            "TOP",
        ]
        not_allowed_values = ["l", "r", "north", "east"]
        # Check that the allowed values are accepted without an exception
        for direction in allowed_values:
            Direction(direction)

        # Check that the not allowed values raise an exception
        err_msg = (
            "is an invalid input for enum `Direction`. Only the following values are "
            "allowed as an input: left, right, top, bottom."
        )
        for direction in not_allowed_values:
            with self.assertRaises(ValueError) as cm:
                Direction(direction)
            self.assertIn(err_msg, str(cm.exception))

    def test_direction_to_orientation(self):
        """Test whether the direction enum is converted to the correct orientation."""
        # Define direction to orientation mapping
        direction_to_orientation_mapping = {
            Direction.LEFT: Orientation.HORIZONTAL,
            Direction.RIGHT: Orientation.HORIZONTAL,
            Direction.TOP: Orientation.VERTICAL,
            Direction.BOTTOM: Orientation.VERTICAL,
        }

        for direction in Direction:
            expected_orientation = direction_to_orientation_mapping.get(direction)
            # Test the conversion from direction to orientation
            self.assertEqual(
                Orientation.from_direction(direction), expected_orientation
            )
            self.assertEqual(direction.to_orientation(), expected_orientation)

    def test_resource_state_creation(self):
        """Test whether the resource state enum is validated correctly."""
        allowed_values = ["t", "s", "T", "S"]
        not_allowed_values = ["x", "invalid"]

        # Check that the allowed values are accepted without an exception
        for state in allowed_values:
            ResourceState(state)

        # Check that the not allowed values raise an exception
        for state in not_allowed_values:
            err_msg = (
                f"`{state}` is an invalid input for enum `ResourceState`. Only the "
                "following values are allowed as an input: t, s."
            )
            with self.assertRaises(ValueError) as cm:
                ResourceState(state)
            self.assertIn(err_msg, str(cm.exception))

    def test_mirror_of_direction(self):
        """Test whether the mirror of a direction is returned correctly."""
        # Define direction to its mirror mapping
        expected_mirror_dict = {
            Direction.LEFT: {
                Orientation.VERTICAL: Direction.RIGHT,
                Orientation.HORIZONTAL: Direction.LEFT,
            },
            Direction.RIGHT: {
                Orientation.VERTICAL: Direction.LEFT,
                Orientation.HORIZONTAL: Direction.RIGHT,
            },
            Direction.TOP: {
                Orientation.HORIZONTAL: Direction.BOTTOM,
                Orientation.VERTICAL: Direction.TOP,
            },
            Direction.BOTTOM: {
                Orientation.HORIZONTAL: Direction.TOP,
                Orientation.VERTICAL: Direction.BOTTOM,
            },
        }

        for direction in Direction:
            for orientation in Orientation:
                expected_mirror = expected_mirror_dict[direction][orientation]
                # Test the mirror method
                self.assertEqual(
                    direction.mirror_across_orientation(orientation), expected_mirror
                )

    def test_diagonal_mirror_of_direction(self):
        """Test whether the diagonal mirror of a direction is returned correctly."""
        # Define direction to its diagonal mirror mapping
        expected_mirror_dict = {
            DiagonalDirection.TOP_LEFT: {
                Orientation.VERTICAL: DiagonalDirection.TOP_RIGHT,
                Orientation.HORIZONTAL: DiagonalDirection.BOTTOM_LEFT,
            },
            DiagonalDirection.TOP_RIGHT: {
                Orientation.VERTICAL: DiagonalDirection.TOP_LEFT,
                Orientation.HORIZONTAL: DiagonalDirection.BOTTOM_RIGHT,
            },
            DiagonalDirection.BOTTOM_LEFT: {
                Orientation.VERTICAL: DiagonalDirection.BOTTOM_RIGHT,
                Orientation.HORIZONTAL: DiagonalDirection.TOP_LEFT,
            },
            DiagonalDirection.BOTTOM_RIGHT: {
                Orientation.VERTICAL: DiagonalDirection.BOTTOM_LEFT,
                Orientation.HORIZONTAL: DiagonalDirection.TOP_RIGHT,
            },
        }

        for diagonal_direction in DiagonalDirection:
            for orientation in Orientation:
                expected_mirror = expected_mirror_dict[diagonal_direction][orientation]
                # Test the mirror method
                self.assertEqual(
                    diagonal_direction.mirror_across_orientation(orientation),
                    expected_mirror,
                )

    def test_direction_along_orientation(self):
        """Test whether the direction along an orientation is returned correctly."""
        # Define orientation to its direction mapping
        expected_direction_dict = {
            DiagonalDirection.BOTTOM_LEFT: {
                Orientation.VERTICAL: Direction.BOTTOM,
                Orientation.HORIZONTAL: Direction.LEFT,
            },
            DiagonalDirection.BOTTOM_RIGHT: {
                Orientation.VERTICAL: Direction.BOTTOM,
                Orientation.HORIZONTAL: Direction.RIGHT,
            },
            DiagonalDirection.TOP_LEFT: {
                Orientation.VERTICAL: Direction.TOP,
                Orientation.HORIZONTAL: Direction.LEFT,
            },
            DiagonalDirection.TOP_RIGHT: {
                Orientation.VERTICAL: Direction.TOP,
                Orientation.HORIZONTAL: Direction.RIGHT,
            },
        }

        for diag in DiagonalDirection:
            for orientation in Orientation:
                expected_direction = expected_direction_dict[diag][orientation]
                # Test the direction method
                self.assertEqual(
                    diag.direction_along_orientation(orientation),
                    expected_direction,
                )


if __name__ == "__main__":
    unittest.main()
