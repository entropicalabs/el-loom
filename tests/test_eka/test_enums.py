import unittest
from loom.eka.utilities.enums import Direction, Orientation, ResourceState


class TestEnums(unittest.TestCase):

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


if __name__ == "__main__":
    unittest.main()
