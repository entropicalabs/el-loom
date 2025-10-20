"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable= duplicate-code
import unittest

from loom.interpreter import Detector, Syndrome


class TestDetector(unittest.TestCase):
    """Tests for the Detector class."""

    def setUp(self):

        self.syndrome1 = Syndrome(
            stabilizer="stab0",
            measurements=(("c_(1,1,1)", 7),),
            block="block0",
            round=3,
            corrections=(("c_(5, 8, 0)", 0),),
            labels={"space_coordinate": (1, 1, 1)},
        )

        self.syndrome2 = Syndrome(
            stabilizer="stab0",
            measurements=(("c_(1,1,1)", 8),),
            block="block0",
            round=4,
            corrections=(),
            labels={"time_coordinate": (9,)},
        )

        self.detector1 = Detector(
            syndromes=[self.syndrome1, self.syndrome2], labels=self.syndrome2.labels
        )
        self.detector2 = Detector(
            syndromes=[self.syndrome1, self.syndrome2], labels={"quantum": 42}
        )

        # Check that two different syndromes have different hashes
        self.syndrome3 = Syndrome(
            stabilizer="stab0",
            measurements=(("c_(1,1,1)", 7),),
            block="block0",
            round=8,
            corrections=(("c_(5, 8, 0)", 0),),
            labels={"space_coordinate": (2, 5, 2)},
        )

        self.detector3 = Detector(syndromes=[self.syndrome3], labels={})

    def test_creation_detector(self):
        """Tests the creation of a Detector object."""

        self.assertIsInstance(self.detector1, Detector)
        self.assertEqual(self.detector1.syndromes, (self.syndrome1, self.syndrome2))
        self.assertEqual(self.detector1.labels, self.syndrome2.labels)

        self.assertIsInstance(self.detector3, Detector)
        self.assertEqual(self.detector3.syndromes, (self.syndrome3,))
        self.assertEqual(self.detector3.labels, {})

    def test_detector_equality(self):
        """Test the equality method"""
        self.assertEqual(self.detector1, self.detector2)
        self.assertNotEqual(self.detector1, self.detector3)

    def test_detector_rounds(self):
        """Test the rounds property of Detector"""
        self.assertEqual(self.detector1.rounds(), (3, 4))
        self.assertEqual(self.detector3.rounds(), (8,))

    def test_detector_stabilizer(self):
        """Test the stabilizer property of the Detector object"""

        self.assertEqual(self.detector1.stabilizer(), ("stab0", "stab0"))
        self.assertEqual(self.detector3.stabilizer(), ("stab0",))

    def test_detector_repr(self):
        """Test the string representation of the Detector object"""

        expected_repr = (
            "Detector(Syndromes: (Syndrome(Measurements: (('c_(1,1,1)', 7),), "
            "Corrections: (('c_(5, 8, 0)', 0),), Round: 3, "
            "Labels: {'space_coordinate': (1, 1, 1)}),"
            " Syndrome(Measurements: (('c_(1,1,1)', 8),), "
            "Corrections: (), Round: 4, "
            "Labels: {'time_coordinate': (9,)})), Labels: {'time_coordinate': (9,)})"
        )
        self.assertEqual(repr(self.detector1), expected_repr)

    def test_detector_hash(self):
        """Test the proper hashing of the Detector object"""

        syndrome1 = Syndrome(
            stabilizer="stab0",
            measurements=(("c_(1,1,1)", 7),),
            block="block0",
            round=3,
            corrections=(("c_(5, 8, 0)", 0),),
            labels={"space_coordinate": (1, 1, 1)},
        )

        syndrome2 = Syndrome(
            stabilizer="stab0",
            measurements=(("c_(1,1,1)", 7),),
            block="block0",
            round=3,
            corrections=(("c_(5, 8, 0)", 0),),
            labels={"time_coordinate": (9,)},
        )

        detector1 = Detector(syndromes=[syndrome1, syndrome2], labels={})
        detector2 = Detector(syndromes=[syndrome1, syndrome2], labels={"quantum": 42})

        # Check that two identical syndromes have the same hash
        self.assertEqual(hash(detector1), hash(detector2))

        # Check that two different syndromes have different hashes
        syndrome3 = Syndrome(
            stabilizer="stab0",
            measurements=(("c_(1,1,1)", 7),),
            block="block0",
            round=8,
            corrections=(("c_(5, 8, 0)", 0),),
            labels={"space_coordinate": (2, 5, 2)},
        )

        detector3 = Detector(syndromes=[syndrome3], labels={})
        self.assertNotEqual(hash(detector1), hash(detector3))


if __name__ == "__main__":
    unittest.main()
