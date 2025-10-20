"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest

from loom.interpreter import Syndrome


class TestSyndrome(unittest.TestCase):
    """Tests for the Syndrome class."""

    def test_creation_syndrome(self):
        """Tests the creation of a Syndrome object."""

        syndrome_attributes = {
            "stab_uuid": ["stab0", "stab1", "stab2"],
            "measurements": [(), (("c_(1,1,1)", 7),), (("c_(2,5,2)", 0),)],
            "block": ["block0", "block1", "block2"],
            "corrections": [(), (("c_(5, 8, 0)", 0),), ()],
            "round": [3, 9, 2],
            "labels": [
                {},
                {"space_coordinate": (1, 1, 1)},
                {"space_coordinate": (2, 5, 2), "time_coordinate": (4,), "color": 2},
            ],
        }

        # Loop over the three examples
        for i in range(len(syndrome_attributes["stab_uuid"])):

            syndrome = Syndrome(
                stabilizer=syndrome_attributes["stab_uuid"][i],
                measurements=syndrome_attributes["measurements"][i],
                block=syndrome_attributes["block"][i],
                round=syndrome_attributes["round"][i],
                corrections=syndrome_attributes["corrections"][i],
                labels=syndrome_attributes["labels"][i],
            )

            self.assertEqual(syndrome.stabilizer, syndrome_attributes["stab_uuid"][i])
            self.assertEqual(
                syndrome.measurements, syndrome_attributes["measurements"][i]
            )
            self.assertEqual(syndrome.block, syndrome_attributes["block"][i])
            self.assertEqual(syndrome.round, syndrome_attributes["round"][i])
            self.assertEqual(
                syndrome.corrections, syndrome_attributes["corrections"][i]
            )
            self.assertEqual(syndrome.labels, syndrome_attributes["labels"][i])

    def test_syndrome_equality(self):
        """Test the equality method"""

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
            labels={"space_coordinate": (2, 5, 2), "time_coordinate": (4,), "color": 2},
        )

        # They are equal despite having different labels
        self.assertEqual(syndrome1, syndrome2)

        # Test for inequality field by field
        syndrome3 = Syndrome(
            stabilizer="stab1",
            measurements=syndrome1.measurements,
            block=syndrome1.block,
            round=syndrome1.round,
            corrections=syndrome1.corrections,
            labels={},
        )

        syndrome4 = Syndrome(
            stabilizer=syndrome1.stabilizer,
            measurements=(("c_(1,1,2)", 7),),
            block=syndrome1.block,
            round=syndrome1.round,
            corrections=syndrome1.corrections,
            labels={},
        )

        syndrome5 = Syndrome(
            stabilizer=syndrome1.stabilizer,
            measurements=syndrome1.measurements,
            block="entropica_labs",
            round=syndrome1.round,
            corrections=syndrome1.corrections,
            labels={},
        )

        syndrome6 = Syndrome(
            stabilizer=syndrome1.stabilizer,
            measurements=syndrome1.measurements,
            block=syndrome1.block,
            round=334,
            corrections=syndrome1.corrections,
            labels={},
        )

        syndrome7 = Syndrome(
            stabilizer=syndrome1.stabilizer,
            measurements=syndrome1.measurements,
            block=syndrome1.block,
            round=syndrome1.round,
            corrections=(("c_(5, 8, 0)", 42),),
            labels={},
        )

        wrong_syndromes = [syndrome3, syndrome4, syndrome5, syndrome6, syndrome7]
        for wrong_syndrome in wrong_syndromes:
            self.assertNotEqual(syndrome1, wrong_syndrome)

    def test_syndrome_repr(self):
        """Test the string representation of the Syndrome object"""

        syndrome = Syndrome(
            stabilizer="stab0",
            measurements=(("c_(1,1,1)", 7),),
            block="block0",
            round=3,
            corrections=(("c_(5, 8, 0)", 0),),
            labels={"space_coordinate": (1, 1, 1)},
        )

        expected_repr = (
            "Syndrome(Measurements: (('c_(1,1,1)', 7),), "
            "Corrections: (('c_(5, 8, 0)', 0),), Round: 3, "
            "Labels: {'space_coordinate': (1, 1, 1)})"
        )
        self.assertEqual(repr(syndrome), expected_repr)

    def test_syndrome_hash(self):
        """Test the proper hashing of the Syndrome object"""

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

        # Check that two identical syndromes have the same hash
        self.assertEqual(hash(syndrome1), hash(syndrome2))

        # Check that two different syndromes have different hashes
        syndrome3 = Syndrome(
            stabilizer="stab0",
            measurements=(("c_(1,1,1)", 7),),
            block="block0",
            round=8,
            corrections=(("c_(5, 8, 0)", 0),),
            labels={"space_coordinate": (2, 5, 2)},
        )
        self.assertNotEqual(hash(syndrome1), hash(syndrome3))


if __name__ == "__main__":
    unittest.main()
