"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest
import uuid

from loom.cliffordsim.classicalreg import ClassicalRegister


class TestCliffordSimClassicalRegister(unittest.TestCase):
    """
    We test the Classical Register and its associated functionality.
    """

    def test_initialization_required(self):
        """
        The Classical Register requires the name and number of bits to be initialized.

        The register created automatically creates bit ids for every bit. The ids are
        uuid4 compatible and should not be identical between bits.

        The bits are all initialized in 0.
        """
        name = "testreg"
        no_of_bits = 10
        creg = ClassicalRegister(name, no_of_bits)

        self.assertEqual(creg.name, name)
        self.assertEqual(creg.no_of_bits, no_of_bits)
        self.assertTrue(len(set(creg.bit_ids)) == len(creg.bit_ids))
        self.assertEqual(creg.bit_reg, list(0 for _ in range(no_of_bits)))

        # Auto-genereated Bit IDs are uuid4 compatible
        list(uuid.UUID(each_id, version=4) for each_id in creg.bit_ids)

    def test_initialization_optional(self):
        """
        The Classical Register can be initialized with bit ids provided as an optional
        arguement.

        Note, custom ids need not be uuid4 compatible.

        The number of IDs provided must be equal to the number of bits in the register.

        The IDs provided cannot be identical.
        """
        name = "testreg"
        no_of_bits = 2
        bit_ids = ["bit_1", "bit_2"]
        creg = ClassicalRegister(name, no_of_bits, bit_ids)

        self.assertEqual(creg.bit_ids, bit_ids)
        self.assertEqual(creg.name, name)
        self.assertEqual(creg.no_of_bits, no_of_bits)
        self.assertEqual(creg.bit_reg, [0 for _ in range(no_of_bits)])

        with self.assertRaises(ValueError):
            _ = ClassicalRegister(name, no_of_bits, ["bit_1"])

        with self.assertRaises(ValueError):
            _ = ClassicalRegister(name, no_of_bits, ["bit_1", "bit_2", "bit_3"])

        with self.assertRaises(ValueError):
            _ = ClassicalRegister(name, no_of_bits, ["bit_1", "bit_1"])

    def test_view_update(self):
        """
        The views within the classical register should be automatically updated when
        the register itself is updated.
        """
        name = "testreg"
        no_of_bits = 2
        bit_ids = ["bit_1", "bit_2"]
        id_bit_reg = {
            "bit_1": 0,
            "bit_2": 0,
        }
        creg = ClassicalRegister(name, no_of_bits, bit_ids)

        # Check Views
        self.assertEqual(creg.no_of_bits, no_of_bits)
        self.assertEqual(creg.id_bit_reg, id_bit_reg)
        self.assertEqual(creg.bit_ids, bit_ids)
        self.assertEqual(creg.bit_reg, [0, 0])

        # New Register
        new_reg = [("bit_1", 0), ("bit_2", 0), ("bit_3", 1)]
        creg.reg = new_reg

        # Check Views have changed
        self.assertEqual(creg.no_of_bits, 3)
        self.assertEqual(creg.id_bit_reg, {"bit_1": 0, "bit_2": 0, "bit_3": 1})
        self.assertEqual(creg.bit_ids, ["bit_1", "bit_2", "bit_3"])
        self.assertEqual(creg.bit_reg, [0, 0, 1])

        # In-Place Changes to the Register would update the view when reg is called.
        creg.reg.append(("bit_4", 0))

        self.assertEqual(creg.no_of_bits, 4)
        self.assertEqual(
            creg.id_bit_reg, {"bit_1": 0, "bit_2": 0, "bit_3": 1, "bit_4": 0}
        )
        self.assertEqual(creg.bit_ids, ["bit_1", "bit_2", "bit_3", "bit_4"])
        self.assertEqual(creg.bit_reg, [0, 0, 1, 0])
        self.assertEqual(
            creg.reg, [("bit_1", 0), ("bit_2", 0), ("bit_3", 1), ("bit_4", 0)]
        )

    def test_invalid_register(self):
        """
        If the register is invalid, contains a value that is non-binary, a ValueError is raised.
        Invalid registers cannot be assigned to the `reg` attribute of Classical
        Register. If the register is assigned, a ValueError is returned.
        """
        name = "testreg"
        no_of_bits = 2
        bit_ids = ["bit_1", "bit_2"]
        creg = ClassicalRegister(name, no_of_bits, bit_ids)

        self.assertEqual(creg.reg, [("bit_1", 0), ("bit_2", 0)])

        # An invalid register, with an invalid bit, is assigned.
        bad_reg = [("bit_1", 2), ("bit_2", 0)]

        with self.assertRaises(ValueError):
            creg.reg = bad_reg

        # An invalid register, with an invalid bit IDs, is assigned.
        bad_reg = [("bit_1", 1), ("bit_1", 0)]

        with self.assertRaises(ValueError):
            creg.reg = bad_reg


if __name__ == "__main__":
    unittest.main()
