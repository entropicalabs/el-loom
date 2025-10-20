"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest

from loom.validator.debug_dataclass import (
    CodeStabilizerCheck,
    LogicalOperatorCheck,
    StabilizerMeasurementCheck,
    AllChecks,
)

from loom.validator.check_stabilizer_measurement import StabilizerMeasurementCheckOutput
from loom.validator.check_code_stabilizers import CodeStabilizerCheckOutput
from loom.validator.check_logical_ops import LogicalOperatorCheckOutput


class TestDebugData(unittest.TestCase):
    """
    Test cases for the debug dataclasses used in the Validator module."
    """

    def test_all_check_iter(self):
        """
        Test the __iter__ method of AllChecks.
        """
        all_checks = AllChecks(
            code_stabilizers=CodeStabilizerCheck(
                output=CodeStabilizerCheckOutput((), ())
            ),
            logical_operators=LogicalOperatorCheck(
                output=LogicalOperatorCheckOutput((), ())
            ),
            stabilizers_measured=StabilizerMeasurementCheck(
                output=StabilizerMeasurementCheckOutput((), ())
            ),
        )

        for check in all_checks:
            self.assertIsInstance(
                check,
                (CodeStabilizerCheck, LogicalOperatorCheck, StabilizerMeasurementCheck),
            )
            self.assertTrue(check.valid)


if __name__ == "__main__":
    unittest.main()
