"""
Copyright 2024 Entropica Labs Pte Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

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
