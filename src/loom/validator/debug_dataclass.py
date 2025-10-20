"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from pydantic import Field
from pydantic.dataclasses import dataclass

from .check_code_stabilizers import CodeStabilizerCheck
from .check_logical_ops import LogicalOperatorCheck
from .check_stabilizer_measurement import StabilizerMeasurementCheck


@dataclass(frozen=True)
class AllChecks:
    """
    Dataclass to store the results of all checks.

    Parameters
    ----------
    CodeStabilizers: CodeStabilizerCheck
        The results of the Code Stabilizer check.
    LogicalState: LogicalOperatorCheck
        The results of the Logical Operator check.
    StabilizersMeasured: StabilizerMeasurementCheck
        The results of the Stabilizer Measurement check.
    """

    code_stabilizers: CodeStabilizerCheck
    logical_operators: LogicalOperatorCheck
    stabilizers_measured: StabilizerMeasurementCheck

    def __iter__(self):
        """Iterate over the checks."""
        return iter(self.__dict__.values())


@dataclass(frozen=True)
class DebugData:
    """Dataclass to store the results of validator verification checks.

    Parameters
    ----------
    checks: AllChecks
        The results of all checks.
    valid: bool
        True if all checks have passed, False otherwise.
    """

    checks: AllChecks
    valid: bool = Field(init_var=False, default=True)

    def __post_init__(self):
        # Check if all checks are valid and set the valid attribute
        valid = all(check.valid for check in self.checks)
        object.__setattr__(self, "valid", valid)

    def __str__(self):
        """String representation of the DebugData."""
        out = f"DebugData: valid = {self.valid}\n"
        for check in self.checks:
            if not check.valid:
                # Add a separator for better readability
                out += "-" * 40 + "\n"
                out += str(check) + "\n"
        return out.rstrip("\n")
