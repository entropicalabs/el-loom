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

from abc import ABC
from typing import Any
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class AbstractValidityCheck(ABC):
    """Abstract dataclass to store the results of a check.

    Parameters
    ----------
    output: Any
        Any output that is relevant to the check. It has to have a __len__ method to be
        used to determine if the check is valid.

    Properties
    ----------
    valid: bool
        True if the check is valid (i.e., no issues found), False otherwise.
    message: str
        A message indicating the result of the check. It will be empty if the check is
        valid, otherwise it will contain a message describing the issue. This must be
        implemented by subclasses.
    """

    output: Any

    def __new__(cls, *args, **kwargs):  # pylint: disable=unused-argument
        if cls == AbstractValidityCheck:
            raise TypeError("Cannot instantiate abstract ValidityCheck class.")
        return super().__new__(cls)

    def __str__(self):
        out = f"{self.__class__.__name__}: valid = {self.valid}\n"
        if not self.valid:
            out += f"message: {self.message}\n"
            out += "output: \n"
            out += str(self.output) + "\n"
        return out.rstrip("\n")  # Remove the last newline character for cleaner output

    @property
    def valid(self) -> bool:
        """
        Determine if the check passed based on the output.
        """
        return len(self.output) == 0

    @property
    def message(self) -> str:
        """
        A message indicating the result of the check. It will be empty if the check is
        valid, otherwise it will contain a message describing the issue.
        """
        raise NotImplementedError(
            "Subclasses must implement the 'message' property to provide a message."
        )
