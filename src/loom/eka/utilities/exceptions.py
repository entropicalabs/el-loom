"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""


class AntiCommutationError(Exception):
    """
    Raised when operators that should commute are found to anti-commute.
    """


class SyndromeMissingError(Exception):
    """
    Exception raised when a stabilizer should have been measured.
    """
