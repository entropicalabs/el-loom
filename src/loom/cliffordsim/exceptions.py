"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""


class EngineRunError(BaseException):
    """An Error has occurred when trying to run the Engine."""


class PropagationError(BaseException):
    """An Error has occurred when trying to propagate a PauliFrame"""


class InvalidTableauError(BaseException):
    """An Error has occured when trying to validate the current Tableau during runtime."""


class TableauSizeError(BaseException):
    """Raised when the Tableau provided 2 tableaus do not have the same size."""


class ClassicalRegisterError(BaseException):
    """Raised when trying to perform an action with the classical register."""


class ClassicalOperationError(BaseException):
    """Raised when trying to perform a Classical Operation."""
