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
