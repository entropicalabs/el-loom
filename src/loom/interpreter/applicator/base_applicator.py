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

from loom.eka import Eka
from loom.eka.operations import Operation

from ..interpretation_step import InterpretationStep


class BaseApplicator:  # pylint: disable=too-few-public-methods
    """
    Base Class for Applicators. BaseOperations are implemented at the level of this
    BaseApplicator. For CodeOperations and LogicalOperations, subclasses are used.

    Classmethods of this class evolve the InterpretationStep object, by applying the
    operation to it. The InterpretationStep object is then returned.
    The signature of applicator methods should be:
    def operation_name(self, interpretation_step: InterpretationStep,
    operation: Operation, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    """

    def __init__(
        self,
        eka: Eka,
    ):
        # Define supported base operations
        self.supported_operations = {}
        self.eka = eka

    def apply(
        self,
        interpretation_step: InterpretationStep,
        operation: Operation,
        same_timeslice: bool,
        debug_mode: bool,
    ) -> InterpretationStep:
        """
        If the `operation` can be found in the `supported_operations` dictionary,
        this method applies the operation to the input `interpretation_step`.

        The class name of the operation is used as a way to determine
        if the operation is supported by this applicator.

        Parameters
        ----------
        interpretation_step : InterpretationStep
            Interpretation step containing the blocks to be modified.
        operation : :class:`loom.eka.operations.base_operation.Operation`
            The operation to be applied. It should be an instance of a class
            that is registered in `supported_operations`.
        same_timeslice : bool
            Flag indicating whether the operation is part of the same timestep as the
            previous operation.
        debug_mode : bool
            Flag indicating whether the interpretation should be done in debug mode.
            Currently, the effects of debug mode are:
            - Disabling the commutation validation of Block

        Returns
        -------
        InterpretationStep
            Interpretation step after the operation is applied.
        """
        operation_function = self.supported_operations.get(operation.__class__.__name__)
        if operation_function is None:
            raise NotImplementedError(
                f"Operation {operation.__class__.__name__} is not supported by "
                f"{self.__class__.__name__}"
            )
        return operation_function(
            interpretation_step, operation, same_timeslice, debug_mode
        )
