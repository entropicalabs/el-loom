"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from copy import deepcopy
from typing import Iterator, Tuple

from .moments.base_moment import Moment
from .operations.base_operation import Operation
from .operations.gate_operation import (
    GateOperation,
    TwoQubitGateOperation,
)
from .operations.resize_operation import ResizeOperation
from .operations.measurement_operation import MeasurementOperation
from .operations.datamanipulation_operation import (
    DataManipulationOperation,
)
from .operations.classical_operation import ClassicalOperation
from .operations.controlled_operation import ControlledOperation


class MomentQueue:
    """This object takes an input of a list of Operation objects, creates a
    Moment objects based on the list then manages the sequence of Moment
    objects.

    The Moment objects will then be sent to the Tableau and
    transform the internal representation.
    """

    def __init__(
        self, input_operations: list[Operation], parallelize: bool = False
    ) -> None:
        self.input_operations = deepcopy(input_operations)
        if parallelize:
            self.parallelized_operations = []
            self._parallel_operation_split(deepcopy(self.input_operations))
        else:
            self.parallelized_operations = [
                [op] for op in deepcopy(self.input_operations)
            ]

        self.moment_generators = self._create_generators(self.parallelized_operations)

    def _create_generators(
        self, parallelised_operations: list[Operation]
    ) -> Iterator[Tuple[Moment, Moment]]:
        """Create a generator that returns a tuple of Moment objects. The first
        Moment object is the forward moment and the second is the backward
        moment.
        """
        n_moments = len(parallelised_operations)
        for i in range(n_moments):
            yield (
                MomentFactory.create_moment(parallelised_operations[i], i),
                MomentFactory.create_moment(
                    parallelised_operations[-i - 1], n_moments - i - 1
                ),
            )

    def _parallel_operation_split(self, input_operations: list[Operation]) -> bool:
        """Parallelize GateOperation Objects in the input list, by returning a
        list of lists."""
        # To Create Exception for Operations that dont have target_qubit.
        while len(input_operations) != 0:
            first_op = input_operations.pop(0)
            parallel_op = [first_op]
            if isinstance(
                first_op,
                (DataManipulationOperation, ClassicalOperation, ControlledOperation),
            ):
                occupied_qubit = []
            else:
                occupied_qubit = [first_op.target_qubit]
            if isinstance(first_op, TwoQubitGateOperation):
                occupied_qubit.append(first_op.control_qubit)
            if isinstance(first_op, GateOperation):
                pop_index = []
                for op_index, next_op in enumerate(input_operations):
                    # Measurement, Resize and DataManipulation Operations are hard walls that
                    # prevent other Operations from being performed
                    if isinstance(
                        next_op,
                        (
                            MeasurementOperation,
                            ResizeOperation,
                            DataManipulationOperation,
                            ClassicalOperation,
                            ControlledOperation,
                        ),
                    ):
                        break
                    if isinstance(next_op, GateOperation):
                        if [
                            qubit_number
                            for qubit_number in next_op.operating_qubit
                            if qubit_number in occupied_qubit
                        ] == []:
                            pop_index.append(op_index)
                            parallel_op.append(next_op)
                        occupied_qubit.extend(next_op.operating_qubit)
                sort_index = sorted(pop_index, reverse=True)
                for each_index in sort_index:
                    input_operations.pop(each_index)
            self.parallelized_operations.append(parallel_op)
        return True

    @property
    def input_operations(self) -> list[Operation]:
        """
        The list of Operation objects that are to be converted into Moment
        objects.
        """
        return self._input_operations

    @input_operations.setter
    def input_operations(self, input_operations: list[Operation]) -> None:
        for each_item in input_operations:
            if not isinstance(each_item, Operation):
                raise TypeError(
                    "The input_operations should be a list of Operation objects."
                )
        self._input_operations = input_operations

    def reset_queue(self):
        """Resets the queue to contain all the instructions again."""
        # self.queue = self.queue_full.copy()
        self.moment_generators = self._create_generators(self.parallelized_operations)


class MomentFactory:  # pylint: disable=too-few-public-methods
    """This object is in charge of creating Moment Objects."""

    @staticmethod
    def create_moment(input_operations: list[Operation], time_step: int) -> Moment:
        """
        Create a Moment object from a list of Operation objects.
        """
        if not input_operations:
            raise ValueError(
                """input_operations cannot be empty. There should be at least 1
                Operation in 1 Moment."""
            )
        # Elements are True if all the inputs are valid Operation Types
        gate_check = [
            isinstance(
                each_operations,
                (
                    GateOperation,
                    MeasurementOperation,
                    ResizeOperation,
                    DataManipulationOperation,
                    ClassicalOperation,
                    ControlledOperation,
                ),
            )
            for each_operations in input_operations
        ]
        if all(gate_check):
            return Moment(input_operations, time_step)
        raise TypeError(
            "There are invalid Operation types in the input_operations list."
        )
