"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest

from loom.cliffordsim.moment_queue import MomentQueue
from loom.cliffordsim.operations import Hadamard, Phase, CNOT, Measurement, X


class TestCliffordSimMomentQueue(unittest.TestCase):
    """
    We test the MomentQueue class and its associated functionality.
    """

    def test_parallel_op_gate(self):
        """We check that for a particular set of gates, the operation has been
        parallelized correctly."""
        operation_list = [
            Hadamard(0),
            Phase(1),
            Hadamard(1),
            CNOT(0, 1),
            Phase(0),
        ]
        final_parallel_op = [
            [Hadamard(0), Phase(1)],
            [Hadamard(1)],
            [CNOT(0, 1)],
            [Phase(0)],
        ]

        moment_q = MomentQueue(operation_list, parallelize=True)

        self.assertEqual(moment_q.parallelized_operations, final_parallel_op)
        self.assertEqual(moment_q.input_operations, operation_list)

        flatten_parallel_op = [
            each_op
            for each_list in moment_q.parallelized_operations
            for each_op in each_list
        ]
        self.assertEqual(len(operation_list), len(flatten_parallel_op))

    def test_parallel_op_gate_2(self):
        """We check that for a particular set of gates, the operation has been
        parallelized correctly."""
        operation_list = [
            Hadamard(0),
            CNOT(0, 1),
            Hadamard(1),
            CNOT(1, 0),
            Phase(2),
        ]
        final_parallel_op = [
            [Hadamard(0), Phase(2)],
            [CNOT(0, 1)],
            [Hadamard(1)],
            [CNOT(1, 0)],
        ]

        moment_q = MomentQueue(operation_list, parallelize=True)

        self.assertEqual(moment_q.parallelized_operations, final_parallel_op)
        self.assertEqual(moment_q.input_operations, operation_list)

        flatten_parallel_op = [
            each_op
            for each_list in moment_q.parallelized_operations
            for each_op in each_list
        ]
        self.assertEqual(len(operation_list), len(flatten_parallel_op))

    def test_parallel_op_gate_3(self):
        """We check that for a particular set of gates, the operation has been
        parallelized correctly."""
        operation_list = [
            Hadamard(0),
            CNOT(0, 1),
            CNOT(1, 2),
            CNOT(0, 3),
            CNOT(1, 3),
            CNOT(0, 4),
            CNOT(2, 4),
            Hadamard(0),
            Hadamard(1),
            Hadamard(2),
        ]
        final_parallel_op = [
            [Hadamard(0)],
            [CNOT(0, 1)],
            [CNOT(1, 2), CNOT(0, 3)],
            [CNOT(1, 3), CNOT(0, 4)],
            [CNOT(2, 4), Hadamard(0), Hadamard(1)],
            [Hadamard(2)],
        ]

        moment_q = MomentQueue(operation_list, parallelize=True)

        self.assertEqual(moment_q.parallelized_operations, final_parallel_op)
        self.assertEqual(moment_q.input_operations, operation_list)

        flatten_parallel_op = [
            each_op
            for each_list in moment_q.parallelized_operations
            for each_op in each_list
        ]
        self.assertEqual(len(operation_list), len(flatten_parallel_op))

    def test_parallel_op_gate_4(self):
        """When parallelizing Operations, the Measurement Operation cannot be
        parallelized with other Operations."""
        meas_op_0 = Measurement(0)
        meas_op_1 = Measurement(1)

        operation_list = [
            Hadamard(0),
            CNOT(0, 1),
            meas_op_0,
            X(0),
            X(1),
            meas_op_1,
        ]
        final_parallel_op = [
            [Hadamard(0)],
            [CNOT(0, 1)],
            [meas_op_0],
            [X(0), X(1)],
            [meas_op_1],
        ]

        moment_q = MomentQueue(operation_list, parallelize=True)

        self.assertEqual(moment_q.parallelized_operations, final_parallel_op)
        self.assertEqual(moment_q.input_operations, operation_list)

        flatten_parallel_op = [
            each_op
            for each_list in moment_q.parallelized_operations
            for each_op in each_list
        ]
        self.assertEqual(len(operation_list), len(flatten_parallel_op))

    def test_non_parallel_op_gate(self):
        """We check that when disabling the parallelisation, each element is grouped
        one by one"""
        operation_list = [
            Hadamard(0),
            CNOT(0, 1),
            CNOT(1, 2),
            CNOT(0, 3),
            CNOT(1, 3),
            CNOT(0, 4),
            CNOT(2, 4),
            Hadamard(0),
            Hadamard(1),
            Hadamard(2),
        ]
        moment_q = MomentQueue(operation_list, parallelize=False)

        non_parallel_op = [[op] for op in operation_list]

        self.assertEqual(moment_q.parallelized_operations, non_parallel_op)
        self.assertEqual(moment_q.input_operations, operation_list)


if __name__ == "__main__":
    unittest.main()
