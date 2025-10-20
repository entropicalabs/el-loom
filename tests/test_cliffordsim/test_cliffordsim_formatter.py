"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest

from loom.cliffordsim.engine import Engine
from loom.cliffordsim.operations import (
    Hadamard,
    CNOT,
    X,
    Phase,
    CreatePauliFrame,
    RecordPauliFrame,
)
from loom.cliffordsim.pauli_frame import PauliFrame


class TestCliffordSimFormatter(unittest.TestCase):
    """
    We test the standard and sparse format for CliffordSim output.
    """

    def test_circuit_standard_output(self):
        """
        Just make sure that circuit still runs as expected and
        the standard output passes expectations.
        """
        operation_list = [
            Hadamard(0),
            CNOT(0, 1),
            # X error on qubit 2
            X(1),
            # bell prepare ancilla states
            Hadamard(2),
            Hadamard(4),
            CNOT(2, 3),
            CNOT(4, 5),
            # "join" between pair
            CNOT(3, 4),
            # couple data and ancilla
            CNOT(0, 2),
            CNOT(1, 4),
            # just to spice things up
            Phase(0),
        ]
        cliffordsim_engine = Engine(operation_list, 6, None)
        cliffordsim_engine.run()

        expected_stab = {
            "+____XX",
            "+_Z_ZZZ",
            "-ZZ____",
            "+__XXX_",
            "+YXX_X_",
            "+Z_ZZ__",
        }
        self.assertEqual(
            expected_stab, cliffordsim_engine.tableau_w_scratch.stabilizer_set
        )

    def test_sparse_format(self):
        """
        Test the sparse format for CliffordSim output
        """
        operation_list = [
            Hadamard(0),
            CNOT(0, 1),
            # X error on qubit 2
            X(1),
            # bell prepare ancilla states
            Hadamard(2),
            Hadamard(4),
            CNOT(2, 3),
            CNOT(4, 5),
            # "join" between pair
            CNOT(3, 4),
            # couple data and ancilla
            CNOT(0, 2),
            CNOT(1, 4),
            # just to spice things up
            Phase(0),
        ]
        cliffordsim_engine = Engine(operation_list, 6, None)
        cliffordsim_engine.run()

        expected_format = [
            {"X": [4, 5], "Y": [], "Z": [], "sign": "+"},
            {"X": [], "Y": [], "Z": [1, 3, 4, 5], "sign": "+"},
            {"X": [], "Y": [], "Z": [0, 1], "sign": "-"},
            {"X": [2, 3, 4], "Y": [], "Z": [], "sign": "+"},
            {"X": [1, 2, 4], "Y": [0], "Z": [], "sign": "+"},
            {"X": [], "Y": [], "Z": [0, 2, 3], "sign": "+"},
        ]

        computed_output = cliffordsim_engine.stabilizer_set_sparse_format
        for op in expected_format:
            self.assertTrue(op in computed_output)

    def test_pf_standard_output(self):
        """
        Test the standard format for Pauliframe string output
        """
        operation_list = [
            Hadamard(0),
            CNOT(0, 1),
            # X error on qubit 2
            X(1),
            # bell prepare ancilla states
            Hadamard(2),
            Hadamard(4),
            CNOT(2, 3),
            CNOT(4, 5),
            # "join" between pair
            CNOT(3, 4),
            # couple data and ancilla
            CNOT(0, 2),
            CNOT(1, 4),
            # just to spice things up
            Phase(0),
        ]
        test_pf = "Z" + "_" * 5
        pf = PauliFrame.from_string(test_pf)
        # Add the PauliFrame and record it
        operation_list = (
            [CreatePauliFrame(pf)] + operation_list + [RecordPauliFrame(pf)]
        )
        cliffordsim_engine = Engine(operation_list, 6, None)

        cliffordsim_engine.run()
        pf_records = cliffordsim_engine.data_store.pf_records["forward"]
        time_step = pf_records["time_step"][0]
        pf_out = pf_records[str(time_step)][pf.id]["recorded_pauli_frame"]
        expected_output = "PauliFrame: YXX_X_"
        self.assertEqual(expected_output, repr(pf_out))

    def test_pf_sparse_output(self):
        """
        Test the sparse format for Pauliframe string output
        """
        operation_list = [
            Hadamard(0),
            CNOT(0, 1),
            # X error on qubit 2
            X(1),
            # bell prepare ancilla states
            Hadamard(2),
            Hadamard(4),
            CNOT(2, 3),
            CNOT(4, 5),
            # "join" between pair
            CNOT(3, 4),
            # couple data and ancilla
            CNOT(0, 2),
            CNOT(1, 4),
            # just to spice things up
            Phase(0),
        ]
        test_pf = "Z" + "_" * 5
        pf = PauliFrame.from_string(test_pf)

        operation_list = [
            CreatePauliFrame(pf),
            *operation_list,
            RecordPauliFrame(pf),
        ]
        cliffordsim_engine = Engine(operation_list, 6, None)

        cliffordsim_engine.run()
        pf_records = cliffordsim_engine.data_store.pf_records["forward"]
        time_step = pf_records["time_step"][0]
        pf_out = pf_records[str(time_step)][pf.id]["recorded_pauli_frame"]
        expected_sparse_output = [{"sign": "+", "X": [1, 2, 4], "Y": [0], "Z": []}]
        self.assertEqual(expected_sparse_output, pf_out.sparse_format())


if __name__ == "__main__":
    unittest.main()
