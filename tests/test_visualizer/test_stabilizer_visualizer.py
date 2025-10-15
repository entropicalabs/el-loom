"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest

from loom.eka import Lattice, PauliOperator, Stabilizer, Block
import loom.visualizer as vis


class TestStabilizerPlot(unittest.TestCase):
    """Unit tests for the StabilizerPlot class in the visualizer module."""

    def setUp(self):
        self.dx = 3
        self.dz = 3
        self.lattice = Lattice.square_2d((self.dx + 2, self.dz + 2))
        # pylint: disable=duplicate-code
        self.q1 = Block(
            unique_label="q1",
            stabilizers=[
                Stabilizer(
                    "ZZZZ",
                    ((2, 1, 0), (1, 1, 0), (2, 2, 0), (1, 2, 0)),
                    ancilla_qubits=((2, 2, 1),),
                ),
                Stabilizer(
                    "ZZZZ",
                    ((3, 2, 0), (2, 2, 0), (3, 3, 0), (2, 3, 0)),
                    ancilla_qubits=((3, 3, 1),),
                ),
                Stabilizer(
                    "XXXX",
                    ((2, 2, 0), (2, 3, 0), (1, 2, 0), (1, 3, 0)),
                    ancilla_qubits=((2, 3, 1),),
                ),
                Stabilizer(
                    "XXXX",
                    ((3, 1, 0), (3, 2, 0), (2, 1, 0), (2, 2, 0)),
                    ancilla_qubits=((3, 2, 1),),
                ),
                Stabilizer(
                    "XX",
                    ((1, 1, 0), (1, 2, 0)),
                    ancilla_qubits=((1, 2, 1),),
                ),
                Stabilizer(
                    "XX",
                    ((3, 2, 0), (3, 3, 0)),
                    ancilla_qubits=((4, 3, 1),),
                ),
                Stabilizer(
                    "ZZ",
                    ((3, 1, 0), (2, 1, 0)),
                    ancilla_qubits=((3, 1, 1),),
                ),
                Stabilizer(
                    "ZZ",
                    ((2, 3, 0), (1, 3, 0)),
                    ancilla_qubits=((2, 4, 1),),
                ),
            ],
            logical_x_operators=[
                PauliOperator("XXX", [(1, 1, 0), (2, 1, 0), (3, 1, 0)])
            ],
            logical_z_operators=[
                PauliOperator("ZZZ", [(1, 1, 0), (1, 2, 0), (1, 3, 0)])
            ],
        )

    def test_create_stabilizer_plot_object(self):
        """Tests whether a StabilizerPlot object is created without any errors."""
        title = f"Rotated Surface Code, {self.dx} x {self.dz}"
        # pylint: disable=protected-access
        stab_plot = vis.StabilizerPlot(
            self.lattice,
            title=title,
        )

        len_data = len(stab_plot._fig.data)
        self.assertEqual(
            len_data, 1
        )  # StabilizerPlot should contain one dummy trace for the data qubits

        # Test get_stabilizer_traces()
        stab_traces = stab_plot.get_stabilizer_traces(self.q1.stabilizers)
        self.assertTrue(
            len(stab_traces) == len(self.q1.stabilizers)
        )  # Check that the right number of traces is created

        # Add stabilizers
        stab_plot.add_stabilizers(self.q1.stabilizers)
        len_data_new = len(stab_plot._fig.data)
        self.assertEqual(
            len_data_new, len_data + len(self.q1.stabilizers)
        )  # There should be n additional traces now where n is the number of stabilizers
        len_data = len_data_new

        # Test get_dqubit_traces()
        dqubit_traces = stab_plot.get_dqubit_traces()
        self.assertTrue(
            len(dqubit_traces)
            == (self.lattice.size[0] * self.lattice.size[1])
            * len(self.lattice.basis_vectors)
        )  # Check that the right number of traces is created

        # Add data qubits
        stab_plot.add_dqubit_traces()
        len_data_new = len(stab_plot._fig.data)
        self.assertEqual(
            len_data_new,
            len_data
            + (self.lattice.size[0] * self.lattice.size[1])
            * len(self.lattice.basis_vectors),
        )  # There should be n additional traces now where n is the number of data qubits

        # Other checks
        self.assertEqual(stab_plot._fig.layout.title.text, title)

    def test_plot_pauli_string(self):
        """Tests the plotting of a pauli operator."""
        # pylint: disable=protected-access
        stab_plot = vis.StabilizerPlot(
            self.lattice,
        )
        stab_plot.plot_pauli_string(
            PauliOperator("XYZ", [(0, 0, 0), (0, 1, 0), (0, 2, 0)]),
            connecting_line=True,
        )
        self.assertEqual(
            len(stab_plot._fig.data),
            1  # Dummy for data qubit legend
            + 3  # 3 X symbols on the involved data qubits
            + 1,  # Connecting line
        )

    def test_plot_logical_operators(self):
        """Tests whether logical operators are plotted without any errors."""
        # pylint: disable=protected-access
        stab_plot = vis.StabilizerPlot(
            self.lattice,
        )
        stab_plot.plot_blocks(self.q1)
        self.assertEqual(
            len(stab_plot._fig.data),
            1  # Dummy for data qubit legend
            + 1  # Dummy for logical qubit
            + self.dx * self.dz
            - 1  # Stabilizers
            + 2  # Dummy for each logical operator
            + 6,
        )

    def test_plot_pauli_charges(self):
        """Tests whether pauli charges are plotted without any errors."""
        # pylint: disable=protected-access
        stab_plot = vis.StabilizerPlot(
            self.lattice,
        )
        stab_plot.plot_blocks(
            self.q1, plot_logical_operators=False, plot_pauli_charges=True
        )
        self.assertEqual(
            len(stab_plot._fig.data),
            1  # Dummy for data qubit legend
            + 1  # Dummy for logical qubit
            + self.dx * self.dz
            - 1  # Stabilizers
            + 8,  # Pauli charges
        )


if __name__ == "__main__":
    unittest.main()
