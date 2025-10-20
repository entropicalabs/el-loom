"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest

from loom.eka import Lattice, Eka, Block, Stabilizer, PauliOperator
from loom.eka.operations import MeasureBlockSyndromes

from loom.eka.utilities import dumps, loads


class TestEka(unittest.TestCase):
    """
    Tests the Eka class. The tests are not exhaustive but check the most important
    functionality.
    """

    def setUp(self):
        super().setUp()
        # pylint: disable=duplicate-code
        self.rotated_surface_code = Block(
            stabilizers=(
                Stabilizer(
                    pauli="ZZZZ",
                    data_qubits=((1, 0, 0), (1, 1, 0), (0, 0, 0), (0, 1, 0)),
                    ancilla_qubits=((1, 1, 1),),
                ),
                Stabilizer(
                    pauli="ZZZZ",
                    data_qubits=((2, 1, 0), (2, 2, 0), (1, 1, 0), (1, 2, 0)),
                    ancilla_qubits=((2, 2, 1),),
                ),
                Stabilizer(
                    pauli="XXXX",
                    data_qubits=((1, 1, 0), (0, 1, 0), (1, 2, 0), (0, 2, 0)),
                    ancilla_qubits=((1, 2, 1),),
                ),
                Stabilizer(
                    pauli="XXXX",
                    data_qubits=((2, 0, 0), (1, 0, 0), (2, 1, 0), (1, 1, 0)),
                    ancilla_qubits=((2, 1, 1),),
                ),
                Stabilizer(
                    pauli="XX",
                    data_qubits=((0, 0, 0), (0, 1, 0)),
                    ancilla_qubits=((0, 1, 1),),
                ),
                Stabilizer(
                    pauli="XX",
                    data_qubits=((2, 1, 0), (2, 2, 0)),
                    ancilla_qubits=((3, 2, 1),),
                ),
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((2, 0, 0), (1, 0, 0)),
                    ancilla_qubits=((2, 0, 1),),
                ),
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((1, 2, 0), (0, 2, 0)),
                    ancilla_qubits=((1, 3, 1),),
                ),
            ),
            logical_x_operators=[
                PauliOperator(
                    pauli="XXX", data_qubits=((0, 0, 0), (1, 0, 0), (2, 0, 0))
                )
            ],
            logical_z_operators=[
                PauliOperator(
                    pauli="ZZZ", data_qubits=((0, 0, 0), (0, 1, 0), (0, 2, 0))
                )
            ],
            unique_label="q1",
        )

        self.rotated_surface_code_alt = Block(
            stabilizers=(
                Stabilizer(
                    pauli="XXXX",
                    data_qubits=((1, 0, 0), (1, 1, 0), (0, 0, 0), (0, 1, 0)),
                    ancilla_qubits=((1, 1, 1),),
                ),
                Stabilizer(
                    pauli="XXXX",
                    data_qubits=((2, 1, 0), (2, 2, 0), (1, 1, 0), (1, 2, 0)),
                    ancilla_qubits=((2, 2, 1),),
                ),
                Stabilizer(
                    pauli="ZZZZ",
                    data_qubits=((1, 1, 0), (0, 1, 0), (1, 2, 0), (0, 2, 0)),
                    ancilla_qubits=((1, 2, 1),),
                ),
                Stabilizer(
                    pauli="ZZZZ",
                    data_qubits=((2, 0, 0), (1, 0, 0), (2, 1, 0), (1, 1, 0)),
                    ancilla_qubits=((2, 1, 1),),
                ),
                Stabilizer(
                    pauli="XX",
                    data_qubits=((2, 0, 0), (2, 1, 0)),
                    ancilla_qubits=((3, 1, 1),),
                ),
                Stabilizer(
                    pauli="XX",
                    data_qubits=((0, 1, 0), (0, 2, 0)),
                    ancilla_qubits=((0, 2, 1),),
                ),
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((2, 2, 0), (1, 2, 0)),
                    ancilla_qubits=((2, 3, 1),),
                ),
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=((1, 0, 0), (0, 0, 0)),
                    ancilla_qubits=((1, 0, 1),),
                ),
            ),
            logical_x_operators=[
                PauliOperator(
                    pauli="XXX", data_qubits=((0, 0, 0), (1, 0, 0), (2, 0, 0))
                )
            ],
            logical_z_operators=[
                PauliOperator(
                    pauli="ZZZ", data_qubits=((0, 0, 0), (0, 1, 0), (0, 2, 0))
                )
            ],
            unique_label="q1",
        )

        self.square_2d_lattice = Lattice.square_2d((10, 20))
        self.square_2d_lattice_inf = Lattice.square_2d()

    def test_creation_of_eka_without_blocks(self):
        """
        Tests the creation of an Eka object without adding blocks.
        """
        eka = Eka(self.square_2d_lattice)

        self.assertTrue(eka.lattice.n_dimensions == 2)
        self.assertTrue(eka.lattice.unit_cell_size == 2)

    def test_creation_of_eka_non_overlapping_blocks(self):
        """
        Tests the creation of an Eka object with valid, non-overlapping blocks.
        """
        q1 = self.rotated_surface_code
        q2 = q1.shift((0, 4)).rename("q2")
        eka = Eka(self.square_2d_lattice, blocks=[q1, q2])

        self.assertTrue(eka.blocks == (q1, q2))

    def test_loads_dumps(self):
        """
        Test that the loads and dumps functions work correctly.
        """
        q1 = self.rotated_surface_code
        q2 = self.rotated_surface_code.shift((0, 4)).rename("q2")
        eka = Eka(self.square_2d_lattice, blocks=[q1, q2])

        eka_json = dumps(eka)
        loaded_eka = loads(Eka, eka_json)

        self.assertEqual(loaded_eka, eka)

    def test_creation_of_eka_infinite_size(self):
        """
        Tests the creation of an Eka object with infinite lattice. This should
        mainly check that the validation of qubit indices also works for infinite
        lattices.
        """
        q1 = self.rotated_surface_code
        q2 = self.rotated_surface_code.shift((0, 4)).rename("q2")
        eka = Eka(self.square_2d_lattice_inf, blocks=[q1, q2])

        self.assertTrue(eka.lattice.n_dimensions == 2)
        self.assertTrue(eka.lattice.unit_cell_size == 2)
        self.assertTrue(eka.lattice.size is None)

    def test_creation_of_eka_overlapping_blocks(self):
        """
        Tests the creation of an Eka object with overlapping blocks. In this case
        there should be an error raised.
        """
        q1 = self.rotated_surface_code
        q2 = self.rotated_surface_code.shift((0, 2)).rename("q2")

        with self.assertRaises(ValueError) as cm:
            Eka(self.square_2d_lattice, blocks=[q1, q2])

        err_msg = (
            "Block 'q1' and block 'q2' share the data qubits"
            + " {(0, 2, 0), (2, 2, 0), (1, 2, 0)}"
        )
        self.assertIn(err_msg, str(cm.exception))

        # Test that an error is raised when different data qubits are used but one ancilla
        # is used in both blocks
        q1 = self.rotated_surface_code
        q2 = self.rotated_surface_code_alt.shift((0, 2)).rename("q2")

        with self.assertRaises(ValueError):
            Eka(self.square_2d_lattice, blocks=[q1, q2])

    def test_validation_blocks_unique_labels(self):
        """
        Tests that an error is raised in validation when block labels are not
        unique.
        """
        q1 = self.rotated_surface_code
        q2 = self.rotated_surface_code.shift((0, 4))

        with self.assertRaises(ValueError) as cm:
            Eka(self.square_2d_lattice, blocks=[q1, q2])

        err_msg = "Not all blocks have unique labels."
        self.assertIn(err_msg, str(cm.exception))

    def test_validation_blocks_bad_indices(self):
        """
        Tests that an error is raised in validation when data qubit indices are
        invalid (negative or larger than the lattice size).
        """
        q1 = self.rotated_surface_code

        # Check that an error is raised when a data qubit has a negative index
        q2 = q1.shift((-2, 0)).rename("q2")
        with self.assertRaises(ValueError) as cm:
            Eka(self.square_2d_lattice, blocks=[q1, q2])

        err_msg = "Block 'q2' has negative data qubit indices."
        self.assertIn(err_msg, str(cm.exception))

        # Check that an error is raised when a data or and ancilla qubit has an index which is too
        # large for the respective lattice dimension.
        # Since the lattice is 10 x 20, the maximum index for the first dimension is 9.
        # A 3x3 code starting a (6,*) is valid while starting at (7,*) is not.
        q3 = q1.shift((6, 0)).rename("q3")
        Eka(self.square_2d_lattice, blocks=[q1, q3])

        q4 = q1.shift((7, 0)).rename("q4")
        with self.assertRaises(ValueError) as cm:
            Eka(self.square_2d_lattice, blocks=[q1, q4])

        err_msg = (
            "Block 'q4' has ancilla qubit indices which are too large for the lattice."
        )
        self.assertIn(err_msg, str(cm.exception))

        # Check the 2nd dimension as well
        q5 = q1.shift((0, 16)).rename("q5")
        Eka(self.square_2d_lattice, blocks=[q1, q5])

        q6 = q1.shift((0, 17)).rename("q6")
        with self.assertRaises(ValueError) as cm:
            Eka(self.square_2d_lattice, blocks=[q1, q6])

        err_msg = (
            "Block 'q6' has ancilla qubit indices which are too large for the lattice."
        )
        self.assertIn(err_msg, str(cm.exception))

    def test_creation_of_eka_with_operations(self):
        """
        Tests the creation of an Eka object with operations.
        """
        q1 = self.rotated_surface_code
        q2 = self.rotated_surface_code.shift((0, 4)).rename("q2")

        meas_q1 = MeasureBlockSyndromes(q1.unique_label)
        meas_q2 = MeasureBlockSyndromes(q2.unique_label)

        eka = Eka(
            self.square_2d_lattice,
            blocks=[q1, q2],
            operations=[meas_q1, meas_q2],
        )

        self.assertEqual(eka.blocks, (q1, q2))
        self.assertEqual(eka.operations, ((meas_q1,), (meas_q2,)))

        # Test that tuple[tuple[Operation, ...], ...] is also accepted and yields the same result
        eka_eq = Eka(
            self.square_2d_lattice,
            blocks=[q1, q2],
            operations=[(meas_q1,), (meas_q2,)],
        )
        self.assertEqual(eka, eka_eq)


if __name__ == "__main__":
    unittest.main()
