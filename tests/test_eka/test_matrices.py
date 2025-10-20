"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import unittest
import networkx as nx
import numpy as np

from loom.eka.utilities import (
    binary_gaussian_elimination,
    verify_css_code_condition,
)
from loom.eka import (
    Stabilizer,
    ClassicalParityCheckMatrix,
    ParityCheckMatrix,
)
from loom.eka.tanner_graphs import ClassicalTannerGraph, TannerGraph


# pylint: disable=duplicate-code
class TestMatrices(unittest.TestCase):
    """
    Test for Eka utilities.
    """

    def test_bge_non_square_matrix(self):
        """
        Tests binary gaussian elimination with a non-square matrix. Non-square
        matrices MxN, M<N do not have a pivot row for every column.
        """
        array = np.array(
            [
                [1, 1, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
            ]
        )

        array_bge = np.array(
            [
                [1, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        self.assertTrue(np.array_equal(binary_gaussian_elimination(array), array_bge))


# pylint: disable=invalid-name, too-many-lines, too-many-instance-attributes, too-many-statements, too-many-locals
class TestMatricesUtilities(unittest.TestCase):
    """Unit tests for matrix utilities in the eka module."""

    def setUp(self):

        ### HAMMING AND STEANE CODE

        # Parity-check matrix for a component of the Hamming code
        self.H_hamming = np.array(
            [[1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1]],
            dtype=int,
        )

        # Edges corresponding to the Tanner graph of Hamming code
        data_supports_ham = [[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 5, 6]]
        datas_ham = list(range(7))
        checks_ham = list(range(7, 10))

        self.nodes_hamming = [((i,), {"label": "check"}) for i in checks_ham] + [
            ((d,), {"label": "data"}) for d in datas_ham
        ]
        self.edges_hamming = [
            ((c,), (d,))
            for c, supp in zip(checks_ham, data_supports_ham, strict=True)
            for d in supp
        ]

        # Tanner Graph for Hamming code
        self.T_hamming = nx.Graph()
        self.T_hamming.add_nodes_from(self.nodes_hamming)
        self.T_hamming.add_edges_from(self.edges_hamming)

        # Parity-check matrix for full quantum code
        self.H_steane = np.vstack(
            (
                np.hstack((self.H_hamming, np.zeros(self.H_hamming.shape, dtype=int))),
                np.hstack((np.zeros(self.H_hamming.shape, dtype=int), self.H_hamming)),
            )
        )

        # Steane code Tanner graph
        self.T_steane = nx.Graph()

        # Add nodes
        self.x_nodes_steane = [((i,), {"label": "X"}) for i in checks_ham]
        self.z_nodes_steane = [((i + 3,), {"label": "Z"}) for i in checks_ham]
        self.data_nodes_steane = [((i,), {"label": "data"}) for i in datas_ham]
        self.T_steane.add_nodes_from(self.x_nodes_steane)
        self.T_steane.add_nodes_from(self.z_nodes_steane)
        self.T_steane.add_nodes_from(self.data_nodes_steane)

        # Add edges
        self.x_edges_steane = self.edges_hamming
        self.z_edges_steane = [((c[0] + 3,), d) for c, d in self.edges_hamming]
        self.T_steane.add_edges_from(self.x_edges_steane + self.z_edges_steane)

        # Steane code stabilizers
        self.stabilizers_steane = [
            Stabilizer(
                pauli=p * 4,
                data_qubits=[(d, 0) for d in support],
                ancilla_qubits=[(i + 3 * j, 1)],
            )
            for i, support in enumerate(data_supports_ham)
            for j, p in enumerate("XZ")
        ]

        ### SHOR CODE VARIABLES

        # Parity-check matrices for the Shor code
        self.Hx_shor = np.array(
            [[1, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1, 1]]
        )
        self.Hz_shor = np.array(
            [
                [1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1],
            ]
        )
        self.H_shor = np.vstack(
            (
                np.hstack((self.Hx_shor, np.zeros(self.Hx_shor.shape, dtype=int))),
                np.hstack((np.zeros(self.Hz_shor.shape, dtype=int), self.Hz_shor)),
            )
        )

        # Tanner graph components and full graph for Shor Code
        self.Tx_shor = nx.Graph()
        self.Tz_shor = nx.Graph()
        self.T_shor = nx.Graph()

        self.data_nodes_shor = [((i,), {"label": "data"}) for i in range(9)]
        self.x_nodes_shor = [((i,), {"label": "X"}) for i in range(9, 11)]
        self.z_nodes_shor = [((i,), {"label": "Z"}) for i in range(11, 17)]

        self.Tx_shor.add_nodes_from(self.x_nodes_shor)
        self.Tx_shor.add_nodes_from(self.data_nodes_shor)

        self.Tz_shor.add_nodes_from(self.z_nodes_shor)
        self.Tz_shor.add_nodes_from(self.data_nodes_shor)

        self.T_shor.add_nodes_from(self.x_nodes_shor)
        self.T_shor.add_nodes_from(self.z_nodes_shor)
        self.T_shor.add_nodes_from(self.data_nodes_shor)

        self.x_edges_shor = [
            ((i + 9,), (j + 3 * i,)) for i in range(2) for j in range(6)
        ]
        self.z_edges_shor = [
            ((i + 11,), (j + 3 * (i // 2) + (i % 2),))
            for i in range(6)
            for j in range(2)
        ]
        self.Tx_shor.add_edges_from(self.x_edges_shor)
        self.Tz_shor.add_edges_from(self.z_edges_shor)

        self.T_shor.add_edges_from(self.x_edges_shor + self.z_edges_shor)

        # Stabilizers for the Shor code
        x_indices = [[(i + 3 * j, 0) for i in range(6)] for j in range(2)]
        z_indices = [
            [(i + j + 3 * k, 0) for i in range(2)] for k in range(3) for j in range(2)
        ]
        paulis = ["X" * 6, "Z" * 2]

        self.stabilizers_shor = [
            Stabilizer(
                pauli=paulis[i],
                data_qubits=data_qubits,
                ancilla_qubits=[(j + 2 * i, 1)],
            )
            for i, indices in enumerate([x_indices, z_indices])
            for j, data_qubits in enumerate(indices)
        ]

        ### d=3 ROTATED SURFACE CODE VARIABLES

        # Parity-check matrix for the full quantum code
        self.H_rsc = np.array(
            [
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            ]
        )

        # Tanner Graph
        self.T_rsc = nx.Graph()

        # Add notes
        self.data_nodes_rsc = [((i,), {"label": "data"}) for i in range(9)]
        self.x_nodes_rsc = [((i,), {"label": "X"}) for i in range(9, 13)]
        self.z_nodes_rsc = [((i,), {"label": "Z"}) for i in range(13, 17)]
        self.T_rsc.add_nodes_from(self.x_nodes_rsc)
        self.T_rsc.add_nodes_from(self.z_nodes_rsc)
        self.T_rsc.add_nodes_from(self.data_nodes_rsc)

        # Add edges
        x_data_supports_rsc = [[0, 3], [1, 2, 4, 5], [3, 4, 6, 7], [5, 8]]
        self.x_edges_rsc = [
            ((x,), (d,))
            for x, supp in zip(range(9, 13), x_data_supports_rsc, strict=True)
            for d in supp
        ]

        z_data_supports_rsc = [[1, 2], [0, 1, 3, 4], [4, 5, 7, 8], [6, 7]]
        self.z_edges_rsc = [
            ((z,), (d,))
            for z, supp in zip(range(13, 17), z_data_supports_rsc, strict=True)
            for d in supp
        ]

        self.T_rsc.add_edges_from(self.x_edges_rsc + self.z_edges_rsc)

        ### REPETITION CODE VARIABLES
        self.distance_rep = 21

        self.H_rep = np.eye(
            self.distance_rep - 1, self.distance_rep, dtype=int
        ) + np.eye(self.distance_rep - 1, self.distance_rep, k=1, dtype=int)

        # Nodes and edges  corresponding to a Tanner graph with a single component
        self.nodes_bitflip_rep = [
            ((i,), {"label": "data"}) for i in range(self.distance_rep)
        ] + [
            ((i,), {"label": "Z"})
            for i in range(self.distance_rep, 2 * self.distance_rep - 1)
        ]
        self.edges_bitflip_rep = [
            ((i + j,), (self.distance_rep + i,))
            for i in range(self.distance_rep - 1)
            for j in range(2)
        ]

        # Tanner Graph for bitflip repetition code
        self.T_bitflip_rep = nx.Graph()
        self.T_bitflip_rep.add_nodes_from(self.nodes_bitflip_rep)
        self.T_bitflip_rep.add_edges_from(self.edges_bitflip_rep)

        # [[5,1,3]] LAFLAMME CODE (aka 5q perfect code)
        self.H_laflamme = np.array(
            [
                [1, 0, 0, 1, 0, 0, 1, 1, 0, 0],  # XZZXI = (XIIXI)(IZZII)
                [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],  # IXZZX = (IXIIX)(IIZZI)
                [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],  # XIXZZ = (XIXII)(IIIZZ)
                [0, 1, 0, 1, 0, 1, 0, 0, 0, 1],  # ZXIXZ = (IXIXI)(ZIIIZ)
            ]
        )

        ### INVALID INPUTS AND ERROR MESSAGES

        # Invalid inputs for initalizing ClassicalParityCheckMatrix or ParityCheckMatrix
        self.invalid_matrix_inputs = [
            {0: [1, 0, 1], 1: [0, 1, 1]},
            "[[1,0,1],[1,0,1]]",
            ([1, 0, 1], [1, 0, 1]),
            [(1, 0, 1), (1, 0, 1)],
        ]

        # Check for non-binary matrix inputs
        self.H_nb_err = np.array([[1, 2, 3], [4, 5, 6]])
        self.err_msg_H_nb = "Parity-check matrix contains non-binary elements."

        # Check for empty matrices
        self.H_zeros = np.array([[0, 0], [0, 0]])
        self.err_msg_H_empty = "Parity-check matrix is empty."

        # Check for 2D array
        self.non_2D_H = np.array([[[1, 1], [0, 1]]])
        self.err_msg_non_2D = "Parity-check matrix must be a 2D array."

        # Check for erroneous input in quantum CSS codes Tanner graph
        self.T_err1 = nx.Graph()  # Non-allowed label
        self.T_err1.add_nodes_from([1, 2, 3], label="X")
        self.T_err1.add_nodes_from([4, 5, 6], label="entropica")
        self.T_err1.add_nodes_from([7, 8], label="Z")
        self.T_err1.add_nodes_from([9, 10], label="data")
        self.err1_msg = (
            "Tanner graph should describe a quantum CSS code and thus"
            " can only contain 'X', 'Z', and 'data' nodes."
        )

        self.T_err2 = nx.Graph()  # Missing data nodes
        self.T_err2.add_nodes_from([1, 2, 3], label="X")
        self.T_err2.add_nodes_from([4, 5, 6], label="Z")
        self.err2_msg = "Tanner graph does not contain any 'data' nodes."

        self.T_err3 = nx.Graph()  # Missing check nodes
        self.T_err3.add_nodes_from([1, 2, 3], label="data")
        self.err3_msg = "Tanner graph must contain 'X' and 'Z' check nodes."

        self.T_err4 = nx.Graph()  # Unlabelled edges
        labeled_edges = [
            (0, 1, {"cardinality": "E"}),
            (1, 2, {"cardinality": "N"}),
            (2, 3, {"cardinality": "S"}),
        ]
        unlabelled_edges = [(0, 5), (3, 4), (5, 4)]
        self.T_err4.add_edges_from(labeled_edges + unlabelled_edges)
        self.err4_msg = f"Edges {unlabelled_edges} do not contain input label."

    def test_verify_css_code_condition(self):
        """Test that condition for a CSS code is correctly identified."""

        # EXAMPLE 1 - Steane code

        # Both X and Z components are equal
        Hx, Hz = self.H_hamming, self.H_hamming

        # Compute full partity-check matrix and verify condition
        valid = verify_css_code_condition(Hx, Hz)

        # The condition should be satisfied by definition of the Steane Code
        self.assertTrue(valid)

        # EXAMPLE 2 - Shor code
        valid = verify_css_code_condition(self.Hx_shor, self.Hz_shor)
        self.assertTrue(valid)

        # EXAMPLE 3 - Random invalid code

        # Random parity-check matrix
        Hx = np.array([[1, 1, 0, 1], [0, 1, 1, 0], [0, 0, 1, 1]])
        Hz = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])

        # Verify condition is False
        valid = verify_css_code_condition(Hx, Hz)
        self.assertTrue(not valid)

        # Check for invalid (non-binary) parity-check matrix
        with self.assertRaises(ValueError) as cm:
            _ = verify_css_code_condition(self.H_hamming, self.H_nb_err)
        self.assertEqual(str(cm.exception), self.err_msg_H_nb)

        with self.assertRaises(ValueError) as cm:
            _ = verify_css_code_condition(self.H_nb_err, self.H_hamming)
        self.assertEqual(str(cm.exception), self.err_msg_H_nb)

        # Check for empty parity-check matrix
        with self.assertRaises(ValueError) as cm:
            _ = verify_css_code_condition(self.H_zeros, self.H_hamming)
        self.assertEqual(str(cm.exception), self.err_msg_H_empty)

        with self.assertRaises(ValueError) as cm:
            _ = verify_css_code_condition(self.H_hamming, self.H_zeros)
        self.assertEqual(str(cm.exception), self.err_msg_H_empty)

    ### CLASSICAL PARITY CHECK MATRIX TESTS

    def test_classical_parity_check_matrix_wrong_input(self):
        """Test that the correct error messages are raised for invalid inputs."""

        err_msg = (
            "A numpy.array, list of list, a tuple of Stabilizers or a "
            "ClassicalTannerGraph must be provided."
        )

        for invalid_input in self.invalid_matrix_inputs:
            with self.assertRaises(TypeError) as cm:
                _ = ClassicalParityCheckMatrix(invalid_input)
            self.assertEqual(str(cm.exception), err_msg)

    def test_classical_parity_check_matrix_from_matrix(self):
        """Test the correct error messages are raised for invalid array-like input."""

        # EXAMPLE 1 - Hamming code
        H_ham = self.H_hamming
        n_check_ham, n_data_ham = H_ham.shape

        # Example 2 - Bitflip repetition code
        H_rep = self.H_rep
        n_check_rep, n_data_rep = H_rep.shape

        # EXAMPLE 3 - Shor code - Cast it as python list for checking different input
        H_shor = self.Hx_shor.tolist()
        n_check_shor, n_data_shor = len(H_shor), len(H_shor[0])

        # EXAMPLE 4 - Valid matrix with empty rows and columns to be removed
        H_with_empty = [[1, 0, 1, 0, 0], [0, 0, 1, 1, 1], [0, 0, 0, 0, 0]]
        clean_H_empty = [[1, 1, 0, 0], [0, 1, 1, 1]]
        n_check_empty, n_data_empty = 2, 4

        # EXAMPLE 5 - Valid matrix with repeated rows to be removed
        H_with_repeat = [[1, 1, 0], [0, 1, 1], [1, 1, 0], [0, 1, 1]]
        clean_H_repeat = [[1, 1, 0], [0, 1, 1]]
        n_check_repeat, n_data_repeat = 2, 3

        # Verify examples
        H_list = [H_ham, H_rep, H_shor, H_with_empty, H_with_repeat]
        clean_H_list = [H_ham, H_rep, H_shor, clean_H_empty, clean_H_repeat]
        n_check_list = [
            n_check_ham,
            n_check_rep,
            n_check_shor,
            n_check_empty,
            n_check_repeat,
        ]
        n_data_list = [n_data_ham, n_data_rep, n_data_shor, n_data_empty, n_data_repeat]

        for H, clean_H, n_check, n_data in zip(
            H_list, clean_H_list, n_check_list, n_data_list, strict=True
        ):

            # Compute Parity Check Matrix
            P = ClassicalParityCheckMatrix(H)

            # Check that the parity-check matrix is correctly stored
            # Turn into tuple and use set to ensure equality up to row permutations
            self.assertEqual(set(map(tuple, P.matrix)), set(map(tuple, clean_H)))

            # Check that the number of rows and columns is correct
            self.assertEqual(P.n_checks, n_check)
            self.assertEqual(P.n_datas, n_data)

            # Check that there are no empty rows
            self.assertTrue(np.all(np.any(P.matrix != 0, axis=1)))

            # Check that there are no repeated rows
            self.assertTrue(np.all(np.unique(P.matrix, axis=0).shape == P.matrix.shape))

        ### Invalid matrix inputs

        # ERROR 1 - Empty matrix
        with self.assertRaises(ValueError) as cm:
            P = ClassicalParityCheckMatrix(self.H_zeros)
        self.assertEqual(str(cm.exception), self.err_msg_H_empty)

        # ERROR 2 - Input matrix is not two-dimensional
        with self.assertRaises(ValueError) as cm:
            P = ClassicalParityCheckMatrix(self.non_2D_H)
        self.assertEqual(str(cm.exception), self.err_msg_non_2D)

        # ERROR 3 - Non-binary elements
        with self.assertRaises(ValueError) as cm:
            P = ClassicalParityCheckMatrix(self.H_nb_err)
        self.assertEqual(str(cm.exception), self.err_msg_H_nb)

    def test_classical_parity_check_matrix_from_stabilizers(self):
        """Test the creation of Parity Check Matrix from set of stabilizers."""

        # EXAMPLE 1 - List of Stabilizers of X type
        stabilizers_x = (
            Stabilizer(
                pauli="XXX",
                data_qubits=[(0, 0, 0), (0, 1, 0), (1, 0, 0)],
                ancilla_qubits=[(0, 0, 1)],
            ),
            Stabilizer(
                pauli="XXX",
                data_qubits=[(0, 0, 0), (0, 2, 0), (2, 0, 0)],
                ancilla_qubits=[(0, 1, 1)],
            ),
            Stabilizer(
                pauli="XXX",
                data_qubits=[(0, 0, 0), (0, 3, 0), (3, 0, 0)],
                ancilla_qubits=[(1, 0, 1)],
            ),
        )

        correct_matrix_x = np.array(
            [[1, 1, 1, 0, 0, 0, 0], [1, 0, 0, 1, 1, 0, 0], [1, 0, 0, 0, 0, 1, 1]]
        )
        correct_n_checks_x, correct_n_datas_x = 3, 7

        # EXAMPLE 2 - List of Stabilizers of Z type - Added repeated stabilizer
        stabilizers_z = (
            Stabilizer(
                pauli="ZZ", data_qubits=[(0, 0), (1, 0)], ancilla_qubits=[(0, 1)]
            ),
            Stabilizer(
                pauli="ZZ", data_qubits=[(1, 0), (2, 0)], ancilla_qubits=[(1, 1)]
            ),
            Stabilizer(
                pauli="ZZ", data_qubits=[(1, 0), (2, 0)], ancilla_qubits=[(1, 1)]
            ),
        )

        correct_matrix_z = np.array([[1, 1, 0], [0, 1, 1]])
        correct_n_checks_z, correct_n_datas_z = 2, 3

        # Verify examples

        stabilizers_list = [stabilizers_x, stabilizers_z]
        correct_matrix_list = [correct_matrix_x, correct_matrix_z]
        correct_datas_list = [correct_n_datas_x, correct_n_datas_z]
        correct_checks_list = [correct_n_checks_x, correct_n_checks_z]

        for stabilizers, H, n_datas, n_checks in zip(
            stabilizers_list,
            correct_matrix_list,
            correct_datas_list,
            correct_checks_list,
            strict=True,
        ):

            # Compute Classical Tanner graph
            P = ClassicalParityCheckMatrix(stabilizers)

            # Check that the parity-check matrix is correctly stored and cleaned
            # Turn into tuple and use set to ensure equality up to column permutations
            self.assertEqual(set(map(tuple, P.matrix.T)), set(map(tuple, H.T)))

            # Check that the number of rows and columns is correct
            self.assertEqual(P.n_checks, n_checks)
            self.assertEqual(P.n_datas, n_datas)

        ### Invalid stabilizer inputs

        # ERROR 1 - Empty stabilizer list
        empty_input = ()
        err_msg = "Input Stabilizer tuple is empty."
        with self.assertRaises(ValueError) as cm:
            P = ClassicalParityCheckMatrix(empty_input)
        self.assertEqual(str(cm.exception), err_msg)

        # ERROR 2 - Non-Classical Stabilizers
        invalid_stabilizers = [
            (
                Stabilizer(
                    pauli="ZZ",
                    data_qubits=[(0, 0), (1, 0)],
                    ancilla_qubits=[(0, 1)],
                ),
                Stabilizer(
                    pauli="XX",
                    data_qubits=[(1, 0), (2, 0)],
                    ancilla_qubits=[(1, 1)],
                ),
            ),
            (
                Stabilizer(
                    pauli="XZ",
                    data_qubits=[(0, 0), (1, 0)],
                    ancilla_qubits=[(0, 1)],
                ),
                Stabilizer(
                    pauli="ZY",
                    data_qubits=[(1, 0), (2, 0)],
                    ancilla_qubits=[(1, 1)],
                ),
            ),
        ]

        err_msg = (
            "Input stabilizers must be of the same type to define a classical"
            " parity check matrix."
        )

        for invalid_stabs in invalid_stabilizers:
            with self.assertRaises(ValueError) as cm:
                P = ClassicalParityCheckMatrix(invalid_stabs)
            self.assertEqual(str(cm.exception), err_msg)

    def test_classical_parity_check_matrix_to_stabilizers(self):
        """Test the conversion of a parity check matrix into list of stabilizers"""

        # EXAMPLE 1 - Hamming code
        H_ham = self.H_hamming
        stabs_ham = [
            Stabilizer(
                pauli="ZZZZ",
                data_qubits=[(0, 0), (1, 0), (2, 0), (3, 0)],
                ancilla_qubits=[(0, 1)],
            ),
            Stabilizer(
                pauli="ZZZZ",
                data_qubits=[(1, 0), (2, 0), (4, 0), (5, 0)],
                ancilla_qubits=[(1, 1)],
            ),
            Stabilizer(
                pauli="ZZZZ",
                data_qubits=[(2, 0), (3, 0), (5, 0), (6, 0)],
                ancilla_qubits=[(2, 1)],
            ),
        ]
        check_type_ham = "Z"

        # EXAMPLE 2 - Shor code
        H_shor = self.Hx_shor
        stabs_shor = [
            Stabilizer(
                pauli="X" * 6,
                data_qubits=[(i, 0) for i in range(6)],
                ancilla_qubits=[(0, 1)],
            ),
            Stabilizer(
                pauli="X" * 6,
                data_qubits=[(i, 0) for i in range(3, 9)],
                ancilla_qubits=[(1, 1)],
            ),
        ]
        check_type_shor = "X"

        # Verify examples
        H_list = [H_ham, H_shor]
        stabs_list = [stabs_ham, stabs_shor]
        check_type_list = [check_type_ham, check_type_shor]

        for check_type, stabs, H in zip(
            check_type_list, stabs_list, H_list, strict=True
        ):
            P = ClassicalParityCheckMatrix(H)

            # Compute stabilizers
            computed_stabs = P.to_stabilizers(pauli_type=check_type)

            # Check Stabilizers are correct up to ancilla definitions
            ordered_computed_stabs = sorted(
                computed_stabs, key=lambda x: x.data_qubits[0]
            )
            ordered_stabs = sorted(stabs, key=lambda x: x.data_qubits[0])

            for stab, comp_stab in zip(
                ordered_stabs, ordered_computed_stabs, strict=True
            ):
                self.assertEqual(stab.pauli, comp_stab.pauli)
                self.assertEqual(stab.data_qubits, comp_stab.data_qubits)

        # ERROR - Invalid Pauli type input
        invalid_pauli_inputs = ["Y", "I", 9]
        err_msg = "Pauli type must be either 'X' or 'Z'."

        P = ClassicalParityCheckMatrix(self.Hx_shor)
        for invalid_pauli in invalid_pauli_inputs:
            with self.assertRaises(ValueError) as cm:
                _ = P.to_stabilizers(pauli_type=invalid_pauli)
            self.assertEqual(str(cm.exception), err_msg)

    def test_classical_parity_check_matrix_from_classical_tanner_graph(self):
        """Test the creation of a parity check matrix from a classical Tanner graph."""

        # EXAMPLE 1 - Hamming code
        T_ham = ClassicalTannerGraph(self.T_hamming)

        # EXAMPLE 2 - Bitflip repetition code
        T_rep = ClassicalTannerGraph(self.T_bitflip_rep)

        # EXAMPLE 3 - Shor code
        T_shor = ClassicalTannerGraph(self.Tx_shor)

        # Verify examples
        T_list = [T_ham, T_rep, T_shor]
        H_list = [self.H_hamming, self.H_rep, self.Hx_shor]

        for T, H in zip(T_list, H_list, strict=True):

            # Compute Parity Check Matrix
            P = ClassicalParityCheckMatrix(T)

            # Check that the parity-check matrix is correctly stored
            # Turn into tuple and use set to ensure equality up to row permutations
            self.assertEqual(set(map(tuple, P.matrix)), set(map(tuple, H)))

            # Check that the number of rows and columns is correct
            self.assertEqual(P.n_checks, H.shape[0])
            self.assertEqual(P.n_datas, H.shape[1])

    def test_classical_parity_check_matrix_eq_method(self):
        """Test the equality method of ClassicalParityCheckMatrix."""

        # EXAMPLE 1 - Hamming code
        P_ham = ClassicalParityCheckMatrix(self.H_hamming)

        # EXAMPLE 2 - Bitflip repetition code
        P_rep = ClassicalParityCheckMatrix(self.H_rep)

        # EXAMPLE 3 - Shor code
        P_shor = ClassicalParityCheckMatrix(self.Hx_shor)

        # Verify equality
        self.assertEqual(P_ham, P_ham)
        self.assertEqual(P_rep, P_rep)
        self.assertEqual(P_shor, P_shor)

        # Verify inequality
        self.assertNotEqual(P_ham, P_rep)
        self.assertNotEqual(P_ham, P_shor)
        self.assertNotEqual(P_rep, P_shor)

        # Check error messages for invalid inputs
        err_msg = (
            "Comparison is only supported with another ClassicalParityCheckMatrix."
        )
        with self.assertRaises(TypeError) as cm:
            _ = P_ham == self.H_hamming
        self.assertEqual(str(cm.exception), err_msg)

    # PARITY CHECK MATRIX TESTS

    def test_parity_check_matrix_wrong_input(self):
        """Test that the correct error messages are raised for invalid inputs."""

        err_msg = (
            "A numpy.array, list of lists, tuple of Stabilizers or a TannerGraph "
            "object must be provided."
        )

        for invalid_input in self.invalid_matrix_inputs:
            with self.assertRaises(TypeError) as cm:
                _ = ParityCheckMatrix(invalid_input)
            self.assertEqual(str(cm.exception), err_msg)

    def test_parity_check_matrix_from_matrix(self):
        """Test that validity condition for a parity-check matrix to describe a code is
        correctly verified. This function also tests for the check_if_css method."""

        # EXAMPLE 1 - Steane code
        H_steane = self.H_steane
        n_stabs_steane, n_data_steane = 6, 7

        # EXAMPLE 2 - Laflamme code
        H_laflamme = self.H_laflamme
        n_stabs_laflamme, n_data_laflamme = 4, 5

        # EXAMPLE 3 - Shor code
        H_shor = self.H_shor
        n_stabs_shor, n_data_shor = 8, 9

        # EXAMPLE 4 - Valid matrix with empty rows and columns to be removed
        H_with_empty = [
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0],
        ]
        clean_H_empty = [[1, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]]
        n_stabs_empty, n_data_empty = 3, 3

        # EXAMPLE 5 - Valid matrix with repeated rows to be removed
        H_with_repeat = [
            [1, 1, 0, 0, 1, 1],
            [0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
        ]
        clean_H_repeat = [[1, 1, 0, 0, 1, 1], [0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]]
        n_stabs_repeat, n_data_repeat = 3, 3

        # Verify code validity
        H_list = [H_steane, H_laflamme, H_shor, H_with_empty, H_with_repeat]

        clean_H_list = [H_steane, H_laflamme, H_shor, clean_H_empty, clean_H_repeat]
        n_stabs_list = [
            n_stabs_steane,
            n_stabs_laflamme,
            n_stabs_shor,
            n_stabs_empty,
            n_stabs_repeat,
        ]
        n_data_list = [
            n_data_steane,
            n_data_laflamme,
            n_data_shor,
            n_data_empty,
            n_data_repeat,
        ]
        is_css_list = [True, False, True, True, False]

        for H, clean_H, n_stabs, n_data, is_css in zip(
            H_list, clean_H_list, n_stabs_list, n_data_list, is_css_list, strict=True
        ):
            P = ParityCheckMatrix(H)

            # Check that the parity-check matrix is correctly stored
            # Turn into tuple and use set to ensure equality up to row permutations
            self.assertEqual(set(map(tuple, P.matrix)), set(map(tuple, clean_H)))

            # Check that the number of rows and columns is correct
            self.assertEqual(P.n_stabs, n_stabs)
            self.assertEqual(P.n_datas, n_data)

            # Check that there are no empty rows
            self.assertTrue(np.all(np.any(P.matrix != 0, axis=1)))

            # Check that there are no repeated rows
            self.assertTrue(np.all(np.unique(P.matrix, axis=0).shape == P.matrix.shape))

            # Check if the CSS flag is set correctly
            self.assertEqual(P.is_css, is_css)

        # Check for invalid (non-binary) parity-check matrix
        with self.assertRaises(ValueError) as cm:
            P = ParityCheckMatrix(self.H_nb_err)
        self.assertEqual(str(cm.exception), self.err_msg_H_nb)

        # Check for empty parity-check matrix
        with self.assertRaises(ValueError) as cm:
            P = ParityCheckMatrix(self.H_zeros)
        self.assertEqual(str(cm.exception), self.err_msg_H_empty)

        # Check for 2D array
        with self.assertRaises(ValueError) as cm:
            P = ClassicalParityCheckMatrix(self.non_2D_H)
        self.assertEqual(str(cm.exception), self.err_msg_non_2D)

        # Check for odd-shaped parity-check matrix
        H_odd = np.array([[1, 1, 0], [0, 1, 1]])
        err_msg = "Parity-check matrix contains odd number of columns."
        with self.assertRaises(ValueError) as cm:
            P = ParityCheckMatrix(H_odd)
        self.assertEqual(str(cm.exception), err_msg)

        # Check for non-commuting stabilizers
        Hx = np.array([[1, 1, 0, 1], [0, 1, 1, 0], [0, 0, 1, 1]])
        Hz = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
        H_invalid = np.vstack(
            (np.hstack((Hx, np.zeros(Hx.shape))), np.hstack((np.zeros(Hz.shape), Hz)))
        )
        err_msg = "Parity-check matrix does not define a quantum code."
        with self.assertRaises(ValueError) as cm:
            P = ParityCheckMatrix(H_invalid)
        self.assertEqual(str(cm.exception), err_msg)

    def test_parity_check_matrix_from_tanner_graph(self):
        """Test the creation of a parity check matrix from a Tanner graph."""

        # EXAMPLE 1 - Steane code
        T_steane = TannerGraph(self.T_steane)

        # EXAMPLE 2 - Shor code
        T_shor = TannerGraph(self.T_shor)

        # EXAMPLE 3 - Rotated surface code
        T_rsc = TannerGraph(self.T_rsc)

        # Verify examples
        T_list = [T_steane, T_shor, T_rsc]
        H_list = [self.H_steane, self.H_shor, self.H_rsc]

        for T, H in zip(T_list, H_list, strict=True):

            # Compute Parity Check Matrix
            P = ParityCheckMatrix(T)

            # Check that the parity-check matrix is correctly stored
            # Turn into tuple and use set to ensure equality up to row permutations
            self.assertEqual(set(map(tuple, P.matrix)), set(map(tuple, H)))

            # Check that the number of rows and columns is correct
            self.assertEqual(P.n_datas, len(T.data_nodes))
            self.assertEqual(P.n_stabs, len(T.x_nodes) + len(T.z_nodes))

            # All input Tanners correspond to CSS codes
            self.assertEqual(P.is_css, True)

    def test_parity_check_matrix_from_stabilizers(self):
        """Test the creation of a parity check matrix from a set of stabilizers."""

        # EXAMPLE 1 - Steane code
        stabilizers_steane = self.stabilizers_steane
        n_stabs_steane, n_data_steane = 6, 7

        # EXAMPLE 2 - Shor code
        stabilizers_shor = self.stabilizers_shor
        n_stabs_shor, n_data_shor = 8, 9

        # Verify examples
        stabilizers_list = [stabilizers_steane, stabilizers_shor]
        H_list = [self.H_steane, self.H_shor]
        n_stabs_list = [n_stabs_steane, n_stabs_shor]
        n_data_list = [n_data_steane, n_data_shor]

        for stabilizers, H, n_stabs, n_data in zip(
            stabilizers_list, H_list, n_stabs_list, n_data_list, strict=True
        ):

            # Compute Parity Check Matrix from Stabilizer tuple
            P = ParityCheckMatrix(tuple(stabilizers))

            # Check that the parity-check matrix is correctly stored
            # Turn into tuple and use set to ensure equality up to row permutations
            self.assertEqual(set(map(tuple, P.matrix)), set(map(tuple, H)))

            # Check that the number of rows and columns is correct
            self.assertEqual(P.n_stabs, n_stabs)
            self.assertEqual(P.n_datas, n_data)

            # Check that the CSS flag is set correctly
            self.assertEqual(P.is_css, True)

    def test_parity_check_matrix_to_stabilizers(self):
        """Test the conversion of a parity check matrix into list of stabilizers."""

        # EXAMPLE 1 - Steane code
        P_steane = ParityCheckMatrix(self.H_steane)
        stabs_steane = P_steane.to_stabilizers()
        expected_stabs_steane = self.stabilizers_steane

        # Check that the stabilizers are correct
        self.assertEqual(set(stabs_steane), set(expected_stabs_steane))

        # EXAMPLE 2 - Shor code
        P_shor = ParityCheckMatrix(self.H_shor)
        stabs_shor = P_shor.to_stabilizers()

        expected_stabs_shor = self.stabilizers_shor

        # Check that the stabilizers are correct
        self.assertEqual(set(stabs_shor), set(expected_stabs_shor))

        # ERROR
        non_comm_stabs = tuple(
            [
                Stabilizer(
                    "XXXX",
                    data_qubits=[(0, 0), (1, 0), (2, 0), (3, 0)],
                    ancilla_qubits=[(0, 1)],
                ),
                Stabilizer("ZZ", data_qubits=[(1, 0), (3, 0)], ancilla_qubits=[(0, 2)]),
                Stabilizer("XY", data_qubits=[(0, 0), (2, 0)], ancilla_qubits=[(0, 3)]),
            ]
        )

        err_msg = (
            f"Input Stabilizers {non_comm_stabs[0]} and {non_comm_stabs[2]}"
            " do not commute."
        )

        with self.assertRaises(ValueError) as cm:
            _ = ParityCheckMatrix(non_comm_stabs)
        self.assertTrue(err_msg in str(cm.exception))

    def test_parity_check_matrix_extract_components(self):
        """Test the extraction of components from a parity-check matrix."""

        # For CSS codes, ensure that X and Z components are correct
        # EXAMPLE 1 - Steane code
        P_steane = ParityCheckMatrix(self.H_steane)
        CPM_hamming = ClassicalParityCheckMatrix(self.H_hamming)
        Hx_steane, Hz_steane = P_steane.get_components()

        self.assertEqual(Hx_steane, CPM_hamming)
        self.assertEqual(Hz_steane, CPM_hamming)

        # EXAMPLE 2 - Shor code
        P_shor = ParityCheckMatrix(self.H_shor)
        Hx_shor, Hz_shor = P_shor.get_components()

        self.assertEqual(Hx_shor, ClassicalParityCheckMatrix(self.Hx_shor))
        self.assertEqual(Hz_shor, ClassicalParityCheckMatrix(self.Hz_shor))

        # For non-CSS codes, ensure error is raised correctly
        err_msg = (
            "Parity-check matrix cannot be split into hx_matrix and hz_matrix as there are"
            " stabilizers with mixed X and Z support, thus it does not define"
            " a CSS code."
        )

        # EXAMPLE 3 - Laflamme code
        P_laflamme = ParityCheckMatrix(self.H_laflamme)
        with self.assertRaises(ValueError) as cm:
            _ = P_laflamme.get_components()
        self.assertEqual(str(cm.exception), err_msg)

    def test_parity_check_matrix_properties(self):
        """Test the properties of a parity-check matrix."""

        # EXAMPLE 1 - Steane code
        P_steane = ParityCheckMatrix(self.H_steane)
        CPM_hamming = ClassicalParityCheckMatrix(self.H_hamming)

        self.assertEqual(P_steane.hx_matrix, CPM_hamming)
        self.assertEqual(P_steane.n_xstabs, len(self.H_hamming))
        self.assertEqual(P_steane.hz_matrix, CPM_hamming)
        self.assertEqual(P_steane.n_zstabs, len(self.H_hamming))

        # EXAMPLE 2 - Shor code
        P_shor = ParityCheckMatrix(self.H_shor)

        self.assertEqual(P_shor.hx_matrix, ClassicalParityCheckMatrix(self.Hx_shor))
        self.assertEqual(P_shor.n_xstabs, len(self.Hx_shor))
        self.assertEqual(P_shor.hz_matrix, ClassicalParityCheckMatrix(self.Hz_shor))
        self.assertEqual(P_shor.n_zstabs, len(self.Hz_shor))

    def test_parity_check_matrix_eq_method(self):
        """Test the equality method of ParityCheckMatrix."""

        # EXAMPLE 1 - Steane code
        P_steane = ParityCheckMatrix(self.H_steane)

        # EXAMPLE 2 - Shor code
        P_shor = ParityCheckMatrix(self.H_shor)

        # Check equality
        self.assertEqual(P_steane, P_steane)
        self.assertEqual(P_shor, P_shor)
        self.assertNotEqual(P_steane, P_shor)

        # Check for error message for invalid inputs
        err_msg = "Comparison is only supported with another ParityCheckMatrix."
        with self.assertRaises(TypeError) as cm:
            _ = P_steane == self.H_steane
        self.assertEqual(str(cm.exception), err_msg)


if __name__ == "__main__":
    unittest.main()
