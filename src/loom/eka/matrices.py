"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from __future__ import annotations
from functools import cached_property

import numpy as np

from .stabilizer import Stabilizer
from . import tanner_graphs


class ClassicalParityCheckMatrix:
    """Classical parity-check matrix for error-correcting codes."""

    def __init__(
        self,
        matrix_input: (
            np.ndarray
            | list[list[int]]
            | tuple[Stabilizer, ...]
            | tanner_graphs.ClassicalTannerGraph
        ),
    ):
        """
        The ClassicalParityCheckMatrix class stores a parity-check matrix
        faithfully describing a classical error-correcting code. This is a binary
        matrix, where each row describes a classical check and each column a
        classical bit.

        The object can be instantiated from a numpy array, a list of lists, or a tuple of
        Stabilizers. For an array-like input, the matrix is verified to be valid and
        cleaned afterwards. We adopt the convention of removing repeated rows and empty
        rows. For a Stabilizers input, we require all of them to be of the same pauli
        type, i.e. either X or Z. The support of the matrix rows is then built from the
        qubit support of each stabilizer.

        The ClassicalParityCheckMatrix object can also be casted as a list of
        Stabilizers.

        Lastly the matrix, can also be generated from a ClassicalTannerGraph, where
        the connectivity of the check nodes is translated into the rows of the
        matrix.

        Parameters
        ----------
        input : np.ndarray | list[list[int]] | tuple[Stabilizer,...] | ClassicalTannerGraph
            Input to instantiate the ClassicalParityCheckMatrix object.
        """

        if isinstance(matrix_input, np.ndarray):
            h_matrix = matrix_input
            self.verify_input(h_matrix)

        elif isinstance(matrix_input, list) and all(
            isinstance(row, list) for row in matrix_input
        ):
            h_matrix = np.array(matrix_input)
            self.verify_input(h_matrix)

        elif isinstance(matrix_input, tuple) and all(
            isinstance(item, Stabilizer) for item in matrix_input
        ):
            h_matrix = self.generate_matrix_from_stabilizers(matrix_input)
        elif isinstance(matrix_input, tanner_graphs.ClassicalTannerGraph):
            h_matrix = self.generate_matrix_from_graph(matrix_input)
        else:
            raise TypeError(
                "A numpy.array, list of list, a tuple of Stabilizers or a "
                "ClassicalTannerGraph must be provided."
            )

        self.matrix = h_matrix
        self.clean_matrix()
        self.n_checks, self.n_datas = self.matrix.shape

    @staticmethod
    def verify_input(h_matrix: np.ndarray) -> None:
        """
        Verify the input parity-check matrix is valid. The matrix is required to
        be binary, two-dimensional, and non-empty.

        Parameters
        ----------
        h_matrix : np.ndarray
            Parity-check matrix to be verified.
        """

        # Check for empty matrix
        if not np.any(h_matrix):
            raise ValueError("Parity-check matrix is empty.")

        # Ensure array is two-dimensional
        if len(h_matrix.shape) != 2:
            raise ValueError("Parity-check matrix must be a 2D array.")

        # Check for non-binary elements
        if not np.all(np.isin(h_matrix, [0, 1])):
            raise ValueError("Parity-check matrix contains non-binary elements.")

    def clean_matrix(self) -> None:
        """
        Clean the parity-check matrix by removing repeated and empty rows and
        columns.
        """

        # Remove repeated rows
        # Rows are manually reversed to counter enforced sorting from numpy.unique
        self.matrix = np.unique(self.matrix, axis=0)[::-1]

        # Remove empty rows
        self.matrix = self.matrix[~np.all(self.matrix == 0, axis=1)]

        # Remove empty columns
        self.matrix = self.matrix[:, ~np.all(self.matrix == 0, axis=0)]

    @staticmethod
    def generate_matrix_from_stabilizers(
        stabilizers: tuple[Stabilizer, ...],
    ) -> np.ndarray:
        """
        Generate parity-check matrix from a set of stabilizers.

        Parameters
        ----------
        stabilizers : tuple[Stabilizer,...]
            Stabilizers to generate the parity-check matrix from.

        Returns
        -------
        h_matrix : np.ndarray
            Parity-check matrix generated from the stabilizers.
        """

        # Verifuy non-empty input
        if len(stabilizers) == 0:
            raise ValueError("Input Stabilizer tuple is empty.")

        # Verify all stabilizers are of the same type
        pauli_type = set(p for stab in stabilizers for p in stab.pauli)
        if len(pauli_type) > 1:
            raise ValueError(
                "Input stabilizers must be of the same type to define a classical"
                " parity check matrix."
            )

        # Extract the number of data qubits
        all_data_qubits = {qubit for stab in stabilizers for qubit in stab.data_qubits}
        n_data = len(all_data_qubits)

        coord_to_index = {qubit: i for i, qubit in enumerate(all_data_qubits)}

        # Initialize parity-check matrix
        h_matrix = np.zeros((len(stabilizers), n_data), dtype=int)

        # Fill in the parity-check matrix
        for k, stabilizer in enumerate(stabilizers):
            for qubit in stabilizer.data_qubits:
                i = coord_to_index[qubit]
                h_matrix[k, i] = 1

        return h_matrix

    def to_stabilizers(self, pauli_type: str) -> list[Stabilizer]:
        """
        Convert the parity-check matrix to a list of Stabilizers.

        Parameters
        ----------
        pauli_type : str
            Pauli type to assign to the stabilizers, either 'X' or 'Z'.

        Returns
        -------
        stabilizers : list[Stabilizer]
            List of Stabilizers generated from the parity-check matrix.
        """

        # Check input
        if pauli_type not in ["X", "Z"]:
            raise ValueError("Pauli type must be either 'X' or 'Z'.")

        stabilizers = []
        for ind_row, row in enumerate(self.matrix):
            data_qubits = [(i, 0) for i in np.where(row == 1)[0]]
            ancilla_qubits = [(ind_row, 1)]
            stabilizers.append(
                Stabilizer(
                    pauli=pauli_type * len(data_qubits),
                    data_qubits=data_qubits,
                    ancilla_qubits=ancilla_qubits,
                )
            )

        return stabilizers

    @staticmethod
    def generate_matrix_from_graph(
        tanner_graph: tanner_graphs.ClassicalTannerGraph,
    ) -> np.ndarray:
        """
        Generate parity-check matrix from an input Tanner graph.

        Parameters
        ----------
        tanner_graph : graphs.ClassicalTannerGraph
            Tanner graph to generate the parity-check matrix from.

        Returns
        -------
        h_matrix : np.ndarray
            Parity-check matrix generated from the Tanner graph.
        """

        # Extract the graph nodes
        data_nodes = {
            n
            for n, attr in tanner_graph.graph.nodes(data=True)
            if attr["label"] == "data"
        }
        check_nodes = {
            n
            for n, attr in tanner_graph.graph.nodes(data=True)
            if attr["label"] != "data"
        }

        # Initialize parity-check matrix
        h_matrix = np.zeros((len(check_nodes), len(data_nodes)), dtype=int)

        # Fill in with checks
        for i, check_node in enumerate(check_nodes):
            for data_node in tanner_graph.graph.neighbors(check_node):
                h_matrix[i, data_node] = 1

        return h_matrix

    def __eq__(self, other: ClassicalParityCheckMatrix) -> bool:
        """
        Check if two parity-check matrices are equal.

        Parameters
        ----------
        other : ClassicalParityCheckMatrix
            Other parity-check matrix to compare with.

        Returns
        -------
        bool
            True if the matrices are equal, False otherwise.
        """
        if not isinstance(other, ClassicalParityCheckMatrix):
            raise TypeError(
                "Comparison is only supported with another ClassicalParityCheckMatrix."
            )

        def are_attributes_not_the_same():
            return self.n_checks != other.n_checks or self.n_datas != other.n_datas

        # Ensure that matrices are equal up to row permutations
        def are_rows_not_the_same():
            return set(map(tuple, self.matrix)) != set(map(tuple, other.matrix))

        return not (are_attributes_not_the_same() or are_rows_not_the_same())


class ParityCheckMatrix:
    """Parity-check matrix for quantum error-correcting codes."""

    def __init__(
        self,
        input: (
            list[list] | np.ndarray | tanner_graphs.TannerGraph | tuple[Stabilizer, ...]
        ),
        # pylint: disable=redefined-builtin
    ):
        """
        The ParityCheckMatrix class stores a parity-check matrix faithfully describing a
        quantum error-correcting code. This is a binary matrix, where each row is
        associated with the symplectic representation of a stabilizer. Therefore, these
        matrices have twice the number of columns as the number of data qubits in the
        code, with the first half representing the X stabilizers and the second half
        representing the Z stabilizers. The number of rows is then equal to the number
        of stabilizers in the code. To represent a valid quantum code, these matrices
        must satisfy the vanishing symplectic product condition, i.e. the symplectic
        product of the matrix with itself must be zero. This corresponds to all the
        stabilizers commuting with each other.

        The object can be instantiated from a numpy array or a list of lists, a
        TannerGraph object or a tuple containing Stabilizer objects. For an
        array-like input, the matrix is verified to be valid and cleaned afterwards.
        We adopt the convention of removing repeated rows and empty rows.
        For a TannerGraph input, the parity-check matrix is generated from the
        connectivity of bipartite graph. For a Stabilizer input, each ancilla and each
        data qubit are mapped into rows and columns of the matrix, respectively.

        Parameters
        ----------
        input : np.ndarray | list[list[int]] | TannerGraph | tuple[Stabilizer, ...]
            Input to instantiate the ParityCheckMatrix object.
        """

        if isinstance(input, np.ndarray):
            h_matrix = input
            self.verify_input(h_matrix)

        elif isinstance(input, list) and all(isinstance(row, list) for row in input):
            h_matrix = np.array(input)
            self.verify_input(h_matrix)

        elif isinstance(input, tanner_graphs.TannerGraph):
            h_matrix = self.generate_matrix_from_graph(input)

        elif isinstance(input, tuple) and all(
            isinstance(item, Stabilizer) for item in input
        ):
            h_matrix = self.generate_matrix_from_stabilizers(input)

        else:
            raise TypeError(
                "A numpy.array, list of lists, tuple of Stabilizers or a TannerGraph "
                "object must be provided."
            )

        self.matrix = h_matrix
        self.clean_matrix()
        self.n_stabs, self.n_datas = self.matrix.shape[0], self.matrix.shape[1] // 2

        # Check if code is CSS
        self.is_css = self.check_if_css

    @staticmethod
    def verify_input(h_matrix: np.ndarray) -> None:
        """
        Verifies that parity-check matrix defines a valid quantum code, through
        structural checks and computing the symplectic product with itself.

        Parameters
        ----------
        h_matrix : np.ndarray
            A full parity-check matrix for a quantum code.
        """

        # Check for empty matrix
        if not np.any(h_matrix):
            raise ValueError("Parity-check matrix is empty.")

        # Ensure array is two-dimensional
        if len(h_matrix.shape) != 2:
            raise ValueError("Parity-check matrix must be a 2D array.")

        # Check input
        if not np.all(np.isin(h_matrix, [0, 1])):
            raise ValueError("Parity-check matrix contains non-binary elements.")

        # Check number of data qubits is well defined
        if len(h_matrix[0]) % 2 == 1:
            raise ValueError("Parity-check matrix contains odd number of columns.")

        # Extract the number of data qubits
        n = len(h_matrix[0]) // 2

        # Define the symplectic matrix
        symplectic_matrix = np.vstack(
            (
                np.hstack((np.zeros((n, n)), np.eye(n))),
                np.hstack((np.eye(n), np.zeros((n, n)))),
            )
        )

        # Compute symplectic product
        product = np.dot(h_matrix, np.dot(symplectic_matrix, h_matrix.T)) % 2

        # Specify validity as a boolean
        valid = not bool(product.any())

        if not valid:
            raise ValueError("Parity-check matrix does not define a quantum code.")

    def clean_matrix(self) -> None:
        """Clean the parity-check matrix by removing repeated and empty rows and
        columns."""

        # Remove repeated rows
        # Rows are manually reversed to counter enforced sorting from numpy.unique
        self.matrix = np.unique(self.matrix, axis=0)[::-1]

        # Remove empty rows
        self.matrix = self.matrix[~np.all(self.matrix == 0, axis=1)]

        # Remove empty columns by checking both sectors
        r = np.shape(self.matrix)[1] // 2

        # Identify columns where both column[i] and column[r+i] are all zeros
        zero_pairs = np.all(self.matrix[:, :r] == 0, axis=0) & np.all(
            self.matrix[:, r:] == 0, axis=0
        )

        # Concatenate the two halves using the mask
        self.matrix = np.hstack(
            (self.matrix[:, :r][:, ~zero_pairs], self.matrix[:, r:][:, ~zero_pairs])
        )

    @staticmethod
    def generate_matrix_from_graph(
        tanner_graph: tanner_graphs.TannerGraph,
    ) -> np.ndarray:
        """
        Generate parity-check matrix from an input Tanner graph.

        Parameters
        ----------
        tanner_graph : TannerGraph
            Tanner graph object to generate the parity-check matrix from.
        Returns
        -------
        h_matrix : np.ndarray
            Parity-check matrix generated from the Tanner graph as numpy array.
        """

        # Initialize parity-check matrix
        h_matrix = np.zeros(
            (
                len(tanner_graph.x_nodes) + len(tanner_graph.z_nodes),
                2 * len(tanner_graph.data_nodes),
            )
        )

        # Map data nodes to column indices - Datas are sorted by their coordinates
        data_to_ind = {d: i for i, d in enumerate(sorted(tanner_graph.data_nodes))}

        # Fill in with X checks
        for i, check_node in enumerate(tanner_graph.x_nodes):
            for data_node in tanner_graph.graph.neighbors(check_node):
                data_ind = data_to_ind[data_node]
                h_matrix[i, data_ind] = 1

        # Fill in with Z checks
        for i, check_node in enumerate(tanner_graph.z_nodes):
            for data_node in tanner_graph.graph.neighbors(check_node):
                data_ind = data_to_ind[data_node] + len(tanner_graph.data_nodes)
                h_matrix[i + len(tanner_graph.x_nodes), data_ind] = 1

        return h_matrix

    @staticmethod
    def generate_matrix_from_stabilizers(
        stabilizers: tuple[Stabilizer, ...],
    ) -> np.ndarray:
        """
        Generate parity-check matrix from a list of stabilizers. Each stabilizer is
        converted into a row of the matrix. For consistency, we sort the data qubits
        according to their coordinates and assign them a column index in the matrix.

        Parameters
        ----------
        stabilizers : tuple[Stabilizer,...]
            Stabilizers to generate the parity-check matrix from.

        Returns
        -------
        h_matrix : np.ndarray
            Parity-check matrix generated from the stabilizers as numpy array.
        """

        # Check that stabilizers commute
        for stab1 in stabilizers:
            for stab2 in stabilizers:
                if not stab1.commutes_with(stab2):
                    raise ValueError(
                        f"Input Stabilizers {stab1} and {stab2} do not commute."
                    )

        # Extract the number of data qubits and sort it
        all_data_qubits = sorted(
            {qubit for stab in stabilizers for qubit in stab.data_qubits}
        )
        n_data = len(all_data_qubits)

        # Initialize parity-check matrix
        h_matrix = np.zeros((len(stabilizers), 2 * n_data), dtype=int)

        # Fill in the parity-check matrix
        for k, stabilizer in enumerate(stabilizers):

            # Last element omitted as it corresponds to the sign
            h_matrix[k, :] = stabilizer.as_signed_pauli_op(all_data_qubits).array[:-1]

        return h_matrix

    def to_stabilizers(self) -> list[Stabilizer]:
        """
        Converts the parity-check matrix to a list of Stabilizers. The stabilizers are
        generated by scanning each row of the matrix and extracting data qubits from
        the position of the non-zero elements. The data qubits are assigned coordinates
        in the form `(index,0)`, where `index` is the column index of the data qubit in
        the symplectic representation of the matrix. The ancilla qubits are
        assigned coordinates in the form `(index,1)`, where `index` is the row index of
        the stabilizer in the matrix.

        Returns
        -------
        stabilizers : list[Stabilizer]
            List of Stabilizers generated from the parity-check matrix.
        """

        # Initialize the list of stabilizers
        stabilizers = []

        # Symplectic definition of the Pauli operators
        ref_string = {(0, 0): "", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}

        for ind_row, row in enumerate(self.matrix):
            # Extract the X and Z parts supports of the stabilizer
            x_row, z_row = row[: self.n_datas], row[self.n_datas :]
            pauli = "".join(
                [ref_string[(x, z)] for x, z in zip(x_row, z_row, strict=True)]
            )
            data_qubits = [
                (i, 0) for i in sorted(np.where((x_row == 1) | (z_row == 1))[0])
            ]

            ancilla_qubits = [(ind_row, 1)]
            stabilizers.append(
                Stabilizer(
                    pauli=pauli, data_qubits=data_qubits, ancilla_qubits=ancilla_qubits
                )
            )

        return stabilizers

    @cached_property
    def check_if_css(self) -> bool:
        """
        Check if the parity-check matrix defines a CSS code. Commutativity is checked
        beforehand by verifying that the symplectic product of the full matrix.

        Returns
        -------
        valid_css : bool
            True if the parity-check matrix defines a valid CSS code, False otherwise.
        """

        valid_css = True

        # For CSS, the support of every row should be non-vanishing only on one side of the
        # matrix, i.e. either X or Z stabilizers.
        for row in self.matrix:
            if np.any(row[: self.n_datas]) and np.any(row[self.n_datas :]):
                valid_css = False
                break

        return valid_css

    def get_components(
        self,
    ) -> tuple[ClassicalParityCheckMatrix, ClassicalParityCheckMatrix]:
        """
        Compute the X and Z components of the parity-check matrix, if possible.

        Returns
        -------
        hx_matrix : ClassicalParityCheckMatrix
            The X component of the parity-check matrix, containing only X stabilizers.
        hz_matrix : ClassicalParityCheckMatrix
            The Z component of the parity-check matrix, containing only Z stabilizers.
        """

        # Check if matrix has already been verified as non CSS
        if not self.is_css:
            raise ValueError(
                "Parity-check matrix cannot be split into hx_matrix and hz_matrix as there are"
                " stabilizers with mixed X and Z support, thus it does not define"
                " a CSS code."
            )

        # Extract the X and Z components of the parity-check matrix
        x_component = self.matrix[np.any(self.matrix[:, : self.n_datas], axis=1)][
            :, : self.n_datas
        ]
        z_component = self.matrix[np.any(self.matrix[:, self.n_datas :], axis=1)][
            :, self.n_datas :
        ]

        # Convert to ClassicalParityCheckMatrix objects
        hx_matrix = ClassicalParityCheckMatrix(x_component)
        hz_matrix = ClassicalParityCheckMatrix(z_component)

        return hx_matrix, hz_matrix

    @cached_property
    def hx_matrix(self) -> ClassicalParityCheckMatrix:
        """Extract the X component of the parity-check matrix."""
        hx_matrix, _ = self.get_components()
        return hx_matrix

    @cached_property
    def n_xstabs(self) -> int:
        """Extract the number of X stabilizers."""
        hx_matrix, _ = self.get_components()
        return hx_matrix.n_checks

    @cached_property
    def hz_matrix(self) -> ClassicalParityCheckMatrix:
        """Extract the Z component of the parity-check matrix."""
        _, hz_matrix = self.get_components()
        return hz_matrix

    @cached_property
    def n_zstabs(self) -> int:
        """Extract the number of Z stabilizers."""
        _, hz_matrix = self.get_components()
        return hz_matrix.n_checks

    def __eq__(self, other: ParityCheckMatrix) -> bool:
        """
        Check if two ParityCheckMatrix objects are equal.

        Parameters
        ----------
        other : ParityCheckMatrix
            Other parity-check matrix to compare with.

        Returns
        -------
        bool
            True if the matrices are equal, False otherwise.
        """

        if not isinstance(other, ParityCheckMatrix):
            raise TypeError(
                "Comparison is only supported with another ParityCheckMatrix."
            )

        def are_attributes_not_equal():
            return self.n_stabs != other.n_stabs or self.n_datas != other.n_datas

        #  that matrices are equal up to row permutations
        def are_rows_not_equal():
            return set(map(tuple, self.matrix)) != set(map(tuple, other.matrix))

        return not (are_attributes_not_equal() or are_rows_not_equal())
