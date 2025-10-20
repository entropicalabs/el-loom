"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy

import numpy as np

from loom.eka.utilities import is_stabarray_equivalent, StabArray, sparse_formatter

from .exceptions import TableauSizeError


@dataclass(frozen=True)
class TableauSnapshot:
    """
    A snapshot of the Tableau at a specific point in time.
    """

    nqubits: int
    rand_gen: int
    tableau_w_scratch: np.ndarray


class Tableau:  # pylint: disable=too-many-instance-attributes
    """
    The Tableau class represents the stabilizer state of a quantum system
    using the tableau representation. It provides methods to manipulate and
    query the stabilizer state.
    """

    dtype = np.int8

    def __init__(
        self, nqubits: int = None, initial_tableau: np.ndarray = None, seed: int = None
    ) -> None:
        """
        Initializes the the state `|00...0>`.
        """
        if not (nqubits is None) ^ (initial_tableau is None):
            raise ValueError(
                "Tableau should be defined using nqubits XOR initial_tableau"
            )

        self.rand_gen = np.random.default_rng(seed)

        if nqubits is not None:
            # construct tableau array as concatenation of an identity array
            # and an empty vector
            self.tableau_w_scratch = np.zeros(
                (2 * nqubits + 1, 2 * nqubits + 1), dtype=self.dtype
            )
            id_array = np.identity(2 * nqubits, dtype=self.dtype)
            self.tableau_w_scratch[:-1, :-1] = id_array
        else:
            self._init_w_tableau(initial_tableau)

        self.update()

    def _init_w_tableau(self, input_tableau: np.ndarray):
        """
        Assigns a custom tableau to the Tableau. The method adds an empty
        scratch row to the custom tableau and assigns it to the tableau_w_scratch
        attribute.
        """
        tableau = np.array(input_tableau, dtype=self.dtype)
        # form the tableau with the scratch row
        scratch_row = np.zeros((1, tableau.shape[1]), dtype=self.dtype)

        # set the tableau with the scratch and update
        self.tableau_w_scratch = np.vstack((tableau, scratch_row))

    @property
    def nqubits(self) -> int:
        """
        The number of qubits in the tableau.
        """
        return self.__nqubits

    @nqubits.setter
    def nqubits(self, n_qubits: int):
        assert n_qubits > 0 and isinstance(
            n_qubits, int
        ), "Number of qubits should be positive integer."

        self.__nqubits = n_qubits

    # pylint: disable=attribute-defined-outside-init
    def _define_tableau_views(self) -> None:
        """
        Defines different views of the tableau array.
        To be called upon initialization or upon adding or deleting a qubit.
        """

        self.tableau = self.tableau_w_scratch[:-1, :]
        self.scratch_row = self.tableau_w_scratch[-1, :]

        # different parts of the tableau
        self.x = self.tableau[:, : self.nqubits]
        self.z = self.tableau[:, self.nqubits : -1]
        self.r = self.tableau[:, -1]

        self.x_w_scratch = self.tableau_w_scratch[:, : self.nqubits]
        self.z_w_scratch = self.tableau_w_scratch[:, self.nqubits : -1]
        self.r_w_scratch = self.tableau_w_scratch[:, -1]

        self.stabilizer_array = self.tableau[self.nqubits :, :]
        self.destabilizer_array = self.tableau[: self.nqubits, :]

        self.z_stabilizers = self.z[self.nqubits :, :]
        self.x_stabilizers = self.x[self.nqubits :, :]
        self.r_stabilizers = self.r[self.nqubits :]

    def update(self):
        """Updates the internal structure after the tableau has
        been modified"""
        dim = self.tableau_w_scratch.shape[0]
        self.nqubits = (dim - 1) // 2

        self._define_tableau_views()

    @property
    def stabilizer_set(self) -> set[str]:
        """
        Generates the stabilizers in human readable format.
        """
        return set(StabArray(self.stabilizer_array).as_paulistrings)

    @property
    def stabilizer_set_sparse_format(self) -> list[dict]:
        """
        Returns the stabilizers in more human readable format.
        This is the sparse format.

        Note: the formatter function can be called
        by user directly as well. Look for it in QCDutils

        Parameters
        ----------

        Returns
        -------
        list[dict]
            A list of dicts, each dict representing one stabilizer operator.
            The keys refer to the Pauli operators, while value refer to the
            qubit indices where the Pauli operator resides.
            E.g. +ZXIIYXZII gets returned as
            {'sign':'+', 'X':(1,5), 'Z':(0,6), 'Y':(4,)}
        """
        return sparse_formatter(set(StabArray(self.stabilizer_array).as_paulistrings))

    def create_snapshot(self) -> TableauSnapshot:
        """
        Creates a TableauSnapshot object, a snapshot of the state of the
        Tableau, that contains important properties of the
        Tableau.

        The TableauSnapshot object can then be used to restore the state of the
        Tableau at the time the TableauSnapshot was created.
        """
        return TableauSnapshot(
            nqubits=deepcopy(self.nqubits),
            rand_gen=deepcopy(self.rand_gen),
            tableau_w_scratch=deepcopy(self.tableau_w_scratch),
        )

    def restore(self, engine_snapshot: TableauSnapshot) -> None:
        """
        This method allows us to restore the state of the Tableau
        with a TableauSnapshot Object. Once the Tableau has been
        restored, the views would be redefined.
        """

        self.nqubits = engine_snapshot.nqubits
        self.rand_gen = engine_snapshot.rand_gen
        self.tableau_w_scratch = engine_snapshot.tableau_w_scratch

        self._define_tableau_views()

    def rewrite_tableau(self, input_tableau: np.ndarray) -> Tableau:
        """
        This method updates the internal tableau of this Tableau by
        returning a new Tableau with the updated tableau. The tableau is
        updated as a whole. This method also checks for compatibility of the updated
        tableau.
        """

        # Check that new Tableau has same number of qubits as old tableau. Raise printable warning.
        if not self.tableau.shape == input_tableau.shape:
            raise TableauSizeError(
                f"The Tableau has been updated from shape {self.tableau.shape} to"
                f" shape {input_tableau.shape}"
            )

        self._init_w_tableau(input_tableau=input_tableau)
        self.update()

        return self


def compare_stabilizer_set(engine_rep_1: Tableau, engine_rep_2: Tableau) -> bool:
    """Returns true if the stabilizer set of both Tableau(s) are
    equal when they are reduced.
    """
    # return false directly in case of different qubits
    if engine_rep_1.nqubits != engine_rep_2.nqubits:
        return False

    # find row echelon form of stabilizer arrays
    return is_stabarray_equivalent(
        StabArray(engine_rep_1.stabilizer_array),
        StabArray(engine_rep_2.stabilizer_array),
    )
