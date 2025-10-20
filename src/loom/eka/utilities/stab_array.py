"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from __future__ import annotations
from itertools import combinations
from typing import Sequence
from functools import reduce
from copy import deepcopy
import re
import numpy as np

from .exceptions import AntiCommutationError
from .pauli_array import PauliArray
from .pauli_array_computation import rowsum
from .pauli_binary_vector_rep import SignedPauliOp, pauliops_anti_commute
from .graph_matrix_utils import binary_gaussian_elimination
from .tableau import tableau_generates_pauli_group


# pylint: disable=too-many-lines
class StabArray(PauliArray):
    """
    A StabArray is a collection of commuting SignedPauliOps acting on the same qubit
    register.

    Parameters
    ----------
    array : np.ndarray
        The numpy array containing the stabilizers.
    validated : bool, optional
        Whether the input array has already been validated, by default False.

    Attributes
    ----------
    nqubits : int
        The number of qubits that the PauliArray operators act on.
    x : np.ndarray
        The array representing the X-component of the PauliArray in binary representation.
    z : np.ndarray
        The array representing the Z-component of the PauliArray in binary representation.
    """

    array: np.ndarray
    nqubits: int
    DTYPE = SignedPauliOp.DTYPE

    def __init__(self, array: np.ndarray, validated: bool = False) -> None:
        """
        Initialization of a StabArray.

        Parameters
        ----------
        array : np.ndarray
            The numpy array containing the stabilizers.
        validated : bool, optional
            Whether the input array has already been validated, by default False.

        Raises
        ------
        TypeError
            When the input is not a numpy array.
        ValueError
            When the input array is not 2-dimensional or contains values other than 0
            and 1.
        AntiCommutationError
            When the input array contains anti-commuting operators.
        """
        if not validated:
            if not isinstance(array, np.ndarray):
                raise TypeError("The input has to be a numpy array.")
                # find the SignedPauliOp's that make up the StabArray

            if not array.ndim == 2:
                raise ValueError("The numpy array has to be 2 dimensional")

            if np.any(array * (array - 1)):
                raise ValueError("The array should only contain 0s and 1s.")

            if array.dtype != SignedPauliOp.DTYPE:
                # convert the array to the correct dtype
                array = array.astype(SignedPauliOp.DTYPE)

            nstabs = array.shape[0]
            # check for anti-commuting pairs of operators
            if any(
                pauliops_anti_commute(
                    SignedPauliOp(array[i], validated=True),
                    SignedPauliOp(array[j], validated=True),
                )
                for i in range(nstabs)
                for j in range(i + 1, nstabs)
            ):
                raise AntiCommutationError(
                    "StabArray should only contain commuting operators."
                )

        self.array = array

    def copy(self) -> StabArray:
        """
        Returns a deep copy of the StabArray object.

        Returns
        -------
        SignedPauliOp
            Deep copy of the StabArray object
        """
        return deepcopy(self)

    @classmethod
    def from_signed_pauli_ops(
        cls, signed_pauli_ops: Sequence[SignedPauliOp], validated: bool = False
    ) -> StabArray:
        """
        Create a StabArray from a sequence of SignedPauliOps.

        Parameters
        ----------
        signed_pauli_ops : Sequence[SignedPauliOp]
            The sequence of SignedPauliOps to be converted to a StabArray.

        Returns
        -------
        StabArray
            The StabArray created from the sequence of SignedPauliOps.

        Raises
        ------
        TypeError
            When the input sequence contains elements that are not of type
            SignedPauliOp.
        ValueError
            When the input sequence contains elements with different numbers of qubits.
        """
        if any(not isinstance(p, SignedPauliOp) for p in signed_pauli_ops):
            raise TypeError(
                "All elements of the input sequence should be of type SignedPauliOp."
            )

        if any(
            p1.nqubits != p2.nqubits
            for p1, p2 in zip(signed_pauli_ops, signed_pauli_ops[1:], strict=False)
        ):
            raise ValueError(
                "All elements of the input sequence should have the same number of"
                " qubits."
            )

        nstabs = len(signed_pauli_ops)
        ncols = 2 * signed_pauli_ops[0].nqubits + 1 if nstabs > 0 else 0

        array = np.zeros((nstabs, ncols), dtype=SignedPauliOp.DTYPE)
        for i, op in enumerate(signed_pauli_ops):
            array[i, :] = op.array

        return cls(array, validated=validated)

    @classmethod
    def trivial(cls) -> StabArray:
        """
        Return a trivial StabArray.
        """
        return StabArray(np.zeros((0, 0), dtype=cls.DTYPE), validated=True)

    @property
    def nstabs(self) -> int:
        """
        The number of stabilizers in the StabArray.
        """
        return self.array.shape[0]

    @property
    def is_trivial(self) -> bool:
        """
        Checks whether the StabArray is trivial, i.e. if it contains no stabilizers.
        """
        return self.nstabs == 0

    @property
    def is_irreducible(self) -> bool:
        """
        Checks whether the StabArray is irreducible, i.e. if it contains no stabilizers that can
        be generated as a product of other stabilizers.
        """
        return reduce_stabarray(self).nstabs == self.nstabs

    @property
    def as_paulistrings(self) -> list[str]:
        """
        Representation of Pauli operators in the StabArray as a list of strings.
        """
        return [str(pauliop) for pauliop in self]

    def __getitem__(self, key: int | slice):
        """
        Overloads the [] operator to return a SignedPauliOp or a list of
        SignedPauliOp.
        """
        # Return a list of SignedPauliOp
        if isinstance(key, slice):
            indices = range(*key.indices(self.nstabs))
            return [SignedPauliOp(self.array[i, :], validated=True) for i in indices]
        if isinstance(key, np.ndarray) and key.ndim == 1 and key.dtype == np.bool_:
            # Support for boolean indexing
            return [
                SignedPauliOp(self.array[i, :], validated=True)
                for i in range(self.nstabs)
                if key[i]
            ]
        # or a single SignedPauliOp
        return SignedPauliOp(self.array[key, :], validated=True)

    def __iter__(self):
        """
        Overloads the `in` operator to go through all SignedPauliOps.
        """
        return (
            SignedPauliOp(self.array[i, :], validated=True) for i in range(self.nstabs)
        )

    def __repr__(self) -> str:
        return f"StabArray({self.as_paulistrings})"


########################################################################################
# Methods of StabArray
########################################################################################


def find_destabarray(  # pylint: disable=too-many-locals,too-many-branches
    stabarr: StabArray, partial_destabarray: StabArray | None = None
) -> StabArray:
    """
    Given a stabilizer array of a state, find the destabilizer array.

    Parameters
    ----------
    stabilizer_array : StabArray
        The stabilizer array of a state. It must contain n independent stabilizers where
        n is the number of qubits.
    partial_destabarray : StabArray, optional
        A destabilizer array that partially destabilizes the state, i.e. it contains
        m < n operators that each anti-commute with exactly one state stabilizer. The
        final destabilizer array will contain these operators in the appropriate rows.

    Returns
    -------
    StabArray
        The destabilizer array.
    """
    if partial_destabarray is None:
        partial_destabarray = StabArray.trivial()

    nrows, ncols = stabarr.array.shape
    nqubs = stabarr.nqubits

    sarr_reduced = reduce_stabarray(stabarr)
    if sarr_reduced.nstabs != stabarr.nstabs:
        raise ValueError(
            "The StabArray should correspond to an irreducible set of stabilizers."
        )
    if stabarr.nstabs != nqubs:
        raise ValueError(
            "The StabArray should describe a full state, i.e. the number of stabilizers"
            f" ({stabarr.nstabs}) should be equal to the number of qubits ({nqubs})."
        )

    if partial_destabarray.nqubits != nqubs and not partial_destabarray.is_trivial:
        raise ValueError(
            "The partial destabilizer array must have the same number of qubits as the"
            " stabilizer array."
        )

    # Find for every operator of the partial destabilizer array the operators of the
    # stabilizer array that anti-commute with it.
    anti_commuting_indices = [
        [i for i, s_op in enumerate(stabarr) if pauliops_anti_commute(d_op, s_op)]
        for d_op in partial_destabarray
    ]

    if any(len(indices) != 1 for indices in anti_commuting_indices):
        raise ValueError(
            "Each operator of the partial destabilizer array must anti-commute with "
            "exactly one stabilizer."
        )

    # Make anti_commuting_indices a list of integers
    anti_commuting_indices = [indices[0] for indices in anti_commuting_indices]

    # Ensure that these indices are unique
    if len(anti_commuting_indices) != len(set(anti_commuting_indices)):
        raise ValueError(
            "Each operator of the partial destabilizer array must anti-commute with "
            "a different stabilizer."
        )

    # Find characteristic indices of stabilizer array
    char_idcs = [np.argmax(sarr_reduced.array[irow, :] == 1) for irow in range(nrows)]

    # Initialize d_array such that d_array and s_array
    # generate the full pauli set
    # Do not include the sign! --> (ncols-1)
    leftover_idx = set(range(ncols - 1)) - set(char_idcs)

    # builds an array which indicates which set of indices occur in leftover_idx
    d_array = np.zeros(stabarr.array.shape, dtype=SignedPauliOp.DTYPE)
    for row_idx, col_idx in enumerate(leftover_idx):
        d_array[row_idx, col_idx] = 1

    # translates column indices into the corresponding qubit indices
    row_qubit = np.array([col_idx % nqubs for col_idx in leftover_idx])

    # builds a dictionary out of qubits (keys) that appear twice and their
    # corresponding column indices (values)
    doubly_idxd_qubits = dict(
        filter(
            lambda tup: len(tup[1]) > 1,
            [(qubit, np.where(row_qubit == qubit)[0]) for qubit in set(row_qubit)],
        )
    )

    # Find which qubits have not been indexed by the d_array
    qubits_not_indexed = list(set(range(nqubs)) - set(row_qubit))

    # All possible anti-commuting single pauli combinations
    possible_combos = (
        ((1, 0), (0, 1)),  # X,Z
        ((1, 0), (1, 1)),  # X,Y
        ((0, 1), (1, 0)),  # Z,X
        ((0, 1), (1, 1)),  # Z,Y
        ((1, 1), (1, 0)),  # Y,X
        ((1, 1), (0, 1)),  # Y,Z
    )

    # Match each non-indexed qubit with the two rows of
    # a doubly indexed qubit.
    # Make sure that these rows eventually commute by using the
    # non-indexed qubit.
    # Simultaneously, ensure that this doesn't break the fact that
    # the concatenation of d_array,s_array generates the full pauli set.
    for non_indexed_qubit, (row1, row2) in zip(
        qubits_not_indexed, doubly_idxd_qubits.values(), strict=True
    ):
        # Cycle through the possible combinations
        # until we find one that doesn't break the
        # generation of the full pauli set
        for row1_vals, row2_vals in possible_combos:
            # assign the pauli of the the non-indexed qubits

            d_array[row1, non_indexed_qubit] = row1_vals[0]
            d_array[row1, non_indexed_qubit + nqubs] = row1_vals[1]

            d_array[row2, non_indexed_qubit] = row2_vals[0]
            d_array[row2, non_indexed_qubit + nqubs] = row2_vals[1]

            tableau_to_check = np.vstack((d_array, stabarr.array))
            if tableau_generates_pauli_group(tableau_to_check):
                # The set generates the pauli group.
                # We can continue to the next set of rows that
                # don't commute.
                break

    # Fix commutation and anti-commutation relations with the
    # stabilizer array.
    for i in range(nrows):
        s_operator = stabarr[i]

        # find with which d_array rows s_operator anti_commutes
        anti_commuting_rows = [
            j
            for j in range(nrows)
            if pauliops_anti_commute(s_operator, SignedPauliOp(d_array[j, :]))
        ]

        # Make sure that s_i anti-commutes
        # with its corresponding row
        if i not in anti_commuting_rows:
            # Get a row index that is larger than i
            # to not mess up progress.
            j_swap = next(j for j in anti_commuting_rows if j > i)

            # swap j_swap with i row
            d_array[[i, j_swap]] = d_array[[j_swap, i]]

            # Now that the row has been swapped into i,
            # remove it from the rows that need to be fixed
            anti_commuting_rows.remove(j_swap)
        else:
            # i is already in place.
            # Remove it from the rows that need to be fixed
            anti_commuting_rows.remove(i)

        # Multiply i into anti-commuting rows such that
        # s_i commutes with all d_j, j!=i
        for idx in anti_commuting_rows:
            d_array[idx, :] ^= d_array[i, :]

    # At this point, d_array is a valid destabilizer of the input StabArray.
    # Now we need to modify it to include the operators of partial_destabarray.

    # Set the rows of the destabilizer array from the partial destabilizer array
    for i, idx in enumerate(anti_commuting_indices):
        d_array[idx] = partial_destabarray.array[i]

    # Fix the anti-commutation relations between the destabilizer array rows occurred
    # due to setting the rows from the partial destabilizer array.
    if not partial_destabarray.is_trivial:
        # This if statement is to avoid unnecessary computations if the partial
        # destabilizer array is trivial.
        for i in set(range(nrows)) - set(anti_commuting_indices):
            # Rows with index i are going to be altered only.
            # This is why i does not include the anti_commuting_indices since they were
            # set and fixed in the previous step.
            for j in range(nrows):
                # When i == j, the operators trivially commute.
                if pauliops_anti_commute(
                    SignedPauliOp(d_array[i, :]), SignedPauliOp(d_array[j, :])
                ):
                    # Because the anti-commutation is not correct, we need to fix it.
                    # We do that by multiplying the j-th stabilizer into the i-th
                    # destabilizer.
                    # This works because [d_i, s_j] = 0 while {d_j, s_j} = 0.
                    d_array[i, :] ^= stabarr.array[j, :]

    return StabArray(d_array, validated=True)


def invert_bookkeeping_matrix(bookkeeping_matrix: np.ndarray) -> np.ndarray:
    """
    Invert the bookkeeping matrix.

    Parameters
    ----------
    bookkeeping_matrix : np.ndarray
        The bookkeeping matrix to be inverted.

    Returns
    -------
    np.ndarray
        The inverted bookkeeping matrix.
    """
    # We need to invert it and cast it to a boolean array
    return np.linalg.inv(bookkeeping_matrix).astype(bool)


def is_stabarray_equivalent(stab_arr0: StabArray, stab_arr1: StabArray) -> bool:
    """
    Check if two stabilizer arrays are equivalent.

    Parameters
    ----------
    stab_arr0 : np.ndarray
        A stabilizer array to be compared with another.
    stab_arr1 : np.ndarray
        A stabilizer array to be compared with another.

    Returns
    -------
    bool
        Whether the stabilizer arrays are equivalent.
    """
    if stab_arr0.nqubits != stab_arr1.nqubits:
        # the stabilizer arrays should have the same number of qubits to be equivalent
        return False

    stab_arr0_bge = reduce_stabarray(stab_arr0)
    stab_arr1_bge = reduce_stabarray(stab_arr1)
    return np.all(stab_arr0_bge.array == stab_arr1_bge.array)


def is_subset_of_stabarray(
    pauli_obj_sub: SignedPauliOp | StabArray, stab_arr_super: StabArray
) -> bool:
    """Check if a StabArray or SignedPauliOp is a subset of a StabArray.

    Parameters
    ----------
    pauli_obj_sub : SignedPauliOp | StabArray
        The SignedPauliOp or StabArray to be checked if it is a subset.
    stab_arr_super : StabArray
        The StabArray to be checked if it is a superset.

    Returns
    -------
    bool
        Whether the StabArray or SignedPauliOp is a subset of the StabArray.
    """
    # Convert the SignedPauliOp to a single row StabArray
    if isinstance(pauli_obj_sub, SignedPauliOp):
        pauli_obj_sub = StabArray.from_signed_pauli_ops([pauli_obj_sub], validated=True)

    # Check the validity of the inputs
    if not isinstance(pauli_obj_sub, StabArray):
        raise TypeError(
            "The pauli_obj_sub input should be a SignedPauliOp or a StabArray, not "
            f"{type(pauli_obj_sub)}."
        )
    if not isinstance(stab_arr_super, StabArray):
        raise TypeError(
            "The stab_arr_super input should be a StabArray, not "
            f"{type(stab_arr_super)}."
        )

    # If the number of qubits of the subarray is not the same as the superarray,
    # then the subarray cannot be a subset
    if pauli_obj_sub.nqubits != stab_arr_super.nqubits:
        return False

    # stack the arrays
    try:
        stab_arrays_merged = merge_stabarrays((stab_arr_super, pauli_obj_sub))
    except AntiCommutationError:
        # merging the arrays failed due to some operators anti-commuting
        return False
    # reduce the super set
    stab_arr_super_reduced = reduce_stabarray(stab_arr_super)
    # reduce the merged array
    stab_arrays_reduced = reduce_stabarray(stab_arrays_merged)
    # if the number of stabilizers of the superset is the same as the reduced,
    # then indeed stab_arr_sub was a subset
    return stab_arrays_reduced.nstabs == stab_arr_super_reduced.nstabs


def merge_stabarrays(stabarr_tuple: tuple[StabArray, ...]) -> StabArray:
    """
    Merges StabArrays.

    Parameters
    ----------
    stabarr_tuple : tuple[StabArray, ...]
        The tuple of StabArrays to be merged.

    Returns
    -------
    StabArray
        The merged StabArray.
    """
    if len(stabarr_tuple) < 2:
        raise ValueError("At least two StabArrays are needed to merge them.")

    # check if the StabArrays have the same number of qubits
    if any(stabarr.nqubits != stabarr_tuple[0].nqubits for stabarr in stabarr_tuple):
        raise ValueError("The StabArrays should have the same number of qubits.")

    for star1, star2 in combinations(stabarr_tuple, 2):
        # check if any of the operators of any pair anti-commute
        if any(pauliops_anti_commute(p1, p2) for p1 in star1 for p2 in star2):
            raise AntiCommutationError(
                "The StabArrays should only contain commuting operators."
            )

    return StabArray.from_signed_pauli_ops(
        [p for stabarr in stabarr_tuple for p in stabarr], validated=True
    )


def reduce_stabarray(stabarr: StabArray) -> StabArray:
    """
    Perform stabilizer BGE and then remove any trivial operators.

    Parameters
    ----------
    stabarr : StabArray
        The stabilizer array to be reduced.

    Returns
    -------
    StabArray
        The reduced stabilizer array.
    """
    stabarr_bge = stabarray_bge(stabarr)
    # remove empty stabilizers
    return StabArray.from_signed_pauli_ops(
        [p_op for p_op in stabarr_bge if not p_op.is_trivial],
        validated=True,
    )


def reduce_stabarray_with_bookkeeping(
    stabarr: StabArray,
) -> tuple[StabArray, np.ndarray]:
    """
    Perform stabilizer BGE and then remove any trivial operators. Also return the
    bookkeeping matrix, which is a matrix that keeps track of the row operations that
    were performed on the stabilizer array.

    Parameters
    ----------
    stabarr : StabArray
        The stabilizer array to be reduced.

    Returns
    -------
    tuple[StabArray, np.ndarray]
        A tuple containing the reduced stabilizer array and the bookkeeping matrix, i.e.
        the matrix that keeps track of the row operations that were performed on the
        stabilizer array.
    """
    stabarr_bge, bookkeeping_matrix = stabarray_bge_with_bookkeeping(stabarr)
    # find which operators are non trivial after BGE
    non_trivial_ops = [p_op for p_op in stabarr_bge if not p_op.is_trivial]
    return (
        StabArray.from_signed_pauli_ops(non_trivial_ops, validated=True),
        bookkeeping_matrix,
    )


def reindex_stabarray(stab_array: StabArray, new_idcs: list[int]) -> StabArray:
    """Returns an instance of a stabilizer array but with the qubit columns reindexed.

    Parameters
    ----------
    stab_array : StabArray
        The StabArray to be reindexed.
    new_idcs : list[int]
        The new indices.

    Returns
    -------
    StabArray
        The reindexed StabArray.

    Raises
    ------
    ValueError
        When the new_idcs are not unique or not as many as the n_qubits.
    """
    nqubits = stab_array.nqubits
    if len(new_idcs) != len(set(new_idcs)):
        raise ValueError("new_idcs has to have unique elements.")
    if len(new_idcs) != nqubits:
        raise ValueError("new_idcs should be as many as the n_qubits")

    array = stab_array.array.copy()
    reindexed_array = array.copy()

    new_idcs = np.array(new_idcs)

    # change the X's
    reindexed_array[:, 0:nqubits] = array[:, new_idcs]
    # change the Z's
    reindexed_array[:, nqubits : 2 * nqubits] = array[:, new_idcs + nqubits]

    return StabArray(reindexed_array, validated=True)


def swap_stabarray_rows(stabarr: StabArray, i: int, j: int) -> StabArray:
    """
    Swap two rows of a stabilizer array in-place.

    Parameters
    ----------
    stabarr : StabArray
        The stabilizer array to be modified.
    i : int
        The index of the first row.
    j : int
        The index of the second row.

    Returns
    -------
    StabArray
        The modified stabilizer array.
    """
    stabarr.array[[i, j]] = stabarr.array[[j, i]]
    return stabarr


def sparse_formatter(stab_list: set[str], **keywords) -> list[dict]:
    """Given a set of Pauli string operators,
    return a list of dicts, where dict keys are ``{'sign', 'X', 'Y', 'Z'}``. For keys
    ``X``, ``Y``, ``Z``, value is a list of indices/positions where the i-th qubit in
    the Pauli operator is respectively either X, Y, Z.

    There is a keyword option, to allow the user to pass a dictionary
    to convert *from* qubit index in CliffordSim, to some other indexing
    that is more convenient for the user.

    Parameters
    ----------
    stab_array : set[str]
        An unordered list of strings, where each string
        represents a Pauli string for a stabilizer operator in the set.

    keywords: {'convert_dict'}
        By supplying a dictionary that converts between qubit indices in CliffordSim
        into qubit IDs in an embedded QEC code, that user can return dicts where the values
        refer directly to IDs in an embedded QEC code.

    Returns
    -------
    output_format : list[dict]
        Reformats each string in the sparse format.
        E.g. +ZXIIYXZII gets returned as ``{'sign':'+', 'X':(1,5), 'Z':(0,6), 'Y':(4,)}``
    """

    output_format = []
    convert = False

    for key in keywords:
        if key == "convert_dict":
            assert isinstance(keywords["convert_dict"], dict)
            convert = True
            convert_dict = keywords["convert_dict"]

    for stab_op in stab_list:
        assert stab_op[0] == "+" or stab_op[0] == "-", "Improper format of Stab Op."
        formatted_dict = {"sign": stab_op[0], "X": [], "Y": [], "Z": []}

        xlist = [m.start() for m in re.finditer("X", stab_op[1:])]
        zlist = [m.start() for m in re.finditer("Z", stab_op[1:])]
        ylist = [m.start() for m in re.finditer("Y", stab_op[1:])]

        if len(xlist) != 0:
            if convert:
                xlist = [convert_dict[item] for item in xlist]
            formatted_dict["X"].extend(xlist)

        if len(zlist) != 0:
            if convert:
                zlist = [convert_dict[item] for item in zlist]
            formatted_dict["Z"].extend(zlist)

        if len(ylist) != 0:
            if convert:
                ylist = [convert_dict[item] for item in ylist]
            formatted_dict["Y"].extend(ylist)

        output_format.append(formatted_dict)

    return output_format


def stabarray_bge(stabarr: StabArray) -> StabArray:
    """
    Perform binary gaussian elimination on a stabilizer array.

    Parameters
    ----------
    stabarr : StabArray
        The stabilizer array to perform binary gaussian elimination on.

    Returns
    -------
    StabArray
        The stabilizer array in its reduced row echelon form.
    """
    return stabarray_bge_with_bookkeeping(stabarr)[0]


def stabarray_bge_with_bookkeeping(stabarr: StabArray) -> tuple[StabArray, np.ndarray]:
    """
    Performs binary gaussian elimination on a stabilizer array and returns
    the array in its reduced row echelon form.
    Reference:
    https://en.wikipedia.org/wiki/Row_echelon_form#Reduced_row_echelon_form

    Bookkeeping matrix explanation:
    It also returns the bookkeeping matrix, which is a matrix that keeps track of the
    row operations that were performed on the stabilizer array, i.e. how to transform
    the input stabilizer array into the reduced row echelon form. If the (i, j)-th
    element of the bookkeeping matrix is True, then the i-th stabilizer of the row
    echelon form contains the j-th row of the initial stabilizer array.
    In other words, to get the i-th stabilizer of the row echelon form, one should
    multiply all the stabilizers of the initial stabilizer array with index j that
    have a True value in the (i, j)-th element of the bookkeeping matrix.

    Parameters
    ----------
    stabarr : StabArray
        The stabilizer array to perform binary gaussian elimination on.

    Returns
    -------
    tuple[StabArray, np.ndarray]
        A tuple containing the reduced stabilizer array and the bookkeeping matrix, i.e.
        the matrix that keeps track of the row operations that were performed on the
        stabilizer array.
    """
    stabarr_copy = stabarr.copy()
    m, n = stabarr_copy.array.shape

    # Initialize the bookkeeping matrix
    bookkeeping_matrix = np.eye(m, dtype=bool)

    h = 0  # Initialization of the pivot row
    k = 0  # Initialization of the pivot column
    while h < m and k < n:
        # Find the k-th pivot:
        i_max = np.argmax(stabarr_copy.array[h:, k]) + h
        if stabarr_copy.array[i_max, k] == 0:
            # No pivot in this column, pass to next column
            k += 1
        else:
            # Swap rows h, imax
            stabarr_copy = swap_stabarray_rows(stabarr_copy, h, i_max)

            # Update the bookkeeping matrix
            bookkeeping_matrix[[h, i_max]] = bookkeeping_matrix[[i_max, h]]

            # Need to apply rowsum to eliminate 1s
            # on column k.

            # Do for all rows excluding pivot
            irange = np.concatenate([np.arange(h), np.arange(h + 1, m)])
            # Find rows with 1 in column k
            irange_with_1_in_col_k = irange[stabarr_copy.array[irange, k] == 1]

            # Apply rowsum to row idx if if st_ar[idx, k] == 1
            stabarr_copy = reduce(
                lambda st, i: rowsum(st, i, h),
                irange_with_1_in_col_k,
                stabarr_copy,
            )

            # XOR h into the irange_with_1_in_col_k in the bookkeeping_matrix
            bookkeeping_matrix[irange_with_1_in_col_k] = np.bitwise_xor(
                bookkeeping_matrix[irange_with_1_in_col_k], bookkeeping_matrix[h]
            )

            # Increase pivot row and column
            h += 1
            k += 1

    return stabarr_copy, bookkeeping_matrix


# pylint: disable=too-many-locals
def stabarray_standard_form(stabarr: StabArray) -> tuple[StabArray, int, list[int]]:
    """Transform a stabilizer array into a standard form. Note that this means that the
    stabilizer array is reduced and reindexed.

        1. Reduce the stabilizer array.
        
        2. Reorder the qubits such that the X part of the stabilizer array becomes of 
        the form
        
            (I, A)
            (0, 0)

        where I is the identity matrix of size r and A is a r x (n-r) matrix and the 0 \
        arrays have n-k-r rows.
        
        3. Multiply rows of the stabilizer array so that it becomes of the form
        
            (I, A1, A2 | B, 0, C)
            (0,  0,  0 | D, I, E)
            
        where I is the identity matrix, A1, A2, B, C, D and E are matrices.
    
    Dimension explanation:
    
        - The left blocks comprise the X part of the stabilizer array and the right \
        blocks comprise the Z part of the stabilizer array.
        - The top blocks have r rows and the bottom blocks have n-k-r rows, where n is \
        the number of qubits, k is the number of logical qubits and r is the rank of \
        the X part of the stabilizer array.
        - From left to right, the number of columns in the blocks are r, n-k-r, k for \
        each of the Z and X parts.
        
    More info:
    
        Nielsen, M. A., & Chuang, I. L. (2011). (p.470-471) Quantum Computation and Quantum
        Information: 10th Anniversary Edition. Cambridge University Press

    Parameters
    ----------
    stabarr : StabArray
        The stabilizer array to be transformed.

    Returns
    -------
    tuple[StabArray, int, list[int]]
        A tuple containing the reindexed stabilizer array, the rank of the X part of the
        stabilizer array and the inverse qubit map. The inverse qubit map can be used to
        reindex the stabilizer array to the original qubit order.
    """
    # Reduce the stabilizer array
    stabarr_reduced = reduce_stabarray(stabarr)

    # Name quantities to shorten notation
    n = stabarr.nqubits  # Number of qubits
    k = stabarr.nqubits - stabarr_reduced.nstabs  # Number of logical qubits

    # Find the characteristic indices of the stabilizer array, i.e. the indices of the
    # first non-zero entry in each row
    characteristic_idx = [
        np.argmax(signed_pauli_op.array) for signed_pauli_op in stabarr_reduced
    ]
    # Find the indices of the characteristic rows that correspond to the X part of the
    # stabilizer array
    characteristic_idx_x = [idx for idx in characteristic_idx if idx < n]

    r = len(characteristic_idx_x)  # The rank of the X part of the stabilizer array

    # Find the indices of the rows that are not characteristic of the X part
    rest_idx = [idx for idx in range(n) if idx not in characteristic_idx_x]

    # We need to order the rest_idx such that the identity array can be generated
    # at the bottom of the stabilizer array. This is a concatenation of the I, E blocks
    # in the standard form of the stabilizer array.
    # 1. Isolate the bottom right block of the stabilizer array
    bottom_right_ie_block = stabarr_reduced.z[r:, np.array(rest_idx)]
    # 2. Perform Binary Gaussian Elimination
    bottom_right_ie_block_bge = binary_gaussian_elimination(bottom_right_ie_block)
    # 3. Find the characteristic indices of the bottom right block
    ie_block_characteristic_idx = [np.argmax(row) for row in bottom_right_ie_block_bge]
    # 4. Find the indices of the rows that are not characteristic
    ie_block_non_characteristic_idx = [
        idx for idx in range(len(rest_idx)) if idx not in ie_block_characteristic_idx
    ]
    # 5. Reorder the rest_idx such that the characteristic indices come first. This
    # means that the identity matrix can be generated at the bottom of the stabilizer
    rest_idx_reordered = [
        rest_idx[idx]
        for idx in ie_block_characteristic_idx + ie_block_non_characteristic_idx
    ]

    # Construct a new map that reorders the qubits such that the X part of the
    # stabilizer array becomes of the form:
    # (I, A)
    # (0, 0)
    # where I is the identity matrix of size r and A is a r x (n-r) matrix and the 0
    # arrays have n-k-r columns
    qbit_map = characteristic_idx_x + rest_idx_reordered
    inv_qbit_map = [qbit_map.index(idx) for idx in range(stabarr.nqubits)]

    # Reindex the stabilizer array according to the new map.
    # Only the qubits are reindexed by swapping columns in the X and Z parts of the
    # stabilizer array.
    reindexed_stabarr = reindex_stabarray(stabarr_reduced, qbit_map)

    # Create the identity matrix on the bottom part of the reindexed stabilizer array
    for i in range(n - k - r):
        # Find the array that is to be made an identity matrix
        arr_to_be_eye = reindexed_stabarr.z[r:, r : n - k]
        # NOTE: Because arr_to_be_eye is a subarray of the reindexed_stabarr, we may
        # need to offset the indices by r when modifying the reindexed_stabarr

        # if the i-th diagonal element is not 1, then swap the i-th row with a row that has
        # 1 in the i-th column
        if arr_to_be_eye[i, i] == 0:
            # find a row with 1 in the i-th column and swap it with the i-th row
            row_with_one = np.argmax(arr_to_be_eye[i:, i] == 1) + i
            # if such a row is not found, then raise an error
            if row_with_one == i:
                raise ValueError(
                    "Cannot create identity matrix inside the reindexed "
                    "stabilizer array"
                )
                # NOTE: if this ever fails for a valid code, it may be worth checking if
                # the error can be resolved by reindexing the StabArray further

            # swap the rows
            reindexed_stabarr = swap_stabarray_rows(
                reindexed_stabarr, r + i, r + row_with_one
            )

        # make the i-th column of the reindexed stabilizer array to be that of an
        # identity matrix
        rows_to_make_zero = [
            j for j in range(n - k - r) if j != i and arr_to_be_eye[j, i] == 1
        ]
        reindexed_stabarr = reduce(
            lambda acc, row, _i=i: rowsum(acc, r + row, r + _i),
            rows_to_make_zero,
            reindexed_stabarr,
        )

    return reindexed_stabarr, r, inv_qbit_map


def subtract_stabarrays(
    stab_array: StabArray, stab_array_to_remove: StabArray
) -> StabArray:
    """Given 2 stabilizer arrays, remove from the first one any
    occurrence of stabilizers found in the second one.

    Parameters
    ----------
    stab_array : StabArray
        The StabArray from which the other is subtracted.
    stab_array_to_remove : StabArray
        The StabArray that is subtracted from the other.

    Returns
    -------
    StabArray
        The resulting StabArray.
    """
    # check if the StabArrays have the same number of qubits
    if stab_array.nqubits != stab_array_to_remove.nqubits:
        raise ValueError("The StabArrays should have the same number of qubits.")

    if any(
        pauliops_anti_commute(p, q) for p in stab_array_to_remove for q in stab_array
    ):
        raise ValueError("The operators of the StabArrays should not anti-commute.")

    # set stab_array in its reduced form
    stab_array_to_remove = reduce_stabarray(stab_array_to_remove)

    # Do the subtraction
    subtracted_paulis = []
    for pauli_op in stab_array:
        # Create a temporary StabArray with the operator from which we will subtract
        # the operators of the other StabArray
        # This gives us access of the rowsum function
        temp_stabarr = StabArray.from_signed_pauli_ops(
            list(stab_array_to_remove) + [pauli_op],
            validated=True,
        )

        # Find for every operator A in stab_array_to_remove the characteristic index:
        # np.argmax(st.array[idx, :] == 1)
        # If at the characteristic index the last operator has 1, then multiply
        # A into the last operator.
        temp_stabarr = reduce(
            lambda st, idx, _pauli=pauli_op: (
                rowsum(st, -1, idx)
                if _pauli.array[np.argmax(st.array[idx, :] == 1)] == 1
                else st
            ),
            range(temp_stabarr.nstabs - 1),
            temp_stabarr,
        )

        subtracted_paulis.append(temp_stabarr[-1])

    # Combine the result
    result_arr = StabArray.from_signed_pauli_ops(subtracted_paulis, validated=True)

    # Reduce the result
    return reduce_stabarray(result_arr)
