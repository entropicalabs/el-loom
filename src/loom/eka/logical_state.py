"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from __future__ import annotations

from functools import cached_property, reduce
from pydantic.dataclasses import dataclass
from pydantic import field_validator, model_validator


import numpy as np

from .block import Block
from .utilities.pauli_binary_vector_rep import SignedPauliOp
from .utilities.stab_array import StabArray, find_destabarray
from .utilities.validation_tools import dataclass_params


@dataclass(**dataclass_params)
class LogicalState:
    """A logical state defined by its logical operators that stabilize it. The state is
    the statevector that is simultaneously the +1 eigenvector of all the logical
    operators stabilizing it. The operators are represented as sparse Pauli operators
    on the logical level.

    Parameters
    ----------
    sparse_logical_paulistrings : tuple[str, ...]
        A tuple of strings representing the logical state stabilizers as sparse Pauli
        operators expressed on the logical level. The strings should be in the format of
        a sparse Pauli operator, e.g. "+Z1Y3".
    """

    sparse_logical_paulistrings: tuple[str, ...]

    @classmethod
    def from_stabarray(
        cls,
        stabarray: StabArray,
    ) -> LogicalState:
        """
        Create a LogicalState from a StabArray. The StabArray should be on the logical
        level, meaning that every column corresponds to a logical qubit.
        """
        if not isinstance(stabarray, StabArray):
            raise TypeError(
                "The input should be a StabArray object representing the logical "
                "operators."
            )
        sparse_logical_paulistrings = tuple(
            signed_pauli_op.as_sparse_string() for signed_pauli_op in stabarray
        )
        return LogicalState(sparse_logical_paulistrings=sparse_logical_paulistrings)

    @field_validator("sparse_logical_paulistrings", mode="before")
    @classmethod
    def _cast_to_uppercase(cls, value: tuple[str]):
        """
        Cast the sparse_logical_operators to uppercase.
        """
        if isinstance(value, str):
            value = (value,)
        return tuple(op.upper() for op in value)

    @field_validator("sparse_logical_paulistrings", mode="before")
    @classmethod
    def cast_str_to_tuple(cls, value):
        """Cast the sparse_logical_paulistrings to a tuple if it is a single string."""
        if isinstance(value, str):
            return (value,)
        return value

    @field_validator("sparse_logical_paulistrings", mode="after")
    @classmethod
    def _validate_correct_format(cls, value: tuple[str]):
        """
        Check if the sparse_logical_paulistrings are in the correct format.
        """
        # Check that the sparse_logical_operators are in the correct format by
        # attempting to initialize SignedPauliOps
        signed_pauli_ops = [SignedPauliOp.from_sparse_string(op) for op in value]

        # Find the maximum number of logical qubits required and check that it matches
        # the number of sparse Pauli strings
        max_qubits = max(op.nqubits for op in signed_pauli_ops)

        if max_qubits != len(value):
            raise ValueError(
                f"The number of sparse stabilizers ({len(value)}) does not "
                "match the maximum logical qubits required by the operators "
                f"({max_qubits})."
            )
        return value

    @model_validator(mode="after")
    def _validate_irreducibility(self):
        """
        Check that the the set of logical paulistrings is not irreducible.
        """
        if not self.stabarray.is_irreducible:
            raise ValueError("The set of logical paulistrings is not irreducible.")
        return self

    def __repr__(self) -> str:
        return f"LogicalState({self.sparse_logical_paulistrings})"

    @property
    def n_logical_qubits(self) -> int:
        """Return the number of logical qubits."""
        return len(self.sparse_logical_paulistrings)

    @cached_property
    def stabarray(self) -> StabArray:
        """
        Return the StabArray representation of the logical operator set that stabilizes
        the state. The representation is on the logical level which means that every
        column in the StabArray corresponds to a logical qubit rather than a
        data/physical qubit.
        """
        return StabArray.from_signed_pauli_ops(
            [
                SignedPauliOp.from_sparse_string(op, nqubits=self.n_logical_qubits)
                for op in self.sparse_logical_paulistrings
            ]
        )

    @cached_property
    def destabarray(self) -> StabArray:
        """
        Return the StabArray representation of the destabilizer array corresponding to
        the stabarray property.
        """
        return find_destabarray(self.stabarray)

    @classmethod
    def convert_logical_to_base_representation(
        cls, block: Block, logical_stabarray: StabArray
    ) -> StabArray:
        """
        Convert a logical StabArray representation to the base StabArray representation
        of a Block object. If the Block object describes a code with n data qubits
        encoding k logical qubits, then the logical_stabarray should be
        indexing the logical qubits from 0 to k-1. The base representation that will be
        returned will be indexing the data qubits from 0 to n-1.

        Parameters
        ----------
        block : Block
            The Block object to use to convert the logical StabArray.
        logical_stabarray : StabArray
            The logical StabArray representation to convert.

        Returns
        -------
        StabArray
            The base StabArray representation of the input logical StabArray.
        """
        if block.n_logical_qubits != logical_stabarray.nqubits:
            raise ValueError(
                f"The number of logical qubits {block.n_logical_qubits} in the Block "
                "does not match the number of logical qubits "
                f"{logical_stabarray.nqubits} in the logical StabArray."
            )

        operators_in_base_repr = [
            cls.convert_logical_pauli_op_to_base_representation(
                log_op, block.x_log_stabarray, block.z_log_stabarray
            )
            for log_op in logical_stabarray
        ]

        return StabArray.from_signed_pauli_ops(operators_in_base_repr)

    @staticmethod
    def convert_logical_pauli_op_to_base_representation(
        log_op: SignedPauliOp, x_log_stabarray: StabArray, z_log_stabarray: StabArray
    ) -> SignedPauliOp:
        """
        Convert a logical Pauli operator to its base representation. The base
        representation is defined as the representation of the operator on the data
        qubits of a Block object. The logical Pauli operator is expected to be
        represented as a SignedPauliOp object, and the x_log_stabarray and
        z_log_stabarray are the StabArray representations of the logical X and Z
        operators, respectively.

        Parameters
        ----------
        log_op : SignedPauliOp
            The logical Pauli operator to convert.
        x_log_stabarray : StabArray
            The StabArray representation of the logical X operators.
        z_log_stabarray : StabArray
            The StabArray representation of the logical Z operators.

        Returns
        -------
        SignedPauliOp
            The base representation of the logical Pauli operator.
        """
        n_data_qubits = x_log_stabarray.nqubits
        where_z = np.where(log_op.z == 1)[0]
        where_x = np.where(log_op.x == 1)[0]

        # for every logical operator find the corresponding Z, X, and Y operators
        y_op_indexes = np.intersect1d(where_z, where_x)
        z_op_indexes = np.setdiff1d(where_z, y_op_indexes)
        x_op_indexes = np.setdiff1d(where_x, y_op_indexes)

        # Construct the Y operators from the Z and X operators.
        # Note that Y = i * X * Z, so we need to have the appropriate order of
        # multiplication to get the correct sign.
        y_operators = [
            x_log_stabarray[y_idx].multiply_with_anticommuting_operator(
                z_log_stabarray[y_idx]
            )
            for y_idx in y_op_indexes
        ]

        # Multiply all the Z, X, and Y operators together to get the logical
        # operator in the base representation.
        operator = reduce(
            lambda a, b: a * b,
            [z_log_stabarray[z_idx] for z_idx in z_op_indexes]
            + [x_log_stabarray[x_idx] for x_idx in x_op_indexes]
            + y_operators,
            SignedPauliOp.identity(n_data_qubits, negative=log_op.sign),
        )

        return operator

    def get_tableau(self, block: Block) -> np.ndarray:
        """Given a Block, return the tableau of the logical state.
        TODO: When shifting to multiple Blocks, we can instead take as input a list of
        Block objects and match for example a 6 logical qubit state on 3 Block objects
        that each have 2 logical qubits.

        Parameters
        ----------
        block : Block
            The Block object to use to generate the tableau.

        Returns
        -------
        np.ndarray
            The tableau of the logical state.
        """
        if block.n_logical_qubits != self.n_logical_qubits:
            raise ValueError(
                "The number of logical qubits in the Block does not match "
                "the number of logical qubits in the LogicalState."
            )
        # get the base representation of the logical operators
        stab_log_ops = self.convert_logical_to_base_representation(
            block, self.stabarray
        )
        # get the base representation of the destabilizers of the logical operators
        destab_log_ops = self.convert_logical_to_base_representation(
            block, self.destabarray
        )
        # return the tableau by concatenating appropriately the arrays
        return np.vstack(
            (
                block.destabarray.array,
                destab_log_ops.array,
                block.reduced_stabarray.array,
                stab_log_ops.array,
            )
        )
