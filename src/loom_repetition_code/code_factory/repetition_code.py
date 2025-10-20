"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=duplicate-code
from __future__ import annotations
from uuid import uuid4
from functools import cached_property
from pydantic.dataclasses import dataclass

from loom.eka import (
    Block,
    Lattice,
    LatticeType,
    PauliOperator,
    Stabilizer,
)

from loom.eka.utilities import Direction, dataclass_params


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-branches
@dataclass(**dataclass_params)
class RepetitionCode(Block):
    """
    A sub-class of `Block` that represents a repetition code block.
    Contains methods to create a repetition code block.
    """

    @classmethod
    def create(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-branches
        cls,
        d: int,
        check_type: str,
        lattice: Lattice,
        unique_label: str | None = None,
        position: tuple[int] = (0,),
        logical_x_operator: PauliOperator | None = None,
        logical_z_operator: PauliOperator | None = None,
        skip_validation: bool = False,
    ) -> RepetitionCode:
        """Create a `Block` object for a repetition code block. The repetition code is
        defined as a linear chain with open boundary conditions.

        Parameters
        ----------
        d : int
            Code distance and size of the chain.
        check_type : str
            Type of code stabilizers, either "X" (phase-flip) or "Z" (bit-flip).
        lattice : Lattice
            Lattice on which the block is defined. The qubit indices depend on the type
            of lattice.
        unique_label : str, optional
            Label for the block. It must be unique among all blocks in the initial CRD.
            If no label is provided, a unique label is generated automatically using the
            uuid module.
        position : tuple[int, ], optional
            Position of the top left corner of the block on the lattice, by
            default (0,).
        logical_x_operator: PauliOperator | None, optional
            Logical X operator. For bit-flip code, if None is provided, by default
            the full chain of qubits is selected. For phase-flip code, if None is
            provided, by default the first qubit is selected.
        logical_z_operator: PauliOperator | None, optional
            Logical Z operator. For phase-flip code, if None is provided, by default
            the full chain of qubits is selected. For bit-flip code, if None is
            provided, by default the first qubit is selected.
        skip_validation : bool, optional
            Skip validation of the block object, by default False.

        Returns
        -------
        Block
            Block object for a repetition code chain
        """

        # Input validation
        # Check distance is a positive integer
        if not isinstance(d, int) or d <= 0:
            raise ValueError(f"`d` must be a positive integer. Got '{d}' instead.")

        # Check check_type is either "X" or "Z"
        if check_type not in ["X", "Z"]:
            raise ValueError(
                f"`check_type` must be either 'X' or 'Z'. Got '{check_type}' instead."
            )

        # Check lattice is linear
        if lattice.lattice_type != LatticeType.LINEAR:
            raise ValueError(
                "The creation of repetition chains is "
                "currently only supported for linear lattices. Instead "
                f"the lattice is of type {lattice.lattice_type}."
            )

        # Ensure position is a tuple of integers
        if not isinstance(position, tuple) or any(
            not isinstance(x, int) for x in position
        ):
            raise ValueError(
                f"`position` must be a tuple of integers. Got '{position}' instead."
            )

        if len(position) != lattice.n_dimensions:
            raise ValueError(
                f"`position` has length {len(position)} while length "
                f"{lattice.n_dimensions} is required to match the lattice dimension."
            )

        # Assign uuid if not label provided
        if unique_label is None:
            unique_label = str(uuid4())

        # Check logical operator lengths
        if logical_x_operator is not None and check_type == "Z":
            if len(logical_x_operator.pauli) != d:
                raise ValueError(
                    "Support of input X logical should be equal to distance"
                )

        if logical_z_operator is not None and check_type == "X":
            if len(logical_z_operator.pauli) != d:
                raise ValueError(
                    "Support of input Z logical should be equal to distance"
                )

        # Create stabilizers
        stabilizers = [
            Stabilizer(
                pauli=check_type * 2,
                data_qubits=[(i, 0), (i + 1, 0)],
                ancilla_qubits=[(i, 1)],
            )
            for i in range(d - 1)
        ]

        # Create logical operators
        if logical_x_operator is None:
            logical_x_operator = (
                PauliOperator(pauli=check_type, data_qubits=[(0, 0)])
                if check_type == "X"
                else PauliOperator(
                    pauli="X" * d, data_qubits=[(i, 0) for i in range(d)]
                )
            )
        if logical_z_operator is None:
            logical_z_operator = (
                PauliOperator(pauli=check_type, data_qubits=[(0, 0)])
                if check_type == "Z"
                else PauliOperator(
                    pauli="Z" * d, data_qubits=[(i, 0) for i in range(d)]
                )
            )
        block = cls(
            unique_label=unique_label,
            stabilizers=stabilizers,
            logical_x_operators=[logical_x_operator],
            logical_z_operators=[logical_z_operator],
            skip_validation=skip_validation,
        )
        if position == (0,):
            return block

        return block.shift(position)

    @property
    def check_type(self) -> str:
        """Extract the check type of the repetition code.

        Returns
        -------
        str
            Check type of the repetition code, either "X" or "Z"
        """
        return self.stabilizers[0].pauli[0]

    def boundary_qubits(self, direction: Direction | str) -> tuple[int, int]:
        """Return the data qubits that are part of the specified boundary.

        Parameters
        ----------
        direction : Direction | str
            Boundary (left or right) for which the data qubits should be
            returned. If a string is provided, it is converted to a Direction enum.

        Returns
        -------
        tuple[int, int]
            Data qubit in the specified boundary
        """
        # Input validation: cast direction to Direction enum if it is not already
        if not isinstance(direction, Direction):
            direction = Direction(direction)

        if direction not in [Direction.LEFT, Direction.RIGHT]:
            raise ValueError(
                f"Invalid direction '{direction}'. "
                "Only 'left' and 'right' are supported."
            )

        # NOTE: Even though the type of Block.data_qubits is tuple[int, ...], the output
        # of this function will in practice always be a tuple of two ints, as the
        # repetition code can only be applied to a linear lattice
        selector_function = max if direction is Direction.RIGHT else min

        return selector_function(self.data_qubits, key=lambda x: x[0])  # type: ignore

    def get_shifted_equivalent_logical_operator(
        self,
        new_qubit: tuple[int, int],
    ) -> tuple[PauliOperator, tuple[Stabilizer, ...]]:
        """Get a shifted version of the single qubit logical operator and the
        stabilizers required to perform the shift.

        Parameters
        ----------
        new_qubit : tuple[int, int]
            New qubit where the logical operator should be shifted.

        Returns
        -------
        tuple[PauliOperator, tuple[Stabilizer, ...]]
            Shifted logical operator and the stabilizers required to perform the shift.
        """

        # Check if the new qubit is in the data qubits
        if new_qubit not in self.data_qubits:
            raise ValueError(
                f"New logical position {new_qubit} is not part of the data qubits"
            )

        short_logical = (
            self.logical_x_operators[0]
            if self.check_type == "X"
            else self.logical_z_operators[0]
        )
        short_logical_data_qubit = short_logical.data_qubits[0]

        # Define shifted operator
        if new_qubit != short_logical_data_qubit:
            shifted_operator = PauliOperator(
                pauli=self.check_type, data_qubits=[new_qubit]
            )
        else:
            shifted_operator = short_logical

        # Define required stabilizers
        if new_qubit < short_logical_data_qubit:
            required_stabilizers = tuple(
                stab
                for stab in self.stabilizers
                if all(
                    (q[0] >= new_qubit[0] and q[0] <= short_logical_data_qubit[0])
                    for q in stab.data_qubits
                )
            )
        elif new_qubit > short_logical_data_qubit:
            required_stabilizers = tuple(
                stab
                for stab in self.stabilizers
                if all(
                    (q[0] <= new_qubit[0] and q[0] >= short_logical_data_qubit[0])
                    for q in stab.data_qubits
                )
            )
        else:
            required_stabilizers = ()

        return shifted_operator, required_stabilizers

    # Define equality method for repetition codes
    def __eq__(self, other: RepetitionCode) -> bool:
        if not isinstance(other, RepetitionCode):
            raise NotImplementedError(
                f"Cannot compare RepetitionCode with {type(other)}"
            )
        return super().__eq__(other)

    def shift(
        self, position: tuple[int], new_label: str | None = None
    ) -> RepetitionCode:
        return super().shift(position, new_label)

    def rename(self, name: str) -> RepetitionCode:
        return super().rename(name)

    @cached_property
    def stabilizers_labels(self) -> dict[str, dict[str, tuple[int, ...]]]:
        return super().stabilizers_labels
