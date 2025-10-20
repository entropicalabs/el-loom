"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from __future__ import annotations
import dataclasses

from pydantic import Field, field_validator, ValidationInfo
from pydantic.dataclasses import dataclass

from .block import Block
from .lattice import Lattice
from .operations import Operation
from .utilities.validation_tools import dataclass_params


@dataclass(**dataclass_params)
class Eka:
    # pylint: disable=line-too-long
    """
    This dataclass contains all the information necessary to execute a
    logical circuit with embedded error correction.

    Parameters
    ------
    lattice : Lattice
        Lattice on which the Eka is defined
    blocks : tuple[Block, ...]
        Inital state of the block defining the logical qubits.
    operations : tuple[:class:`loom.eka.operations.base_operation.Operation`, ...] | tuple[tuple[:class:`loom.eka.operations.base_operation.Operation`, ...], ...]
        Operations to be executed on the block. Each tuple of operations is executed
        in parallel. The operations are executed in the order they are given in the
        tuple but their circuits are combined such that they can be executed in
        parallel. If the operations are specified as a tuple of operations, they are
        executed sequentially.
    """

    lattice: Lattice
    blocks: tuple[Block, ...] = Field(default_factory=tuple, validate_default=False)
    operations: tuple[Operation, ...] | tuple[tuple[Operation, ...], ...] = Field(
        default_factory=tuple, validate_default=False
    )
    # hardware: Hardware # Remove comment after implementing Hardware

    @field_validator("blocks", mode="after")
    @classmethod
    def blocks_valid_indices(
        cls, blocks: tuple[Block, ...], info: ValidationInfo
    ) -> tuple[Block, ...]:
        """Check that blocks have valid qubit indices. Every element in the tuple has
        to be larger or equal to 0 and smaller than the size of the lattice in the
        respective dimension (if the lattice has a finite size)."""
        # Check if any data/ancilla qubit has a negative index
        for block in blocks:
            if any(any(coord < 0 for coord in qubit) for qubit in block.data_qubits):
                raise ValueError(
                    f"Block '{block.unique_label}' has negative data qubit indices."
                )
            if any(any(coord < 0 for coord in qubit) for qubit in block.ancilla_qubits):
                raise ValueError(
                    f"Block '{block.unique_label}' has negative ancilla qubit indices."
                )

        # Check if any data qubit has an index larger than the lattice size
        lattice_size = info.data["lattice"].size
        if lattice_size is not None:  # For infinite lattices, skip this check
            # To check that the last element in qubit indices (specifying the basis vector)
            # is valid, add the number of allowed values to `lattice_size`
            lattice_size = list(lattice_size)
            lattice_size.append(len(info.data["lattice"].basis_vectors))
            for block in blocks:
                if any(
                    any(coord >= lattice_size[i] for i, coord in enumerate(qubit))
                    for qubit in block.data_qubits
                ):
                    raise ValueError(
                        f"Block '{block.unique_label}' has data qubit indices which "
                        "are too large for the lattice."
                    )
                if any(
                    any(coord >= lattice_size[i] for i, coord in enumerate(qubit))
                    for qubit in block.ancilla_qubits
                ):
                    raise ValueError(
                        f"Block '{block.unique_label}' has ancilla qubit indices which "
                        "are too large for the lattice."
                    )

        # TODO: Add checks for ancilla qubits # pylint: disable=fixme

        return blocks

    _validate_blocks_no_overlap = field_validator("blocks", mode="after")(
        Block.blocks_no_overlap
    )

    @field_validator("blocks", mode="after")
    @classmethod
    def blocks_unique_labels(cls, blocks: tuple[Block, ...]) -> tuple[Block, ...]:
        """Check that blocks have unique labels."""
        unique_labels = set(block.unique_label for block in blocks)
        if len(unique_labels) != len(blocks):
            raise ValueError("Not all blocks have unique labels.")
        return blocks

    @field_validator("operations", mode="after")
    @classmethod
    def operations_disjoint(
        cls, operations: tuple[Operation] | tuple[tuple[Operation, ...], ...]
    ):
        """Check that operations are disjoint, i.e. no block is used in multiple
        operations. Casts the operations to a tuple of tuples if it is a tuple of
        operations."""
        # Cast into tuple of tuples if operations is a tuple of operations
        if all(isinstance(operation, Operation) for operation in operations):
            return tuple((operation,) for operation in operations)
        # Check that operations in the same timestep are disjoint
        for timestep in operations:
            used_inputs = set()
            for operation in timestep:
                op_inputs = set(getattr(operation, "_inputs"))
                if conflict := used_inputs.intersection(op_inputs):
                    raise ValueError(
                        f"Operations are not disjoint, {conflict} is subject to "
                        f"multiple operations at the same time."
                    )
                used_inputs.update(op_inputs)
        return operations

    @classmethod
    def fromdict(cls, data_dict: dict) -> Eka:
        """Instantiate an Eka object from a dictionary."""

        operation_input = data_dict.pop("operations", None)
        if operation_input is not None:
            data_dict["operations"] = tuple(
                tuple(Operation.fromdict(operation) for operation in timestep)
                for timestep in operation_input
            )
        return cls(**data_dict)

    def asdict(self) -> dict:
        """Create a dictionary representation of the Eka object."""
        # Leverage dataclasses.asdict excluding the 'operations' key
        class_dict = dataclasses.asdict(
            self, dict_factory=lambda d: {k: v for k, v in d if k != "operations"}
        )
        # Use custom asdict method to add the 'operations' key: value
        class_dict["operations"] = tuple(
            tuple(
                Operation.asdict(operation) for operation in timestep  # custom asdict
            )
            for timestep in self.operations
        )
        return class_dict
