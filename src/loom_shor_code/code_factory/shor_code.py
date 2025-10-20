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
    SyndromeCircuit,
    Circuit,
    Channel,
)
from loom.eka.utilities import dataclass_params


@dataclass(**dataclass_params)
class ShorCode(Block):
    """
    A sub-class of ``Block`` that represents a Shor code block.
    """

    @classmethod
    def create(
        cls,
        lattice: Lattice,
        unique_label: str | None = None,
        position: tuple[int, ...] = (0,),
        skip_validation: bool = False,
    ) -> ShorCode:
        """Create datablock of Shor Code as a Block object. The Shor Code is a
        [[9, 1, 3]] quantum error-correcting CSS code, resulting from the concatenation
        of two repetition codes. There are two X stabilizers of weight 6, and six Z
        stabilizers of weight 2. We implement a naive syndrome extraction scheme, where
        measure all X stabilizers first and all the Z stabilizers later. We choose to
        embed the code in a linear lattice, where the logical operators are defined on
        all the data qubits.

        Parameters
        ----------
        lattice : Lattice
            Lattice on which the block is defined. The qubit indices depend on the type
            of lattice.
        unique_label : str, optional
            Label for the block. It must be unique among all blocks in the initial CRD.
            If no label is provided, a unique label is generated automatically using the
            uuid module.
        position : tuple[int], optional
            Position of the top left boundary of the block on the lattice,
            by default (0,).
        skip_validation : bool, optional
            Skip validation of the block object, by default False.

        Returns
        -------
        Block
            Block object for the Shor Code.
        """

        # Input validation
        if lattice.lattice_type != LatticeType.LINEAR:
            raise ValueError(
                "The creation of Shor code blocks is "
                "currently only supported for 1D linear lattices. Instead "
                f"the lattice is of type {lattice.lattice_type}."
            )

        if not isinstance(position, tuple) or any(
            not isinstance(x, int) for x in position
        ):
            raise ValueError(
                f"`position` must be a tuple of integers. Got '{position}' instead."
            )

        if unique_label is None:
            unique_label = str(uuid4())

        # Define the stabilizers
        x_indices = [[(i + 3 * j, 0) for i in range(6)] for j in range(2)]
        x_stabilizers = [
            Stabilizer(
                pauli="X" * 6,
                data_qubits=x_ind,
                ancilla_qubits=[(i, 1)],
            )
            for i, x_ind in enumerate(x_indices)
        ]

        z_indices = [
            [(i + j + 3 * k, 0) for i in range(2)] for k in range(3) for j in range(2)
        ]
        z_stabilizers = [
            Stabilizer(
                pauli="Z" * 2,
                data_qubits=z_ind,
                ancilla_qubits=[(i + 2, 1)],
            )
            for i, z_ind in enumerate(z_indices)
        ]

        stabilizers = x_stabilizers + z_stabilizers

        # Define the syndrome circuits
        x_syndrome_circuits = cls.generate_syndrome_extraction_circuits("X" * 6)
        z_syndrome_circuits = cls.generate_syndrome_extraction_circuits("Z" * 2)
        syndrome_circuits = [x_syndrome_circuits, z_syndrome_circuits]

        # Define mapping of stabilizers to syndrome circuits
        stabilizer_to_circuit = {
            stab.uuid: x_syndrome_circuits.uuid for stab in x_stabilizers
        } | {stab.uuid: z_syndrome_circuits.uuid for stab in z_stabilizers}

        # Define the logical operators
        all_datas = [(i, 0) for i in range(9)]
        logical_x_operator = PauliOperator(pauli="X" * 9, data_qubits=all_datas)
        logical_z_operator = PauliOperator(pauli="Z" * 9, data_qubits=all_datas)

        block = cls(
            unique_label=unique_label,
            stabilizers=stabilizers,
            logical_x_operators=[logical_x_operator],
            logical_z_operators=[logical_z_operator],
            syndrome_circuits=syndrome_circuits,
            stabilizer_to_circuit=stabilizer_to_circuit,
            skip_validation=skip_validation,
        )

        if position == (0,):
            return block

        return block.shift(position)

    # Syndrome extraction circuits
    @staticmethod
    def generate_syndrome_extraction_circuits(pauli: str) -> SyndromeCircuit:
        """
        Construct a syndrome extraction circuit for the given Pauli operator. We chose
        the convention of measuring all X stabilizers first and then all Z, thus,
        appropriate idling steps are added depending on the type of Pauli operator.

        Parameters
        ----------
        pauli : str
            The Pauli operator as a string.

        Returns
        -------
        syndrome_circuit : SyndromeCircuit
            The syndrome extraction circuit for the given Pauli operator.
        """

        # Extract pauli information
        pauli_type = pauli[0]
        supp = len(pauli)
        circ_name = f"{pauli}_syndrome_extraction"

        # Define channels
        data_channels = [Channel(type="quantum", label=f"d{i}") for i in range(supp)]
        cbit_channel = Channel(type="classical", label="c0")
        ancilla_channel = Channel(type="quantum", label="a0")

        # Add ancilla reset
        reset = [Circuit("Reset_0", channels=[ancilla_channel])]

        # Hadamard layers
        h_initial = [Circuit("H", channels=[ancilla_channel])]
        h_final = [Circuit("H", channels=[ancilla_channel])]

        # Add entangling layer
        entangling_layer = [
            [Circuit(f"C{p}", channels=[ancilla_channel, data_channels[i]])]
            for i, p in enumerate(pauli)
        ]

        ## Add idling steps
        if pauli_type == "X":
            entangling_layer += [()] * 2
        else:
            entangling_layer = [()] * 6 + entangling_layer

        # Add ancilla measurement
        measurement = [Circuit("Measurement", channels=[ancilla_channel, cbit_channel])]

        circuit_list = [reset, h_initial] + entangling_layer + [h_final, measurement]

        syndrome_circuit = SyndromeCircuit(
            pauli=pauli,
            name=circ_name,
            circuit=Circuit(
                name=circ_name,
                circuit=circuit_list,
                channels=data_channels + [ancilla_channel, cbit_channel],
            ),
        )

        return syndrome_circuit

    # Instance methods
    def __eq__(self, other: ShorCode) -> bool:
        if not isinstance(other, ShorCode):
            raise NotImplementedError(f"Cannot compare ShorCode with {type(other)}")
        return super().__eq__(other)

    def shift(
        self, position: tuple[int, ...], new_label: str | None = None
    ) -> ShorCode:
        return super().shift(position, new_label)

    def rename(self, name: str) -> ShorCode:
        return super().rename(name)

    @cached_property
    def stabilizers_labels(self) -> dict[str, dict[str, tuple[int, ...]]]:
        return super().stabilizers_labels
