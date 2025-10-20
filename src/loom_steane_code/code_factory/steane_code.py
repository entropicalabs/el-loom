"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=duplicate-code
from __future__ import annotations
from functools import cached_property
from uuid import uuid4
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
class SteaneCode(Block):
    """
    A sub-class of ``Block`` that represents a Steane code block.
    """

    @classmethod
    def create(  # pylint: disable=too-many-locals
        cls,
        lattice: Lattice,
        unique_label: str = None,
        position: tuple[int, ...] = (0, 0),
        skip_validation: bool = False,
    ) -> SteaneCode:
        """Create Steane Code Block object. The Steane Code is a
        [[7, 1, 3]] quantum error-correcting CSS code. There are 6 stabilizers, three
        of X type and three of Z type all of support size 4. We implement a naive
        syndrome extraction scheme, where measure all X stabilizers first and all the
        Z stabilizers later. We choose to embed the code in a 2D square lattice, where
        the logical operators are defined on the three lowest data qubits.

        Parameters
        ----------
        lattice : Lattice
            Lattice on which the block is defined. The qubit indices depend on the type
            of lattice.
        unique_label : str, optional
            Label for the block. It must be unique among all blocks in the initial CRD.
            If no label is provided, a unique label is generated automatically using the
            uuid module.
        position : tuple[int, ...], optional
            Position of the top left corner of the block on the lattice,
            by default (0, 0).
        skip_validation : bool, optional
            Skip validation of the block object, by default False.

        Returns
        -------
        Block
            Block object for the Steane Code.
        """

        # Input validation
        if lattice.lattice_type != LatticeType.SQUARE_2D:
            raise ValueError(
                "The creation of Steane code blocks is "
                "currently only supported for 2D square lattices. Instead "
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
        stabilizer_supports = [
            [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
            [(1, 0, 0), (1, 1, 0), (2, 1, 0), (2, 0, 0)],
            [(2, 1, 0), (1, 2, 0), (0, 1, 0), (1, 1, 0)],
        ]

        x_stabilizers = [
            Stabilizer(pauli="X" * 4, data_qubits=supp, ancilla_qubits=[(i, 0, 1)])
            for i, supp in enumerate(stabilizer_supports)
        ]
        z_stabilizers = [
            Stabilizer(pauli="Z" * 4, data_qubits=supp, ancilla_qubits=[(0, i + 1, 1)])
            for i, supp in enumerate(stabilizer_supports)
        ]

        stabilizers = x_stabilizers + z_stabilizers

        # Define the logical operators
        logical_data_qubits = [(i, 0, 0) for i in range(3)]

        logical_x_operator = PauliOperator(
            pauli="XXX",
            data_qubits=logical_data_qubits,
        )
        logical_z_operator = PauliOperator(
            pauli="ZZZ",
            data_qubits=logical_data_qubits,
        )

        # Define the syndrome extraction circuits
        x_syndrome_circuits = cls.generate_syndrome_extraction_circuits("XXXX")
        z_syndrome_circuits = cls.generate_syndrome_extraction_circuits("ZZZZ")
        syndrome_circuits = [x_syndrome_circuits, z_syndrome_circuits]

        # Define the stabilizer to circuit mapping
        stabilizer_to_circuit = {
            stab.uuid: x_syndrome_circuits.uuid for stab in x_stabilizers
        } | {stab.uuid: z_syndrome_circuits.uuid for stab in z_stabilizers}

        block = cls(
            unique_label=unique_label,
            stabilizers=stabilizers,
            logical_x_operators=[logical_x_operator],
            logical_z_operators=[logical_z_operator],
            syndrome_circuits=syndrome_circuits,
            stabilizer_to_circuit=stabilizer_to_circuit,
            skip_validation=skip_validation,
        )

        if position == (0, 0):
            return block

        return block.shift(position)

    @staticmethod
    def generate_syndrome_extraction_circuits(pauli: str) -> SyndromeCircuit:
        """
        Generate syndrome extraction circuit for a stabilizers from the Steane Code.
        The prescription followed here is to first measure all stabilizers of a given
        type simultaneously in four layers (arbitrarily chosen to be entangling in the
        data qubits with the ancilla in a clockwise order). We choose X stabilizers to
        be measured first, thus we need to add four idling steps at the end, and four
        at the beginning for the Z stabilizers. The ancilla is then measured and reset.

        Parameters
        ----------
        pauli: str
            Pauli operator for which the syndrome extraction circuit is generated.

        Returns
        -------
        SyndromeCircuit
            Syndrome extraction circuit for the given Pauli operator.
        """

        # Extract parameters
        name = f"{pauli}_syndrome_extraction"
        pauli_type = pauli[0]

        # Define channels
        data_channels = [Channel(type="quantum", label=f"d{i}") for i in range(4)]
        cbit_channel = Channel(type="classical", label="c0")
        ancilla_channel = Channel(type="quantum", label="a0")

        # Define Hadamard gates
        hadamard1 = tuple([Circuit("H", channels=[ancilla_channel])])
        hadamard2 = tuple([Circuit("H", channels=[ancilla_channel])])

        # Entangling layer
        entangle_ancilla = [
            [Circuit(f"C{p}", channels=[ancilla_channel, data_channels[i]])]
            for i, p in enumerate(pauli)
        ]

        # Add idling step
        if pauli_type == "Z":
            entangle_ancilla = [(), (), (), ()] + entangle_ancilla
        else:
            entangle_ancilla += [(), (), (), ()]

        # Add ancilla measurement and reset
        measurement = tuple(
            [Circuit("Measurement", channels=[ancilla_channel, cbit_channel])]
        )
        reset = tuple([Circuit("Reset_0", channels=[ancilla_channel])])

        # Compose circuit operations as a list
        circuit_list = [reset, hadamard1] + entangle_ancilla + [hadamard2, measurement]

        # Return the syndrome extraction circuit
        return SyndromeCircuit(
            pauli=pauli,
            name=name,
            circuit=Circuit(
                name=name,
                circuit=circuit_list,
                channels=data_channels + [ancilla_channel, cbit_channel],
            ),
        )

    # Instance methods
    def __eq__(self, other: SteaneCode) -> bool:
        if not isinstance(other, SteaneCode):
            raise NotImplementedError(f"Cannot compare SteaneCode with {type(other)}")
        return super().__eq__(other)

    def shift(
        self, position: tuple[int, ...], new_label: str | None = None
    ) -> SteaneCode:
        return super().shift(position, new_label)

    def rename(self, name: str) -> SteaneCode:
        return super().rename(name)

    @cached_property
    def stabilizers_labels(self) -> dict[str, dict[str, tuple[int, ...]]]:
        return super().stabilizers_labels
