"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=too-many-locals

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
class FiveQubitPerfectCode(Block):
    """
    Represents the five qubit perfect code, also known as the
    Laflamme code, as a subclass of ``Block``.
    """

    @classmethod
    def create(
        cls,
        lattice: Lattice,
        unique_label: str | None = None,
        position: tuple[int, int] = (0, 0),
        skip_validation: bool = False,
    ) -> FiveQubitPerfectCode:
        """Create a `Block` object for a five qubit perfect code/Laflamme code block.
        The five qubit code is a [[5, 1, 3]] quantum error-correcting code that can
        correct arbitrary single-qubit errors on any one qubit, and detect errors on any
        two qubits. There are 4 stabilizers of support 4 that are symmetric under cyclic
        permutation. The block is defined on a pentagonal lattice.

        NOTE: There are two stabilizers labelled with "XZZX" but with different data
        qubits. The two correspond to XZZXI and IXZZX stabilizers, but as the stabilizer
        class does not accept "I" as an input, these two become labelled identically.

        The circuit implementation is not fault tolerant, and is based on the stabilizer
        circuit from https://en.wikipedia.org/wiki/Five-qubit_error_correcting_code


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
            Block object for the five qubit perfect code
        """

        # Input validation
        if lattice.lattice_type != LatticeType.POLY_2D:
            raise ValueError(
                "The creation of five qubit perfect code blocks is "
                "currently only supported for 2D pentagonal lattices. Instead "
                f"the lattice is of type {lattice.lattice_type}."
            )
        if lattice.unit_cell_size != 9:
            raise ValueError(
                "The five qubit perfect code block requires a lattice with exactly 5 "
                f"qubits and 4 ancilla in each unit cell. Got {lattice.unit_cell_size} "
                "qubits and ancilla instead."
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
        looped_stabilizer_supports = [(0, 0, i % 5) for i in range(8)]
        stab_labels = ["XZZX", "XZZX", "XXZZ", "ZXXZ"]
        stabilizers = [
            Stabilizer(
                pauli=stab_labels[i],
                data_qubits=sorted(looped_stabilizer_supports[i : i + 4]),
                ancilla_qubits=[(0, 0, 5 + i)],
            )
            for i in range(4)
        ]

        # Define the logical operators
        all_data_qubits = [(0, 0, i) for i in range(5)]

        logical_x_operator = PauliOperator(
            pauli="X" * 5,
            data_qubits=all_data_qubits,
        )
        logical_z_operator = PauliOperator(
            pauli="Z" * 5,
            data_qubits=all_data_qubits,
        )

        # Define the syndrome extraction circuits
        syndrome_circuits = [
            cls.generate_syndrome_extraction_circuits("XZZXI"),
            cls.generate_syndrome_extraction_circuits("IXZZX"),
            cls.generate_syndrome_extraction_circuits("XIXZZ"),
            cls.generate_syndrome_extraction_circuits("ZXIXZ"),
        ]

        # Define the stabilizer to syndrome mapping
        stabilizer_to_circuit = {
            stab.uuid: syndrome_circuits[i].uuid for i, stab in enumerate(stabilizers)
        }

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
        """Generate syndrome extraction circuits for the five qubit perfect code.

        Parameters
        ----------
        pauli : str
            The Pauli operator for which to generate the syndrome extraction circuits.

        Returns
        -------
        SyndromeCircuit
            A SyndromeCircuit object containing the circuit for syndrome extraction.
        """

        # Stabilizer info
        p_index = pauli.index("I")
        start_t_index = (p_index + 1) % 5
        name = f"{pauli}_syndrome_extraction"

        # Define Channels
        data_channels = [Channel(type="quantum", label=f"d{i}") for i in range(4)]
        anc_channel = [Channel(type="quantum", label="a0")]
        cbit_channel = [Channel(type="classical", label="c0")]

        # Define the circuit
        reset_layer = [Circuit(name="Reset_0", channels=anc_channel)]

        hadamard_initial = [Circuit(name="H", channels=anc_channel)]
        hadamard_final = [Circuit(name="H", channels=anc_channel)]

        measurement_layer = [
            Circuit(name="Measurement", channels=anc_channel + cbit_channel)
        ]

        insert_entangling_layer = []
        i = 0  # len(pauli) == 5, so we track index for data channels
        for p in pauli:
            if p == "I":
                insert_entangling_layer.append([])
            else:
                insert_entangling_layer.append(
                    [Circuit(name=f"C{p}", channels=anc_channel + [data_channels[i]])]
                )
                i += 1

        entangling_layer = [[], [], []]
        entangling_layer = (
            entangling_layer[:start_t_index]
            + insert_entangling_layer
            + entangling_layer[start_t_index:]
        )

        circuit_list = (
            [reset_layer, hadamard_initial]
            + entangling_layer
            + [hadamard_final, measurement_layer]
        )

        # Return the syndrome extraction circuit
        return SyndromeCircuit(
            pauli=pauli[:p_index] + pauli[p_index + 1 :],
            name=name,
            circuit=Circuit(
                name=name,
                circuit=circuit_list,
                channels=data_channels + anc_channel + cbit_channel,
            ),
        )

    # Instance methods
    def __eq__(self, other: FiveQubitPerfectCode) -> bool:
        if not isinstance(other, FiveQubitPerfectCode):
            raise NotImplementedError(
                f"Cannot compare FiveQubitPerfectCode with {type(other)}"
            )
        return super().__eq__(other)

    def shift(
        self, position: tuple[int, ...], new_label: str | None = None
    ) -> FiveQubitPerfectCode:
        return super().shift(position, new_label)

    def rename(self, name: str) -> FiveQubitPerfectCode:
        return super().rename(name)

    @cached_property
    def stabilizers_labels(self) -> dict[str, dict[str, tuple[int, ...]]]:
        return super().stabilizers_labels
