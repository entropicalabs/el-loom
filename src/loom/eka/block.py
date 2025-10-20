"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=too-many-lines

from __future__ import annotations
from itertools import combinations, product, chain
from functools import cached_property
from uuid import uuid4

import numpy as np

from pydantic.dataclasses import Field, dataclass
from pydantic import field_validator, model_validator, ValidationInfo
from pydantic_core import ArgsKwargs

from .stabilizer import Stabilizer
from .pauli_operator import PauliOperator
from .syndrome_circuit import SyndromeCircuit
from .utilities.pauli_format_conversion import paulixz_to_char
from .utilities.stab_array import (
    merge_stabarrays,
    reduce_stabarray_with_bookkeeping,
    find_destabarray,
    StabArray,
    invert_bookkeeping_matrix,
)
from .utilities.validation_tools import (
    dataclass_params,
    ensure_tuple,
    empty_list_error,
    retrieve_field,
)


@dataclass(**dataclass_params)
class Block:  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """
    Block describing one or multiple logical qubits. It is defined by a code type, a
    list of stabilizers, two logical operators per logical qubit and a unique label. The
    block also contains information about how its syndromes are measured.

    Parameters
    ----------
    stabilizers : tuple[Stabilizer, ...]
        The stabilizers that define the block.
    logical_x_operators : tuple[PauliOperator, ...]
        The logical X operators associated to the block.
    logical_z_operators : tuple[PauliOperator, ...]
        The logical Z operators associated to the block.
    syndrome_circuits : tuple[SyndromeCircuit, ...]
        The syndrome circuits with which the stabilizers of this block are measured.
        NOTE: Block does not have a check for distinct Syndrome Circuits(s)
    stabilizer_to_circuit : dict[str, str]
        A dictionary mapping stabilizer uuids to the uuids of the syndrome circuits that
        measure them.
    unique_label : str, optional
        Label for the block. It must be unique among all blocks in the initial Eka.
        If no label is provided, a unique label is generated automatically using the
        uuid module.
    skip_validation : bool, optional
        Boolean that allows to skip some validation of the Block. Default is False.
        The validation skipped are the following:

        - The qubits coordinates have the same dimension.
        - The stabilizers commute with each other.
        - The logical operators commute with each other.
        - The stabilizers commute with the logical operators.
        - The logical X and Z operators anti-commute at the same index and commute at \
        different indices.
        - The number of qubits and stabilizers in the Block is compatible with the \
        number of logical qubits.
    uuid : str, optional
        Unique identifier for the block. If no uuid is provided, a unique uuid is
        generated automatically using the uuid module.

    Attributes
    ----------
    __version__ : str
        Version of the Block class implementation.
    """

    stabilizers: tuple[Stabilizer, ...]
    logical_x_operators: tuple[PauliOperator, ...]
    logical_z_operators: tuple[PauliOperator, ...]
    syndrome_circuits: tuple[SyndromeCircuit, ...] = Field(
        default_factory=tuple, validate_default=True
    )
    stabilizer_to_circuit: dict[str, str] = Field(
        default_factory=dict, validate_default=True
    )
    unique_label: str = Field(default_factory=lambda: str(uuid4()))
    skip_validation: bool = Field(default=False, validate_default=True)
    uuid: str = Field(default_factory=lambda: str(uuid4()), validate_default=True)

    # version of the Block class implementation
    __version__: str = "1.0.0"

    # Model validators with mode="before".
    # Note that these are executed in the reverse order in which they are defined.
    # This is where the last "before" validator is executed
    @model_validator(mode="before")
    @classmethod
    def _validate_qubits_included(cls, data: dict):
        """
        Check that all qubits used in the logical operators are included in the
        set of stabilizers.

        Parameters
        ----------
        data : dict
            The data to be validated.
        """
        logical_operator_qubits = set(
            qubit
            for operator in data["logical_x_operators"] + data["logical_z_operators"]
            for qubit in operator.data_qubits
        )
        stabilizers_qubits = set(
            qubit
            for stabilizer in data["stabilizers"]
            for qubit in stabilizer.data_qubits
        )
        qubits_not_in_stabilizers = logical_operator_qubits - stabilizers_qubits
        if qubits_not_in_stabilizers:
            raise ValueError(
                f"Qubits {qubits_not_in_stabilizers} are not included in the"
                f" stabilizers but are used in the logical operators"
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def _validate_number_of_logical_operators(cls, data: dict):
        """
        Check that the number of logical X operators is equal to the number of logical Z
        """
        if len(data["logical_x_operators"]) != len(data["logical_z_operators"]):
            raise ValueError(
                "The number of logical X operators must be equal to the number of "
                "logical Z operators."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def _assign_types(cls, data: ArgsKwargs):
        """
        Assign the types of the stabilizers and logical operators and casts
        ArgsKwargs into a dictionary. This is necessary to ensure that the data is
        loaded in the correct format when the Block is created from a JSON. It also
        tests for empty inputs for stabilizers, logical_x_operators and
        logical_z_operators.
        """
        # Cast ArgsKwargs into a dictionary
        if not isinstance(data, dict):
            data = {**dict(data.kwargs), **dict(data.args)}
        # Check empty arguments
        list(
            map(
                empty_list_error,
                [
                    data["stabilizers"],
                    data["logical_x_operators"],
                    data["logical_z_operators"],
                ],
            )
        )
        # Assign types explicitly
        if isinstance(data["stabilizers"][0], dict):
            data["stabilizers"] = [Stabilizer(**stab) for stab in data["stabilizers"]]
        if isinstance(data["logical_x_operators"][0], dict):
            data["logical_x_operators"] = [
                PauliOperator(**{k: v for k, v in op.items() if k != "nr_of_ancillae"})
                for op in data["logical_x_operators"]
            ]
        if isinstance(data["logical_z_operators"][0], dict):
            data["logical_z_operators"] = [
                PauliOperator(**{k: v for k, v in op.items() if k != "nr_of_ancillae"})
                for op in data["logical_z_operators"]
            ]
        return data

    @model_validator(mode="before")
    @classmethod
    def _validate_workbench_json_version(cls, data):
        """
        Checks that the major version of the incoming Workbench JSON matches
        the major version of the Block class.
        """
        if hasattr(data, "kwargs") and data.kwargs is not None:
            if "__version__" not in data.kwargs:
                return data
            if data.kwargs["__version__"][0] != cls.__version__[0]:
                raise ValueError(
                    """
                    The major version of the Workbench export is not 
                    compatible with the major version of Eka.Block.
                    """
                )
        return data

    # Field validators are executed after model_validator with mode="before"
    @field_validator("stabilizers", mode="before")
    @classmethod
    def _validate_distinct_stabilizers(cls, stabilizers: tuple[Stabilizer, ...]):
        """
        Check that stabilizers are distinct.
        """
        for stab1, stab2 in combinations(stabilizers, 2):
            if stab1 == stab2:
                raise ValueError("Stabilizers must be distinct.")
        return stabilizers

    _validate_stabilizers_list = field_validator("stabilizers", mode="before")(
        ensure_tuple
    )

    @field_validator("logical_x_operators", mode="before")
    @classmethod
    def _validate_distinct_logical_x_operators(
        cls, logical_x_operators: tuple[PauliOperator, ...]
    ):
        """
        Check that logical X operators are distinct.
        """
        for log1, log2 in combinations(logical_x_operators, 2):
            if log1 == log2:
                raise ValueError("Logical X operators must be distinct.")
        return logical_x_operators

    _validate_logical_x_list = field_validator("logical_x_operators", mode="before")(
        ensure_tuple
    )

    @field_validator("logical_z_operators", mode="before")
    @classmethod
    def _validate_distinct_logical_z_operators(
        cls, logical_z_operators: tuple[PauliOperator, ...]
    ):
        """
        Check that logical Z operators are distinct.
        """
        for log1, log2 in combinations(logical_z_operators, 2):
            if log1 == log2:
                raise ValueError("Logical Z operators must be distinct.")
        return logical_z_operators

    _validate_logical_z_list = field_validator("logical_z_operators", mode="before")(
        ensure_tuple
    )

    @field_validator("syndrome_circuits", mode="after")
    @classmethod
    def _create_default_syndrome_circuits(
        cls, syndrome_circuits: tuple[SyndromeCircuit, ...], info: ValidationInfo
    ) -> dict[str, str]:
        """
        Check that for every pauli string of the stabilizers there is a corresponding
        syndrome circuit. If not, create a default syndrome circuit for that pauli
        string.
        """
        stab_paulis = {stab.pauli for stab in retrieve_field("stabilizers", info)}
        synd_circ_paulis = {synd_circ.pauli for synd_circ in syndrome_circuits}
        additional_circuits = [
            SyndromeCircuit(
                name=f"default_{pauli}",
                pauli=pauli,
            )
            for pauli in stab_paulis - synd_circ_paulis
        ]
        return tuple(list(syndrome_circuits) + additional_circuits)

    @field_validator("stabilizer_to_circuit", mode="after")
    @classmethod
    def _check_stabilizer_to_circuit_map_uuids(
        cls, stabilizer_to_circuit: dict[str, str], info: ValidationInfo
    ) -> dict[str, str]:
        """
        Check that all uuids in the stabilizer to circuit map are valid.
        I.e. check that all stabilizer uuids actually appear in the set of stabilizers
        and that all syndrome circuit uuids actually appear in the list of syndrome
        circuits. Also check that pauli strings match.
        """
        syndrome_circs_dict = {
            circ.uuid: circ for circ in retrieve_field("syndrome_circuits", info)
        }
        stabilizer_dict = {
            stab.uuid: stab for stab in retrieve_field("stabilizers", info)
        }
        for stab_uuid, syndrome_circ_uuid in stabilizer_to_circuit.items():
            if stab_uuid not in stabilizer_dict.keys():
                raise ValueError(
                    f"Stabilizer with uuid {stab_uuid} is not present in the stabilizers."
                )
            if syndrome_circ_uuid not in syndrome_circs_dict.keys():
                raise ValueError(
                    f"Syndrome circuit with uuid {syndrome_circ_uuid} is not present in "
                    f"the syndrome circuits."
                )
            if (
                stabilizer_dict[stab_uuid].pauli
                != syndrome_circs_dict[syndrome_circ_uuid].pauli
            ):
                raise ValueError(
                    f"Stabilizer with uuid {stab_uuid} has a different pauli string "
                    f"{stabilizer_dict[stab_uuid].pauli} than the syndrome circuit "
                    f"with uuid {syndrome_circ_uuid} (pauli string: "
                    f"{syndrome_circs_dict[syndrome_circ_uuid].pauli})."
                )
        return stabilizer_to_circuit

    @field_validator("stabilizer_to_circuit", mode="after")
    @classmethod
    def _associate_stabs_to_syndrome_circuits(
        cls, stabilizer_to_circuit: dict[str, str], info: ValidationInfo
    ) -> dict[str, str]:
        """
        For every stabilizer which is not yet associated to a syndrome circuit,
        associate it with a default syndrome circuit. If there are multiple syndrome
        circuits for the same pauli string, raise an exception.
        """
        synd_circs = retrieve_field("syndrome_circuits", info)
        for stab in retrieve_field("stabilizers", info):
            if stab.uuid not in stabilizer_to_circuit.keys():
                matching_circuits = [
                    circ for circ in synd_circs if circ.pauli == stab.pauli
                ]
                if len(matching_circuits) > 1:
                    raise ValueError(
                        "Multiple syndrome circuits for the same stabilizer pauli "
                        f"string {stab.pauli} found. Could not automatically associate "
                        f"stabilizer {stab} with a syndrome circuit. Please do the "
                        "association manually."
                    )
                stabilizer_to_circuit[stab.uuid] = matching_circuits[0].uuid
        return stabilizer_to_circuit

    # Model validators, mode="after"
    # NOTE that the order of the validators is important
    @model_validator(mode="after")
    def _validate_coordinate_dimension(self):
        """
        Check that all qubits coordinates have the same dimension.
        """
        # Bypass validation
        if self.skip_validation:
            return self

        dimension = len(self.qubits[0])
        if any(len(coordinate) != dimension for coordinate in self.qubits):
            raise ValueError("All qubits coordinates must have the same dimension.")
        return self

    @model_validator(mode="after")
    def _validate_commutation_stabilizers(self):
        """
        Check that stabilizers commute with each other.
        """
        # Bypass validation
        if self.skip_validation:
            return self

        for stab1, stab2 in combinations(self.stabilizers, 2):
            if not stab1.commutes_with(stab2):
                raise ValueError(
                    f"Stabilizers must commute with each other:\n"
                    f"{stab1} and\n"
                    f"{stab2} do not commute.\n"
                )
        return self

    @model_validator(mode="after")
    def _validate_commutation_logical_operators(self):
        """
        Check that logical operators commute with each other.
        """
        # Bypass validation
        if self.skip_validation:
            return self

        for log1, log2 in combinations(self.logical_x_operators, 2):
            if not log1.commutes_with(log2):
                raise ValueError(
                    f"Logical X operators must commute with each other:\n"
                    f"{log1} and\n"
                    f"{log2} do not commute.\n"
                )
        for log1, log2 in combinations(self.logical_z_operators, 2):
            if not log1.commutes_with(log2):
                raise ValueError(
                    f"Logical Z operators must commute with each other:\n"
                    f"{log1} and\n"
                    f"{log2} do not commute.\n"
                )
        return self

    @model_validator(mode="after")
    def _validate_commutation_stabilizers_logical_operators(self):
        """
        Check that stabilizers commute with logical operators.
        """
        # Bypass validation
        if self.skip_validation:
            return self

        for stab, log_x in product(self.stabilizers, self.logical_x_operators):
            if not stab.commutes_with(log_x):
                raise ValueError(
                    f"Stabilizers must commute with logical X operators:\n"
                    f"{repr(stab)} and\n"
                    f"{repr(log_x)} do not commute.\n"
                )
        for stab, log_z in product(self.stabilizers, self.logical_z_operators):
            if not stab.commutes_with(log_z):
                raise ValueError(
                    f"Stabilizers must commute with logical Z operators:\n"
                    f"{repr(stab)} and\n"
                    f"{repr(log_z)} do not commute.\n"
                )
        return self

    @model_validator(mode="after")
    def _validate_anticommutation_logical_operators_one_to_one(self):
        """
        Check that the logical X and Z operators anti-commute at the same index
        anticommute and commute at different indices.
        """
        # Bypass validation
        if self.skip_validation:
            return self

        for (i, log_x), (j, log_z) in combinations(
            chain(
                enumerate(self.logical_x_operators), enumerate(self.logical_z_operators)
            ),
            2,
        ):
            if i == j and log_x.commutes_with(log_z):
                raise ValueError(
                    f"Logical X and Z operators at the same index must anticommute "
                    f"with each other:\n{log_x} and\n{log_z} do not anticommute.\n"
                )
            if i != j and not log_x.commutes_with(log_z):
                raise ValueError(
                    f"Logical X and Z operators at different indices must commute with "
                    f"each other:\n{log_x} and\n {log_z} do not commute.\n"
                )
        return self

    # This is the last "after" validator
    @model_validator(mode="after")
    def _validate_dimensional_compatibility(self):
        """
        Check that the number of qubits and stabilizers in the Block is compatible
        with the number of logical qubits.

        E.g. if L is the number of logical operators, N the number of data qubits and k
        the number of independent stabilizers, then L = N - k.
        """
        # Bypass validation
        if self.skip_validation:
            return self

        if self.reduced_stabarray.nstabs + len(self.logical_x_operators) != len(
            self.data_qubits
        ):
            raise ValueError(
                "The number of qubits and independent stabilizers in the Block is "
                "not compatible with the number of logical qubits."
            )
        return self

    # Properties
    @property
    def data_qubits(self) -> tuple[tuple[int, ...], ...]:
        """
        Return the set of all data qubits in the block.

        Returns
        -------
        tuple[tuple[int, ...], ...] :
            A tuple of coordinates representing the data qubits.
        """
        return tuple(
            sorted(
                # sort the qubits so that the order is not dependent on the
                # order of the stabilizers
                set(
                    data_qubit
                    for stabilizer in self.stabilizers
                    for data_qubit in stabilizer.data_qubits
                )
            )
        )

    @property
    def ancilla_qubits(self) -> tuple[tuple[int, ...], ...]:
        """
        Return the set of all ancilla qubits in the block.

        Returns
        -------
        tuple[tuple[int, ...], ...] :
            A tuple of coordinates representing the ancilla qubits.
        """
        return tuple(
            set(
                ancilla_qubit
                for stabilizer in self.stabilizers
                for ancilla_qubit in stabilizer.ancilla_qubits
            )
        )

    @property
    def qubits(self) -> tuple[tuple[int, ...], ...]:
        """
        Return the set of all qubits in the block.

        Returns
        -------
        tuple[tuple[int, ...], ...] :
            A tuple of coordinates representing all qubits in the block.
        """
        return tuple(set(self.data_qubits + self.ancilla_qubits))

    @property
    def n_data_qubits(self) -> int:
        """
        Return the number of data qubits in the block.
        """
        return len(self.data_qubits)

    @property
    def n_logical_qubits(self) -> int:
        """
        Return the number of logical qubits in the block.
        """
        return len(self.logical_x_operators)

    @property
    def n_irreducible_stabs(self) -> int:
        """
        Return the number of irreducible stabilizers in the embedding, i.e. the
        minimum number of stabilizers needed to generate the code space.
        """
        return self.reduced_stabarray.nstabs

    @cached_property
    def original_stabarray(self) -> StabArray:
        """
        Return the stabilizer array of the block as defined by the stabilizers.
        """
        signed_pauli_ops = [
            s.as_signed_pauli_op(self.data_qubits) for s in self.stabilizers
        ]
        return StabArray.from_signed_pauli_ops(signed_pauli_ops, validated=False)

    @cached_property
    def reduced_stabarray_with_bookkeeping(self) -> tuple[StabArray, np.ndarray]:
        """
        Return the reduced stabilizer array of the embedding with bookkeeping.
        """
        return reduce_stabarray_with_bookkeeping(self.original_stabarray)

    @property
    def reduced_stabarray(self) -> StabArray:
        """
        Return the reduced stabilizer array of the embedding.
        """
        return self.reduced_stabarray_with_bookkeeping[0]

    @property
    def bookkeeping(self) -> np.ndarray:
        """
        Return the bookkeeping array of the reduced stabilizer array.
        """
        return self.reduced_stabarray_with_bookkeeping[1]

    @property
    def reduced_bookkeeping(self) -> np.ndarray:
        """
        Return the reduced bookkeeping array of the reduced stabilizer array. This
        entails slicing out the last rows of the bookkeeping array that should cancel
        out and give a zero row.
        """
        return self.bookkeeping[: self.n_irreducible_stabs, :]

    @cached_property
    def bookkeeping_inv(self) -> np.ndarray:
        """
        Return the inverted bookkeeping array of the reduced stabilizer array.
        """
        return invert_bookkeeping_matrix(self.reduced_stabarray_with_bookkeeping[1])

    @property
    def reduced_bookkeeping_inv(self) -> np.ndarray:
        """
        Return the inverted reduced bookkeeping array of the reduced stabilizer
        array. This entails slicing out the last trivial columns of the inverted
        bookkeeping array that correspond to the zero rows of the reduced StabArray.
        """
        return self.bookkeeping_inv[:, : self.n_irreducible_stabs]

    @cached_property
    def x_log_stabarray(self) -> StabArray:
        """
        Return the X stabilizer array of the logical operator set.
        """
        signed_pauli_ops = [
            x.as_signed_pauli_op(self.data_qubits) for x in self.logical_x_operators
        ]
        return StabArray.from_signed_pauli_ops(signed_pauli_ops, validated=False)

    @cached_property
    def z_log_stabarray(self) -> StabArray:
        """
        Return the Z stabilizer array of the logical operator set.
        """
        signed_pauli_ops = [
            z.as_signed_pauli_op(self.data_qubits) for z in self.logical_z_operators
        ]
        return StabArray.from_signed_pauli_ops(signed_pauli_ops, validated=False)

    @cached_property
    def destabarray(self) -> StabArray:
        """
        Return the destabilizer array of the block. The operators in the
        destabarray anti-commute with exactly one stabilizer of the block each. In
        particular, the one with a matching index. They also commute with all of the
        logical operators. Note that the destabilizer is calculated using the reduced
        stabilizer array and not the original one.
        """
        z_state_stabarray = merge_stabarrays(
            (self.reduced_stabarray, self.z_log_stabarray)
        )
        full_destabarray = find_destabarray(
            z_state_stabarray, partial_destabarray=self.x_log_stabarray
        )
        # The last n_logical_qubits rows are the destabilizers of the logical operators
        # and as such they should be omitted from the full_destabarray.
        return StabArray.from_signed_pauli_ops(
            full_destabarray[: -self.n_logical_qubits]
        )

    @cached_property
    def pauli_charges(self) -> dict[tuple[int, ...], str]:
        """
        Calculate Pauli charges for all data qubits in the given Block. The Pauli
        charges are calculated from the stabilizers of the Block. For every data qubit,
        one counts how often the data qubit is included in stabilizers in the X, Y, and Z
        basis respectively. If it is included in an odd number of X, Y, or Z stabilizers,
        the data qubit has a Pauli charge of X, Y, or Z respectively. We only report a
        single Pauli charge where multiple charges are combined into one charge according to
        the product of Pauli matrices. E.g. if a data qubit has a Pauli charge of both X and
        Z, the combined Pauli charge is Y since Y=iXZ.

        E.g. for a d=5 rotated surface code, plotting the Pauli charges on top of the data
        qubits would look like this (where we omitted plotting data qubits with no Pauli
        charge):

        .. code-block::

            Y -- X -- X -- X -- Y
            |    |    |    |    |
            Z --   --   --   -- Z
            |    |    |    |    |
            Z --   --   --   -- Z
            |    |    |    |    |
            Z --   --   --   -- Z
            |    |    |    |    |
            Y -- X -- X -- X -- Y

        In this example of the rotated surface code, the four logical corners have Pauli
        charge Y, and there are two boundaries with Pauli charge X and two with Pauli charge
        Z. The bulk data qubits have no Pauli charge. No Pauli charge is represented by "_"
        in the output of this function but we omitted plotting these in the example above.

        Returns
        -------
        dict[tuple[int, ...], str]
            Dict mapping data qubits to their pauli charges ("_", "X", "Y", or "Z")
        """
        # Initialize the dictionary of pauli charge numbers
        pauli_charge_numbers = {qb: {"X": 0, "Y": 0, "Z": 0} for qb in self.data_qubits}

        # Count the number of X, Y, and Z numbers for each data qubit in each stabilizer
        for stab in self.stabilizers:
            for i, qb in enumerate(stab.data_qubits):
                pauli_charge_numbers[qb][stab.pauli[i]] += 1

        # Construct the dictionary of pauli charges
        pauli_charges = {
            # Get Pauli charge ("_", "X", "Y", or "Z") for each qubit
            qb: paulixz_to_char(
                (charge["X"] + charge["Y"]) % 2,  # The X charge
                (charge["Z"] + charge["Y"]) % 2,  # The Z charge
                # Note that Y contributes to both X and Z charges
            )
            for qb, charge in pauli_charge_numbers.items()
        }

        return pauli_charges

    # Magic methods
    def __eq__(self, other) -> bool:
        """
        Check whether two blocks are equivalent. The order of stabilizers does not
        matter for the comparison. Also the different fields are checked independently
        on the uuids of objects.

        Returns
        -------
        bool
            True if the two blocks are equivalent, False otherwise
        """

        def check_stabilizer_to_circuit_eq(block1: Block, block2: Block) -> bool:
            """
            Check whether the stabilizer to circuit maps are equivalent. Since the
            stabilizers and syndrome circuits can have different uuids in the two
            blocks respectively, the `stabilizer_to_circuit` dict cannot directly be
            compared. Instead it has to be checked if the dict maps the same elements
            to each other by comparing them element-wise.
            """
            for stab_uuid, circ_uuid in block1.stabilizer_to_circuit.items():
                # Find the stabilizer object in block1
                stab1 = [stab for stab in block1.stabilizers if stab.uuid == stab_uuid][
                    0
                ]
                # Find the corresponding stabilizer object in block2 using the == method
                # which ignores the stabilizer uuid. This is done to check that the two
                # stabilizers are mapped to the same syndrome circuits in both blocks
                # respectively, irrespective of different uuids
                stab2 = [stab for stab in block2.stabilizers if stab == stab1][0]
                # Uuid of the syndrome circuit which stab2 is mapped to
                circ2_uuid = block2.stabilizer_to_circuit[stab2.uuid]
                # Find the two syndrome circuit objects
                circ1 = [
                    circ for circ in block1.syndrome_circuits if circ.uuid == circ_uuid
                ][0]
                circ2 = [
                    circ for circ in block2.syndrome_circuits if circ.uuid == circ2_uuid
                ][0]
                # Compared the two using the __eq__ method which ignores uuids
                if circ1 != circ2:
                    return False

            return True

        if isinstance(other, Block):
            blocks_not_equal = (
                self.unique_label != other.unique_label
                or set(self.stabilizers) != set(other.stabilizers)
                or self.logical_x_operators != other.logical_x_operators
                or self.logical_z_operators != other.logical_z_operators
                or sorted(self.syndrome_circuits, key=lambda x: x.name)
                != sorted(other.syndrome_circuits, key=lambda x: x.name)
                or not check_stabilizer_to_circuit_eq(self, other)
            )
            if blocks_not_equal:
                return False
            # Else, No differences found, the blocks are equivalent
            return True
        # Else, cannot compare the block for equivalence with another object with is
        # not a block
        return NotImplemented

    # Constructors
    @classmethod
    def from_blocks(cls, blocks: tuple[Block, ...]) -> Block:
        """
        Combine multiple blocks into a single block. The blocks must not overlap
        with each other. The output block will be of code type "custom".

        Parameters
        ----------
        blocks : tuple[Block, ...]
            The blocks to be combined.

        Returns
        -------
        Block
            The combined block.
        """
        # Check that the input is indeed an iterable of blocks
        if not all(isinstance(block, Block) for block in blocks):
            raise ValueError("The input argument blocks must only contain blocks.")

        # Check that the blocks do not overlap
        cls.blocks_no_overlap(blocks)

        # Combine the stabilizers and logical operators
        stabilizers = tuple(stab for block in blocks for stab in block.stabilizers)
        logical_x_operators = tuple(
            log_x for block in blocks for log_x in block.logical_x_operators
        )
        logical_z_operators = tuple(
            log_z for block in blocks for log_z in block.logical_z_operators
        )

        # Get syndrome_circuits without duplicates and create a mapping from old to new
        # uuids
        syndrome_circuits_with_duplicates = tuple(
            circ for block in blocks for circ in block.syndrome_circuits
        )
        syndrome_circuits = tuple(set(syndrome_circuits_with_duplicates))

        synd_circ_map_old_to_new_uuid = {
            circ.uuid: syndrome_circuits[syndrome_circuits.index(circ)].uuid
            for circ in syndrome_circuits_with_duplicates
        }

        # Combine the stabilizer to circuit maps
        stabilizer_to_circuit = {
            stab_uuid: synd_circ_map_old_to_new_uuid[circ_uuid]
            for block in blocks
            for stab_uuid, circ_uuid in block.stabilizer_to_circuit.items()
        }

        return cls(
            stabilizers=stabilizers,
            logical_x_operators=logical_x_operators,
            logical_z_operators=logical_z_operators,
            syndrome_circuits=syndrome_circuits,
            stabilizer_to_circuit=stabilizer_to_circuit,
            unique_label="+".join(block.unique_label for block in blocks),
            skip_validation=True,
        )

    # Static methods
    @staticmethod
    def blocks_no_overlap(blocks: tuple[Block, ...]) -> tuple[Block, ...]:
        """
        Check that blocks do no overlap, meaning that no blocks should be defined
        on the same data or ancilla qubits.

        NOTE: We do not check that logical operators are not overlapping. In the `Block`
        class there is a validation checking that logical operators are defined on the
        same qubits as the stabilizers.
        """
        for block_i, block_j in combinations(blocks, 2):
            shared_data_qubits = set(block_i.data_qubits) & set(block_j.data_qubits)
            if len(shared_data_qubits) > 0:
                raise ValueError(
                    f"Block '{block_i.unique_label}' and block "
                    f"'{block_j.unique_label}' share the data qubits "
                    f"{shared_data_qubits}."
                )

            shared_ancilla_qubits = set(block_i.ancilla_qubits) & set(
                block_j.ancilla_qubits
            )
            if len(shared_ancilla_qubits) > 0:
                raise ValueError(
                    f"Block '{block_i.unique_label}' and block "
                    f"'{block_j.unique_label}' share the ancilla qubits "
                    f"{shared_ancilla_qubits}."
                )
        return blocks

    # Methods
    def rename(self, name: str) -> Block:
        """
        Return a copy of the Block with the new name.
        """
        return self.__class__(
            unique_label=name,
            stabilizers=self.stabilizers,
            logical_x_operators=self.logical_x_operators,
            logical_z_operators=self.logical_z_operators,
            syndrome_circuits=self.syndrome_circuits,
            stabilizer_to_circuit=self.stabilizer_to_circuit,
        )

    def shift(
        self,
        position: tuple[int, ...],
        new_label: str | None = None,
    ) -> Block:
        """
        Return a copy of the Block where all qubit coordinates are shifted by a given
        position.

        Parameters
        ----------
        position : tuple[int, ...]
            Vector by which the block should be shifted
        new_label : str | None, optional
            New label for the block. If None, the same label is used.

        Returns
        -------
        Block
            A new Block with the shifted qubit coordinates.
        """

        if (
            len(position) != len(self.logical_x_operators[0].data_qubits[0]) - 1
        ):  # Remove unit vector
            raise ValueError(
                f"The shift position has a wrong dimension of {len(position)}. "
                f"Expected {len(self.logical_x_operators[0].data_qubits[0]) - 1} instead."
            )

        if new_label is None:
            new_label = self.unique_label

        def shift_coord(
            coord: tuple[int, ...], shift: tuple[int, ...]
        ) -> tuple[int, ...]:
            if coord is None or len(coord) == 0:
                return coord
            return [
                (coord[i] + shift[i] if i < len(shift) else coord[i])
                for i in range(len(coord))
            ]  # Append the old unit vector

        new_stabilizers = [
            Stabilizer(
                pauli=stab.pauli,
                data_qubits=[shift_coord(qb, position) for qb in stab.data_qubits],
                ancilla_qubits=[
                    shift_coord(qb, position) for qb in stab.ancilla_qubits
                ],
            )
            for stab in self.stabilizers
        ]
        logical_x_operators = [
            PauliOperator(
                log_x.pauli,
                data_qubits=[shift_coord(qb, position) for qb in log_x.data_qubits],
            )
            for log_x in self.logical_x_operators
        ]
        logical_z_operators = [
            PauliOperator(
                log_z.pauli,
                data_qubits=[shift_coord(qb, position) for qb in log_z.data_qubits],
            )
            for log_z in self.logical_z_operators
        ]

        # Replacing old stabilizer uuids with new ones
        stabs_map_old_to_new_uuid = {
            stab.uuid: new_stabilizers[i].uuid
            for i, stab in enumerate(self.stabilizers)
        }
        new_stabilizer_to_circuit = {
            stabs_map_old_to_new_uuid[old_stab_uuid]: circ_uuid
            for old_stab_uuid, circ_uuid in self.stabilizer_to_circuit.items()
        }

        return self.__class__(
            unique_label=new_label,
            stabilizers=new_stabilizers,
            logical_x_operators=logical_x_operators,
            logical_z_operators=logical_z_operators,
            syndrome_circuits=self.syndrome_circuits,
            stabilizer_to_circuit=new_stabilizer_to_circuit,
            skip_validation=True,  # Skip validation since the block has been validated
        )

    @cached_property
    def stabilizers_labels(self) -> dict[str, dict[str, tuple[int, ...]]]:
        """
        Builds a dictionary associating stabilizers, via their uuid, with a
        set of labels defined through a dictionary. Inside Block, this is generically
        populated with the space coordinates of the stabilizer check, corresponding to
        the ancilla which measures each stabilizer.

        This functionality can be leveraged to later provide these labels to Syndromes
        and Detectors associated with a given Stabilizer.

        Returns
        -------
        dict[str, dict[str, tuple[int, ...]]]
            Dictionary associating stabilizer uuids with their labels.
        """

        # Retrieve space coordinates
        labels = {
            stab.uuid: {"space_coordinates": stab.ancilla_qubits[0]}
            for stab in self.stabilizers
        }

        return labels

    def get_stabilizer_label(self, stabilizer_uuid: str) -> dict[str, tuple[int, ...]]:
        """
        Get the labels of a stabilizer, specified by its uuid.

        Parameters
        ----------
        stabilizer_uuid : str
            uuid of the stabilizer.

        Returns
        -------
        dict[str, tuple[int, ...]]
            Labels of the stabilizer.
        """

        return self.stabilizers_labels.get(stabilizer_uuid, {})
