"""
Copyright 2024 Entropica Labs Pte Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

# pylint: disable=too-many-lines
from __future__ import annotations
from itertools import compress
from collections.abc import Sequence

from pydantic.dataclasses import dataclass
from pydantic import Field

from loom.eka.circuit import Circuit, Channel, ChannelType
from loom.eka.block import Block
from loom.eka.pauli_operator import PauliOperator
from loom.eka.stabilizer import Stabilizer
from loom.eka.utilities import SyndromeMissingError
from loom.eka.operations import LogicalMeasurement

from .syndrome import Syndrome
from .detector import Detector
from .logical_observable import LogicalObservable
from .block_history import BlockHistory, BlocksAlreadySeenError, BlocksNotPresentError
from .utilities import Cbit, CompositeOperationSession


def check_frozen(func):
    """
    Decorator to check if the InterpretationStep is frozen before calling a method.
    """

    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if self.is_frozen:
            raise ValueError(
                "Cannot change properties of the final InterpretationStep after the "
                "interpretation is finished."
            )
        return result

    return wrapper


@dataclass
class InterpretationStep:  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """
    The `InterpretationStep` class stores all relevant information which was
    generated during interpretation up to the `Operation` which is currently
    interpreted. In every interpretation step, the old `InterpretationStep` instance is
    replaced with an updated instance. After all `Operation`\\s have been interpreted,
    the last `InterpretationStep` instance contains the final output.

    NOTE on mutability: During the interpretation of an EKA, there is a lot of data
    generated and modified which is stored inside the `InterpretationStep` objects.
    Having the `InterpretationStep` dataclass mutable makes it a lot easier to modify
    the data during interpretation. Therefore many of the convenience methods here have
    side effects on `InterpretationStep`. To make these side effects explicit, those
    methods with side effects have the suffix `_MUT` in their name. This follows the
    Julia convention where functions with side effects have a ``!`` at the end of their
    name.

    NOTE: The operations implemented inside the `applicators` are not pure
    functions, they take the previous `InterpretationStep` as an input and returning an
    updated `InterpretationStep`, effectively emulating that behavior. Make sure to
    always keep track of the `InterpretationStep` you are currently working on,
    since it is updated with every operation.

    Parameters
    ----------
    intermediate_circuit_sequence : tuple[tuple[Circuit, ...], ...]
        The circuit implementing all `Operation` s which have been interpreted until
        now. It consists of a tuple of timeslices, where each timeslice is a tuple of
        `Circuit` objects. They can be composite circuits. At the final step, this is
        used to generate final_circuit.
    final_circuit : Circuit | None
        The final circuit object which is generated after interpreting all operations.
        This is the circuit which is used for the final output of the interpretation, it
        is generated automatically by interpreting all operations.
    block_history : BlockHistory
        A `BlockHistory` object which keeps track of the block UUIDs present at every
        timestamp.
    block_registry : dict[str, Block]
        A dictionary storing all Block objects which have been created during the
        interpretation. The keys are the UUIDs of the blocks. This is used to retrieve
        block objects based on their UUIDs which are stored in the `block_history`
        field.
    syndromes : tuple[Syndrome, ...]
        A tuple of `Syndrome`s which are created due to all syndrome extraction cycles
        up to the `Operation` which is currently interpreted.
    detectors : tuple[Detector, ...]
        A tuple of `Detector`s which are created due to all syndrome extraction cycles
        up to the `Operation` which is currently interpreted.
    logical_observables : tuple[LogicalObservable, ...]
        A tuple of `LogicalObservable` s which were measured until now.
    stabilizer_evolution : dict[str, tuple[str, ...]]
        Keeps track of which stabilizers transformed into which other stabilizers due to
        operations such as shrink or split. The dictionary is a FINAL-to-INITIAL
        mapping. In most cases both key and value will be a single string and there is a
        1:1 mapping from an old stabilizer to a new stabilizer. If there is a case where
        multiple stabilizers are combined into a single stabilizer, the value will be a
        tuple of strings. Conversely, if a single stabilizer is split into multiple
        stabilizers, two keys would be associated with the same value.
        E.g. for a split we match `new_stab1.uuid` to `(old_stab.uuid,)` and
        `new_stab2.uuid` to `(old_stab.uuid,)`. For a situation where we merge two
        stabilizers, we match `merged_stab.uuid` to `(old_stab1.uuid, old_stab.uuid)` .
    logical_x_evolution : dict[str, tuple[str, ...]]
        Keeps track of which logical X operator(s) transformed into which other logical
        X operator(s) due to operations such as shrink or split and eventual
        stabilizer(s) required to go from one to the next. The dictionary is a
        FINAL-to-INITIAL mapping.
        E.g. for a split we match `split_x_op1.uuid` to `(old_x_op.uuid,)` and
        `split_x_op2.uuid` to `(old_x_op.uuid,)`. For a shrink that moved the X operator
        using adjacent stabilizers, we match `new_x_op.uuid` to
        `(old_x_op.uuid, stab1.uuid, stab2.uuid)`.
    logical_z_evolution : dict[str, tuple[str, ...]]
        Keeps track of which logical Z operator(s) transformed into which other logical
        Z operator(s) due to operations such as shrink or split and eventual
        stabilizer(s) required to go from one to the next. The dictionary is a
        FINAL-to-INITIAL mapping.
        E.g. for a split we match `split_z_op1.uuid` to `(old_z_op.uuid,)` and
        `split_z_op2.uuid` to `(old_z_op.uuid,)`. For a shrink that moved the Z operator
        using adjacent stabilizers, we match `new_z_op.uuid` to
        `(old_z_op.uuid, stab1.uuid, stab2.uuid)`.
    block_evolution : dict[str, tuple[str, ...]]
        Keeps track of which block(s) transformed into which other block(s) due to
        operations such as merge and split. If there is a 1:1 mapping between and old
        block and a new block (e.g. due to renaming), the value will be a
        tuple containing a single string. If one block is split into two blocks, two
        keys will be associated to the same value that is a tuple containing a single
        string. If two blocks are merged into a single block, the key will be a single
        string and the value will be a tuple of two strings.
        E.g. for a merge, we match `merged_block.uuid` to `(block1.uuid, block2.uuid)`.
    block_qec_rounds : dict[str, int]
        A dictionary storing for every block id how many syndrome extraction rounds have
        been performed on this block. This is needed for creating new `Syndrome` and
        `Detector` objects which have a `round` attribute, specifying the syndrome
        extraction round of the block in which they were measured.
    cbit_counter : dict[str, int]
        A dictionary storing how many measurements have been performed and stored in
        each classical register. The keys are the labels of the classical registers
        which are used as the first element in `Cbit`.
    block_decoding_starting_round : dict[str, int]
        A dictionary storing for every block the round from which the decoding of this
        block should start the next time real-time decoding is performed. E.g. if we
        encounter a non-Clifford gate on a block at time t, we need to decode until this
        time t. Then in this dictionary, we store that the next decoding round has to
        include detectors up to time t+1.
    logical_x_operator_updates : dict[str, tuple[Cbit, ...]]
        A dictionary storing for every logical X operator, the measurements (in the form
        of Cbits) which need to be taken into account for updating the Pauli frame of
        this logical operator once this operator is measured. Elements will be added
        here when some of the data qubits of the respective logical operator are
        measured, e.g. in a shrink or split operation. In this case, these measurements
        lead to a change of pauli frame and need to be included in the next readout of
        this operator. This is also needed for real-time decoding. The values can be
        accessed via `logical_x_operator_updates[logical_x.uuid]`.
        E.g. for a shrink of length 2 we match `new_x_op.uuid` to `(cbit1, cbit2,)`.
    logical_z_operator_updates : dict[str, tuple[Cbit, ...]]
        A dictionary storing for every logical Z operator, the measurements (in the form
        of Cbits) which need to be taken into account for updating the Pauli frame of
        this logical operator once this operator is measured. Elements will be added
        here when some of the data qubits of the respective logical operator are
        measured, e.g. in a shrink or split operation. In this case, these measurements
        lead to a change of pauli frame and need to be included in the next readout of
        this operator. This is also needed for real-time decoding. The values can be
        accessed via `logical_z_operator_updates[logical_x.uuid]`.
        E.g. for a shrink of length 2 we match `new_z_op.uuid` to `(cbit1, cbit2,)`.
    stabilizer_updates : dict[str, tuple[Cbit, ...]]
        A dictionary storing updates for stabilizers which need to be included when the
        stabilizer is measured the next time. Elements will be added here when some of
        the data qubits of the respective stabilizer are measured (in other words when
        the weight of the stabilizer is reduced), e.g. in a shrink or split operation.
        The keys of the dictionary are uuids of stabilizers.
        E.g. for a shrink that changes a weight 4 stabilizer to a weight 2 stabilizer
        we match `new_stab.uuid` to `(cbit1, cbit2)`.
        CAUTION:
        Some applicators may pop the entries from the stabilizer_updates field of the
        interpretation step to compute corrections. This may cause issues in the future
        if the information in this field also needs to be accessed somewhere else.
    reset_single_qubit_stabilizers: dict[str, set[Stabilizer]]
        A dictionary storing the qubits that need to be reset before the support
        of a stabilizer or logical operator is increased. Elements will be added here
        when new data qubits need to be initialized before they are added to a block,
        for example in a grow or merge operation.
        The qubits are stored as single-qubit stabilizers, i.e. stabilizers containing
        a single data qubit.
        The keys of the dictionary are block uuids and the values are sets of
        single-qubit stabilizers.
    measured_single_qubit_stabilizers: dict[str, set[Stabilizer]]
        A dictionary storing the qubits that need to be measured when the support
        of a stabilizer or logical operator is reduced. Elements will be added here
        when measured data qubits are removed from a block, for example in a shrink
        or split operation.
        The qubits are stored as single-qubit stabilizers, i.e. stabilizers containing
        a single data qubit.
        The keys of the dictionary are block uuids and the values are sets of
        single-qubit stabilizers.
    channel_dict : dict[str, Channel]
        A dictionary storing all channels which have been created during the
        interpretation. The keys are the labels of the channels (which are either the
        qubit coordinates or the Cbit tuple). The values are the `Channel` objects.
        Only one Channel is created per qubit. Measurements are associated to individual
        channels. I.e. for every Cbit, there is a separate Channel object.
    composite_operation_session_stack : list[CompositeOperationSession]
        A stack of composite operation sessions which are currently open. Every time a
        composite operation is started, a new session is created and added to this
        stack. When the composite operation is ended, the session is removed from the
        stack.
    timeslice_durations : list[int]
        A list storing the duration of each timeslice in the
        `intermediate_circuit_sequence` field.
    is_frozen : bool
        A boolean flag, indicating whether the `InterpretationStep` is frozen. If it is
        set to True (frozen), calling methods which mutate the `InterpretationStep` will
        raise an exception. Defaults to False.
    """

    intermediate_circuit_sequence: tuple[tuple[Circuit, ...], ...] = Field(
        default_factory=tuple, validate_default=False
    )
    final_circuit: Circuit | None = Field(
        default=None, validate_default=False, init=False
    )
    block_history: BlockHistory = Field()
    block_registry: dict[str, Block] = Field(
        default_factory=dict, validate_default=True
    )
    syndromes: tuple[Syndrome, ...] = Field(
        default_factory=tuple, validate_default=True
    )
    detectors: tuple[Detector, ...] = Field(
        default_factory=tuple, validate_default=True
    )
    logical_observables: tuple[LogicalObservable, ...] = Field(
        default_factory=tuple, validate_default=True
    )
    stabilizer_evolution: dict[str, tuple[str, ...]] = Field(
        default_factory=dict, validate_default=True
    )
    logical_x_evolution: dict[str, tuple[str, ...]] = Field(
        default_factory=dict, validate_default=True
    )
    logical_z_evolution: dict[str, tuple[str, ...]] = Field(
        default_factory=dict, validate_default=True
    )
    block_evolution: dict[str, tuple[str, ...]] = Field(
        default_factory=dict, validate_default=True
    )
    block_qec_rounds: dict[str, int] = Field(
        default_factory=dict, validate_default=True
    )
    cbit_counter: dict[str, int] = Field(default_factory=dict, validate_default=True)
    block_decoding_starting_round: dict[str, int] = Field(
        default_factory=dict, validate_default=True
    )
    logical_x_operator_updates: dict[str, tuple[Cbit, ...]] = Field(
        default_factory=dict, validate_default=True
    )
    logical_z_operator_updates: dict[str, tuple[Cbit, ...]] = Field(
        default_factory=dict, validate_default=True
    )
    stabilizer_updates: dict[str, tuple[Cbit, ...]] = Field(
        default_factory=dict, validate_default=True
    )
    reset_single_qubit_stabilizers: dict[str, set[Stabilizer]] = Field(
        default_factory=dict, validate_default=True
    )
    measured_single_qubit_stabilizers: dict[str, set[Stabilizer]] = Field(
        default_factory=dict, validate_default=True
    )
    channel_dict: dict[str, Channel] = Field(
        default_factory=dict, validate_default=True
    )
    logical_measurements: dict[LogicalMeasurement, tuple[Cbit, ...]] = Field(
        default_factory=dict, validate_default=True
    )
    composite_operation_session_stack: list[CompositeOperationSession] = Field(
        default_factory=list, validate_default=True, init=False
    )
    timeslice_durations: list[int] = Field(
        default_factory=list, validate_default=True, init=False
    )
    is_frozen: bool = False

    @classmethod
    def create(cls, initial_blocks: Sequence[Block], **kwargs) -> InterpretationStep:
        """
        Create a new InterpretationStep with the given initial blocks. It is also
        possible to provide additional fields as keyword arguments for testing purposes.

        Parameters
        ----------
        initial_blocks : Sequence[Block]
            The blocks which are present at the beginning of the interpretation.

        Returns
        -------
        InterpretationStep
            New InterpretationStep instance
        """
        if not isinstance(initial_blocks, Sequence):
            raise TypeError(
                f"Type {type(initial_blocks)} not supported for initial_blocks "
                f"parameter. It must be a Sequence of Block objects."
            )
        if not all(isinstance(block, Block) for block in initial_blocks):
            raise TypeError("All elements of initial_blocks must be Block objects.")
        if not len(initial_blocks) == len(set(block.uuid for block in initial_blocks)):
            raise ValueError("All blocks in initial_blocks must have distinct UUIDs.")

        block_history = BlockHistory.create(
            blocks_at_0={block.uuid for block in initial_blocks}
        )
        block_registry = {block.uuid: block for block in initial_blocks}
        return cls(
            block_history=block_history,
            block_registry=block_registry,
            **kwargs,
        )

    def get_block(self, label: str) -> Block:
        """
        Get the block with the given label from the current block configuration.

        Parameters
        ----------
        label : str
            Unique label of the block

        Returns
        -------
        Block
            Block with the given label
        """
        for block_uuid in self.block_history.blocks_at(self.get_timestamp()):
            block = self.block_registry[block_uuid]
            if block.unique_label == label:
                return block
        raise RuntimeError(
            f"No block with label '{label}' found in the current configuration."
        )

    def get_blocks_at_index(self, index: int) -> tuple[Block, ...]:
        """
        Get the blocks present at the given index in the block history. The order of the
        blocks is not guaranteed within the returned tuple.

        Parameters
        ----------
        index : int
            Index in the block history to get the blocks from

        Returns
        -------
        tuple[Block, ...]
            Blocks present at the given index
        """
        block_uuids = self.block_history.block_uuids_at_index(index)
        return tuple(self.block_registry[block_uuid] for block_uuid in block_uuids)

    @check_frozen
    def update_block_history_and_evolution_MUT(  # pylint: disable=invalid-name
        self,
        new_blocks: Sequence[Block] = tuple(),
        old_blocks: Sequence[Block] = tuple(),
        update_evolution: bool = True,
    ) -> None:
        """
        Update the block history and the block evolution with the new blocks and
        remove the old blocks from the new state of blocks. If update_evolution is set
        to True, the new blocks are added to the evolution with the assumption that
        they are correlated to all previous blocks, e.g. two blocks merged in one.
        For more subtle operations, one can play with the evolution flag, e.g. resetting
        the state of a block creates a new block not related to the previous one (for
        detector generation).

        NOTE: This function has side effects on the current InterpretationStep! The
        `block_history`, `block_evolution`, and `block_registry` fields are updated.


        Parameters
        ----------
        new_blocks : Sequence[Block]
            New blocks to be added to the block history and evolution
        old_blocks : Sequence[Block]
            Old blocks to be removed from the block history and evolution
        update_evolution : bool
            Flag that enables the addition of the new and old blocks to the block
            evolution.
        """
        # Validation of input types
        for arg in [new_blocks, old_blocks]:
            if not isinstance(arg, Sequence):
                raise TypeError(
                    f"Type {type(arg)} not supported for new_blocks/old_blocks "
                    f"parameter. It must be a Sequence of Block objects."
                )
            if not all(isinstance(block, Block) for block in arg):
                raise TypeError(
                    "All elements of new_blocks/old_blocks must be Block objects."
                )
            # Ensure all blocks have distinct UUIDs
            if not len(arg) == len(set(block.uuid for block in arg)):
                raise ValueError(
                    "All blocks in new_blocks/old_blocks must have distinct UUIDs."
                )

        # Update block_registry dictionary
        self.block_registry |= {block.uuid: block for block in new_blocks}

        # Update BlockHistory object
        try:
            self.block_history.update_blocks_MUT(
                timestamp=self.get_timestamp(),
                old_blocks=set(block.uuid for block in old_blocks),
                new_blocks=set(block.uuid for block in new_blocks),
            )
        except BlocksNotPresentError as e:
            # Reraise with more informative message
            block_labels = [
                block.unique_label for block in old_blocks if block.uuid in e.blocks
            ]
            raise RuntimeError(
                "Failed to update block history. Some old_blocks are not present "
                f"in the current block configuration: {block_labels}."
            ) from e
        except BlocksAlreadySeenError as e:
            # Reraise with more informative message
            block_labels = [
                block.unique_label for block in new_blocks if block.uuid in e.blocks
            ]
            raise RuntimeError(
                "Failed to update block history. Some new_blocks have already been "
                f"present in the block history: {block_labels}."
            ) from e

        # Update block evolution
        if old_blocks and new_blocks and update_evolution:
            self.block_evolution.update(
                {
                    new_block.uuid: tuple(block.uuid for block in old_blocks)
                    for new_block in new_blocks
                }
            )

    @check_frozen
    def update_logical_operator_updates_MUT(  # pylint: disable=invalid-name
        self,
        operator_type: str,
        logical_operator_id: str,
        new_updates: tuple[Cbit, ...],
        inherit_updates: bool,
    ) -> None:
        """
        Update the logical_operator_updates dictionary with the new updates for the
        given logical operator. The updates from the previous logical operator are also
        included in the new updates.

        NOTE: This function has side effects on the current InterpretationStep! The
        `logical_x_operator_updates` or `logical_z_operator_updates` field is updated.

        Parameters
        ----------
        operator_type : str
            Type of the logical operator, either 'X' or 'Z'
        logical_operator_id : str
            ID of the new logical operator that inherits the given updates
        new_updates : tuple[Cbit, ...]
            New updates to be added to the logical_operator_updates
        inherit_updates : bool
            If True, the updates from the previous logical operators are also included
            in the new updates. If False, only the new updates are added.
        """

        # Separate cases for X and Z operators because
        # they are located in different dictionaries
        if operator_type == "X":
            logical_evolution = self.logical_x_evolution
            logical_updates = self.logical_x_operator_updates
        elif operator_type == "Z":
            logical_evolution = self.logical_z_evolution
            logical_updates = self.logical_z_operator_updates
        else:
            raise ValueError("Operator type must be labelled either 'X' or 'Z'.")

        # If inherit_updates is True, add the old updates to the new updates
        # Retrieve the previous logical updates
        if inherit_updates:
            old_logical_ids = logical_evolution.get(logical_operator_id, ())
            old_logical_updates = tuple(
                cbit
                for logical_id in old_logical_ids
                for cbit in logical_updates.get(logical_id, ())
            )
            # Add the old updates to the new updates
            new_updates += old_logical_updates

        # Add the updates only if there are new updates
        if new_updates:
            # If the new logical has no updates yet, create an empty tuple
            if logical_operator_id not in logical_updates.keys():
                logical_updates[logical_operator_id] = ()
            # Add the new updates to the logical operator update
            logical_updates[logical_operator_id] += new_updates

    @check_frozen
    def update_reset_single_qubit_stabilizers_MUT(  # pylint: disable=invalid-name
        self,
        block_id: str,
        new_single_qubit_stabilizers: set[Stabilizer],
    ) -> None:
        """
        Update the reset_single_qubit_stabilizers dictionary with the provided
        single-qubit stabilizers for the specified block.

        NOTE: This function has side effects on the current InterpretationStep! The
        `reset_single_qubit_stabilizers` field is updated.

        Parameters
        ----------
        block_id : str
            The uuid of the block associated with the provided single-qubit stabilizers.
        new_single_qubit_stabilizers : set[Stabilizer]
            New single-qubit stabilizers to be associated with the specified block
        """
        # Check that the provided stabilizers are single-qubit stabilizers
        for stabilizer in new_single_qubit_stabilizers:
            if not isinstance(stabilizer, Stabilizer):
                raise TypeError(
                    f"Invalid single-qubit stabilizer: '{stabilizer}'. Must be of type "
                    "`Stabilizer`"
                )
            if len(stabilizer.data_qubits) != 1:
                raise ValueError(
                    "Each single-qubit stabilizer must contain exactly one data qubit."
                )

        # Check that the specified block exists at the current timestep
        if block_id not in self.block_history.block_uuids_at_index(-1):
            raise ValueError(
                f"Block {block_id} not present at current timestep in block history."
            )

        # Retrieve the existing stabilizers for the block
        current_single_qubit_stabilizers = self.reset_single_qubit_stabilizers.get(
            block_id, set()
        )

        # Update reset_single_qubit_stabilizers
        self.reset_single_qubit_stabilizers[block_id] = (
            current_single_qubit_stabilizers | new_single_qubit_stabilizers
        )

    @check_frozen
    def update_measured_single_qubit_stabilizers_MUT(  # pylint: disable=invalid-name
        self,
        block_id: str,
        new_single_qubit_stabilizers: set[Stabilizer],
    ) -> None:
        """
        Update the measured_single_qubit_stabilizers dictionary with the provided
        single-qubit stabilizers for the specified block.

        NOTE: This function has side effects on the current InterpretationStep! The
        `measured_single_qubit_stabilizers` field is updated.

        Parameters
        ----------
        block_id : str
            The uuid of the block associated with the provided single-qubit stabilizers.
        new_single_qubit_stabilizers : set[Stabilizer]
            New single-qubit stabilizers to be associated with the specified block
        """
        # Check that the provided stabilizers are single-qubit stabilizers
        for stabilizer in new_single_qubit_stabilizers:
            if not isinstance(stabilizer, Stabilizer):
                raise TypeError(
                    f"Invalid single-qubit stabilizer: '{stabilizer}'. Must be of type "
                    "`Stabilizer`"
                )
            if len(stabilizer.data_qubits) != 1:
                raise ValueError(
                    "Each single-qubit stabilizer must contain exactly one data qubit."
                )

        # Check that the specified block exists at the current timestep
        if block_id not in self.block_history.block_uuids_at_index(-1):
            raise ValueError(
                f"Block {block_id} not present at current timestep in block history."
            )

        # Retrieve the existing stabilizers for the block
        current_single_qubit_stabilizers = self.measured_single_qubit_stabilizers.get(
            block_id, set()
        )

        # Update measured_single_qubit_stabilizers
        self.measured_single_qubit_stabilizers[block_id] = (
            current_single_qubit_stabilizers | new_single_qubit_stabilizers
        )

    @check_frozen
    def get_channel_MUT(  # pylint: disable=invalid-name
        self, label: str, channel_type: ChannelType = ChannelType.QUANTUM
    ) -> Channel:
        """
        Get the channel for the given label. If no channel exists yet, create one and
        add it to the `channel_dict` dictionary.

        NOTE: This function has side effects on the current InterpretationStep! The
        `channel_dict` field is updated. The channel which is returned will eventually
        be contained in the `circuit` field of `InterpretationStep` as well by adding
        the circuit generated by the respective operation. However the channel might be
        needed several times for the new circuit, therefore it is important to store it
        in the `channel_dict` field, so that it can be reused.

        Parameters
        ----------
        label : str
            Label of the channel (which is the qubit coordinates or the Cbit tuple)
        channel_type : ChannelType
            Type of the channel, only needed if the channel does not exist yet and a new
            channel has to be created. If a channel already exists for the given label,
            the channel_type parameter is ignored. Defaults to ChannelType.QUANTUM.

        Returns
        -------
        Channel
            Corresponding channel
        """
        # Convert label (either coordinate tuple or Cbit) to string
        label = str(label)
        # Create Channel if it does not exist yet
        if label not in self.channel_dict.keys():
            self.channel_dict[label] = Channel(
                type=channel_type,
                label=label,
            )
        return self.channel_dict[label]

    @check_frozen
    def append_circuit_MUT(  # pylint: disable=invalid-name
        self, circuit: Circuit, same_timeslice: bool = False
    ) -> None:
        """
        Append a circuit to the current circuit.

        NOTE: This function has side effects on the current InterpretationStep! The
        `intermediate_circuit_sequence` and `timeslice_durations` fields are updated.

        Parameters
        ----------
        circuit : Circuit
            The circuit to be appended to the current circuit of the InterpretationStep.
            It can only be a single circuit in recursive form.
        same_timeslice : bool
            If True, the circuit is appended to the last timeslice of
            intermediate_circuit_sequence. If False, a new timeslice is created.
        """
        if not isinstance(circuit, Circuit):
            raise TypeError(
                f"Type {type(circuit)} not supported for circuit field. The circuit"
                f" must be a Circuit object"
            )

        # Validation: If there is an open composite operation session, the first circuit
        # of the session cannot be added to the same timeslice as the previous circuit
        first_circuit_of_composite_operations = [
            session
            for session in self.composite_operation_session_stack
            if session.start_timeslice_index == len(self.intermediate_circuit_sequence)
        ]
        if same_timeslice and first_circuit_of_composite_operations:
            raise ValueError(
                "The first circuit of a composite operation session cannot be "
                "added to the same timeslice as the previous circuit. Please set "
                "same_timeslice to False for the first circuit of composite operation "
                "with circuit name "
                f"'{first_circuit_of_composite_operations[-1].circuit_name}'."
            )

        # Append the new circuit to intermediate_circuit_sequence
        if same_timeslice and len(self.intermediate_circuit_sequence) > 0:
            existing_channels = [
                chan
                for circuit in self.intermediate_circuit_sequence[-1]
                for chan in circuit.channels
            ]
            if any(channel in existing_channels for channel in circuit.channels):
                raise ValueError(
                    "The channels of the new circuit are already in use in the current "
                    "timeslice. Please use a new timeslice."
                )

            # Add the circuit to the last timeslice
            self.intermediate_circuit_sequence = self.intermediate_circuit_sequence[
                :-1
            ] + (self.intermediate_circuit_sequence[-1] + (circuit,),)
            # Update the timeslice duration if needed
            self.timeslice_durations[-1] = max(
                self.timeslice_durations[-1], circuit.duration
            )
        else:
            self.intermediate_circuit_sequence += (
                (circuit,),
            )  # Add the circuit as a single timeslice
            # Append the duration of the new timeslice
            self.timeslice_durations.append(circuit.duration)

    @check_frozen
    def _pop_intermediate_circuit_MUT(  # pylint: disable=invalid-name
        self, length: int
    ) -> tuple[tuple[Circuit, ...], ...]:
        """
        Gets the last `length` timeslices of the intermediate circuit sequence and
        removes it from self.intermediate_circuit_sequence.

        Parameters
        ----------
        length : int
            Number of timeslices to pop

        Returns
        -------
        tuple[tuple[Circuit, ...], ...]
            Popped tuple of `length` timeslices.
        """
        if length > len(self.intermediate_circuit_sequence):
            raise ValueError(
                "The number of timeslices to pop exceeds the number of timeslices in "
                "the intermediate circuit sequence."
            )
        if length == 0:
            raise ValueError("The number of timeslices to pop must be greater than 0.")
        popped_circuits = self.intermediate_circuit_sequence[-length:]
        self.intermediate_circuit_sequence = self.intermediate_circuit_sequence[
            :-length
        ]
        # Also remove the durations of the popped timeslices
        self.timeslice_durations = self.timeslice_durations[:-length]
        return popped_circuits

    @check_frozen
    def begin_composite_operation_session_MUT(  # pylint: disable=invalid-name
        self, same_timeslice: bool, circuit_name: str
    ) -> None:
        """
        Marks the beginning of a composite operation in the interpretation step.

        NOTE: This function has side effects on the current InterpretationStep! The
        `composite_operation_session_stack` field is updated.

        Parameters
        ----------
        same_timeslice : bool
            If True, the composite operation will be appended to the last timeslice.
        circuit_name : str
            Name for the wrapped composite circuit.

        Examples
        --------
        Below is an example of how to use the composite operation session methods:

        .. code-block:: python

            # Beginning of composite operation applicator:
            interpretation_step.begin_composite_operation_session_MUT(
                same_timeslice=same_timeslice,
                circuit_name="composite_op_circuit_name",
            )

            # Main body of composite operation applicator
            ...

            # Before exiting the composite operation applicator:
            circuit = interpretation_step.end_composite_operation_session_MUT()
            interpretation_step.append_circuit_MUT(
                circuit,
                same_timeslice=same_timeslice,
            )
        """
        # Create a new session and add it to the composite operation stack
        session = CompositeOperationSession(
            start_timeslice_index=len(self.intermediate_circuit_sequence),
            same_timeslice=same_timeslice,
            circuit_name=circuit_name,
        )
        self.composite_operation_session_stack.append(session)

    @check_frozen
    def end_composite_operation_session_MUT(  # pylint: disable=invalid-name
        self,
    ) -> Circuit:
        """
        Marks the end of a composite operation in the interpretation step.

        NOTE: This function has side effects on the current InterpretationStep! The
        `composite_operation_session_stack`, `intermediate_circuit_sequence`, and
        `timeslice_durations` fields are updated.

        NOTE: It is advised to always follow this function call with an
        `append_circuit_MUT` call to add the returned circuit back to the intermediate
        circuit sequence.

        Returns
        -------
        Circuit
            The wrapped composite circuit.

        Examples
        --------
        Refer to the examples in the `begin_composite_operation_session_MUT` method.
        """
        # Validate that there is a session to end and it matches the most recent one
        if not self.composite_operation_session_stack:
            raise ValueError(
                "No composite operation session to end. Please begin a session first."
            )

        # Pop a session from the stack
        session = self.composite_operation_session_stack.pop()

        # Extract the circuit sequence for the composite operation
        operation_length = (
            len(self.intermediate_circuit_sequence) - session.start_timeslice_index
        )
        circuit_sequence = self._pop_intermediate_circuit_MUT(operation_length)

        # Wrap the circuit sequence into a single Circuit object with proper alignment
        # and padding
        wrapped_circuit_sequence = []
        for timeslice in circuit_sequence:
            # Compute the timespan of the timeslice
            timespan = max(circ.duration for circ in timeslice)
            # Create a template circuit: first element is the original timeslice, rest
            # are empty tuples
            template_circ = [timeslice] + [()] * (timespan - 1)
            # Append the template circuit to the wrapped circuit sequence
            wrapped_circuit_sequence.extend(template_circ)

        # Create the final Circuit object with the given name and return it
        return Circuit(
            name=session.circuit_name, circuit=tuple(wrapped_circuit_sequence)
        )

    def get_timestamp(self) -> int:
        """
        Get the current timestamp of the interpretation step. The timestamp indicates
        the time when the last circuit that was appended to the intermediate circuit
        sequence ends.

        The timestamp is calculated by summing the maximum duration of each timeslice in
        the intermediate circuit sequence, omitting the timeslices which are just before
        active composite operation sessions that are marked as same_timeslice=True. This
        is because these timeslices run in parallel with the previous timeslice and thus
        the previous timeslice's duration should not be considered in the total time.

        NOTE: Access the time after appending the circuits of the current operation, so
        that it includes all relevant timeslices.

        NOTE: To sector the final circuit based on timestamps, unroll the circuit and
        sum the durations of the timeslices until the desired timestamp is reached.

        Returns
        -------
        int
            The current timestamp of the interpretation step.

        Examples
        --------
        Below we illustrate how the timestamp is calculated in different scenarios. Note
        that the values shown correspond to the values after appending the circuit they
        are associated with.

        - No composite operations:

        .. code-block:: text

            "circuit_a": |--3--|                                time = 3
            "circuit_b":       |--2--|                          time = 5
            "circuit_c":       |-----5-----|                    time = 8
            "circuit_d":       |---3---|                        time = 6
            "circuit_e":                   |---3---|            time = 11

        - Nested composite operation sessions:

        .. code-block:: text

            Base circuit:
                "some circuit":                   |--2--|                      time = 2

            Session 0:
                    "ses0 circuit":                     |---------9---------|  time = 11

            Session 1:
                    "ses1 first circuit":               |--2--|                time = 4

                    Nested session 0:
                            "parallel circuit_0":             |----4----|      time = 8
                    Nested session 1:
                            "parallel circuit_1":             |--2--|          time = 6
                    Nested session 2:
                            "parallel circuit_2":             |-----5-----|    time = 9
                    Nested session 3:
                            "parallel circuit_3":             |-1-|            time = 5
        """
        # Build a set of indices to omit
        timeslice_idxs_to_omit = {
            session.start_timeslice_index - 1
            for session in self.composite_operation_session_stack
            if session.same_timeslice
        }

        # Initialize timestamp
        timestamp = 0

        # Sum the durations of the all previous timeslices, omitting the ones in the set
        # and the final timeslice (which is added later)
        timestamp += sum(
            compress(
                self.timeslice_durations,
                # Generator: True = keep, False = omit
                (
                    idx not in timeslice_idxs_to_omit
                    for idx in range(len(self.intermediate_circuit_sequence) - 1)
                ),
            )
        )

        # Add the duration of the final element of the final timeslice
        # (this was appended last)
        if self.intermediate_circuit_sequence:
            # if statement to avoid index error when no circuits have been appended yet
            timestamp += self.intermediate_circuit_sequence[-1][-1].duration

        return timestamp

    @check_frozen
    def get_new_cbit_MUT(  # pylint: disable=invalid-name
        self, register_name: str
    ) -> tuple[str, int]:
        """
        Create a new Cbit for the given register name, considering how often that
        register has been used for measurements before. Increase the respective counter.

        NOTE: This function has side effects on the current InterpretationStep! The
        `cbit_counter` field is updated.

        Parameters
        ----------
        register_name : str
            Classical register name

        Returns
        -------
        Cbit
            Cbit for the new measurement
        """
        # If the register does not exist yet in the counter, create it
        if register_name not in self.cbit_counter.keys():
            self.cbit_counter[register_name] = 0

        # Create the new Cbit, increase the counter and return the Cbit
        cbit = (register_name, self.cbit_counter[register_name])
        self.cbit_counter[register_name] += 1
        return cbit

    @check_frozen
    def append_syndromes_MUT(  # pylint: disable=invalid-name
        self, syndromes: Syndrome | tuple[Syndrome, ...]
    ) -> None:
        """
        Append a new syndrome to the list of syndromes.

        NOTE: This function has side effects on the current InterpretationStep! The
        `syndromes` field is updated.

        Parameters
        ----------
        syndromes : Syndrome | tuple[Syndrome, ...]
            New syndrome(s) to be appended
        """
        if isinstance(syndromes, tuple):
            if any(not isinstance(s, Syndrome) for s in syndromes):
                raise TypeError("All elements in the tuple must be Syndrome objects.")
            self.syndromes += syndromes
        elif isinstance(syndromes, Syndrome):
            self.syndromes += (syndromes,)
        else:
            raise TypeError(
                "Syndrome must be a Syndrome object or a tuple of Syndromes"
            )

    @check_frozen
    def append_detectors_MUT(  # pylint: disable=invalid-name
        self, detectors: Detector | tuple[Detector]
    ) -> None:
        """
        Append new detector(s) to the list of detectors.

        NOTE: This function has side effects on the current InterpretationStep! The
        `detectors` field is updated.

        Parameters
        ----------
        detectors : Detector | tuple[Detector]
            New detector(s) to be appended
        """
        if isinstance(detectors, tuple):
            if any(not isinstance(d, Detector) for d in detectors):
                raise TypeError(
                    "Some elements in the input tuple are not of type Detector"
                )
            self.detectors += detectors
        elif isinstance(detectors, Detector):
            self.detectors += (detectors,)
        else:
            raise TypeError(
                "Input detectors must be of type Detector or tuple of Detectors"
            )

    def get_all_syndromes(self, stab_id: str, block_id: str) -> list[Syndrome]:
        """
        Returns all syndromes associated with a given stabilizer id.

        Parameters
        ----------
        stab_id : str
            Stabilizer uuid to search for.
        block_id : str
            block uuid to search for.

        Returns
        -------
        list[Syndrome]
            List of all syndromes associated with the given stabilizer and block id.
        """
        return [
            syndrome
            for syndrome in self.syndromes
            if syndrome.stabilizer == stab_id and syndrome.block == block_id
        ]

    def get_prev_syndrome(
        self, stabilizer_id: str, block_id: str, current_round: int | None = None
    ) -> list[Syndrome]:
        """
        Finds the latest syndrome for a given stabilizer_id. If current_round is
        given, this function returns the latest syndrome for the associated stabilizer
        such that the round is less than current_round. If None is given, the latest
        syndrome is returned.

        Parameters
        ----------
        stabilizer_id : str
            Stabilizer uuid to search for.
        block_id: str
            block uuid to search for.
        current_round : int | None, optional
            Round to compare to, by default None

        Returns
        -------
        list[Syndrome]
            The latest syndrome for the given stabilizer_id, block_id and current_round.
            Returns an empty list if no Syndrome is found.
        """

        # - Whenever syndromes_id is populated, we exit the all while loops.
        # - We start with the current stabilizer and look for syndromes by traversing
        #   the block history backwards, i.e. we look for syndromes in the current
        #   block and then in the blocks it evolved from - and so on.
        # - If the above fails, we find the stabilizers that the current stabilizer
        #   evolved from and repeat the process until we find syndromes or we fully
        #   traverse the block and stabilizer history of block_id and stabilizer_id.
        syndromes_id = []
        current_stabilizers_id = [stabilizer_id]
        while current_stabilizers_id and not syndromes_id:
            current_blocks_id = [block_id]
            while current_blocks_id and not syndromes_id:
                syndromes_id = [
                    syndrome
                    for prev_block_id in current_blocks_id
                    for stab_id in current_stabilizers_id
                    for syndrome in self.get_all_syndromes(stab_id, prev_block_id)
                ]
                current_blocks_id = [
                    prev_block_id
                    for current_block_id in current_blocks_id
                    for prev_block_id in self.block_evolution.get(current_block_id, [])
                ]
            current_stabilizers_id = [
                prev_stab_id
                for stab_id in current_stabilizers_id
                for prev_stab_id in self.stabilizer_evolution.get(stab_id, [])
            ]

        # If current_round is given, filter the syndromes to only include those
        # that were measured before the current round.
        if current_round is not None:
            syndromes_id = [
                syndrome for syndrome in syndromes_id if syndrome.round < current_round
            ]

        # If no syndromes were found to match the criteria, return an empty list.
        if not syndromes_id:
            return []
        # Return the most recent syndromes, i.e. those with the highest round
        # number.
        max_round = max(syndrome.round for syndrome in syndromes_id)
        most_recent_syndromes = [
            syndrome for syndrome in syndromes_id if syndrome.round == max_round
        ]
        return most_recent_syndromes

    def retrieve_cbits_from_stabilizers(
        self, stabs_required: tuple[Stabilizer, ...], current_block: Block
    ) -> tuple[Cbit, ...]:
        """
        Retrieve the cbits associated with the most recent syndrome extraction of the
        stabilizers required to move the logical operator.

        Parameters
        ----------
        stabs_required : tuple[Stabilizer, ...]
            Stabilizers required to update the logical operator.
        current_block : Block
            Current block in which the stabilizers were measured.

        Returns
        -------
        tuple[Cbit, ...]
            Cbits associated with the measurement of the logical operator displacement.
        """
        last_syndrome_per_stab = [
            self.get_prev_syndrome(stab.uuid, current_block.uuid)
            for stab in stabs_required
        ]
        stabilizers_without_syndrome = [
            stab
            for stab, synd_list in zip(
                stabs_required, last_syndrome_per_stab, strict=True
            )
            if synd_list == []
        ]
        if any(stabilizers_without_syndrome):
            raise SyndromeMissingError(
                "Could not find a syndrome for some stabilizers. "
                f"Stabilizers without syndrome: {stabilizers_without_syndrome}"
            )
        # Because the syndromes are given as a list of a single syndrome, we extract
        # the syndrome from the list
        last_syndrome_per_stab = tuple(
            synd_list[0] for synd_list in last_syndrome_per_stab
        )

        return tuple(
            cbit for synd in last_syndrome_per_stab for cbit in synd.measurements
        )

    @property
    def stabilizers_dict(self) -> dict[str, Stabilizer]:
        """
        Return a dictionary of stabilizers with stabilizer uuid as keys.
        """
        # flatten the block history tuple of tuples
        return {
            stabilizer.uuid: stabilizer
            for block in self.block_registry.values()
            for stabilizer in block.stabilizers
        }

    @property
    def logical_x_operators_dict(self) -> dict[str, PauliOperator]:
        """
        Return a dictionary of logical X operators with logical operator uuid as keys.
        """
        return {
            logical_x.uuid: logical_x
            for block in self.block_registry.values()
            for logical_x in block.logical_x_operators
        }

    @property
    def logical_z_operators_dict(self) -> dict[str, PauliOperator]:
        """
        Return a dictionary of logical Z operators with logical operator uuid as keys.
        """
        return {
            logical_z.uuid: logical_z
            for block in self.block_registry.values()
            for logical_z in block.logical_z_operators
        }
