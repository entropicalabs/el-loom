"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

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
from .utilities import Cbit


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
class InterpretationStep:  # pylint: disable=too-many-instance-attributes
    """
    The `InterpretationStep` class stores all relevant information which was
    generated during interpretation up to the `Operation` which is currently
    interpreted. In every interpretation step, the old `InterpretationStep` instance is
    replaced with an updated instance. After all `Operation`\\s have been interpreted, the
    last `InterpretationStep` instance contains the final output.

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
        The circuit implementing all `Operation` s which have been interpreted until now.
        It consists of a tuple of timeslices, where each timeslice is a tuple of
        `Circuit` objects. They can be composite circuits. At the final step, this is
        used to generate final_circuit.
    final_circuit : Circuit | None
        The final circuit object which is generated after interpreting all operations.
        This is the circuit which is used for the final output of the interpretation, it
        is generated automatically by interpreting all operations.
    block_history : tuple[tuple[Block, ...], ...]
        A history of block configurations. The last element in the tuple is the current
        configuration of blocks. With the interpretation of every `Operation` a new
        tuple of blocks is added to this `block_history` field. While mostly only the
        last configuration of blocks is relevant, the whole history is stored which
        might be useful for plotting.
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
        operations such as shrink or split. The dictionary is a FINAL-to-INITIAL mapping.
        In most cases both key and value will be a single string and there is a 1:1
        mapping from an old stabilizer to a new stabilizer. If there is a case where
        multiple stabilizers are combined into a single stabilizer, the value will be a
        tuple of strings. Conversely, if a single stabilizer is split into multiple
        stabilizers, two keys would be associated with the same value.
        E.g. for a split we match `new_stab1.uuid` to `(old_stab.uuid,)` and `new_stab2.uuid` to
        `(old_stab.uuid,)`. For a situation where we merge two stabilizers, we match
        `merged_stab.uuid` to `(old_stab1.uuid, old_stab.uuid)` .
    logical_x_evolution : dict[str, tuple[str, ...]]
        Keeps track of which logical X operator(s) transformed into which other logical
        X operator(s) due to operations such as shrink or split and eventual
        stabilizer(s) required to go from one to the next. The dictionary is a
        FINAL-to-INITIAL mapping.
        E.g. for a split we match `split_x_op1.uuid` to `(old_x_op.uuid,)` and `split_x_op2.uuid`
        to `(old_x_op.uuid,)`. For a shrink that moved the X operator using adjacent stabilizers,
        we match `new_x_op.uuid` to `(old_x_op.uuid, stab1.uuid, stab2.uuid)`.
    logical_z_evolution : dict[str, tuple[str, ...]]
        Keeps track of which logical Z operator(s) transformed into which other logical
        Z operator(s) due to operations such as shrink or split and eventual
        stabilizer(s) required to go from one to the next. The dictionary is a
        FINAL-to-INITIAL mapping.
        E.g. for a split we match `split_z_op1.uuid` to `(old_z_op.uuid,)` and `split_z_op2.uuid`
        to `(old_z_op.uuid,)`. For a shrink that moved the Z operator using adjacent stabilizers,
        we match `new_z_op.uuid` to `(old_z_op.uuid, stab1.uuid, stab2.uuid)`.
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
    channel_dict : dict[str, Channel]
        A dictionary storing all channels which have been created during the
        interpretation. The keys are the labels of the channels (which are either the
        qubit coordinates or the Cbit tuple). The values are the `Channel` objects.
        Only one Channel is created per qubit. Measurements are associated to individual
        channels. I.e. for every Cbit, there is a separate Channel object.
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
    block_history: tuple[tuple[Block, ...], ...] = Field(
        default_factory=tuple, validate_default=True
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
    channel_dict: dict[str, Channel] = Field(
        default_factory=dict, validate_default=True
    )
    logical_measurements: dict[LogicalMeasurement, tuple[Cbit, ...]] = Field(
        default_factory=dict, validate_default=True
    )
    is_frozen: bool = False

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
        for block in self.block_history[-1]:
            if block.unique_label == label:
                return block
        raise RuntimeError(
            f"No block with label '{label}' found in the current configuration."
        )

    @check_frozen
    def update_block_history_and_evolution_MUT(  # pylint: disable=invalid-name
        self,
        new_blocks: tuple[Block, ...] = tuple(),
        old_blocks: tuple[Block, ...] = tuple(),
        update_evolution: bool = True,  # pylint: disable=unused-argument
    ) -> None:
        """
        Update the block history and the block evolution with the new blocks and
        remove the old blocks from the new state of blocks. If update_evolution is set
        to True, the new blocks are added to the evolution with the assumption that
        they are correlated to all previous blocks, e.g. two blocks merged in one.
        For more subtle operations, one can play with the evolution flag, e.g. resetting
        the state of a block creates a nw block not related to the previous one (for
        detector generation).

        NOTE: This function has side effects on the current InterpretationStep! The
        `block_history` and `block_evolution` fields are updated.

        Parameters
        ----------
        new_blocks : tuple[Block, ...]
            New blocks to be added to the block history and evolution
        old_blocks : tuple[Block, ...]
            Old blocks to be removed from the block history and evolution
        update_evolution : bool
            Flag that enables the addition of the new and old blocks to the block
            evolution.
        """
        current_block_ids = tuple(block.uuid for block in self.block_history[-1])
        # Test for existence of old blocks
        for old_block in old_blocks:
            if old_block.uuid not in current_block_ids:
                raise ValueError(
                    f"Block '{old_block.unique_label}' is not in the current block "
                    "configuration."
                )
        # Test for non-existence of new blocks
        for new_block in new_blocks:
            if new_block.uuid in current_block_ids:
                raise ValueError(
                    f"Block '{new_block.unique_label}' is already in the current block "
                    "configuration."
                )

        # Update block evolution
        if old_blocks and new_blocks:
            self.block_evolution.update(
                {
                    new_block.uuid: tuple(block.uuid for block in old_blocks)
                    for new_block in new_blocks
                }
            )

        new_state_of_blocks = (
            tuple(
                block
                for block in self.block_history[-1]
                if block.uuid not in (old_block.uuid for old_block in old_blocks)
            )
            + new_blocks
        )
        self.block_history += (new_state_of_blocks,)

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

        # Separate cases for X and Z operators because they are located in different dictionaries
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
        `intermediate_circuit_sequence` field is updated.

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
        else:
            self.intermediate_circuit_sequence += (
                (circuit,),
            )  # Add the circuit as a single timeslice

    @check_frozen
    def pop_intermediate_circuit_MUT(  # pylint: disable=invalid-name
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

        return popped_circuits

    @check_frozen
    def get_new_cbit_MUT(  # pylint: disable=invalid-name
        self, register_name: str
    ) -> Cbit:
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
            for block_tuple in self.block_history
            for block in block_tuple
            for stabilizer in block.stabilizers
        }

    @property
    def logical_x_operators_dict(self) -> dict[str, PauliOperator]:
        """
        Return a dictionary of logical X operators with logical operator uuid as keys.
        """
        return {
            logical_x.uuid: logical_x
            for block_tuple in self.block_history
            for block in block_tuple
            for logical_x in block.logical_x_operators
        }

    @property
    def logical_z_operators_dict(self) -> dict[str, PauliOperator]:
        """
        Return a dictionary of logical Z operators with logical operator uuid as keys.
        """
        return {
            logical_z.uuid: logical_z
            for block_tuple in self.block_history
            for block in block_tuple
            for logical_z in block.logical_z_operators
        }
