"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from __future__ import annotations
from enum import Enum
from functools import reduce
from typing import Optional, Union
from uuid import uuid4
import logging
from pydantic import Field, field_validator, ValidationInfo
from pydantic.dataclasses import dataclass
from .utilities.serialization import apply_to_nested
from .utilities.validation_tools import (
    uuid_error,
    dataclass_params,
    distinct_error,
    ensure_tuple,
    retrieve_field,
    no_name_error,
)

logging.basicConfig(format="%(name)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


class ChannelType(str, Enum):
    """
    The type of the channel: QUANTUM or CLASSICAL
    More types should be added when we feel the need for it
    """

    QUANTUM = "quantum"
    CLASSICAL = "classical"


def create_default_label(channel_type: ChannelType):
    """
    Creates a default label for the channel.

    Parameters
    ----------
    channel_type : ChannelType
        The type of the channel: QUANTUM, ANCILLA or CLASSICAL

    Returns
    -------
    str
        The default label for the channel
    """
    match channel_type:
        case ChannelType.QUANTUM:
            return "data_qubit"
        case ChannelType.CLASSICAL:
            return "classical_bit"
        case _:
            raise ValueError(f"Channel type {type} not recognized")


@dataclass(**dataclass_params)
class Channel:
    """
    Identifies information channels connecting the Circuit elements: examples are
    classical or quantum bit channels

    Parameter
    ---------
    type: ChannelType
        The type of the channel: QUANTUM or CLASSICAL, default is QUANTUM

    label: str
        The label of the channel, allowing it to be grouped in a user friendly way, E.g.
        can be "red", "ancilla_qubit" or "my_favourite_qubit"

    id: str
        The unique identifier of the channel
    """

    type: ChannelType = Field(default=ChannelType.QUANTUM)
    label: Optional[str] = Field(default=None, validate_default=True)
    id: str = Field(default_factory=lambda: str(uuid4()))
    _validate_uuid = field_validator("id")(uuid_error)

    @field_validator("label", mode="after")
    @classmethod
    def set_default_label(cls, v: str, info: ValidationInfo) -> str:
        """
        Set the default label based on the type of the channel, according to
        the following scheme:
        ChannelType.QUANTUM:   "data_qubit"
        ChannelType.ANCILLA:   "ancilla_qubit"
        ChannelType.CLASSICAL: "classical_bit"
        """
        if v is None and "type" in info.data.keys():
            v = create_default_label(info.data["type"])
        return v

    def __eq__(self, other):
        if isinstance(other, Channel):
            return (self.type, self.id) == (other.type, other.id)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.type, self.id))

    def is_quantum(self) -> bool:
        """Check if the channel is a quantum channel.

        Returns
        -------
        bool
            True if the channel is a quantum channel, False otherwise.
        """
        return self.type == ChannelType.QUANTUM

    def is_classical(self) -> bool:
        """Check if the channel is a classical channel.
        Returns
        -------
        bool
            True if the channel is a classical channel, False otherwise.
        """
        return self.type == ChannelType.CLASSICAL


@dataclass(**dataclass_params)
class Circuit:
    """
    A serializable, recursive circuit representation. Previously defined circuit
    structures can be either reused as nested circuit elements or identified by name
    only for compression

    Parameter
    ---------
    name: str
        The name of the circuit/operation, identifying its behaviour: e.g. "Hadamard",
        "CNOT" or "entangle" for the combination of a Hadamard and a CNOT gate.

    circuit: tuple[tuple[Circuit, ...], ...]
        Alternative inputs are: Circuit, list[Circuit], list[list[Circuit]], ...
        Pydantic type conversion accepts all inputs stated here:
        https://docs.pydantic.dev/latest/concepts/conversion_table/

        The list of nested circuit elements. The outer tuple represents the time step
        (tick) that the enclosed circuits are executed at. The inner tuple
        contains parallel circuits that are executed at the same time step. So each
        channel can only operated on by one circuit each tick.

        An input of a 1D list/tuple of circuits is interpreted as a sequence of
        circuits. So each circuit is executed at its own tick and after the execution
        of the previous circuit is complete.

    channels: tuple[Channel, ...]
        Alternative inputs are: Channel, list[Channel], set[Channel], ...
        Pydantic type conversion accepts all inputs stated in the link above.

        The list of channels involved in the circuit

    id: str
        The unique identifier of the circuit
    """

    name: str
    circuit: tuple[tuple[Circuit, ...], ...] = Field(
        default_factory=tuple, validate_default=True
    )
    channels: tuple[Channel, ...] = Field(default_factory=tuple, validate_default=True)
    duration: Optional[int] = Field(default=None, validate_default=True)
    id: str = Field(default_factory=lambda: str(uuid4()))
    _validate_name = field_validator("name")(no_name_error)
    _validate_uuid = field_validator("id")(uuid_error)

    @field_validator("circuit", mode="before")
    @classmethod
    def format_circuit(
        cls, circuit: Union[Circuit, list[Circuit], list[list[Circuit]]]
    ):
        """
        Format the circuit field to be a 2D tuple of circuits.
        """
        if isinstance(circuit, Circuit):
            return ((circuit,),)
        if circuit in ((), []):
            return ()
        if circuit and all((isinstance(gate, Circuit) for gate in circuit)):
            return reduce(
                lambda x, y: x + ((y,),) + ((),) * (y.duration - 1), circuit, ()
            )
        return circuit

    _validate_channels_tuple = field_validator("channels", mode="before")(ensure_tuple)

    @field_validator("circuit")
    @classmethod
    def validate_timing(
        cls, circuit: tuple[tuple[Circuit, ...], ...]
    ) -> tuple[tuple[Circuit, ...], ...]:
        """
        Validates, that all time steps and durations are consistent. I.e. that no two
        gates are scheduled on the same channel at the same time.
        """
        occupancy_dict = {}
        for tick, time_step in enumerate(circuit):
            for gate in time_step:
                for channel in gate.channels:
                    if occupancy_dict.get(channel.id, -1) < tick:
                        occupancy_dict[channel.id] = tick + gate.duration - 1
                    else:
                        raise ValueError(
                            f"Error while setting up composite circuit: Channel {channel.label}({channel.id[0:6]}..) is subject to more than one operation at tick {tick}."
                        )
        return circuit

    @field_validator("channels")
    @classmethod
    def adjust_channels(cls, channels: Union[Channel, list[Channel]], values: dict):
        """
        Adjusts the channels of the circuit based on the channels of the nested circuits.

        Parameters
        ----------
        channels : Union(list[Channel], Channel)
            The channels of the circuit.

        values : dict
            Values of other fields of the Circuit object.

        Returns
        -------
        list[Channel]
            The adjusted list of channels of the circuit.
        """

        def derive_channels(circuit: tuple[tuple[Circuit, ...], ...]):
            """
            Derive channels from nested circuits when the `channels` field is empty.
            The order of the channels of each type is not conserved (or important).
            The output channels are ordered by type.
            """
            all_channels = {
                channel
                for tick in circuit
                if tick
                for circ in tick
                for channel in circ.channels
            }
            typing_order = (ChannelType.QUANTUM, ChannelType.CLASSICAL)
            ordered_channels = sorted(
                all_channels, key=lambda x: typing_order.index(x.type)
            )
            return tuple(ordered_channels)

        circuit = retrieve_field("circuit", values)
        match (len(circuit) == 0, len(channels) == 0):
            case [True, True]:
                return (Channel(),)
            case [True, False]:
                return distinct_error(channels)
            case [False, True]:
                return derive_channels(circuit)
            case [False, False]:
                if set(derive_channels(circuit)) != set(channels):
                    raise ValueError(
                        "\nError while setting up composite circuit: Provided channels do not match the channels of the sub-circuits. \nMake sure that the sub-circuits channel ids and types match the ones provided.\n"
                    )
        return channels

    @field_validator("duration")
    @classmethod
    def adjust_duration(cls, duration: int, info: ValidationInfo) -> int:
        """
        Sets the duration of the circuit based on the duration of the nested circuits
        (if any).
        """

        def derive_duration(circuit: tuple[tuple[Circuit, ...], ...]) -> int:
            """
            Derive the duration of the circuit from the nested circuits: calculate the
            end tick of each operation and return the maximum.
            """
            return max(
                reduce(
                    lambda x, y: x + y,
                    map(
                        lambda tick: [op.duration + tick[0] for op in tick[1]],
                        enumerate(circuit),
                    ),
                    [],
                )
                + [len(circuit)]
            )

        circuit = retrieve_field("circuit", info)
        match (not circuit, duration is None):
            case [True, True]:
                return 1
            case [True, False]:
                if duration < 1:
                    raise ValueError("Duration must be a positive integer.")
            case [False, True]:
                return derive_duration(circuit)
            case [False, False]:
                derived_duration = derive_duration(circuit)
                if derived_duration != duration:
                    raise ValueError(
                        f"Error while setting up composite circuit: Provided duration ({duration}) does not match the duration of the sub-circuits ({derived_duration})."
                    )
        return duration

    @classmethod
    def as_gate(
        cls, name: str, nr_qchannels: int, nr_cchannels: int = 0, duration: int = 1
    ):
        """
        Create a base gate by specifying the name and optionally the number of quantum
        and classical channels and the duration.

        Parameters
        ----------
        name : str
            The name of the gate.

        nr_qchannels : int
            The number of quantum channels it acts on.

        nr_cchannels : int
            The number of classical channels it acts on.

        duration : int
            The duration of the base gate.

        Returns
        -------
        Circuit
            The base gate Circuit object.
        """
        qchannels = [Channel(type=ChannelType.QUANTUM) for _ in range(nr_qchannels)]
        cchannels = [Channel(type=ChannelType.CLASSICAL) for _ in range(nr_cchannels)]
        return cls(name, channels=qchannels + cchannels, duration=duration)

    @classmethod
    def from_circuits(cls, name: str, circuit=None):
        """
        Create a Circuit object from a list of circuits with relative qubit indices.

        Parameters
        ----------
        name : str
            The name of the circuit.

        circuit : list[tuple[Circuit, list[int]]], list[list[tuple[Circuit, list[int]]]]
            The list of circuits with relative qubit indices.

        Returns
        -------
        Circuit
            The Circuit object.
        """

        def make_chan(
            cidx: int, ctype: ChannelType, cmap: dict[str, Channel]
        ) -> Channel:
            """
            Create a Channel for a relative index if it does not exist in the mapping
            dictionary yet. Check for Channel type inconsistencies of relative indices
            used in multiple sub-Circuits.
            """
            if cidx in cmap.keys():
                if cmap[cidx].type != ctype:
                    raise ValueError(
                        f"Provided channel indices are not consistent with respect to their types. Offending channel {cidx} has type {ctype} but has previously been used with a channel of type {cmap[cidx].type}."
                    )
                return cmap[cidx]
            new_chan = Channel(type=ctype)
            cmap[cidx] = new_chan
            return new_chan

        def make_circ(circtup: tuple[Circuit, list[int]], cmap: dict) -> Circuit:
            """
            Build a sub-Circuit with the correct Channels based on the given relative
            indices.
            """
            nr_prov_channels = len(circtup[1])
            nr_circ_channels = len(circtup[0].channels)
            if nr_prov_channels != nr_circ_channels:
                raise ValueError(
                    f"Provided number of channels {nr_prov_channels} does not match the number of channels {nr_circ_channels} in circuit {circtup[0].name}."
                )
            new_channels = [
                make_chan(cidx, ctype, cmap)
                for cidx, ctype in zip(
                    circtup[1],
                    map(lambda chan: chan.type, circtup[0].channels),
                    strict=True,
                )
            ]
            return circtup[0].clone(new_channels)

        if circuit is None:
            circuit = []
        if isinstance(circuit, tuple) or len(circuit) < 2:
            raise ValueError(
                "Error while creating circuit via from_circuit(): The circuit must be a list of circuits. If the intention is to copy a circuit to deal with Channel objects directly, use the clone() method instead."
            )
        cmap = {}
        new_circ = apply_to_nested(circuit, make_circ, cmap)
        return cls(name, new_circ)

    def clone(self, channels: list[Channel] = None) -> Circuit:
        """
        Convenience method to clone a circuit structure that was defined before.

        Parameter
        ---------
        channels: list[Channel]
            Channels of the new circuit.

        Returns
        -------
        Circuit
        """

        def make_channel_map(
            old_channels: list[Channel], new_channels: list[Channel]
        ) -> dict[str, Channel]:
            """
            Create a mapping from old channel ids to new channels.

            Parameter
            ---------
            old_channels: list[Channel]
                The old channels to be mapped.

            new_channels: list[Channel]
                The new channels to be mapped.

            Returns
            -------
            dict[str, Channel]
                A dictionary that maps the old channel ids to the new channels.
            """

            def match_channel(
                old_channel: Channel, index: int, new_channels: list[Channel]
            ) -> Channel:
                if index < len(new_channels):
                    if old_channel.is_quantum() != new_channels[index].is_quantum():
                        raise ValueError(
                            "Error while cloning circuit: CLASSICAL channels cannot be assigned to QUANTUM channels and vice versa."
                        )
                    return new_channels[index]
                return Channel(type=old_channel.type)

            new_channels = list(
                map(
                    lambda old_channel: match_channel(
                        old_channel[1], old_channel[0], new_channels
                    ),
                    enumerate(old_channels),
                )
            )
            old_chan_ids = list(map(lambda channel: channel.id, old_channels))
            return dict(zip(old_chan_ids, new_channels, strict=True))

        def update_sub_circuit(
            circuit: Circuit, channel_map: dict[str, Circuit]
        ) -> Circuit:
            """
            Update the nested sub-circuits with new channels recursively.

            Parameter
            ---------
            circuit: Circuit
                The circuit to be updated.

            channel_map: dict[str, Channel]
                The channel map to used for looking up the new circuit's channels.

            Returns
            -------
            Circuit
                The updated circuit.
            """
            new_channels = [channel_map[channel.id] for channel in circuit.channels]
            return Circuit(
                circuit.name,
                tuple(
                    (
                        tuple((update_sub_circuit(circ, channel_map) for circ in tick))
                        for tick in circuit.circuit
                    )
                ),
                new_channels,
                circuit.duration,
            )

        if channels is None:
            channels = []
        if isinstance(channels, Channel):
            channels = [channels]
        channel_map = make_channel_map(self.channels, channels)
        return update_sub_circuit(self, channel_map)

    def nr_of_qubits_in_circuit(self):
        """
        Returns the number of qubits in the circuit.

        Parameters
        ----------
        circuit : Circuit
            recursive graph circuit representation

        Returns
        -------
        int
            the number of qubits in the circuit
        """
        return len(list(filter(lambda channel: channel.is_quantum(), self.channels)))

    def circuit_seq(self):
        """
        Returns the sequence of sub-circuits in the circuit field.

        Returns
        -------
        tuple[Circuit, ...]
            The list of sub-circuits in sequence, disregarding ticks.
        """
        return reduce(lambda x, y: x + y, self.circuit, ())

    def flatten(self) -> Circuit:
        """
        Returns the flattened circuit as a copy where all elements in the circuit
        list are physical operations, and there is no further nesting.

        Returns
        -------
        Circuit
            The flattened circuit
        """
        flat_circuit = []
        queue = [self]
        while len(queue) > 0:
            next_circuit = queue.pop()
            if len(next_circuit.circuit) == 0:
                flat_circuit.append(next_circuit)
            else:
                for tick in next_circuit.circuit:
                    for circ in tick:
                        if circ != ():
                            queue.append(circ)
        flat_circuit.reverse()
        return Circuit(self.name, circuit=flat_circuit, channels=self.channels)

    @classmethod
    def unroll(cls, circuit: Circuit) -> tuple[tuple[Circuit, ...], ...]:
        """
        Unrolls the circuits within the time slices using a Depth First Search
        algorithm until the final sequence is composed of only base gates. This method
        preserves the time structure of the circuit (unlike flatten).
        Note that this method returns the unrolled circuit sequence, not a new Circuit.

        Returns
        -------
        tuple[tuple[Circuit, ...], ...]
            The unrolled circuit time sequence
        """
        unrolled_circuit_time_sequence = [
            () for _ in range(max(len(circuit.circuit), circuit.duration))
        ]
        stack = [(0, circuit)]
        while stack:
            time, circ = stack.pop()
            if not circ.circuit:
                unrolled_circuit_time_sequence[time] += (circ,)
            else:
                for i, tick in enumerate(circ.circuit):
                    for sub_circ in reversed(tick):
                        stack.append((time + i, sub_circ))
        return tuple(unrolled_circuit_time_sequence)

    def __eq__(self, other) -> bool:
        """
        Check whether two circuits perform the same gate sequence. I.e. check if the
        same gates are applied to the same qubits in the same order. Circuit and qubit
        names are ignored. It only matters that gates are applied to the same qubits, no
        matter what their internal id or their label is. Any nested structure of the
        circuits is ignored, i.e. the two circuits are unrolled before comparison.
        Note that the order of gates within a timeslice does not matter. It can be
        checked for if one compares a tuple of tuples of gates (Circuit.circuit)
        instead of using `Circuit.__eq__`. The order of the timeslices themselves does
        matter. Empty timeslices are also taken into account.

        Note that this overwrites the default `__eq__` method which would check for
        exact equality, including the equality of all uuids. There are only very few
        cases where one would need to check for exact equality including equality of
        uuids. If such a function will every be needed, it should be implemented as a
        separate method like `is_identical(self, other)` or similar. Since checking for
        equality but ignoring the uuids is the much common use case, overwriting the
        == operator for this check is the better default.

        Returns
        -------
        bool
            True if the two circuits are equivalent, False otherwise
        """
        channel_map = {}
        if isinstance(other, Circuit):
            circ_sequence1 = Circuit.unroll(self)
            circ_sequence2 = Circuit.unroll(other)
            if len(circ_sequence1) != len(circ_sequence2):
                log.info("The two circuits have a different number of time slices.")
                log.debug("%s != %s\n", len(circ_sequence1), len(circ_sequence2))
                return False
            for time_step, (time_slice1, time_slice2) in enumerate(
                zip(circ_sequence1, circ_sequence2, strict=False)
            ):
                if len(time_slice1) == 0 and len(time_slice2) == 0:
                    continue
                if len(time_slice1) != len(time_slice2):
                    log.info(
                        "The two circuits have a different number of gates in a time slice."
                    )
                    log.debug(
                        "%s != %s for time slices %s and %s\n",
                        len(time_slice1),
                        len(time_slice2),
                        time_slice1,
                        time_slice2,
                    )
                    return False
                for gate1, gate2 in zip(
                    sorted((gate for gate in time_slice1), key=lambda x: x.name),
                    sorted((gate for gate in time_slice2), key=lambda x: x.name),
                    strict=False,
                ):
                    if gate1.name != gate2.name:
                        log.info(
                            "The two circuits have different gates in a time slice."
                        )
                        log.debug(
                            "For time steps %s: %s and %s: %s, \n    %s != %s for gates %s and %s\n",
                            time_step,
                            time_slice1,
                            time_step,
                            time_slice2,
                            gate1.name,
                            gate2.name,
                            gate1,
                            gate2,
                        )
                        return False
                    for ch1, ch2 in zip(gate1.channels, gate2.channels, strict=False):
                        if ch1.id not in channel_map:
                            channel_map[ch1.id] = ch2.id
                    if [ch.id for ch in gate2.channels] != [
                        channel_map.get(ch.id) for ch in gate1.channels
                    ]:
                        log.info("The two circuits have different channels in a gate.")
                        log.debug(
                            "\n    %s\n        !=\n    %s\n",
                            [(ch.type, ch.label) for ch in gate2.channels],
                            [(ch.type, ch.label) for ch in gate1.channels],
                        )
                        return False
            return True
        return NotImplemented

    def __repr__(self):
        n_ticks = len(self.circuit)
        if n_ticks == 0:
            title = f"{self.name} (base gate)\n"
        else:
            title = f"{self.name} ({n_ticks} ticks)\n"
        tick_str = title
        for i, tick in enumerate(self.circuit):
            if len(tick) != 0:
                tick_str += f"{i}: {' '.join((gate.name for gate in tick))}\n"
        tick_str = tick_str[:-1]
        return tick_str

    def detailed_str(self):
        """
        Detailed string representation for a `Circuit`, displaying the gates
        and channels per tick.
        """
        tick_str = f"{self.name}\n"
        for i, tick in enumerate(self.circuit):
            tick_str += f"{i}: "
            for gate in tick:
                tick_str += f"{gate.name} - "
                tick_str += f"{' '.join((str(chan.label) for chan in gate.channels))}"
                tick_str += "\n"
        return tick_str

    @staticmethod
    def construct_padded_circuit_time_sequence(
        circuit_time_sequence: tuple[tuple[Circuit, ...], ...],
    ) -> tuple[tuple[Circuit, ...], ...]:
        """
        Construct a padded circuit time sequence.

        The input is a tuple of tuples of circuits, where each tuple of circuits
        represents a time step. Each time step may be of variable duration.
        The output is a tuple of tuples of circuits that includes empty tuples which
        represent time steps where the circuit is busy because of a composite
        sub-circuit.

        Note that the scheduling is done following the time structure of the input,
        if two composite circuits exist in the same time step, they will start at the
        same time but may end at different times. If there are conflicts between
        subsequent circuits, add the minimum amount of padding such that the circuit
        can be executed. The last composite circuit's padding will automatically be
        added since it is the last element in the sequence.

        E.g.:

        .. code-block:: python

            hadamard = Circuit("hadamard", channels=channels[0], duration=1)
            cnot = Circuit("cnot", channels=channels[0:2], duration=2)
            circuit_time_sequence = (
                (hadamard),
                (cnot,),
            )

        Constructing the padded circuit time sequence would result in:

        .. code-block:: python

            padded_circuit_time_sequence = (
                (hadamard,),
                (cnot,),
                (),
            )

        Similarly, if the input is:

        .. code-block:: python

            circuit_time_sequence = (
                (cnot),
                (hadamard,),
            )

        The padded circuit time sequence would be:

        .. code-block:: python

            padded_circuit_time_sequence = (
                (cnot,),
                (),
                (hadamard,),
            )

        To illustrate two circuits that are executed at the same time, but of variable
        duration:

        .. code-block:: python

            hadamard_2 = Circuit("hadamard_2", channels=channels[2], duration=1)
            circuit_time_sequence = (
                (cnot, hadamard_2,)
            )

        The padded circuit time sequence would be:

        .. code-block:: python

            padded_circuit_time_sequence = (
                (cnot, hadamard_2,),
                (),
            )

        where the cnot would span two time steps and hadamard_2 only one.

        Parameters
        ----------
        circuit_time_sequence : tuple[tuple[Circuit, ...], ...]
            The circuit time sequence to be padded.

        Returns
        -------
        tuple[tuple[Circuit, ...], ...]
            The padded circuit time sequence.
        """
        padded_circuit_time_sequence = ()
        occupancy_dictionary = {}
        for tick in circuit_time_sequence:
            current_tick_occupancy = {
                channel.label: circuit.duration
                for circuit in tick
                for channel in circuit.channels
            }
            conflicting_channels = set(occupancy_dictionary.keys()).intersection(
                set(current_tick_occupancy.keys())
            )
            if conflicting_channels:
                duration = max(
                    (occupancy_dictionary[channel] for channel in conflicting_channels)
                )
                padded_circuit_time_sequence += ((),) * (duration - 1)
            else:
                duration = 1
            padded_circuit_time_sequence += (tick,)
            free_channels = set(occupancy_dictionary.keys()).difference(
                set(current_tick_occupancy.keys())
            )
            for channel in free_channels:
                if (new_duration := (occupancy_dictionary.pop(channel) - duration)) > 0:
                    occupancy_dictionary[channel] = new_duration
            occupancy_dictionary.update(current_tick_occupancy)
        if occupancy_dictionary:
            duration = max(occupancy_dictionary.values())
            padded_circuit_time_sequence += ((),) * (duration - 1)
        return padded_circuit_time_sequence
