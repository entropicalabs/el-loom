"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from __future__ import annotations

from functools import cached_property
import textwrap
from uuid import uuid4
from typing import ClassVar

from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass

from .circuit import Circuit
from .channel import Channel, ChannelType
from .utilities import BoolOp
from .utilities.validation_tools import dataclass_config


# pylint: disable=arguments-differ, arguments-renamed, unnecessary-lambda-assignment
@dataclass(config=dataclass_config)
class IfElseCircuit(Circuit):
    """
    Branching circuit: executes if_circuit or else_circuit depending on some classical
    condition circuit.
    """

    # Marker to allow other modules to detect IfElseCircuit instances without
    # importing the class (prevents circular imports). This is a ClassVar so
    # pydantic dataclasses won't treat it as a field.
    _loom_ifelse_marker: ClassVar[bool] = True

    # Define class-specific fields for IfElseCircuit
    if_circuit: Circuit | None = Field(
        default_factory=lambda: Circuit("empty_branch"), validate_default=True
    )
    else_circuit: Circuit | None = Field(
        default_factory=lambda: Circuit("empty_branch"), validate_default=True
    )
    condition_circuit: Circuit | None = Field(
        default_factory=lambda: Circuit(
            name=BoolOp.MATCH,
            channels=[Channel(type=ChannelType.CLASSICAL)],
        ),
        validate_default=True,
    )
    id: str = Field(default_factory=lambda: str(uuid4()))

    # Override fields from Circuit
    name: str = Field(default="if-else_circuit", init=False, frozen=True)
    circuit: tuple[tuple[Circuit, ...], ...] = Field(default_factory=tuple, init=False)
    channels: tuple[Channel, ...] = Field(default_factory=tuple, init=False)

    # Validation functions
    @field_validator("condition_circuit")
    @classmethod
    def validate_condition_circuit(cls, value: Circuit) -> Circuit:
        """
        Provide a default condition circuit if none is provided. Otherwise, check if the
        condition circuit is classical. Throw an error if that is not the case.
        """
        if value is None:
            return Circuit(
                BoolOp.MATCH,
                channels=[Channel(type=ChannelType.CLASSICAL)],
            )

        if not all(channel.is_classical() for channel in value.channels):
            raise ValueError(
                "IfElseCircuit `condition_circuit` must be a circuit with classical "
                f"channels only. Found channels: {value.channels}"
            )

        if value.name in BoolOp.multi_bit_list():
            if len(value.channels) < 2:
                raise ValueError(
                    f"Condition circuit with BoolOp '{value.name}' must have at least "
                    "two classical channels."
                )
        elif value.name in BoolOp.mono_bit_list():
            if len(value.channels) != 1:
                raise ValueError(
                    f"Condition circuit with BoolOp '{value.name}' must have only one "
                    "classical channel."
                )
        else:
            raise ValueError(
                f"Unsupported BoolOp '{value.name}' for condition circuit. Supported "
                "BoolOps are: "
                f"{', '.join(BoolOp.multi_bit_list() + BoolOp.mono_bit_list())}."
            )

        return value

    @field_validator("if_circuit", "else_circuit")
    @classmethod
    def validate_circuit_branches(cls, circuit: Circuit) -> Circuit:
        """
        Assign default empty Circuit if None, and wrap base gates into Circuit if
        needed.
        """
        if circuit is None or circuit.name == "empty_branch":
            return Circuit(name="empty_branch")
        if not circuit.circuit:
            return Circuit(name=f"wrapped_{circuit.name}", circuit=circuit)
        return circuit

    def __post_init__(self):
        """
        Post-initialization to set derived fields. As Circuit objects are immutable,
        we use object.__setattr__ to set these fields. Additionally, we do not perform
        validation again here as the fields are derived from already validated fields.
        """
        # Format circuit field as ((if_circuit, else_circuit),)
        object.__setattr__(self, "circuit", ((self.if_circuit, self.else_circuit),))

        # Gather all unique channels from both branches and prepend condition channels
        # Note that the order of channels is:
        #   condition channels, quantum channels, classical channels
        branch_channels = set(self.if_circuit.channels) | set(
            self.else_circuit.channels
        )
        typing_order = (
            ChannelType.QUANTUM,
            ChannelType.CLASSICAL,
        )
        ordered_branch_channels = tuple(
            sorted(branch_channels, key=lambda x: typing_order.index(x.type))
        )
        all_channels = self.condition_circuit.channels + ordered_branch_channels
        object.__setattr__(self, "channels", all_channels)

        # Set duration to max of branches
        object.__setattr__(
            self, "duration", max(self.if_circuit.duration, self.else_circuit.duration)
        )

    # Override Parent Methods
    @classmethod
    def as_gate(cls):
        """Represent IfElseCircuit as a gate."""
        raise NotImplementedError("IfElseCircuit cannot be represented as a gate.")

    def circuit_seq(self):
        """
        Returns the sequence of sub-circuits in the circuit field.
        """
        raise NotImplementedError(
            "IfElseCircuit cannot be converted into a Circuit sequence."
        )

    def flatten(self) -> IfElseCircuit:
        """
        Flatten the IfElseCircuit by flattening its branches and condition circuit.
        """
        flat_if = self.if_circuit.flatten()
        flat_else = self.else_circuit.flatten()
        flat_condition = self.condition_circuit.flatten()

        return IfElseCircuit(
            if_circuit=flat_if,
            else_circuit=flat_else,
            condition_circuit=flat_condition,
        )

    @classmethod
    def unroll(cls, input_circuit: IfElseCircuit) -> tuple[IfElseCircuit, ...]:

        unrolled_if = input_circuit.if_circuit.unroll(input_circuit.if_circuit)
        unrolled_else = input_circuit.else_circuit.unroll(input_circuit.else_circuit)

        wrapped_unrolled_if = Circuit(
            name=input_circuit.if_circuit.name, circuit=unrolled_if
        )
        wrapped_unrolled_else = Circuit(
            name=input_circuit.else_circuit.name, circuit=unrolled_else
        )

        return (
            IfElseCircuit(
                if_circuit=wrapped_unrolled_if,
                else_circuit=wrapped_unrolled_else,
                condition_circuit=input_circuit.condition_circuit,
            ),
        )

    @cached_property
    def is_condition_single_bit(self) -> bool:
        """Check if the condition circuit is a single-bit condition."""
        return self.condition_circuit.name in BoolOp.mono_bit_list()

    @cached_property
    def is_single_gate_conditioned(self) -> bool:
        """Whether this is just a single gate conditioned by a classical condition."""
        return (
            len(self.if_circuit.circuit) == 1
            and len(self.if_circuit.circuit[0]) == 1
            and not self.if_circuit.circuit[0][0].circuit
            and self.else_circuit.name == "empty_branch"
        )

    def __eq__(self, other: IfElseCircuit) -> bool:
        """Check equality between two IfElseCircuit instances."""
        if not isinstance(other, IfElseCircuit):
            return False
        return (
            self.if_circuit == other.if_circuit
            and self.else_circuit == other.else_circuit
            and self.condition_circuit == other.condition_circuit
        )

    def __repr__(self) -> str:
        """Return a concise string representation of the circuit."""
        return (
            f"{self.name}\n"
            f"  if: {self.if_circuit.name}\n"
            f"  else: {self.else_circuit.name}\n"
            f"  condition: {self.condition_circuit.name}"
        )

    @staticmethod
    def construct_padded_circuit_time_sequence():
        """Construct a padded circuit time sequence."""
        raise NotImplementedError(
            "IfElseCircuit cannot construct a padded circuit time sequence."
        )

    def detailed_str(self) -> str:
        """Return a detailed string representation of the circuit."""
        _skip_firstline = lambda s: s.detailed_str().splitlines()[1:]

        if_str = textwrap.indent("\n".join(_skip_firstline(self.if_circuit)), "    ")
        else_str = textwrap.indent(
            "\n".join(_skip_firstline(self.else_circuit)), "    "
        )

        return (
            f"{self.name}\n"
            f"  if: {self.if_circuit.name}\n"
            f"{if_str}\n"
            f"  else: {self.else_circuit.name}\n"
            f"{else_str}\n"
            f"  condition: {self.condition_circuit.name}\n"
        )
