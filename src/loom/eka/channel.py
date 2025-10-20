"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import Field, field_validator, ValidationInfo
from pydantic.dataclasses import dataclass

from .utilities.validation_tools import (
    dataclass_params,
    uuid_error,
    ValidationInfo,
)


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
        The type of the channel: QUANTUM or CLASSICAL

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

    # Validation functions
    _validate_uuid = field_validator("id")(uuid_error)

    @field_validator("label", mode="after")
    @classmethod
    def set_default_label(cls, v: str, info: ValidationInfo) -> str:
        """
        Set the default label based on the type of the channel, according to
        the following scheme:
        ChannelType.QUANTUM:   "data_qubit"
        ChannelType.CLASSICAL: "classical_bit"
        """
        if v is None and "type" in info.data.keys():
            v = create_default_label(info.data["type"])
        return v

    # Magic methods
    def __eq__(self, other):
        if isinstance(other, Channel):
            return (self.type, self.id) == (other.type, other.id)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.type, self.id))

    # Convenience methods

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
