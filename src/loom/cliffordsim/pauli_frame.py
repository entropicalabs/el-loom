"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from __future__ import annotations
from uuid import uuid4, UUID

import numpy as np
from loom.eka.utilities import (
    paulichar_to_xz_npfunc,
    paulixz_to_char_npfunc,
    sparse_formatter,
)


class PauliFrame:
    """
    A class representing a Pauli Frame.
    """

    def __init__(
        self,
        x: np.ndarray,
        z: np.ndarray,
        direction: str = "forward",
        id: str = None,  # pylint: disable=redefined-builtin
    ) -> None:
        """Initializes the the Pauli Frame."""
        z = np.array(z)
        x = np.array(x)

        if len(x) != len(z):
            raise ValueError("The length of z and x vectors should be equal.")

        if not np.all((z == 0) | (z == 1)):
            raise ValueError("z vector should contain only 0 and 1.")

        if not np.all((x == 0) | (x == 1)):
            raise ValueError("x vector should contain only 0 and 1.")

        # broadcast to int8
        self.z = np.array(z, dtype=np.int8)
        self.x = np.array(x, dtype=np.int8)

        # Check validity of direction
        if direction not in ["forward", "backward"]:
            raise ValueError(
                f"Invalid direction '{direction}'. Must be 'forward' or 'backward'."
            )
        self.direction = direction
        # Check the validity of the ID
        self.id = str(uuid4()) if id is None else id
        try:
            uuid_obj = UUID(self.id)
        except ValueError as exc:
            raise ValueError(
                f"Invalid uuid: {self.id}. UUID must be version 4."
            ) from exc
        if uuid_obj.version != 4:
            raise ValueError(f"Invalid uuid: {self.id}. UUID must be version 4.")

    def __repr__(self) -> str:
        str_array = paulixz_to_char_npfunc(self.x, self.z)
        paulistr = "".join(str_array)
        return f"PauliFrame: {paulistr}"

    def __len__(self):
        return len(self.x)

    def copy(self) -> PauliFrame:
        """
        Returns a copy of the Pauli Frame.
        """
        return PauliFrame(self.x.copy(), self.z.copy())

    def __eq__(self, o_pf: PauliFrame) -> bool:
        # check if they have the same length
        if len(self) != len(o_pf):
            return False
        # return whether x's and z's match
        return np.all(self.x == o_pf.x) and np.all(self.z == o_pf.z)

    @classmethod
    def from_string(
        cls,
        pauli_string: str,
        direction: str = "forward",
        id: str = None,  # pylint: disable=redefined-builtin
    ) -> PauliFrame:
        """
        Create a PauliFrame from a string representation of a Pauli operator.
        """
        ps_array = np.array(list(pauli_string))

        x, z = paulichar_to_xz_npfunc(ps_array)

        return PauliFrame(x, z, direction, id)

    def sparse_format(self) -> dict:
        """
        Returns the sparse format of the Pauli Frame.
        """
        s = repr(self)
        return sparse_formatter({"+" + s[12:]})
