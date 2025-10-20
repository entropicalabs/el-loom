"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from typing import Optional
import uuid
from copy import deepcopy
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassicalRegisterSnapshot:
    """
    A snapshot of the classical register at a specific point in time.
    """

    name: str
    no_of_bits: int
    reg: list[tuple[str, int]]


class ClassicalRegister:
    """
    A register that stores classical bits that can be accessed at runtime by the Engine.

    Comments:
    Consider Big Registry Design with Smaller Classical Registers as Views. Similar
    design to Tableau where all quantum data is in a Tableau.

    Parameters
    ----------
    name: str
        The name of this classical register.
    no_of_bits: int
        The number of bits the register will be initialized with.
    bit_ids: Optional[list[str]]
        A list containing the ids of every bit in the classical register. The ids
        provided do not have to be uuid compatible but must be of equal length to the
        number of bits. If the bit ids are not provided, the bits will be initialized
        with randomized ids generated using uuid4.
    """

    def __init__(
        self, name: str, no_of_bits: int, bit_ids: Optional[list[str]] = None
    ) -> None:
        self.name = name
        _checked_bit_ids = self._check_bit_ids(no_of_bits, bit_ids)
        # Able to modify register and easily update all views/attributes in classical
        # register accordingly.
        self.reg = list(
            zip(_checked_bit_ids, [0 for _ in range(no_of_bits)], strict=True)
        )

    def __eq__(self, input_register: "ClassicalRegister") -> bool:
        """
        2 Classical Registers are equal if their reg and name attribute are equal.
        """
        if (
            isinstance(input_register, type(self))
            and input_register.reg == self.reg
            and input_register.name == self.name
        ):
            return True
        return False

    @property
    def reg(self) -> list[tuple[str, int]]:
        """
        The classical register represented as a list of tuples whose first entry is the
        ID of the bit and the second entry is the value of the bit.
        """
        self._valid_register()
        return self._reg

    @reg.setter
    def reg(self, completed_register: list[tuple[str, int]]) -> None:
        self._reg = completed_register
        self._valid_register()

    def _check_bit_ids(
        self, no_of_bits: int, bit_ids: Optional[list[str]]
    ) -> list[str]:
        """
        Check if bit IDs is provided.
        If not provided, new IDs are generated for every bit in the register.
        If provided, checks that the bit IDs are unique and exists for every bit in
        the register.
        """
        if bit_ids is None:
            # Generate random bit ids equal to the length of self.reg
            return [str(uuid.uuid4()) for i in range(no_of_bits)]
        if len(set(bit_ids)) != len(bit_ids):
            raise ValueError("The bit IDs provided for each bit must be unique")
        if len(bit_ids) != no_of_bits:
            raise ValueError(
                "The number of bit ids in the classical register must be equal to the "
                "number of bits."
            )
        return bit_ids

    def _valid_register(self) -> None:
        """
        The classical register should have bit values that are binary.
        The bit IDs should be valid.
        """
        for each_value in self.bit_reg:
            if each_value not in [0, 1]:
                raise ValueError("The register has a non-binary value.")
        self._check_bit_ids(self.no_of_bits, self.bit_ids)

    @property
    def no_of_bits(self) -> int:
        """
        The total number of bits in the current classical register.
        """
        return len(self._reg)

    @property
    def id_bit_reg(self) -> dict[str, int]:
        """
        A dictionary whose key, value pairs are the bit IDs and their respective bit
        values.
        """
        return dict(self._reg)

    @property
    def bit_ids(self) -> list[str]:
        """
        The bit IDs of all the bits in the classical register.
        """
        return list(self.id_bit_reg.keys())

    @property
    def bit_reg(self) -> list[int]:
        """
        The bit values of all the bits in the classical register.
        """
        return list(self.id_bit_reg.values())

    def create_snapshot(self) -> ClassicalRegisterSnapshot:
        """
        Creates a ClassicalRegisterSnapshot object, a snapshot of the state of the
        ClassicalRegister, that contains important properties of the
        ClassicalRegister.

        The ClassicalRegisterSnapshot object can then be used to restore the state of the
        ClassicalRegister at the time the snapshot was created.
        """
        return ClassicalRegisterSnapshot(
            name=deepcopy(self.name),
            no_of_bits=deepcopy(self.no_of_bits),
            reg=deepcopy(self.reg),
        )

    @classmethod
    def restore(cls, cr_snapshot: ClassicalRegisterSnapshot) -> "ClassicalRegister":
        """
        This method allows us to restore the state of the ClassicalRegister
        with a ClassicalRegisterSnapshot Object.
        The method returns an initialized ClassicalRegister with the Snapshot state.
        """

        creg = cls(name=cr_snapshot.name, no_of_bits=cr_snapshot.no_of_bits)

        creg.reg = cr_snapshot.reg

        return creg
