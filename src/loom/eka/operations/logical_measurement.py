"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from __future__ import annotations
from functools import cached_property
from pydantic.dataclasses import dataclass
from ..utilities import dataclass_params


@dataclass(**dataclass_params)
class LogicalMeasurement:
    """
    LogicalMeasurement acts as a wrapper to describe the type of logical measurement
    based on the block(s) involved and the pauli product of the observable.
    NOTE: Assumes that each block only consist of one logical qubit, and refers only to
    the latest logical measurement of the block(s).

    E.g. LogicalMeasurement(("block0", "block1"), "ZZ") describes the joint ZZ
    measurement between block0 and block1.

    Parameters
    ----------
    blocks : tuple[str, ...]
        List of names of blocks that are involved in the logical measurement.
    observable : str
        Pauli string describing the type of logical measurement.
    """

    blocks: tuple[str, ...]
    observable: str  # e.g. 'X', 'ZZ'

    @cached_property
    def _as_pairs(self) -> frozenset[tuple[str, str]]:
        """Internal representation as a frozenset of (block, observable) pairs.

        This ensures that two equivalent LogicalMeasurement objects
        (e.g., with the same blocks and observables in different orders)
        compare equal and can be used interchangeably in sets or dicts.
        """
        return frozenset(zip(self.blocks, self.observable, strict=True))

    def __eq__(self, other: LogicalMeasurement) -> bool:
        return self._as_pairs == other._as_pairs

    def __hash__(self) -> int:
        return hash(self._as_pairs)

    def __repr__(self) -> str:
        blocks_str = ", ".join(self.blocks)
        return (
            f"LogicalMeasurement(blocks=({blocks_str}), "
            f"observable='{self.observable}')"
        )
