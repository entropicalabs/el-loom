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

import bisect
import uuid
from collections.abc import Iterator

from pydantic.dataclasses import dataclass


@dataclass
class BlockHistory:
    """
    Class to track the history of blocks over time.

    Attributes
    ----------
    - _blocks_by_timestamp: dict[int, set[str]]
        A mapping from timestamps to sets of block UUIDs present at those times.
    - _timestamps_sorted_asc: list[int]
        A sorted list of timestamps in ascending order. This allows for efficient
        searching of previous and next timestamps.
    - _timestamps_set: set[int]
        A set of timestamps for quick existence checks.
    - _all_blocks_set: set[str]
        A set of all block UUIDs that have ever been present in the history.
    """

    _blocks_by_timestamp: dict[int, set[str]]
    _timestamps_sorted_asc: list[int]
    _timestamps_set: set[int]
    _all_blocks_set: set[str]

    @classmethod
    def create(cls, blocks_at_0: set[str]) -> "BlockHistory":
        """
        Create a BlockHistory instance with initial blocks at timestamp 0.

        Parameters
        ----------
        blocks_at_0 : set[str]
            Set of block UUIDs present at timestamp 0.

        Returns
        -------
        BlockHistory
            An instance of BlockHistory initialized with the given blocks at time 0.
        """
        if not cls.is_set_of_uuid4(blocks_at_0):
            raise ValueError("blocks_at_0 must be a set of valid UUID4 strings.")
        return cls(
            _blocks_by_timestamp={0: blocks_at_0.copy()},
            _timestamps_sorted_asc=[0],
            _timestamps_set={0},
            _all_blocks_set=blocks_at_0.copy(),
        )

    def max_timestamp_below_ref_value(self, ref_timestamp: int) -> int:
        """
        Return the largest timestamp less than the input timestamp that exists
        in the block history.

        Parameters
        ----------
        ref_timestamp : int
            The reference timestamp.

        Returns
        -------
        int
            The largest timestamp less than the input timestamp.
        """
        self.validate_timestamp(ref_timestamp)
        if ref_timestamp == 0:
            raise IndexError("No previous timestamp exists for timestamp 0.")

        idx = bisect.bisect_left(self._timestamps_sorted_asc, ref_timestamp)
        return self._timestamps_sorted_asc[idx - 1]

    def min_timestamp_above_ref_value(self, ref_timestamp: int) -> int | None:
        """
        Return the smallest timestamp greater than the input timestamp that exists
        in the block history.

        Parameters
        ----------
        ref_timestamp : int
            The reference timestamp.

        Returns
        -------
        int | None
            The smallest timestamp greater than the input timestamp, or None if
            no such timestamp exists.
        """
        self.validate_timestamp(ref_timestamp)

        # Use bisect to find the right position and retrieve the next timestamp
        idx = bisect.bisect_right(self._timestamps_sorted_asc, ref_timestamp)
        return (
            self._timestamps_sorted_asc[idx]
            if idx < len(self._timestamps_sorted_asc)
            else None
        )

    def blocks_at(self, timestamp: int) -> set[str]:
        """
        Return the blocks present at the given timestamp. If the exact timestamp
        does not exist, return the blocks at the most recent previous timestamp.

        Parameters
        ----------
        timestamp : int
            The timestamp to query.

        Returns
        -------
        set[str]
            The set of block UUIDs present at the given timestamp.
        """
        self.validate_timestamp(timestamp)

        timestamp = (
            timestamp
            if timestamp in self._timestamps_set
            else self.max_timestamp_below_ref_value(timestamp)
        )
        return self._blocks_by_timestamp[timestamp].copy()

    def blocks_over_time(
        self, t_start: int | None = None, t_stop: int | None = None
    ) -> Iterator[tuple[int, set[str]]]:
        """
        Return the timestamps and corresponding blocks over the specified time range.

        Parameters
        ----------
        t_start : int | None
            The start timestamp (inclusive). If None, starts from the first timestamp.
        t_stop : int | None
            The stop timestamp (exclusive). If None, goes up to the last timestamp.

        Returns
        -------
        Iterator[tuple[int, set[str]]]
            An iterator over (timestamp, set of block UUIDs) tuples within the specified
            time range.
        """
        # Determine start and stop timestamps
        t_start = t_start if t_start is not None else self._timestamps_sorted_asc[0]
        t_stop = t_stop if t_stop is not None else self._timestamps_sorted_asc[-1] + 1

        self.validate_timestamp(t_start)
        self.validate_timestamp(t_stop)

        # Find indices for slicing using bisect
        start_idx = bisect.bisect_left(self._timestamps_sorted_asc, t_start)
        stop_idx = bisect.bisect_left(self._timestamps_sorted_asc, t_stop)

        for idx in range(start_idx, stop_idx):
            timestamp = self._timestamps_sorted_asc[idx]
            # Return a copy of the block set to prevent external modification,
            # since sets are mutable
            yield timestamp, self._blocks_by_timestamp[timestamp].copy()

    def update_blocks(
        self, timestamp: int, old_blocks: set[str], new_blocks: set[str]
    ) -> None:
        """
        Modify the blocks at a given timestamp by replacing old_blocks with new_blocks.
        If the timestamp does not exist, it is created based on the previous timestamp's
        blocks.

        Parameters
        ----------
        timestamp : int
            The timestamp at which to modify the blocks.
        old_blocks : set[str]
            The set of block UUIDs to be removed.
        new_blocks : set[str]
            The set of block UUIDs to be added.
        """
        # Validate timestamp
        self.validate_timestamp(timestamp)
        # Check that all old_blocks and new_blocks are sets of UUIDs (strings)
        if not self.is_set_of_uuid4(old_blocks):
            raise ValueError("old_blocks must be a set of valid UUID4 strings.")
        if not self.is_set_of_uuid4(new_blocks):
            raise ValueError("new_blocks must be a set of valid UUID4 strings.")

        # Ensure that new_blocks have not been seen before
        blocks_seen_before = new_blocks.intersection(self._all_blocks_set)
        if blocks_seen_before:
            raise ValueError(
                "Some new_blocks have already been present in the block history. "
                f"Blocks seen before: {blocks_seen_before}"
            )
        # Update the set of all blocks ever seen
        self._all_blocks_set.update(new_blocks)

        # Check whether timestamp exists
        timestamp_exists = timestamp in self._timestamps_set

        # Determine which block set to modify
        blocks_to_modify = self._blocks_by_timestamp[
            (
                timestamp
                if timestamp_exists
                else self.max_timestamp_below_ref_value(timestamp)
            )
        ]

        # Ensure that old_blocks are present in blocks_to_modify
        if not old_blocks.issubset(blocks_to_modify):
            missing_blocks = old_blocks - blocks_to_modify
            raise ValueError(
                "Some old_blocks are not present in the blocks at the previous "
                f"timestamp. Missing blocks: {missing_blocks}"
            )

        # Compute updated blocks for this timestamp
        updated_blocks = (blocks_to_modify - old_blocks) | new_blocks

        # Update the blocks at the relevant timestamp
        self._blocks_by_timestamp[timestamp] = updated_blocks

        if not timestamp_exists:
            # If timestamp did not exist, insert it into sorted list and set
            bisect.insort(self._timestamps_sorted_asc, timestamp)
            self._timestamps_set.add(timestamp)

        # For operations that run in parallel, there is a chance we might be modifying
        # blocks at a timestamp that is not the latest one.
        # Since timestamp lies in the past, we need to propagate the changes forward
        # to ensure consistency.
        # NOTE: The operations running in parallel with the current one cannot have
        # modified `old_blocks` at any future timestamp since parallel operations
        # cannot have overlapping Block objects.
        for t, block_set in self.blocks_over_time(t_start=timestamp + 1):
            if not old_blocks.issubset(block_set):
                missing_blocks = old_blocks - block_set
                raise ValueError(
                    "Inconsistent block update detected. The following blocks were "
                    f"not present: {missing_blocks} at timestamp {t}."
                )
            # Update all subsequent timestamps to reflect the changes
            block_difference = block_set - old_blocks
            self._blocks_by_timestamp[t] = block_difference | new_blocks

    # Utilities for Validation of Inputs

    @staticmethod
    def validate_timestamp(timestamp: int) -> None:
        """
        Raise ValueError if timestamp is not a non-negative integer.
        """
        if not isinstance(timestamp, int) or timestamp < 0:
            raise ValueError(
                f"Timestamp must be a non-negative integer. Got {timestamp} of type "
                f"{type(timestamp).__name__} instead."
            )

    @staticmethod
    def is_uuid4(s: str) -> bool:
        """
        Return True if s is a valid UUID4 string, False otherwise.
        """
        try:
            u = uuid.UUID(s)
        except ValueError:
            return False
        return u.version == 4

    @classmethod
    def is_set_of_uuid4(cls, blocks: set[str]) -> bool:
        """
        Return True if blocks is a set of valid UUID4 strings, False otherwise.
        """
        return isinstance(blocks, set) and all(cls.is_uuid4(b) for b in blocks)
