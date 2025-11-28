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

import uuid

import pytest

from loom.interpreter.block_history import BlockHistory

# pylint: disable=redefined-outer-name, protected-access


@pytest.fixture
def initial_blocks() -> set[str]:
    """Pytest fixture for a set of valid UUIDs."""
    return {str(uuid.uuid4()) for _ in range(3)}


@pytest.fixture
def block_history(initial_blocks: set[str]) -> BlockHistory:
    """Pytest fixture for a BlockHistory instance."""
    return BlockHistory.create(initial_blocks)


class TestBlockHistory:
    """Unit tests for the BlockHistory class."""

    def test_create_success(self, initial_blocks: set[str]):
        """Test successful creation of BlockHistory."""
        bh = BlockHistory.create(initial_blocks)
        assert bh.blocks_at(0) == initial_blocks
        assert bh._timestamps_sorted_asc == [0]

    def test_create_with_invalid_uuid(self):
        """Test that creating a BlockHistory with an invalid UUID raises a
        ValueError."""
        with pytest.raises(
            ValueError, match="blocks_at_0 must be a set of valid UUID4 strings."
        ):
            BlockHistory.create({"not-a-uuid"})

    def test_max_timestamp_below_ref_value(self, block_history: BlockHistory):
        """Test getting the previous timestamp."""
        block_history.update_blocks(10, set(), set())
        assert block_history.max_timestamp_below_ref_value(10) == 0
        assert block_history.max_timestamp_below_ref_value(5) == 0
        with pytest.raises(IndexError):
            block_history.max_timestamp_below_ref_value(0)

    def test_max_timestamp_below_ref_value_invalid_input(
        self, block_history: BlockHistory
    ):
        """Test max_timestamp_below_ref_value with invalid input."""
        with pytest.raises(
            ValueError,
            match="Timestamp must be a non-negative integer.",
        ):
            block_history.max_timestamp_below_ref_value(-1)

    def test_min_timestamp_above_ref_value(self, block_history: BlockHistory):
        """Test getting the next timestamp."""
        block_history.update_blocks(10, set(), set())
        block_history.update_blocks(20, set(), set())
        assert block_history.min_timestamp_above_ref_value(0) == 10
        assert block_history.min_timestamp_above_ref_value(10) == 20
        assert block_history.min_timestamp_above_ref_value(15) == 20
        assert block_history.min_timestamp_above_ref_value(20) is None

    def test_min_timestamp_above_ref_value_invalid_input(
        self, block_history: BlockHistory
    ):
        """Test min_timestamp_above_ref_value with invalid input."""
        with pytest.raises(
            ValueError, match="Timestamp must be a non-negative integer."
        ):
            block_history.min_timestamp_above_ref_value(-1)

    def test_blocks_at(self, block_history: BlockHistory, initial_blocks: set[str]):
        """Test getting blocks at a specific timestamp."""
        new_block = {str(uuid.uuid4())}
        block_history.update_blocks(10, set(), new_block)

        assert block_history.blocks_at(0) == initial_blocks
        assert block_history.blocks_at(5) == initial_blocks
        assert block_history.blocks_at(10) == initial_blocks | new_block
        assert block_history.blocks_at(15) == initial_blocks | new_block

    def test_blocks_at_invalid_input(self, block_history: BlockHistory):
        """Test blocks_at with invalid input."""
        with pytest.raises(
            ValueError, match="Timestamp must be a non-negative integer."
        ):
            block_history.blocks_at(-1)

    def test_blocks_over_time(
        self, block_history: BlockHistory, initial_blocks: set[str]
    ):
        """Test getting blocks over a time range."""
        new_block_10 = {str(uuid.uuid4())}
        new_block_20 = {str(uuid.uuid4())}
        block_history.update_blocks(10, set(), new_block_10)
        block_history.update_blocks(20, set(), new_block_20)

        # Full range
        timestamps, blocks = zip(*block_history.blocks_over_time(), strict=True)
        assert timestamps == (0, 10, 20)
        assert blocks == (
            initial_blocks,
            initial_blocks | new_block_10,
            initial_blocks | new_block_10 | new_block_20,
        )

        # Partial range
        timestamps, blocks = zip(
            *block_history.blocks_over_time(t_start=5, t_stop=15), strict=True
        )
        assert timestamps == (10,)
        assert blocks == (initial_blocks | new_block_10,)

        # Open-ended start
        timestamps, blocks = zip(
            *block_history.blocks_over_time(t_stop=10), strict=True
        )
        assert timestamps == (0,)
        assert blocks == (initial_blocks,)

        # Open-ended stop
        timestamps, blocks = zip(
            *block_history.blocks_over_time(t_start=10), strict=True
        )
        assert timestamps == (10, 20)
        assert blocks == (
            initial_blocks | new_block_10,
            initial_blocks | new_block_10 | new_block_20,
        )

    def test_blocks_over_time_invalid_input(self, block_history: BlockHistory):
        """Test blocks_over_time with invalid input."""
        with pytest.raises(ValueError):
            for _ in block_history.blocks_over_time(t_start=-1):
                pass
        with pytest.raises(ValueError):
            for _ in block_history.blocks_over_time(t_stop=-1):
                pass

    def test_update_blocks(self, block_history: BlockHistory, initial_blocks: set[str]):
        """Test updating blocks."""

        # Make change for t = 10
        old_block_10 = {list(initial_blocks)[0]}
        new_block_10 = {str(uuid.uuid4())}

        block_history.update_blocks(10, old_block_10, new_block_10)

        expected_blocks_10 = (initial_blocks - old_block_10) | new_block_10
        assert block_history.blocks_at(10) == expected_blocks_10
        assert 10 in block_history._timestamps_set
        assert block_history._timestamps_sorted_asc == [0, 10]

        # Make change for t = 5
        old_block_5 = {(initial_blocks - old_block_10).pop()}
        new_block_5 = {str(uuid.uuid4())}

        block_history.update_blocks(5, old_block_5, new_block_5)

        expected_blocks_5 = (initial_blocks - old_block_5) | new_block_5
        assert block_history.blocks_at(5) == expected_blocks_5
        # Ensure that the change is reflected at t = 10 as well
        assert (
            block_history.blocks_at(10)
            == (expected_blocks_10 - old_block_5) | new_block_5
        )
        assert 5 in block_history._timestamps_set
        assert block_history._timestamps_sorted_asc == [0, 5, 10]

    def test_update_blocks_existing_timestamp(
        self, block_history: BlockHistory, initial_blocks: set[str]
    ):
        """Test updating blocks at an existing timestamp."""
        new_block1 = {str(uuid.uuid4())}
        block_history.update_blocks(10, set(), new_block1)
        assert block_history.blocks_at(10) == initial_blocks | new_block1

        new_block2 = {str(uuid.uuid4())}
        block_history.update_blocks(10, new_block1, new_block2)
        assert block_history.blocks_at(10) == initial_blocks | new_block2
        assert block_history._timestamps_sorted_asc == [0, 10]

    def test_update_blocks_invalid_timestamp(self, block_history: BlockHistory):
        """Test update_blocks with invalid timestamp."""
        with pytest.raises(
            ValueError, match="Timestamp must be a non-negative integer."
        ):
            block_history.update_blocks(-1, set(), set())

    def test_update_blocks_invalid_old_blocks(self, block_history: BlockHistory):
        """Test update_blocks with invalid old_blocks."""
        with pytest.raises(
            ValueError, match="old_blocks must be a set of valid UUID4 strings."
        ):
            block_history.update_blocks(10, {"not-a-uuid"}, set())

    def test_update_blocks_invalid_new_blocks(self, block_history: BlockHistory):
        """Test update_blocks with invalid new_blocks."""
        with pytest.raises(
            ValueError, match="new_blocks must be a set of valid UUID4 strings."
        ):
            block_history.update_blocks(10, set(), {"not-a-uuid"})

    def test_update_blocks_old_blocks_not_present(self, block_history: BlockHistory):
        """Test update_blocks when old_blocks are not present."""
        non_existent_block = {str(uuid.uuid4())}
        with pytest.raises(ValueError, match="Some old_blocks are not present"):
            block_history.update_blocks(10, non_existent_block, set())

    def test_is_timestamp_valid(self):
        """Test the is_timestamp_valid static method."""
        BlockHistory.validate_timestamp(0)
        BlockHistory.validate_timestamp(100)
        with pytest.raises(
            ValueError,
            match=(
                "Timestamp must be a non-negative integer. Got -1 of type int instead."
            ),
        ):
            BlockHistory.validate_timestamp(-1)
        with pytest.raises(
            ValueError,
            match=(
                "Timestamp must be a non-negative integer. Got 1.5 of type "
                "float instead."
            ),
        ):
            BlockHistory.validate_timestamp(1.5)
        with pytest.raises(
            ValueError,
            match=(
                "Timestamp must be a non-negative integer. Got 1 of type str instead."
            ),
        ):
            BlockHistory.validate_timestamp("1")

    def test_is_set_of_uuid4(self):
        """Test the is_set_of_uuid4 class method."""
        valid_set = {str(uuid.uuid4()), str(uuid.uuid4())}
        invalid_set = {str(uuid.uuid4()), "not-a-uuid"}
        not_a_set = [str(uuid.uuid4())]

        assert BlockHistory.is_set_of_uuid4(valid_set) is True
        assert BlockHistory.is_set_of_uuid4(invalid_set) is False
        assert BlockHistory.is_set_of_uuid4(set()) is True
        assert BlockHistory.is_set_of_uuid4(not_a_set) is False

    def test_update_blocks_inconsistent_subsequent_state(
        self, block_history: BlockHistory, initial_blocks: set[str]
    ):
        """
        Test that an error is raised if an update causes an inconsistency in a
        subsequent timestamp.
        """
        block_to_remove_later = list(initial_blocks)[0]
        other_block = list(initial_blocks)[1]

        # At t=10, remove one block
        t_later = 10
        block_history.update_blocks(t_later, {block_to_remove_later}, set())

        # At t=5, try to remove a set of blocks including the one already removed at
        # t=10. This should cause an inconsistency when propagating the change to t=10
        msg = (
            "Inconsistent block update detected. The following blocks were "
            f"not present: {set([block_to_remove_later])} at timestamp {t_later}."
        )
        with pytest.raises(ValueError, match=msg):
            block_history.update_blocks(5, {block_to_remove_later, other_block}, set())

    def test_update_blocks_with_previously_seen_block_raises_error(
        self, block_history: BlockHistory, initial_blocks: set[str]
    ):
        """
        Test that update_blocks raises a ValueError if new_blocks contains a block
        that has already been part of the history.
        """
        # Take one of the blocks that already exists from the initial setup
        reused_block = {list(initial_blocks)[0]}

        # Expect a ValueError because the reused_block is already in _all_blocks_set
        with pytest.raises(
            ValueError,
            match="Some new_blocks have already been present in the block history. "
            f"Blocks seen before: {reused_block}",
        ):
            block_history.update_blocks(
                timestamp=10, old_blocks=set(), new_blocks=reused_block
            )
