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

from pydantic.dataclasses import dataclass

from loom.eka.utilities import Direction, dataclass_config
from loom.eka.operations import CodeOperation


@dataclass(config=dataclass_config)
class AuxCNOT(CodeOperation):
    """
    Apply a CNOT operation between two blocks using a grow-split-merge-shrink sequence.

    Parameters
    ----------
    input_blocks_name : tuple[str, str]
        Names of the control and target blocks, respectively.
    """

    input_blocks_name: tuple[str, str]


@dataclass(config=dataclass_config)
class TransversalHadamard(CodeOperation):
    """
    Apply a Transversal Hadamard on a block.

    Parameters
    ----------
    input_block_name : str
        Name of the block to apply the transversal hadamard gates.
    """

    input_block_name: str


# pylint: disable=line-too-long
@dataclass(config=dataclass_config)
class MoveBlock(CodeOperation):
    """
    Move the selected block 1 unit in the chosen direction in a fault-tolerant manner.
    The SWAP operation occurs as part of the syndrome extraction round, allowing the block
    to remain fault tolerant throughout the move.

    .. note::
        1 unit is defined as the distance between two adjacent qubits (with equal unit vectors) in the block.
        For example, moving from (0, 0, 0) to (1, 0, 0) is a 1-unit move in the x-direction (i.e., to the "right").

    The possible directions are:

    1. Direction.TOP (top) – Moving the block toward the top; all qubits are shifted by (0, -1, 0)
    2. Direction.BOTTOM (bottom) – Moving the block toward the bottom; all qubits are shifted by (0, 1, 0)
    3. Direction.LEFT (left) – Moving the block toward the left; all qubits are shifted by (-1, 0, 0)
    4. Direction.RIGHT (right) – Moving the block toward the right; all qubits are shifted by (1, 0, 0)

    Parameters
    ----------
    input_block_name : str
        Name of the block to move.
    direction: Direction
        Direction in which to move the block.
    """

    input_block_name: str
    direction: Direction


@dataclass(config=dataclass_config)
class LogicalPhaseViaYwall(CodeOperation):
    """
    Apply a logical phase gate to a RotatedSurfaceCode block by:

    1. Growing it towards the right/left or bottom/top depending on the orientation of
    the X boundary of the block. If the X boundary is horizontal, the block can be
    grown to the right or left. If the X boundary is vertical, the block can be
    grown to the top or bottom.
    2. Moving corners
    3. Measuring a wall of qubits in the Y basis
    4. Getting the block back to its original size

    Parameters
    ----------
    input_block_name : str
        Name of the block to apply the logical phase gate.
    growth_direction : Direction
        Direction in which to grow the block. This is the direction of the X boundary
        of the block. If the X boundary is horizontal, the block has to be grown to the
        right or left. If the X boundary is vertical, the block has to be grown to the
        top or bottom.
    """

    input_block_name: str
    growth_direction: Direction


@dataclass(config=dataclass_config)
class RotateBlock(CodeOperation):
    """
    Rotate a block by 90 degrees in a fault-tolerant manner. This modifies both the bulk
    and the boundaries of the block.

    E.g. RotateBlock on the following rotated surface code block::

                     x                               z
            o --- o --- o                   o --- o --- o
          z |  x  |  z  |                 x |  z  |  x  |
            o --- o --- o      Rotate       o --- o --- o
            |  z  |  x  | z      ->         |  x  |  z  | x
            o --- o --- o                   o --- o --- o
               x                               z

    Parameters
    ----------
    input_block_name : str
        Name of the block to rotate.
    grow_direction : Direction
        Direction in which to grow the block so that the rotation can be done in
        a fault-tolerant manner.
    """

    input_block_name: str
    grow_direction: Direction
