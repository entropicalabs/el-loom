"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from loom.eka import Eka
from loom.interpreter.applicator import CodeApplicator

from .grow import grow
from .shrink import shrink
from .merge import merge
from .split import split
from .auxcnot import auxcnot

# from .transversalhadamard import transversalhadamard
from .move_block import move_block
from .rotate_block import rotate_block
from .transversalhadamard import transversalhadamard
from ..code_factory import RotatedSurfaceCode
from .logical_phase_via_ywall import logical_phase_via_ywall
from .state_injection import state_injection


class RotatedSurfaceCodeApplicator(  # pylint: disable=too-few-public-methods
    CodeApplicator
):
    """
    Contains the implementation logic for each operation, for the Rotated Surface Code.
    """

    def __init__(
        self,
        eka: Eka,
    ):
        # Ensure that all blocks are typed RotatedSurfaceCode
        if any(not isinstance(block, RotatedSurfaceCode) for block in eka.blocks):
            raise ValueError("All blocks must be of type RotatedSurfaceCode.")
        super().__init__(eka)
        # Add the extra operations that are supported by the Rotated Surface Code
        self.supported_operations |= {
            "Grow": grow,
            "Shrink": shrink,
            "Merge": merge,
            "Split": split,
            "AuxCNOT": auxcnot,
            "TransversalHadamard": transversalhadamard,
            "MoveBlock": move_block,
            "LogicalPhaseViaYwall": logical_phase_via_ywall,
            "StateInjection": state_injection,
            "RotateBlock": rotate_block,
        }
