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

from ..code_factory import RepetitionCode


# pylint: disable=too-few-public-methods
class RepetitionCodeApplicator(CodeApplicator):
    """
    Contains the implementation logic for each operation, for the Repetition Code.
    """

    def __init__(
        self,
        eka: Eka,
    ):
        # Ensure that all blocks are typed RepetitionCode
        if any(not isinstance(block, RepetitionCode) for block in eka.blocks):
            raise ValueError("All blocks must be of type RepetitionCode.")
        super().__init__(eka)
        # Add the extra operations that are supported by the Repetition Code
        self.supported_operations |= {
            "Grow": grow,
            "Shrink": shrink,
            "Split": split,
            "Merge": merge,
        }
