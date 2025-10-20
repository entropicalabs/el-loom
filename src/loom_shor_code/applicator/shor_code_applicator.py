"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from loom.eka import Eka
from loom.interpreter.applicator import CodeApplicator

from ..code_factory import ShorCode


class ShorCodeApplicator(CodeApplicator):  # pylint: disable=too-few-public-methods
    """
    Contains the implementation logic for each operation, for the Shor code.
    """

    def __init__(
        self,
        eka: Eka,
    ):
        # Ensure that all blocks are typed ShorCode
        if any(not isinstance(block, ShorCode) for block in eka.blocks):
            raise ValueError("All blocks must be of type ShorCode.")
        super().__init__(eka)
        # Add the extra operations that are supported by the Shor Code
        self.supported_operations |= {}
