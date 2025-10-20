"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from loom.eka import Eka
from loom.interpreter.applicator import CodeApplicator

from ..code_factory import SteaneCode


# pylint: disable=duplicate-code
class SteaneCodeApplicator(CodeApplicator):  # pylint: disable=too-few-public-methods
    """
    Contains the implementation logic for each operation, for the Steane code.
    """

    def __init__(
        self,
        eka: Eka,
    ):
        # Ensure that all blocks are typed SteaneCode
        if any(not isinstance(block, SteaneCode) for block in eka.blocks):
            raise ValueError("All blocks must be of type SteaneCode.")
        super().__init__(eka)
        # Add the extra operations that are supported by the Steane Code
        self.supported_operations |= {}
