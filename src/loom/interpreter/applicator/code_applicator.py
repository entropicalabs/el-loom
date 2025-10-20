"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from loom.eka import Eka

from .base_applicator import BaseApplicator
from .measure_block_syndromes import measureblocksyndromes
from .measure_logical_pauli import measurelogicalpauli
from .logical_pauli import logical_pauli
from .reset_all_data_qubits import reset_all_data_qubits
from .reset_all_ancilla_qubits import reset_all_ancilla_qubits
from .conditional_logical_pauli import conditional_logical_pauli


class CodeApplicator(BaseApplicator):  # pylint: disable=too-few-public-methods
    """
    Contains the implementation logic for operations at the level of a code.
    CodeOperations are implemented at the level of this CodeApplicator. For more
    specific codes, subclasses are used.
    """

    def __init__(
        self,
        eka: Eka,
    ):
        super().__init__(eka)
        # Add the extra operations that are supported by the all codes
        self.supported_operations |= {
            "MeasureBlockSyndromes": measureblocksyndromes,
            "MeasureLogicalX": measurelogicalpauli,
            "MeasureLogicalY": measurelogicalpauli,
            "MeasureLogicalZ": measurelogicalpauli,
            "ResetAllDataQubits": reset_all_data_qubits,
            "LogicalX": logical_pauli,
            "LogicalY": logical_pauli,
            "LogicalZ": logical_pauli,
            "ResetAllAncillaQubits": reset_all_ancilla_qubits,
            "ConditionalLogicalX": conditional_logical_pauli,
            "ConditionalLogicalY": conditional_logical_pauli,
            "ConditionalLogicalZ": conditional_logical_pauli,
        }
