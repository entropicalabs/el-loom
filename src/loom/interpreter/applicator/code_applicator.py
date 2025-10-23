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
