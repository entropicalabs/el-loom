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

from enum import Enum
from functools import cached_property, partial
from typing import Any

from pydantic import BaseModel, Field

from ..eka import Circuit
from ..interpreter import InterpretationStep


from .converter import Converter
from .eka_to_pennylane_converter import EkaToPennylaneConverter
from .eka_to_qasm_converter import EkaToQasmConverter
from .eka_to_guppylang_converter import EkaToGuppylangConverter
from .eka_to_stim_converter import EkaToStimConverter
from .eka_to_mimiq_converter import EkaToMimiqConverter
from .eka_to_cudaq_converter import EkaToCudaqConverter


class TargetLanguage(Enum):
    """Enumeration of the available converters."""

    MIMIQ = EkaToMimiqConverter
    OPEN_QASM3 = EkaToQasmConverter
    GUPPYLANG = EkaToGuppylangConverter
    STIM = EkaToStimConverter
    PENNYLANE = partial(EkaToPennylaneConverter, is_catalyst=False)
    PENNYLANE_CATALYST = partial(EkaToPennylaneConverter, is_catalyst=True)
    CUDAQ = EkaToCudaqConverter


class Executor(BaseModel):
    """
    Factory class for executing quantum circuits using different target languages.

    Parameters
    ----------
    target_language : TargetLanguage
        The target language/platform for circuit execution
    """

    target_language: TargetLanguage = Field(
        description="Target language for circuit conversion and execution"
    )

    # Pydantic configuration
    model_config = {
        "arbitrary_types_allowed": True,  # Allow custom types like Converter
    }

    @cached_property
    def converter(self) -> Converter:
        """Return the converter for the target language."""
        return self.target_language.value()  # pylint: disable=no-member

    def export_eka(self, input_data: InterpretationStep) -> Any:
        """
        Factory function that selects the right converter and runs it.

        Parameters
        ----------
        input_data : InterpretationStep
            The EKA interpretation step to convert

        Returns
        -------
        Any
            The converted output in the target language format
        """
        return self.converter.convert(input_data)

    def export_eka_circuit(self, input_data: Circuit) -> Any:
        """
        Export the EKA circuit to the target language format.

        Parameters
        ----------
        input_data : Circuit
            The EKA circuit to convert

        Returns
        -------
        Any
            The converted output in the target language format
        """
        return self.converter.convert_circuit(input_data)

    def parse_outcome(self, run_output: Any) -> dict[str, int | list[int]]:
        """
        Function to parse the measurement outcomes from the target language run output.

        Parameters
        ----------
        run_output : Any
            The output from running the converted circuit

        Returns
        -------
        dict[str, int | list[int]]
            Parsed measurement outcomes as a dictionary mapping channels labels to
            outcomes or lists of outcomes (for multiple shots)
        """
        return self.converter.parse_target_run_outcome(run_output)
