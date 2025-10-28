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

# Import utility library to check for the availability of the cudaq package
import importlib.util as _importlib_util

from .main import convert_circuit_to_cliffordsim
from .converter import detector_reference_states, detector_outcomes
from .utilities import format_channel_label_to_tuple
from .circuit_error_model import (
    CircuitErrorModel,
    ErrorType,
    ApplicationMode,
    ErrorProbProtocol,
    HomogeneousTimeIndependentCEM,
    HomogeneousTimeDependentCEM,
    AsymmetricDepolarizeCEM,
)

try:
    from .eka_circuit_to_stim_converter import (
        EkaCircuitToStimConverter,
        noise_annotated_stim_circuit,
    )
except ModuleNotFoundError:
    pass

try:
    from .eka_circuit_to_qasm_converter import convert_circuit_to_qasm
except ModuleNotFoundError:
    pass

try:
    from .eka_circuit_to_pennylane_converter import convert_circuit_to_pennylane
except ModuleNotFoundError:
    pass

# Import the CUDAQ converter only if the cudaq package is available
if _importlib_util.find_spec("cudaq"):
    from .eka_circuit_to_cudaq_converter import EkaToCudaqConverter, Converter
