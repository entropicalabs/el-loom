"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

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

# from .stim_to_circuit_converter import StimToEkaCircuitConverter
# from .qasm_to_circuit_converter import QasmToEkaCircuitConverter
from .eka_circuit_to_stim_converter import (
    EkaCircuitToStimConverter,
    noise_annotated_stim_circuit,
)
from .eka_circuit_to_qasm_converter import convert_circuit_to_qasm
from .eka_circuit_to_pennylane_converter import convert_circuit_to_pennylane
from .eka_circuit_to_mimiq_converter import convert_circuit_to_mimiq

# Import the CUDAQ converter only if the cudaq package is available
if _importlib_util.find_spec("cudaq"):
    from .eka_circuit_to_cudaq_converter import EkaToCudaqConverter, Converter
