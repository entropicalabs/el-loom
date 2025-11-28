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

from .enums import (
    SingleQubitPauliEigenstate,
    Direction,
    Orientation,
    ResourceState,
    DiagonalDirection,
    BoolOp,
)
from .exceptions import SyndromeMissingError, AntiCommutationError
from .graph_matrix_utils import (
    binary_gaussian_elimination,
    minimum_edge_coloring,
    extract_subgraphs_from_edge_labels,
    find_maximum_matching,
    cardinality_distribution,
    verify_css_code_condition,
)
from .logical_operator_finding import find_logical_operator_set
from .pauli_array import PauliArray
from .pauli_array_computation import rowsum, ndarray_rowsum
from .pauli_binary_vector_rep import (
    PauliOp,
    SignedPauliOp,
    UnsignedPauliOp,
    pauliops_anti_commute,
)
from .pauli_commutation import paulis_anti_commute, anti_commutes_npfunc
from .pauli_computation import g, g_npfunc
from .pauli_format_conversion import (
    paulichar_to_xz,
    paulichar_to_xz_npfunc,
    paulixz_to_char,
    paulixz_to_char_npfunc,
)
from .serialization import (
    findall,
    apply_to_nested,
    dumps,
    loads,
)
from .stab_array import (
    StabArray,
    find_destabarray,
    invert_bookkeeping_matrix,
    is_stabarray_equivalent,
    is_subset_of_stabarray,
    merge_stabarrays,
    reduce_stabarray,
    reduce_stabarray_with_bookkeeping,
    reindex_stabarray,
    swap_stabarray_rows,
    sparse_formatter,
    stabarray_bge,
    stabarray_bge_with_bookkeeping,
    stabarray_standard_form,
    subtract_stabarrays,
)
from .tableau import is_tableau_valid, tableau_generates_pauli_group
from .validation_tools import (
    uuid_error,
    retrieve_field,
    dataclass_config,
    larger_than_zero_error,
)
