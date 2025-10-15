"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from .enums import SingleQubitPauliEigenstate, Direction, Orientation, ResourceState
from .exceptions import SyndromeMissingError, AntiCommutationError
from .graph_matrix_utils import (
    binary_gaussian_elimination,
    minimum_edge_coloring,
    extract_subgraphs_from_edge_labels,
    find_maximum_matching,
    cardinality_distribution,
    verify_css_code_condition,
)
from .serialization import (
    findall,
    apply_to_nested,
    dumps,
    # dump,
    loads,
    # load
)
from .validation_tools import (
    uuid_error,
    retrieve_field,
    dataclass_params,
    larger_than_zero_error,
)

# # from .pauli_array import PauliArray
# # from .pauli_array_computation import rowsum
from .pauli_binary_vector_rep import (
    # PauliOp,
    SignedPauliOp,
    pauliops_anti_commute,
)
from .pauli_commutation import paulis_anti_commute  # , anti_commutes_npfunc
from .pauli_computation import g, g_npfunc
from .pauli_format_conversion import (
    paulichar_to_xz,
    paulichar_to_xz_npfunc,
    paulixz_to_char,
    paulixz_to_char_npfunc,
)
from .stab_array import (
    StabArray,
    merge_stabarrays,
    # #     swap_stabarray_rows,
    stabarray_bge,
    # #     reduce_stabarray,
    stabarray_bge_with_bookkeeping,
    reduce_stabarray_with_bookkeeping,
    invert_bookkeeping_matrix,
    find_destabarray,
    is_stabarray_equivalent,
)
from .tableau import is_tableau_valid  # tableau_generates_pauli_group
