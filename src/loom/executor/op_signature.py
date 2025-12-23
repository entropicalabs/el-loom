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
from itertools import chain
from pydantic import field_validator
from pydantic.dataclasses import dataclass

from ..eka.utilities import BoolOp


class OpType(Enum):
    """Enum for operation types."""

    SINGLE_QUBIT = "single_qubit"
    TWO_QUBIT = "two_qubit"
    MEASUREMENT = "measurement"
    RESET = "reset"
    CONTROL_FLOW = "control_flow"
    SUBCIRCUIT = "subcircuit"
    BOOL_LOGIC = "bool_logic"
    UTILS = "utils"
    CUSTOM = "custom"


@dataclass(frozen=True)
class OpSignature:
    """
    A class to represent a channel signature for quantum operations.

    Parameters
    ----------
    name : str
        The name of the channel signature.
    quantum_input : int
        The number of quantum inputs required by the operation.
    classical_input : int
        The number of classical inputs required by the operation.
    description : str, optional
        A description of the channel signature.
        Default is an empty string.
    """

    name: str
    op_type: OpType
    quantum_input: int = 0
    classical_input: int = 0
    is_clifford: bool = True
    description: str = ""

    @field_validator("quantum_input", "classical_input")
    @classmethod
    def validate_positive_inputs(cls, v):
        """Validate that quantum_input and classical_input are non-negative."""
        if v < 0:
            raise ValueError("Input count must be non-negative")
        return v

    @classmethod
    def single_qubit_op_signature(
        cls, name: str, is_clifford: bool = True
    ) -> "OpSignature":
        """Create a single qubit operation signature."""
        return cls(
            name=name,
            op_type=OpType.SINGLE_QUBIT,
            quantum_input=1,
            is_clifford=is_clifford,
        )

    @classmethod
    def reset_op_signature(cls, name: str) -> "OpSignature":
        """Create a reset operation signature."""
        return cls(
            name=name,
            op_type=OpType.RESET,
            quantum_input=1,
            is_clifford=True,
        )

    @classmethod
    def two_qubit_op_signature(cls, name: str) -> "OpSignature":
        """Create a two qubit operation signature."""
        return cls(name=name, op_type=OpType.TWO_QUBIT, quantum_input=2)

    @classmethod
    def measurement_op_signature(cls, name: str) -> "OpSignature":
        """Create a measurement operation signature."""
        return cls(
            name=name, op_type=OpType.MEASUREMENT, quantum_input=1, classical_input=1
        )


CONTROL_FLOW_OP_SIGNATURE = frozenset(
    {
        OpSignature(
            name="classical_if",
            op_type=OpType.CONTROL_FLOW,
        ),
        OpSignature(
            name="classical_else",
            op_type=OpType.CONTROL_FLOW,
        ),
        OpSignature(
            name="end_if",
            op_type=OpType.CONTROL_FLOW,
        ),
    }
)

UTILS_SIGNATURE = frozenset(
    {
        OpSignature(
            name="comment",
            op_type=OpType.UTILS,
        ),
        OpSignature(
            name="indent_more",
            op_type=OpType.UTILS,
        ),
        OpSignature(
            name="indent_less",
            op_type=OpType.UTILS,
        ),
    }
)

BOOL_LOGIC_OP_SIGNATURE = frozenset(
    {
        OpSignature(
            name=n,
            op_type=OpType.BOOL_LOGIC,
            classical_input=2,
            is_clifford=False,
        )
        for n in BoolOp.multi_bit_list()
    }
    | {
        OpSignature(
            name=n,
            op_type=OpType.BOOL_LOGIC,
            classical_input=1,
            is_clifford=False,
        )
        for n in BoolOp.mono_bit_list()
    }
)

STR_SINGLE_QUBIT_CLIFFORD_GATE = frozenset(
    {"i", "x", "y", "z", "h", "phase", "phaseinv"}
)
STR_NONCLIFFORD_SINGLE_QUBIT_GATE = frozenset({"t"})
STR_TWO_QUBIT_GATE = frozenset({"cnot", "cx", "cy", "cz", "swap"})
STR_RESET = frozenset(
    {
        "reset",
        "reset_0",
        "reset_1",
        "reset_+",
        "reset_-",
        "reset_+i",
        "reset_-i",
    }
)
STR_MEAS = frozenset({"measurement", "measure_z", "measure_x", "measure_y"})

CLIFFORD_GATES_SIGNATURE = frozenset(
    chain(
        (
            OpSignature.single_qubit_op_signature(name=g)
            for g in STR_SINGLE_QUBIT_CLIFFORD_GATE
        ),
        (OpSignature.two_qubit_op_signature(name=g) for g in STR_TWO_QUBIT_GATE),
        (OpSignature.measurement_op_signature(name=g) for g in STR_MEAS),
        (OpSignature.reset_op_signature(name=g) for g in STR_RESET),
    )
)

NONCLIFFORD_GATES_SIGNATURE = frozenset(
    OpSignature.single_qubit_op_signature(name=g, is_clifford=False)
    for g in STR_NONCLIFFORD_SINGLE_QUBIT_GATE
)
ALL_EKA_OP_SIGNATURES = frozenset(
    CONTROL_FLOW_OP_SIGNATURE
    | BOOL_LOGIC_OP_SIGNATURE
    | CLIFFORD_GATES_SIGNATURE
    | NONCLIFFORD_GATES_SIGNATURE
    | UTILS_SIGNATURE
)

USUAL_QUANTUM_GATES = frozenset(CLIFFORD_GATES_SIGNATURE)
USUAL_CLIFFORD_GATES = frozenset(CLIFFORD_GATES_SIGNATURE)
