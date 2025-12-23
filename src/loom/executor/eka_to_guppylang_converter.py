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

from __future__ import annotations

from collections import defaultdict
from functools import partial, cached_property
import textwrap
from typing import TYPE_CHECKING, Any, Union

from pydantic import Field, field_validator


from ..eka import ChannelType, Circuit
from ..eka.utilities import BoolOp
from .op_signature import (
    ALL_EKA_OP_SIGNATURES,
    BOOL_LOGIC_OP_SIGNATURE,
    CLIFFORD_GATES_SIGNATURE,
    NONCLIFFORD_GATES_SIGNATURE,
    CONTROL_FLOW_OP_SIGNATURE,
    UTILS_SIGNATURE,
    OpType,
    OpSignature,
)
from .converter import Converter, OpAndTargetToInstrCallable

if TYPE_CHECKING:
    from hugr.qsystem.result import QsysResult, QsysShot

else:
    QsysResult = "QsysResult"  # pylint: disable=invalid-name
    QsysShot = "QsysShot"  # pylint: disable=invalid-name

ProgramAndRegister = tuple[str, dict[str, int], dict[str, int]]


class EkaToGuppylangConverter(Converter[ProgramAndRegister, Any]):
    """Convert Eka circuit format to Guppylang program.

    In the program, the quantum registers are represented as an array of qubits named
    `q`, and the classical registers are represented as an array of booleans named `c`.

    The program is generated as a string of Guppylang code, assuming the following
    import prefixes by default:

    - guppylang.std.quantum as `guppyQ.`
    - guppylang.std.qsystem as `guppyQSys.`
    - guppylang.std.builtins as `builtins.`

    Note: the user also needs to import `array` from `guppylang.std.builtins` to use
    arrays in the generated program (it is not exposed by the builtins by default).

    This can be customized via the constructor parameters.

    Parameters
    ----------
    quantum_prefix : str
        The import prefix for quantum operations in Guppylang.
    qsystem_prefix : str
        The import prefix for Qsys operations in Guppylang.
    builtins_prefix : str
        The import prefix for built-in functions in Guppylang.
    """

    SUPPORTED_OPERATIONS: frozenset[OpSignature] = ALL_EKA_OP_SIGNATURES

    ALLOW_ERROR_MODELS: bool = Field(default=False, frozen=True, init=False)

    quantum_prefix: str = Field(default="guppyQ.", frozen=True)
    qsystem_prefix: str = Field(default="guppyQSys.", frozen=True)
    builtins_prefix: str = Field(default="builtins.", frozen=True)

    @field_validator("quantum_prefix", "qsystem_prefix", "builtins_prefix")
    # pylint: disable=no-self-argument
    def validate_prefixes(cls, v):
        """Validate that the prefixes end with a dot."""
        cls._validate_import_prefix(v)
        return v

    @cached_property
    # pylint: disable=unused-argument
    def operations_map(
        self,
    ) -> dict[str, OpAndTargetToInstrCallable]:
        """Map operation names to their corresponding Guppy functions.
        The functions returned by this map take as input the quantum and classical
        targets, and return the Guppy function(s) to apply, handling assignments of
        parameters.

        Returns
        -------
        dict[str, str]
            A mapping from operation names to Guppylang instruction with the proper
            parameters applied.
        """

        def _single_qubit_op(
            q_targets: list[str],
            c_targets: list[str],
            op: str | list[str],
            desc: str = "",
        ) -> str:
            if op == "":
                return ""
            if isinstance(op, list):
                return "\n".join(f"{o}({q_targets[0]})" for o in op)
            return f"{op}({q_targets[0]})"

        def _two_qubit_op(
            q_targets: list[str],
            c_targets: list[str],
            op: str,
            desc: str = "",
        ) -> str:
            if op == "swap":
                return "\n".join(
                    [
                        f"{self.quantum_prefix}cx({q_targets[0]}, {q_targets[1]})",
                        f"{self.quantum_prefix}cx({q_targets[1]}, {q_targets[0]})",
                        f"{self.quantum_prefix}cx({q_targets[0]}, {q_targets[1]})",
                    ]
                )
            return f"{op}({', '.join(q_targets)})"

        def _measurement_op(
            q_targets: list[str],
            c_targets: list[str],
            op: str | list[str],
            desc: str = "",
        ) -> str:
            instructions = []
            if not isinstance(op, list):
                op = [op]
            for single_op in op[:-1]:
                instructions.append(f"{single_op}({q_targets[0]})")
            instructions.append(f"{c_targets[0]} = {op[-1]}({q_targets[0]})")
            instructions.append(
                f"{self.builtins_prefix}result('{desc}', {c_targets[0]})"
            )
            return "\n".join(instructions)

        def _control_flow_op(
            q_targets: list[str],
            c_targets: list[str],
            op: str | list[str],
            desc: str = "",
        ) -> str:
            """
            Generate Guppy code for classical control-flow constructs.
            str_instr is just a string like 'ifelse_circuit', 'if_circuit', etc.
            """
            match op:
                case "classical_if":
                    return f"if ({desc}):"
                case "classical_else":
                    return "else:"
                case "end_if":
                    return ""
                case _:
                    raise ValueError(f"Unsupported control flow operation: {op}")

        def _bool_logic_op(
            q_targets: tuple[str],
            c_targets: tuple[str],
            op: str,
            desc: str = "",
        ) -> str:
            condition = ""
            match op:
                case BoolOp.MATCH:
                    condition = c_targets[0]
                case BoolOp.NOT:
                    condition = f"not {c_targets[0]}"
                case BoolOp.AND:
                    condition = " and ".join(c_targets)
                case BoolOp.OR:
                    condition = " or ".join(c_targets)
                case BoolOp.XOR:
                    condition = " ^ ".join(c_targets)
                case BoolOp.NAND:
                    condition = f"not ({' and '.join(c_targets)})"
                case BoolOp.NOR:
                    condition = f"not ({' or '.join(c_targets)})"
                case _:
                    condition = ValueError(f"Unsupported BoolOp '{op}'.")
            return condition

        def _utils_op(
            q_targets: tuple[str],
            c_targets: tuple[str],
            op: str,
            desc: str = "",
        ) -> str:
            if op != "comment":
                raise ValueError(f"Unsupported utils operation: {op}")
            return f"# {desc}"

        # Map operation types to their corresponding application function
        op_type_handlers = {
            OpType.SINGLE_QUBIT: _single_qubit_op,
            OpType.TWO_QUBIT: _two_qubit_op,
            OpType.MEASUREMENT: _measurement_op,
            OpType.RESET: _single_qubit_op,
            OpType.CONTROL_FLOW: _control_flow_op,
            OpType.BOOL_LOGIC: _bool_logic_op,
            OpType.UTILS: _utils_op,
        }

        eka_to_guppy_ops = {
            "i": "",
            "x": f"{self.quantum_prefix}x",
            "y": f"{self.quantum_prefix}y",
            "z": f"{self.quantum_prefix}z",
            "h": f"{self.quantum_prefix}h",
            "t": f"{self.quantum_prefix}t",
            "phase": f"{self.quantum_prefix}s",
            "phaseinv": f"{self.quantum_prefix}sdg",
            "cnot": f"{self.quantum_prefix}cx",
            "cy": f"{self.quantum_prefix}cy",
            "cz": f"{self.quantum_prefix}cz",
            "cx": f"{self.quantum_prefix}cx",
            "swap": "swap",
            "reset": f"{self.quantum_prefix}reset",
            "reset_0": f"{self.quantum_prefix}reset",
            "reset_1": [f"{self.quantum_prefix}reset", f"{self.quantum_prefix}x"],
            "reset_+": [f"{self.quantum_prefix}reset", f"{self.quantum_prefix}h"],
            "reset_-": [
                f"{self.quantum_prefix}reset",
                f"{self.quantum_prefix}x",
                f"{self.quantum_prefix}h",
            ],
            "reset_+i": [
                f"{self.quantum_prefix}reset",
                f"{self.quantum_prefix}h",
                f"{self.quantum_prefix}s",
            ],
            "reset_-i": [
                f"{self.quantum_prefix}reset",
                f"{self.quantum_prefix}x",
                f"{self.quantum_prefix}h",
                f"{self.quantum_prefix}s",
            ],
            "measurement": f"{self.qsystem_prefix}measure_and_reset",
            "measure_z": f"{self.qsystem_prefix}measure_and_reset",
            "measure_x": [
                f"{self.quantum_prefix}h",
                f"{self.qsystem_prefix}measure_and_reset",
            ],
            "measure_y": [
                f"{self.quantum_prefix}sdg",
                f"{self.quantum_prefix}h",
                f"{self.qsystem_prefix}measure_and_reset",
            ],
        }

        return {
            op.name: partial(
                op_type_handlers[op.op_type],
                op=eka_to_guppy_ops[op.name],
            )
            for op in CLIFFORD_GATES_SIGNATURE | NONCLIFFORD_GATES_SIGNATURE
        } | {
            op.name: partial(op_type_handlers[op.op_type], op=op.name)
            for op in CONTROL_FLOW_OP_SIGNATURE
            | UTILS_SIGNATURE
            | BOOL_LOGIC_OP_SIGNATURE
        }

    def convert_circuit(self, input_circuit: Circuit) -> str:
        """Convert a Circuit to Guppy program string.

        Parameters
        ----------
        input_circuit : Circuit
            The Eka Circuit to convert.

        Returns
        -------
        str
            The Guppy program string representing the converted circuit.
        """

        if not isinstance(input_circuit, Circuit):
            raise TypeError("Input must be a Circuit")

        # Return an empty program if there are no channels or the circuit is empty.
        if (
            not input_circuit.circuit and input_circuit.name not in self.operations_map
        ) or not input_circuit.channels:
            return "", {}, {}

        init_program_line, q_registers, c_registers = self.emit_init_instructions(
            input_circuit
        )

        program_lines = []
        program_lines.extend(init_program_line.splitlines())

        circuit_instructions = self.emit_circuit_program(
            input_circuit, q_registers, c_registers
        )
        program_lines.extend(circuit_instructions.splitlines())

        # Free the quantum resources at the end of the program
        # (this is required in Guppylang)
        program_lines.append("for e in q:")
        program_lines.append(f"   {self.quantum_prefix}discard(e) ")

        return (
            "\n".join(program_lines),
            q_registers,
            c_registers,
        )

    @staticmethod
    def parse_target_run_outcome(
        outcome: Union[QsysResult, QsysShot],
    ) -> dict[str, list[int]]:
        """
        Convert a QsysResult into a dict mapping label -> list of int per shot.
        """
        result_dict = defaultdict(list)

        # Use duck typing to avoid runtime dependency on hugr types
        if hasattr(outcome, "results"):
            # QsysResult case: has 'results' attribute containing shots
            for shot in outcome.results:
                for label, value in shot.entries:
                    result_dict[label].append(value)
        elif hasattr(outcome, "entries"):
            # QsysShot case: has 'entries' attribute directly
            for label, value in outcome.entries:
                result_dict[label].append(value)
        return dict(result_dict)

    def emit_init_instructions(
        self,
        circuit: Circuit,
    ) -> tuple[str, dict[str, str], dict[str, str]]:
        """Generate Guppy code to initialize quantum and classical registers.

        Parameters
        ----------
        circuit : Circuit
            The circuit for which to initialize registers.

        Returns
        -------
        tuple[str, dict[str, str], dict[str, str]]
            A tuple containing:

            - The Guppy code string to initialize the registers.
            - A mapping from quantum channel IDs to a string of the variable name in
              the program (e.g. "q[0]").
            - A mapping from classical channel IDs to a string of the variable name in
              the program (e.g. "c[0]").

        """

        if not isinstance(circuit, Circuit):
            raise TypeError("Input must be a Circuit instance.")

        q_channels = sorted(
            [chan for chan in circuit.channels if chan.type == ChannelType.QUANTUM],
            key=lambda ch: ch.label,
        )
        qubit_id_to_idx_map = {v.id: f"q[{i}]" for i, v in enumerate(q_channels)}

        c_channels = sorted(
            [chan for chan in circuit.channels if chan.type == ChannelType.CLASSICAL],
            key=lambda ch: ch.label,
        )
        creg_id_to_idx_map = {v.id: f"c[{i}]" for i, v in enumerate(c_channels)}

        q_len = len(q_channels)
        c_len = len(c_channels)

        # pylint: disable=line-too-long
        init_code = "\n".join(
            [
                f"q = array({self.quantum_prefix}qubit() for _ in range({q_len}))",
                f"c = array(False for _ in range({c_len}))",
            ]
        )

        # Remove common leading whitespace and top/bottom empty lines
        init_code_clean = textwrap.dedent(init_code).strip()

        return init_code_clean, qubit_id_to_idx_map, creg_id_to_idx_map

    def emit_leaf_circuit_instruction(
        self,
        circuit: Circuit,
        quantum_channel_map: dict[str, str],
        classical_channel_map: dict[str, str],
    ) -> str:
        """Generate a string of the Guppy code of an given Eka instruction.

        Parameters
        ----------
        circuit : Circuit
            The Eka atomic circuit to convert to Guppy code.
        quantum_channel_map : dict[str, str]
            A mapping from quantum channel IDs to their variable names in the Guppylang
            code.
        classical_channel_map : dict[str, str]
            A mapping from classical channel IDs to their variable names in the
            Guppylang code.

        Returns
        -------
        str
            The Guppy code representation of the Eka instruction.

        Raises
        ------
        ValueError
            If the circuit is not atomic (leaf in the Eka circuit tree).
        KeyError
            If the instruction is not supported by the converter.
        """

        if not isinstance(circuit, Circuit):
            raise TypeError("Input must be a Circuit instance.")

        if circuit.circuit and len(circuit.circuit) != 0:
            raise ValueError("The circuit must be an leaf circuit with channels.")

        if circuit.name not in self.operations_map:
            raise ValueError(f"Unsupported operation: {circuit.name}")

        # Get quantum targets
        q_targets = [
            quantum_channel_map[q_chan.id]
            for q_chan in circuit.channels
            if q_chan.type == ChannelType.QUANTUM
        ]

        # Get classical targets
        c_targets = [
            classical_channel_map[c_chan.id]
            for c_chan in circuit.channels
            if c_chan.type == ChannelType.CLASSICAL
        ]

        self._validate_ops_args(circuit.name, len(q_targets), len(c_targets))

        c_label = [
            c_chan.label
            for c_chan in circuit.channels
            if c_chan.type == ChannelType.CLASSICAL
        ]

        if self.op_by_eka_name[circuit.name].op_type == OpType.MEASUREMENT:
            desc = c_label[0] if c_label else ""
        else:
            desc = circuit.description

        return self.operations_map[circuit.name](q_targets, c_targets, desc=desc)
