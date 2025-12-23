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

from functools import cached_property, partial
from typing import Any
from pydantic import Field, field_validator
import numpy as np


from ..eka import Circuit
from ..eka.utilities import BoolOp

from .op_signature import (
    BOOL_LOGIC_OP_SIGNATURE,
    CONTROL_FLOW_OP_SIGNATURE,
    USUAL_QUANTUM_GATES,
    NONCLIFFORD_GATES_SIGNATURE,
    UTILS_SIGNATURE,
    OpType,
    OpSignature,
)
from .converter import Converter, OpAndTargetToInstrCallable

# Alias for the output of a PennyLane/catalyst run.
PennyLaneResult = dict[str, Any]


class EkaToPennylaneConverter(Converter[str, PennyLaneResult]):
    """Convert Eka InterpretationStep to PennyLane.

    Here's a simple example of how to use this method to execute Eka experiment with
    PennyLane:

    .. code-block:: python

        from loom.executor import EkaToPennylaneConverter

        pl_converter = EkaToPennylaneConverter(is_catalyst=False)
        pl_program, q_register, c_register = pl_converter.convert(interpreted_eka)
        n_qubits = len(q_register)

        # Indent the program body so it fits inside a function
        indented_program = "\\n    ".join(pl_program.splitlines())

        # Construct the Python program string
        s_prog = f\"\"\"
        import pennylane as qml

        def circuit():
            {indented_program}
            return {{k:qml.sample(measurements[k]) for k in measurements.keys()}}

        # Use a PennyLane device
        dev = qml.device("lightning.qubit", wires=n_qubits, shots=5)
        circ = qml.QNode(circuit, dev)

        results = circ()
        \"\"\"
        local_ns = {}
        exec(s_prog, globals(), local_ns)

        results = local_ns["results"]

        parsed_outcome = pl_converter.parse_target_run_outcome(results)

    Parameters
    ----------
    is_catalyst : bool
        Whether the PennyLane program is intended to run on Catalyst.
    import_prefix : str
        The import alias for PennyLane defaults to "qml.".
    """

    SUPPORTED_OPERATIONS: frozenset[OpSignature] = USUAL_QUANTUM_GATES

    ALLOW_ERROR_MODELS: bool = Field(default=False, frozen=True, init=False)

    # Specify whether the circuit is meant to run on catalyst or not.
    is_catalyst: bool = Field(default=False, frozen=True, init=True)

    import_prefix: str = Field(
        default="qml.",
        frozen=True,
        init=True,
        description="The import alias for PennyLane.",
    )

    @field_validator("import_prefix")
    # pylint: disable=no-self-argument
    def validate_import_prefix(cls, v: str) -> str:
        """Ensure the import prefix ends with a dot if not empty."""
        if v and not v.endswith("."):
            v += "."
        return v

    @cached_property
    # pylint: disable=unused-argument, too-many-statements
    def operations_map(
        self,
    ) -> dict[str, OpAndTargetToInstrCallable]:
        """Map Eka operations to PennyLane instructions. The map return a sequence of
        PennyLane instructions, which are specified by name, wires, classical register
        label, and whether it is a measurement or classically controlled operation."""

        def _quantum_gate_op(
            q_targets: list[int],
            c_targets: list[str],
            op: str | list[str],
            desc: str = "",
        ) -> str:
            instructions = []
            if not isinstance(op, list):
                op = [op]
            for o in op:
                if o.startswith(f"{self.import_prefix}") or o.startswith("catalyst_"):
                    prefix = ""
                else:
                    prefix = self.import_prefix
                instructions.append(f"{prefix}{o}({q_targets})")
            return "\n".join(instructions)

        def _measurement_op(
            q_targets: list[int],
            c_targets: list[str],
            op: str | list[str],
            desc: str = "",
        ) -> str:
            res = ""
            if not isinstance(op, list):
                op = [op]
            for o in op[:-1]:
                res += f"{self.import_prefix}{o}({str(q_targets[0])})\n"
            res += f"{desc} = {op[-1]}({str(q_targets[0])})"
            return res

        def _reset_op(
            q_targets: list[int],
            c_targets: list[str],
            op: str | list[str],
            desc: str = "",
        ) -> str:
            """Handle reset operations by calling measure with reset=True, followed by
            any gates."""
            instructions = []
            if not isinstance(op, list):
                op = [op]

            # The first operation should be the measure with reset=True
            measure_func = op[0]
            instructions.append(f"{measure_func}({str(q_targets[0])}, reset=True)")

            # Apply any subsequent gates
            for o in op[1:]:
                if o.startswith(f"{self.import_prefix}") or o.startswith("catalyst_"):
                    prefix = ""
                else:
                    prefix = self.import_prefix
                instructions.append(f"{prefix}{o}({q_targets})")

            return "\n".join(instructions)

        def _control_flow_op(
            q_targets: list[int],
            c_targets: list[str],
            op: str | list[str],
            desc: str = "",
        ) -> str:
            if_block_callable = "call_if_true"
            else_block_callable = "call_else"
            match op:
                case "classical_if":
                    if_instructions = f"def {if_block_callable}():"
                    return if_instructions
                case "classical_else":
                    else_instructions = f"def {else_block_callable}():"
                    return else_instructions
                case "end_if":
                    cond, is_else_present = tuple(
                        desc.split(self.separator_for_else_in_condition)
                    )
                    apply_cond = []
                    apply_cond.append(
                        f"{self.import_prefix}cond({cond}, {if_block_callable})()"
                    )
                    if is_else_present == "True":
                        apply_cond.append(
                            f"{self.import_prefix}cond(~({cond}), "
                            f"{else_block_callable})()"
                        )

                    return "\n".join(apply_cond)
                case _:
                    raise NotImplementedError(
                        f"Control flow operation '{op}' is not supported by PennyLane."
                    )

        def _bool_logic_op(
            q_targets: list[int],
            c_targets: list[str],
            op: str,
            desc: str = "",
        ):
            condition = ""
            match op:
                case BoolOp.MATCH:
                    condition = f"{c_targets[0]} == 1"
                case BoolOp.NOT:
                    condition = f"~({c_targets[0]} == 1)"
                case BoolOp.AND:
                    condition = f"({' == 1) & ('.join(c_targets)} == 1)"
                case BoolOp.OR:
                    condition = f"({' == 1) | ('.join(c_targets)} == 1)"
                case BoolOp.XOR:
                    condition = f"({' == 1) ^ ('.join(c_targets)} == 1)"
                case BoolOp.NAND:
                    condition = f"~ (({' == 1) & ('.join(c_targets)} == 1)) "
                case BoolOp.NOR:
                    condition = f"~ (({' == 1) | ('.join(c_targets)} == 1))"
                case _:
                    condition = ValueError(f"Unsupported BoolOp '{op}'.")
            return condition

        def _utils_op(
            q_targets: list[int],
            c_targets: list[str],
            op: str | list[str],
            desc: str = "",
        ) -> str:
            if op == "comment":
                return f"# {desc}"
            raise ValueError(f"Unsupported utils operation: {op}")

        # Map operation types to their corresponding function
        op_type_handlers = {
            OpType.SINGLE_QUBIT: _quantum_gate_op,
            OpType.TWO_QUBIT: _quantum_gate_op,
            OpType.MEASUREMENT: _measurement_op,
            OpType.RESET: _reset_op,
            OpType.CONTROL_FLOW: _control_flow_op,
            OpType.UTILS: _utils_op,
            OpType.BOOL_LOGIC: _bool_logic_op,
        }

        measure_op = (
            "catalyst_measure" if self.is_catalyst else f"{self.import_prefix}measure"
        )
        # PennyLane implements resets via measure(qubit, reset=True)
        # We use a separate variable for semantic clarity, though it points to the same
        # function
        reset_op = measure_op

        eka_to_pennylane_ops = {
            "i": "Identity",
            "x": "PauliX",
            "y": "PauliY",
            "z": "PauliZ",
            "t": "T",
            "h": "Hadamard",
            "phase": "S",
            "phaseinv": f"adjoint({self.import_prefix}S)",
            "cnot": "CNOT",
            "cy": "CY",
            "cz": "CZ",
            "cx": "CNOT",
            "swap": "SWAP",
            "reset": reset_op,
            "reset_0": reset_op,
            "reset_1": [reset_op, "PauliX"],
            "reset_+": [reset_op, "Hadamard"],
            "reset_-": [reset_op, "PauliX", "Hadamard"],
            "reset_+i": [reset_op, "Hadamard", "S"],
            "reset_-i": [
                reset_op,
                "PauliX",
                "Hadamard",
                "S",
            ],
            "measurement": measure_op,
            "measure_z": measure_op,
            "measure_x": ["Hadamard", measure_op],
            "measure_y": [f"adjoint({self.import_prefix}S)", "Hadamard", measure_op],
        }
        return {
            op.name: partial(
                op_type_handlers[op.op_type], op=eka_to_pennylane_ops[op.name]
            )
            for op in USUAL_QUANTUM_GATES | NONCLIFFORD_GATES_SIGNATURE
        } | {
            op.name: partial(op_type_handlers[op.op_type], op=op.name)
            for op in CONTROL_FLOW_OP_SIGNATURE
            | BOOL_LOGIC_OP_SIGNATURE
            | UTILS_SIGNATURE
        }

    def convert_circuit(
        self, input_circuit: Circuit
    ) -> tuple[str, dict[str, int], dict[str, str]]:
        """Convert an Eka Circuit to PennyLane code string along with quantum and
        classical channel maps.

        Returns
        -------
            A tuple containing:

            - The PennyLane code string representing the circuit.
            - A dictionary mapping Eka quantum channel IDs to PennyLane wire
              indices.
            - A dictionary mapping Eka classical channel IDs to PennyLane
              measurement keys.

        """
        if not isinstance(input_circuit, Circuit):
            raise TypeError("Input must be a Circuit")

        pennylane_lines = []

        pennylane_lines.append("# PennyLane Program Generated from Eka")
        if (
            not input_circuit.circuit and input_circuit.name not in self.operations_map
        ) or not input_circuit.channels:
            return "# empty input", {}, {}

        init_instr, q_register, c_register = self.emit_init_instructions(input_circuit)
        pennylane_lines.append("# Initialize Wires from Eka quantum channels Ids")
        pennylane_lines.extend(init_instr.splitlines())

        pennylane_lines.append("")

        circuit_instr = self.emit_circuit_program(
            input_circuit,
            q_register,
            c_register,
        )
        pennylane_lines.extend(circuit_instr.splitlines())

        return "\n".join(pennylane_lines), q_register, c_register

    @staticmethod
    def parse_target_run_outcome(outcome: PennyLaneResult) -> dict[str, list[int]]:
        """Parse the PennyLane/catalyst run outcome into a dictionary mapping the Eka
        classical channels labels to the measurement outcomes.
        """
        result: dict[str, list[int]] = {}
        for key, val in outcome.items():
            result[key] = [int(v) for v in np.ravel(val)]
        return result

    def emit_init_instructions(
        self, circuit: Circuit
    ) -> tuple[str, dict[str, int], dict[str, str]]:
        """Provide a mapping from eka channel label to initialized PennyLane
        wire index."""

        if not isinstance(circuit, Circuit):
            raise TypeError("Input must be a Circuit instance.")

        q_channels = sorted(
            [chan for chan in circuit.channels if chan.is_quantum()],
            key=lambda c: c.label,
        )
        q_register = {q_chan.id: n for n, q_chan in enumerate(q_channels)}
        init_dict = {q_chan.id: 1 for n, q_chan in enumerate(q_channels)}
        c_register = {
            c_chan.id: f'measurements["{c_chan.label}"]'
            for c_chan in circuit.channels
            if c_chan.is_classical()
        }

        instructions = (
            f"measurements = {{}}\n{self.import_prefix}registers({init_dict})"
        )

        return (instructions, q_register, c_register)

    def emit_leaf_circuit_instruction(
        self,
        circuit: Circuit,
        quantum_channel_map: dict[str, int],
        classical_channel_map: dict[str, str],
    ) -> str:
        """Provide the python code (as a string) to emit an Eka instruction in the
        target language."""

        if not isinstance(circuit, Circuit):
            raise TypeError("Input must be a Circuit instance.")

        if circuit.circuit and len(circuit.circuit) != 0:
            raise ValueError("The circuit must be an leaf circuit with channels.")

        q_targets = [
            quantum_channel_map[q_chan.id]
            for q_chan in circuit.channels
            if q_chan.is_quantum()
        ]

        # Get classical targets
        c_targets = [
            classical_channel_map[c_chan.id]
            for c_chan in circuit.channels
            if c_chan.is_classical()
        ]

        self._validate_ops_args(circuit.name, len(q_targets), len(c_targets))

        if self.op_by_eka_name[circuit.name].op_type == OpType.MEASUREMENT:
            desc = c_targets[0] if c_targets else ""
        else:
            desc = circuit.description

        return self.operations_map[circuit.name](
            q_targets=q_targets, c_targets=c_targets, desc=desc
        )
