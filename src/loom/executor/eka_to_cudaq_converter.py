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
from pydantic import Field

from ..eka.utilities import BoolOp
from ..eka import ChannelType, Circuit, Channel

from .converter import Converter, OpAndTargetToInstrCallable, OpSignature
from .op_signature import (
    ALL_EKA_OP_SIGNATURES,
    BOOL_LOGIC_OP_SIGNATURE,
    NONCLIFFORD_GATES_SIGNATURE,
    CONTROL_FLOW_OP_SIGNATURE,
    USUAL_QUANTUM_GATES,
    UTILS_SIGNATURE,
    OpType,
)


# Type of the output of the Convert function.
CudaqProgramAndRegister = tuple[str, dict[Channel, str], dict[Channel, str | None]]


class EkaToCudaqConverter(Converter[CudaqProgramAndRegister, Any]):
    """Convert an InterpretationStep to a CudaQ circuit."""

    SUPPORTED_OPERATIONS: frozenset[OpSignature] = ALL_EKA_OP_SIGNATURES

    ALLOW_ERROR_MODELS: bool = Field(default=False, frozen=True, init=False)

    @cached_property
    # pylint: disable=unused-argument
    def operations_map(
        self,
    ) -> dict[str, OpAndTargetToInstrCallable]:
        """Map of operation signatures to their corresponding CudaQ operations."""

        def _quantum_gate_op(
            q_target: tuple[str],
            c_target: tuple[str],
            op: str | list[str],
            desc: str = "",
        ):
            if op == "":
                return ""
            if isinstance(op, list):
                return "\n".join(
                    f"{single_op}({', '.join(q_target)})" for single_op in op
                )
            return f"{op}({', '.join(q_target)})"

        def _measurement_op(
            q_target: tuple[str],
            c_target: tuple[str],
            op: str,
            desc: str = "",
        ):
            return f"{c_target[0]} = {op}({q_target[0]}, regName='{desc}')"

        def _control_flow_op(
            q_target: tuple[str], c_target: tuple[str], op: str, desc: str = ""
        ):
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
            q_target: tuple[str],
            c_target: tuple[str],
            op: str,
            desc: str = "",
        ):
            condition = ""
            match op:
                case BoolOp.MATCH:
                    condition = c_target[0]
                case BoolOp.NOT:
                    condition = f"not {c_target[0]}"
                case BoolOp.AND:
                    condition = " and ".join(c_target)
                case BoolOp.OR:
                    condition = " or ".join(c_target)
                case BoolOp.XOR:
                    condition = " ^ ".join(c_target)
                case BoolOp.NAND:
                    condition = f"not ({' and '.join(c_target)})"
                case BoolOp.NOR:
                    condition = f"not ({' or '.join(c_target)})"
                case _:
                    condition = ValueError(f"Unsupported BoolOp '{op}'.")
            return condition

        def _utils_op(
            q_target: tuple[str],
            c_target: tuple[str],
            op: str,
            desc: str = "",
        ):
            if op != "comment":
                raise ValueError(f"Unsupported utils operation: {op}")
            return f"# {desc}"

        # Map operation types to their corresponding function
        op_type_handlers = {
            OpType.SINGLE_QUBIT: _quantum_gate_op,
            OpType.TWO_QUBIT: _quantum_gate_op,
            OpType.MEASUREMENT: _measurement_op,
            OpType.RESET: _quantum_gate_op,
            OpType.CONTROL_FLOW: _control_flow_op,
            OpType.UTILS: _utils_op,
            OpType.BOOL_LOGIC: _bool_logic_op,
        }

        eka_to_cudaq_ops = {
            "i": "",
            "x": "x",
            "y": "y",
            "z": "z",
            "h": "h",
            "t": "t",
            "phase": "s",
            "phaseinv": ["z", "s"],
            "cnot": "cx",
            "cy": "cy",
            "cz": "cz",
            "cx": "cx",
            "swap": "swap",
            "reset": "reset",
            "reset_0": "reset",
            "reset_1": ["reset", "x"],
            "reset_+": ["reset", "h"],
            "reset_-": [
                "reset",
                "x",
                "h",
            ],
            "reset_+i": [
                "reset",
                "h",
                "s",
            ],
            "reset_-i": [
                "reset",
                "x",
                "h",
                "s",
            ],
            "measurement": "mz",
            "measure_z": "mz",
            "measure_x": "mx",
            "measure_y": "my",
        }

        return {
            op.name: partial(op_type_handlers[op.op_type], op=eka_to_cudaq_ops[op.name])
            for op in USUAL_QUANTUM_GATES | NONCLIFFORD_GATES_SIGNATURE
        } | {
            op.name: partial(op_type_handlers[op.op_type], op=op.name)
            for op in CONTROL_FLOW_OP_SIGNATURE
            | UTILS_SIGNATURE
            | BOOL_LOGIC_OP_SIGNATURE
        }

    def convert_circuit(
        self,
        input_circuit: Circuit,
    ) -> CudaqProgramAndRegister:
        """Convert a Circuit to a cudaq kernel.

        Parameters
        ----------
        input_circuit : Circuit
            The input circuit to convert.

        Returns
        -------
        str
            The converted cudaq circuit program.
        dict[Channel, str]
            A dictionary mapping quantum channel to their allocated registers.
        dict[Channel, str | None]
            A dictionary mapping classical channel to their allocated registers.
            If a classical channel is not allocated, its value will be None.

        Raises
        ------
        TypeError
            If the input is not a Circuit.
        ValueError
            If the input circuit is empty or does not contain any quantum channels.
        """

        # Create a context kernel for the converter.
        if not isinstance(input_circuit, Circuit):
            raise TypeError("Input must be a Circuit")

        if (
            not input_circuit.circuit and input_circuit.name not in self.operations_map
        ) or not input_circuit.channels:
            return "# empty input", {}, {}

        # Create a kernel for the circuit.
        init_program_line, q_registers, c_registers = self.emit_init_instructions(
            input_circuit
        )

        kernel_program_lines = []
        kernel_program_lines.extend(init_program_line.splitlines())

        circuit_instructions = self.emit_circuit_program(
            input_circuit, q_registers, c_registers
        )
        kernel_program_lines.extend(circuit_instructions.splitlines())

        return (
            "\n".join(kernel_program_lines),
            q_registers,
            c_registers,
        )

    @staticmethod
    def parse_target_run_outcome(
        # cudaq.SampleResult (we avoid direct import to reduce dependencies)
        run_output,
        classical_reg_mapping: dict[Channel, str],
    ) -> dict[str, list[int]]:
        """Parse the run output of the target language into a dictionary mapping the
        eka channel labels to int values measured at each shot."""

        result_dict = {}

        # Extract measurements from each register (Channels labels are register names)
        for register_name in run_output.register_names:
            # Skip the global register which contains concatenated results
            if register_name == "__global__":
                continue

            # Get sequential measurement data for this register/channel
            sequential_data = run_output.get_sequential_data(register_name)

            # Convert string measurements to boolean values
            measurements = [int(bit) for bit in sequential_data]

            # Store in result dictionary using the EKA channel label (register name)
            # Multiple shots: return list of boolean values
            result_dict[register_name] = measurements

        map_str_to_label = {v.label: k for k, v in classical_reg_mapping.items()}

        decoded_varname = {map_str_to_label[k]: v for k, v in result_dict.items()}

        return decoded_varname

    def emit_leaf_circuit_instruction(
        self,
        circuit: Circuit,
        quantum_channel_map: dict[Channel, str],
        classical_channel_map: dict[Channel, str],
    ) -> str:
        """Emit the instruction for a leaf circuit (a circuit with no sub-circuits)."""

        if not isinstance(circuit, Circuit):
            raise TypeError("Input must be a Circuit instance.")

        if circuit.circuit and len(circuit.circuit) != 0:
            raise ValueError("The circuit must be an leaf circuit with channels.")

        # Get quantum targets
        q_targets = [
            quantum_channel_map[q_chan]
            for q_chan in circuit.channels
            if q_chan.type == ChannelType.QUANTUM
        ]

        # Get classical targets
        c_targets = [
            classical_channel_map[c_chan]
            for c_chan in circuit.channels
            if c_chan.type == ChannelType.CLASSICAL
        ]

        c_label = [
            c_chan.label
            for c_chan in circuit.channels
            if c_chan.type == ChannelType.CLASSICAL
        ]

        self._validate_ops_args(circuit.name, len(q_targets), len(c_targets))

        if self.op_by_eka_name[circuit.name].op_type == OpType.MEASUREMENT:
            desc = c_label[0] if c_label else ""
        else:
            desc = circuit.description

        if circuit.circuit or len(circuit.circuit) != 0:
            raise ValueError("The circuit must be an leaf circuit with channels.")

        if circuit.name not in self.operations_map:
            raise ValueError(f"Unsupported operation: {circuit.name}")

        return self.operations_map[circuit.name](q_targets, c_targets, desc=desc)

    def emit_init_instructions(
        self, input_circuit: Circuit
    ) -> tuple[str, dict[Channel, str], dict[Channel, str]]:
        """Provide the python or c++ code (as a string) to initializes the quantum and
        classical registers, and return the mapping from eka channel id to register."""

        if not isinstance(input_circuit, Circuit):
            raise TypeError("Input must be a Circuit instance.")
        q_channels = sorted(
            [
                chan
                for chan in input_circuit.channels
                if chan.type == ChannelType.QUANTUM
            ],
            key=lambda ch: ch.label,
        )

        c_channels = sorted(
            [
                chan
                for chan in input_circuit.channels
                if chan.type == ChannelType.CLASSICAL
            ],
            key=lambda ch: ch.label,
        )

        instructions = []

        # Allocate quantum registers for each quantum channel.
        q_registers = {c: f"q[{i}]" for i, c in enumerate(q_channels)}
        instructions.append(f"q = cudaq.qvector({len(q_channels)})")

        # Allocate classical registers for each classical channel.
        c_registers = {c: f"c{i}" for i, c in enumerate(c_channels)}

        return "\n".join(instructions), q_registers, c_registers
