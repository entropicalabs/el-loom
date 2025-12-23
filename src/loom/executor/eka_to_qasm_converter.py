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


from ..eka import Circuit, Channel
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


QasmProgram = tuple[str, dict[Channel, int], dict[Channel, int]]


class EkaToQasmConverter(Converter[QasmProgram, Any]):
    """Convert EKA circuits to OpenQASM3.0 format.
    This converter translates Eka operations into their corresponding OpenQASM3.0
    instructions, handling quantum and classical channels appropriately.

    Here's a simple example of how to use this method to execute Eka experiment in
    Qiskit through QASM conversion:

    .. code-block:: python

        from loom.executor import EkaToQasmConverter

        qasm_program, q_register, c_register = EkaToQasmConverter().convert(eka)

        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        import qiskit.qasm3

        circuit = qiskit.qasm3.loads(qasm_program)
        simulator = AerSimulator()
        qc_t = transpile(circuit, simulator)

        result = simulator.run(qc_t, shots=5).result().get_counts()

        parsed_outcome = EkaToQasmConverter.parse_target_run_outcome(
            (result, c_register)
        )

    """

    SUPPORTED_OPERATIONS: frozenset[OpSignature] = ALL_EKA_OP_SIGNATURES

    ALLOW_ERROR_MODELS: bool = Field(default=False, frozen=True, init=False)

    @cached_property
    # pylint: disable=unused-argument
    def operations_map(self) -> dict[str, OpAndTargetToInstrCallable]:
        """Map of operation signatures to their corresponding QASM instructions."""
        eka_to_qasm_ops = {
            "i": "id",
            "x": "x",
            "y": "y",
            "z": "z",
            "h": "h",
            "t": "t",
            "phase": "s",
            "phaseinv": "sdg",
            "cnot": "cx",
            "cx": "cx",
            "cy": "cy",
            "cz": "cz",
            "swap": "swap",
            "reset": "reset",
            "reset_0": "reset",
            "reset_1": ["reset", "x"],
            "reset_+": ["reset", "h"],
            "reset_-": ["reset", "x", "h"],
            "reset_+i": ["reset", "h", "s"],
            "reset_-i": ["reset", "h", "sdg"],
            "measurement": "measure",
            "measure_z": "measure",
            "measure_x": ["h", "measure"],
            "measure_y": ["sdg", "h", "measure"],
        }

        def _single_qubit_op(
            q_target: list[str],
            c_target: list[str],
            op: str | list[str],
            desc: str = "",
        ) -> str:
            if not isinstance(op, list):
                op = [op]
            qasm_ops = []
            for o in op:
                qasm_ops.append(f"{o} {q_target[0]};")
            return "\n".join(qasm_ops)

        def _two_qubit_op(
            q_target: list[str],
            c_target: list[str],
            op: str,
            desc: str = "",
        ) -> str:
            return f"{op} {q_target[0]}, {q_target[1]};"

        def _measurement_op(
            q_target: list[str],
            c_target: list[str],
            op: str | list[str],
            desc: str = "",
        ) -> list[str]:
            qasm_ops = []
            if not isinstance(op, list):
                op = [op]
            for o in op[:-1]:
                qasm_ops.append(f"{o} {q_target[0]};")
            qasm_ops.append(f"{op[-1]} {q_target[0]} -> {c_target[0]};")
            return "\n".join(qasm_ops)

        def _control_flow_op(
            q_target: list[str], c_target: list[str], op: str, desc: str = ""
        ) -> str:
            """Handling control flow operations in QASM. c_target[0] is the classical
            condition instruction string already formatted."""
            match op:
                case "classical_if":
                    return f"if ({desc}) {{"
                case "classical_else":
                    return "} else {"
                case "end_if":
                    return "}"
                case _:
                    raise ValueError(f"Unsupported control flow operation: {op}")

        def _bool_logic_op(
            q_target: list[str], c_target: list[str], op: str, desc: str = ""
        ) -> str:
            """Handling boolean logic operations in QASM. c_target contains the
            classical channel targets."""
            condition = ""
            match op:
                case BoolOp.MATCH:
                    condition = c_target[0]
                case BoolOp.NOT:
                    condition = f"!{c_target[0]}"
                case BoolOp.AND:
                    condition = " == 1 && ".join(c_target) + " == 1"
                case BoolOp.OR:
                    condition = " == 1 || ".join(c_target) + " == 1"
                case BoolOp.XOR:
                    condition = " == 1 ^ ".join(c_target) + " == 1"
                case BoolOp.NAND:
                    condition = f"!({' == 1 && '.join(c_target)} == 1)"
                case BoolOp.NOR:
                    condition = f"!({' == 1 || '.join(c_target)} == 1)"
                case _:
                    condition = ValueError(f"Unsupported BoolOp '{op}'.")
            return condition

        def _utils_op(
            q_target: list[str], c_target: list[str], op: str, desc: str = ""
        ) -> str:
            """Handling utility operations in QASM."""
            match op:
                case "comment":
                    return f"// {desc}"
                case _:
                    return None

        # Map operation types to their corresponding function
        op_type_handlers = {
            OpType.SINGLE_QUBIT: _single_qubit_op,
            OpType.TWO_QUBIT: _two_qubit_op,
            OpType.MEASUREMENT: _measurement_op,
            OpType.RESET: _single_qubit_op,
            OpType.CONTROL_FLOW: _control_flow_op,
            OpType.BOOL_LOGIC: _bool_logic_op,
            OpType.UTILS: _utils_op,
        }

        return {
            # Quantum gate instructions
            op.name: partial(op_type_handlers[op.op_type], op=eka_to_qasm_ops[op.name])
            for op in CLIFFORD_GATES_SIGNATURE | NONCLIFFORD_GATES_SIGNATURE
        } | {
            # Bool op and control flow mapping are less straightforward so they are
            # handled in the handlers methods
            op.name: partial(op_type_handlers[op.op_type], op=op.name)
            for op in BOOL_LOGIC_OP_SIGNATURE
            | CONTROL_FLOW_OP_SIGNATURE
            | UTILS_SIGNATURE
        }

    def convert_circuit(self, input_circuit: Circuit) -> QasmProgram:
        """
        Convert an Eka circuit to OpenQASM3.0 format.

        Parameters
        ----------
        input_data : Circuit
            The Eka circuit to convert.

        Returns
        -------
        QasmProgram (tuple[str, dict[Channel, int], dict[Channel, int]])
            A tuple containing the QASM program as a string, a mapping from quantum
            channels to their indices, and a mapping from classical channels to their
            indices.
        """
        # List, where each element is a line in the QASM program
        qasm_str_program = ["OPENQASM 3.0;", 'include "stdgates.inc";']

        # pylint: disable=unsupported-membership-test
        if (
            not input_circuit.circuit and input_circuit.name not in self.operations_map
        ) or not input_circuit.channels:
            return "# empty input", {}, {}

        init_instructions, quant_ch_to_idx, classical_ch_to_idx = (
            self.emit_init_instructions(input_circuit)
        )
        qasm_str_program.append("// Init registers")
        qasm_str_program.extend(init_instructions.splitlines())

        circuit_instructions = self.emit_circuit_program(
            input_circuit, quant_ch_to_idx, classical_ch_to_idx
        )

        qasm_str_program.extend(circuit_instructions.splitlines())

        return (
            "\n".join(qasm_str_program),
            quant_ch_to_idx,
            classical_ch_to_idx,
        )

    @staticmethod
    def parse_target_run_outcome(
        run_output: tuple[dict[str, int], dict[Channel, int]],
    ) -> dict[str, int | list[int]]:
        """
        Parse the output of a target run by mapping the bitstring to the register
        labels.

        Parameters
        ----------
        run_output : tuple[dict[str, int], dict[Channel, int]]
            Output of the simulation run as a tuple of:

            - bitstrings: dict with keys as bitstrings and values as their counts,
            - channel_to_idx: a mapping from channels to their bitstring indices,
              this mapping is given by the converter when exporting the circuit.

        Returns
        -------
        dict[str, int | list[int]]
            dict mapping register labels to their corresponding values (a list where
            each element represents the value of the register in a specific shot).
        """
        bitstrings, channel_to_idx = run_output

        bitstrings = [key for key, count in bitstrings.items() for _ in range(count)]
        # Implement the parsing logic here
        result = {chan.label: [] for chan in channel_to_idx.keys()}

        if bitstrings == []:
            raise ValueError("No bitstrings found in the run output.")
        num_shots = len(bitstrings[0])

        for channel in channel_to_idx.keys():
            idx = channel_to_idx[channel]
            # Qiskit returns bitstrings in reverse order
            reversed_index = num_shots - 1 - idx
            result[channel.label] = [int(b[reversed_index]) for b in bitstrings]

        return result

    def emit_init_instructions(
        self, circuit: Circuit
    ) -> tuple[str, dict[Channel, Any], dict[Channel, Any]]:
        """Provide the python code to initializes the
        quantum and classical registers, and return the mapping from eka channel to
        register index."""

        if not isinstance(circuit, Circuit):
            raise TypeError("Input must be a Circuit instance.")

        if circuit.circuit == [] and circuit.channels == []:
            return "", {}, {}

        q_channels = sorted(
            [channel for channel in circuit.channels if channel.is_quantum()],
            key=lambda ch: ch.label,
        )

        c_channels = sorted(
            [channel for channel in circuit.channels if channel.is_classical()],
            key=lambda ch: ch.label,
        )

        qasm_str_program = []
        # Init registers
        if len(q_channels) > 0:
            qasm_str_program.append(f"qubit[{len(q_channels)}] q;")
        if len(c_channels) > 0:
            qasm_str_program.append(f"bit[{len(c_channels)}] c;")

        quant_ch_to_idx = {ch: idx for idx, ch in enumerate(q_channels)}
        classical_ch_to_idx = {ch: idx for idx, ch in enumerate(c_channels)}

        return "\n".join(qasm_str_program), quant_ch_to_idx, classical_ch_to_idx

    def emit_leaf_circuit_instruction(
        self,
        circuit: Circuit,
        quantum_channel_map: dict[Channel, int],
        classical_channel_map: dict[Channel, int],
    ) -> str:
        """Provide the python code to emit an Eka instruction in the target
        language."""

        if not isinstance(circuit, Circuit):
            raise TypeError("Input must be a Circuit instance.")

        if circuit.circuit and len(circuit.circuit) != 0:
            raise ValueError("The circuit must be an leaf circuit with channels.")

        op_name = circuit.name

        q_target = [
            f"q[{quantum_channel_map[qt]}]"
            for qt in circuit.channels
            if qt.is_quantum()
        ]

        c_target = [
            f"c[{classical_channel_map[ct]}]"
            for ct in circuit.channels
            if ct.is_classical()
        ]

        self._validate_ops_args(op_name, len(q_target), len(c_target))

        # pylint: disable=unsupported-membership-test
        if op_name not in self.operations_map:
            raise KeyError(f"Operation {op_name} not supported in QASM target.")

        return self.operations_map[op_name](
            q_target, c_target, desc=circuit.description
        )
