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

from ..eka import Circuit, Channel, IfElseCircuit
from ..eka.utilities import BoolOp


from .op_signature import (
    CLIFFORD_GATES_SIGNATURE,
    USUAL_QUANTUM_GATES,
    UTILS_SIGNATURE,
    NONCLIFFORD_GATES_SIGNATURE,
    OpSignature,
    OpType,
)
from .converter import Converter, OpAndTargetToInstrCallable

MimiqResult = tuple[Any, dict[Channel, int]]


class EkaToMimiqConverter(Converter[Channel, MimiqResult]):
    """Convert an InterpretationStep to a Mimiq circuit.

    .. code-block:: python

        import mimiqcircuits as mymc
        mimiq_alias = "mymc"
        from loom.executor import EkaToMimiqConverter

        conn = mymc.MimiqConnection()
        conn.connect("username", "pwd")

        mimiq_exec = EkaToMimiqConverter(
            mimiq_import_prefix=f"{mimiq_alias}.", circuit_varname="my_circuit"
        )

        program_str, qreg, creg = mimiq_exec.convert(interpreted_eka)

        p = (
            f"import mimiqcircuits as {mimiq_alias}\\n{program_str}"
        )

        local_ns = {}
        exec(p, {}, local_ns)

        res = local_ns["my_circuit"]

        job = conn.execute(res, algorithm="mps", nsamples=5)
        mcres = conn.get_result(job)

        parsed_outcome = EkaToMimiqConverter.parse_target_run_outcome((mcres, creg))

    Parameters
    ----------
    mimiq_import_prefix : str
        The import alias for Mimiq. defaults to "mc.".
    circuit_varname : str
        The name of the Mimiq circuit variable.
    """

    SUPPORTED_OPERATIONS: frozenset[OpSignature] = USUAL_QUANTUM_GATES | UTILS_SIGNATURE

    ALLOW_ERROR_MODELS: bool = Field(default=False, frozen=True, init=False)

    MIMIQ_CLASSICALLY_CONTROLLED_OPS: frozenset[OpSignature] = frozenset(
        {
            OpSignature(
                name=m,
                op_type=OpType.CUSTOM,
                classical_input=1,
                quantum_input=1,
            )
            for m in {
                f"classical_controlled_{n.name}" for n in CLIFFORD_GATES_SIGNATURE
            }
        }
    )

    mimiq_import_prefix: str = Field(
        default="mc.",
        frozen=True,
        init=True,
        description="The import alias for Mimiq.",
    )

    circuit_varname: str = Field(
        default="circuit",
        frozen=True,
        init=True,
        description="The name of the Mimiq circuit variable.",
    )

    @field_validator("mimiq_import_prefix")
    # pylint: disable=no-self-argument
    def validate_import_prefix(cls, v: str) -> str:
        """Ensure the import prefix ends with a dot if not empty."""
        if v and not v.endswith("."):
            v += "."
        return v

    @cached_property
    # pylint: disable=unused-argument
    def operations_map(self) -> dict[str, OpAndTargetToInstrCallable]:
        """Map of operation signatures to their corresponding Mimiq operations."""

        def _push_string(op: str, targets: list[int]) -> str:
            t = f"{', '.join(map(str, targets))}"
            return f"{self.circuit_varname}.push({self.mimiq_import_prefix}{op}, {t})"

        def _quantum_op(
            q_target: list[int],
            c_target: list[int],
            op: str | list[str],
            desc: str = "",
        ):

            if isinstance(op, list):
                return "\n".join(_push_string(o, q_target + c_target) for o in op)
            return _push_string(op, q_target + c_target)

        def _utils_op(
            q_target: list[int],
            c_target: list[int],
            op: str,
            desc: str = "",
        ):
            if op == "comment":
                return f"# {desc}"
            raise ValueError(f"Unsupported utils operation: {op}")

        def _control_flow_op(
            q_target: list[int],
            c_target: list[int],
            op: str | list[str],
            desc: str = "",
        ):
            op = eka_to_mimiq_ops[op.replace("classical_controlled_", "")]
            cond = f'{self.mimiq_import_prefix}BitString("{desc}")'

            def if_then_op(o):
                return (
                    f"{self.mimiq_import_prefix}IfStatement("
                    f"{self.mimiq_import_prefix}{o}, {cond})"
                )

            targets = f"{', '.join(map(str, q_target+c_target))}"
            if not isinstance(op, list):
                op = [op]
            return "\n".join(
                f"{self.circuit_varname}.push({if_then_op(o)}, {targets})" for o in op
            )

        # Map operation types to their corresponding function
        op_type_handlers = {
            OpType.SINGLE_QUBIT: _quantum_op,
            OpType.TWO_QUBIT: _quantum_op,
            OpType.MEASUREMENT: _quantum_op,
            OpType.RESET: _quantum_op,
            OpType.UTILS: _utils_op,
            OpType.CUSTOM: _control_flow_op,
        }

        eka_to_mimiq_ops = {
            "i": "GateID()",
            "x": "GateX()",
            "y": "GateY()",
            "z": "GateZ()",
            "h": "GateH()",
            "t": "GateT()",
            "phase": "GateS()",
            "phaseinv": "GateSDG()",
            "cnot": "GateCX()",
            "cy": "GateCY()",
            "cz": "GateCZ()",
            "cx": "GateCX()",
            "swap": "GateSWAP()",
            "reset": "ResetZ()",
            "reset_0": "ResetZ()",
            "reset_1": ["ResetZ()", "GateX()"],
            "reset_+": "ResetX()",
            "reset_-": ["ResetX()", "GateZ()"],
            "reset_+i": "ResetY()",
            "reset_-i": ["ResetY()", "GateZ()"],
            "measurement": "MeasureZ()",
            "measure_z": "MeasureZ()",
            "measure_x": "MeasureX()",
            "measure_y": "MeasureY()",
        }

        return {
            op.name: partial(op_type_handlers[op.op_type], op=eka_to_mimiq_ops[op.name])
            for op in USUAL_QUANTUM_GATES | NONCLIFFORD_GATES_SIGNATURE
        } | {
            op.name: partial(op_type_handlers[op.op_type], op=op.name)
            for op in self.MIMIQ_CLASSICALLY_CONTROLLED_OPS | UTILS_SIGNATURE
        }

    def convert_circuit(
        self,
        input_circuit: Circuit,
    ) -> tuple[str, dict[Channel, int], dict[Channel, int]]:
        """Convert a Circuit to a MimiqCircuitAndRegisterMap.

        Parameters
        ----------
        input_circuit : Circuit
            The input circuit to convert.

        Returns
        -------
        MimiqCircuitAndRegisterMap
            The converted Mimiq circuit program and register map.
        """

        if not isinstance(input_circuit, Circuit):
            raise TypeError("Input must be a Circuit")

        if (
            not input_circuit.circuit and input_circuit.name not in self.operations_map
        ) or not input_circuit.channels:
            return "# empty input", {}, {}

        mimiq_program_lines = []

        mimiq_program_lines.append("# Mimiq Program Generated from Eka")

        init_instr, q_register, c_register = self.emit_init_instructions(input_circuit)

        mimiq_program_lines.extend(init_instr.splitlines())

        mimiq_program_lines.append("")

        unrolled_circuit = Circuit.unroll(input_circuit)

        for layer in unrolled_circuit:
            for eka_op in layer:
                if eka_op.name == "if-else_circuit":
                    eka_op = self.parse_if_operation(eka_op)
                if eka_op.name not in self.operations_map:
                    raise ValueError(f"Unsupported operation: {eka_op.name}")
                mimiq_program_lines.append(
                    self.emit_leaf_circuit_instruction(eka_op, q_register, c_register)
                )

        return "\n".join(mimiq_program_lines), q_register, c_register

    @staticmethod
    def parse_target_run_outcome(
        run_output: MimiqResult,
    ) -> dict[str, int | list[int]]:
        """Parse the run output of the target language into a dictionary mapping the
        eka channel labels to boolean values measured at each shot."""
        mimiq_result, channel_to_idx = run_output
        result = mimiq_result.histogram()
        bitstrings = [key for key, count in result.items() for _ in range(count)]

        result = {chan.label: [] for chan in channel_to_idx.keys()}

        if bitstrings == []:
            raise ValueError("No bitstrings found in the run output.")

        for channel in channel_to_idx.keys():
            idx = channel_to_idx[channel]
            result[channel.label] = [int(b[idx]) for b in bitstrings]

        return result

    def emit_init_instructions(
        self, input_circuit: Circuit
    ) -> tuple[str, dict[Channel, int], dict[Channel, int]]:
        """Provide the python code (as a string) to initializes the
        quantum and classical registers, and return the mapping from eka channel to
        register."""
        if not isinstance(input_circuit, Circuit):
            raise TypeError("Input must be a Circuit instance.")

        q_channels = sorted(
            [qc for qc in input_circuit.channels if qc.is_quantum()],
            key=lambda c: c.label,
        )
        c_channels = sorted(
            [cc for cc in input_circuit.channels if cc.is_classical()],
            key=lambda c: c.label,
        )
        q_register = {qc: idx for idx, qc in enumerate(q_channels)}
        c_register = {cc: idx for idx, cc in enumerate(c_channels)}
        # Initialize the Mimiq circuit and register map
        return (
            f"{self.circuit_varname} = {self.mimiq_import_prefix}Circuit()",
            q_register,
            c_register,
        )

    def emit_leaf_circuit_instruction(
        self,
        input_circuit: Circuit,
        quantum_channel_map: dict[Channel, int],
        classical_channel_map: dict[Channel, int],
    ) -> str:
        """Provide the python code (as a string) to emit an Eka instruction in the
        target language."""

        if not isinstance(input_circuit, Circuit):
            raise TypeError("Input must be a Circuit instance.")

        if input_circuit.circuit and len(input_circuit.circuit) != 0:
            raise ValueError("The circuit must be an leaf circuit with channels.")

        if input_circuit.name not in self.operations_map:
            raise ValueError(f"Unsupported operation: {input_circuit.name}")

        q_targets = [
            str(quantum_channel_map[q_chan])
            for q_chan in input_circuit.channels
            if q_chan.is_quantum()
        ]

        c_targets = [
            classical_channel_map[c_chan]
            for c_chan in input_circuit.channels
            if c_chan.is_classical()
        ]

        if not input_circuit.name.startswith("classical_controlled_"):
            self._validate_ops_args(input_circuit.name, len(q_targets), len(c_targets))

        return self.operations_map[input_circuit.name](
            q_targets, c_targets, desc=input_circuit.description
        )

    def parse_if_operation(self, if_circuit: IfElseCircuit) -> Circuit:
        """
        Parse control flow operation, allowing only specific cases supported by MIMIQ.
        MIMIQ only supports conditional unitary operations.

        Parameters
        ----------
        if_circuit : IfElseCircuit
            The IfElseCircuit to parse.

        Returns
        -------
        Circuit
            A Circuit (gate) representing the parsed if operation in MIMIQ.
        """

        if not if_circuit.is_single_gate_conditioned:
            raise ValueError(
                "Unsupported operation for Mimiq conversion: "
                "Mimiq only supports single gate conditioned if-else circuits."
            )

        applied_unitary = if_circuit.if_circuit.circuit[0][0].name
        if applied_unitary in self.op_by_eka_name:
            if self.op_by_eka_name[applied_unitary] not in CLIFFORD_GATES_SIGNATURE:
                raise ValueError(
                    "Unsupported operation for Mimiq conversion: "
                    "Mimiq only supports single qubit unitary operations in if-else "
                    "circuits."
                )

        desc = ""
        match if_circuit.condition_circuit.name:
            case BoolOp.MATCH:
                desc = "1"
            case BoolOp.NOT:
                desc = "0"
            case BoolOp.AND:
                desc = "1" * len(if_circuit.condition_circuit.channels)
            case BoolOp.NOR:
                desc = "0" * len(if_circuit.condition_circuit.channels)
            case _:
                raise ValueError(
                    f"Unsupported bool operator for Mimiq"
                    f"conditional statement: {if_circuit.condition_circuit.name}"
                )
        return Circuit(
            name=f"classical_controlled_{applied_unitary}",
            channels=if_circuit.if_circuit.channels
            + if_circuit.condition_circuit.channels,
            description=desc,
        )
