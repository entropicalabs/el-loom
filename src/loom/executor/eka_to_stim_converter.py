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

import ast
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property, partial
from typing import Any, Callable
import traceback

from numpy.typing import NDArray
import numpy as np
from pydantic import Field
import stim


from ..eka import Circuit, Channel, IfElseCircuit
from ..interpreter import InterpretationStep

from .op_signature import (
    CLIFFORD_GATES_SIGNATURE,
    OpSignature,
    OpType,
)
from .converter import Converter
from .circuit_error_model import CircuitErrorModel, ErrorType, ApplicationMode

# pylint: disable=no-member

# Type of the output of the Convert function.
# stim.Circuit and dictionaries mapping classical channel to stim measurement record
# indices and quantum channel to stim qubit coordinate and target.
StimCircuitAndRegisters = tuple[
    stim.Circuit, dict[Channel, tuple[str, stim.GateTarget]], dict[Channel, int]
]


StimCallable = Callable[
    [
        list[
            list[stim.GateTarget]
        ],  # List of targets for one stim instruction (e.g. [[0,1], [2,3]] for two CX)
    ],
    list[str],  # Stim instructions strings
]
StimCallable.__doc__ = """
    Type alias that describes a callable that maps an Eka op to stim instruction strings 
    with targets.
    Takes a list of list of targets, allowing multiple instructions of the 
    same type to be grouped.
    For example, two CX gates on different pair of targets:
    [[q1, c1], [q2, c2]] → "CX q1 c1 q2 c2".
    Returns a list of stim instruction strings with targets, since one Eka instruction
    can correspond to multiple stim instructions.
"""

# Type of the run result from stim.sample (NDArray of shape (shots, num_qubits)),
# where each entry is a boolean, holding the measurement outcome of the latest measure
# of each qubits.
StimOutputRunResult = tuple[
    NDArray[np.bool], dict[Channel, tuple[str, stim.GateTarget]]
]


class EkaToStimConverter(Converter[StimCircuitAndRegisters, StimOutputRunResult]):
    """
    Convert an InterpretationStep to a stim.Circuit.

    Here's a simple example of how to use this method to execute Eka experiment in
    Stim:

    .. code-block:: python

        from loom.executor import EkaToStimConverter

        stim_converter = EkaToStimConverter()
        stim_program, quant_register, class_register = stim_converter.convert(
            eka
        )

        import stim
        st = stim_program
        sampler = st.compile_sampler()
        result = sampler.sample(shots=5)

        print(stim_converter.parse_target_run_outcome((result, class_register)))

    """

    # Support only Clifford operations
    SUPPORTED_OPERATIONS: frozenset[OpSignature] = CLIFFORD_GATES_SIGNATURE

    SUPPORTED_ERROR_TYPES: frozenset[ErrorType] = frozenset(
        {
            ErrorType.PAULI_X,
            ErrorType.PAULI_Y,
            ErrorType.PAULI_Z,
            ErrorType.PAULI_CHANNEL,
            ErrorType.BIT_FLIP,
            ErrorType.PHASE_FLIP,
            ErrorType.DEPOLARIZING1,
            ErrorType.DEPOLARIZING2,
        }
    )

    ALLOW_ERROR_MODELS: bool = Field(default=True, frozen=True, init=False)

    # Special classically controlled operations supported by Stim, only allow
    # conditional pauli operations.
    STIM_CLASSICALLY_CONTROLLED_OPS: frozenset[str] = frozenset(
        {
            OpSignature(
                name=n,
                op_type=OpType.CUSTOM,
                classical_input=1,
                quantum_input=1,
            )
            for n in {
                f"classical_controlled_{applied_pauli}"
                for applied_pauli in ["x", "y", "z"]
            }
        }
    )

    @cached_property
    # pylint: disable=unused-argument
    def operations_map(
        self,
    ) -> dict[str, StimCallable]:
        """Map of operation signatures to their corresponding StimCallable, which is a
        set of stim instructions (as strings) associated with values for the targets and
        parameters."""

        def _make_stim_instruction_string(
            name: str,
            targets: list[list[stim.GateTarget]],
            gate_args: list[float] | None = None,
        ) -> str:
            """
            Construct a Stim instruction string for a gate.

            Arguments
            ---------
            name : str
                Gate name (e.g., "X", "H", "CX").
            targets : list[list[stim.GateTarget]]
                Nested list of stim.GateTarget to be flattened.
            gate_args : list[float] | None
                Optional list of float parameters.

            Returns
            -------
            str
                A single Stim instruction string.
            """
            # Flatten targets and convert to Stim string representation
            flattened_targets = [t for sublist in targets for t in sublist]
            targets_str = " ".join(str(t.value) for t in flattened_targets)

            # Handle optional gate arguments
            if gate_args:
                args_str = ", ".join(map(str, gate_args))
                return f"{name}({args_str}) {targets_str}".strip()
            return f"{name} {targets_str}".strip()

        def _handle_op(
            targets: list[list[stim.GateTarget]],
            op,
            gate_args: list[float] | None = None,
        ) -> list[str]:
            """Sort the targets and call the corresponding operation handler.
            Also handle the case where op is a tuple of operations by creating a list of
            instructions.
            """
            if isinstance(op, tuple):
                return [
                    _make_stim_instruction_string(o, targets, gate_args=gate_args)
                    for o in op
                ]
            return [_make_stim_instruction_string(op, targets, gate_args=gate_args)]

        def _control_flow_op(
            targets: list[list[stim.GateTarget]],
            op: str,
            gate_args: list[float] | None = None,
        ) -> list[str]:
            """Handle control flow operations in stim. Only allow conditional paulis.
            targets are used as follow: [[control, target]]

            """
            pauli = op.replace("classical_controlled_", "")
            return [f"C{pauli.upper()} rec[{targets[0][0]}] {str(targets[0][1].value)}"]

        eka_to_stim_ops = {
            "i": "I",
            "hadamard": "H",
            "h": "H",
            "phase": "S",
            "phaseinv": "S_DAG",
            "x": "X",
            "y": "Y",
            "z": "Z",
            "cx": "CX",
            "cnot": "CX",
            "cz": "CZ",
            "cy": "CY",
            "swap": "SWAP",
            "reset": "R",
            "reset_0": "R",
            "reset_1": ("R", "X"),
            "reset_+": "RX",
            "reset_-": ("RX", "Z"),
            "reset_+i": "RY",
            "reset_-i": ("RY", "Z"),
            "measurement": "M",
            "measure_z": "M",
            "measure_x": "MX",
            "measure_y": "MY",
            ErrorType.PAULI_X.label: "X_ERROR",
            ErrorType.PAULI_Y.label: "Y_ERROR",
            ErrorType.PAULI_Z.label: "Z_ERROR",
            ErrorType.BIT_FLIP.label: "X_ERROR",
            ErrorType.PHASE_FLIP.label: "Z_ERROR",
            ErrorType.PAULI_CHANNEL.label: "PAULI_CHANNEL_1",
            ErrorType.DEPOLARIZING1.label: "DEPOLARIZE1",
            ErrorType.DEPOLARIZING2.label: "DEPOLARIZE2",
        }

        return (
            {
                op.name: partial(_handle_op, op=eka_to_stim_ops[op.name])
                for op in CLIFFORD_GATES_SIGNATURE
            }
            | {
                error_type.label: partial(  # type: ignore
                    _handle_op, op=eka_to_stim_ops[error_type.label]
                )
                for error_type in self.SUPPORTED_ERROR_TYPES
            }
            | {
                op.name: partial(_control_flow_op, op=op.name)
                for op in self.STIM_CLASSICALLY_CONTROLLED_OPS
            }
        )

    @staticmethod
    def parse_if_operation(if_circuit: IfElseCircuit) -> Circuit:
        """
        Parse control flow operation, allowing only specific cases supported by Stim.
        Stim only supports conditional pauli operations, with single bit conditions.

        Parameters
        ----------
        if_circuit : IfElseCircuit
            The IfElseCircuit to parse.

        Returns
        -------
        Circuit
            A Circuit (gate) representing the parsed if operation in Stim, which is a
            CX (resp. CY or CZ) operation with a classical channel as control.
        """
        if not if_circuit.is_condition_single_bit:
            raise ValueError(
                "Unsupported operation for Stim conversion: "
                "Stim only supports single bit conditions for if-else circuits."
            )

        if not if_circuit.is_single_gate_conditioned:
            raise ValueError(
                "Unsupported operation for Stim conversion: "
                "Stim only supports single gate conditioned if-else circuits."
            )

        applied_pauli = if_circuit.if_circuit.circuit[0][0].name
        if applied_pauli not in {
            "x",
            "y",
            "z",
        }:
            raise ValueError(
                "Unsupported operation for Stim conversion: "
                "Stim only supports conditional pauli operations in if-else circuits."
            )

        return Circuit(
            name=f"classical_controlled_{applied_pauli}",
            channels=[
                if_circuit.condition_circuit.channels[0],
                if_circuit.if_circuit.channels[0],
            ],
        )

    # pylint: disable=too-many-locals, too-many-nested-blocks, too-many-statements,
    # pylint: disable=too-many-branches
    def emit_circuit_program(
        self,
        input_circuit: Circuit,
        error_models: list[CircuitErrorModel] | None = None,
        with_ticks: bool = False,
    ) -> tuple[
        str, dict[Channel, tuple[str, stim.GateTarget]], dict[Channel, int], int
    ]:
        """Convert a Circuit to a text in stim format and
        a mapping of channel labels to stim target or measurement record indices.

        Parameters
        ----------
        input_circuit : Circuit
            The EKA circuit to convert.
        with_ticks : bool, optional
            Whether to include TICK instructions in the stim circuit, by default False.

        Returns
        -------
        str, dict[Channel, tuple[str, stim.GateTarget]], dict[Channel, int]
            A tuple containing the stim circuit as a string, a dictionary mapping
            quantum channel ids to stim qubit targets and coordinates and a dictionary
            mapping classical channel ids to stim measurement record indices,
        """
        if not isinstance(input_circuit, Circuit):
            raise TypeError("Input must be a Circuit instance.")

        # Initialize an empty stim circuit
        final_instructions = []

        # If the circuit is empty, return the empty stim circuit and empty mappings
        if (
            not input_circuit.channels
            or not input_circuit.circuit
            or len(input_circuit.circuit) == 0
        ):
            return stim.Circuit("\n".join(final_instructions)), {}, {}, 0

        init_instructions, q_chan_to_stim_qubits_map, c_channel_to_rec_idx = (
            self.emit_init_instructions(input_circuit)
        )

        for init_instr in init_instructions.split("\n"):
            if init_instr:  # avoid adding empty lines
                final_instructions.append(init_instr)

        @dataclass(frozen=True)
        class NoiseArgsKey:
            """This class is used to provide a hashable key for information that defines
            a noise instruction."""

            e_type: ErrorType
            a_mode: ApplicationMode
            prob: tuple[float]

        # For ease of access, define the set of classically controlled op names
        classically_controlled_op_names = {
            o.name for o in self.STIM_CLASSICALLY_CONTROLLED_OPS
        }

        # Group operations (=gates) in a layer (tick) by their type
        # stim_instructions_by_layer is a list of dicts, where each dict corresponds to
        # a tick, and the dict map operations name to their targets
        stim_instructions_by_layer: list[
            dict[
                tuple[str, tuple[NoiseArgsKey, ...]],
                list[tuple[list[stim.GateTarget], list[Channel]]],
            ]
        ] = []

        for layer in Circuit.unroll(input_circuit):

            # dict mapping (op_name, *NoiseArgsKey) to list of quantum targets and
            # classical channels
            # This is useful to group operations of the same type in one stim
            # instruction, Including associated noise instructions (if any).
            op_dict = defaultdict(list)
            for op in layer:
                if op.name == "if-else_circuit":
                    op = self.parse_if_operation(op)
                if op.name not in self.operations_map:
                    raise KeyError(f"Operation {op.name} not found in operations map.")
                classical_channels = [c for c in op.channels if c.is_classical()]
                targets = [
                    q_chan_to_stim_qubits_map[c][1]
                    for c in op.channels
                    if c.is_quantum()
                ]

                # Validation of operation arguments
                if op.name in classically_controlled_op_names:
                    if len(classical_channels) != 1 or len(targets) != 1:
                        raise ValueError(
                            f"Classically controlled operation {op.name} must have one "
                            "classical control channel and one quantum target channel."
                        )
                else:
                    self._validate_ops_args(
                        op.name, len(targets), len(classical_channels)
                    )

                noises = []
                for cem in error_models or []:
                    noise_args = cem.get_gate_error_probability(op)
                    if noise_args:
                        noises.append(
                            NoiseArgsKey(
                                e_type=cem.error_type,
                                a_mode=cem.application_mode,
                                prob=tuple(noise_args),
                            )
                        )

                # Group by operation name and associated noise instructions
                op_dict[(op.name, tuple(noises))].append((targets, classical_channels))
            # Append the dict of operations for this layer
            stim_instructions_by_layer.append(op_dict)

        # Keep track of the measurement record index to map them to classical channels
        measurement_record_counter = 0
        tick_count = 0
        # Build the stim circuit from the grouped operations
        for s_layer in stim_instructions_by_layer:
            for ops, targets_and_channels in s_layer.items():
                targets, classical_channels = zip(*targets_and_channels)
                # here the op is a tuple of (op_name, *NoiseArgsKey)
                # unpack it to get the op_name and noise_args
                op, noise_instructions = ops

                # Handle classically controlled operations and measurement operations
                if op in classically_controlled_op_names:
                    # Intertwine the classical target with the quantum targets for
                    # classically controlled operations
                    classical_targets = [
                        [
                            c_channel_to_rec_idx[chan] - measurement_record_counter
                            for chan in chan_list
                        ]
                        for chan_list in classical_channels
                    ]
                    # Set quantum target to be:
                    # [
                    #   [classical_targets[0], quantum_targets[0]],
                    #   [classical_targets[1], quantum_targets[1]],
                    #  ...
                    # ]
                    targets = [
                        [t_pair[0][0], t_pair[1][0]]
                        for t_pair in zip(classical_targets, targets)
                    ]
                elif self.op_by_eka_name[op].op_type == OpType.MEASUREMENT:
                    # Map classical channels to measurement record indices
                    # for measurement operations
                    for c in classical_channels:
                        c_channel_to_rec_idx[c[0]] = measurement_record_counter
                        measurement_record_counter += 1

                # Get the stim instructions for this operation
                stim_instructions = self.operations_map[op](targets)
                # Add noise instructions associated with this operation
                for noise in noise_instructions:
                    if noise.a_mode == ApplicationMode.BEFORE_GATE:
                        stim_instructions = (
                            self.operations_map[noise.e_type.label](
                                targets=targets, gate_args=noise.prob
                            )
                            + stim_instructions
                        )
                    if noise.a_mode == ApplicationMode.AFTER_GATE:
                        stim_instructions = stim_instructions + self.operations_map[
                            noise.e_type.label
                        ](targets=targets, gate_args=noise.prob)

                final_instructions.extend(stim_instructions)

            # Add tick-wise noise if error model is provided
            noise_instruction = []
            for cem in error_models or []:
                if cem.application_mode == ApplicationMode.END_OF_TICK:
                    noise_args = cem.get_tick_error_probability(tick_count)
                    noise_instruction += self.operations_map[cem.error_type.label](
                        targets=[
                            [
                                value[1]
                                for key, value in q_chan_to_stim_qubits_map.items()
                            ]
                        ],
                        gate_args=noise_args,
                    )
                if cem.application_mode == ApplicationMode.IDLE_END_OF_TICK:
                    # group all the target with the same error probabilities together
                    idle_args = {}
                    for c, stim_reg in q_chan_to_stim_qubits_map.items():
                        args = cem.get_idle_tick_error_probability(tick_count, c.id)
                        if args:
                            if tuple(args) in idle_args:
                                idle_args[tuple(args)].append(stim_reg[1])
                            else:
                                idle_args[tuple(args)] = [stim_reg[1]]
                    for args, targets in idle_args.items():
                        noise_instruction += self.operations_map[cem.error_type.label](
                            targets=[targets], gate_args=args
                        )

            final_instructions.extend(noise_instruction)

            tick_count += 1
            # Append tick instruction for every eka layer
            if with_ticks is True:
                final_instructions.append("TICK")

        return (
            final_instructions,
            q_chan_to_stim_qubits_map,
            c_channel_to_rec_idx,
            measurement_record_counter,
        )

    def convert_circuit(
        self,
        input_circuit: Circuit,
        error_models: list[CircuitErrorModel] | None = None,
        with_ticks: bool = False,
    ):
        """Convert a Circuit to a text in stim format and
        a mapping of channel labels to stim target or measurement record indices.

        Parameters
        ----------
        input_circuit : Circuit
            The EKA circuit to convert.
        with_ticks : bool, optional
            Whether to include TICK instructions in the stim circuit, by default False.

        Returns
        -------
        StimCircuitAndRegisters
            A tuple containing the stim circuit, a dictionary mapping quantum channel
            ids to stim qubit targets and coordinates and a dictionary mapping
            classical channel ids to stim measurement record indices,
        """
        if not input_circuit.circuit or len(input_circuit.circuit) == 0:
            return "", {}, {}
        circ_text, quant_dict, rec_dict, _ = self.emit_circuit_program(
            input_circuit, error_models=error_models, with_ticks=with_ticks
        )
        return stim.Circuit("\n".join(circ_text)), quant_dict, rec_dict

    def convert(
        self,
        interpreted_eka: InterpretationStep,
        error_models: list[CircuitErrorModel] | None = None,
        with_ticks: bool = False,
    ) -> StimCircuitAndRegisters:
        """
        Convert a InterpretationStep.
        Call the default convert method to convert the final circuit of the step.
        Then, add the detectors and logical observables to the stim circuit.
        """

        # Convert the final circuit of the interpretation step
        input_circuit = interpreted_eka.final_circuit
        if not input_circuit.circuit or len(input_circuit.circuit) == 0:
            return stim.Circuit(), {}, {}
        stim_circuit_text, quant_dict, rec_dict, measurement_counter = (
            self.emit_circuit_program(
                input_circuit, error_models=error_models, with_ticks=with_ticks
            )
        )
        # Get and add detectors to the stim circuit
        detector_instructions = EkaToStimConverter.emit_detectors_and_observables(
            interpreted_eka, rec_dict, measurement_counter
        )
        if detector_instructions:
            stim_circuit_text.extend(detector_instructions)

        return stim.Circuit("\n".join(stim_circuit_text)), quant_dict, rec_dict

    @staticmethod
    def parse_target_run_outcome(
        output: StimOutputRunResult,
    ) -> dict[str, int | list[int]]:
        """
        Parse the run output of a stim circuit into a dictionary mapping the
        Eka classical channels labels to their measurement outcomes.

        Parameters
        ----------
        output : StimOutputRunResult
            The output from stim simulation, which is a tuple of
            (NDArray of shape (shots, num_qubits), dict mapping classical channel to
            stim measurement record index).

        Returns
        -------
        dict[Channel, int | list[int]]
            A dictionary mapping the Eka classical channel labels to their measurement
            outcomes. If multiple shots were run, the outcomes are lists of integers
            (0 or 1), otherwise a single integer (0 or 1).
        """
        output, c_reg = output
        res: dict[Channel, int | list[int]] = {}
        n_shots = output.shape[0]

        for key, idx in c_reg.items():
            values = output[:, idx].astype(int).tolist()
            # Flatten to int if only one shot
            res[key.label] = values[0] if n_shots == 1 else values

        return res

    @staticmethod
    # pylint: disable=too-many-return-statements
    def _eka_to_stim_coordinates(
        eka_coord: tuple[int, ...] | str,
    ) -> int | tuple[float, ...]:
        """
        Convert EKA coordinates to Stim coordinates. If possible (i.e. if the label of
        the quantum channel
        is in the form of (x, y, z) or (x, y)), otherwise return 0
        """
        if isinstance(eka_coord, tuple):
            coords = eka_coord
        else:
            try:
                coords = ast.literal_eval(eka_coord)
            except (ValueError, SyntaxError):
                coords = eka_coord  # treat it as a raw string

        if not isinstance(coords, tuple):
            return 0

        match len(coords):
            case 2:
                # For linear lattice codes
                if coords[1] == 1:
                    return (coords[0], 0)
                if coords[1] == 0:
                    return (coords[0] - 0.5, 0.5)
                # Removed error and kept this temporarily to host HGP codes until
                # Lattice refactor
                return coords
            case 3:
                # For square lattice codes
                if coords[2] == 1:
                    return (coords[0], coords[1])
                if coords[2] == 0:
                    return (coords[0] + 0.5, coords[1] + 0.5)
                # Patched up case for proper handling of Color Codes until Lattice
                # refactor
                if coords[2] == 2:
                    return (coords[0] + 0.05, coords[1] + 0.05)
                raise ValueError(
                    f"Invalid coordinate {coords}. "
                    "Coordinates should be in the form (x, y, 0), (x, y, 1)"
                    " or (x, y, 2)."
                )
            case _:
                # For other lattice codes raise an error
                raise ValueError(f"Invalid channel label {eka_coord}.")

    @staticmethod
    def emit_detectors_and_observables(
        input_istep: InterpretationStep,
        rec_dict: dict[str, int],
        measurement_offset: int = 0,
    ) -> list[str]:
        """Get the detector instructions for the given interpretation step, and the
        given measurement offset which corresponds to the number of measurements in the
        circuit before adding detectors.

        Parameters
        ----------
        input_istep : InterpretationStep
            The interpretation step to get detector instructions for.
        rec_dict : dict[str, int]
            A mapping from channel labels to their corresponding record indices.
        measurement_offset : int, optional
            An offset to apply to the measurement indices, by default 0

        Returns
        -------
        list[str]
            A list of stim circuit instructions for the detector measurements.

        Raises
        ------
        ValueError
            If the detector labels are not in the expected format.
        """
        detector_instructions = []
        # Add detectors to the stim circuit
        for det in input_istep.detectors:
            # Find the channels corresponding to the measurements in the detector
            # According to some obscure convention.
            channels_in_det = [
                channel
                for meas in det.measurements
                for channel in input_istep.final_circuit.channels
                if isinstance(meas, tuple) and channel.label == f"{meas[0]}_{meas[1]}"
                # ignore constant cbits with the isinstance check
            ]

            # substract the number of measurements in the circuit to get the
            # correct indices in the stim circuit (which track records with negative
            # indices, -1 is the latest measurement, -2 the one before, etc.)
            targets = [
                f"rec[{rec_dict[chan] - measurement_offset}]"
                for chan in channels_in_det
            ]

            # Eze's shinegan, I have no idea why this works, this follows obscure
            # conventions again.
            det_args = []
            try:
                if (
                    "space_coordinates" in det.labels
                    and "time_coordinate" in det.labels
                ):
                    det_args = list(
                        EkaToStimConverter._eka_to_stim_coordinates(
                            det.labels.get("space_coordinates")
                        )
                        + det.labels.get("time_coordinate")
                    )
                if "color" in det.labels:
                    det_args.append(det.labels.get("color"))

            except Exception as e:
                traceback.print_exc()
                raise e
            if det_args:
                det_args = [str(arg) for arg in det_args]
                str_instruction = f"DETECTOR({', '.join(det_args)}) {' '.join(targets)}"
            else:
                str_instruction = f"DETECTOR {' '.join(targets)}"

            detector_instructions.append(str_instruction)

        # Add observables to the stim circuit
        for i, logical_observable in enumerate(input_istep.logical_observables):
            # ignore constant cbits with the isinstance check
            target_channel = [
                channel
                for meas in logical_observable.measurements
                for channel in input_istep.final_circuit.channels
                if isinstance(meas, tuple) and channel.label == f"{meas[0]}_{meas[1]}"
            ]

            # Get the targets in stim format from the measurement record indices
            targets = [
                f"rec[{rec_dict[chan] - measurement_offset}]" for chan in target_channel
            ]
            detector_instructions.append(
                f"OBSERVABLE_INCLUDE({str(i)}) {' '.join(targets)}"
            )

        return detector_instructions

    def emit_init_instructions(
        self, circuit: Circuit
    ) -> tuple[str, dict[Channel, Any], dict[Channel, Any]]:
        """Provide the python code (as a string) to initializes the
        quantum and classical registers, and return the mapping from eka channel to
        register.

        In the beginning of the circuit, the mapping of classical channels to
        measurement record indices is None, since no classical channels have been
        measured yet."""

        if not isinstance(circuit, Circuit):
            raise TypeError("Input must be a Circuit instance.")

        instructions = []

        # Sort quantum channels by label to ensure consistent qubit ordering across runs
        q_channels = sorted(
            [c for c in circuit.channels if c.is_quantum()],
            key=lambda x: x.label,
        )

        # Map of EKA qubits to Stim qubits idx and coordinates
        q_chan_to_stim_qubits_map = {}

        # Create mapping of Eka quantum channels to stim qubit indices
        # and add QUBIT_COORDS instructions to the stim circuit if coordinates
        # are provided.
        for i, chan in enumerate(q_channels):
            stim_coords = EkaToStimConverter._eka_to_stim_coordinates(chan.label)
            if stim_coords != 0:
                q_chan_to_stim_qubits_map[chan] = (stim_coords, stim.GateTarget(i))
                instructions.append(f"QUBIT_COORDS{stim_coords} {i}")
            else:
                # If the label is not a coordinate, just map to the qubit index
                q_chan_to_stim_qubits_map[chan] = (None, stim.GateTarget(i))

        # Create mapping of Eka classical channels to stim measurement record indices
        # (None if no classical channels have not been measured yet)
        c_channels = [c for c in circuit.channels if c.is_classical()]
        c_channel_to_rec_idx = {c: None for c in c_channels}

        return (
            "\n".join(instructions),
            q_chan_to_stim_qubits_map,
            c_channel_to_rec_idx,
        )

    def emit_leaf_circuit_instruction(
        self,
        circuit: Circuit,
        quantum_channel_map: dict[Channel, tuple[Any, stim.GateTarget]],
        classical_channel_map: dict[Channel, Any],
        measurement_record_counter: int = 0,
    ) -> str:
        """Provide the python code (as a string) to emit an Eka instruction in the
        target language."""

        if not isinstance(circuit, Circuit):
            raise TypeError("Input must be a Circuit instance.")

        if circuit.circuit and len(circuit.circuit) != 0:
            raise ValueError("The circuit must be an leaf circuit with channels.")

        instructions = []
        targets = [
            [quantum_channel_map[c][1]] for c in circuit.channels if c.is_quantum()
        ]
        classical_target = [c for c in circuit.channels if c.is_classical()]
        # Build the stim circuit from the grouped operations

        self._validate_ops_args(circuit.name, len(targets), len(classical_target))

        if circuit.name in self.operations_map:
            stim_instructions = self.operations_map[circuit.name](targets)
            for instr in stim_instructions:
                instructions.append(str(instr))

        if self.op_by_eka_name[circuit.name].op_type == OpType.MEASUREMENT:
            classical_channel_map[classical_target[0]] = measurement_record_counter
            measurement_record_counter += 1

        return "\n".join(instructions)

    @classmethod
    def stim_polygons(cls, interpreted_eka: InterpretationStep) -> str:
        """Define stim polygons using data qubits coordinates involved
        with each stabilizer on patches passed as argument to the function

        DEMO SYNTAX: #!pragma POLYGON(1,0,0,0.25) 5 11 16 23
            POLYGON(<X>, <Y>, <Z>, <color intensity>) <data qubits involved>

        Since polygon definitions are added as comments in the stim circuit body,
        and there is no way to add comments programmatically in stim.Circuit
        This function is only available to print polygon instructions from the
        patch stabilizers. The user *MUST* add these comments manually to the
        stim.Circuit string body to display the polygons in crumble

        Parameters
        ----------
        interpreted_eka: InterpretationStep
            The `InterpretationStep` object containing
            information on the code stabilizers

        Returns
        -------
        stim_polygons: str
            Stim polygon instructions as a string
        """
        # list of stabilizers to define the polygons
        pauli_polyarg_map = {
            "X": "(1,0,0,0.5)",
            "Y": "(0,1,0,0.5)",
            "Z": "(0,0,1,0.5)",
        }
        # Get all blocks at the initial timestamp (timestamp 0)
        initial_block_uuids = interpreted_eka.block_history.blocks_at(0)
        all_stabilizers = [
            stab
            for block_uuid in initial_block_uuids
            for stab in interpreted_eka.block_registry[block_uuid].stabilizers
        ]

        eka_circuit = interpreted_eka.final_circuit
        eka_channels = [
            ast.literal_eval(chan.label)
            for chan in eka_circuit.channels
            if chan.is_quantum()
        ]

        eka_coord_to_stim_qubit_map = {
            coords: stim.CircuitInstruction(
                name="QUBIT_COORDS",
                targets=[i],
                gate_args=cls._eka_to_stim_coordinates(coords),
            )
            for i, coords in enumerate(
                sorted(eka_channels, key=cls._eka_to_stim_coordinates)
            )
        }

        polygon_instructions = []
        for stab in all_stabilizers:
            if len(set(list(stab.pauli))) == 1:
                polyarg = pauli_polyarg_map[stab.pauli[0]]
                data_qubits = [
                    eka_coord_to_stim_qubit_map[data_qubit]
                    for data_qubit in stab.data_qubits
                ]
                # arrange qubits in cyclically to be visualized properly by crumble
                polygon_ordered_data_qubits = (
                    data_qubits[int(len(data_qubits) / 2) :]
                    + data_qubits[: int(len(data_qubits) / 2)][::-1]
                )
            else:
                raise ValueError(
                    f"Unsupported {stab.pauli}."
                    "Currently only CSS type codes are supported."
                )
            polygon_instructions.append(
                f"#!pragma POLYGON{polyarg} "
                + " ".join(
                    str(qubit.targets_copy()[0].value)
                    for qubit in polygon_ordered_data_qubits
                )
            )
            polygon_instructions_string = "\n".join(
                instruction for instruction in polygon_instructions
            )
        return polygon_instructions_string

    def print_stim_circuit_for_crumble(self, final_step: InterpretationStep) -> str:
        """Print the stim circuit along with polygon instructions to be
        used for crumble

        Parameters
        ----------
        stim_circ : stim.Circuit
            input stim circuit
        """
        polygon_instructions = self.stim_polygons(final_step)
        stim_circuit, _, _ = self.convert(final_step)

        total_output = polygon_instructions + "\n" + str(stim_circuit)

        return total_output
