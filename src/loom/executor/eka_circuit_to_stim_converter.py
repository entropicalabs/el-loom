"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import ast
from collections import defaultdict

# pylint: disable=no-name-in-module
from stim import CircuitInstruction, GateTarget, target_rec
from stim import Circuit as StimCircuit


from ..interpreter import InterpretationStep, Syndrome, LogicalObservable, Cbit
from ..eka import Circuit, Channel

from .circuit_error_model import ApplicationMode, CircuitErrorModel, ErrorType


def noise_annotated_stim_circuit(
    stim_circ: StimCircuit,
    before_measure_flip_probability: float = 0,
    after_clifford_depolarization: float = 0,
    after_reset_flip_probability: float = 0,
) -> StimCircuit:
    """
    This function takes as input a pure (sans noise) stim
    circuit, and outputs a the circuit with the desired noise model

    Parameters
    ----------
    stim_circ : StimCircuit
        The input noiseless stim circuit
    before_measure_flip_probability: float, optional
        X_ERROR probability before a measurement. Default set to
        0 will add no measurement errors
    after_clifford_depolarization: float, optional
        applies DEPOLARIZING_ERROR1 and DEPOLARIZING_ERROR2
        after each single and two qubit clifford gate in the circuit.
        Default set to 0 will add no depolarization errors
    after_reset_flip_probability: float, optional
        Apply an X_ERROR with this probability after a reset gate.
        Default set to 0 will add no reset errors

    Returns
    -------
    StimCircuit
        stim circuit annotated with the input noise model
    """

    stim_one_qubit_ops = ["H", "X", "Y", "Z", "I"]
    stim_two_qubit_ops = ["CX", "CY", "CZ", "SWAP"]

    def return_annotated_operation(op: CircuitInstruction):
        """
        Append/Prepend an appropriate stim annotation to the
        corresponding stim operation based on how the
        converter has been configured.

        For e.g., this operation appends X_ERROR annotation before each measurement
        round, if the corresponding error is turned on.
        """
        op_name = op.name
        targets = op.targets_copy()

        annotated_ops_list = [
            {"name": op_name, "targets": targets, "gate_args": op.gate_args_copy()}
        ]

        if op_name == "M" and before_measure_flip_probability > 0:
            annotation = [
                {
                    "name": "X_ERROR",
                    "targets": targets,
                    "gate_args": [before_measure_flip_probability],
                }
            ]
            annotated_ops_list = annotation + annotated_ops_list
        if op_name in stim_one_qubit_ops and after_clifford_depolarization > 0:
            annotation = [
                {
                    "name": "DEPOLARIZE1",
                    "targets": targets,
                    "gate_args": [after_clifford_depolarization],
                }
            ]
            annotated_ops_list = annotated_ops_list + annotation
        if op_name in stim_two_qubit_ops and after_clifford_depolarization > 0:
            annotation = [
                {
                    "name": "DEPOLARIZE2",
                    "targets": targets,
                    "gate_args": [after_clifford_depolarization],
                }
            ]
            annotated_ops_list = annotated_ops_list + annotation
        if op_name == "R" and after_reset_flip_probability > 0:
            annotation = [
                {
                    "name": "X_ERROR",
                    "targets": targets,
                    "gate_args": [after_reset_flip_probability],
                }
            ]
            annotated_ops_list = annotated_ops_list + annotation

        annotated_stim_ops_list = [
            CircuitInstruction(
                name=args_dict["name"],
                targets=args_dict["targets"],
                gate_args=args_dict["gate_args"],
            )
            for args_dict in annotated_ops_list
        ]

        return annotated_stim_ops_list

    annotated_stim_circuit = StimCircuit()
    for op in stim_circ:
        stim_annotated_op_list = return_annotated_operation(op)
        for annotated_op in stim_annotated_op_list:
            annotated_stim_circuit.append(annotated_op)

    return annotated_stim_circuit


class EkaCircuitToStimConverter:
    """
    Convert an Eka circuit description in Stim
    """

    def stim_polygons(self, interpreted_eka: InterpretationStep) -> str:
        """Define stim polygons using data qubits coordinates involved
        with each stabilizer on blocks passed as argument to the function

        DEMO SYNTAX: #!pragma POLYGON(1,0,0,0.25) 5 11 16 23
            POLYGON(<X>, <Y>, <Z>, <color intensity>) <data qubits involved>

        Since polygon definitions are added as comments in the stim circuit body,
        and there is no way to add comments programmatically in StimCircuit
        This function is only available to print polygon instructions from the
        block stabilizers. The user *MUST* add these comments manually to the
        StimCircuit string body to display the polygons in crumble

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
        all_stabilizers = [
            stab
            for block in interpreted_eka.block_history[0]
            for stab in block.stabilizers
        ]

        eka_coord_to_stim_qubit_instruction = (
            self.eka_coords_to_stim_qubit_instruction_mapper(
                interpreted_eka.final_circuit
            )
        )

        polygon_instructions = []
        for stab in all_stabilizers:
            if len(set(list(stab.pauli))) == 1:
                polyarg = pauli_polyarg_map[stab.pauli[0]]
                data_qubits = [
                    eka_coord_to_stim_qubit_instruction[data_qubit]
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

    def eka_coords_to_stim_qubit_instruction_mapper(
        self, eka_circuit: Circuit
    ) -> dict[tuple[int, ...], CircuitInstruction]:
        """Generate stim qubits from all input blocks

        Parameters
        ----------
        circuit: Circuit
            The input EKA circuit object

        Returns
        -------
        dict[tuple, CircuitInstruction]:
            Mapping between Eka coordinates and stim qubit instructions
        """
        eka_stim_qubits_map = {}
        # pylint: disable=attribute-defined-outside-init
        self.eka_stim_qubits_coords_map = {}

        eka_all_qubits = [
            ast.literal_eval(chan.label)
            for chan in eka_circuit.channels
            if chan.type in ["quantum", "ancilla"]
        ]

        def eka_to_stim_coordinates(coords):
            """
            Convert EKA coordinates to Stim coordinates.
            """
            match len(coords):
                case 2:
                    # For linear lattice codes
                    if coords[1] == 1:
                        return (coords[0], 0)
                    if coords[1] == 0:
                        return (coords[0] - 0.5, 0.5)
                    raise ValueError(
                        f"Invalid coordinate {coords}. "
                        "Coordinates should be in the form (x, 0) or (x, 1)."
                    )
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
                        "Coordinates should be in the form (x, y, 0) or (x, y, 1)."
                    )
                case _:
                    # For other lattice codes raise an error
                    raise ValueError(f"Invalid coordinate {coords}.")

        for i, coords in enumerate(
            # pylint: disable=unnecessary-lambda
            sorted(eka_all_qubits, key=lambda x: eka_to_stim_coordinates(x))
        ):

            new_coords = eka_to_stim_coordinates(coords)

            # Store Loom mapping as a class attribute to be used outside this function
            self.eka_stim_qubits_coords_map.update({coords: new_coords})
            eka_stim_qubits_map.update(
                {
                    coords: self.generate_stim_circuit_instruction(
                        name="QUBIT_COORDS",
                        targets=[i],
                        gate_args=new_coords,
                    )
                }
            )
        return eka_stim_qubits_map

    def eka_channel_to_stim_qubit_instruction_mapper(
        self,
        circuit,
        eka_coords_to_stim_qubit_instruction: dict[tuple[int, ...], CircuitInstruction],
    ) -> dict[Channel, CircuitInstruction]:
        """
        Based on the input EKA circuit, define stim qubits
        as QUBIT_COORDS
        Each non-classical channel in the input eka circuit is mapped
        to the respective qubit declaration in stim. Its a map containing
        elements, for e.g. :

        ``{Channel('q',id,): CircuitInstruction("QUBIT_COORDS", q, [x, y]}``

        Parameters
        ----------
        circuit: Circuit
            The EKA circuit object
        eka_coords_to_stim_qubit_instruction: dict[tuple[int,...], CircuitInstruction]
            The mapping between EKA coordinates and stim qubit instructions

        Returns
        -------
        dict[Channel, CircuitInstruction]:
            The mapping between circuit channels and stim qubit instructions
        """
        eka_stim_qubits_map = {}
        for channel in circuit.channels:
            # there are no classical channels in stim
            # measurements are chronologically indexed
            if channel.is_quantum():
                coords = ast.literal_eval(channel.label)
                stim_qubit = eka_coords_to_stim_qubit_instruction[coords]
                eka_stim_qubits_map.update({channel: stim_qubit})
        return eka_stim_qubits_map

    # NOISE ANNOTATION MAP DEFINITION
    @property
    def noise_annotation_mapper(self) -> dict[ErrorType, str]:
        """
        A dictionary mapping error types to the corresponding Stim operation name.
        """
        return {
            ErrorType.PAULI_X: "X_ERROR",
            ErrorType.PAULI_Y: "Y_ERROR",
            ErrorType.PAULI_Z: "Z_ERROR",
            ErrorType.PAULI_CHANNEL: "PAULI_CHANNEL_1",
            ErrorType.DEPOLARIZING1: "DEPOLARIZE1",
            ErrorType.DEPOLARIZING2: "DEPOLARIZE2",
        }

    # OPERATION MAP DEFINITION

    @property
    def single_qubit_gate_mapper(self) -> dict[str, str]:
        """
        A dictionary mapping Circuit single qubit gates to their corresponding Stim
        operations.
        """
        return {
            "identity": "I",
            "hadamard": "H",
            "phase": "S",
            "phase_inv": "S_DAG",
            "h": "H",
            "x": "X",
            "y": "Y",
            "z": "Z",
            "i": "I",
        }

    @property
    def classically_controlled_gate_mapper(self) -> dict[str, str]:
        """
        A dictionary mapping Circuit classically controlled gates to their corresponding Stim
        operations.
        """
        return {
            f"classically_controlled_{pauli.lower()}": f"C{pauli}"
            for pauli in ["X", "Y", "Z"]
        }

    @property
    def two_qubit_gate_mapper(self) -> dict[str, str]:
        """
        A dictionary mapping Circuit two qubit gates to their corresponding Stim operations.
        """
        return {
            "cx": "CX",
            "cnot": "CX",
            "cz": "CZ",
            "cy": "CY",
            "swap": "SWAP",
        }

    @property
    def measurement_mapper(self) -> dict[str, str]:
        """
        A dictionary mapping Eka measurement operations to their corresponding Stim
        operations.
        """
        return {
            "measurement": "M",
            "measure_z": "M",
            "measure_x": "MX",
            "measure_y": "MY",
        }

    @property
    def reset_mapper(self) -> dict[str, str | tuple[str]]:
        """
        A dictionary mapping Eka measurement operations to their corresponding Stim
        operations or list of operations.
        """
        return {
            "reset": "R",
            "reset_0": "R",
            "reset_1": ("R", "X"),
            "reset_+": "RX",
            "reset_-": ("RX", "Z"),
            "reset_+i": "RY",
            "reset_-i": ("RY", "Z"),
        }

    @property
    def misc_mapper(self) -> dict[str, str]:
        """
        A dictionary mapping Eka miscellaneous operations to their corresponding Stim
        operations.
        """
        return {
            "detector": "DETECTOR",
            "observable": "OBSERVABLE_INCLUDE",
            "[]": "TICK",
        }

    @property
    def to_stim_ops_mapper(self) -> dict[str, str | tuple[str]]:
        """This property is a dict mapping Eka operations to their corresponding Stim
        operations. The keys are the Eka operation names and the values are the Stim
        operations.

        Returns
        -------
        dict:
            A mapping between Circuit instruction names and stim instruction names
        """

        # NOTE: Some operations are not natively supported in stim (e.g. reset_+)
        # and are mapped to a combination of stim operations.
        # In the future we may want this to be the responsibility of a transpiler
        # rather than the converter.
        return (
            self.single_qubit_gate_mapper
            | self.two_qubit_gate_mapper
            | self.measurement_mapper
            | self.reset_mapper
            | self.misc_mapper
            | self.classically_controlled_gate_mapper
        )

    @staticmethod
    def _stim_target_sort_key(t: GateTarget) -> tuple[int, int]:
        """
        Sort key function for stim targets.
        Orders qubit targets first, then measurement record targets, then others.

        Parameters
        ----------
        t : stim target
            The stim target to generate a sort key for

        Returns
        -------
        tuple[int, int]
            The sort key tuple (priority, value)
        """
        if t.is_qubit_target:
            return (0, t.value)
        if t.is_measurement_record_target:
            return (1, t.value)
        return (2, 0)

    def generate_stim_circuit_instruction(
        self,
        name: str,
        targets: list[int],
        gate_args: list | None = None,
    ):
        """
        Return the StimCircuit.CircuitInstruction corresponding
        to the input operator

        Parameters
        ----------
        name: str
            The input stim circuit operator name
        targets: list[int]
            The list of indices the gate is acting on
        gate_args: str
            A list of arguments for parameterizing the gate. For e.g.

                * noise_probability for noise instruction
                * location coordinate for qubit declarations

            NOTE: This is a parameter of the CircuitInstruction, which depends on the
            type of instruction specified

        Returns
        -------
        circuit_instructions: list[CircuitInstruction]
            Stim circuit instructions for an input name and qubit target
        """
        if gate_args is None:
            gate_args = []

        stim_targets = [GateTarget(i) for i in targets]
        return CircuitInstruction(
            name=name,
            targets=stim_targets,
            gate_args=gate_args,
        )

    # pylint: disable=too-many-locals, too-many-statements, too-many-branches, too-many-nested-blocks
    def convert(
        self,
        interpreted_eka: InterpretationStep,
        with_ticks=False,
        error_models: list[CircuitErrorModel] | None = None,
    ) -> StimCircuit:
        """
        A method to convert the eka_circuit into stim circuit

        Properties of the interpreted_eka used for the conversion:

            1.) Circuit
            2.) Blocks
            3.) Syndrome history
            4.) Detectors list
            5.) Logical observables list

        Parameters
        ----------
        interpreted_eka: InterpretationStep
            The final interpretation step for the complete circuit
            containing necessary information on the program

        with_ticks: bool, optional
            If True, append a TICK instruction after each layer

        Returns
        -------
        stim_circuit: StimCircuit
            The iteratively generated StimCircuit.
        """
        if error_models is None:
            error_models = []

        # flatten the input circuit for ease of iterability
        input_eka_circuit = interpreted_eka.final_circuit

        for em in error_models:
            if input_eka_circuit != em.circuit:
                raise ValueError(
                    "The circuit of one of the error model does not match the input circuit."
                )

        def channels_from_eka_cbits(cbits: tuple[Cbit]) -> list[Channel]:
            channels = [
                channel
                for meas in cbits
                for channel in input_eka_circuit.channels
                if isinstance(meas, tuple) and channel.label == f"{meas[0]}_{meas[1]}"
                # ignore constant cbits with the isinstance check
            ]
            return channels

        def channels_from_eka_meas_objects(
            eka_meas_object: Syndrome | LogicalObservable,
        ) -> list[Channel]:
            """Get the Channel corresponding to the input
            eka measurement object. The object can be a Syndrome
            or a LogicalObservable

            Parameters
            ----------
            eka_meas_object : Syndrome | LogicalObservable

            Returns
            -------
            channels: list[Channel]
                The list of channels corresponding to measurements
            """
            channels = [
                channel
                for meas in eka_meas_object.measurements  # + eka_meas_object.corrections
                for channel in input_eka_circuit.channels
                if isinstance(meas, tuple) and channel.label == f"{meas[0]}_{meas[1]}"
                # ignore constant cbits with the isinstance check
            ]
            return channels

        # dynamically updated while iterating through instruction list
        measurement_channel_order_map = {}

        # initialize empty stim circuit
        stim_circ = StimCircuit()

        eka_coords_to_stim_qubit_instruction = (
            self.eka_coords_to_stim_qubit_instruction_mapper(input_eka_circuit)
        )
        eka_channel_to_stim_qubit_instruction = (
            self.eka_channel_to_stim_qubit_instruction_mapper(
                input_eka_circuit, eka_coords_to_stim_qubit_instruction
            )
        )

        # append qubit instructions first
        for qubit in eka_coords_to_stim_qubit_instruction.values():
            stim_circ.append(qubit)

        # append through operations
        meas_pointer = 0
        tick_count = 0
        eka_unrolled_circuit = Circuit.unroll(input_eka_circuit)
        for eka_layer in eka_unrolled_circuit:
            # stim_layers keeps track of the decomposition of a single eka gate into
            # multiple stim gates. Each index corresponds to a layer of gates, e.g. a
            # reset in |1> on qubit 0 is decomposed in [{"R": (0,)}, {"X": (0,)}]
            # The key is the stim gate name and the value is the targets.
            stim_layers: list[dict[str, tuple[int, ...]]] = [{}]
            # Create a list of dict to keep track of operations of each of the step in
            # the layer. So that we can group qubit targetted by the same op into a
            # single stim operation.

            # Also, create a dictionary mapping stim measurement targets to their
            # corresponding classical channels
            # This assumes that in each layer of an unrolled circuit, there is at most
            # one measurement operation per qubit and classical channel.
            measurement_targets_to_classical_channels: dict[int, Channel] = {}

            # This is not pretty. As a consequence of the converter design, we need to
            # group noise operations for each target. Other operations are identified
            # by their name and targets, so we can group them directly.
            # For noise operations, we need to group them by their name and parameters,
            # because the same noise operation can be applied to the same target
            # with different parameters. Since parameters are lists, and we need to
            # use them as keys in a dictionary, we create a map to hashed values,
            # and use the hash until we need the actual parameters.
            hashed_noise_args = defaultdict(lambda: None)

            # This loop map eka_instructions to decomposition in Stim op :
            # eg. reset_1 => [{"R": (0,)}, {"X": (0,)}]
            # if a eka instruction contains 2 qubits op and single qubit op,
            # it is supported.
            for instruction in eka_layer:
                eka_op_name = instruction.name
                # Translates the instruction to stim
                stim_op_names = self.to_stim_ops_mapper[eka_op_name]

                # If stim_op_names is a tuple, it means that the eka_op needs to be
                # decomposed into stim base gates. If there is a sequence of operations,
                # each gate will have the same target.
                # If it is a string, make it a tuple :
                stim_op_names = (
                    (stim_op_names,)
                    if isinstance(stim_op_names, str)
                    else stim_op_names
                )

                classical_channels = [
                    channel
                    for channel in instruction.channels
                    if channel.is_classical()
                ]
                quantum_channels = [
                    channel for channel in instruction.channels if channel.is_quantum()
                ]

                # Get the quantum targets for the instruction
                targets = [
                    target.value
                    for channel in quantum_channels
                    for target in eka_channel_to_stim_qubit_instruction[
                        channel
                    ].targets_copy()
                ]

                # Handle measurement operations
                if instruction.name in self.measurement_mapper:
                    if len(classical_channels) != 1 or len(quantum_channels) != 1:
                        raise ValueError(
                            f"Measurement operation {instruction.name} requires "
                            "exactly one classical and one quantum channel."
                        )
                    # Find the measurement channel corresponding to the target
                    measurement_targets_to_classical_channels[targets[0]] = (
                        classical_channels[0]
                    )
                # Handle classically controlled gates
                elif instruction.name in self.classically_controlled_gate_mapper:
                    if len(classical_channels) != 1 or len(quantum_channels) != 1:
                        raise ValueError(
                            f"Classically controlled operation {instruction.name} "
                            "requires exactly one classical and one quantum channel."
                        )
                    # Find the control pointer in the measurement_channel_order_map
                    control_pointer = measurement_channel_order_map.get(
                        classical_channels[0], None
                    )
                    if control_pointer is None:
                        raise KeyError(
                            f"Control channel {classical_channels[0].label} is used for"
                            " a classically controlled operation but not found in "
                            "the measurements."
                        )
                    # The classical register is defined as a negative number in
                    # stim. The previous measurement is given by -1, the next
                    # one by -2, etc.
                    stim_lookback_target = target_rec(control_pointer - meas_pointer)
                    # Put the control pointer at the start of the targets
                    # as it is going to be the control for the operation
                    targets = [stim_lookback_target] + targets

                noise_op_before = []
                noise_op_after = []

                for m in error_models:
                    annot = self.noise_annotation_mapper[m.error_type]
                    params = m.get_gate_error_probability(instruction)
                    if params:
                        k = hash(tuple(params))
                        hashed_noise_args[k] = params

                        if m.application_mode == ApplicationMode.BEFORE_GATE:
                            noise_op_before.append((annot, k))
                        elif m.application_mode == ApplicationMode.AFTER_GATE:
                            noise_op_after.append((annot, k))

                # map op to tuple op, args
                stim_op_name_and_args = list((n, None) for n in stim_op_names)

                # Add the noise operations before and after the main operation
                stim_op_name_and_args = (
                    noise_op_before + stim_op_name_and_args + noise_op_after
                )

                if len(stim_op_name_and_args) > len(stim_layers):
                    # If the number of stim operations is greater than the number of
                    # layers, add enough new layers
                    stim_layers += [
                        {} for _ in range(len(stim_op_name_and_args) - len(stim_layers))
                    ]

                # Append the targets to the stim_layer dictionaries
                # We use tuples (name, hash(param)) as keys because a user could stack several
                # Similar Noise operations on the same target, with different parameters..
                # If param is None, it is not an issue since it is hashable.
                for i, name_and_args_hash in enumerate(stim_op_name_and_args):
                    stim_layers[i][name_and_args_hash] = stim_layers[i].get(
                        name_and_args_hash, ()
                    ) + (targets,)

            for stim_layer in stim_layers:
                for (stim_op_name, args_hash), targets_groups in stim_layer.items():
                    for targets in targets_groups:
                        if stim_op_name in self.measurement_mapper.values():
                            # Track the order of measurements using a measurement
                            # pointer. To ensure correct ordering of the measurements,
                            # it needs to be done whenever a measurement operation is
                            # appended to the stim circuit.
                            meas_channel = measurement_targets_to_classical_channels[
                                targets[0]
                            ]
                            measurement_channel_order_map.update(
                                {meas_channel: meas_pointer}
                            )
                            meas_pointer += 1
                        op = (
                            self.generate_stim_circuit_instruction(
                                name=stim_op_name,
                                targets=targets,
                                gate_args=hashed_noise_args[args_hash],
                            ),
                        )

                        for stim_op in op:
                            # Append the stim operation to the stim circuit
                            stim_circ.append(stim_op)

            # Add noise to all targets at the end of the tick
            for model in error_models or []:
                if model.application_mode == ApplicationMode.IDLE_END_OF_TICK:
                    # get all the channels considered in the stim circuit
                    channels = eka_channel_to_stim_qubit_instruction.keys()
                    instructions = {}
                    for ch in channels:
                        target = eka_channel_to_stim_qubit_instruction[
                            ch
                        ].targets_copy()
                        name = self.noise_annotation_mapper[model.error_type]
                        error_args = model.get_idle_tick_error_probability(
                            tick_index=tick_count, channel_id=ch.id
                        )
                        if error_args:
                            key = (name, tuple(error_args))
                            if instructions.get(key):
                                instructions[key].append(*target)
                            else:
                                instructions[key] = [*target]
                    for (name, error_args), targets in instructions.items():
                        if error_args is not None:
                            stim_circ.append(
                                CircuitInstruction(
                                    name=name,
                                    # We sort the targets for convenience in testing
                                    targets=sorted(
                                        targets, key=self._stim_target_sort_key
                                    ),
                                    gate_args=error_args,
                                )
                            )
                if model.application_mode == ApplicationMode.END_OF_TICK:
                    # Apply the error model after the layer
                    error_args = model.get_tick_error_probability(tick_count)
                    if error_args is not None:
                        targets = sorted(
                            [
                                t
                                for target in eka_channel_to_stim_qubit_instruction.values()
                                for t in target.targets_copy()
                            ],
                            key=self._stim_target_sort_key,
                        )
                        stim_circ.append(
                            CircuitInstruction(
                                name=self.noise_annotation_mapper[model.error_type],
                                targets=targets,
                                gate_args=error_args,
                            )
                        )
            tick_count += 1
            # Append tick instruction for every eka layer
            if with_ticks is True:
                stim_circ.append(CircuitInstruction("TICK"))

        for detector in interpreted_eka.detectors:
            channels_list = channels_from_eka_cbits(detector.measurements)
            # Define lookback targets for the detectors
            targets = [
                measurement_channel_order_map[channel] - meas_pointer
                for channel in channels_list
            ]
            # Add space-time coordinates of detectors and extra indices such as color
            detector_labels = detector.labels

            # Transform space coordinates from Loom convention into real coordinates
            detector_args = list(
                self.eka_stim_qubits_coords_map[detector_labels["space_coordinates"]]
                + detector_labels["time_coordinate"]
            )

            # Add color index if present
            if detector_labels.get("color") is not None:
                detector_args.append(detector_labels["color"])
            stim_circ.append(
                self.generate_stim_circuit_instruction(
                    name="DETECTOR",
                    targets=[target_rec(idx) for idx in targets],
                    gate_args=detector_args,
                )
            )

        for i, logical_observable in enumerate(interpreted_eka.logical_observables):
            channels_list = channels_from_eka_meas_objects(logical_observable)
            # Define lookback targets for the logical observables
            targets = [
                measurement_channel_order_map[channel] - meas_pointer
                for channel in channels_list
            ]
            stim_circ.append(
                self.generate_stim_circuit_instruction(
                    name="OBSERVABLE_INCLUDE",
                    targets=[target_rec(idx) for idx in targets],
                    gate_args=[i],
                )
            )

        return stim_circ

    def print_stim_circuit_for_crumble(self, final_step: InterpretationStep) -> str:
        """Print the stim circuit along with polygon instructions to be
        used for crumble

        Parameters
        ----------
        stim_circ : StimCircuit
            input stim circuit
        """
        polygon_instructions = self.stim_polygons(final_step)
        stim_circuit = self.convert(final_step)

        total_output = polygon_instructions + "\n" + str(stim_circuit)

        return total_output
