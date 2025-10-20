"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import re
from functools import reduce

from ..eka import Circuit, Channel, ChannelType


# pylint: disable=too-many-locals, too-many-statements
def convert_circuit_to_qasm(
    input_circuit: Circuit,
    syndromes: list | None = None,
    detectors: list | None = None,
    logicals: list | None = None,
    ancilla_channels: list[Channel] | None = None,
) -> dict:
    """
    Converts a Circuit object into a qasm string

    Parameter
    ---------
    input_circuit: Circuit
        The circuit to be converted into a QASM string. The input circuit must be
        unrolled
    syndromes: list
        List of syndromes that are measured in the eka circuit
    detectors: list
        List of detectors that are measured in the eka circuit
    logicals: list
        List of logicals that are measured in the eka circuit
    ancilla_channels: list[Channel]
        List of channels to be considered as ancillas

    Returns
    -------
    dict
        A dictionary containing the following:
        - `eka_to_qasm_syndromes`: A dictionary that maps the EKA syndromes to the
        QASM measurements
        - `eka_to_qasm_detectors`: A dictionary that maps the EKA detectors to the
        QASM measurements
        - `qasm_circuit`: A QASM string containing the operations that represent
        the original `input_circuit`.
    """
    if ancilla_channels is None:
        ancilla_channels = []

    def define_regs(
        input_circuit: Circuit,
        syndromes: list | None = None,
        detectors: list | None = None,
        logicals: list | None = None,
    ):  # pylint: disable=too-many-branches,too-many-statements
        """
        Given the input circuit and the syndromes and/or detectors, this function:
        
        - maps the data qubits, ancilla qubits, and data measurements and \
        ancilla measurements from loom.eka to indices of qasm registers.
        - creates a mapping between the EKA syndromes and the QASM measurements
        - creates a mapping between the EKA detectors and the QASM measurements.

        Parameters
        ----------
        input_circuit: Circuit
            The circuit to be converted into a QASM string. The input circuit must be
            unrolled
        syndromes: list | None
            List of syndromes that are measured in the eka circuit
        detectors: list | None
            List of detectors that are measured in the eka circuit
        logicals: list | None
            List of logicals that are measured in the eka circuit

        Returns
        -------
        tuple
            A tuple containing the following: 
            
            - data_qubits_idx_mapping: dict
                A dictionary that maps the data qubits to their index in the data register
            - ancilla_qubits_idx_mapping: dict
                A dictionary that maps the ancilla qubits to their index in the ancilla register
            - classical_bits_idx_mapping: dict
                A dictionary that maps the classical bits to their index in the classical register
            - classical_registers: dict
                A dictionary that maps the classical register name to the number of bits
            - eka_to_qasm_syndromes: dict
                A dictionary that maps the EKA syndromes to the QASM measurements
            - eka_to_qasm_detectors: dict
                A dictionary that maps the EKA detectors to the QASM measurements
        """

        data_qubits_idx_mapping = {}
        ancilla_qubits_idx_mapping = {}
        classical_bits_idx_mapping = {}
        classical_registers = {}

        classical_channels = []
        qubit_channels = []
        for channel in input_circuit.channels:
            if channel.is_classical():
                classical_channels.append(channel)
            else:
                qubit_channels.append(channel)

        def classical_sorter(channel: Channel):
            """
            Sorts the classical channels by their index and their coordinate.
            The label is of the form "c_(x,y,...)_i" where i is the measurement index.
            """
            assert channel.is_classical(), "Channel type needs to be classical"
            _, coords, index = channel.label.split("_")
            loc = (index,) + tuple(
                int(coord) for coord in re.sub("[c( )]", "", coords).split(",")
            )
            return loc

        def qubit_sorter(channel: Channel):
            """
            Sorts the qubit channels by their coordinates.
            The label is of the form "(x,y,...)" where i is the index.
            """
            assert channel.is_quantum(), "Channel type needs to be quantum or ancilla"

            loc = tuple(
                int(coord)
                for coord in re.sub("[c( )]", "", str(channel.label)).split(",")
            )
            return loc

        # sorted lists of classical and qubit channels
        classical_channels: list[Channel] = sorted(
            classical_channels, key=classical_sorter
        )
        qubit_channels: list[Channel] = sorted(qubit_channels, key=qubit_sorter)

        d_idx, a_idx = 0, 0
        for channel in classical_channels:
            _, coord, layer = channel.label.split("_")
            qubit_type_str = int(re.sub("[( )]", "", coord).split(",")[-1])
            if qubit_type_str == 0:
                if classical_registers.get(f"data_creg{layer}") is None:
                    classical_registers[f"data_creg{layer}"] = 0
                    idx = 0
                else:
                    idx = classical_registers[f"data_creg{layer}"]
                classical_bits_idx_mapping[channel.id] = (f"data_creg{layer}", idx)
                classical_registers[f"data_creg{layer}"] += 1
            elif qubit_type_str == 1:
                if classical_registers.get(f"anc_creg{layer}") is None:
                    classical_registers[f"anc_creg{layer}"] = 0
                    idx = 0
                else:
                    idx = classical_registers[f"anc_creg{layer}"]
                classical_bits_idx_mapping[channel.id] = (f"anc_creg{layer}", idx)
                classical_registers[f"anc_creg{layer}"] += 1

        for channel in qubit_channels:
            if channel in ancilla_channels:
                ancilla_qubits_idx_mapping[channel.id] = a_idx
                a_idx += 1
            else:
                data_qubits_idx_mapping[channel.id] = d_idx
                d_idx += 1

        # mapping between the EKA syndromes and the QASM measurements
        eka_to_qasm_syndromes = {}
        if syndromes is not None:
            classical_label_to_id_map = {
                channel.label: channel.id for channel in classical_channels
            }

            for syndrome in syndromes:
                # if syndrome.measurements != ():
                measurement_ids = [
                    classical_label_to_id_map["_".join(str(part) for part in label)]
                    for label in syndrome.measurements
                ]
                eka_to_qasm_syndromes.update(
                    {
                        syndrome: [
                            classical_bits_idx_mapping[meas_id]
                            for meas_id in measurement_ids
                        ]
                    }
                )

        # mapping between the EKA detectors and the QASM measurements
        eka_to_qasm_detectors = {}
        if detectors is not None:
            for detector in detectors:
                detector_measurements = [
                    eka_to_qasm_syndromes[syndrome] for syndrome in detector.syndromes
                ]
                detector_measurements = reduce(
                    lambda x, y: x + y, detector_measurements
                )
                eka_to_qasm_detectors[detector] = detector_measurements

        # mapping between the EKA logicals and the QASM measurements
        eka_to_qasm_logicals = {}
        if logicals is not None:
            for logical in logicals:
                measurement_ids = [
                    classical_label_to_id_map["_".join(str(part) for part in label)]
                    for label in logical.measurements
                ]
                eka_to_qasm_logicals.update(
                    {
                        logical: [
                            classical_bits_idx_mapping[meas_id]
                            for meas_id in measurement_ids
                        ]
                    }
                )

        return (
            data_qubits_idx_mapping,
            ancilla_qubits_idx_mapping,
            classical_bits_idx_mapping,
            classical_registers,
            eka_to_qasm_syndromes,
            eka_to_qasm_detectors,
            eka_to_qasm_logicals,
        )

    single_qubits_op_map = {
        "identity": "id",
        "x": "x",
        "y": "y",
        "z": "z",
        "h": "h",
        "phase": "s",
        "phaseinv": "sdg",
        "t": "t",
        "tinv": "tdg",
    }

    two_qubits_op_map = {
        "cnot": "cx",
        "cx": "cx",
        "cy": "cy",
        "cz": "cz",
        "swap": "swap",
    }

    meas_op_map = {
        "measurement": "measure",
        "measure_z": ("measure"),
        "measure_x": ("h", "measure"),
        "measure_y": ("sdg", "h", "measure"),
    }

    reset_operation_mapper = {
        "reset": "reset",
        "reset_0": "reset",
        "reset_1": ("reset", "x"),
        "reset_+": ("reset", "h"),
        "reset_-": ("reset", "x", "h"),
        "reset_+i": ("reset", "h", "s"),
        "reset_-i": ("reset", "h", "sdg"),
    }

    misc_op_map = {
        "barrier": "barrier",
    }

    op_map = (
        single_qubits_op_map
        | two_qubits_op_map
        | meas_op_map
        | reset_operation_mapper
        | misc_op_map
    )

    # Here lies the instantiation of quantum and classical registers
    def instantiate_registers(
        data_qubits_dict: dict[str, int],
        ancilla_qubits_dict: dict[str, int],
        classical_registers: dict[str, int],
    ):
        """Creates the QASM string that instantiate the classical and quantum registers

        Parameters
        ----------
        data_qubits_dict: dict
            Dictionary that maps a quantum channel id to its index in the data register
        ancilla_qubits_dict: dict
            Dictionary that maps a quantum channel id to its index in the ancilla register
        classical_registers: dict
            Dictionary that maps the classical register name to the number of bits

        Returns
        -------
        str
            QASM string instantiating the quantum and classical registers
        """
        reg_declaration = []

        # data register
        reg_declaration.append(f"qubit[{len(data_qubits_dict)}] data_qreg;\n")

        # ancilla register
        reg_declaration.append(f"qubit[{len(ancilla_qubits_dict)}] anc_qreg;\n")

        # classical registers one for each layer
        for reg_name, reg_size in classical_registers.items():
            reg_declaration.append(f"bit[{reg_size}] {reg_name};\n")

        return "".join(reg_declaration)

    def extract_operators(
        input_circuit: Circuit,
        data_qubits_map: dict[str, int],
        ancilla_qubits_map: dict[str, int],
        classical_bits_map: dict[str, int],
    ):  # pylint: disable=too-many-branches,too-many-statements
        """Extracts the operator describing the quantum circuit as a QASM string.
        Each line describes a single operation applied to its respective quantum
        and classical registers.

        Parameters
        ----------
        input_circuit: Circuit
            Circuit that holds the information about which operator has to be applied
            to which qubits
        data_qubits_map: dict
            Dictionary that maps a quantum channel id to its index in the data register
        ancilla_qubits_map: dict
            Dictionary that maps a ancilla channel id to its index in the ancilla register
        classical_bits_map: dict
            Dictionary that maps the classical channel id to register name,idx in QASM

        Returns
        -------
        output_ops: str
            QASM string that describes the operations performed in the circuit
        """

        # First unroll the input circuit
        input_circuit_sequence = Circuit.unroll(input_circuit)

        def get_reg_name_from_chan_id(chan_id):
            return "data_qreg" if chan_id in data_qubits_map else "anc_qreg"

        all_qubits_map = data_qubits_map | ancilla_qubits_map

        string_list = []
        # unrolled circuit is 2 dimensional and the loop parses over each
        # operation in each timeslice of the circuit to write QASM circuit
        for timeslice in input_circuit_sequence:
            for subcircuit in timeslice:
                # check if the subcircuit (operation) is valid
                try:
                    ops = op_map[subcircuit.name.lower()]
                    ops = [ops] if isinstance(ops, str) else ops
                except KeyError as e:
                    # check if the operation is conditioned
                    if subcircuit.name.startswith("ifelse_condition_"):
                        name = "ifelse_condition"
                    else:
                        raise NotImplementedError(
                            f"Operation {subcircuit.name} not supported"
                        ) from e

                for name in ops:
                    # Parse the operation as follows, if its a measurement op
                    if name == "measure":
                        if len(subcircuit.channels) != 2:
                            raise ValueError(
                                f"Measurement operation {subcircuit.name} must have "
                                f"exactly two channels, got {len(subcircuit.channels)}"
                            )
                        # Each measurement is associated to one qubit and one c-bit
                        q_channel, c_channel = subcircuit.channels
                        if (
                            q_channel.type not in (ChannelType.QUANTUM)
                        ) or c_channel.type != ChannelType.CLASSICAL:
                            raise ValueError(
                                f"Measurement operation {subcircuit.name} must have "
                                f"one quantum/ancilla and one classical channel in "
                                f"order, got {q_channel.type} and {c_channel.type}"
                            )

                        qubit_reg_name = get_reg_name_from_chan_id(q_channel.id)
                        qasm_qubit_id = all_qubits_map[q_channel.id]

                        creg_name, creg_idx = classical_bits_map[c_channel.id]

                        # string syntax to describe measurement operation in QASM
                        string = (
                            f"{creg_name}[{classical_registers[creg_name]-1-creg_idx}] = measure "
                            f"{qubit_reg_name}[{qasm_qubit_id}];\n"
                        )

                    # If its a conditional operation, parse as follows
                    elif name == "ifelse_condition":

                        # get the qubit channels in QASM syntax
                        q_channels_qasm = []
                        for q_channel in subcircuit.channels:
                            qubit_reg_name = get_reg_name_from_chan_id(q_channel.id)
                            qasm_qubit_id = all_qubits_map[q_channel.id]

                            q_channels_qasm.append(f"{qubit_reg_name}[{qasm_qubit_id}]")

                        # Define the if-else condition in QASM syntax using Eka classical channels
                        openqasm_conditions = (
                            f"{classical_bits_map[subcircuit.condition.channels[0].id][0]} == "
                            f"{int(''.join(str(int(b)) for b in subcircuit.condition.value), 2)}"
                        )

                        # Extract the conditional circuits for both `if` and `else` branches
                        conditional_branching_circuits = ["", ""]
                        for i, branch in enumerate(subcircuit.circuit):
                            for op in branch:
                                for channel in q_channels_qasm:
                                    conditional_branching_circuits[
                                        i
                                    ] += f"    {op_map[op.name.lower()]} {channel};\n"

                        # QASM string syntax for the if condition
                        # pylint: disable=line-too-long
                        if_syntax = (
                            f"""if ({openqasm_conditions}) {{\n{conditional_branching_circuits[0]}}}"""
                            if conditional_branching_circuits[0]
                            else ""
                        )
                        # QASM string syntax for the else condition
                        else_syntax = (
                            (f""" else {{\n{conditional_branching_circuits[1]}\n}}""")
                            if conditional_branching_circuits[1]
                            else ""
                        )
                        string = if_syntax + else_syntax + "\n"

                    # If its a regular quantum operation, parse as follows
                    else:
                        string = name
                        for q_channel in subcircuit.channels:
                            qubit_reg_name = get_reg_name_from_chan_id(q_channel.id)
                            qasm_qubit_id = all_qubits_map[q_channel.id]
                            string += f" {qubit_reg_name}[{qasm_qubit_id}],"
                        string = string[:-1]
                        string += ";\n"
                    string_list.append(string)

            # instructions to add barriers
            # pylint: disable=fixme
            # TODO: Improve barrier placement. Currently too many!
            data_indices = ", ".join(
                f"data_qreg[{i}]" for i in range(len(data_qubits_map))
            )
            ancilla_indices = ", ".join(
                f"anc_qreg[{i}]" for i in range(len(ancilla_qubits_map))
            )
            string_list.append(f"barrier {data_indices}, {ancilla_indices};\n")

        output_ops = "".join(string_list)

        return output_ops

    # Instantiate the quantum and classical mappings, and measurement maps for the
    # input circuit
    (
        data_qubits_mapping,
        ancilla_qubits_mapping,
        classical_bits_mapping,
        classical_registers,
        eka_to_qasm_syndromes,
        eka_to_qasm_detectors,
        eka_to_qasm_logicals,
    ) = define_regs(input_circuit, syndromes, detectors, logicals)

    # The QASM string declaring the quantum and classical registers
    reg_declaration = instantiate_registers(
        data_qubits_mapping, ancilla_qubits_mapping, classical_registers
    )

    # The QASM string describing all operations in the circuit
    output_ops = extract_operators(
        input_circuit,
        data_qubits_mapping,
        ancilla_qubits_mapping,
        classical_bits_mapping,
    )

    # Standard header included in QASM programs
    header_qasm = """OPENQASM 3.0;\ninclude "stdgates.inc";\n"""

    # complete QASM string describing the full circuit
    qasm_string = header_qasm + reg_declaration + output_ops
    output = {
        "qasm_circuit": qasm_string,
        "eka_to_qasm_syndromes": eka_to_qasm_syndromes,
        "eka_to_qasm_detectors": eka_to_qasm_detectors,
        "eka_to_qasm_logicals": eka_to_qasm_logicals,
    }
    return output
