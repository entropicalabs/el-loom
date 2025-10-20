"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=line-too-long, fixme, invalid-name
from functools import partial  # , reduce

from loom import cliffordsim
import loom.cliffordsim.operations as cops
from ..eka import Circuit, Channel


# pylint: disable=too-many-locals
def convert_circuit_to_cliffordsim(
    input_circuit: Circuit, index_to_channel_map: dict[int, Channel]
) -> list[cliffordsim.operations.Operation]:
    """
    Converts a Circuit object into a CliffordSim Circuit.

    NOTE: Qubit indices are remapped for now as the Circuit Object
    does not contain any information about qubit indices.

    NOTE: If input_circuit contains classical channels, additional operations will be
    added that create classical registers at the start and record the classical
    registers at the end of the circuit. The classical registers are created based on
    the labels of the classical channels. The first set of letters, separated by an _,
    in the label of the classical channel is the name of the classical register.

    TODO: Resolved Mapping Issue but external requirement of data qubit map as input
    might be an issue. Maybe dqubit_dict or the final output from extract_qubits should
    be information "uploaded" onto the CRD as well. (As this contains information about
    the ancilla qubit which was previously not available. i.e. what qubit indices are they)


    Parameter
    ---------
    input_circuit: Circuit
    dqubits_dict: dict

    Returns
    -------
    list[cliffordsim.operations.Operation]
        A list containing CliffordSim Operations that represent the original
        `input_circuit`. This list can be passed into CliffordSim's Engine
        to be simulated.
    """
    channel_id_to_index_map = {
        channel.id: index for index, channel in index_to_channel_map.items()
    }

    # If classical channels exist, create classical register.
    # We follow the convention where the first set of letters, separated by an _, in
    # the label of the classical channel is the name of the classical register.
    creg_cbits = [
        (input.label.split("_")[0], input.label)
        for input in input_circuit.channels
        if input.is_classical()
    ]
    # Returns a dictionary whose keys are the classical register names and values are
    # lists of classical bit labels.
    reg_dict = {}
    for each_reg_name, each_cbit_label in creg_cbits:
        if each_reg_name not in reg_dict:
            reg_dict[each_reg_name] = []
        reg_dict[each_reg_name].append(each_cbit_label)

    # creg_op_instr will contain all the Operations that create the Classical Registers.
    # creg_record_op_instr will contain all the Operations that record the Classical
    # Registers at the end of the circuit.
    creg_op_instr = []
    creg_record_op_instr = []
    for each_reg_name, cbit_labels in reg_dict.items():
        creg_op_instr.append(
            cops.CreateClassicalRegister(
                reg_name=each_reg_name, no_of_bits=len(cbit_labels), bit_ids=cbit_labels
            )
        )
        creg_record_op_instr.append(
            cops.RecordClassicalRegister(reg_name=each_reg_name)
        )

    SQ_GATE_MAP = {
        "identity": cops.Identity,
        "h": cops.Hadamard,
        "x": cops.X,
        "y": cops.Y,
        "z": cops.Z,
        "phase": cops.Phase,
        "phaseinv": cops.PhaseInv,
    }

    TQ_GATE_MAP = {
        "cnot": cops.CNOT,
        "cy": cops.CY,
        "cz": cops.CZ,
        "cx": cops.CNOT,
        "swap": cops.SWAP,
    }

    MEASUREMENT_OPS_MAP = {
        "measurement": cops.Measurement,
        "measurementbias0": partial(cops.Measurement, bias=0),
        "measurementbias1": partial(cops.Measurement, bias=1),
    } | {
        # All measure_{basis} operations
        f"measure_{basis.lower()}": partial(cops.Measurement, basis=basis)
        for basis in ["X", "Y", "Z"]
    }

    RESET_OPS_MAP = {
        "reset": cops.Reset,
    } | {
        # All reset_{state} operations
        f"reset_{state}": partial(cops.Reset, state=state)
        for state in ["0", "1", "+", "-", "+i", "-i"]
    }

    CLASSICALLY_CONTROLLED_GATES_MAP = {
        f"classically_controlled_{sq_gate_name}": (cops.ControlledOperation, op)
        for sq_gate_name, op in SQ_GATE_MAP.items()
    }

    op_map = {
        **TQ_GATE_MAP,
        **SQ_GATE_MAP,
        **MEASUREMENT_OPS_MAP,
        **RESET_OPS_MAP,
        **CLASSICALLY_CONTROLLED_GATES_MAP,
    }

    def operator_selector(item: Circuit) -> cops.Operation:
        if item.name not in op_map.keys():
            raise NotImplementedError(f'Invalid operation name "{item.name}"')
        try:
            # Get classical channel labels of circuit object.
            cc_labels = [
                each_channel.label
                for each_channel in item.channels
                if each_channel.is_classical()
            ]
            # Find the qubit indices of the corresponding quantum and ancilla channels.
            target_qubit_idx = [
                channel_id_to_index_map[input.id]
                for input in item.channels
                if input.is_quantum()
            ]

            if item.name in MEASUREMENT_OPS_MAP:
                # Measurement operations can only have one target qubit.
                if len(target_qubit_idx) != 1:
                    raise ValueError(
                        "Measurement operation can only have one target qubit."
                    )
                target_qubit = target_qubit_idx[0]

                # Measurements can be written onto a classical register if it exists.
                # No Classical Channel == No writing output to Classical Register
                # UUIDs generated for measurements since no Classical Channel available.
                if len(cc_labels) == 0:
                    return op_map[item.name](target_qubit=target_qubit)
                if len(cc_labels) == 1:
                    cc_label = cc_labels[0]
                    reg_name = next(
                        (
                            each_reg_name
                            for each_reg_name, cbit_labels in reg_dict.items()
                            if cc_label in cbit_labels
                        )
                    )

                    # BOTH the label of the measurement operation and the bit id of the
                    # classical bit == Label of the Classical Channel in the crd.Circuit
                    # measurement Object.
                    return op_map[item.name](
                        target_qubit=target_qubit,
                        label=cc_label,
                        reg_name=reg_name,
                        bit_id=cc_label,
                    )

                raise ValueError(
                    "Measurement operation can only have either one or no classical channel."
                )
            if item.name in CLASSICALLY_CONTROLLED_GATES_MAP:
                # Prepare the Operation that will be controlled.
                inner_op = op_map[item.name][1](*target_qubit_idx)

                # Classical Control X requires a classical register and classical bit to be specified.
                if len(cc_labels) != 1:
                    raise ValueError(
                        "Classically controlled operation must have exactly one classical channel."
                    )
                cc_label = cc_labels[0]
                reg_name = next(
                    (
                        each_reg_name
                        for each_reg_name, cbit_labels in reg_dict.items()
                        if cc_label in cbit_labels
                    )
                )

                return op_map[item.name][0](
                    app_operation=inner_op,
                    reg_name=reg_name,
                    bit_id=cc_label,
                )
            # else
            # map the circuit operation onto a catalyst operation using only
            # quantum channels
            return op_map[item.name](*target_qubit_idx)

        except KeyError as exc:
            raise KeyError(f"Channel index {exc} not found in qubit_map.") from exc

    output_ops = (
        creg_op_instr
        + [
            operator_selector(subcirc)
            for tick in input_circuit.flatten().circuit
            for subcirc in tick
        ]
        + creg_record_op_instr
    )

    return output_ops
