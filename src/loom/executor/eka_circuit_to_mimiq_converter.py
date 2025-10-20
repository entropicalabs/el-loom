"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from types import MappingProxyType
import mimiqcircuits as mc

from ..eka import Circuit, Channel


# Define the mapping between EKA operations and MIMIQ operations as MappingProxyType
# to ensure that the mapping is immutable
SINGLE_QUBIT_GATE_OPERATIONS_MAP = MappingProxyType(
    {
        "h": mc.GateH(),
        "x": mc.GateX(),
        "y": mc.GateY(),
        "z": mc.GateZ(),
        "phase": mc.GateS(),
        "phaseinv": mc.GateSDG(),
    }
)
TWO_QUBIT_GATE_OPERATIONS_MAP = MappingProxyType(
    {
        "cnot": mc.GateCX(),
        "cy": mc.GateCY(),
        "cz": mc.GateCZ(),
        "cx": mc.GateCX(),
        "swap": mc.GateSWAP(),
    }
)

RESET_OPERATIONS_MAP = MappingProxyType(
    {
        "reset": mc.ResetZ(),
        "reset_0": mc.ResetZ(),
        "reset_1": (mc.ResetZ(), mc.GateX()),
        "reset_+": mc.ResetX(),
        "reset_-": (mc.ResetX(), mc.GateZ()),
        "reset_+i": mc.ResetY(),
        "reset_-i": (mc.ResetY(), mc.GateZ()),
    }
)
# The measurement operations will also register the measurement result
# in the measurement dictionary
MEASUREMENT_OPERATIONS_MAP = MappingProxyType(
    {
        "measurement": mc.MeasureZ(),
        "measure": mc.MeasureZ(),
        "measure_x": mc.MeasureX(),
        "measure_y": mc.MeasureY(),
    }
)

# Declare the classically controlled operations as tuples to be composed into operations
CLASSICALLY_CONTROLLED_OPERATIONS_SQ_MAP = MappingProxyType(
    {
        f"classically_controlled_{op}": mc.IfStatement(mimiq_op, mc.BitString("1"))
        for op, mimiq_op in SINGLE_QUBIT_GATE_OPERATIONS_MAP.items()
    }
)
CLASSICALLY_CONTROLLED_OPERATIONS_TQ_MAP = MappingProxyType(
    {
        f"classically_controlled_{op}": mc.IfStatement(mimiq_op, mc.BitString("1"))
        for op, mimiq_op in TWO_QUBIT_GATE_OPERATIONS_MAP.items()
    }
)
# Combine all the classically controlled operations into a single mapping
CLASSICALLY_CONTROLLED_OPERATIONS_MAP = MappingProxyType(
    CLASSICALLY_CONTROLLED_OPERATIONS_SQ_MAP | CLASSICALLY_CONTROLLED_OPERATIONS_TQ_MAP
)

# Combine all the operations into a single mapping
ALL_OPERATIONS_MAP = MappingProxyType(
    SINGLE_QUBIT_GATE_OPERATIONS_MAP
    | TWO_QUBIT_GATE_OPERATIONS_MAP
    | RESET_OPERATIONS_MAP
    | MEASUREMENT_OPERATIONS_MAP
    | CLASSICALLY_CONTROLLED_OPERATIONS_MAP
)


def convert_circuit_to_mimiq(
    input_circuit: Circuit,
) -> tuple[mc.Circuit, dict[str, dict[int, Channel]]]:
    """
    Convert a circuit from EKA to MIMIQ format.

    Parameters
    ----------
    input_circuit : Circuit
        The input circuit to be converted. The circuit should be in EKA format.

    Returns
    -------
    mimiqcircuits.Circuit
        The converted circuit in MIMIQ format. The circuit is represented as a
        MIMIQ circuit object.
    dict[str, dict[int, Channel]]
        A dictionary containing the mapping of channels to their indices in the
        circuit. The dictionary contains two keys: "quantum" and "classical",
        which map the quantum and classical indices of MIMIQ to their
        corresponding channels in the input circuit, respectively.
    """
    # Create a dictionary of all elementary mimiq operations as items to their class
    # names

    # Find all the channels in the input circuit and separate them into quantum and
    # classical channels
    c_channels = sorted(
        [chan for chan in input_circuit.channels if chan.is_classical()],
        key=lambda x: x.label,
    )
    q_channels = sorted(
        [chan for chan in input_circuit.channels if chan.is_quantum()],
        key=lambda x: x.label,
    )

    # Make a mapping the channels to their indices in the circuit
    register_dict = {
        "quantum": {chan: idx for idx, chan in enumerate(q_channels)},
        "classical": {chan: idx for idx, chan in enumerate(c_channels)},
    }
    # Get the inverse mapping of the channels to their indices
    register_dict_inv = {
        "quantum": {idx: chan for chan, idx in register_dict["quantum"].items()},
        "classical": {idx: chan for chan, idx in register_dict["classical"].items()},
    }

    mimiq_circuit = mc.Circuit()
    for eka_layer in Circuit.unroll(input_circuit):

        mimiq_layers = get_mimiq_layers(register_dict, eka_layer)

        for mimiq_layer in mimiq_layers:
            for mimiq_op, targets_list in mimiq_layer:
                n_repeats = len(targets_list)

                # Parallelize if the operation is repeated more than once and is a gate
                # operation
                # NOTE: These constraints are set by the MIMIQ library
                parallelize = n_repeats > 1 and isinstance(mimiq_op, mc.Gate)

                if parallelize:
                    # Flatten the targets list
                    targets_flat = [
                        item for sublist in targets_list for item in sublist
                    ]
                    # Parallelize the operation
                    mimiq_circuit.push(mc.Parallel(n_repeats, mimiq_op), *targets_flat)
                else:
                    for targets in targets_list:
                        mimiq_circuit.push(mimiq_op, *targets)

    return mimiq_circuit, register_dict_inv


def get_mimiq_layers(
    register_dict: dict[str, dict[Channel, int]],
    eka_layer: tuple[Circuit, ...],
) -> list[list[tuple[mc.Operation, list[list[int]]]]]:
    """
    Convert a layer of EKA operations to MIMIQ operations.
    The function takes a layer of EKA operations and converts them to MIMIQ
    operations.

    Parameters
    ----------
    register_dict : dict[str, dict[Channel, int]]
        A dictionary containing the mapping of channels to their indices in the
        circuit. The dictionary contains two keys: "quantum" and "classical",
        which map the quantum and classical channels to their indices, respectively.
    eka_layer : tuple[Circuit, ...]
        A tuple containing the EKA operations in a signle unrolled layer. Each operation
        is represented as a Circuit object.

    Returns
    -------
    list[list[tuple[mc.Operation, list[list[int]]]]]
        The converted MIMIQ operations in a list of layers.
        mimiq_layers[layer_idx][op_idx] is a tuple containing the MIMIQ operation and
        a list of targets for the operation.
    """

    mimiq_layers: list[list[tuple[mc.Operation, list[list[int]]]]] = []
    for base_op in eka_layer:
        if base_op.name not in ALL_OPERATIONS_MAP.keys():
            raise NotImplementedError(f'Invalid operation name "{base_op.name}"')

        # Transform the EKA operation to MIMIQ operation(s)
        mimiq_operations = ALL_OPERATIONS_MAP[base_op.name]
        # Cast it into a tuple if it is not already
        mimiq_operations = (
            mimiq_operations
            if isinstance(mimiq_operations, tuple)
            else (mimiq_operations,)
        )

        if len(mimiq_operations) > len(mimiq_layers):
            # If the number of MIMIQ operations is greater than the number of
            # layers, add enough new layers
            mimiq_layers += [
                [] for _ in range(len(mimiq_operations) - len(mimiq_layers))
            ]

            # Find quantum and classical channels of the operation
        op_q_channels = [chan for chan in base_op.channels if chan.is_quantum()]
        op_c_channels = [chan for chan in base_op.channels if chan.is_classical()]

        # Set the targets of the operation to be the indices of these channels
        # by putting first the quantum channels and then the classical channels
        # This is the convention used in MIMIQ
        targets = [register_dict["quantum"][chan] for chan in op_q_channels] + [
            register_dict["classical"][chan] for chan in op_c_channels
        ]

        for layer_idx, mimiq_op in enumerate(mimiq_operations):
            # Check if the operation already exists in the layer and if so, add the
            # targets to its target_list
            mimiq_op_idx = next(
                (
                    idx
                    for idx in range(len(mimiq_layers[layer_idx]))
                    if mimiq_layers[layer_idx][idx][0] == mimiq_op
                ),
                None,
            )
            if mimiq_op_idx is None:
                # If the operation is not already in the layer, add it
                mimiq_layers[layer_idx].append((mimiq_op, [targets]))
            else:
                # If the operation is already in the layer, add the targets to it
                mimiq_layers[layer_idx][mimiq_op_idx][1].append(targets)

    return mimiq_layers
