"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from typing import Callable
from types import MappingProxyType
from functools import partial
from math import pi

import pennylane as qml
from catalyst import measure, cond
from jax import Array

from ..eka import Circuit

# pylint: disable=unnecessary-lambda-assignment, comparison-with-callable

# Define the mapping between EKA operations and PennyLane operations as MappingProxyType
# to ensure that the mapping is immutable
# NOTE: For operations including measurement, the measurement operation needs to be
# given as an argument to the function so as to support both PennyLane and Catalyst
# measurement operations.
SINGLE_QUBIT_GATE_OPERATIONS_MAP = MappingProxyType(
    {
        "h": qml.Hadamard,
        "x": qml.PauliX,
        "y": qml.PauliY,
        "z": qml.PauliZ,
        "phase": qml.S,
        "phaseinv": qml.adjoint(qml.S),
        "t": partial(qml.RZ, phi=pi / 4),
        "tinv": partial(qml.RZ, phi=-pi / 4),
    }
)
TWO_QUBIT_GATE_OPERATIONS_MAP = MappingProxyType(
    {
        "cnot": qml.CNOT,
        "cy": qml.CY,
        "cz": qml.CZ,
        "cx": qml.CNOT,
        "swap": qml.SWAP,
    }
)
RESET_OPERATIONS_MAP = lambda is_catalyst: MappingProxyType(
    {
        "reset": partial(
            measure_op := measure if is_catalyst else qml.measure, reset=True
        ),
        "reset_0": partial(measure_op, reset=True),
        "reset_1": (partial(measure_op, reset=True), qml.PauliX),
        "reset_+": (partial(measure_op, reset=True), qml.Hadamard),
        "reset_-": (partial(measure_op, reset=True), qml.PauliX, qml.Hadamard),
        "reset_+i": (partial(measure_op, reset=True), qml.Hadamard, qml.S),
        "reset_-i": (partial(measure_op, reset=True), qml.PauliX, qml.Hadamard, qml.S),
    }
)
# The measurement operations will also register the measurement result
# in the measurement dictionary
MEASUREMENT_OPERATIONS_MAP = lambda is_catalyst: MappingProxyType(
    {
        "measurement": (measure_op := measure if is_catalyst else qml.measure),
        "measure": measure_op,
        "measure_x": (qml.Hadamard, measure_op),
        "measure_y": (qml.adjoint(qml.S), qml.Hadamard, measure_op),
    }
)

# Declare the classically controlled operations
CLASSICALLY_CONTROLLED_OPERATIONS_SQ_MAP = MappingProxyType(
    {
        f"classically_controlled_{op}": qml_op
        for op, qml_op in SINGLE_QUBIT_GATE_OPERATIONS_MAP.items()
    }
)
CLASSICALLY_CONTROLLED_OPERATIONS_TQ_MAP = MappingProxyType(
    {
        f"classically_controlled_{op}": qml_op
        for op, qml_op in TWO_QUBIT_GATE_OPERATIONS_MAP.items()
    }
)
# Combine all the classically controlled operations into a single mapping
CLASSICALLY_CONTROLLED_OPERATIONS_MAP = MappingProxyType(
    CLASSICALLY_CONTROLLED_OPERATIONS_SQ_MAP | CLASSICALLY_CONTROLLED_OPERATIONS_TQ_MAP
)


def convert_circuit_to_pennylane(
    input_circuit: Circuit, is_catalyst: bool
) -> tuple[Callable, dict]:
    """Converts an EKA circuit to a PennyLane circuit. The PennyLane circuit is returned
    as a function that can be called to execute the circuit.

    Parameters
    ----------
    input_circuit : Circuit
        The EKA circuit to be converted.
    is_catalyst : bool
        If True, the Catalyst measurement operation and cond will be used. Otherwise,
        the PennyLane ones will be used.

    Returns
    -------
    Callable
        A function that can be called to execute the PennyLane circuit. Can only be used
        within the @qjit decorator.
    dict
        A dictionary containing the qubit register of the PennyLane circuit.
    """
    # If the catalyst_measure is set to True, we will use the measure operation from
    # the catalyst library. Otherwise, we will use the measure operation from PennyLane.
    measure_op = measure if is_catalyst else qml.measure

    # Find the quantum channels in the input circuit
    q_channels = [chan for chan in input_circuit.channels if chan.is_quantum()]
    # Create an entry of size 1 in the qubit register for each quantum channel
    qml_register = qml.registers({q_chan.label: 1 for q_chan in q_channels})

    # Combine the gate and measurement operations into a single dictionary
    operations_map = (
        SINGLE_QUBIT_GATE_OPERATIONS_MAP
        | TWO_QUBIT_GATE_OPERATIONS_MAP
        | CLASSICALLY_CONTROLLED_OPERATIONS_MAP
        | MEASUREMENT_OPERATIONS_MAP(is_catalyst)
        | RESET_OPERATIONS_MAP(is_catalyst)
    )

    def operator_selector(
        item: Circuit, measurements: dict[str, Array]
    ) -> dict[str, Array]:
        """Selects the PennyLane operation to be executed based on the EKA operation.

        Parameters
        ----------
        item : Circuit
            The EKA operation to be executed.
        measurements : dict[str, Array]
            A dictionary containing the measurements of the circuit. It is updated
            with the measurement results of the current operation and returned.

        Returns
        -------
        dict[str, Array]
            The updated dictionary containing the measurements of the circuit.
        """
        if item.name not in operations_map.keys():
            raise NotImplementedError(f'Invalid operation name "{item.name}"')

        # Obtain the list of operations to be executed
        pennylane_operations = operations_map[item.name]
        pennylane_operations = (
            pennylane_operations
            if isinstance(pennylane_operations, tuple)
            else [pennylane_operations]
        )

        # Find the quantum and classical channels of the operation
        c_channels = [chan for chan in item.channels if chan.is_classical()]
        q_channels = [chan for chan in item.channels if chan not in c_channels]

        # Find the wires to which the operation will be applied
        wires = [qml_register[input_qchan.label][0] for input_qchan in q_channels]
        wires = wires if len(wires) > 1 else wires[0]

        # Check if the eka operation is a measurement operation that requires
        # registering the measurement result
        is_measurement_operation = (
            item.name in MEASUREMENT_OPERATIONS_MAP(is_catalyst).keys()
        )
        # Check if the eka operation is a classically controlled operation
        is_classically_controlled_operation = (
            item.name in CLASSICALLY_CONTROLLED_OPERATIONS_MAP.keys()
        )

        for qml_op in pennylane_operations:
            if qml_op == measure_op and is_measurement_operation:
                # If the operation is a measurement and it's part of the
                # measurement_bookkeeping_ops, we need to bookkeep the
                # measurement result

                # Find the classical channel to which the measurement result will be
                # stored.
                # We append the measurement result to a dictionary with the key being
                # the label of the classical channel
                classical_channel = next(
                    input_chan
                    for input_chan in item.channels
                    if input_chan.is_classical()
                )
                measurements[classical_channel.label] = qml_op(wires=wires)
            elif is_classically_controlled_operation:
                # If the operation is classically controlled, we need to find the
                # control channel and the value of the control channel in the
                # measurements dictionary
                if len(c_channels) != 1:
                    raise ValueError(
                        f"{item.name} operation is classically controlled and it "
                        f"requires exactly one classical channel. "
                        f"Found {len(c_channels)} classical channels."
                    )
                control_channel_label = c_channels[0].label

                # We need to find the value of the control channel in the measurements
                # dictionary
                if control_channel_label not in measurements.keys():
                    raise KeyError(
                        f"{item.name} operation is classically controlled but the "
                        f"control channel {control_channel_label} could not be found in"
                        f" the measurements dictionary."
                    )
                control_value = measurements[control_channel_label]

                # pylint: disable=cell-var-from-loop
                if is_catalyst:
                    # Use cond to apply the operation only if the control value is True
                    @cond(control_value)
                    def apply_op():
                        qml_op(wires=wires)

                    apply_op()

                else:
                    qml.cond(control_value, qml_op)(wires=wires)

            else:
                # Bookkeeping is not required for this operation so we just
                # execute it
                qml_op(wires=wires)

        return measurements

    def circuit_function() -> dict:
        """Generator of the PennyLane circuit. To be called within @qjit.

        Returns
        -------
        dict
            A dictionary containing the measurements of the circuit.
        """

        measurements = {}
        for tick in Circuit.unroll(input_circuit):
            for subcirc in tick:
                measurements = operator_selector(subcirc, measurements)
        return measurements

    return circuit_function, qml_register
