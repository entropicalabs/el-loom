"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from loom.eka import Circuit
from loom.eka.operations import (
    MeasureLogicalX,
    MeasureLogicalY,
    MeasureLogicalZ,
    LogicalMeasurement,
)

from .generate_syndromes import generate_syndromes
from .generate_detectors import generate_detectors
from ..interpretation_step import InterpretationStep
from ..logical_observable import LogicalObservable


def measurelogicalpauli(  # pylint: disable=too-many-locals
    interpretation_step: InterpretationStep,
    operation: MeasureLogicalX | MeasureLogicalY | MeasureLogicalZ,
    same_timeslice: bool,
    debug_mode: bool,  # pylint: disable=unused-argument
) -> InterpretationStep:
    """
    Measure a logical Pauli operator. Y logical measurements are not supported yet.

    The algorithm is the following:
    
    - A.) Measure all data qubits in the block
    - B.) Update Syndromes for all stabilizers involved in the data qubits measured
    - C.) Create the logical observable including measured data qubits and all \
    previous corrections

    Parameters
    ----------
    interpretation_step : InterpretationStep
        The interpretation step to which the operation should be applied.
    operation : MeasureLogicalX | MeasureLogicalY | MeasureLogicalZ
        The operation to be applied, can either be a logical X, Y or Z measurement.
    same_timeslice : bool
        Flag indicating whether the operation is part of the same timestep as the
        previous operation.
    debug_mode : bool
        Flag indicating whether the interpretation should be done in debug mode.
        Activating debug mode will enable commutation validation for Block.

    Returns
    -------
    InterpretationStep
        New InterpretationStep containing all modifications due to the logical pauli
        measurement.
    """
    block = interpretation_step.get_block(operation.input_block_name)
    logical_qubit_index = operation.logical_qubit

    # Check for correct operation
    match operation.__class__.__name__:
        case "MeasureLogicalX":
            basis = "X"
        case "MeasureLogicalY":
            raise ValueError("Logical measurement in Y basis is not supported")
        case "MeasureLogicalZ":
            basis = "Z"
        case _:
            raise ValueError(f"Operation {operation.__class__.__name__} not supported")

    # 1 - Measure all data qubits in a block and keep track of the Cbits
    meas_circuit_seq = []

    # Add Hadamard layer for X basis
    if basis == "X":
        hadamard_layer = [
            Circuit(
                "H", channels=interpretation_step.get_channel_MUT(str(q), "quantum")
            )
            for q in block.data_qubits
        ]
        meas_circuit_seq += [hadamard_layer]

    # Create the measurement layer in Z basis
    cbit_labels = [f"c_{qubit}" for qubit in block.data_qubits]
    measurements = tuple(
        interpretation_step.get_new_cbit_MUT(label) for label in cbit_labels
    )
    measurement_layer = [
        Circuit(
            "measurement",
            channels=[
                interpretation_step.get_channel_MUT(
                    str(qubit), "quantum"
                ),  # qubit to measure
                interpretation_step.get_channel_MUT(
                    f"{cbit[0]}_{cbit[1]}", "classical"
                ),  # associated classical bit
            ],
        )
        for qubit, cbit in zip(block.data_qubits, measurements, strict=True)
    ]

    meas_circuit_seq += [measurement_layer]
    meas_circuit = Circuit(
        f"Measure logical {basis} of {block.unique_label}", circuit=meas_circuit_seq
    )

    # Append the circuit
    interpretation_step.append_circuit_MUT(meas_circuit, same_timeslice)

    # 2 - Update Syndromes for all stabilizers involved in the data qubits measured
    # Only use the stabilizers of the right pauli type
    relevant_stabs = [stab for stab in block.stabilizers if set(stab.pauli) == {basis}]
    # Get the classical bits associated with these stabilizers
    stab_cbits = [
        [
            cbit
            for cbit in measurements
            if cbit[0].split("_")[1] in map(str, stab.data_qubits)
        ]
        for stab in relevant_stabs
    ]
    # Create new Syndromes from these measurements
    new_syndromes = generate_syndromes(
        interpretation_step, relevant_stabs, block, stab_cbits
    )
    # Create Detectors for the new syndromes
    new_detectors = generate_detectors(interpretation_step, new_syndromes)

    # 3 - Create the logical observable including measured data qubits and all previous corrections

    if basis == "X":
        logical_qubit = block.logical_x_operators[logical_qubit_index]
        operator_updates = interpretation_step.logical_x_operator_updates
    else:
        logical_qubit = block.logical_z_operators[logical_qubit_index]
        operator_updates = interpretation_step.logical_z_operator_updates

    corrections = list(operator_updates.get(logical_qubit.uuid, ()))
    qubits_in_logical = [str(q) for q in logical_qubit.data_qubits]
    logical_measurements = [
        cbit for cbit in measurements if cbit[0].split("_")[1] in qubits_in_logical
    ]
    logical_observable = LogicalObservable(
        label=f"{block.unique_label}_{basis}_{logical_qubit_index}",
        measurements=logical_measurements + corrections,
    )
    interpretation_step.logical_measurements[
        LogicalMeasurement(blocks=(block.unique_label,), observable=basis)
    ] = tuple(logical_measurements + corrections)
    interpretation_step.append_detectors_MUT(new_detectors)
    interpretation_step.append_syndromes_MUT(new_syndromes)
    interpretation_step.logical_observables += (logical_observable,)

    return interpretation_step
