"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=duplicate-code
from itertools import product

from loom.eka import Circuit, ChannelType, PauliOperator, Stabilizer
from loom.eka.operations import Shrink
from loom.eka.utilities import Orientation
from loom.interpreter import InterpretationStep
from loom.interpreter.utilities import Cbit
from loom.interpreter.applicator import generate_detectors, generate_syndromes

from ..code_factory import RotatedSurfaceCode


def shrink_consistency_checks(
    interpretation_step: InterpretationStep,
    operation: Shrink,
) -> RotatedSurfaceCode:
    """
    Check if the Shrink operation is valid for the given state of blocks

    Parameters
    ----------
    interpretation_step : InterpretationStep
        Interpretation step containing the block to shrink.
    operation : Shrink
        Shrink operation description.

    Returns
    -------
    RotatedSurfaceCode
        The block to be shrunk.
    """
    block = interpretation_step.get_block(operation.input_block_name)

    # Check block type
    if not isinstance(block, RotatedSurfaceCode):
        raise TypeError(
            f"The shrink operation is not supported for {type(block)} blocks."
        )

    is_horizontal = (
        Orientation.from_direction(operation.direction) == Orientation.HORIZONTAL
    )
    block_distance = block.size[0] if is_horizontal else block.size[1]

    # Limit the possible shrink lengths (between 1 and block_distance-2)
    if operation.length < 1 or operation.length > block_distance - 2:
        raise ValueError(
            f"Shrink length {operation.length} is not valid. "
            f"Must be between 1 and {block_distance - 2} for the selected block."
        )

    return block


def shrink(  # pylint: disable=too-many-statements, too-many-locals, line-too-long
    interpretation_step: InterpretationStep,
    operation: Shrink,
    same_timeslice: bool,
    debug_mode: bool,
) -> InterpretationStep:
    """
    Shrink a Block in the specified direction.

    The algorithm is the following:

    - A.) DATA QUBITS
    
        - A.1) Find data qubits which are measured in the shrink
        
    - B.) CIRCUIT
    
        - B.1) Create classical channels for all data qubit measurements
        - B.2) Create a measurement circuit for every measured data qubit
        - B.3) Append the measurement circuits to the InterpretationStep circuit. \
        If needed, apply a basis change
            
    - C.) STABILIZERS
    
        - C.1) Find stabilizers which are completely removed
        - C.2) Find stabilizers which have to be reduced in weight
        - C.3) Create new stabilizers with reduced weight
        - C.4) Update ``stabilizer_evolution`` and ``stabilizer_updates`` for the \
        stabilizers which have been reduced in weight
        - C.5) Combine the stabilizers to get a new set of stabilizers
        - C.6) Create syndromes for fully measured stabilizers
        - C.7) Create detectors for fully measured stabilizers
        
    - D.) LOGICAL OPERATORS
    
        - D.1) Find logical operators which do not have to be modified
        
    - E.) NEW BLOCK AND NEW INTERPRETATION STEP
    
        - E.1) Update the stabilizer to circuit mapping
        - E.2) Create the new block
        - E.3) Update the block history

    Parameters
    ----------
    interpretation_step : InterpretationStep
        Interpretation step containing the block to shrink.
    operation : Shrink
        Shrink operation description.
    same_timeslice : bool
        Flag indicating whether the operation is part of the same timestep as the
        previous operation.
    debug_mode : bool
        Flag indicating whether the interpretation should be done in debug mode.
        Activating debug mode will enable commutation validation for Block

    Returns
    -------
    InterpretationStep
        Interpretation step after the shrink operation.
    """

    block = shrink_consistency_checks(
        interpretation_step=interpretation_step, operation=operation
    )
    boundary_type = block.boundary_type(operation.direction)

    # A) DATA QUBITS

    # Generate a list of vectors by which the boundary qubits should be shifted to
    # obtain all data qubits that should be measured
    match operation.direction:
        case "left":
            shift_vectors = [(i, 0, 0) for i in range(operation.length)]
        case "right":
            shift_vectors = [(-i, 0, 0) for i in range(operation.length)]
        case "top":
            shift_vectors = [(0, i, 0) for i in range(operation.length)]
        case "bottom":
            shift_vectors = [(0, -i, 0) for i in range(operation.length)]

    # Get a list of all data qubits to measure
    combinations = product(
        block.boundary_qubits(direction=operation.direction), shift_vectors
    )
    qubits_to_measure = [
        tuple(coord1 + coord2 for coord1, coord2 in zip(qubit, shift_vect, strict=True))
        for qubit, shift_vect in combinations
    ]

    # B) CIRCUIT
    # B.1) Create classical channels for all data qubit measurements
    # So far, the classical channels are all named "c_{q}" for some qubit q.
    cbits = [interpretation_step.get_new_cbit_MUT(f"c_{q}") for q in qubits_to_measure]
    cbit_channels = [
        interpretation_step.get_channel_MUT(
            f"{cbit[0]}_{cbit[1]}", channel_type=ChannelType.CLASSICAL
        )
        for cbit in cbits
    ]
    # B.2) Create a measurement circuit for every measured data qubit and create
    # the sequence of gates
    measure_circuit_seq = [
        [
            Circuit(
                "Measurement",
                channels=[interpretation_step.get_channel_MUT(qb), cbit_channels[i]],
            )
            for i, qb in enumerate(qubits_to_measure)
        ]
    ]
    # B.3) Append the measurement circuits to the InterpretationStep circuit
    # If needed, apply a basis change before the measurement to measure in the right
    # basis
    if boundary_type == "X":
        # If the boundary type is X, the data qubit can directly be read out in the
        # Z basis
        shrink_circuit_list = measure_circuit_seq
    else:
        # If the boundary type is Z, the 2-bodies stabilizers are X, the data qubits
        # have to be read out in the X basis. Apply Hadamard gates before the
        # Z measurement to effectively measure in the X basis
        basis_change_circuit_seq = [
            [
                Circuit("H", channels=[interpretation_step.get_channel_MUT(qb)])
                for qb in qubits_to_measure
            ]
        ]
        # We add two sequences of gates to the circuit, i.e. 2 timesteps
        shrink_circuit_list = basis_change_circuit_seq + measure_circuit_seq
    # Construct the final circuit and append it to InterpretationStep.circuit
    shrink_circuit = Circuit(
        name=f"Shrink {block.unique_label} by {operation.length} from {operation.direction}",
        circuit=shrink_circuit_list,
    )
    interpretation_step.append_circuit_MUT(shrink_circuit, same_timeslice)

    # C) STABILIZERS

    # C.1) Find stabilizers which are completely removed
    stabs_to_remove = [
        stab
        for stab in block.stabilizers
        if any(qb in qubits_to_measure for qb in stab.data_qubits)
    ]

    # C.2) Find stabilizers which have to be reduced in weight
    old_stabs_to_reduce_weight = [
        stab
        for stab in block.stabilizers
        if stab.pauli[0] != boundary_type
        and len([qb for qb in stab.data_qubits if qb in qubits_to_measure]) == 2
        and len(stab.data_qubits) == 4
    ]

    # C.3) Create new stabilizers with reduced weight
    new_stabs_reduced_weight = [
        Stabilizer(
            pauli="".join(
                stab.pauli[i]
                for i, qb in enumerate(stab.data_qubits)
                if qb not in qubits_to_measure
            ),
            data_qubits=[qb for qb in stab.data_qubits if qb not in qubits_to_measure],
            ancilla_qubits=stab.ancilla_qubits,
        )
        for stab in old_stabs_to_reduce_weight
    ]
    stab_map_weight4_to_weight2 = {
        new_stabs_reduced_weight[i].uuid: (stab.uuid,)
        for i, stab in enumerate(old_stabs_to_reduce_weight)
    }
    # C.4) Update `stabilizer_evolution` and `stabilizer_updates` for the
    #      stabilizers which have been reduced in weight
    interpretation_step.stabilizer_evolution.update(stab_map_weight4_to_weight2)
    # Stabilizer updates: Take data qubit measurements into account
    for i, stab in enumerate(new_stabs_reduced_weight):
        previous_updates = (
            interpretation_step.stabilizer_updates[stab.uuid]
            if stab.uuid in interpretation_step.stabilizer_updates.keys()
            else ()
        )
        old_stab = old_stabs_to_reduce_weight[i]
        # Find the data qubits of this stabilizer which are measured and whose
        # corresponding cbit has to be included in the stabilizer update
        qbs_measured = [qb for qb in old_stab.data_qubits if qb in qubits_to_measure]
        # Find the cbits and include them in `stabilizer_updates`
        cbit_indices = [qubits_to_measure.index(qb) for qb in qbs_measured]
        new_updates = tuple(cbits[cbit_idx] for cbit_idx in cbit_indices)
        interpretation_step.stabilizer_updates[stab.uuid] = (
            previous_updates + new_updates
        )

    # C.5) Combine the stabilizers to get the new set of stabilizers
    new_stabs = set(block.stabilizers) - set(stabs_to_remove) | set(
        new_stabs_reduced_weight
    )

    # C.6) Create syndromes for fully measured stabilizers (in the right basis)
    fully_measured_stabs = [
        stab
        for stab in block.stabilizers
        if (
            all(qb in qubits_to_measure for qb in stab.data_qubits)
            and set(stab.pauli) == ({"Z"} if boundary_type == "X" else {"X"})
        )
    ]
    # Create the syndromes fo fully measuerd stabilizers, we need to order the
    # measurements in an order matching the fully_measured_stabs
    syndromes = generate_syndromes(
        interpretation_step,
        stabilizers=fully_measured_stabs,
        block=block,
        stab_measurements=tuple(
            tuple(
                cbit
                for q, cbit in product(stab.data_qubits, cbits)
                if cbit[0].split("_")[1] == str(q)
            )  # Match the order of cbits and stabilizers
            for stab in fully_measured_stabs
        ),
    )

    # C.7) Create the detectors for the fully measured stabilizers
    detectors = generate_detectors(interpretation_step, syndromes)
    # Finally append both syndromes and detectors
    interpretation_step.append_syndromes_MUT(syndromes)
    interpretation_step.append_detectors_MUT(detectors)

    # D) LOGICAL OPERATORS
    # D.1) Find logical operators which do not have to be modified

    def shorten_log_op(
        old_op: PauliOperator, cbits: list[Cbit]
    ) -> tuple[PauliOperator, dict]:
        """
        Return a shortened logical operator where the measured data qubits are
        removed. Note that this only works if the data qubits are measured in the
        right basis. Also, a new logical operator update dict is returned containing
        the new cbit measurements for the respective operator.

        Parameters
        ----------
        old_op : PauliOperator
            Old logical X/Z operator which has to be shortened
        cbits : list[Cbit]
            Measurements tied to the logical X/Z operator

        Returns
        -------
        PauliOperator
            Shortened logical X/Z operator
        tuple[Cbit, ...]
            Measurements tied to the short logical X/Z operator
        """
        # New shorter operator
        new_op = PauliOperator(
            pauli="".join(
                old_op.pauli[i]
                for i, qb in enumerate(old_op.data_qubits)
                if qb not in qubits_to_measure
            ),
            data_qubits=[
                qb for qb in old_op.data_qubits if qb not in qubits_to_measure
            ],
            uuid=old_op.uuid,
        )

        # Logical operator updates: Include the new cbit measurements
        cbit_indices = [
            qubits_to_measure.index(qb)
            for qb in old_op.data_qubits
            if qb in qubits_to_measure
        ]
        measurements = tuple(cbits[idx] for idx in cbit_indices)

        return new_op, measurements

    def get_new_log_op(
        int_step: InterpretationStep,
        old_op: PauliOperator,
        new_upper_left_corner: tuple[int, int, int],
        cbits: list[Cbit],
    ) -> tuple[PauliOperator, tuple[Cbit, ...], dict]:
        """
        Return new `PauliOperator` for a logical operator which has to be
        modified due to the shrink. Also, a new logical operator update dict is
        returned containing the cbit measurements for the respective operator and a
        new logical operator evolution dict is returned containing the mapping from
        the new operator to the old operator and eventual necessary stabilizers.

        Parameters
        ----------
        int_step : InterpretationStep
            Interpretation step containing the block to shrink
        old_op : PauliOperator
            Old logical X/Z operator to be changed
        new_upper_left_corner : tuple[int, int, int]
            New upper-left corner of the block after the shrink

        Returns
        -------
        PauliOperator
            New logical X/Z operator
        tuple[Cbit, ...]
            Measurements tied to the new logical X/Z operator
        dict
            Updated logical X/Z operator evolution dictionary
        """
        # Check in which bases the measured data qubits are included in the logical
        # operator
        measured_paulis = set(
            old_op.pauli[i]
            for i, qb in enumerate(old_op.data_qubits)
            if qb in qubits_to_measure
        )
        if len(measured_paulis) > 1:
            raise RuntimeError(
                f"Cannot update logical operator {old_op} during the shrink "
                "because it contains multiple different Paulis inside the region "
                "of data qubits that are measured in the shrink."
            )
        if measured_paulis == {"Y"}:
            raise RuntimeError(
                f"Cannot update logical operator {old_op} during the shrink "
                "because it contains a Y Pauli inside the region of data qubits "
                "that are measured in the shrink."
            )
        if list(measured_paulis)[0] != boundary_type:
            # This is the case where the data qubits are measured in the "right"
            # basis, i.e. in the same basis in which they are included in the
            # logical operator. In this case, the logical operator is simply shrunk
            # and those data qubits are added to the update list for later
            # processing
            new_op, measurements = shorten_log_op(old_op, cbits)
            new_log_op_evolution = {new_op.uuid: (old_op.uuid,)}
            return new_op, measurements, new_log_op_evolution
        if list(measured_paulis)[0] == boundary_type:
            # This is the case where the data qubits are measured in a basis which
            # is different to the basis in which they are included in the logical
            # operator.  In this case, the logical operator has to be moved into the
            # region of the block that is not measured. This can be done by
            # multiplying with stabilizers
            new_op, stabs_required = block.get_shifted_equivalent_logical_operator(
                old_op, new_upper_left_corner
            )
            # Add to the updates the cbits associated with the measured qubits and the
            # stabilizers required for the shift.
            cbits_required = int_step.retrieve_cbits_from_stabilizers(
                stabs_required, block
            )
            new_log_op_evolution = {
                new_op.uuid: (old_op.uuid,)
                + tuple(stab.uuid for stab in stabs_required)
            }
            return new_op, cbits_required, new_log_op_evolution

        raise RuntimeError("This should not happen. Please check the code for errors.")

    # Those operators which have at least one qubit inside the measured region
    # have to be updated
    new_qubits = [q for stab in new_stabs for q in stab.data_qubits]
    new_upleft_qubit = min(new_qubits, key=lambda x: x[0] + x[1])
    new_log_x_ops, new_log_z_ops = [], []
    for op in block.logical_x_operators:
        if any(qb in qubits_to_measure for qb in op.data_qubits):
            new_op, measurements, new_evolution = get_new_log_op(
                interpretation_step,
                op,
                new_upleft_qubit,
                cbits,
            )
            new_log_x_ops.append(new_op)
            interpretation_step.logical_x_evolution.update(new_evolution)
            interpretation_step.update_logical_operator_updates_MUT(
                operator_type="X",
                logical_operator_id=new_op.uuid,
                new_updates=measurements,
                inherit_updates=(new_op.uuid != op.uuid),
            )
        else:
            new_log_x_ops.append(op)
    for op in block.logical_z_operators:
        if any(qb in qubits_to_measure for qb in op.data_qubits):
            new_op, measurements, new_evolution = get_new_log_op(
                interpretation_step,
                op,
                new_upleft_qubit,
                cbits,
            )
            new_log_z_ops.append(new_op)
            interpretation_step.logical_z_evolution.update(new_evolution)
            interpretation_step.update_logical_operator_updates_MUT(
                operator_type="Z",
                logical_operator_id=new_op.uuid,
                new_updates=measurements,
                inherit_updates=(new_op.uuid != op.uuid),
            )
        else:
            new_log_z_ops.append(op)

    # E) NEW BLOCK
    # E.1) Update the stabilizer to circuit mapping
    new_stabilizer_to_circuit = block.stabilizer_to_circuit
    # Remove those stabilizers which do not exist anymore
    for stab in stabs_to_remove:
        new_stabilizer_to_circuit.pop(stab.uuid)
    # Associate the new stabilizers with the right syndrome circuits
    for stab in new_stabs_reduced_weight:
        # The default creation in the RotatedSurfaceCode class creates
        # SyndromeCircuits with the name "{direction}-{pauli}" where direction is
        # left, right, top or bottom. Find the SyndromeCircuit for this name and
        # take its uuid
        circ_name = f"{operation.direction}-{stab.pauli.lower()}"
        new_stabilizer_to_circuit[stab.uuid] = [
            circ for circ in block.syndrome_circuits if circ.name == circ_name
        ][0].uuid

    # E.2) Create the new block
    new_block = RotatedSurfaceCode(
        stabilizers=list(new_stabs),
        logical_x_operators=new_log_x_ops,
        logical_z_operators=new_log_z_ops,
        unique_label=block.unique_label,
        syndrome_circuits=block.syndrome_circuits,
        stabilizer_to_circuit=new_stabilizer_to_circuit,
        skip_validation=not debug_mode,
    )

    # E.3) Update the block history
    interpretation_step.update_block_history_and_evolution_MUT(
        new_blocks=(new_block,),
        old_blocks=(block,),
    )

    return interpretation_step
