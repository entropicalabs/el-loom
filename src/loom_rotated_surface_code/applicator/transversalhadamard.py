"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

# pylint: disable=line-too-long

from loom.eka import Circuit, Stabilizer, PauliOperator, SyndromeCircuit
from loom.interpreter import InterpretationStep

from ..operations import TransversalHadamard
from ..code_factory import RotatedSurfaceCode

# # pylint: disable=duplicate-code


def update_synd_circ(
    input_synd_circ: SyndromeCircuit, new_pauli: str
) -> SyndromeCircuit:
    """Updates the input Measurement SyndromeCircuit to measure a new pauli.
    The input SyndromeCircuit requires a specific design where the ancilla qubits
    are initialized and measured in the X basis with a CX, CY or CZ gate to the data
    qubit.
    The function replaces the two-qubit gate with a new gate that measures the new_pauli.
    If the new pauli is a Z, the function replaces the two-qubit gate with a CZ gate.
    If the new pauli is an X, the function replaces the two-qubit gate with a CX gate.
    If the new pauli is a Y, the function replaces the two-qubit gate with a CY gate.

    Parameters
    ----------
    input_synd_circ : SyndromeCircuit
        The input syndrome circuit.
    new_pauli : str
        The new pauli the circuit will be measuring now.

    Returns
    -------
        SyndromeCircuit: The updated syndrome circuit.
    """
    # Iterate through Syndrome Circuit and change the two-qubit gates.
    # Get actual circuit object.
    actual_circ_obj = input_synd_circ.circuit
    inner_circ_tuple = actual_circ_obj.circuit
    new_circ = tuple()

    # Check that Syndrome Circuit has the same Pauli Length as New Stabilizer
    if len(new_pauli) != len(input_synd_circ.pauli):
        raise ValueError(
            "The new stabilizer has a different pauli length than the input syndrome circuit."
        )

    iterator = 0
    for each_tick in inner_circ_tuple:
        # If tick is empty, add an empty tuple.
        if len(each_tick) == 0:
            new_circ += ((),)
            continue

        # Else, Iterate through each gate in the tick.
        for each_gate in each_tick:
            if each_gate.name in ["cz", "cx", "cy"]:
                # Recreate the gate with a new name.
                new_circ += (
                    (
                        Circuit(
                            name=f"c{new_pauli[iterator]}", channels=each_gate.channels
                        ),
                    ),
                )
                iterator += 1
            else:
                new_circ += ((each_gate,),)

    # Get new circuit name.
    new_name = (
        actual_circ_obj.name[: -1 * len(input_synd_circ.pauli)] + new_pauli.lower()
    )

    # Recreate Circuit Block with updated name.
    # Channels need to be assigned properly. Automatic assignment jumbles ordering.
    new_circ_obj = Circuit(
        name=new_name,
        circuit=new_circ,
        channels=actual_circ_obj.channels,
    )

    # Create new Syndrome Circuit object.
    new_synd_circ = SyndromeCircuit(
        pauli=new_pauli,
        name=new_name,
        circuit=new_circ_obj,
    )

    return new_synd_circ


def transversal_hadamard_syndrome_circuits(
    block: RotatedSurfaceCode,
    interpretation_step: InterpretationStep,
    new_stabs: list[Stabilizer],
    new_stabilizer_to_circuit: dict[str, str],
    new_synd_circs: tuple[SyndromeCircuit] = tuple(),
) -> tuple[tuple[SyndromeCircuit], dict[str, str]]:
    """For a given Block, InterpretationStep and list of new Stabilizers that
    result from a Transversal Hadamard operation, this generates the new SyndromeCircuit
    objects that are associated with the new Stabilizers. Along with the new
    SyndromeCircuit objects, it also generates a mapping of the new Stabilizers to
    the UUIDs of the new SyndromeCircuits. An initial collection of new SyndromeCircuits
    and a partially populated mapping of the new Stabilizers to the UUIDs of the new
    SyndromeCircuits can be provided as optional arguments.

    Parameters
    ----------
    block : RotatedSurfaceCode
        The block of the surface code.
    interpretation_step : InterpretationStep
        The interpretation step that contains the evolution of the new Stabilizers.
    new_stabs : list[Stabilizer]
        The new stabilizers that result from the Transversal Hadamard operation.
    new_stabilizer_to_circuit : dict[str, str]
        The partially populated mapping of the new Stabilizers to the UUIDs of the new
        SyndromeCircuits.
    new_synd_circs : tuple[SyndromeCircuit], optional
        The new SyndromeCircuit objects that are ALREADY associated with the new
        Stabilizers.

    Returns
    -------
    tuple[tuple[SyndromeCircuit], dict[str,str]]
        A tuple containing the new SyndromeCircuit objects and a dictionary mapping
        the new Stabilizers to the UUIDs of the new SyndromeCircuits.
    """
    # D) SYNDROMECIRCUITS
    # D.1) Update Syndrome Circuits with new UUIDs.
    for each_new_stab in new_stabs:
        # D.2) Determine the Syndrome Circuit(s) associated with the old Stabilizer(s).
        # If more than one stabilizers evolved into the same stabilizer, we only
        # consider the first one.
        old_stab_uuid = interpretation_step.stabilizer_evolution[each_new_stab.uuid][0]
        old_synd_circ_uuid = block.stabilizer_to_circuit[old_stab_uuid]
        selected_synd_circ = next(
            (
                each_selected_synd
                for each_selected_synd in block.syndrome_circuits
                if each_selected_synd.uuid == old_synd_circ_uuid
            )
        )
        # D.3) Update the Syndrome Circuit(s) for the new Stabilizer(s).
        updated_synd_circ = update_synd_circ(selected_synd_circ, each_new_stab.pauli)

        # D.4) Update Collection of New Syndrome Circuits and New Stabilizer to Circuit Mapping
        # We consider Syndrome Circuits duplicates if all their properties except UUID are the same.
        duplicate_synd_circ = next(
            (
                new_synd_circ
                for new_synd_circ in new_synd_circs
                if new_synd_circ == updated_synd_circ
            ),
            None,
        )
        if not duplicate_synd_circ:
            # If there is no duplicate, we add the new Syndrome Circuit to the collection.
            new_synd_circs += (updated_synd_circ,)
            # We map the stabilizer to the new Syndrome Circuit's UUID.
            new_stabilizer_to_circuit.update(
                {each_new_stab.uuid: updated_synd_circ.uuid}
            )
        else:
            # If there is a duplicate, we map the stabilizer to the duplicate's UUID instead.
            new_stabilizer_to_circuit.update(
                {each_new_stab.uuid: duplicate_synd_circ.uuid}
            )

    return new_synd_circs, new_stabilizer_to_circuit


def transversalhadamard(
    interpretation_step: InterpretationStep,
    operation: TransversalHadamard,
    same_timeslice: bool,
    debug_mode: bool,
) -> InterpretationStep:
    """
    Apply a Transversal Hadamard gate to a Block.

    The algorithm is as follows:

    A) CIRCUIT
        - A.1) Apply Hadamard gates over all data qubits of the block.
        - A.2) Append Hadamard gates to the circuit within the InterpretationStep.
    B) STABILIZERS
        - B.1) Update Stabilizers (XX -> ZZ, ZZ -> XX, for 2-body stabilizers) with new UUIDs.
        - B.2) Record the old UUIDs and the new UUIDs of the Stabilizers. (stabilizer_evolution)
    C) PAULIOPERATORS
        - C.1) Update PauliOperators (X to Z Operators) with new UUIDs
               and logical updates.
        - C.2) Update PauliOperators (Z to X Operators) with new UUIDs
               and logical updates.
    D) SYNDROMECIRCUITS
        - D.1) Update Syndrome Circuits with new UUIDs.
        - D.2) Determine the Syndrome Circuit(s) associated with the old Stabilizer(s).
        - D.3) Update the Syndrome Circuit(s) for the new Stabilizer(s).
        - D.4) Update Collection of New Syndrome Circuits and New Stabilizer to Circuit Mapping
    E) NEW BLOCK
        - E.1) Create new Block (Same unique_label as Input Block)
    G) BLOCK HISTORY
        - G.1) Update the latest Block Configuration in InterpretationStep.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        Interpretation step containing the block to apply the transversal hadamard to.
    operation : TransversalHadamard
        Transversal hadamard operation description.
    same_timeslice : bool
        Flag indicating whether the operation is part of the same timestep as the
        previous operation.
    debug_mode : bool
        Flag indicating whether the interpretation should be done in debug mode.
        Activating debug mode will enable commutation validation for Block.

    Returns
    -------
    InterpretationStep
        Interpretation step after the transversal hadamard operation.
    """

    # A) CIRCUIT
    # A.1) Apply Hadamard gates over all data qubits of the block.
    block = interpretation_step.get_block(operation.input_block_name)

    data_channels = tuple(
        interpretation_step.get_channel_MUT(str(each_dqbit), "quantum")
        for each_dqbit in block.data_qubits
    )
    hadamard_circ_seq = [
        [Circuit("H", channels=(each_channel,)) for each_channel in data_channels]
    ]

    hadamard_block = Circuit(
        name=f"Transversal Hadamard on {block.unique_label}",
        circuit=hadamard_circ_seq,
    )

    # A.2) Append Hadamard gates to the circuit within the InterpretationStep.
    interpretation_step.append_circuit_MUT(hadamard_block, same_timeslice)

    # B) STABILIZERS
    # Mapping for the effect of the Hadamard gates on the Paulis.
    # Only able to convert X -> Z and Z -> X.
    had_trans = {"X": "Z", "Z": "X"}

    # B.1) Update Stabilizers (XX -> ZZ, ZZ -> XX, for 2-body stabilizers) with new UUIDs.
    new_stabs = tuple(
        Stabilizer(
            pauli="".join(
                had_trans[each_pauli] for each_pauli in each_stabilizer.pauli
            ),
            data_qubits=each_stabilizer.data_qubits,
            ancilla_qubits=each_stabilizer.ancilla_qubits,
        )
        for each_stabilizer in block.stabilizers
        if "Y" not in each_stabilizer.pauli
    )

    # TODO: Raises a ValueError if there is a Pauli Y Stabilizer. Message is vague and doesn't specify the exact Stabilizer that is problematic.
    if len(new_stabs) != len(block.stabilizers):
        raise ValueError(
            "One of the Stabilizer in the Block contains a Pauli Y, which is not supported by Hadamard."
        )

    # B.2) Record the new UUIDs and the old UUIDs of the Stabilizers. (stabilizer_evolution)
    interpretation_step.stabilizer_evolution.update(
        {
            each_new_stab.uuid: (each_stabilizer.uuid,)
            for each_new_stab, each_stabilizer in zip(
                new_stabs, block.stabilizers, strict=True
            )
        }
    )

    # C) PAULIOPERATORS
    # C.1) Update PauliOperators (X to Z Pauli String in Operators) with new UUIDs
    # and logical updates.

    new_x_pauli_ops = tuple(
        PauliOperator(
            pauli="".join(had_trans[each_pauli] for each_pauli in each_pauli_op.pauli),
            data_qubits=each_pauli_op.data_qubits,
        )
        for each_pauli_op in block.logical_x_operators
    )

    for new_op, old_op in zip(new_x_pauli_ops, block.logical_x_operators, strict=True):
        old_op_updates = interpretation_step.logical_x_operator_updates.get(
            old_op.uuid, ()
        )
        interpretation_step.logical_x_evolution[new_op.uuid] = (old_op.uuid,)
        interpretation_step.update_logical_operator_updates_MUT(
            new_op.pauli[0],
            new_op.uuid,
            new_updates=old_op_updates,
            inherit_updates=False,
        )

    # C.2) Update PauliOperators (Z to X String in Operators) with new UUIDs
    # and logical updates.

    new_z_pauli_ops = tuple(
        PauliOperator(
            pauli="".join(had_trans[each_pauli] for each_pauli in each_pauli_op.pauli),
            data_qubits=each_pauli_op.data_qubits,
        )
        for each_pauli_op in block.logical_z_operators
    )

    for new_op, old_op in zip(new_z_pauli_ops, block.logical_z_operators, strict=True):
        old_op_updates = interpretation_step.logical_z_operator_updates.get(
            old_op.uuid, ()
        )
        interpretation_step.logical_z_evolution[new_op.uuid] = (old_op.uuid,)
        interpretation_step.update_logical_operator_updates_MUT(
            new_op.pauli[0],
            new_op.uuid,
            new_updates=old_op_updates,
            inherit_updates=False,
        )
    # D) SYNDROMECIRCUITS
    #   D.1) Update Syndrome Circuits with new UUIDs.
    #   D.2) Determine the Syndrome Circuit(s) associated with the old Stabilizer(s).
    #   D.3) Update the Syndrome Circuit(s) for the new Stabilizer(s).
    #   D.4) Update Collection of New Syndrome Circuits and New Stabilizer to Circuit Mapping
    new_stabilizer_to_circuit = {}
    new_synd_circ_tuple, new_stabilizer_to_circuit = (
        transversal_hadamard_syndrome_circuits(
            block, interpretation_step, new_stabs, new_stabilizer_to_circuit
        )
    )

    # E) NEW BLOCK
    # E.1) Create new Block (Same unique_label as Input Block)
    new_block = RotatedSurfaceCode(
        stabilizers=new_stabs,
        logical_x_operators=new_z_pauli_ops,
        logical_z_operators=new_x_pauli_ops,
        syndrome_circuits=new_synd_circ_tuple,
        stabilizer_to_circuit=new_stabilizer_to_circuit,
        unique_label=block.unique_label,
        skip_validation=not debug_mode,
    )

    # G) BLOCK HISTORY
    # G.1) Update the latest Block Configuration in InterpretationStep.
    interpretation_step.update_block_history_and_evolution_MUT((new_block,), (block,))

    return interpretation_step
