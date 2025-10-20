"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from loom.eka import Stabilizer, Circuit, Block

from ..syndrome import Syndrome
from ..interpretation_step import InterpretationStep
from ..utilities import Cbit


def generate_syndromes(
    interpretation_step: InterpretationStep,
    stabilizers: tuple[Stabilizer, ...],
    block: Block,
    stab_measurements: tuple[tuple[Cbit, ...], ...],
) -> tuple[Syndrome, ...]:
    """
    Generate new Syndromes for the given stabilizers and its associated block.
    Stabilizers are passed explicitly as they follow the same order as the
    `stab_measurements` variable used to compute the Syndromes.

    CAUTION: This function pops the entries from the stabilizer_updates field of the
    interpretation step to compute corrections. This may cause issues in the future if
    the information in this field also needs to be accessed somewhere else.


    Parameters
    ----------
    interpretation_step : InterpretationStep
        The updated interpretation step implementing the operation
    stabilizers : tuple[Stabilizer, ...]
        Stabilizers that were measured, results are included in `stab_measurements`
    block : Block
        Block containing the stabilizers
    stab_measurements : tuple[tuple[Cbit, ...], ...]
        Measurements used to create the syndromes. Each index contains a tuple of Cbits
        associated to the stabilizer at the same index in `stabilizers`.

    Returns
    -------
    tuple[Syndrome, ...]
        Syndromes created for the stabilizers, they are returned in the same order as
        the stabilizers are given.
    """
    # Find the round that needs to be associated with the new syndromes
    # If the block exists in block_qec_rounds, then we increment the round
    if block.uuid in interpretation_step.block_qec_rounds.keys():
        new_round = interpretation_step.block_qec_rounds[block.uuid]  # Get the index
        interpretation_step.block_qec_rounds[block.uuid] += 1  # Then increment
    # If the block does not exist in block_qec_rounds, then we create it
    else:
        new_round = 0
        interpretation_step.block_qec_rounds[block.uuid] = 1

    # Extract stabilizer labels to be inherited by the Syndromes and add the time stamp
    # This is generically not a true time-stamp as operations sometimes are added and
    # popped - It currently stands as a placeholder until a global time tracker is added
    time_stamp = len(
        Circuit.construct_padded_circuit_time_sequence(
            interpretation_step.intermediate_circuit_sequence
        )
    )

    # Create new Syndromes
    new_syndromes = tuple(
        Syndrome(
            stabilizer=stabilizer.uuid,
            measurements=measurements,
            block=block.uuid,
            round=new_round,
            labels=block.get_stabilizer_label(stabilizer.uuid)
            | {"time_coordinate": (time_stamp,)},
            corrections=interpretation_step.stabilizer_updates.pop(
                stabilizer.uuid, tuple()
            ),  # get and remove the correction from int_step.stabilizer_updates
        )
        for stabilizer, measurements in zip(stabilizers, stab_measurements, strict=True)
    )

    return new_syndromes
