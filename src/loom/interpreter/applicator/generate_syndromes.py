"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from loom.eka import Stabilizer

from ..syndrome import Syndrome
from ..interpretation_step import InterpretationStep
from ..utilities import Cbit


def generate_syndromes(
    interpretation_step: InterpretationStep,
    stabilizers: tuple[Stabilizer, ...],
    current_block_id: str,
    stab_measurements: tuple[tuple[Cbit, ...], ...],
) -> tuple[Syndrome, ...]:
    """
    Generate new Syndromes for the given stabilizers and the given block id.

    CAUTION: This function pops the entries from the stabilizer_updates field of the
    interpretation step to compute corrections. This may cause issues in the future if
    the information in this field also needs to be accessed somewhere else.

    Parameters
    ----------
    interpretation_step : InterpretationStep
        The updated interpretation step implementing the operation
    stabilizers : tuple[Stabilizer, ...]
        Stabilizers that were measured, results are included in `stab_measurements`
    current_block_id : str
        UUID of the block where the stabilizers are located
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
    if current_block_id in interpretation_step.block_qec_rounds.keys():
        new_round = interpretation_step.block_qec_rounds[
            current_block_id
        ]  # Get the index
        interpretation_step.block_qec_rounds[current_block_id] += 1  # Then increment
    else:
        new_round = 0
        interpretation_step.block_qec_rounds[current_block_id] = 1

    # Create new Syndromes
    new_syndromes = tuple(
        Syndrome(
            stabilizer=stabilizer.uuid,
            measurements=measurements,
            block=current_block_id,
            round=new_round,
            corrections=interpretation_step.stabilizer_updates.pop(
                stabilizer.uuid, tuple()
            ),  # get and remove the correction from int_step.stabilizer_updates
        )
        for stabilizer, measurements in zip(stabilizers, stab_measurements, strict=True)
    )

    return new_syndromes
