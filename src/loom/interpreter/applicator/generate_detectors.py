"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from loom.eka import Stabilizer

from ..syndrome import Syndrome
from ..detector import Detector
from ..interpretation_step import InterpretationStep


def generate_detectors(
    interpretation_step: InterpretationStep,
    stabilizers: tuple[Stabilizer, ...],
    current_block_id: str,
    new_syndromes: tuple[Syndrome, ...],
) -> tuple[Detector, ...]:
    """
    Generate `Detector` objects for the input `new_syndromes`. The syndromes can be
    provided for multiple rounds and multiple stabilizers.

    The detectors are created using previous syndromes contained within the input
    `interpretation_step`. It also uses the block, stabilizer and logical operator evolutions
    in `interpretation_step` to group the syndromes into detectors.


    Parameters
    ----------
    interpretation_step: InterpretationStep
        The updated interpretation step implementing the operation
    stabilizers : tuple[Stabilizer, ...]
        List of block stabilizers
    current_block_id : str
        The UUID of the current block
    new_syndromes: tuple[Syndrome, ...]
        The list of all syndromes (this can include multiple stabilizers but not
        multiple rounds)

    Returns
    -------
    tuple[Detector, ...]
        Detectors created for the given syndromes, included enventual previous existing
        syndromes.
    """

    syndrome_vector_per_stab = {}
    for stab in stabilizers:
        # add the last round of syndromes for unchanged and morphed stabilizers
        prev_syndrome = interpretation_step.get_prev_syndrome(
            stab.uuid,
            current_block_id,
        )
        syndrome_vector_per_stab.update({stab.uuid: prev_syndrome})

    for syndrome in new_syndromes:
        stab_id = syndrome.stabilizer
        syndrome_vector_per_stab[stab_id] += (syndrome,)

    # Generate Detectors comparing syndromes at rounds t and t-1
    new_detectors = tuple()
    for stab in stabilizers:
        # temporally ordered syndrome list
        syndrome_list = syndrome_vector_per_stab[stab.uuid]
        if len(syndrome_list) > 1:
            # placed new detectors with syndromes in increasing temporal order
            new_detectors += (Detector(syndromes=syndrome_list),)

    return new_detectors
