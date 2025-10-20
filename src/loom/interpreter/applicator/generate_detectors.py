"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from ..syndrome import Syndrome
from ..detector import Detector
from ..interpretation_step import InterpretationStep


def generate_detectors(
    interpretation_step: InterpretationStep,
    new_syndromes: tuple[Syndrome, ...],
) -> tuple[Detector, ...]:
    """
    Generates the detectors, by matching the new syndromes to their associated old
    syndromes. This relation is obtained via the stabilizer flow specified in the
    operation applicator.

    Detectors inherit the labels from the new syndromes.

    Parameters
    ----------
    interpretation_step: InterpretationStep
        The updated interpretation step implementing the operation.
    new_syndromes: tuple[Syndrome, ...]
        The newly generated syndromes during the operation.

    Returns
    -------
    tuple[Detector, ...]
        Detectors associated with the newly generated syndromes in the operation.
    """

    # Extract old syndromes
    old_syndromes = tuple(
        interpretation_step.get_prev_syndrome(syndrome.stabilizer, syndrome.block)
        for syndrome in new_syndromes
    )

    # Get new detectors
    new_detectors = tuple(
        Detector(
            syndromes=old_syndromes_list + [new_syndrome], labels=new_syndrome.labels
        )
        for old_syndromes_list, new_syndrome in zip(
            old_syndromes, new_syndromes, strict=True
        )
        if old_syndromes_list
    )

    return new_detectors
