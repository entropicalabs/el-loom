"""
Copyright 2024 Entropica Labs Pte Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

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
