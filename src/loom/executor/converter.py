"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

import numpy as np

# from abc import ABC, abstractmethod
# from ..eka import Circuit, Channel


def detector_reference_states(
    reference_measurement_sample: list,
    detector_observable_list: list,
) -> tuple[list[bool], list[bool]]:
    """Compute the state of detectors in the reference state (without noise)

    Parameters
    ----------
    reference_measurement_sample : list
        A list of syndrome measurements in the circuit under zero noise case.
    detector_observable_list : list
        The list of all detectors and observables present in stim circuit.
        NOTE: Assumes that there is only one observable appended at the end of the list
        and all others before are detectors. To be fixed later

    Returns
    -------
    detector_ref_parity : list[bool]
        The outcome of each detectors' reference state under zero noise
    observable_ref_parity : list[bool]
        The outcome of the observable's reference state under zero noise
    """
    detector_list, observable = (
        detector_observable_list[:-1],
        detector_observable_list[-1],
    )
    reference_measurement_sample = np.array(reference_measurement_sample, dtype=int)

    detector_ref_parity = []
    for detector in detector_list:

        # extract measurement index from classical channel (measurement channel) name
        meas_idx = [int(channel.name.split("_")[-1]) for channel in detector[1]]
        detector_measurements = [reference_measurement_sample[x] for x in meas_idx]

        # compute parity check for detector w.r.t defining measurements
        parity = not sum(detector_measurements) % 2 == 0
        detector_ref_parity.append(parity)

    meas_idx = [int(channel.name.split("_")[-1]) for channel in observable[1]]
    observable_measurements = [reference_measurement_sample[i] for i in meas_idx]
    observable_ref_parity = [False] if sum(observable_measurements) % 2 == 0 else [True]

    return detector_ref_parity, observable_ref_parity


# pylint: disable=too-many-locals
def detector_outcomes(
    measurement_counts_dict: dict,
    detector_observable_list: list[tuple],
    detector_ref_parities: list[bool] | None = None,
    observable_ref_parities: list[bool] | None = None,
):
    """Construct detector and observable parities from the raw device measurement
    results. If the reference parities of detectors and observable are not specified,
    the state of the detectors are computed w.r.t all zeros reference state.

    Parameters
    ----------
    measurement_counts_dict : dict
        The measurement outcomes from the device
    detector_observable_list : list[tuple]
        List containing detector definitions in absolute indices
        e.g. [("DETECTOR", [Channel(classical_register_12, ...)]),
        ("DETECTOR", [Channel(classical_register_5, ...)])]
    detector_ref_parities : list[bool] | None
        reference parities of detectors
    observable_ref_parities : list[bool] | None
        reference parities of observables

    Returns
    -------
    detectors_parity, observable_parity: tuple(np.ndarray, np.ndarray)
        detectors_parity: Each detector's parity w.r.t reference state
        observable_parity: Each observable's parity w.r.t reference state
    """
    detector_list, observable = (
        detector_observable_list[:-1],
        detector_observable_list[-1],
    )
    detectors_parity = []
    observable_parity = []
    measurement_counts = [
        key for key, value in measurement_counts_dict.items() for _ in range(value)
    ]

    for meas in measurement_counts:
        # reverse order of measurement in qiskit
        meas = [int(x) for x in meas.split(" ")][::-1]
        detector_parity = []
        for j, detector in enumerate(detector_list):

            # extract measurement index from classical channel (measurement) name
            meas_idx = [int(channel.name.split("_")[-1]) for channel in detector[1]]
            detector_measurements = [meas[x] for x in meas_idx]

            # compute parity check for detector w.r.t defining measurements
            parity = not sum(detector_measurements) % 2 == 0
            parity_match = (
                parity ^ detector_ref_parities[j]
                if detector_ref_parities is not None
                else parity
            )
            detector_parity.append(parity_match)
        detectors_parity.append(detector_parity)

        meas_idx = [int(channel.name.split("_")[-1]) for channel in observable[1]]
        observable_measurements = [meas[i] for i in meas_idx]
        parity = not sum(observable_measurements) % 2 == 0
        parity_match = (
            [parity ^ observable_ref_parities[0]]
            if observable_ref_parities is not None
            else [parity]
        )
        observable_parity.append(parity_match)

    return np.array(detectors_parity, dtype=np.bool_), np.array(
        observable_parity, dtype=np.bool_
    )
