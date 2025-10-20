"""
Copyright (c) Entropica Labs Pte Ltd 2025.

Use, distribution and reproduction of this program in its source or compiled
form is prohibited without the express written consent of Entropica Labs Pte
Ltd.

"""

from .pauli_frame import PauliFrame
from .classicalreg import ClassicalRegister


class DataStore:
    """
    An Object that records relevant information users might want recorded from
    a particular computation.
    """

    def __init__(self):
        self.measurements = {"time_step": []}
        self.pf_records = {"forward": {"time_step": []}, "backward": {"time_step": []}}
        self.cr_records = {"time_step": []}

    def __str__(self):
        return str(self.measurements)

    def set_time_step(self, time_step: int) -> None:
        """
        Sets the time step of the current Moment being processed.
        """
        self.time_step = time_step  # pylint: disable=attribute-defined-outside-init

    def record_measurements(
        self,
        measurement_id: str,
        measurement_result: int,
        is_result_random: int,
    ) -> None:
        """
        Records the measurement result in the DataStore. Raises a ValueError if
        no time step was found in the Moment.
        """
        if self.time_step is not None:
            meas_result = {
                "measurement_result": measurement_result,
                "is_random": is_result_random,
            }
            # If the measurements of a particular time step has not been
            # recorded. Record the current measurement results in that
            # time step.
            if self.time_step not in self.measurements["time_step"]:
                self.measurements["time_step"].append(self.time_step)
                self.measurements.update({str(self.time_step): {}})
            self.measurements[str(self.time_step)].update({measurement_id: meas_result})
        else:
            raise ValueError("The time step of the Moment cannot be None.")

    def record_pauli_frame(
        self,
        init_pauliframe: PauliFrame,
        recorded_pauliframe: PauliFrame,
    ) -> None:
        """
        Records the PauliFrame in the DataStore. Raises a ValueError if
        no time step was found in the Moment.
        """
        if self.time_step is not None:
            direction = init_pauliframe.direction
            pf_id = init_pauliframe.id
            record_pf = {
                "initial_pauli_frame": init_pauliframe,
                "recorded_pauli_frame": recorded_pauliframe,
            }
            if self.time_step not in self.pf_records[direction]["time_step"]:
                self.pf_records[direction]["time_step"].append(self.time_step)
                self.pf_records[direction].update({str(self.time_step): {}})
            self.pf_records[direction][str(self.time_step)].update({pf_id: record_pf})

        else:
            raise ValueError("The time step of the Moment cannot be None.")

    def record_classical_register(
        self, input_classical_register: ClassicalRegister
    ) -> None:
        """
        Records the Snapshot of the Classical Register into the DataStore.
        """
        if self.time_step is not None:
            cr_snapshot = {
                input_classical_register.name: input_classical_register.create_snapshot()
            }
            # Records the state of a classical register at a particular time step.
            if self.time_step not in self.cr_records["time_step"]:
                self.cr_records["time_step"].append(self.time_step)
                self.cr_records.update({str(self.time_step): {}})
            self.cr_records[str(self.time_step)].update(cr_snapshot)
        else:
            raise ValueError("The time step of the Moment cannot be None.")

    def record_measurement_from_pauliframes(
        self,
        measurement_id: str,
        results: list[int],
        init_pauliframes: list[PauliFrame],
    ) -> None:
        """
        Records measurement flips that result from the interaction of
        Pauli frames and measurement operations.

        Measurement flips is passed as a list as there could be multiple Pframes.
        """
        if self.time_step is not None:

            # first we check whether "time_step" is already written into datastore
            # if not, then we update the dict.
            if self.time_step not in self.measurements["time_step"]:
                self.measurements["time_step"].append(self.time_step)
                self.measurements.update({str(self.time_step): {}})

                self.measurements[str(self.time_step)].update(
                    {
                        measurement_id: {
                            "flip_results": {},
                        }
                    }
                )
            # if "time_step" already is present, then we open a new dict keyed by "flip_results"
            # for particular measurement_id. If "measurement_id" has not been previously processed
            # update with key.
            elif measurement_id not in self.measurements[str(self.time_step)]:
                self.measurements[str(self.time_step)].update(
                    {
                        measurement_id: {
                            "flip_results": {},
                        }
                    }
                )

            elif (
                "flip_results"
                not in self.measurements[str(self.time_step)][measurement_id]
            ):
                self.measurements[str(self.time_step)][measurement_id].update(
                    {"flip_results": {}}
                )

            for pframe, result in zip(init_pauliframes, results, strict=True):
                self.measurements[str(self.time_step)][measurement_id][
                    "flip_results"
                ].update({pframe.id: result})
        else:
            raise ValueError("The time step of the Moment cannot be None.")

    def get_pframes(
        self,
        temporal_direction: str = "forward",
    ) -> tuple[dict]:
        """
        A helper function to return a tuple of all Pframes (dict)

        `temporal_direction` can be either "forward" or "backward".

        returns:
            tuple: A tuple of dictionaries, where each dictionary
            corresponds to a PauliFrame that was created in the circuit,
            and the computed output --- for frames propagated forwards.
        """
        raw_data = self.pf_records[temporal_direction]
        raw_data_keys = raw_data["time_step"]

        return tuple(fr for key in raw_data_keys for fr in raw_data[str(key)].values())

    def get_pframe_measurements(
        self,
        pauliframe_id: str,
    ) -> dict:
        """
        Helper function to aggregate all measurement results
        belonging to the same Pframe. The keys of returned dict
        are the uuids of meausurements in the circuit.

        returns:
            dict: A dictionary where keys are measurement ids and values are
            the PauliFrame flip results for the given Pframe id.
        """

        result_data = {}
        raw_data_timekeys = self.measurements["time_step"]

        for ts in raw_data_timekeys:
            for m_id in self.measurements[str(ts)].keys():

                measurement_entry = self.measurements[str(ts)][m_id]
                if "flip_results" in measurement_entry:
                    for pf_id_sel, m_val in measurement_entry["flip_results"].items():
                        if pf_id_sel == pauliframe_id:
                            result_data.update({m_id: m_val})

        return result_data
